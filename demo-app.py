import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import networkx as nx
import numpy as np


GraphType = Literal["line", "star", "cycle", "erdos_renyi", "custom"]


@dataclass
class AppConfig:
    graph_type: GraphType = "custom"
    num_nodes: int = 10
    custom_edges: Optional[list[tuple[int, int]]] = field(
        default_factory=lambda: [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (1, 5),
            (5, 6),
            (2, 7),
            (7, 8),
            (8, 9),
        ]
    )

    initial_infected: list[int] = field(default_factory=lambda: [0])

    steps: int = 12
    beta: float = 0.45
    random_seed: int = 42
    layout_seed: int = 7

    erdos_renyi_p: float = 0.3
    node_labels: Optional[list[str]] = None


# -----------------------------
# Graph creation
# -----------------------------

def create_graph(config: AppConfig) -> nx.Graph:
    if config.graph_type == "line":
        graph = nx.path_graph(config.num_nodes)
    elif config.graph_type == "star":
        if config.num_nodes < 2:
            raise ValueError("star graph requires at least 2 nodes")
        graph = nx.star_graph(config.num_nodes - 1)
    elif config.graph_type == "cycle":
        if config.num_nodes < 3:
            raise ValueError("cycle graph requires at least 3 nodes")
        graph = nx.cycle_graph(config.num_nodes)
    elif config.graph_type == "erdos_renyi":
        graph = nx.erdos_renyi_graph(
            config.num_nodes,
            config.erdos_renyi_p,
            seed=config.random_seed,
        )
    elif config.graph_type == "custom":
        if config.custom_edges is None:
            raise ValueError("custom_edges must be provided for graph_type='custom'")
        graph = nx.Graph()
        graph.add_nodes_from(range(config.num_nodes))
        graph.add_edges_from(config.custom_edges)
    else:
        raise ValueError(f"Unsupported graph_type: {config.graph_type}")

    return graph


# -----------------------------
# State and simulators
# -----------------------------

def create_initial_state(num_nodes: int, infected_nodes: list[int]) -> np.ndarray:
    state = np.zeros(num_nodes, dtype=int)
    for node in infected_nodes:
        if node < 0 or node >= num_nodes:
            raise ValueError(f"Invalid infected node index: {node}")
        state[node] = 1
    return state


def deterministic_step(A: np.ndarray, state: np.ndarray) -> np.ndarray:
    """
    Movable infection version.

    The adjacency matrix acts as an operator:
        y_t = A @ x_t

    Then we threshold the result to binary values:
        x_{t+1} = 1(y_t > 0)

    Interpretation:
    - a node is infected in the next step if it currently has at least one infected neighbor
    - infection does not automatically persist
    """
    influence = A @ state
    next_state = (influence > 0).astype(int)
    return next_state



def probabilistic_step(
    A: np.ndarray,
    state: np.ndarray,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Movable probabilistic version.

    First compute neighbor influence:
        y_t = A @ x_t

    For each node with k infected neighbors, infection probability is:
        p = 1 - (1 - beta)^k

    Since this is the movable version, the next state is sampled only from current influence.
    Current infection does not automatically persist.
    """
    influence = A @ state
    probs = 1.0 - (1.0 - beta) ** influence
    random_values = rng.random(len(state))
    next_state = (random_values < probs).astype(int)
    return next_state



def run_deterministic_history(A: np.ndarray, initial_state: np.ndarray, steps: int) -> list[np.ndarray]:
    history = [initial_state.copy()]
    current = initial_state.copy()

    for _ in range(steps):
        current = deterministic_step(A, current)
        history.append(current.copy())

    return history



def run_probabilistic_history(
    A: np.ndarray,
    initial_state: np.ndarray,
    steps: int,
    beta: float,
    random_seed: int,
) -> list[np.ndarray]:
    history = [initial_state.copy()]
    current = initial_state.copy()
    rng = np.random.default_rng(random_seed)

    for _ in range(steps):
        current = probabilistic_step(A, current, beta=beta, rng=rng)
        history.append(current.copy())

    return history


# -----------------------------
# Visualization helpers
# -----------------------------

def build_labels(config: AppConfig, num_nodes: int) -> dict[int, str]:
    if config.node_labels is None:
        return {i: str(i) for i in range(num_nodes)}

    if len(config.node_labels) != num_nodes:
        raise ValueError("node_labels must have the same length as num_nodes")

    return {i: label for i, label in enumerate(config.node_labels)}



def state_to_colors(state: np.ndarray) -> list[str]:
    return ["red" if value == 1 else "skyblue" for value in state]



def changed_nodes(prev_state: np.ndarray, curr_state: np.ndarray) -> list[int]:
    return [i for i in range(len(curr_state)) if prev_state[i] != curr_state[i]]



def draw_panel(
    ax,
    graph: nx.Graph,
    pos: dict,
    state: np.ndarray,
    labels: dict[int, str],
    title: str,
    subtitle: str,
) -> None:
    ax.clear()
    nx.draw(
        graph,
        pos,
        ax=ax,
        labels=labels,
        with_labels=True,
        node_color=state_to_colors(state),
        node_size=850,
        edge_color="gray",
        linewidths=1.2,
        font_weight="bold",
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    ax.axis("off")


# -----------------------------
# Interactive viewer
# -----------------------------

class StepViewer:
    def __init__(
        self,
        graph: nx.Graph,
        A: np.ndarray,
        det_history: list[np.ndarray],
        prob_history: list[np.ndarray],
        pos: dict,
        labels: dict[int, str],
        config: AppConfig,
    ) -> None:
        self.graph = graph
        self.A = A
        self.det_history = det_history
        self.prob_history = prob_history
        self.pos = pos
        self.labels = labels
        self.config = config
        self.step_index = 0
        self.max_step = len(det_history) - 1

        self.fig = plt.figure(figsize=(13, 7))
        self.ax_det = self.fig.add_axes([0.05, 0.22, 0.40, 0.68])
        self.ax_prob = self.fig.add_axes([0.55, 0.22, 0.40, 0.68])

        self.ax_prev = self.fig.add_axes([0.28, 0.07, 0.12, 0.07])
        self.ax_next = self.fig.add_axes([0.44, 0.07, 0.12, 0.07])
        self.ax_reset = self.fig.add_axes([0.60, 0.07, 0.12, 0.07])

        self.btn_prev = Button(self.ax_prev, "Previous")
        self.btn_next = Button(self.ax_next, "Next")
        self.btn_reset = Button(self.ax_reset, "Reset")

        self.btn_prev.on_clicked(self.on_previous)
        self.btn_next.on_clicked(self.on_next)
        self.btn_reset.on_clicked(self.on_reset)

        self.info_text = self.fig.text(0.05, 0.95, "", fontsize=12, va="top")
        self.operator_text = self.fig.text(0.05, 0.165, "", fontsize=10, va="top", family="monospace")

        self.fig.suptitle(
            "Infection spread on a graph: adjacency operator-based simulation",
            fontsize=15,
            y=0.995,
        )

        self.render()

    def build_panel_subtitle(self, history: list[np.ndarray]) -> str:
        state = history[self.step_index]
        infected = int(state.sum())

        if self.step_index == 0:
            changed = []
        else:
            changed = changed_nodes(history[self.step_index - 1], state)

        return f"step = {self.step_index}, infected = {infected}/{len(state)}, changed = {changed}"

    def build_operator_text(self) -> str:
        current = self.det_history[self.step_index]
        influence = self.A @ current
        return (
            "Deterministic operator view (current step):\n"
            f"x_t     = {current.tolist()}\n"
            f"A @ x_t = {influence.tolist()}"
        )

    def render(self) -> None:
        det_state = self.det_history[self.step_index]
        prob_state = self.prob_history[self.step_index]

        draw_panel(
            self.ax_det,
            self.graph,
            self.pos,
            det_state,
            self.labels,
            "Deterministic movable model",
            self.build_panel_subtitle(self.det_history),
        )

        draw_panel(
            self.ax_prob,
            self.graph,
            self.pos,
            prob_state,
            self.labels,
            "Probabilistic movable model",
            self.build_panel_subtitle(self.prob_history),
        )

        self.info_text.set_text(
            f"Graph type: {self.config.graph_type} | beta = {self.config.beta} | "
            f"initial infected = {self.config.initial_infected} | step = {self.step_index}/{self.max_step}"
        )

        self.operator_text.set_text(self.build_operator_text())

        # Force immediate redraw to avoid UI freezing on some backends
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_previous(self, _event) -> None:
        new_step = max(0, self.step_index - 1)
        if new_step != self.step_index:
            self.step_index = new_step
            self.render()

    def on_next(self, _event) -> None:
        new_step = min(self.max_step, self.step_index + 1)
        if new_step != self.step_index:
            self.step_index = new_step
            self.render()

    def on_reset(self, _event) -> None:
        if self.step_index != 0:
            self.step_index = 0
            self.render()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    config = AppConfig(
        graph_type="custom",
        num_nodes=10,
        custom_edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (1, 5),
            (5, 6),
            (2, 7),
            (7, 8),
            (8, 9),
        ],
        initial_infected=[0],
        steps=8,
        beta=0.45,
        random_seed=42,
        node_labels=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    )

    # Alternative quick examples:
    # config = AppConfig(graph_type="line", num_nodes=6, initial_infected=[0], steps=6, beta=0.5)
    # config = AppConfig(graph_type="star", num_nodes=8, initial_infected=[0], steps=5, beta=0.4)
    # config = AppConfig(graph_type="cycle", num_nodes=8, initial_infected=[0], steps=6, beta=0.35)
    # config = AppConfig(graph_type="erdos_renyi", num_nodes=10, initial_infected=[0, 4], steps=7, beta=0.4, random_seed=12)

    graph = create_graph(config)
    initial_state = create_initial_state(config.num_nodes, config.initial_infected)
    labels = build_labels(config, config.num_nodes)

    A = nx.to_numpy_array(graph, dtype=int)
    pos = nx.spring_layout(graph, seed=config.layout_seed)

    det_history = run_deterministic_history(A, initial_state, config.steps)
    prob_history = run_probabilistic_history(
        A,
        initial_state,
        config.steps,
        beta=config.beta,
        random_seed=config.random_seed,
    )

    StepViewer(
        graph=graph,
        A=A,
        det_history=det_history,
        prob_history=prob_history,
        pos=pos,
        labels=labels,
        config=config,
    )

    plt.show()


if __name__ == "__main__":
    main()
