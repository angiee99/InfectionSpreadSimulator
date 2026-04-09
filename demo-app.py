from dataclasses import dataclass, field
from typing import Literal, Optional

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
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

    steps: int = 8
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
    influence = A @ state
    next_state = (influence > 0).astype(int)
    return next_state


def probabilistic_step(
    A: np.ndarray,
    state: np.ndarray,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
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


def build_det_annotations(influence: np.ndarray, show_influence: bool) -> dict[int, str]:
    annotations = {}
    if show_influence:
        for i, value in enumerate(influence):
            if value > 0:
                annotations[i] = f"A@x={int(value)}"
    return annotations


def build_prob_annotations(
    influence: np.ndarray,
    probs: np.ndarray,
    show_influence: bool,
    show_probs: bool,
) -> dict[int, str]:
    annotations = {}
    for i in range(len(influence)):
        parts = []
        if show_influence and influence[i] > 0:
            parts.append(f"A@x={int(influence[i])}")
        if show_probs and probs[i] > 0:
            parts.append(f"p={probs[i]:.2f}")
        if parts:
            annotations[i] = "\n".join(parts)
    return annotations


def draw_panel(
    ax,
    graph: nx.Graph,
    pos: dict[int, tuple[float, float]],
    state: np.ndarray,
    labels: dict[int, str],
    title: str,
    subtitle: str,
    top_annotations: Optional[dict[int, str]] = None,
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

    if top_annotations:
        for node, text in top_annotations.items():
            x, y = pos[node]
            ax.text(
                x,
                y + 0.12,
                text,
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                zorder=5,
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
        pos: dict[int, tuple[float, float]],
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

        self.sidebar_visible = True

        self.show_det_panel = True
        self.show_prob_panel = True
        self.det_show_influence = False
        self.prob_show_influence = False
        self.prob_show_probability = False

        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle(
            "Infection spread on a graph: adjacency operator-based simulation",
            fontsize=15,
            y=0.985,
        )

        self.sidebar_texts: list = []
        self.ax_det = None
        self.ax_prob = None

        self.ax_sidebar_bg = None
        self.ax_det_main = None
        self.ax_det_opts = None
        self.ax_prob_main = None
        self.ax_prob_opts = None

        self.chk_det_main = None
        self.chk_det_opts = None
        self.chk_prob_main = None
        self.chk_prob_opts = None

        self._create_static_widgets()
        self._create_sidebar()
        self._create_dynamic_axes()
        self.render()

    def _create_static_widgets(self) -> None:
        self.ax_toggle_sidebar = self.fig.add_axes([0.01, 0.93, 0.09, 0.055])
        self.btn_toggle_sidebar = Button(self.ax_toggle_sidebar, "Hide menu")
        self.btn_toggle_sidebar.on_clicked(self.on_toggle_sidebar)

        self.info_text = self.fig.text(0.16, 0.93, "", fontsize=11, va="top")

        self.ax_prev = self.fig.add_axes([0.34, 0.03, 0.10, 0.07])
        self.ax_next = self.fig.add_axes([0.46, 0.03, 0.10, 0.07])
        self.ax_reset = self.fig.add_axes([0.58, 0.03, 0.10, 0.07])

        self.btn_prev = Button(self.ax_prev, "Previous")
        self.btn_next = Button(self.ax_next, "Next")
        self.btn_reset = Button(self.ax_reset, "Reset")

        self.btn_prev.on_clicked(self.on_previous)
        self.btn_next.on_clicked(self.on_next)
        self.btn_reset.on_clicked(self.on_reset)

    def _style_checkbuttons(self, check: CheckButtons) -> None:
        # Makes the active state easier to see across matplotlib versions.
        for rect in getattr(check, "rectangles", []):
            rect.set_facecolor("white")
            rect.set_edgecolor("black")
            rect.set_linewidth(1.0)

        for lines in getattr(check, "lines", []):
            for line in lines:
                line.set_color("black")
                line.set_linewidth(2.0)

    def _create_sidebar(self) -> None:
        self.ax_sidebar_bg = self.fig.add_axes([0.01, 0.17, 0.22, 0.74])
        self.ax_sidebar_bg.set_facecolor("#f3f3f3")
        self.ax_sidebar_bg.set_xticks([])
        self.ax_sidebar_bg.set_yticks([])
        for spine in self.ax_sidebar_bg.spines.values():
            spine.set_visible(False)

        self.sidebar_texts = [
            self.fig.text(0.03, 0.88, "Display settings", fontsize=12, fontweight="bold"),
            self.fig.text(0.03, 0.73, "Deterministic options", fontsize=10, fontweight="bold"),
            self.fig.text(0.03, 0.47, "Probabilistic options", fontsize=10, fontweight="bold"),
        ]

        self.ax_det_main = self.fig.add_axes([0.03, 0.77, 0.18, 0.07])
        self.chk_det_main = CheckButtons(
            self.ax_det_main,
            ["Deterministic model"],
            [self.show_det_panel],
        )
        self._style_checkbuttons(self.chk_det_main)
        self.chk_det_main.on_clicked(self.on_det_main_toggle)

        self.ax_det_opts = self.fig.add_axes([0.03, 0.61, 0.18, 0.09])
        self.chk_det_opts = CheckButtons(
            self.ax_det_opts,
            ["show A @ x_t"],
            [self.det_show_influence],
        )
        self._style_checkbuttons(self.chk_det_opts)
        self.chk_det_opts.on_clicked(self.on_det_option_toggle)

        self.ax_prob_main = self.fig.add_axes([0.03, 0.47, 0.18, 0.07])
        self.chk_prob_main = CheckButtons(
            self.ax_prob_main,
            ["Probabilistic model"],
            [self.show_prob_panel],
        )
        self._style_checkbuttons(self.chk_prob_main)
        self.chk_prob_main.on_clicked(self.on_prob_main_toggle)

        self.ax_prob_opts = self.fig.add_axes([0.03, 0.28, 0.18, 0.17])
        self.chk_prob_opts = CheckButtons(
            self.ax_prob_opts,
            ["show A @ x_t", "show probability of infection"],
            [self.prob_show_influence, self.prob_show_probability],
        )
        self._style_checkbuttons(self.chk_prob_opts)
        self.chk_prob_opts.on_clicked(self.on_prob_option_toggle)

    def _remove_sidebar(self) -> None:
        for attr in ["ax_sidebar_bg", "ax_det_main", "ax_det_opts", "ax_prob_main", "ax_prob_opts"]:
            ax = getattr(self, attr, None)
            if ax is not None:
                ax.remove()
                setattr(self, attr, None)

        for text in self.sidebar_texts:
            text.remove()
        self.sidebar_texts = []

        self.chk_det_main = None
        self.chk_det_opts = None
        self.chk_prob_main = None
        self.chk_prob_opts = None

    def _set_sidebar_visible(self, visible: bool) -> None:
        if visible == self.sidebar_visible:
            return

        if visible:
            self._create_sidebar()
        else:
            self._remove_sidebar()

        self.sidebar_visible = visible
        self.btn_toggle_sidebar.label.set_text("Hide menu" if visible else "Show menu")
        self._create_dynamic_axes()
        self.render()

    def _create_dynamic_axes(self) -> None:
        if self.ax_det is not None:
            self.ax_det.remove()
            self.ax_det = None

        if self.ax_prob is not None:
            self.ax_prob.remove()
            self.ax_prob = None

        left = 0.26 if self.sidebar_visible else 0.05
        width_total = 0.69 if self.sidebar_visible else 0.90
        visible_count = int(self.show_det_panel) + int(self.show_prob_panel)

        if visible_count == 2:
            panel_width = width_total * 0.42
            self.ax_det = self.fig.add_axes([left, 0.20, panel_width, 0.60])
            self.ax_prob = self.fig.add_axes([left + width_total * 0.54, 0.20, panel_width, 0.60])

        elif visible_count == 1:
            if self.show_det_panel:
                self.ax_det = self.fig.add_axes([left, 0.18, width_total, 0.64])
            elif self.show_prob_panel:
                self.ax_prob = self.fig.add_axes([left, 0.18, width_total, 0.64])

    def build_panel_subtitle(self, history: list[np.ndarray]) -> str:
        state = history[self.step_index]
        infected = int(state.sum())

        if self.step_index == 0:
            changed = []
        else:
            changed = changed_nodes(history[self.step_index - 1], state)

        return f"step = {self.step_index}, infected = {infected}/{len(state)}, changed = {changed}"

    def render(self) -> None:
        visible_count = int(self.show_det_panel) + int(self.show_prob_panel)

        if visible_count == 0:
            if self.ax_det is not None:
                self.ax_det.clear()
            if self.ax_prob is not None:
                self.ax_prob.clear()

            self.info_text.set_text("No model selected.")
            self.fig.canvas.draw()
            return

        self.info_text.set_text(
            f"Graph type: {self.config.graph_type} | beta = {self.config.beta} | "
            f"initial infected = {self.config.initial_infected} | step = {self.step_index}/{self.max_step}"
        )

        if self.ax_det is not None:
            det_state = self.det_history[self.step_index]
            det_influence = self.A @ det_state
            det_annotations = build_det_annotations(det_influence, self.det_show_influence)

            draw_panel(
                self.ax_det,
                self.graph,
                self.pos,
                det_state,
                self.labels,
                "Deterministic movable model",
                self.build_panel_subtitle(self.det_history),
                top_annotations=det_annotations if det_annotations else None,
            )

        if self.ax_prob is not None:
            prob_state = self.prob_history[self.step_index]
            prob_influence = self.A @ prob_state
            prob_probs = 1.0 - (1.0 - self.config.beta) ** prob_influence

            prob_annotations = build_prob_annotations(
                prob_influence,
                prob_probs,
                show_influence=self.prob_show_influence,
                show_probs=self.prob_show_probability,
            )

            draw_panel(
                self.ax_prob,
                self.graph,
                self.pos,
                prob_state,
                self.labels,
                "Probabilistic movable model",
                self.build_panel_subtitle(self.prob_history),
                top_annotations=prob_annotations if prob_annotations else None,
            )

        self.fig.canvas.draw()

    def on_toggle_sidebar(self, _event) -> None:
        self._set_sidebar_visible(not self.sidebar_visible)

    def on_det_main_toggle(self, _label) -> None:
        if self.chk_det_main is not None:
            self.show_det_panel = self.chk_det_main.get_status()[0]
        self._create_dynamic_axes()
        self.render()

    def on_prob_main_toggle(self, _label) -> None:
        if self.chk_prob_main is not None:
            self.show_prob_panel = self.chk_prob_main.get_status()[0]
        self._create_dynamic_axes()
        self.render()

    def on_det_option_toggle(self, _label) -> None:
        if self.chk_det_opts is not None:
            self.det_show_influence = self.chk_det_opts.get_status()[0]
        self.render()

    def on_prob_option_toggle(self, _label) -> None:
        if self.chk_prob_opts is not None:
            statuses = self.chk_prob_opts.get_status()
            self.prob_show_influence = statuses[0]
            self.prob_show_probability = statuses[1]
        self.render()

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