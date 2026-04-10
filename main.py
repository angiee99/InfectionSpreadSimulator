from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx

from config import AppConfig
from graph_builder import GraphBuilder
from simulation import InfectionSimulator
from viewer import StepViewer
from visualization import GraphVisualizer


def prepare_simulation_data(config: AppConfig):
    graph = GraphBuilder.create_graph(config)

    config.num_nodes = graph.number_of_nodes()
    config.initial_infected = GraphBuilder.get_initial_infected_for_graph(graph)

    # default labels from node ids for predefined graphs
    if config.graph_type == "custom":
        labels = GraphVisualizer.build_labels(config, config.num_nodes)
    else:
        config.node_labels = None
        labels = GraphBuilder.build_labels_for_graph(graph)

    initial_state = InfectionSimulator.create_initial_state(config.num_nodes, config.initial_infected)

    A = nx.to_numpy_array(graph, dtype=int)
    pos = nx.spring_layout(graph, seed=config.layout_seed)

    det_history = InfectionSimulator.run_deterministic_history(A, initial_state, config.steps)
    prob_history = InfectionSimulator.run_probabilistic_history(
        A,
        initial_state,
        config.steps,
        beta=config.beta,
        random_seed=config.random_seed,
    )

    return graph, A, det_history, prob_history, pos, labels


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

    graph, A, det_history, prob_history, pos, labels = prepare_simulation_data(config)

    viewer = None

    def handle_graph_change(graph_type: str) -> None:
        nonlocal viewer, config

        config.graph_type = graph_type

        if graph_type == "custom":
            config.num_nodes = 10
            config.custom_edges = [
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
            config.node_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

        graph, A, det_history, prob_history, pos, labels = prepare_simulation_data(config)

        viewer.update_data(
            graph=graph,
            A=A,
            det_history=det_history,
            prob_history=prob_history,
            pos=pos,
            labels=labels,
            config=config,
        )

    viewer = StepViewer(
        graph=graph,
        A=A,
        det_history=det_history,
        prob_history=prob_history,
        pos=pos,
        labels=labels,
        config=config,
        on_graph_change=handle_graph_change,
    )

    plt.show()


if __name__ == "__main__":
    main()