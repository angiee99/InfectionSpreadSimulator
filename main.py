# main.py

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx

from config import AppConfig
from graph_builder import GraphBuilder
from simulation import InfectionSimulator
from viewer import StepViewer
from visualization import GraphVisualizer


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

    graph = GraphBuilder.create_graph(config)
    initial_state = InfectionSimulator.create_initial_state(config.num_nodes, config.initial_infected)
    labels = GraphVisualizer.build_labels(config, config.num_nodes)

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