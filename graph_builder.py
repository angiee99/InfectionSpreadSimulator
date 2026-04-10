from __future__ import annotations

import networkx as nx

from config import AppConfig


class GraphBuilder:
    PREDEFINED_GRAPH_OPTIONS = [
        ("Custom", "custom"),
        ("Line", "line"),
        ("Star", "star"),
        ("Cycle", "cycle"),
        ("Binary tree", "binary_tree"),
        ("2D grid", "grid_2d"),
        ("Erdos-Renyi", "erdos_renyi")
    ]

    @staticmethod
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

        elif config.graph_type == "binary_tree":
            # height=3 gives 15 nodes
            graph = nx.balanced_tree(r=2, h=3)

        elif config.graph_type == "grid_2d":
            # 3x4 grid = 12 nodes, relabeled to integers
            g2 = nx.grid_2d_graph(3, 4)
            graph = nx.convert_node_labels_to_integers(g2, ordering="sorted")

        elif config.graph_type == "wheel":
            graph = nx.wheel_graph(10)

        elif config.graph_type == "ladder":
            graph = nx.ladder_graph(6)

        else:
            raise ValueError(f"Unsupported graph_type: {config.graph_type}")

        return graph

    @staticmethod
    def get_initial_infected_for_graph(graph: nx.Graph) -> list[int]:
        if graph.number_of_nodes() == 0:
            return []
        return [0]

    @staticmethod
    def build_labels_for_graph(graph: nx.Graph) -> dict[int, str]:
        return {node: str(node) for node in graph.nodes()}