import networkx as nx

from config import AppConfig


class GraphBuilder:
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

        else:
            raise ValueError(f"Unsupported graph_type: {config.graph_type}")

        return graph