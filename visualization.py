from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from config import AppConfig


class GraphVisualizer:
    INFECTED_COLOR = "#ff4d4f"
    HEALTHY_COLOR = "#7ec8e3"

    @staticmethod
    def build_labels(config: AppConfig, num_nodes: int) -> dict[int, str]:
        if config.node_labels is None:
            return {i: str(i) for i in range(num_nodes)}

        if len(config.node_labels) != num_nodes:
            raise ValueError("node_labels must have the same length as num_nodes")

        return {i: label for i, label in enumerate(config.node_labels)}

    @staticmethod
    def state_to_colors(state: np.ndarray) -> list[str]:
        return [
            GraphVisualizer.INFECTED_COLOR if value == 1 else GraphVisualizer.HEALTHY_COLOR
            for value in state
        ]

    @staticmethod
    def changed_nodes(prev_state: np.ndarray, curr_state: np.ndarray) -> list[int]:
        return [i for i in range(len(curr_state)) if prev_state[i] != curr_state[i]]

    @staticmethod
    def build_det_annotations(influence: np.ndarray, show_influence: bool) -> dict[int, str]:
        annotations = {}
        if show_influence:
            for i, value in enumerate(influence):
                if value > 0:
                    annotations[i] = f"A@x={int(value)}"
        return annotations

    @staticmethod
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

    @staticmethod
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
            node_color=GraphVisualizer.state_to_colors(state),
            node_size=760,
            edge_color="#8c8c8c",
            width=1.3,
            linewidths=1.0,
            font_weight="bold",
            font_size=11,
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
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="white",
                        ec="none",
                        alpha=0.8,
                    ),
                    zorder=5,
                )

        ax.set_title(f"{title}\n{subtitle}", fontsize=12, color="#222222", pad=10)
        ax.axis("off")