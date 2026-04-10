from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


GraphType = Literal[
    "line",
    "star",
    "cycle",
    "erdos_renyi",
    "custom",
    "binary_tree",
    "grid_2d",
    "wheel",
    "ladder",
]


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