from __future__ import annotations

from typing import Callable, Optional

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
import networkx as nx
import numpy as np

from config import AppConfig
from graph_builder import GraphBuilder
from visualization import GraphVisualizer


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
        on_graph_change: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.graph = graph
        self.A = A
        self.det_history = det_history
        self.prob_history = prob_history
        self.pos = pos
        self.labels = labels
        self.config = config
        self.on_graph_change = on_graph_change

        self.step_index = 0
        self.max_step = len(det_history) - 1

        self.sidebar_visible = True

        self.show_det_panel = True
        self.show_prob_panel = True
        self.det_show_influence = False
        self.prob_show_influence = False
        self.prob_show_probability = False

        # Sidebar geometry
        self.sidebar_left = 0.015
        self.sidebar_bottom = 0.02
        self.sidebar_width = 0.235
        self.sidebar_height = 0.86

        self.sidebar_title_x = 0.035
        self.sidebar_main_x = 0.055
        self.sidebar_main_w = 0.17
        self.sidebar_sub_x = 0.075
        self.sidebar_sub_w = 0.15

        self.fig = plt.figure(figsize=(15, 8))
        self.fig.patch.set_facecolor("#f8f9fb")
        self.fig.suptitle(
            "Infection spread on a graph: adjacency operator-based simulation",
            fontsize=16,
            y=0.985,
            color="#111111",
            fontweight="semibold",
        )

        self.sidebar_texts: list = []
        self.graph_buttons: list[Button] = []
        self.graph_button_axes: list = []

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

    def update_data(
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
        self.render()

    def _create_static_widgets(self) -> None:
        self.ax_toggle_sidebar = self.fig.add_axes([0.015, 0.93, 0.10, 0.055])
        self.btn_toggle_sidebar = Button(
            self.ax_toggle_sidebar,
            "Hide menu",
            color="#e9ecef",
            hovercolor="#dfe6ec",
        )
        self.btn_toggle_sidebar.on_clicked(self.on_toggle_sidebar)

        self.ax_toggle_sidebar.set_facecolor("#e9ecef")
        for spine in self.ax_toggle_sidebar.spines.values():
            spine.set_edgecolor("#c7ced6")
            spine.set_linewidth(1.0)

        self.info_text = self.fig.text(
            0.18,
            0.93,
            "",
            fontsize=12,
            va="top",
            color="#222222",
        )

        self.ax_prev = self.fig.add_axes([0.36, 0.03, 0.10, 0.065])
        self.ax_next = self.fig.add_axes([0.48, 0.03, 0.10, 0.065])
        self.ax_reset = self.fig.add_axes([0.60, 0.03, 0.10, 0.065])

        self.btn_prev = Button(
            self.ax_prev,
            "Previous",
            color="#e9ecef",
            hovercolor="#dfe6ec",
        )
        self.btn_next = Button(
            self.ax_next,
            "Next",
            color="#e9ecef",
            hovercolor="#dfe6ec",
        )
        self.btn_reset = Button(
            self.ax_reset,
            "Reset",
            color="#e9ecef",
            hovercolor="#dfe6ec",
        )

        for ax in [self.ax_prev, self.ax_next, self.ax_reset]:
            ax.set_facecolor("#e9ecef")
            for spine in ax.spines.values():
                spine.set_edgecolor("#c7ced6")
                spine.set_linewidth(1.0)

        self.btn_prev.on_clicked(self.on_previous)
        self.btn_next.on_clicked(self.on_next)
        self.btn_reset.on_clicked(self.on_reset)

    def _style_checkbuttons(self, check: CheckButtons) -> None:
        for rect in getattr(check, "rectangles", []):
            rect.set_facecolor("#ffffff")
            rect.set_edgecolor("#444444")
            rect.set_linewidth(1.0)

        for lines in getattr(check, "lines", []):
            for line in lines:
                line.set_color("#111111")
                line.set_linewidth(2.0)

        for label in getattr(check, "labels", []):
            label.set_fontsize(10)
            label.set_color("#222222")

    def _style_sidebar_option_axis(self, ax) -> None:
        ax.set_facecolor("none")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _style_button(self, ax, button: Button, selected: bool = False) -> None:
        ax.set_facecolor("#eef6ff" if selected else "#ffffff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#8bbcff" if selected else "#444444")
            spine.set_linewidth(1.4 if selected else 1.1)
        button.label.set_fontsize(10)
        button.label.set_color("#222222")

    def _create_graph_buttons(self, title_y: float, start_y: float) -> None:
        self.sidebar_texts.append(
            self.fig.text(
                self.sidebar_title_x,
                title_y,
                "Choose graph",
                fontsize=15,
                fontweight="bold",
                color="#111111",
            )
        )

        button_height = 0.030
        button_gap = 0.006

        for index, (label, graph_type) in enumerate(GraphBuilder.PREDEFINED_GRAPH_OPTIONS):
            y = start_y - index * (button_height + button_gap)
            ax = self.fig.add_axes([self.sidebar_main_x, y, self.sidebar_main_w, button_height])

            btn = Button(ax, label)
            self._style_button(ax, btn, selected=(self.config.graph_type == graph_type))

            def make_callback(gt: str):
                return lambda _event: self.on_graph_selected(gt)

            btn.on_clicked(make_callback(graph_type))

            self.graph_button_axes.append(ax)
            self.graph_buttons.append(btn)

    def _refresh_graph_button_styles(self) -> None:
        for (_, graph_type), ax, btn in zip(
            GraphBuilder.PREDEFINED_GRAPH_OPTIONS,
            self.graph_button_axes,
            self.graph_buttons,
        ):
            self._style_button(ax, btn, selected=(self.config.graph_type == graph_type))

    def _create_sidebar(self) -> None:
        self.ax_sidebar_bg = self.fig.add_axes(
            [self.sidebar_left, self.sidebar_bottom, self.sidebar_width, self.sidebar_height]
        )
        self.ax_sidebar_bg.set_facecolor("#f4f6f8")
        self.ax_sidebar_bg.set_xticks([])
        self.ax_sidebar_bg.set_yticks([])
        for spine in self.ax_sidebar_bg.spines.values():
            spine.set_visible(False)

        # Vertical layout tuned so all blocks stay inside the grey background
        display_title_y = 0.84

        det_main_y = 0.745
        det_opts_y = 0.695

        prob_main_y = 0.605
        prob_opts_y = 0.535

        graph_title_y = 0.355
        graph_buttons_start_y = 0.275

        self.sidebar_texts = [
            self.fig.text(
                self.sidebar_title_x,
                display_title_y,
                "Display options",
                fontsize=15,
                fontweight="bold",
                color="#111111",
            ),
        ]

        self.ax_det_main = self.fig.add_axes([self.sidebar_main_x, det_main_y, self.sidebar_main_w, 0.04])
        self._style_sidebar_option_axis(self.ax_det_main)
        self.chk_det_main = CheckButtons(
            self.ax_det_main,
            ["Deterministic model"],
            [self.show_det_panel],
        )
        self._style_checkbuttons(self.chk_det_main)
        self.chk_det_main.on_clicked(self.on_det_main_toggle)

        self.ax_det_opts = self.fig.add_axes([self.sidebar_sub_x, det_opts_y, self.sidebar_sub_w, 0.04])
        self._style_sidebar_option_axis(self.ax_det_opts)
        self.chk_det_opts = CheckButtons(
            self.ax_det_opts,
            ["show A @ x_t"],
            [self.det_show_influence],
        )
        self._style_checkbuttons(self.chk_det_opts)
        self.chk_det_opts.on_clicked(self.on_det_option_toggle)

        self.ax_prob_main = self.fig.add_axes([self.sidebar_main_x, prob_main_y, self.sidebar_main_w, 0.04])
        self._style_sidebar_option_axis(self.ax_prob_main)
        self.chk_prob_main = CheckButtons(
            self.ax_prob_main,
            ["Probabilistic model"],
            [self.show_prob_panel],
        )
        self._style_checkbuttons(self.chk_prob_main)
        self.chk_prob_main.on_clicked(self.on_prob_main_toggle)

        self.ax_prob_opts = self.fig.add_axes([self.sidebar_sub_x, prob_opts_y, self.sidebar_sub_w, 0.08])
        self._style_sidebar_option_axis(self.ax_prob_opts)
        self.chk_prob_opts = CheckButtons(
            self.ax_prob_opts,
            ["show A @ x_t", "show probability"],
            [self.prob_show_influence, self.prob_show_probability],
        )
        self._style_checkbuttons(self.chk_prob_opts)
        self.chk_prob_opts.on_clicked(self.on_prob_option_toggle)

        self._create_graph_buttons(
            title_y=graph_title_y,
            start_y=graph_buttons_start_y,
        )

    def _remove_sidebar(self) -> None:
        for attr in [
            "ax_sidebar_bg",
            "ax_det_main",
            "ax_det_opts",
            "ax_prob_main",
            "ax_prob_opts",
        ]:
            ax = getattr(self, attr, None)
            if ax is not None:
                ax.remove()
                setattr(self, attr, None)

        for ax in self.graph_button_axes:
            ax.remove()
        self.graph_button_axes = []
        self.graph_buttons = []

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
            self.ax_prob = self.fig.add_axes(
                [left + width_total * 0.54, 0.20, panel_width, 0.60]
            )
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
            changed = GraphVisualizer.changed_nodes(history[self.step_index - 1], state)

        return (
            f"step = {self.step_index}, infected = {infected}/{len(state)}, "
            f"changed = {changed}"
        )

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
            f"initial infected = {self.config.initial_infected} | "
            f"step = {self.step_index}/{self.max_step}"
        )

        self._refresh_graph_button_styles()

        if self.ax_det is not None:
            det_state = self.det_history[self.step_index]
            det_influence = self.A @ det_state
            det_annotations = GraphVisualizer.build_det_annotations(
                det_influence,
                self.det_show_influence,
            )

            GraphVisualizer.draw_panel(
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

            prob_annotations = GraphVisualizer.build_prob_annotations(
                prob_influence,
                prob_probs,
                show_influence=self.prob_show_influence,
                show_probs=self.prob_show_probability,
            )

            GraphVisualizer.draw_panel(
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

    def on_graph_selected(self, graph_type: str) -> None:
        if self.on_graph_change is not None and graph_type != self.config.graph_type:
            self.on_graph_change(graph_type)

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