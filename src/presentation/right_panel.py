"""Right sidebar panel — Workspace: layers, view options.

Provides layer management (selection, visibility toggle, lock toggle),
view options (show image, show other layers, show missing pixels), and
AI auto-labeling controls (plugin selection and run button).
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from domain.layer_config import LayerConfig
from application.app_state import ToolbarState
from presentation.style import (
    PRIMARY,
    ON_PRIMARY,
    ON_SURFACE,
    ON_SURFACE_VARIANT,
    OUTLINE_VARIANT,
    SURFACE_CONTAINER_HIGH,
    SURFACE_CONTAINER_HIGHEST,
    FONT_SIZE_SM,
    FONT_SIZE_XS,
    SIDEBAR_WIDTH,
)


# (LayerMappingDialog has been moved to presentation.model_config_dialog)


# ------------------------------------------------------------------
# Slider with click-to-position + drag support
# ------------------------------------------------------------------

class _OpacitySlider(QSlider):
    """QSlider que salta al punto clicado y permite arrastrar normalmente."""

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            if self.orientation() == Qt.Orientation.Horizontal:
                val = self.minimum() + (self.maximum() - self.minimum()) * event.position().x() / self.width()
            else:
                val = self.maximum() - (self.maximum() - self.minimum()) * event.position().y() / self.height()
            self.setValue(int(round(val)))
        super().mousePressEvent(event)


# ------------------------------------------------------------------
# Layer row widget
# ------------------------------------------------------------------

class _LayerRow(QFrame):
    """Single layer row with visibility, lock, name, and colored left border."""

    def __init__(
        self,
        index: int,
        config: LayerConfig,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._index = index
        self._config = config
        self._selected = False
        self._visible = True
        self._locked = False

        self._cb_selected: Optional[Callable[[int], None]] = None
        self._cb_visibility: Optional[Callable[[int, bool], None]] = None
        self._cb_lock: Optional[Callable[[int], None]] = None

        self.setFixedHeight(36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("LayerRow")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 4, 0)
        layout.setSpacing(4)

        # Name label (left, stretches)
        label_text = f"{index + 1}. {config.name.capitalize()}"
        self._name_label = QLabel(label_text)
        self._name_label.setStyleSheet(
            f"color: {ON_SURFACE}; font-size: {FONT_SIZE_SM}px; "
            f"font-weight: 500;"
        )
        layout.addWidget(self._name_label, 1)

        # Visibility button (right side)
        self._vis_btn = QPushButton("\U0001F441")
        self._vis_btn.setFixedSize(28, 28)
        self._vis_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._vis_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._vis_btn.setToolTip("Toggle visibility")
        self._vis_btn.clicked.connect(self._toggle_visibility)
        layout.addWidget(self._vis_btn)

        # Lock button (right side)
        self._lock_btn = QPushButton("\U0001F512")
        self._lock_btn.setFixedSize(28, 28)
        self._lock_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._lock_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._lock_btn.setToolTip("Toggle lock")
        self._lock_btn.clicked.connect(self._toggle_lock)
        layout.addWidget(self._lock_btn)

        self._update_btn_styles()
        self._apply_style()

    def on_selected(self, cb: Callable[[int], None]):
        self._cb_selected = cb

    def on_visibility_changed(self, cb: Callable[[int, bool], None]):
        self._cb_visibility = cb

    def on_lock_toggled(self, cb: Callable[[int], None]):
        self._cb_lock = cb

    def set_selected(self, selected: bool):
        self._selected = selected
        self._apply_style()

    def set_visible(self, visible: bool):
        self._visible = visible
        self._update_btn_styles()
        self._apply_style()

    def set_locked(self, locked: bool):
        self._locked = locked
        self._update_btn_styles()
        self._apply_style()

    @property
    def is_locked(self) -> bool:
        return self._locked

    def mousePressEvent(self, event):
        self._fire_selected()

    def _fire_selected(self):
        if self._cb_selected:
            self._cb_selected(self._index)

    def _toggle_visibility(self):
        self._visible = not self._visible
        self._update_btn_styles()
        self._apply_style()
        if self._cb_visibility:
            self._cb_visibility(self._index, self._visible)

    def _toggle_lock(self):
        self._locked = not self._locked
        self._update_btn_styles()
        if self._cb_lock:
            self._cb_lock(self._index)

    def _btn_style(self, active: bool) -> str:
        bg = "rgba(161, 250, 255, 0.25)" if active else "transparent"
        hover_bg = "rgba(161, 250, 255, 0.40)" if active else "rgba(161, 250, 255, 0.15)"
        return (
            f"QPushButton {{ background: {bg}; border: none; "
            f"font-size: 16px; padding: 0; border-radius: 4px; }}"
            f"QPushButton:hover {{ background: {hover_bg}; }}"
        )

    def _update_btn_styles(self):
        self._vis_btn.setStyleSheet(self._btn_style(self._visible))
        self._lock_btn.setStyleSheet(self._btn_style(self._locked))

    def _apply_style(self):
        color = self._config.color_hex
        # Row frame: always transparent background, only the left border stripe
        self.setStyleSheet(f"""
            QFrame#LayerRow {{
                background-color: transparent;
                border-left: 4px solid {color};
                border-radius: 8px;
                border-top-left-radius: 0; border-bottom-left-radius: 0;
            }}
        """)
        if not self._visible:
            self._name_label.setStyleSheet(
                f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_SM}px;"
                f" font-weight: 400; background: transparent; border-radius: 4px;"
                f" padding: 1px 4px;"
            )
        elif self._selected:
            self._name_label.setStyleSheet(
                f"color: {ON_PRIMARY}; font-size: {FONT_SIZE_SM}px; font-weight: 700;"
                f" background-color: {PRIMARY}; border-radius: 4px; padding: 1px 4px;"
            )
        else:
            self._name_label.setStyleSheet(
                f"color: {ON_SURFACE}; font-size: {FONT_SIZE_SM}px; font-weight: 500;"
                f" background: transparent; border-radius: 4px; padding: 1px 4px;"
            )


# ------------------------------------------------------------------
# RightPanel
# ------------------------------------------------------------------

class RightPanel(QWidget):
    """Right sidebar with layers, view options, and auto-labeling."""

    def __init__(
        self,
        layer_configs: list[LayerConfig],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._layer_configs = layer_configs
        self.setFixedWidth(SIDEBAR_WIDTH)
        self.setStyleSheet(f"background-color: {SURFACE_CONTAINER_HIGH};")

        self._cb_layer_selected: Optional[Callable[[int], None]] = None
        self._cb_open_project: Optional[Callable[[], None]] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Project header (sticky top, above scroll area)
        self._build_project_header()
        outer.addWidget(self._project_header_widget, 0)

        # Scrollable area for layers + view options
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_content = QWidget()
        self._layout = QVBoxLayout(scroll_content)
        self._layout.setContentsMargins(14, 16, 14, 16)
        self._layout.setSpacing(12)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(scroll_content)
        outer.addWidget(scroll, 1)
        self._scroll_area = scroll

        self._build_layers_section(layer_configs)
        self._build_opacity_section()
        self._build_view_options()
        self._layout.addStretch()

        # Nothing to interact with until an image is loaded
        self.set_image_loaded(False)

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def set_image_loaded(self, loaded: bool) -> None:
        """Enable or disable everything below the project header."""
        self._scroll_area.setEnabled(loaded)

    def on_layer_selected(self, cb: Callable[[int], None]) -> None:
        self._cb_layer_selected = cb

    def on_gallery_clicked(self, cb: Callable[[], None]) -> None:
        self._cb_gallery_toggle = cb

    def on_open_project(self, cb: Callable[[], None]) -> None:
        self._cb_open_project = cb

    def on_show_image_changed(self, cb: Callable[[bool], None]) -> None:
        self._q_show_image.clicked.connect(lambda: cb(self._show_image_active))

    def on_show_missing_pixels_changed(self, cb: Callable[[bool], None]) -> None:
        self._q_missing_pixels.clicked.connect(lambda: cb(self._missing_pixels_active))

    def on_show_grid_changed(self, cb: Callable[[bool], None]) -> None:
        self._q_show_grid.clicked.connect(lambda: cb(self._grid_visible_active))

    def on_layer_visibility_changed(self, cb: Callable[[int, bool], None]) -> None:
        for row in self._layer_rows:
            row.on_visibility_changed(cb)

    def on_layer_lock_toggled(self, cb: Callable[[int], None]) -> None:
        for row in self._layer_rows:
            row.on_lock_toggled(cb)

    def on_toggle_all_visibility(self, cb: Callable[[], None]) -> None:
        self._cb_toggle_all_visibility = cb

    def on_toggle_all_lock(self, cb: Callable[[], None]) -> None:
        self._cb_toggle_all_lock = cb

    def on_opacity_changed(self, cb: Callable[[float], None]) -> None:
        self._cb_opacity_changed = cb

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def set_active_layer(self, layer_index: int) -> None:
        for i, row in enumerate(self._layer_rows):
            row.set_selected(i == layer_index)

    def set_locked_layers(self, locked: set) -> None:
        for i, row in enumerate(self._layer_rows):
            row.set_locked(i in locked)

    def sync(self, state: ToolbarState) -> None:
        """Apply ToolbarState to the right panel."""
        self.set_active_layer(state.active_layer)
        self.set_locked_layers(state.locked_layers)
        for i, row in enumerate(self._layer_rows):
            row.set_visible(i not in state.hidden_layers)

        # View toggles
        self._show_image_active = state.show_image
        self._update_toggle_style(self._q_show_image, state.show_image)

        self._missing_pixels_active = state.show_missing_pixels
        self._update_toggle_style(self._q_missing_pixels, state.show_missing_pixels)

        self._grid_visible_active = state.show_grid
        self._update_toggle_style(self._q_show_grid, state.show_grid)

        self._q_opacity_slider.blockSignals(True)
        self._q_opacity_slider.setValue(int(state.global_layer_opacity * 100))
        self._q_opacity_slider.blockSignals(False)
        self._q_opacity_value_label.setText(f"{int(state.global_layer_opacity * 100)}%")

    def set_project_name(self, name: str) -> None:
        """Update the displayed project name."""
        self._project_name_label.setText(name or "—")

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_project_header(self) -> None:
        """Build the sticky project section header at the top."""
        self._project_header_widget = QWidget()
        self._project_header_widget.setStyleSheet(
            f"background-color: {SURFACE_CONTAINER_HIGH};"
        )
        outer = QVBoxLayout(self._project_header_widget)
        outer.setContentsMargins(14, 8, 8, 8)
        outer.setSpacing(4)

        # Top row: "Project" section label + open-project button
        top_row = QWidget()
        top_row.setStyleSheet("background: transparent; border: none;")
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        section_lbl = QLabel("PROJECT")
        section_lbl.setStyleSheet(
            f"color: {PRIMARY}; font-size: {FONT_SIZE_SM}px; "
            f"font-weight: 700; letter-spacing: 1px;"
        )
        top_layout.addWidget(section_lbl)
        top_layout.addStretch()

        open_btn = QPushButton("📂")
        open_btn.setFixedSize(28, 28)
        open_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_btn.setToolTip("Open another project folder")
        open_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; "
            f"border-radius: 6px; font-size: 14px; padding: 0; }}"
            f"QPushButton:hover {{ background: rgba(255,255,255,0.07); }}"
        )
        open_btn.clicked.connect(self._on_open_project_clicked)
        top_layout.addWidget(open_btn)
        outer.addWidget(top_row)

        # Project name label (read-only — always the folder name)
        self._project_name_label = QLabel("—")
        self._project_name_label.setStyleSheet(
            f"color: {ON_SURFACE}; font-size: {FONT_SIZE_SM}px; "
            f"border: none; padding-bottom: 2px;"
        )
        outer.addWidget(self._project_name_label)

        # Gallery toggle button
        self._cb_gallery_toggle: Optional[Callable[[], None]] = None
        gallery_btn = QPushButton("\U0001f5bc  Gallery")
        gallery_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        gallery_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        gallery_btn.setFixedHeight(28)
        gallery_btn.setToolTip("Show / hide image gallery")
        gallery_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; "
            f"border-radius: 6px; color: {ON_SURFACE_VARIANT}; "
            f"font-size: {FONT_SIZE_SM}px; font-weight: 600; "
            f"text-align: left; padding: 0 4px; }}"
            f"QPushButton:hover {{ background: rgba(161,250,255,0.1); color: {PRIMARY}; }}"
        )
        gallery_btn.clicked.connect(lambda: self._cb_gallery_toggle and self._cb_gallery_toggle())
        self._q_gallery_btn = gallery_btn
        outer.addWidget(gallery_btn)

    def _on_open_project_clicked(self) -> None:
        if self._cb_open_project:
            self._cb_open_project()

    def _build_layers_section(self, layer_configs: list[LayerConfig]) -> None:
        self._cb_toggle_all_visibility: Optional[Callable[[], None]] = None
        self._cb_toggle_all_lock: Optional[Callable[[], None]] = None

        # Layers header (text + count)
        h_row = QHBoxLayout()
        h_row.setContentsMargins(0, 0, 0, 4)
        title = QLabel("LAYERS")
        title.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; "
            f"font-weight: 700; letter-spacing: 1.5px;"
        )
        count = QLabel(str(len(layer_configs)))
        count.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; font-weight: 600;"
        )
        h_row.addWidget(title)
        h_row.addStretch()
        h_row.addWidget(count)
        self._layout.addLayout(h_row)

        # "All layers" control row — buttons aligned with individual layer rows
        _all_btn_style = (
            "QPushButton { background: transparent; border: none; "
            "font-size: 16px; padding: 0; border-radius: 4px; }"
            "QPushButton:hover { background: rgba(161, 250, 255, 0.15); }"
        )
        all_row = QFrame()
        all_row.setFixedHeight(30)
        all_layout = QHBoxLayout(all_row)
        all_layout.setContentsMargins(6, 0, 4, 0)
        all_layout.setSpacing(4)
        all_lbl = QLabel("All layers")
        all_lbl.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; font-weight: 500;"
        )
        all_layout.addWidget(all_lbl, 1)

        self._vis_all_btn = QPushButton("\U0001F441")
        self._vis_all_btn.setFixedSize(28, 28)
        self._vis_all_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._vis_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._vis_all_btn.setToolTip("Show/Hide all layers (V)")
        self._vis_all_btn.setStyleSheet(_all_btn_style)
        self._vis_all_btn.clicked.connect(self._on_toggle_all_visibility_clicked)
        all_layout.addWidget(self._vis_all_btn)

        self._lock_all_btn = QPushButton("\U0001F512")
        self._lock_all_btn.setFixedSize(28, 28)
        self._lock_all_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._lock_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._lock_all_btn.setToolTip("Lock/Unlock all layers (L)")
        self._lock_all_btn.setStyleSheet(_all_btn_style)
        self._lock_all_btn.clicked.connect(self._on_toggle_all_lock_clicked)
        all_layout.addWidget(self._lock_all_btn)

        self._layout.addWidget(all_row)

        # Layer rows
        self._layer_rows: list[_LayerRow] = []
        for i, lc in enumerate(layer_configs):
            row = _LayerRow(i, lc)
            row.on_selected(self._on_layer_row_selected)
            self._layer_rows.append(row)
            self._layout.addWidget(row)

        if self._layer_rows:
            self._layer_rows[0].set_selected(True)

    def _build_opacity_section(self) -> None:
        self._cb_opacity_changed: Optional[Callable[[float], None]] = None

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {OUTLINE_VARIANT}; max-height: 1px; margin-top: 4px;")
        self._layout.addWidget(sep)

        lbl_header = QLabel("LAYER OPACITY")
        lbl_header.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; "
            f"font-weight: 700; letter-spacing: 1.5px; margin-top: 4px;"
        )
        self._layout.addWidget(lbl_header)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        lbl_zero = QLabel("0")
        lbl_zero.setStyleSheet(f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px;")
        self._q_opacity_value_label = QLabel("70%")
        self._q_opacity_value_label.setStyleSheet(
            f"color: {ON_SURFACE}; font-size: {FONT_SIZE_XS}px; font-weight: 600;"
        )
        lbl_full = QLabel("100")
        lbl_full.setStyleSheet(f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px;")
        row.addWidget(lbl_zero)
        row.addStretch()
        row.addWidget(self._q_opacity_value_label)
        row.addStretch()
        row.addWidget(lbl_full)
        self._layout.addLayout(row)

        self._q_opacity_slider = _OpacitySlider(Qt.Orientation.Horizontal)
        self._q_opacity_slider.setRange(0, 100)
        self._q_opacity_slider.setValue(70)
        self._q_opacity_slider.setToolTip("Global annotation layer opacity (0 = invisible, 100 = solid)")
        self._q_opacity_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._q_opacity_slider.valueChanged.connect(self._on_opacity_slider_changed)
        self._layout.addWidget(self._q_opacity_slider)

    def _on_opacity_slider_changed(self, value: int) -> None:
        self._q_opacity_value_label.setText(f"{value}%")
        if self._cb_opacity_changed is not None:
            self._cb_opacity_changed(value / 100.0)

    def _build_view_options(self) -> None:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {OUTLINE_VARIANT}; max-height: 1px; margin-top: 4px;")
        self._layout.addWidget(sep)

        workspace_lbl = QLabel("WORKSPACE")
        workspace_lbl.setStyleSheet(
            f"color: {PRIMARY}; font-size: {FONT_SIZE_SM}px; "
            f"border-top: 1px solid {OUTLINE_VARIANT}; padding-top: 12px; "
            f"font-weight: 700; letter-spacing: 1px;"
        )
        self._layout.addWidget(workspace_lbl)

        lbl = QLabel("VIEW OPTIONS")
        lbl.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; "
            f"font-weight: 700; letter-spacing: 1.5px; margin-top: 4px;"
        )
        self._layout.addWidget(lbl)

        self._show_image_active = True
        self._missing_pixels_active = False

        self._q_show_image = self._create_toggle_button("👁  Show Image (I)", True)
        self._q_show_image.clicked.connect(self._toggle_show_image)
        self._layout.addWidget(self._q_show_image)

        self._q_missing_pixels = self._create_toggle_button("⚠  Missing Pixels (M)", False)
        self._q_missing_pixels.clicked.connect(self._toggle_missing_pixels)
        self._layout.addWidget(self._q_missing_pixels)

        self._grid_visible_active = False
        self._q_show_grid = self._create_toggle_button("⊞  Show Grid (G)", False)
        self._q_show_grid.clicked.connect(self._toggle_show_grid)
        self._layout.addWidget(self._q_show_grid)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_layer_row_selected(self, index: int) -> None:
        for i, row in enumerate(self._layer_rows):
            row.set_selected(i == index)
        if self._cb_layer_selected:
            self._cb_layer_selected(index)

    def _on_toggle_all_visibility_clicked(self) -> None:
        if self._cb_toggle_all_visibility:
            self._cb_toggle_all_visibility()

    def _on_toggle_all_lock_clicked(self) -> None:
        if self._cb_toggle_all_lock:
            self._cb_toggle_all_lock()

    def _create_toggle_button(self, text: str, active: bool) -> QPushButton:
        btn = QPushButton(text)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setMinimumHeight(34)
        self._update_toggle_style(btn, active)
        return btn

    def _update_toggle_style(self, btn: QPushButton, active: bool) -> None:
        if active:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: rgba(161, 250, 255, 0.1);
                    color: {PRIMARY};
                    border: 1px solid rgba(161, 250, 255, 0.2);
                    border-radius: 8px;
                    text-align: left; padding: 6px 12px;
                    font-size: {FONT_SIZE_SM}px; font-weight: 600;
                    letter-spacing: 0.5px;
                }}
            """)
        else:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {SURFACE_CONTAINER_HIGHEST};
                    color: {ON_SURFACE_VARIANT};
                    border: none; border-radius: 8px;
                    text-align: left; padding: 6px 12px;
                    font-size: {FONT_SIZE_SM}px; font-weight: 600;
                    letter-spacing: 0.5px;
                }}
                QPushButton:hover {{ color: {ON_SURFACE}; }}
            """)

    def _toggle_show_image(self):
        self._show_image_active = not self._show_image_active
        self._update_toggle_style(self._q_show_image, self._show_image_active)

    def _toggle_missing_pixels(self):
        self._missing_pixels_active = not self._missing_pixels_active
        self._update_toggle_style(self._q_missing_pixels, self._missing_pixels_active)

    def _toggle_show_grid(self):
        self._grid_visible_active = not self._grid_visible_active
        self._update_toggle_style(self._q_show_grid, self._grid_visible_active)
