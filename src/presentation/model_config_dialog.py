"""Model configuration dialog — plugin layer mapping and fine-tuning controls."""
from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from presentation.style import (
    PRIMARY,
    ON_PRIMARY,
    ON_SURFACE,
    ON_SURFACE_VARIANT,
    OUTLINE_VARIANT,
    SURFACE_BRIGHT,
    SURFACE_CONTAINER_HIGH,
    SURFACE_CONTAINER_HIGHEST,
    SURFACE_VARIANT,
    FONT_SIZE_SM,
    FONT_SIZE_XS,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_ITEM_HEIGHT = 28  # px — height of one combo item in the popup list


def _fix_combo_scroll(combo: QComboBox) -> None:
    """Force the popup list to show all items without a scrollbar.

    ``setMaxVisibleItems`` is ignored on macOS (native style overrides it).
    Setting a fixed height on the underlying view is the reliable cross-
    platform solution.
    """
    n = combo.count()
    combo.setMaxVisibleItems(n)
    combo.view().setFixedHeight(n * _ITEM_HEIGHT + 8)  # 8px vertical padding
    combo.view().setSizeAdjustPolicy(
        QAbstractItemView.SizeAdjustPolicy.AdjustToContents
    )


# ------------------------------------------------------------------
# Dialog
# ------------------------------------------------------------------

class LayerMappingDialog(QDialog):
    """Modal dialog for configuring the plugin-to-app layer mapping and the
    conflict resolution strategy used when running the AI model.
    """

    _RADIO_STYLE = (
        f"QRadioButton {{ color: {ON_SURFACE}; font-size: {FONT_SIZE_SM}px; }}"
        f"QRadioButton::indicator {{ width: 14px; height: 14px; }}"
    )

    def __init__(
        self,
        plugin_display_name: str,
        plugin_layers: list[str],
        app_layers: list[str],
        current_mapping: dict[str, str],
        current_strategy: str = "argmax",
        current_priorities: dict[str, int] | None = None,
        can_fine_tune: bool = False,
        available_versions: list[dict] | None = None,
        current_version: str = "v1",
        is_training: bool = False,
        training_plugin_name: str = "",
        on_fine_tune: Callable | None = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Model Configuration — {plugin_display_name}")
        self.setModal(True)
        self.setMinimumWidth(480)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {SURFACE_CONTAINER_HIGH};
            }}
            QComboBox {{
                background-color: {SURFACE_CONTAINER_HIGHEST};
                color: {ON_SURFACE};
                border: 1px solid {OUTLINE_VARIANT};
                border-radius: 6px;
                padding: 4px 28px 4px 10px;
                font-size: {FONT_SIZE_SM}px;
                min-height: 28px;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 22px;
                border: none;
            }}
            QComboBox::down-arrow {{
                width: 10px;
                height: 10px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {SURFACE_CONTAINER_HIGHEST};
                color: {ON_SURFACE};
                selection-background-color: {PRIMARY};
                selection-color: {ON_PRIMARY};
                padding: 4px;
                border: 1px solid {OUTLINE_VARIANT};
                border-radius: 4px;
                outline: none;
            }}
        """)

        self._plugin_layers = list(plugin_layers)
        self._combos: dict[str, QComboBox] = {}
        self._prio_combos: dict[str, QComboBox] = {}
        self._blocking = False  # re-entrancy guard for swap logic
        if current_priorities is None:
            current_priorities = {}
        self._version_combo: Optional[QComboBox] = None
        self._fine_tune_btn: Optional[QPushButton] = None
        self._on_fine_tune = on_fine_tune
        self._training_plugin_name = training_plugin_name

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # ---- Version selector (solo para plugins reentrenables) ----------
        if can_fine_tune:
            version_lbl = QLabel("VERSION")
            version_lbl.setStyleSheet(
                f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; "
                f"font-weight: 700; letter-spacing: 1.2px;"
            )
            layout.addWidget(version_lbl)

            self._version_combo = QComboBox()
            versions_list = available_versions or []
            for v in versions_list:
                if v.get("is_original"):
                    display = f"{v['label']} — original"
                else:
                    display = f"{v['label']} — {v['date']}"
                self._version_combo.addItem(display, v["label"])
            _fix_combo_scroll(self._version_combo)

            # Seleccionar la versión actual
            default_idx = 0
            for i in range(self._version_combo.count()):
                if self._version_combo.itemData(i) == current_version:
                    default_idx = i
                    break
            self._version_combo.setCurrentIndex(default_idx)

            layout.addWidget(self._version_combo)

            sep_version = QFrame()
            sep_version.setFrameShape(QFrame.Shape.HLine)
            sep_version.setStyleSheet(f"color: {OUTLINE_VARIANT}; max-height: 1px;")
            layout.addWidget(sep_version)

        # ---- Conflict resolution strategy (radio buttons) ----------------
        strategy_lbl = QLabel("CONFLICT RESOLUTION STRATEGY")
        strategy_lbl.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; "
            f"font-weight: 700; letter-spacing: 1.2px;"
        )
        layout.addWidget(strategy_lbl)

        self._rb_argmax = QRadioButton("Highest model confidence")
        self._rb_argmax.setStyleSheet(self._RADIO_STYLE)
        self._rb_priority = QRadioButton("Layer priority")
        self._rb_priority.setStyleSheet(self._RADIO_STYLE)

        self._strategy_group = QButtonGroup(self)
        self._strategy_group.addButton(self._rb_argmax, 0)
        self._strategy_group.addButton(self._rb_priority, 1)

        if current_strategy == "layer_priority":
            self._rb_priority.setChecked(True)
        else:
            self._rb_argmax.setChecked(True)

        radio_row = QHBoxLayout()
        radio_row.setSpacing(20)
        radio_row.addWidget(self._rb_argmax)
        radio_row.addWidget(self._rb_priority)
        radio_row.addStretch()
        layout.addLayout(radio_row)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet(f"color: {OUTLINE_VARIANT}; max-height: 1px;")
        layout.addWidget(sep1)

        # ---- Layer mapping + priorities ----------------------------------
        mapping_lbl = QLabel("ASSIGN MODEL LAYERS TO ANNOTATION LAYERS")
        mapping_lbl.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; "
            f"font-weight: 700; letter-spacing: 1.2px;"
        )
        layout.addWidget(mapping_lbl)

        # Column headers
        col_header = QHBoxLayout()
        col_header.setSpacing(10)
        hdr_model = QLabel("Model layer")
        hdr_model.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; font-weight: 600;"
        )
        hdr_model.setMinimumWidth(100)
        hdr_app = QLabel("Annotation layer")
        hdr_app.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; font-weight: 600;"
        )
        self._hdr_prio = QLabel("Priority")
        hdr_prio = self._hdr_prio
        hdr_prio.setFixedWidth(72)
        hdr_prio.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; font-weight: 600;"
        )
        col_header.addWidget(hdr_model)
        col_header.addWidget(QLabel(""), 0)  # arrow spacer
        col_header.addWidget(hdr_app, 1)
        col_header.addWidget(hdr_prio)
        layout.addLayout(col_header)

        rows_widget = QWidget()
        rows_layout = QVBoxLayout(rows_widget)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_layout.setSpacing(6)

        n_layers = len(plugin_layers)

        for idx, plugin_layer in enumerate(plugin_layers):
            row = QHBoxLayout()
            row.setSpacing(10)

            lbl = QLabel(plugin_layer.capitalize())
            lbl.setStyleSheet(
                f"color: {ON_SURFACE}; font-size: {FONT_SIZE_SM}px; "
                f"font-weight: 600; min-width: 100px;"
            )
            arrow = QLabel("→")
            arrow.setStyleSheet(
                f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_SM}px;"
            )

            # Annotation layer combo
            combo = QComboBox()
            combo.addItem("-- No assignment --", None)
            for app_layer in app_layers:
                combo.addItem(app_layer.capitalize(), app_layer)
            _fix_combo_scroll(combo)

            if plugin_layer in current_mapping:
                target = current_mapping[plugin_layer]
                restore_idx = next(
                    (i for i in range(combo.count()) if combo.itemData(i) == target),
                    0,
                )
                combo.setCurrentIndex(restore_idx)

            # Priority combo (1 = highest)
            prio_combo = QComboBox()
            for p in range(1, n_layers + 1):
                prio_combo.addItem(str(p), p)
            prio_combo.setFixedWidth(72)
            prio_combo.setToolTip(
                "Layer priority (1 = highest). Used with the \"Layer priority\" strategy."
            )
            saved_prio = current_priorities.get(plugin_layer, idx + 1)
            prio_idx = next(
                (i for i in range(prio_combo.count()) if prio_combo.itemData(i) == saved_prio),
                idx,
            )
            prio_combo.setCurrentIndex(prio_idx)
            _fix_combo_scroll(prio_combo)

            self._combos[plugin_layer] = combo
            self._prio_combos[plugin_layer] = prio_combo

            # Wire priority swap
            prio_combo.currentIndexChanged.connect(
                lambda _checked, layer=plugin_layer: self._on_priority_changed(layer)
            )

            row.addWidget(lbl)
            row.addWidget(arrow)
            row.addWidget(combo, 1)
            row.addWidget(prio_combo)
            rows_layout.addLayout(row)

        layout.addWidget(rows_widget)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"color: {OUTLINE_VARIANT}; max-height: 1px;")
        layout.addWidget(sep2)

        # ---- Buttons -----------------------------------------------------
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        # Botón Fine-tune (solo para plugins reentrenables)
        if can_fine_tune:
            self._fine_tune_btn = QPushButton("Fine-tune")
            self._fine_tune_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._fine_tune_btn.setMinimumHeight(32)
            self._fine_tune_btn.setMinimumWidth(110)
            self._fine_tune_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {SURFACE_BRIGHT}; color: {ON_SURFACE};
                    border: none; border-radius: 6px;
                    font-size: {FONT_SIZE_SM}px; font-weight: 700;
                }}
                QPushButton:hover {{ background: {SURFACE_VARIANT}; }}
                QPushButton:pressed {{ background: {OUTLINE_VARIANT}; }}
                QPushButton:disabled {{
                    background: {SURFACE_CONTAINER_HIGHEST};
                    color: {ON_SURFACE_VARIANT};
                }}
            """)
            if is_training:
                self._fine_tune_btn.setEnabled(False)
                name = training_plugin_name or ""
                self._fine_tune_btn.setText(
                    f"Training {name} in progress" if name else "Training in progress"
                )
            self._fine_tune_btn.clicked.connect(self._on_fine_tune_clicked)
            btn_row.addWidget(self._fine_tune_btn)

        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        cancel_btn.setMinimumHeight(32)
        cancel_btn.setMinimumWidth(90)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: {SURFACE_CONTAINER_HIGHEST};
                color: {ON_SURFACE_VARIANT};
                border: none; border-radius: 6px;
                font-size: {FONT_SIZE_SM}px;
            }}
            QPushButton:hover {{ color: {ON_SURFACE}; }}
        """)
        cancel_btn.clicked.connect(self.reject)

        apply_btn = QPushButton("Apply")
        apply_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        apply_btn.setMinimumHeight(32)
        apply_btn.setMinimumWidth(90)
        apply_btn.setStyleSheet(f"""
            QPushButton {{
                background: {PRIMARY}; color: {ON_PRIMARY};
                border: none; border-radius: 6px;
                font-size: {FONT_SIZE_SM}px; font-weight: 700;
            }}
            QPushButton:hover {{ background: #b5fcff; }}
            QPushButton:pressed {{ background: #00e5ee; }}
        """)
        apply_btn.clicked.connect(self.accept)

        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(apply_btn)
        layout.addLayout(btn_row)

        # Wire radio buttons to enable/disable priority combos
        self._rb_argmax.toggled.connect(self._on_strategy_changed)
        self._on_strategy_changed()  # set initial enabled state

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_fine_tune_clicked(self) -> None:
        if self._on_fine_tune is not None:
            self._on_fine_tune()
        if self._fine_tune_btn is not None:
            self._fine_tune_btn.setEnabled(False)
            self._fine_tune_btn.setText(
                f"Training {self._training_plugin_name} in progress"
                if self._training_plugin_name
                else "Training in progress"
            )

    def _on_strategy_changed(self) -> None:
        is_priority = self._rb_priority.isChecked()
        self._hdr_prio.setVisible(is_priority)
        for pc in self._prio_combos.values():
            pc.setVisible(is_priority)

    def _on_priority_changed(self, changed_layer: str) -> None:
        """Swap priority values so no two layers share the same priority."""
        if self._blocking:
            return
        new_val = self._prio_combos[changed_layer].currentData()
        # Find another layer that already holds this value
        for other_layer, pc in self._prio_combos.items():
            if other_layer == changed_layer:
                continue
            if pc.currentData() == new_val:
                # Find what the changed layer previously held and put it there
                # We don't store previous values, so use the first free value
                used = {c.currentData() for n, c in self._prio_combos.items() if n != other_layer}
                free_val = next(
                    v for v in range(1, len(self._plugin_layers) + 1) if v not in used
                )
                self._blocking = True
                other_idx = next(
                    i for i in range(pc.count()) if pc.itemData(i) == free_val
                )
                pc.setCurrentIndex(other_idx)
                self._blocking = False
                break

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def selected_version(self) -> str:
        """Devuelve el label de la versión seleccionada en el combo (ej. 'v1', 'v2')."""
        if self._version_combo is None:
            return "v1"
        return self._version_combo.currentData() or "v1"

    def set_training_state(self, is_training: bool, plugin_name: str = "") -> None:
        """Actualiza el estado del botón Fine-tune desde fuera del diálogo."""
        if self._fine_tune_btn is None:
            return
        self._training_plugin_name = plugin_name
        if is_training:
            self._fine_tune_btn.setEnabled(False)
            self._fine_tune_btn.setText(
                f"Training {plugin_name} in progress" if plugin_name else "Training in progress"
            )
        else:
            self._fine_tune_btn.setEnabled(True)
            self._fine_tune_btn.setText("Fine-tune")

    def get_config(self) -> dict:
        """Return the full configuration as a plain dict.

        Keys: ``layer_mapping``, ``conflict_strategy``, ``layer_priorities``.
        """
        return {
            "layer_mapping": {
                layer: combo.currentData()
                for layer, combo in self._combos.items()
            },
            "conflict_strategy": (
                "layer_priority" if self._rb_priority.isChecked() else "argmax"
            ),
            "layer_priorities": {
                layer: pc.currentData()
                for layer, pc in self._prio_combos.items()
            },
        }
