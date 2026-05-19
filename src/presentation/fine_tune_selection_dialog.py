"""Fine-tune image selection dialog.

Shows a grid of fully-annotated images (100% coverage) so the user can
pick which ones to include in a fine-tuning run.  After ``exec()``
returns ``Accepted``, call ``get_selected_filenames()`` to retrieve the
chosen image filenames.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from domain.layer_config import LayerConfig, sanitize_name
from presentation.gallery_panel import _load_thumbnail
from presentation.style import (
    FONT_SIZE_SM,
    FONT_SIZE_XS,
    ON_PRIMARY,
    ON_SURFACE,
    ON_SURFACE_VARIANT,
    OUTLINE_VARIANT,
    PRIMARY,
    SURFACE_CONTAINER_HIGH,
    SURFACE_CONTAINER_HIGHEST,
)

_THUMB_SIZE = 100  # px — matches gallery_panel._THUMB_SIZE


# ------------------------------------------------------------------
# Coverage helper
# ------------------------------------------------------------------

def _compute_full_coverage(
    image_path: str,
    annotations_dir: str,
    layers: list[LayerConfig],
) -> float:
    """Return the annotation coverage at full resolution (0.0–1.0).

    A pixel is considered annotated if it is non-zero in *any* layer.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        return 0.0
    h, w = bgr.shape[:2]
    total = h * w
    if total == 0:
        return 0.0
    any_ann = np.zeros((h, w), dtype=bool)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    for layer in layers:
        ann_path = os.path.join(
            annotations_dir, f"{stem}_{sanitize_name(layer.name)}.png"
        )
        mask = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        any_ann |= mask > 0
    return float(np.sum(any_ann)) / total


# ------------------------------------------------------------------
# Selectable thumbnail card
# ------------------------------------------------------------------

class _SelectableThumbnail(QFrame):
    """Clickable thumbnail card used inside the selection grid."""

    def __init__(
        self,
        filename: str,
        images_dir: str,
        annotations_dir: str,
        layers: list[LayerConfig],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._filename = filename
        self._selected = False
        self._click_cb = None

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(_THUMB_SIZE + 16, _THUMB_SIZE + 30)
        self._apply_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setFixedSize(_THUMB_SIZE + 8, _THUMB_SIZE)
        self._img_label.setStyleSheet("background: transparent; border: none;")

        image_path = os.path.join(images_dir, filename)
        pixmap = _load_thumbnail(image_path, annotations_dir, layers)
        if pixmap:
            self._img_label.setPixmap(pixmap)
        else:
            self._img_label.setText("?")
            self._img_label.setStyleSheet(
                f"color: {ON_SURFACE_VARIANT}; font-size: 24px;"
                " background: transparent; border: none;"
            )
        layout.addWidget(self._img_label)

        name_lbl = QLabel(filename)
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_lbl.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px;"
            " background: transparent; border: none;"
        )
        name_lbl.setWordWrap(True)
        layout.addWidget(name_lbl)

    def set_click_cb(self, cb) -> None:
        self._click_cb = cb

    def set_selected(self, value: bool) -> None:
        self._selected = value
        self._apply_style()

    def mousePressEvent(self, event) -> None:
        if self._click_cb:
            self._click_cb(self._filename)

    def _apply_style(self) -> None:
        border = f"2px solid {PRIMARY}" if self._selected else "1px solid transparent"
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {SURFACE_CONTAINER_HIGHEST};
                border: {border};
                border-radius: 8px;
            }}
        """)


# ------------------------------------------------------------------
# Dialog
# ------------------------------------------------------------------

class FineTuneSelectionDialog(QDialog):
    """Modal dialog for selecting 100%-annotated images for fine-tuning."""

    def __init__(
        self,
        images_dir: str,
        annotations_dir: str,
        layer_configs: list[LayerConfig],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select images for fine-tuning")
        self.setModal(True)
        self.setMinimumSize(580, 420)

        self._images_dir = images_dir
        self._annotations_dir = annotations_dir
        self._layer_configs = layer_configs
        self._selected: set[str] = set()
        self._items: dict[str, _SelectableThumbnail] = {}
        self._fully_annotated: list[str] = []

        self.setStyleSheet(
            f"QDialog {{ background-color: {SURFACE_CONTAINER_HIGH}; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # Header
        header_lbl = QLabel("SELECT IMAGES FOR FINE-TUNING")
        header_lbl.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px;"
            " font-weight: 700; letter-spacing: 1.2px;"
        )
        layout.addWidget(header_lbl)

        subtitle_lbl = QLabel("Only fully annotated images (100%) are shown.")
        subtitle_lbl.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_SM}px;"
        )
        layout.addWidget(subtitle_lbl)

        # Scroll area with thumbnail grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._grid_container = QWidget()
        self._grid_container.setStyleSheet(
            f"background: {SURFACE_CONTAINER_HIGH};"
        )
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setContentsMargins(8, 8, 8, 8)
        self._grid_layout.setSpacing(10)
        scroll.setWidget(self._grid_container)
        layout.addWidget(scroll, 1)

        # Empty-state label (hidden until needed)
        self._empty_lbl = QLabel(
            "No fully annotated images found in this project."
        )
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_lbl.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_SM}px;"
            " padding: 40px;"
        )
        self._empty_lbl.hide()
        layout.addWidget(self._empty_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {OUTLINE_VARIANT}; max-height: 1px;")
        layout.addWidget(sep)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
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

        self._fine_tune_btn = QPushButton("Fine-tune")
        self._fine_tune_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._fine_tune_btn.setMinimumHeight(32)
        self._fine_tune_btn.setMinimumWidth(110)
        self._fine_tune_btn.setEnabled(False)
        self._fine_tune_btn.setStyleSheet(f"""
            QPushButton {{
                background: {PRIMARY}; color: {ON_PRIMARY};
                border: none; border-radius: 6px;
                font-size: {FONT_SIZE_SM}px; font-weight: 700;
            }}
            QPushButton:hover {{ background: #b5fcff; }}
            QPushButton:pressed {{ background: #00e5ee; }}
            QPushButton:disabled {{
                background: {SURFACE_CONTAINER_HIGHEST};
                color: {ON_SURFACE_VARIANT};
            }}
        """)
        self._fine_tune_btn.clicked.connect(self.accept)

        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self._fine_tune_btn)
        layout.addLayout(btn_row)

        self._populate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_selected_filenames(self) -> list[str]:
        """Return the selected filenames in their original order."""
        return [f for f in self._fully_annotated if f in self._selected]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _populate(self) -> None:
        """Scan images_dir, keep only 100%-annotated ones, fill the grid."""
        if not os.path.isdir(self._images_dir):
            self._show_empty()
            return

        exts = {".jpg", ".jpeg", ".png", ".gif"}
        all_images = sorted(
            f for f in os.listdir(self._images_dir)
            if os.path.splitext(f.lower())[1] in exts
        )

        self._fully_annotated = []
        for fname in all_images:
            image_path = os.path.join(self._images_dir, fname)
            cov = _compute_full_coverage(
                image_path, self._annotations_dir, self._layer_configs
            )
            if cov >= 1.0:
                self._fully_annotated.append(fname)

        if not self._fully_annotated:
            self._show_empty()
            return

        cols = 4
        for idx, fname in enumerate(self._fully_annotated):
            item = _SelectableThumbnail(
                fname,
                self._images_dir,
                self._annotations_dir,
                self._layer_configs,
                parent=self._grid_container,
            )
            item.set_click_cb(self._on_item_clicked)
            self._items[fname] = item
            self._grid_layout.addWidget(item, idx // cols, idx % cols)

    def _show_empty(self) -> None:
        self._grid_container.hide()
        self._empty_lbl.show()

    def _on_item_clicked(self, filename: str) -> None:
        if filename in self._selected:
            self._selected.discard(filename)
            self._items[filename].set_selected(False)
        else:
            self._selected.add(filename)
            self._items[filename].set_selected(True)
        self._fine_tune_btn.setEnabled(bool(self._selected))
