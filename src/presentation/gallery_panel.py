"""Gallery panel — thumbnail browser for available images.

Opens as a slide-out panel between the left sidebar and the canvas.
Shows image thumbnails from the repository; clicking one loads it
through the existing image-loading flow.
"""
from __future__ import annotations

import os
import logging
from typing import Callable, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QSize, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QPainter, QFont, QColor
from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from presentation.style import (
    PRIMARY,
    ON_SURFACE,
    ON_SURFACE_VARIANT,
    OUTLINE_VARIANT,
    SURFACE,
    SURFACE_CONTAINER,
    SURFACE_CONTAINER_HIGH,
    SURFACE_CONTAINER_HIGHEST,
    SURFACE_CONTAINER_LOW,
    FONT_SIZE_SM,
    FONT_SIZE_XS,
)

logger = logging.getLogger(__name__)

_THUMB_SIZE = 100  # px
_ANNOTATION_ALPHA = 0.45  # opacity for annotation colour overlay

# Module-level pixmap cache: maps cache-key tuple -> QPixmap
# Survives gallery close/open cycles without re-reading from disk.
_thumb_cache: dict[tuple, "QPixmap"] = {}


def _draw_coverage_badge(pixmap: QPixmap, coverage: float) -> QPixmap:
    """Draw a small percentage badge on the bottom-right of *pixmap*."""
    text = f"{int(coverage * 100)}%"
    result = pixmap.copy()
    painter = QPainter(result)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    font = QFont()
    font.setPointSize(8)
    font.setBold(True)
    painter.setFont(font)
    metrics = painter.fontMetrics()
    rect = metrics.boundingRect(text)
    badge_rect = QRect(
        result.width() - rect.width() - 8,
        result.height() - rect.height() - 4,
        rect.width() + 6,
        rect.height() + 2,
    )
    painter.fillRect(badge_rect, QColor(0, 0, 0, 160))
    painter.setPen(QColor(255, 255, 255))
    painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, text)
    painter.end()
    return result


class _ThumbnailWorker(QThread):
    """Background thread that regenerates a thumbnail with annotation overlay."""

    thumbnail_ready = pyqtSignal(str, QPixmap)  # (filename, pixmap)

    def __init__(self, filename: str, doc, layer_configs: list, thumb_size: int, parent=None):
        super().__init__(parent)
        self._filename = filename
        self._doc = doc
        self._layer_configs = layer_configs
        self._thumb_size = thumb_size

    def run(self):
        try:
            bgr = self._doc.image
            h, w = bgr.shape[:2]
            scale = self._thumb_size / max(h, w)
            tw = max(1, int(round(w * scale)))
            th = max(1, int(round(h * scale)))
            base = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB).astype(np.float32)

            annotations = self._doc.annotations
            alpha = _ANNOTATION_ALPHA
            for i, layer in enumerate(self._layer_configs):
                if i >= len(annotations):
                    break
                mask = annotations[i]
                mask_small = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
                nonzero = mask_small > 0
                if not nonzero.any():
                    continue
                r, g, b = layer.color_rgb
                for c, val in enumerate((r, g, b)):
                    channel = rgb[:, :, c]
                    channel[nonzero] = channel[nonzero] * (1 - alpha) + val * alpha

            out = np.clip(rgb, 0, 255).astype(np.uint8)
            qimg = QImage(out.tobytes(), tw, th, 3 * tw, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            coverage = self._doc.annotation_coverage
            pixmap = _draw_coverage_badge(pixmap, coverage)
            self.thumbnail_ready.emit(self._filename, pixmap)
        except Exception:
            logger.exception("Error regenerating active thumbnail")


def _load_thumbnail(path: str, annotations_dir: str = "", layers=None) -> Optional["QPixmap"]:
    """Load, scale and annotate a thumbnail, using a module-level cache."""
    from domain.layer_config import sanitize_name
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError:
        return None

    stem = os.path.splitext(os.path.basename(path))[0]
    ann_mtimes: list[int] = []
    if layers and annotations_dir:
        for layer in layers:
            ann_path = os.path.join(annotations_dir, f"{stem}_{sanitize_name(layer.name)}.png")
            try:
                ann_mtimes.append(os.stat(ann_path).st_mtime_ns)
            except OSError:
                ann_mtimes.append(0)

    key = (path, mtime, *ann_mtimes)
    if key in _thumb_cache:
        return _thumb_cache[key]

    bgr = cv2.imread(path)
    if bgr is None:
        return None

    h, w = bgr.shape[:2]
    scale = _THUMB_SIZE / max(h, w)
    tw = max(1, int(round(w * scale)))
    th = max(1, int(round(h * scale)))
    base = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB).astype(np.float32)

    if layers and annotations_dir:
        for layer in layers:
            ann_path = os.path.join(annotations_dir, f"{stem}_{sanitize_name(layer.name)}.png")
            mask = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask_small = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            nonzero = mask_small > 0
            if not nonzero.any():
                continue
            r, g, b = layer.color_rgb
            for c, val in enumerate((r, g, b)):
                channel = rgb[:, :, c]
                channel[nonzero] = channel[nonzero] * (1 - _ANNOTATION_ALPHA) + val * _ANNOTATION_ALPHA

    out = np.clip(rgb, 0, 255).astype(np.uint8)
    qimg = QImage(out.tobytes(), tw, th, 3 * tw, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)

    # Compute coverage badge from annotation files
    if layers and annotations_dir:
        total_pixels = tw * th
        any_annotated = np.zeros((th, tw), dtype=bool)
        for layer in layers:
            ann_path = os.path.join(annotations_dir, f"{stem}_{sanitize_name(layer.name)}.png")
            mask = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask_small = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            any_annotated |= mask_small > 0
        coverage = float(np.sum(any_annotated)) / total_pixels if total_pixels > 0 else 0.0
        pixmap = _draw_coverage_badge(pixmap, coverage)

    _thumb_cache[key] = pixmap
    return pixmap


class _ThumbnailItem(QFrame):
    """A single thumbnail card with image preview and filename."""

    def __init__(self, filename: str, images_dir: str, annotations_dir: str = "", layers=None, parent=None):
        super().__init__(parent)
        self._filename = filename
        self._selected = False
        self._cb_clicked: Optional[Callable[[str], None]] = None

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(_THUMB_SIZE + 16, _THUMB_SIZE + 32)
        self._apply_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Thumbnail image
        self._thumb_label = QLabel()
        self._thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb_label.setFixedSize(_THUMB_SIZE + 8, _THUMB_SIZE)
        self._thumb_label.setStyleSheet("background: transparent; border: none;")

        image_path = os.path.join(images_dir, filename)
        pixmap = _load_thumbnail(image_path, annotations_dir, layers)
        if pixmap:
            self._thumb_label.setPixmap(pixmap)
        else:
            self._thumb_label.setText("?")
            self._thumb_label.setStyleSheet(
                f"color: {ON_SURFACE_VARIANT}; font-size: 24px; "
                f"background: {SURFACE}; border-radius: 4px;"
            )
        layout.addWidget(self._thumb_label)

        # Filename label
        name_label = QLabel(filename)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; "
            f"background: transparent; border: none;"
        )
        name_label.setWordWrap(True)
        layout.addWidget(name_label)

    def on_clicked(self, cb: Callable[[str], None]):
        self._cb_clicked = cb

    def set_selected(self, selected: bool):
        self._selected = selected
        self._apply_style()

    def mousePressEvent(self, event):
        if self._cb_clicked:
            self._cb_clicked(self._filename)

    def _apply_style(self):
        if self._selected:
            self.setStyleSheet(f"""
                _ThumbnailItem {{
                    background-color: {SURFACE_CONTAINER_HIGHEST};
                    border: 1px solid rgba(161, 250, 255, 0.3);
                    border-radius: 8px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                _ThumbnailItem {{
                    background-color: {SURFACE_CONTAINER_HIGHEST};
                    border: 1px solid transparent;
                    border-radius: 8px;
                }}
                _ThumbnailItem:hover {{
                    border: 1px solid rgba(72, 72, 72, 0.3);
                }}
            """)

    def update_pixmap(self, pixmap: QPixmap) -> None:
        """Replace the displayed thumbnail pixmap."""
        self._thumb_label.setPixmap(pixmap)




# ------------------------------------------------------------------
# GalleryPanel
# ------------------------------------------------------------------

class GalleryPanel(QWidget):
    """Slide-out gallery showing thumbnails of available images."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(280)
        self.setStyleSheet(f"background-color: {SURFACE_CONTAINER_HIGH};")

        self._images_dir = ""
        self._annotations_dir = ""
        self._layers = None
        self._items: list[_ThumbnailItem] = []
        self._cb_image_selected: Optional[Callable[[str], None]] = None
        self._current_filename: Optional[str] = None
        self._populated_filenames: list[str] = []

        # Live thumbnail update (FEATURE-020)
        self._thumb_update_timer = QTimer(self)
        self._thumb_update_timer.setSingleShot(True)
        self._thumb_update_timer.setInterval(300)
        self._thumb_update_timer.timeout.connect(self._regenerate_active_thumbnail)
        self._pending_doc = None
        self._pending_layer_configs: list = []
        self._thumb_worker: Optional[_ThumbnailWorker] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setStyleSheet(f"background-color: {SURFACE_CONTAINER_HIGH};")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 12, 14, 12)

        title = QLabel("GALLERY")
        title.setStyleSheet(
            f"color: {PRIMARY}; font-size: {FONT_SIZE_SM}px; "
            f"font-weight: 700; letter-spacing: 1px;"
        )
        header_layout.addWidget(title)
        header_layout.addStretch()

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; border: none;
                color: {ON_SURFACE_VARIANT}; font-size: 14px;
                border-radius: 4px;
            }}
            QPushButton:hover {{ color: {ON_SURFACE}; background: {SURFACE_CONTAINER_HIGHEST}; }}
        """)
        close_btn.clicked.connect(self.hide)
        header_layout.addWidget(close_btn)
        layout.addWidget(header)

        # Scroll area with grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setContentsMargins(10, 8, 10, 8)
        self._grid_layout.setSpacing(8)
        self._grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self._grid_container)
        layout.addWidget(scroll, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_image_selected(self, cb: Callable[[str], None]) -> None:
        self._cb_image_selected = cb

    def populate(self, filenames: list[str], images_dir: str, annotations_dir: str = "", layers=None) -> None:
        """Fill the gallery with thumbnail items."""
        # Skip expensive rebuild when nothing has changed
        if (
            filenames == self._populated_filenames
            and images_dir == self._images_dir
            and annotations_dir == self._annotations_dir
            and layers == self._layers
        ):
            return

        self._images_dir = images_dir
        self._annotations_dir = annotations_dir
        self._layers = layers
        self._populated_filenames = list(filenames)

        # Clear existing items
        for item in self._items:
            self._grid_layout.removeWidget(item)
            item.deleteLater()
        self._items.clear()

        cols = 2
        for idx, filename in enumerate(filenames):
            item = _ThumbnailItem(filename, images_dir, annotations_dir, layers)
            item.on_clicked(self._on_item_clicked)
            if filename == self._current_filename:
                item.set_selected(True)
            self._items.append(item)
            row, col = divmod(idx, cols)
            self._grid_layout.addWidget(item, row, col)

    def set_current_filename(self, filename: Optional[str]) -> None:
        """Highlight the currently loaded image."""
        self._current_filename = filename
        for item in self._items:
            item.set_selected(item._filename == filename)

    def schedule_thumbnail_update(
        self,
        filename: Optional[str],
        doc,               # ImageDocument
        layer_configs: list,
    ) -> None:
        """Schedule a debounced regeneration of the active image thumbnail."""
        if filename is None or doc is None:
            return
        self._current_filename = filename
        self._pending_doc = doc
        self._pending_layer_configs = layer_configs
        self._thumb_update_timer.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _regenerate_active_thumbnail(self) -> None:
        """Launch a worker thread to regenerate the active thumbnail."""
        if self._pending_doc is None or not self._current_filename:
            return
        # Stop any in-flight worker
        if self._thumb_worker is not None and self._thumb_worker.isRunning():
            self._thumb_worker.thumbnail_ready.disconnect()
            self._thumb_worker.quit()
            self._thumb_worker.wait()
        worker = _ThumbnailWorker(
            self._current_filename,
            self._pending_doc,
            self._pending_layer_configs,
            _THUMB_SIZE,
            parent=self,
        )
        worker.thumbnail_ready.connect(self._on_thumbnail_ready)
        self._thumb_worker = worker
        worker.start()

    def _on_thumbnail_ready(self, filename: str, pixmap: QPixmap) -> None:
        """Update the matching thumbnail item in the UI (main thread)."""
        # Invalidate cache entry so the disk-based loader picks up changes
        for key in list(_thumb_cache.keys()):
            if key[0] == os.path.join(self._images_dir, filename):
                del _thumb_cache[key]
        for item in self._items:
            if item._filename == filename:
                item.update_pixmap(pixmap)
                break

    def _on_item_clicked(self, filename: str) -> None:
        self.set_current_filename(filename)
        if self._cb_image_selected:
            self._cb_image_selected(filename)
