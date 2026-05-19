"""Stats panel — edit statistics for the current image.

Shows three sections:
  1. Operation counts (pen, eraser, fill, selector, erase-all, autolabel)
  2. Annotated pixels per layer + total
  3. Annotation time per layer + total

Opens as a slide-in panel between the left toolbar and the canvas,
following the same pattern as GalleryPanel.

Data is fed externally via :meth:`update_stats`.  The panel emits
:attr:`refresh_requested` every time it becomes visible so the caller
can provide fresh data without polling.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from presentation.style import (
    FONT_SIZE_SM,
    FONT_SIZE_XS,
    ON_SURFACE,
    ON_SURFACE_VARIANT,
    OUTLINE_VARIANT,
    PRIMARY,
    SIDEBAR_WIDTH,
    SURFACE_CONTAINER_HIGH,
    SURFACE_CONTAINER_HIGHEST,
)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_OP_ROWS: list[tuple[str, str, str]] = [
    ("pen_stroke",      "✏",  "Pen strokes"),
    ("erase_stroke",    "◻",  "Eraser strokes"),
    ("fill_commit",     "▣",  "Fill commits"),
    ("selector_commit", "⬚",  "Selector commits"),
]


def _fmt_time(seconds: float) -> str:
    """Format *seconds* as M:SS or H:MM:SS."""
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ------------------------------------------------------------------
# Panel widget
# ------------------------------------------------------------------

class StatsPanel(QWidget):
    """Lateral panel that displays edit statistics for the current image.

    Signals
    -------
    close_requested
        Emitted when the user clicks the × button.  The caller is
        responsible for hiding the panel and unchecking the toolbar button.
    refresh_requested
        Emitted every time the panel becomes visible (``showEvent``).  The
        caller should respond by calling :meth:`update_stats` with fresh data.
    """

    close_requested = pyqtSignal()
    refresh_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(SIDEBAR_WIDTH)
        self.setStyleSheet(f"background-color: {SURFACE_CONTAINER_HIGH};")

        self._current_layer_names: list[str] = []
        self._count_labels:       dict[str, QLabel] = {}
        self._layer_pixel_labels: dict[str, QLabel] = {}
        self._layer_time_labels:  dict[str, QLabel] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())
        root.addWidget(self._build_separator())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        self._cl = QVBoxLayout(content)
        self._cl.setContentsMargins(14, 12, 14, 14)
        self._cl.setSpacing(0)
        self._cl.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._build_ops_section()
        self._cl.addSpacing(14)
        self._build_pixels_section()
        self._cl.addSpacing(14)
        self._build_time_section()
        self._cl.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.refresh_requested.emit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_stats(self, stats: dict) -> None:
        """Update all displayed values from *stats* (produced by the controller)."""
        # operations
        for action_type, lbl in self._count_labels.items():
            count = int(stats.get(action_type, 0))
            lbl.setText(str(count))
            color = PRIMARY if count > 0 else ON_SURFACE
            lbl.setStyleSheet(
                f"color: {color}; font-size: {FONT_SIZE_SM}px; font-weight: 700;"
            )

        # rebuild dynamic layer rows if layer names changed
        layer_names: list[str] = stats.get("layer_names", [])
        if layer_names and layer_names != self._current_layer_names:
            self._rebuild_layer_rows(layer_names)

        # pixels
        ppl: dict[str, int] = stats.get("pixels_per_layer", {})
        for name, lbl in self._layer_pixel_labels.items():
            lbl.setText(f"{ppl.get(name, 0):,}")
        self._pixels_total_lbl.setText(
            f"{stats.get('total_pixels_annotated', 0):,}"
        )

        # time
        tbl: dict[str, float] = stats.get("time_by_layer", {})
        for name, lbl in self._layer_time_labels.items():
            lbl.setText(_fmt_time(tbl.get(name, 0.0)))
        self._time_total_lbl.setText(_fmt_time(stats.get("total_time", 0.0)))

    def reset(self) -> None:
        """Reset all counters to zero (called on image change)."""
        for lbl in self._count_labels.values():
            lbl.setText("0")
            lbl.setStyleSheet(
                f"color: {ON_SURFACE}; font-size: {FONT_SIZE_SM}px; font-weight: 700;"
            )
        for lbl in self._layer_pixel_labels.values():
            lbl.setText("0")
        self._pixels_total_lbl.setText("0")
        for lbl in self._layer_time_labels.values():
            lbl.setText("0:00")
        self._time_total_lbl.setText("0:00")

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setStyleSheet(f"background-color: {SURFACE_CONTAINER_HIGHEST};")
        lay = QHBoxLayout(header)
        lay.setContentsMargins(14, 10, 8, 10)
        lay.setSpacing(0)

        title = QLabel("ANNOTATION STATISTICS")
        title.setStyleSheet(
            f"color: {PRIMARY}; font-size: {FONT_SIZE_XS}px; "
            f"font-weight: 700; letter-spacing: 1.5px;"
        )
        lay.addWidget(title)
        lay.addStretch()

        close_btn = QPushButton("×")
        close_btn.setFixedSize(28, 28)
        close_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        close_btn.setToolTip("Close panel")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; border: none; border-radius: 6px;
                color: {ON_SURFACE_VARIANT}; font-size: 18px; padding: 0;
            }}
            QPushButton:hover {{
                background: rgba(255, 255, 255, 0.07); color: {ON_SURFACE};
            }}
        """)
        close_btn.clicked.connect(self.close_requested)
        lay.addWidget(close_btn)
        return header

    def _build_separator(self) -> QFrame:
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {OUTLINE_VARIANT};")
        return sep

    def _section_header_lbl(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {PRIMARY}; font-size: {FONT_SIZE_XS}px; "
            f"font-weight: 700; letter-spacing: 1.2px;"
        )
        return lbl

    def _thin_sep(self) -> QFrame:
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {OUTLINE_VARIANT};")
        return sep

    def _value_label(self, initial: str = "0") -> QLabel:
        lbl = QLabel(initial)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lbl.setStyleSheet(
            f"color: {ON_SURFACE}; font-size: {FONT_SIZE_SM}px; font-weight: 700;"
        )
        return lbl

    # --- Operations section ---

    def _build_ops_section(self) -> None:
        self._cl.addWidget(self._section_header_lbl("OPERATIONS"))
        self._cl.addSpacing(8)
        for action_type, icon, label_text in _OP_ROWS:
            row = QHBoxLayout()
            row.setSpacing(8)

            icon_lbl = QLabel(icon)
            icon_lbl.setFixedWidth(18)
            icon_lbl.setStyleSheet(
                f"font-size: 12px; color: {ON_SURFACE_VARIANT};"
            )
            row.addWidget(icon_lbl)

            name_lbl = QLabel(label_text)
            name_lbl.setStyleSheet(
                f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_SM}px;"
            )
            row.addWidget(name_lbl, 1)

            count_lbl = self._value_label("0")
            self._count_labels[action_type] = count_lbl
            row.addWidget(count_lbl)

            self._cl.addLayout(row)
            self._cl.addSpacing(6)

    # --- Pixels section ---

    def _build_pixels_section(self) -> None:
        self._cl.addWidget(self._section_header_lbl("PIXELS"))
        self._cl.addSpacing(8)

        self._pixels_rows_widget = QWidget()
        self._pixels_rows_widget.setStyleSheet("background: transparent;")
        self._pixels_rows_layout = QVBoxLayout(self._pixels_rows_widget)
        self._pixels_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._pixels_rows_layout.setSpacing(6)
        self._cl.addWidget(self._pixels_rows_widget)

        self._cl.addSpacing(4)
        self._cl.addWidget(self._thin_sep())
        self._cl.addSpacing(4)

        total_row = QHBoxLayout()
        total_lbl = QLabel("Total")
        total_lbl.setStyleSheet(
            f"color: {ON_SURFACE}; font-size: {FONT_SIZE_SM}px; font-weight: 600;"
        )
        self._pixels_total_lbl = self._value_label("0")
        total_row.addWidget(total_lbl, 1)
        total_row.addWidget(self._pixels_total_lbl)
        self._cl.addLayout(total_row)

    # --- Time section ---

    def _build_time_section(self) -> None:
        self._cl.addWidget(self._section_header_lbl("TIME"))
        self._cl.addSpacing(8)

        self._time_rows_widget = QWidget()
        self._time_rows_widget.setStyleSheet("background: transparent;")
        self._time_rows_layout = QVBoxLayout(self._time_rows_widget)
        self._time_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._time_rows_layout.setSpacing(6)
        self._cl.addWidget(self._time_rows_widget)

        self._cl.addSpacing(4)
        self._cl.addWidget(self._thin_sep())
        self._cl.addSpacing(4)

        total_row = QHBoxLayout()
        total_lbl = QLabel("Total")
        total_lbl.setStyleSheet(
            f"color: {ON_SURFACE}; font-size: {FONT_SIZE_SM}px; font-weight: 600;"
        )
        self._time_total_lbl = self._value_label("0:00")
        total_row.addWidget(total_lbl, 1)
        total_row.addWidget(self._time_total_lbl)
        self._cl.addLayout(total_row)

    # --- Dynamic layer rows ---

    def _rebuild_layer_rows(self, layer_names: list[str]) -> None:
        """Clear and rebuild pixel + time rows for *layer_names*."""
        self._clear_layout(self._pixels_rows_layout)
        self._layer_pixel_labels.clear()
        self._clear_layout(self._time_rows_layout)
        self._layer_time_labels.clear()

        for name in layer_names:
            # pixel row
            pix_row = QHBoxLayout()
            pix_row.setContentsMargins(0, 0, 0, 0)
            pix_row.setSpacing(6)
            pix_name = QLabel(name)
            pix_name.setStyleSheet(
                f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_SM}px;"
            )
            pix_val = self._value_label("0")
            self._layer_pixel_labels[name] = pix_val
            pix_row.addWidget(pix_name, 1)
            pix_row.addWidget(pix_val)
            pix_w = QWidget()
            pix_w.setStyleSheet("background: transparent;")
            pix_w.setLayout(pix_row)
            self._pixels_rows_layout.addWidget(pix_w)

            # time row
            time_row = QHBoxLayout()
            time_row.setContentsMargins(0, 0, 0, 0)
            time_row.setSpacing(6)
            time_name = QLabel(name)
            time_name.setStyleSheet(
                f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_SM}px;"
            )
            time_val = self._value_label("0:00")
            self._layer_time_labels[name] = time_val
            time_row.addWidget(time_name, 1)
            time_row.addWidget(time_val)
            time_w = QWidget()
            time_w.setStyleSheet("background: transparent;")
            time_w.setLayout(time_row)
            self._time_rows_layout.addWidget(time_w)

        self._current_layer_names = list(layer_names)

    @staticmethod
    def _clear_layout(layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)  # type: ignore[arg-type]
                w.deleteLater()
