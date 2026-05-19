"""Custom status bar — bottom bar with file info, coordinates, and active tool.

Matches the design system footer from design/code.html.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from presentation.style import (
    PRIMARY,
    ON_SURFACE_VARIANT,
    OUTLINE_VARIANT,
    SURFACE,
    FONT_SIZE_SM,
    FONT_SIZE_XS,
    STATUSBAR_HEIGHT,
)


class StatusBar(QWidget):
    """Bottom status bar showing file info, dimensions, coords, and active tool."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(STATUSBAR_HEIGHT)
        self.setStyleSheet(
            f"background-color: {SURFACE}; "
            f"border-top: 1px solid rgba(72, 72, 72, 0.2);"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(0)

        # Left section
        left = QHBoxLayout()
        left.setSpacing(12)

        # Filename with green dot
        self._dot = QLabel("●")
        self._dot.setStyleSheet(
            f"color: {PRIMARY}; font-size: 8px; padding: 0; margin: 0;"
        )
        left.addWidget(self._dot)

        self._filename_label = QLabel("No file")
        self._filename_label.setStyleSheet(self._info_style())
        left.addWidget(self._filename_label)

        left.addWidget(self._separator())

        self._dims_label = QLabel("—")
        self._dims_label.setStyleSheet(self._info_style())
        left.addWidget(self._dims_label)

        left.addWidget(self._separator())

        self._coords_label = QLabel("X: — Y: —")
        self._coords_label.setStyleSheet(self._info_style())
        left.addWidget(self._coords_label)

        layout.addLayout(left)
        layout.addStretch()

        # Right section
        right = QHBoxLayout()
        right.setSpacing(8)

        tool_label = QLabel("Active Tool:")
        tool_label.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_SM}px; "
            f"letter-spacing: 1px; font-weight: 500;"
        )
        right.addWidget(tool_label)

        self._tool_name = QLabel("Pen")
        self._tool_name.setStyleSheet(
            f"color: {PRIMARY}; font-size: {FONT_SIZE_SM}px; "
            f"font-weight: 700; letter-spacing: 1px;"
        )
        right.addWidget(self._tool_name)

        layout.addLayout(right)

    # ------------------------------------------------------------------
    # Public update methods
    # ------------------------------------------------------------------

    def set_filename(self, filename: Optional[str]) -> None:
        self._filename_label.setText(filename or "No file")

    def set_dimensions(self, width: int, height: int) -> None:
        self._dims_label.setText(f"{width}×{height}")

    def set_coordinates(self, x: Optional[int], y: Optional[int]) -> None:
        if x is not None and y is not None:
            self._coords_label.setText(f"X: {x}  Y: {y}")
        else:
            self._coords_label.setText("X: —  Y: —")

    def set_active_tool(self, tool_name: str) -> None:
        self._tool_name.setText(tool_name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _info_style() -> str:
        return (
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_SM}px; "
            f"letter-spacing: 1px; font-weight: 500;"
        )

    @staticmethod
    def _separator() -> QLabel:
        sep = QLabel("|")
        sep.setStyleSheet(
            f"color: rgba(172, 171, 170, 0.4); font-size: {FONT_SIZE_SM}px; "
            f"padding: 0 4px;"
        )
        return sep
