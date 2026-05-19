"""Welcome screen shown when no project is open.

Displays a centered "Open Folder" button that lets the user pick a
project folder.  Replaces the entire main-window content until a valid
project is selected.
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from presentation.style import (
    ON_SURFACE,
    ON_SURFACE_VARIANT,
    ON_PRIMARY,
    PRIMARY,
    SURFACE,
    SURFACE_CONTAINER_HIGH,
    SURFACE_CONTAINER_HIGHEST,
    FONT_SIZE_MD,
    FONT_SIZE_SM,
)


class WelcomeScreen(QWidget):
    """Full-window placeholder when no project folder is loaded."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._cb_open: Optional[Callable[[], None]] = None
        self.setStyleSheet(f"background-color: {SURFACE};")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(16)

        icon = QLabel("📂")
        icon.setStyleSheet("font-size: 48px; background: transparent;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon)

        title = QLabel("PixelSmart")
        title.setStyleSheet(
            f"color: {PRIMARY}; font-size: 20px; font-weight: 700; "
            f"letter-spacing: -0.5px; background: transparent;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Open a project folder to start annotating")
        subtitle.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_MD}px; background: transparent;"
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        btn = QPushButton("Open Folder")
        btn.setFixedSize(200, 44)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {PRIMARY}; color: {ON_PRIMARY};
                font-weight: 700; border-radius: 8px;
                font-size: {FONT_SIZE_MD}px; letter-spacing: 1px;
            }}
            QPushButton:hover {{ background: #b5fcff; }}
            QPushButton:pressed {{ background: #00e5ee; }}
        """)
        btn.clicked.connect(self._on_open_clicked)
        layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def on_open_folder(self, cb: Callable[[], None]) -> None:
        self._cb_open = cb

    def _on_open_clicked(self) -> None:
        if self._cb_open:
            self._cb_open()
