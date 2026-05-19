"""Help dialog showing all registered keyboard shortcuts."""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class HelpWindow(QDialog):
    """Non-modal dialog that lists all keyboard shortcuts by category."""

    def __init__(self, shortcuts: list[tuple[str, str, str]], parent=None) -> None:
        """
        Parameters
        ----------
        shortcuts:
            List of (category, key_label, description) tuples.
        """
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setModal(False)
        self.resize(480, 520)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(4)

        by_cat: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for cat, key, desc in shortcuts:
            by_cat[cat].append((key, desc))

        for cat, entries in by_cat.items():
            header = QLabel(f"<b>{cat}</b>")
            header.setStyleSheet("font-size: 13px; margin-top: 8px;")
            layout.addWidget(header)
            for key, desc in entries:
                row = QLabel(f"  <code>{key}</code>&nbsp;&nbsp;{desc}")
                row.setTextFormat(Qt.TextFormat.RichText)
                layout.addWidget(row)

        layout.addStretch()
        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.addWidget(scroll)

    @staticmethod
    def show_help(
        shortcuts: list[tuple[str, str, str]],
        parent=None,
    ) -> "HelpWindow":
        """Create, show, and return a HelpWindow instance."""
        win = HelpWindow(shortcuts, parent)
        win.show()
        win.raise_()
        return win
