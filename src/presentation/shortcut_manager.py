"""Centralised keyboard shortcut handler.

Installed as a QApplication event filter so that shortcuts respond regardless
of which widget currently has focus, while explicitly NOT firing when the user
is typing in a text input widget.

Layer shortcuts:
- V          → toggle all layers visibility.
- L          → toggle all layers lock.
- Ctrl + 1-9 → toggle visibility of that specific layer.
- Alt  + 1-9 → toggle lock of that specific layer.
"""
from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import QAbstractSpinBox, QApplication, QComboBox, QLineEdit

if TYPE_CHECKING:
    from application.annotator_controller import AnnotatorController

_TEXT_INPUT_TYPES = (QLineEdit, QAbstractSpinBox, QComboBox)

# Keys mapped to symbolic names; must stay consistent with qt_viewer._key_name.
_KEY_MAP = {
    Qt.Key.Key_P:      "P",
    Qt.Key.Key_S:      "S",
    Qt.Key.Key_F:      "F",
    Qt.Key.Key_Z:      "Z",
    Qt.Key.Key_Y:      "Y",
    Qt.Key.Key_E:      "E",
    Qt.Key.Key_R:      "R",
    Qt.Key.Key_I:      "I",
    Qt.Key.Key_M:      "M",
    Qt.Key.Key_G:      "G",
    Qt.Key.Key_V:      "V",
    Qt.Key.Key_L:      "L",
    Qt.Key.Key_Plus:   "Plus",
    Qt.Key.Key_Equal:  "Plus",   # unshifted + on some keyboards
    Qt.Key.Key_Minus:  "Minus",
    Qt.Key.Key_Space:  "Space",
    Qt.Key.Key_Escape: "Escape",
    Qt.Key.Key_Return: "Return",
    Qt.Key.Key_Enter:  "Return",
    Qt.Key.Key_H:      "H",
    Qt.Key.Key_F1:     "F1",
    **{getattr(Qt.Key, f"Key_{i}"): str(i) for i in range(1, 10)},
}

_DIGITS = frozenset("123456789")


class ShortcutManager(QObject):
    """Application-level event filter that routes key events to the controller.

    Usage::

        mgr = ShortcutManager(main_window)
        QApplication.instance().installEventFilter(mgr)
        # Later, when a controller becomes available:
        mgr.set_controller(controller)
    """

    # (category, key_label, description)
    SHORTCUTS: list[tuple[str, str, str]] = [
        # Tools
        ("Tools", "P", "Pen tool"),
        ("Tools", "E", "Eraser tool"),
        ("Tools", "S", "Selector tool"),
        ("Tools", "F", "Fill tool"),
        # Edit
        ("Edit", "Ctrl+Z", "Undo"),
        ("Edit", "Ctrl+Y / Ctrl+Shift+Z", "Redo"),
        ("Edit", "+", "Increase brush / threshold size"),
        ("Edit", "-", "Decrease brush / threshold size"),
        # View
        ("View", "Space", "Hide annotations (hold)"),
        ("View", "I", "Toggle source image"),
        ("View", "M", "Toggle missing pixels overlay"),
        ("View", "G", "Toggle grid"),
        # Layers
        ("Layers", "V", "Toggle all layers visibility"),
        ("Layers", "L", "Toggle all layers lock"),
        ("Layers", "Ctrl+1-9", "Toggle visibility of layer N"),
        ("Layers", "Alt+1-9", "Toggle lock of layer N"),
        # Help
        ("Help", "H", "Show this help window"),
    ]

    def __init__(self, main_window, parent=None) -> None:
        super().__init__(parent)
        self._window = main_window
        self._controller: Optional["AnnotatorController"] = None
        self._on_help: Optional[Callable] = None
        # Digit keys consumed on press (Ctrl/Alt+digit) so their release
        # is also consumed, preventing an unintended layer-selection.
        self._consumed_digits: set[str] = set()

    def set_controller(self, controller: Optional["AnnotatorController"]) -> None:
        self._controller = controller

    def set_help_callback(self, cb: Callable) -> None:
        self._on_help = cb

    # ------------------------------------------------------------------
    # Event filter
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        t = event.type()
        if t not in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            return False

        if self._controller is None:
            return False

        focused = QApplication.focusWidget()
        if isinstance(focused, _TEXT_INPUT_TYPES):
            return False

        key_name = _KEY_MAP.get(event.key(), "")
        if not key_name:
            return False

        mods: set[str] = set()
        m = event.modifiers()
        if m & Qt.KeyboardModifier.ControlModifier:
            mods.add("ctrl")
        if m & Qt.KeyboardModifier.ShiftModifier:
            mods.add("shift")
        if m & Qt.KeyboardModifier.AltModifier:
            mods.add("alt")
        mods_fs = frozenset(mods)

        if t == QEvent.Type.KeyPress:
            return self._handle_press(key_name, mods_fs)
        else:
            return self._handle_release(key_name, mods_fs)

    # ------------------------------------------------------------------
    # Key press / release dispatch
    # ------------------------------------------------------------------

    def _handle_press(self, key_name: str, mods: frozenset) -> bool:
        if key_name == "H":
            if self._on_help:
                self._on_help()
            return True

        if key_name == "V":
            if self._controller:
                self._controller.toggle_all_visibility()
            return True

        if key_name == "L":
            if self._controller:
                self._controller.toggle_all_lock()
            return True

        if key_name in _DIGITS and "ctrl" in mods:
            self._consumed_digits.add(key_name)
            if self._controller:
                self._controller.toggle_layer_visibility_by_index(int(key_name) - 1)
            return True

        if key_name in _DIGITS and "alt" in mods:
            self._consumed_digits.add(key_name)
            if self._controller:
                self._controller.toggle_layer_lock(int(key_name) - 1)
            return True

        self._controller.handle_key_press(key_name, mods)
        return False

    def _handle_release(self, key_name: str, mods: frozenset) -> bool:
        if key_name in ("V", "L"):
            return True  # no action on release

        if key_name in self._consumed_digits:
            self._consumed_digits.discard(key_name)
            return True

        self._controller.handle_key_release(key_name, mods)
        return True
