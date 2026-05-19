"""Application state model with change notification.

Three kinds of state are distinguished to prevent uncontrolled coupling:

``ToolState``
    Which annotation tool is active and its settings (pen size, threshold…).
    Owned by the controller; reflected in the toolbar.

``ViewState``
    Zoom level, pan centre, visibility flags.  Pure rendering/view concerns
    that do not belong to the domain.

``SessionState``
    State tied to the currently loaded image: active layer, working masks.

``AppState`` is the observable aggregate.  Any component can *subscribe* to
a named topic and will be notified whenever ``AppState.notify(topic)`` is
called for that topic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ------------------------------------------------------------------
# Plugin configuration value object
# ------------------------------------------------------------------

@dataclass
class PluginConfig:
    """Per-plugin inference configuration set by the user.

    Attributes
    ----------
    layer_mapping:
        ``{plugin_layer_name -> app_layer_name}`` dict that overrides the
        default name-based mapping.  When empty the plugin layer names are
        matched to app layer names automatically.
    conflict_strategy:
        ``"argmax"`` (default) — assign each pixel to the layer with the
        highest model probability.  ``"layer_priority"`` — assign each pixel
        to the highest-priority layer (lowest priority number) whose model
        probability ≥ 0.5; falls back to argmax when no layer exceeds the
        threshold.
    layer_priorities:
        ``{plugin_layer_name -> priority_int}`` where 1 is the highest
        priority.  Used only when ``conflict_strategy == "layer_priority"``.
    """
    layer_mapping: dict = field(default_factory=dict)
    conflict_strategy: str = "argmax"
    layer_priorities: dict = field(default_factory=dict)
    selected_version: str = "v1"


# ------------------------------------------------------------------
# State value objects (plain dataclasses — no notification logic)
# ------------------------------------------------------------------

@dataclass
class ToolState:
    """Settings for the currently active annotation tool."""
    active: str = "selector"              # "pen" | "selector" | "fill" | "eraser"
    pen_size: int = 5
    eraser_size: int = 5
    selector_threshold: int = 32
    selector_auto_smooth: bool = True
    fill_all: bool = False
    is_drawing: bool = False
    selector_origin: Optional[tuple[int, int]] = None


@dataclass
class ViewState:
    """Visual-presentation settings (no domain semantics)."""
    zoom: int = 1
    center_pos: Optional[tuple[float, float]] = None
    show_image: bool = True
    show_other_layers: bool = True
    show_missing_pixels: bool = False
    show_grid: bool = False
    global_layer_opacity: float = 0.7


@dataclass
class SessionState:
    """Transient state for the current annotation session."""
    active_layer: int = 0
    selection_mask: Optional[np.ndarray] = None
    tool_preview_mask: Optional[np.ndarray] = None
    locked_layers: set = field(default_factory=set)  # set of locked layer indices
    hidden_layers: set = field(default_factory=set)  # set of hidden layer indices


@dataclass
class ToolbarState:
    """A flat snapshot of all state that the toolbar must reflect.

    Built by the controller from ``ToolState``, ``ViewState``, and
    ``SessionState`` on every relevant state change.  Consumed by
    ``ToolbarPanel.sync()`` as a single atomic update, so callers never
    need to know which individual widget maps to which internal field.

    Adding a new toolbar control requires only:
      1. A new field here (with a sensible default).
      2. Population in ``AnnotatorController.toolbar_state``.
      3. Application in ``ToolbarPanel.sync()``.
    """
    # -- Tool settings --------------------------------------------------
    active_tool: str = "pen"
    pen_size: int = 5
    eraser_size: int = 5
    selector_threshold: int = 32
    selector_auto_smooth: bool = True
    fill_all: bool = False
    # -- Session --------------------------------------------------------
    active_layer: int = 0
    locked_layers: set = field(default_factory=set)
    hidden_layers: set = field(default_factory=set)
    # -- View toggles ---------------------------------------------------
    show_image: bool = True
    show_other_layers: bool = True
    show_missing_pixels: bool = False
    show_grid: bool = True
    global_layer_opacity: float = 0.7


# ------------------------------------------------------------------
# Observable aggregate
# ------------------------------------------------------------------

class AppState:
    """Central application state with pub/sub change notification.

    Usage example::

        state = AppState(layer_names=["bg", "fg"])
        state.subscribe("tool", my_callback)
        state.tool.pen_size = 5
        state.notify("tool")          # triggers my_callback

    The state object does *not* copy or snapshot values — subscribers are
    responsible for reading the current state when called back.
    """

    def __init__(self, layer_names: list[str]) -> None:
        self.layer_names: list[str] = list(layer_names)
        self.tool = ToolState()
        self.view = ViewState()
        self.session = SessionState()
        # Per-plugin inference configuration saved by the user.
        self.plugin_configs: dict[str, PluginConfig] = {}
        self._listeners: dict[str, list[Callable[[], None]]] = {}

    # ------------------------------------------------------------------
    # Pub/sub
    # ------------------------------------------------------------------

    def subscribe(self, topic: str, callback: Callable[[], None]) -> None:
        """Register *callback* to be called whenever *topic* is notified."""
        self._listeners.setdefault(topic, []).append(callback)

    def notify(self, *topics: str) -> None:
        """Fire all callbacks registered for each of *topics*.

        A callback is called at most once per ``notify`` call even if it
        is registered under multiple topics that are all notified at once.
        """
        seen: set[Callable] = set()
        for topic in topics:
            for cb in self._listeners.get(topic, []):
                if cb not in seen:
                    seen.add(cb)
                    cb()
