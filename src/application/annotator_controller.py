"""AnnotatorController — the application orchestrator.

Responsibilities:
- Respond to viewer input events (mouse, keyboard, scroll).
- Execute domain operations (annotation, undo, autolabel).
- Drive persistence (saving annotation files and metadata).
- Keep viewer overlays in sync with domain state.
- Expose a clean Python API for the presentation layer.

The controller knows nothing about QPushButton, QLabel, or any other
widget.  It is the only component that holds direct references to both
the viewer and the domain repository.
"""
from __future__ import annotations

import os
import shutil
import time
import uuid
import logging
from typing import Callable, Optional

import cv2
import numpy as np

from domain.image_document import ImageDocument
from domain.layer_config import LayerConfig
from infrastructure.image_repository import ImageRepository
from application.app_state import AppState, ToolbarState
from application.annotation_tools import (
    compute_pen_mask,
    compute_fill_mask,
    apply_overwrite_guard,
    smooth_mask,
    expand_mask,
    shrink_mask,
    build_annotation_rgba,
    interpolate_points,
)
from application.autolabel_service import AutolabelService
from viewer.interface import IImageAnnotationViewer
from infrastructure.metadata import ImageMetadata
from infrastructure.time_tracker import TimeTracker
from infrastructure.action_logger import ActionLogger

logger = logging.getLogger(__name__)


class AnnotatorController:
    """Wires together the viewer, state, domain layer, and infrastructure.

    Construction
    ------------
    Pass a viewer, the initial layer configuration, and a repository.
    The presenter (main window) then wires up its own callbacks.

    Input flow
    ----------
    Viewer fires raw events → controller translates to domain operations →
    domain state mutates → controller syncs viewer overlays.
    """

    def __init__(
        self,
        viewer: IImageAnnotationViewer,
        layer_configs: list[LayerConfig],
        image_repo: ImageRepository,
    ) -> None:
        self._viewer = viewer
        self._layer_configs = layer_configs
        self._image_repo = image_repo

        layer_names = [lc.name for lc in layer_configs]
        self._state = AppState(layer_names)
        self._document: Optional[ImageDocument] = None
        self._current_filename: Optional[str] = None
        self._metadata: Optional[ImageMetadata] = None
        self._time_tracker: Optional[TimeTracker] = None
        self._action_logger: Optional[ActionLogger] = None
        self._autolabel = AutolabelService(layer_names)
        self._last_mouse_pos: Optional[tuple[int, int]] = None

        # Callbacks for the presentation layer to observe.
        self._on_progress_changed:        list[Callable[[int], None]] = []
        self._on_web_service_image_ready: list[Callable[[str, object], None]] = []
        self._on_status_changed:          list[Callable[[], None]] = []
        self._image_dimensions: tuple[int, int] = (0, 0)

        # Wheel scroll accumulator for throttling (units: angle delta, 120 = one notch).
        self._wheel_accum: float = 0.0

        # Wire viewer input events → our handlers.
        viewer.register_mouse_press(self._handle_mouse_press)
        viewer.register_mouse_release(self._handle_mouse_release)
        viewer.register_mouse_move(self._handle_mouse_move)
        viewer.register_scroll(self._handle_scroll)

    # ------------------------------------------------------------------
    # Public read-only API for the presentation layer
    # ------------------------------------------------------------------

    @property
    def state(self) -> AppState:
        return self._state

    @property
    def document(self) -> Optional[ImageDocument]:
        return self._document

    @property
    def layer_configs(self) -> list[LayerConfig]:
        return self._layer_configs

    @property
    def autolabel_service(self) -> AutolabelService:
        return self._autolabel

    # ------------------------------------------------------------------
    # Callback registration (for the presentation layer)
    # ------------------------------------------------------------------

    def on_progress_changed(self, cb: Callable[[int], None]) -> None:
        self._on_progress_changed.append(cb)

    def on_web_service_image_ready(self, cb: Callable[[str, object], None]) -> None:
        self._on_web_service_image_ready.append(cb)

    def on_status_changed(self, cb: Callable[[], None]) -> None:
        """Register *cb* to be called when status-bar-relevant state changes."""
        self._on_status_changed.append(cb)

    def on_toolbar_state_changed(self, cb: Callable[[], None]) -> None:
        """Register *cb* to be called whenever any toolbar-reflected state changes.

        The callback receives no arguments; the caller should read
        ``controller.toolbar_state`` to obtain a fresh snapshot.
        Subscribes to the ``"tool"``, ``"session"``, and ``"view"`` topics
        so that any source of change (keyboard shortcut, viewer event,
        programmatic call) triggers a single, consistent toolbar refresh.
        """
        self._state.subscribe("tool", cb)
        self._state.subscribe("session", cb)
        self._state.subscribe("view", cb)

    @property
    def toolbar_state(self) -> ToolbarState:
        """Return a fresh ``ToolbarState`` snapshot reflecting the current ``AppState``."""
        t = self._state.tool
        v = self._state.view
        s = self._state.session
        return ToolbarState(
            active_tool=t.active,
            pen_size=t.pen_size,
            eraser_size=t.eraser_size,
            selector_threshold=t.selector_threshold,
            selector_auto_smooth=t.selector_auto_smooth,
            fill_all=t.fill_all,
            active_layer=s.active_layer,
            locked_layers=set(s.locked_layers),
            hidden_layers=set(s.hidden_layers),
            show_image=v.show_image,
            show_other_layers=v.show_other_layers,
            show_missing_pixels=v.show_missing_pixels,
            show_grid=v.show_grid,
            global_layer_opacity=v.global_layer_opacity,
        )

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def load_image(self, filename: str) -> bool:
        """Load *filename* from the image repository.

        Returns True on success.  The viewer is updated immediately.
        """
        self._finalize_autolabel_session()
        # Flush pending time so that the close event records accurate totals.
        if self._time_tracker:
            self._time_tracker.tick()
        if self._metadata:
            self._metadata.save()
        # Log close of previous image
        if self._action_logger and self._metadata and self._current_filename:
            self._action_logger.log_image_close(
                self._current_filename, self._metadata.time_by_layer,
            )

        nlayers = len(self._layer_configs)
        layer_names = [lc.name for lc in self._layer_configs]
        try:
            doc = self._image_repo.load(filename, nlayers, layer_names)
        except FileNotFoundError as exc:
            logger.error("Cannot load image: %s", exc)
            return False

        self._document = doc
        self._current_filename = filename
        self._image_dimensions = (doc.width, doc.height)

        meta_path = self._image_repo.metadata_path(filename)
        layer_names = [lc.name for lc in self._layer_configs]
        self._metadata = ImageMetadata.load(meta_path, layer_names)
        self._metadata.update_pixel_stats(
            doc.annotations, image_size=(doc.height, doc.width)
        )
        self._time_tracker = TimeTracker(self._metadata)

        # Action logger — one log file per image
        log_path = self._image_repo.log_path(filename)
        self._action_logger = ActionLogger(log_path, layer_names)
        self._action_logger.log_image_open(
            filename=filename,
            width=doc.width,
            height=doc.height,
            num_layers=nlayers,
            pixels_per_layer=dict(self._metadata.pixels_per_layer),
        )

        # Start timing layer 0 immediately so that the elapsed time up to the
        # first edit is captured.  change() starts the clock but only flushes
        # on the *next* call, so calling it here mirrors the old behaviour where
        # set_state("selected_layer") triggered track_time on image load.
        self._time_tracker.change(0)

        # Reset interaction state.
        s = self._state
        s.tool.is_drawing = False
        s.tool.selector_origin = None
        s.session.active_layer = 0
        s.session.selection_mask = None
        s.session.tool_preview_mask = None
        s.view.zoom = 1
        s.view.center_pos = None

        # Initialise the viewer.
        self._viewer.set_base_image(doc.image)
        self._sync_annotation_overlay()
        self._viewer.zoom_reset()
        self._viewer.set_grid_visible(self._state.view.show_grid)
        self._state.notify("view")
        self._viewer.update_cursor(
            s.tool.active,
            self._layer_configs[0].color_rgb,
        )
        self._notify_progress()
        self._notify_status()
        return True

    # ------------------------------------------------------------------
    # Tool selection
    # ------------------------------------------------------------------

    def select_tool(self, tool: str) -> None:
        """Switch the active annotation tool (``"pen"``, ``"selector"``, ``"fill"``, ``"eraser"``)."""
        self._state.tool.active = tool
        self._state.tool.is_drawing = False
        self._state.session.selection_mask = None
        self._viewer.set_selection_mask(None, (255, 255, 255))
        self._viewer.set_tool_preview(None, (255, 255, 255))
        self._viewer.update_cursor(
            tool,
            self._layer_configs[self._state.session.active_layer].color_rgb,
        )
        self._state.notify("tool")
        self._notify_status()

    def set_active_layer(self, layer_index: int) -> None:
        self._state.session.active_layer = layer_index
        if self._time_tracker:
            self._time_tracker.change(layer_index)
        self._sync_annotation_overlay()
        self._viewer.update_cursor(
            self._state.tool.active,
            self._layer_configs[layer_index].color_rgb,
        )
        self._state.notify("session")

    def set_pen_size(self, size: int) -> None:
        self._state.tool.pen_size = max(1, min(50, size))
        self._state.notify("tool")
        # Refresh the cursor preview circle at the current mouse position
        # so the user sees the new size immediately without moving the mouse.
        if (
            self._state.tool.active == "pen"
            and not self._state.tool.is_drawing
            and self._last_mouse_pos is not None
            and self._document is not None
        ):
            px, py = self._last_mouse_pos
            layer = self._state.session.active_layer
            color = self._layer_configs[layer].color_rgb
            preview = compute_pen_mask(
                self._document.height, self._document.width,
                px, py, self._state.tool.pen_size,
            )
            self._viewer.set_tool_preview(preview, color)

    def set_selector_threshold(self, threshold: int) -> None:
        self._state.tool.selector_threshold = max(1, min(128, threshold))
        # Recompute selector mask if a selection is in progress.
        if (
            self._state.tool.is_drawing
            and self._state.tool.selector_origin is not None
            and self._document is not None
        ):
            x, y = self._state.tool.selector_origin
            self._run_selector(x, y)
        self._state.notify("tool")

    def set_selector_auto_smooth(self, enabled: bool) -> None:
        self._state.tool.selector_auto_smooth = enabled
        self._state.notify("tool")

    def set_fill_all(self, enabled: bool) -> None:
        self._state.tool.fill_all = enabled
        self._state.notify("tool")

    def set_eraser_size(self, size: int) -> None:
        self._state.tool.eraser_size = max(1, min(50, size))
        self._state.notify("tool")
        if (
            self._state.tool.active == "eraser"
            and not self._state.tool.is_drawing
            and self._last_mouse_pos is not None
            and self._document is not None
        ):
            px, py = self._last_mouse_pos
            preview = compute_pen_mask(
                self._document.height, self._document.width,
                px, py, self._state.tool.eraser_size,
            )
            self._viewer.set_tool_preview(preview, (200, 200, 200))

    def set_eraser_all_layers(self, enabled: bool) -> None:
        self._state.tool.eraser_all_layers = enabled
        self._state.notify("tool")

    def toggle_layer_lock(self, layer_index: int) -> None:
        """Toggle the locked state of a layer."""
        locked = self._state.session.locked_layers
        if layer_index in locked:
            locked.discard(layer_index)
        else:
            locked.add(layer_index)
        self._state.notify("session")

    def toggle_all_lock(self) -> None:
        """Lock all layers if any is unlocked; unlock all if all are locked."""
        locked = self._state.session.locked_layers
        if locked:
            locked.clear()
        else:
            for i in range(len(self._layer_configs)):
                locked.add(i)
        self._state.notify("session")

    def is_layer_locked(self, layer_index: int) -> bool:
        return layer_index in self._state.session.locked_layers

    def toggle_layer_visibility(self, layer_index: int, visible: bool) -> None:
        """Show or hide a layer's annotation mask."""
        hidden = self._state.session.hidden_layers
        if visible:
            hidden.discard(layer_index)
        else:
            hidden.add(layer_index)
        self._sync_annotation_overlay()
        self._state.notify("session")

    def toggle_layer_visibility_by_index(self, index: int) -> None:
        """Toggle visibility of the layer at *index*. No-op if out of range."""
        if index < 0 or index >= len(self._layer_configs):
            return
        visible = index not in self._state.session.hidden_layers
        self.toggle_layer_visibility(index, not visible)

    def toggle_all_visibility(self) -> None:
        """Show all layers if any is hidden; hide all if all are visible."""
        hidden = self._state.session.hidden_layers
        if hidden:
            hidden.clear()
        else:
            for i in range(len(self._layer_configs)):
                hidden.add(i)
        self._sync_annotation_overlay()
        self._state.notify("session")

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    # Standard zoom levels (as scale factor; 1.0 = 100%).
    # Match common levels in GIMP, Photoshop, Acrobat Reader.
    _ZOOM_LEVELS = [
        0.125, 0.25, 0.5, 0.75,
        1.0, 1.5, 2.0, 3.0, 4.0,
        6.0, 8.0, 10.0, 12.0, 16.0,
    ]

    def zoom_in(self, center: Optional[tuple[float, float]] = None) -> None:
        zoom = self._state.view.zoom
        c = center or self._viewer.get_view_center()
        new_zoom = next(
            (lv for lv in self._ZOOM_LEVELS if lv > zoom + 1e-6),
            None,
        )
        if new_zoom is None:
            return
        self._state.view.zoom = new_zoom
        self._state.view.center_pos = c
        self._viewer.set_zoom(new_zoom, c)
        self._state.notify("view")

    def zoom_out(self, center: Optional[tuple[float, float]] = None) -> None:
        zoom = self._state.view.zoom
        c = center or self._viewer.get_view_center()
        new_zoom = next(
            (lv for lv in reversed(self._ZOOM_LEVELS) if lv < zoom - 1e-6),
            None,
        )
        if new_zoom is None:
            return
        self._state.view.zoom = new_zoom
        self._state.view.center_pos = c
        self._viewer.set_zoom(new_zoom, c)
        self._state.notify("view")

    def zoom_fit(self) -> None:
        """Reset zoom to 1× (fit to window)."""
        self._state.view.zoom = 1
        self._state.view.center_pos = None
        self._viewer.set_zoom(1)
        self._state.notify("view")

    def get_zoom_percent(self) -> int:
        """Return the current zoom as a percentage (100 = 1×)."""
        return int(round(self._state.view.zoom * 100))

    # ------------------------------------------------------------------
    # View toggles
    # ------------------------------------------------------------------

    def toggle_show_image(self, visible: bool) -> None:
        self._state.view.show_image = visible
        self._viewer.set_base_image_visible(visible)
        self._state.notify("view")

    def toggle_show_other_layers(self, visible: bool) -> None:
        self._state.view.show_other_layers = visible
        self._sync_annotation_overlay()
        self._state.notify("view")

    def toggle_show_missing_pixels(self, visible: bool) -> None:
        self._state.view.show_missing_pixels = visible
        if self._document is not None:
            mask = self._document.get_missing_annotations_mask()
            self._viewer.set_missing_pixels_visible(mask, visible)
        else:
            self._viewer.set_missing_pixels_visible(None, False)
        self._state.notify("view")

    def toggle_show_grid(self, visible: bool) -> None:
        self._state.view.show_grid = visible
        self._viewer.set_grid_visible(visible)
        self._state.notify("view")

    def set_global_layer_opacity(self, opacity: float) -> None:
        """Set the global annotation layer opacity multiplier (0.0–1.0)."""
        clamped = max(0.0, min(1.0, opacity))
        self._state.view.global_layer_opacity = clamped
        self._viewer.set_global_layer_opacity(clamped)
        self._state.notify("view")

    def toggle_annotations_visible(self, visible: bool) -> None:
        self._viewer.set_annotations_visible(visible)
        if not visible:
            self._viewer.set_missing_pixels_visible(None, False)
        elif self._state.view.show_missing_pixels and self._document:
            mask = self._document.get_missing_annotations_mask()
            self._viewer.set_missing_pixels_visible(mask, True)

    # ------------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------------

    def undo(self) -> None:
        if self._document:
            if self._document.undo():
                if self._action_logger:
                    self._action_logger.pop_last_annotation_entry()
                self._image_repo.save_annotations(self._document, self._current_filename, [lc.name for lc in self._layer_configs])
                self._sync_annotation_overlay()
                self._notify_progress()
                self._state.notify("session")

    def redo(self) -> None:
        """Redo the last undone annotation. No-op if the redo stack is empty."""
        if self._document is None:
            return
        if self._document.redo():
            if self._action_logger and self._action_logger._redo_log_stack:
                group = self._action_logger._redo_log_stack[-1]
                self._action_logger.replay_entry(group)
            self._image_repo.save_annotations(self._document, self._current_filename, [lc.name for lc in self._layer_configs])
            self._sync_annotation_overlay()
            self._notify_progress()
            self._state.notify("session")

    @property
    def can_redo(self) -> bool:
        return self._document is not None and bool(self._document._redo_stack)

    def erase_all(self) -> None:
        """Clear every annotation on the current image and reset all statistics.

        The canvas erase is undoable via Ctrl+Z.  Statistics (pixel counts,
        time) and the edit log are permanently cleared and cannot be recovered.
        """
        if not self._document:
            return
        self._document.clear_all_annotations()
        if self._metadata:
            self._metadata.update_pixel_stats(self._document.annotations)
        if self._time_tracker:
            current_layer = self._time_tracker._current_layer
            self._time_tracker.reset()
            if current_layer is not None:
                self._time_tracker.change(current_layer)
        if self._action_logger:
            self._action_logger.clear_log()
        self._image_repo.save_annotations(self._document, self._current_filename, [lc.name for lc in self._layer_configs])
        self._autolabel.finalize_session()
        self._autolabel.last_ai_metrics = None
        self._sync_annotation_overlay()
        self._notify_progress()

    # ------------------------------------------------------------------
    # Autolabeling
    # ------------------------------------------------------------------

    def run_autolabel(self, plugin_id: str) -> Optional[str]:
        """Run *plugin_id* on the current document (background-thread safe).

        Only performs the heavy model inference and saves annotations.
        Call ``finalize_autolabel()`` on the main thread afterwards.
        Returns an error message string on failure, or ``None`` on success.
        """
        if not self._document or not self._metadata:
            return "No image loaded"

        plugin_config = self._state.plugin_configs.get(plugin_id)

        if self._action_logger:
            config_dict = vars(plugin_config) if plugin_config is not None else None
            self._action_logger.discard_redo_log_entries()
            self._action_logger.log_autolabel_start(plugin_id, model_config=config_dict)

        t0 = time.time()
        success, error = self._autolabel.run(plugin_id, self._document, self._metadata, plugin_config)
        duration = time.time() - t0

        if self._action_logger:
            self._action_logger.log_autolabel_end(
                plugin_id=plugin_id,
                duration_seconds=duration,
                success=success,
                error=error,
                pixels_per_layer=dict(self._metadata.pixels_per_layer) if success else None,
            )

        if not success:
            return error

        self._image_repo.save_annotations(self._document, self._current_filename, [lc.name for lc in self._layer_configs])
        if self._metadata:
            for name in self._metadata.layer_names:
                self._metadata.pixels_per_layer[name] = 0
            self._metadata.save()
        if self._time_tracker:
            current_layer = self._time_tracker._current_layer
            self._time_tracker.reset()
            if current_layer is not None:
                self._time_tracker.change(current_layer)
        if self._action_logger:
            self._action_logger.clear_log()
        return None

    def finalize_autolabel(self) -> None:
        """Refresh the viewer and progress after a successful autolabel run.

        Must be called on the main (Qt) thread.
        """
        self._sync_annotation_overlay()
        self._notify_progress()

    # ------------------------------------------------------------------
    # Layers reconfiguration
    # ------------------------------------------------------------------

    def update_layers(self, layer_configs: list[LayerConfig]) -> None:
        """Reconfigure layers (called after the layers file is reloaded)."""
        self._layer_configs = layer_configs
        layer_names = [lc.name for lc in layer_configs]
        self._state.layer_names = layer_names
        self._autolabel.refresh_plugins(layer_names)

    # ------------------------------------------------------------------
    # Persistence / session lifecycle
    # ------------------------------------------------------------------

    def save_current(self) -> None:
        """Flush metadata to disk."""
        if self._metadata:
            self._metadata.save()

    def get_display_stats(self) -> dict:
        """Return all display statistics for the current image.

        Returns a dict with:
        - operation counts (pen_stroke, erase_stroke, …) from the log file
        - pixels_per_layer, total_pixels_annotated from metadata
        - time_by_layer, total_time including the ongoing un-flushed tick
        - layer_names: ordered list for display
        """
        import time as _time

        op_keys = (
            "pen_stroke",
            "erase_stroke",
            "fill_commit",
            "selector_commit",
            "erase_all",
            "autolabel_end",
        )
        ops: dict[str, int] = {k: 0 for k in op_keys}

        # Operation counts from log file
        if self._action_logger is not None:
            path = self._action_logger._path
            if os.path.exists(path):
                try:
                    import json as _json
                    with open(path, "r", encoding="utf-8") as fh:
                        entries: list[dict] = _json.load(fh)
                    for entry in entries:
                        action = entry.get("action")
                        if action in ops:
                            ops[action] += 1
                except (OSError, ValueError):
                    pass

        # Pixel stats from metadata (already up to date after each commit)
        pixels_per_layer: dict[str, int] = {}
        total_pixels_annotated = 0
        layer_names: list[str] = [lc.name for lc in self._layer_configs]
        if self._metadata is not None:
            pixels_per_layer = dict(self._metadata.pixels_per_layer)
            total_pixels_annotated = sum(pixels_per_layer.values())

        # Time stats — add the current un-flushed tick so the display is live
        time_by_layer: dict[str, float] = {n: 0.0 for n in layer_names}
        if self._metadata is not None:
            for name in layer_names:
                time_by_layer[name] = self._metadata.time_by_layer.get(name, 0.0)
            if (
                self._time_tracker is not None
                and self._time_tracker._current_layer is not None
                and self._time_tracker._start_time is not None
            ):
                elapsed = _time.time() - self._time_tracker._start_time
                idx = self._time_tracker._current_layer
                if 0 <= idx < len(layer_names):
                    time_by_layer[layer_names[idx]] += elapsed
        total_time = sum(time_by_layer.values())

        return {
            **ops,
            "pixels_per_layer": pixels_per_layer,
            "total_pixels_annotated": total_pixels_annotated,
            "time_by_layer": time_by_layer,
            "total_time": total_time,
            "layer_names": layer_names,
        }

    # Back-compat alias
    def get_edit_stats(self) -> dict:
        return self.get_display_stats()

    def close(self) -> None:
        """Clean up before the application terminates."""
        self._finalize_autolabel_session()
        # Flush pending time so that the close event records accurate totals.
        if self._time_tracker:
            self._time_tracker.tick()
        if self._action_logger and self._metadata and self._current_filename:
            self._action_logger.log_image_close(
                self._current_filename, self._metadata.time_by_layer,
            )
        self.save_current()

    # ------------------------------------------------------------------
    # Web service entry point
    # ------------------------------------------------------------------

    def load_web_service_image(self, image_path: str, request: object) -> None:
        """Copy *image_path* to the images directory and load it."""
        unique_filename = f"ws_{uuid.uuid4().hex[:8]}.png"
        target_path = os.path.join(self._image_repo._images_dir, unique_filename)
        os.makedirs(self._image_repo._images_dir, exist_ok=True)
        shutil.copy2(image_path, target_path)
        self.load_image(unique_filename)
        for cb in self._on_web_service_image_ready:
            cb(unique_filename, request)

    # ------------------------------------------------------------------
    # Viewer event handlers (registered in __init__)
    # ------------------------------------------------------------------

    def _handle_mouse_press(self, px: int, py: int, button: str) -> None:
        if not self._document or button != "left":
            return
        tool = self._state.tool
        layer = self._state.session.active_layer
        if tool.active != "eraser" and layer in self._state.session.locked_layers:
            return
        if tool.active == "pen":
            self._pen_draw(px, py)
            tool.is_drawing = True
        elif tool.active == "eraser":
            self._erase_draw(px, py)
            tool.is_drawing = True
        elif tool.active == "selector":
            tool.is_drawing = True
            tool.selector_origin = (px, py)
            self._run_selector(px, py)
        elif tool.active == "fill":
            self._run_fill(px, py)

    def _handle_mouse_release(self, px: int, py: int, button: str) -> None:
        if not self._document or button != "left":
            return
        if self._state.tool.active == "pen" and self._state.tool.is_drawing:
            self._commit_annotation()
            self._state.tool.is_drawing = False
        elif self._state.tool.active == "eraser" and self._state.tool.is_drawing:
            self._commit_erase()
            self._state.tool.is_drawing = False

    def _handle_mouse_move(self, px: int, py: int) -> None:
        if not self._document:
            self._last_mouse_pos = (px, py)
            return
        tool = self._state.tool
        layer = self._state.session.active_layer
        color = self._layer_configs[layer].color_rgb

        if tool.active == "pen" and tool.is_drawing:
            # FEATURE-031: interpolate to avoid gaps on fast moves
            if self._last_mouse_pos is not None:
                lx, ly = self._last_mouse_pos
                step = max(1, tool.pen_size // 2)
                for ix, iy in interpolate_points(lx, ly, px, py, step):
                    self._pen_draw(ix, iy)
            else:
                self._pen_draw(px, py)
            # FEATURE-037: keep cursor circle visible while drawing
            preview = compute_pen_mask(
                self._document.height, self._document.width, px, py, tool.pen_size
            )
            self._viewer.set_tool_preview(preview, color)
        elif tool.active == "pen":
            preview = compute_pen_mask(
                self._document.height, self._document.width, px, py, tool.pen_size
            )
            self._viewer.set_tool_preview(preview, color)
        elif tool.active == "eraser" and tool.is_drawing:
            # FEATURE-031: interpolate to avoid gaps on fast moves
            if self._last_mouse_pos is not None:
                lx, ly = self._last_mouse_pos
                step = max(1, tool.eraser_size // 2)
                for ix, iy in interpolate_points(lx, ly, px, py, step):
                    self._erase_draw(ix, iy)
            else:
                self._erase_draw(px, py)
            # FEATURE-037: keep cursor circle visible while drawing
            preview = compute_pen_mask(
                self._document.height, self._document.width, px, py, tool.eraser_size
            )
            self._viewer.set_tool_preview(preview, (200, 200, 200))
        elif tool.active == "eraser":
            preview = compute_pen_mask(
                self._document.height, self._document.width, px, py, tool.eraser_size
            )
            self._viewer.set_tool_preview(preview, (200, 200, 200))
        elif tool.active in ("selector", "fill"):
            preview = compute_pen_mask(
                self._document.height, self._document.width, px, py, 1
            )
            self._viewer.set_tool_preview(preview, color)
        self._last_mouse_pos = (px, py)
        self._notify_status()

    def _handle_scroll(
        self,
        dy: int,
        dx: int,
        px: int,
        py: int,
        mods: frozenset,
    ) -> None:
        if "ctrl" in mods:
            self._wheel_accum += dy
            while self._wheel_accum >= 120:
                self._wheel_accum -= 120
                self.zoom_in(center=(px, py))
            while self._wheel_accum <= -120:
                self._wheel_accum += 120
                self.zoom_out(center=(px, py))

    def handle_key_press(self, key: str, mods: frozenset) -> None:
        if key == "Space":
            self._viewer.set_annotations_visible(False)

    def handle_key_release(self, key: str, mods: frozenset) -> None:  # noqa: C901
        tool = self._state.tool

        if key == "P":
            self.select_tool("pen")

        elif key == "S":
            self.select_tool("selector")

        elif key == "F":
            self.select_tool("fill")

        elif key == "I":
            self.toggle_show_image(not self._state.view.show_image)

        elif key == "M":
            self.toggle_show_missing_pixels(not self._state.view.show_missing_pixels)

        elif key == "G":
            self.toggle_show_grid(not self._state.view.show_grid)

        elif key == "Z" and "ctrl" in mods and "shift" not in mods:
            if tool.is_drawing:
                self._cancel_tool()
            else:
                self.undo()

        elif (key == "Y" and "ctrl" in mods) or (key == "Z" and "ctrl" in mods and "shift" in mods):
            self.redo()

        elif key == "Plus":
            if tool.active == "pen":
                self.set_pen_size(tool.pen_size + 1)
            elif tool.active == "eraser":
                self.set_eraser_size(tool.eraser_size + 1)
            elif tool.active == "selector" and tool.is_drawing:
                mask = self._state.session.selection_mask
                if mask is not None and self._document is not None:
                    layer = self._state.session.active_layer
                    locked = self._state.session.locked_layers
                    locked_mask = self._document.get_locked_other_annotations_mask(layer, locked)
                    cur_layer_mask = self._document.annotations[layer]
                    barrier = np.bitwise_or(locked_mask, cur_layer_mask)
                    self._state.session.selection_mask = expand_mask(
                        mask, barrier if np.any(barrier) else None
                    )
                    self._sync_selection_mask()
            elif tool.active == "selector":
                self.set_selector_threshold(tool.selector_threshold + 1)

        elif key == "Minus":
            if tool.active == "pen":
                self.set_pen_size(tool.pen_size - 1)
            elif tool.active == "eraser":
                self.set_eraser_size(tool.eraser_size - 1)
            elif tool.active == "selector" and tool.is_drawing:
                mask = self._state.session.selection_mask
                if mask is not None:
                    self._state.session.selection_mask = shrink_mask(mask)
                    self._sync_selection_mask()
            elif tool.active == "selector":
                self.set_selector_threshold(tool.selector_threshold - 1)

        elif key == "Space":
            self._viewer.set_annotations_visible(True)
            if self._state.view.show_missing_pixels and self._document:
                mask = self._document.get_missing_annotations_mask()
                self._viewer.set_missing_pixels_visible(mask, True)

        elif key == "Escape":
            self._cancel_tool()

        elif key == "E" and tool.active == "selector" and tool.is_drawing:
            mask = self._state.session.selection_mask
            if mask is not None and self._document is not None:
                layer = self._state.session.active_layer
                locked = self._state.session.locked_layers
                locked_mask = self._document.get_locked_other_annotations_mask(layer, locked)
                annotated = locked_mask if np.any(locked_mask) else None
                self._state.session.selection_mask = expand_mask(mask, annotated)
                self._sync_selection_mask()

        elif key == "E":
            self.select_tool("eraser")

        elif key == "R" and tool.active == "selector" and tool.is_drawing:
            mask = self._state.session.selection_mask
            if mask is not None:
                self._state.session.selection_mask = shrink_mask(mask)
                self._sync_selection_mask()

        elif key == "Return" and tool.active == "selector" and tool.is_drawing:
            self._commit_annotation()
            tool.is_drawing = False

        else:
            # Numeric keys 1–9 → layer selection.
            try:
                idx = int(key) - 1
                if 0 <= idx < len(self._layer_configs):
                    self.set_active_layer(idx)
            except (ValueError, TypeError):
                pass

    # ------------------------------------------------------------------
    # Tool implementations (private)
    # ------------------------------------------------------------------

    def _pen_draw(self, px: int, py: int) -> None:
        doc = self._document
        tool = self._state.tool
        pen_mask = compute_pen_mask(doc.height, doc.width, px, py, tool.pen_size)
        cur = self._state.session.selection_mask
        merged = cv2.bitwise_or(
            cur if cur is not None else np.zeros_like(pen_mask),
            pen_mask,
        )
        layer = self._state.session.active_layer
        locked = self._state.session.locked_layers
        # Exclude pixels blocked by locked other layers (won't be committed).
        locked_mask = doc.get_locked_other_annotations_mask(layer, locked)
        if np.any(locked_mask):
            merged = apply_overwrite_guard(merged, locked_mask)
        # Exclude pixels already annotated in the current layer — the mask
        # shows only the *new* pixels that will be added.
        merged = apply_overwrite_guard(merged, doc.annotations[layer])
        self._state.session.selection_mask = merged
        self._sync_selection_mask()

    def _run_selector(self, px: int, py: int) -> None:
        doc = self._document
        tool = self._state.tool
        layer = self._state.session.active_layer
        locked = self._state.session.locked_layers
        locked_mask = doc.get_locked_other_annotations_mask(layer, locked)
        mask = doc.compute_similarity_mask(
            px, py, tool.selector_threshold, ignore_annotations=True,
            extra_barrier_mask=locked_mask if np.any(locked_mask) else None,
        )
        if tool.selector_auto_smooth:
            mask = smooth_mask(mask)
        self._state.session.selection_mask = mask
        self._sync_selection_mask()

    def _run_fill(self, px: int, py: int) -> None:
        doc = self._document
        layer = self._state.session.active_layer
        if layer in self._state.session.locked_layers:
            return
        mask = compute_fill_mask(doc.annotations, layer, px, py)
        if self._action_logger:
            self._action_logger.snapshot_before(doc.annotations)
        self._autolabel.begin_correction(doc.annotations)
        doc.annotate_mask(mask, layer)
        self._autolabel.end_correction(doc.annotations)
        self._post_annotation_commit(layer, tool="fill")

    def _erase_draw(self, px: int, py: int) -> None:
        """Accumulate eraser brush strokes into the session selection mask."""
        doc = self._document
        tool = self._state.tool
        erase_mask = compute_pen_mask(doc.height, doc.width, px, py, tool.eraser_size)
        cur = self._state.session.selection_mask
        merged = cv2.bitwise_or(
            cur if cur is not None else np.zeros_like(erase_mask),
            erase_mask,
        )
        # Only show pixels that will actually be erased: annotated in at
        # least one unlocked layer. Locked-layer-only and unannotated
        # pixels are excluded so the preview reflects the real delta.
        locked = self._state.session.locked_layers
        erasable = np.zeros_like(merged)
        for i in range(doc.num_layers):
            if i not in locked:
                erasable = cv2.bitwise_or(erasable, doc.annotations[i])
        display = cv2.bitwise_and(merged, erasable)
        self._state.session.selection_mask = merged
        self._viewer.set_selection_mask(display, (200, 200, 200))

    def _commit_erase(self) -> None:
        """Apply the accumulated erase mask to all unlocked layers."""
        mask = self._state.session.selection_mask
        if mask is None:
            return
        doc = self._document
        locked = self._state.session.locked_layers
        layer = self._state.session.active_layer
        if self._action_logger:
            self._action_logger.snapshot_before(doc.annotations)
        doc.erase_mask_unlocked(mask, locked)
        self._state.session.selection_mask = None
        self._sync_selection_mask()
        self._post_annotation_commit(layer, tool="eraser")

    def _commit_annotation(self) -> None:
        mask = self._state.session.selection_mask
        if mask is None:
            return
        doc = self._document
        layer = self._state.session.active_layer
        locked = self._state.session.locked_layers
        if self._action_logger:
            self._action_logger.snapshot_before(doc.annotations)
        self._autolabel.begin_correction(doc.annotations)
        doc.annotate_mask_respecting_locks(mask, layer, locked)
        self._autolabel.end_correction(doc.annotations)
        self._state.session.selection_mask = None
        self._sync_selection_mask()
        self._post_annotation_commit(layer, tool=self._state.tool.active)

    def _post_annotation_commit(self, layer: int, tool: str = "") -> None:
        """Persist, update metadata, sync viewer, and notify progress."""
        doc = self._document
        if self._action_logger and tool:
            delta = self._action_logger.compute_delta(doc.annotations)
            self._action_logger.discard_redo_log_entries()
            self._action_logger.log_annotation_commit(
                tool=tool,
                layer_index=layer,
                layer_name=self._layer_configs[layer].name,
                delta=delta,
            )
        self._image_repo.save_annotations(doc, self._current_filename, [lc.name for lc in self._layer_configs])
        if self._metadata:
            self._metadata.update_pixel_stats(
                doc.annotations, image_size=(doc.height, doc.width)
            )
        if self._time_tracker:
            self._time_tracker.change(layer)
        self._sync_annotation_overlay()
        self._notify_progress()

    def _cancel_tool(self) -> None:
        self._state.tool.is_drawing = False
        self._state.session.selection_mask = None
        self._sync_selection_mask()

    # ------------------------------------------------------------------
    # Viewer synchronisation helpers
    # ------------------------------------------------------------------

    def _sync_annotation_overlay(self) -> None:
        if self._document is None:
            return
        colors = [lc.color_rgb for lc in self._layer_configs]
        layer = self._state.session.active_layer
        show_others = self._state.view.show_other_layers
        rgba = build_annotation_rgba(
            self._document.annotations, colors, layer, show_others,
            hidden_layers=self._state.session.hidden_layers,
        )
        self._viewer.set_annotation_overlay(rgba)
        if self._state.view.show_missing_pixels:
            mask = self._document.get_missing_annotations_mask()
            self._viewer.set_missing_pixels_visible(mask, True)

    def _sync_selection_mask(self) -> None:
        mask = self._state.session.selection_mask
        color = self._layer_configs[self._state.session.active_layer].color_rgb
        self._viewer.set_selection_mask(mask, color)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _finalize_autolabel_session(self) -> None:
        self._autolabel.finalize_session()

    def _notify_progress(self) -> None:
        if self._document:
            progress = self._document.get_progress()
            for cb in self._on_progress_changed:
                cb(progress)

    def _notify_status(self) -> None:
        for cb in self._on_status_changed:
            cb()

    # ------------------------------------------------------------------
    # Status bar info
    # ------------------------------------------------------------------

    @property
    def current_filename(self) -> Optional[str]:
        return self._current_filename

    @property
    def image_dimensions(self) -> tuple[int, int]:
        return self._image_dimensions

    @property
    def mouse_position(self) -> Optional[tuple[int, int]]:
        return self._last_mouse_pos

    @property
    def active_tool_name(self) -> str:
        return self._state.tool.active.capitalize()
