"""Viewer abstraction: the contract between the application and the rendering backend.

Every component *above* this module (application layer, controller, tests)
must depend only on ``IImageAnnotationViewer``.  Qt-specific code lives
exclusively in ``viewer.qt_viewer``.

Design rationale
----------------
The viewer is responsible for:

* Displaying the base image.
* Displaying annotation overlays.
* Displaying interaction feedback (selection mask, tool cursor preview,
  missing-pixel highlight).
* Managing zoom and scroll transforms.
* Translating raw GUI input into **semantic callbacks expressed in
  image-pixel coordinates** before handing control to the controller.

Changing the rendering technology (e.g. Qt → OpenGL) requires only a new
concrete class that implements this interface; no changes to the controller,
the application layer, or the domain layer are needed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np


class IImageAnnotationViewer(ABC):
    """Contract for the image/annotation viewer component."""

    # ------------------------------------------------------------------
    # Image and overlay display
    # ------------------------------------------------------------------

    @abstractmethod
    def set_base_image(self, bgr: np.ndarray) -> None:
        """Load *bgr* as the displayed base image.

        Replaces the current image and resets all overlay items.
        """

    @abstractmethod
    def set_base_image_visible(self, visible: bool) -> None:
        """Show or hide the base image layer."""

    @abstractmethod
    def set_annotation_overlay(self, rgba: np.ndarray) -> None:
        """Replace the annotation overlay (RGBA, same H×W as the base image)."""

    @abstractmethod
    def set_annotations_visible(self, visible: bool) -> None:
        """Toggle visibility of the annotation overlay."""

    @abstractmethod
    def set_selection_mask(
        self,
        mask: Optional[np.ndarray],
        color_rgb: tuple[int, int, int],
    ) -> None:
        """Show or hide the in-progress selection / stroke overlay.

        Pass ``mask=None`` to hide the overlay.
        """

    @abstractmethod
    def set_tool_preview(
        self,
        mask: Optional[np.ndarray],
        color_rgb: tuple[int, int, int],
    ) -> None:
        """Show or hide the tool cursor preview overlay.

        Pass ``mask=None`` to hide the overlay.
        """

    @abstractmethod
    def set_missing_pixels_visible(
        self,
        mask: Optional[np.ndarray],
        visible: bool,
    ) -> None:
        """Show or hide the missing-pixels highlight.

        *mask* is the binary mask from ``ImageDocument.get_missing_annotations_mask()``.
        Pass ``visible=False`` to hide the overlay without clearing the mask.
        """

    @abstractmethod
    def set_grid_visible(self, visible: bool) -> None:
        """Show or hide the pixel grid overlay.

        The grid is only drawn when zoom >= the viewer's minimum grid zoom
        threshold *and* this flag is ``True``.
        """

    @abstractmethod
    def set_global_layer_opacity(self, opacity: float) -> None:
        """Set a global opacity multiplier (0.0–1.0) applied to all annotation layers."""
        ...

    # ------------------------------------------------------------------
    # Zoom and viewport
    # ------------------------------------------------------------------

    @abstractmethod
    def set_zoom(
        self,
        zoom: int,
        center: Optional[tuple[float, float]] = None,
    ) -> None:
        """Set the zoom level.

        If *center* is provided (image-pixel coordinates), the viewport is
        scrolled to keep that point at the centre of the visible area.
        """

    @abstractmethod
    def zoom_reset(self) -> None:
        """Reset zoom to 1:1 (100%) and center the image."""
        ...

    @abstractmethod
    def get_zoom(self) -> int:
        """Return the current zoom level."""

    @abstractmethod
    def get_view_center(self) -> tuple[float, float]:
        """Return the centre of the visible area in image-pixel coordinates."""

    # ------------------------------------------------------------------
    # Cursor
    # ------------------------------------------------------------------

    @abstractmethod
    def update_cursor(
        self,
        tool: str,
        layer_color: tuple[int, int, int],
    ) -> None:
        """Update the pointer cursor to match the current tool and layer colour."""

    # ------------------------------------------------------------------
    # Widget integration
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def widget(self):
        """Return the underlying GUI widget (for insertion into a layout)."""

    # ------------------------------------------------------------------
    # Input event registration
    # ------------------------------------------------------------------

    @abstractmethod
    def register_mouse_press(
        self,
        cb: Callable[[int, int, str], None],
    ) -> None:
        """Register *cb(pixel_x, pixel_y, button)* for mouse-press events.

        *button* is ``"left"``, ``"right"``, or ``"middle"``.
        Coordinates are always in image-pixel space.
        """

    @abstractmethod
    def register_mouse_release(
        self,
        cb: Callable[[int, int, str], None],
    ) -> None:
        """Register *cb(pixel_x, pixel_y, button)* for mouse-release events."""

    @abstractmethod
    def register_mouse_move(
        self,
        cb: Callable[[int, int], None],
    ) -> None:
        """Register *cb(pixel_x, pixel_y)* for mouse-move events."""

    @abstractmethod
    def register_scroll(
        self,
        cb: Callable[[int, int, int, int, frozenset], None],
    ) -> None:
        """Register *cb(delta_y, delta_x, pixel_x, pixel_y, modifiers)*.

        *modifiers* is a frozenset of modifier-key names (``"ctrl"``,
        ``"shift"``, ``"alt"``).
        """


