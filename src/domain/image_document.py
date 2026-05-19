"""Image document — core domain entity.

Holds the base image and per-layer annotation masks.  Contains only
pure domain state and operations.  No file I/O, no Qt, no persistence.

Persistence is the responsibility of ``infrastructure.ImageRepository``.
"""
from __future__ import annotations

import cv2
import numpy as np
import logging
from typing import Optional
from collections import deque

logger = logging.getLogger(__name__)

_MAX_UNDO = 20


class ImageDocument:
    """Domain entity for a loaded image with its annotation layers.

    Responsibilities:
    - Hold the base image array and per-layer annotation masks.
    - Apply annotation edits (pen, fill, selector result).
    - Maintain an undo stack.
    - Answer domain queries (progress, similarity mask, etc.).

    Everything else — file I/O, rendering, state management — lives in
    the infrastructure, application and viewer layers respectively.
    """

    def __init__(
        self,
        image: np.ndarray,
        annotations: np.ndarray,
        source_path: str,
    ) -> None:
        """
        Parameters
        ----------
        image : np.ndarray
            BGR image array of shape (H, W, C).
        annotations : np.ndarray
            Annotation masks of shape (N, H, W), dtype uint8, values 0 or 255.
        source_path : str
            Original file path (kept as reference; no I/O performed here).
        """
        self._image = image
        self._annotations = np.asarray(annotations, dtype=np.uint8)
        self._source_path = source_path
        self._undo_stack: deque[np.ndarray] = deque(maxlen=_MAX_UNDO)
        self._redo_stack: deque[np.ndarray] = deque(maxlen=_MAX_UNDO)
        # Greyscale cache — computed once; used by compute_similarity_mask.
        self._gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def image(self) -> np.ndarray:
        """Base image in BGR format, shape (H, W, C)."""
        return self._image

    @property
    def annotations(self) -> np.ndarray:
        """All annotation layers, shape (N, H, W)."""
        return self._annotations

    @property
    def source_path(self) -> str:
        return self._source_path

    @property
    def height(self) -> int:
        return int(self._image.shape[0])

    @property
    def width(self) -> int:
        return int(self._image.shape[1])

    @property
    def num_layers(self) -> int:
        return len(self._annotations)

    @property
    def annotation_coverage(self) -> float:
        """Fraction of pixels annotated in any layer (0.0–1.0)."""
        if self._annotations.size == 0:
            return 0.0
        any_annotated = np.any(self._annotations > 0, axis=0)
        return float(np.sum(any_annotated)) / (self.height * self.width)

    # ------------------------------------------------------------------
    # Annotation mutations
    # ------------------------------------------------------------------

    def annotate_mask(self, mask: np.ndarray, layer: int) -> None:
        """Paint *mask* into *layer*, clearing those pixels in every other layer."""
        self._push_undo()
        binary = np.where(mask > 0, np.uint8(255), np.uint8(0))
        self._annotations[layer] = np.maximum(self._annotations[layer], binary)
        self._clear_mask_from_other_layers(layer)
        self._trim_undo_if_unchanged()

    def set_from_labelmap(self, label_map: np.ndarray) -> None:
        """Replace all annotations from a 2-D integer label map.

        Each pixel value in *label_map* is taken as a layer index
        (0 … num_layers - 1).
        """
        self._push_undo()
        for i in range(self.num_layers):
            self._annotations[i] = np.where(label_map == i, 255, 0).astype(np.uint8)
        self._trim_undo_if_unchanged()

    def erase_mask(self, mask: np.ndarray, layer: int) -> None:
        """Clear annotated pixels in *layer* where *mask* is nonzero."""
        self._push_undo()
        inv = np.where(mask > 0, np.uint8(0), np.uint8(255))
        self._annotations[layer] = np.bitwise_and(self._annotations[layer], inv)
        self._trim_undo_if_unchanged()

    def erase_mask_all(self, mask: np.ndarray) -> None:
        """Clear annotated pixels in *all* layers where *mask* is nonzero."""
        self._push_undo()
        inv = np.where(mask > 0, np.uint8(0), np.uint8(255))
        for i in range(self.num_layers):
            self._annotations[i] = np.bitwise_and(self._annotations[i], inv)
        self._trim_undo_if_unchanged()

    def erase_mask_unlocked(self, mask: np.ndarray, locked_layers: set) -> None:
        """Clear annotated pixels in all *unlocked* layers where *mask* is nonzero."""
        self._push_undo()
        inv = np.where(mask > 0, np.uint8(0), np.uint8(255))
        for i in range(self.num_layers):
            if i not in locked_layers:
                self._annotations[i] = np.bitwise_and(self._annotations[i], inv)
        self._trim_undo_if_unchanged()

    def clear_all_annotations(self) -> None:
        """Reset every annotation layer to all-zeros (undoable)."""
        self._push_undo()
        self._annotations[:] = 0
        self._trim_undo_if_unchanged()

    def undo(self) -> bool:
        """Revert to the previous annotation state.

        Returns True if a state was available to revert to.
        """
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._annotations.copy())
        self._annotations = self._undo_stack.pop()
        return True

    def redo(self) -> bool:
        """Redo the last undone annotation operation.

        Returns True if there was something to redo, False otherwise.
        """
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._annotations.copy())
        self._annotations = self._redo_stack.pop()
        return True

    # ------------------------------------------------------------------
    # Domain queries
    # ------------------------------------------------------------------

    def get_other_annotations_mask(self, layer: int) -> np.ndarray:
        """OR-combination of all layers except *layer*."""
        other = np.delete(self._annotations, layer, axis=0)
        return np.bitwise_or.reduce(other, axis=0)

    def get_locked_other_annotations_mask(self, layer: int, locked_layers: set) -> np.ndarray:
        """OR-combination of annotations from locked layers, excluding *layer* itself."""
        h, w = self._annotations[0].shape
        result = np.zeros((h, w), dtype=np.uint8)
        for i in locked_layers:
            if i != layer:
                result = np.bitwise_or(result, self._annotations[i])
        return result

    def annotate_mask_respecting_locks(
        self, mask: np.ndarray, layer: int, locked_layers: set
    ) -> None:
        """Paint *mask* into *layer*, respecting locked layers.

        Pixels annotated in a locked layer are excluded from the painted region.
        The mask is cleared from unlocked other layers but not from locked ones.
        """
        self._push_undo()
        # Exclude pixels that are annotated in any locked other layer
        for i in locked_layers:
            if i != layer:
                mask = np.where(self._annotations[i] > 0, np.uint8(0), mask)
        binary = np.where(mask > 0, np.uint8(255), np.uint8(0))
        self._annotations[layer] = np.maximum(self._annotations[layer], binary)
        # Clear newly painted pixels from unlocked other layers only
        inv = np.where(binary > 0, np.uint8(0), np.uint8(255))
        for i in range(self.num_layers):
            if i != layer and i not in locked_layers:
                self._annotations[i] = np.bitwise_and(self._annotations[i], inv)
        self._trim_undo_if_unchanged()

    def get_all_annotations_mask(self) -> np.ndarray:
        """OR-combination of all layers."""
        return np.bitwise_or.reduce(self._annotations, axis=0)

    def get_missing_annotations_mask(self) -> np.ndarray:
        """Pixels that have no annotation in any layer."""
        return np.bitwise_not(self.get_all_annotations_mask())

    def get_unannotated_mask(self, x: int, y: int, connected: bool = True) -> np.ndarray:
        """Return a mask of unannotated pixels.

        When *connected* is True, only pixels reachable from (x, y) via a
        flood-fill (no colour threshold) are returned.  Otherwise every
        unannotated pixel in the image is included.
        """
        if connected:
            return self.compute_similarity_mask(x, y, threshold=255, ignore_annotations=False)
        return self.get_missing_annotations_mask()

    def compute_similarity_mask(
        self,
        x: int,
        y: int,
        threshold: int,
        ignore_annotations: bool = False,
        extra_barrier_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return the 8-connected region of pixels spectrally similar to (x, y).

        A pixel is *similar* when its greyscale value differs from the seed
        by at most *threshold*.  Already-annotated regions act as hard
        boundaries (barriers to flood propagation) unless *ignore_annotations*
        is True.  *extra_barrier_mask*, if provided, is an additional uint8
        mask whose non-zero pixels are treated as barriers regardless of
        *ignore_annotations* (e.g. pixels from locked layers).

        Implementation
        --------------
        Rather than a Python-level BFS, the algorithm uses two C-level passes:

        1. A vectorised NumPy comparison builds the full similarity bitmap in
           a single O(H×W) operation — no Python loop per pixel.
        2. ``cv2.connectedComponents`` (two-pass scanline, 8-connectivity)
           extracts the connected region that contains the seed pixel.

        This is orders of magnitude faster than the equivalent Python BFS on
        large images.
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return np.zeros(self._gray.shape, dtype=np.uint8)
        
        seed_value = int(self._gray[y, x])

        # Fast path: threshold ≥ 255 means every uint8 pixel is spectrally
        # similar to the seed (|v − seed| ≤ 255 is always true).  Skip the
        # similarity bitmap and use the unannotated region directly as the
        # binary map for connected-component labelling.
        if threshold >= 255:
            if not ignore_annotations:
                reachable = np.bitwise_not(self.get_all_annotations_mask())
            else:
                reachable = np.ones(self._gray.shape, dtype=np.uint8) * 255
        else:
            # 1. Vectorised similarity bitmap.
            similar = (
                np.abs(self._gray.astype(np.int16) - seed_value) <= threshold
            ).astype(np.uint8)

            # 2. Zero out annotation-boundary pixels so they act as barriers.
            if not ignore_annotations:
                blocked = self.get_all_annotations_mask()
                similar = np.where(blocked > 0, np.uint8(0), similar)

            reachable = similar

        # Apply any extra barriers (e.g. locked layer pixels) on top.
        if extra_barrier_mask is not None and np.any(extra_barrier_mask):
            reachable = np.where(extra_barrier_mask > 0, np.uint8(0), reachable)

        # If the seed pixel is itself blocked or dissimilar, return an empty mask.
        if reachable[y, x] == 0:
            return np.zeros(self._gray.shape, dtype=np.uint8)

        # 3. Label connected components; select the one containing the seed.
        _, labels = cv2.connectedComponents(reachable, connectivity=8)
        seed_label = int(labels[y, x])
        return np.where(labels == seed_label, np.uint8(255), np.uint8(0))

    def get_progress(self) -> int:
        """Fraction of pixels annotated in any layer, as an integer percentage."""
        combined = np.any(self._annotations, axis=0)
        total = self.height * self.width
        annotated = int(np.count_nonzero(combined))
        pct = int(annotated / total * 100) if total else 0
        # Clamp to 99 % if not every single pixel is covered (rounding artefact).
        if pct == 100 and annotated < total:
            pct = 99
        return pct

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _push_undo(self) -> None:
        self._redo_stack.clear()
        self._undo_stack.append(self._annotations.copy())

    def _trim_undo_if_unchanged(self) -> None:
        """Pop the last undo entry if annotations are unchanged after a mutation."""
        if self._undo_stack and np.array_equal(self._annotations, self._undo_stack[-1]):
            self._undo_stack.pop()

    def _clear_mask_from_other_layers(self, layer: int) -> None:
        """Remove pixels present in *layer* from all other layers."""
        inv = np.bitwise_not(self._annotations[layer])
        other = np.arange(self.num_layers) != layer
        self._annotations[other] = np.bitwise_and(self._annotations[other], inv)
