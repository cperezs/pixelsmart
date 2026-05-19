"""Time tracking per annotation layer.

Measures time spent annotating in each layer and persists the values
through the shared ImageMetadata object.
"""
from __future__ import annotations

import logging
import time


class TimeTracker:
    """Tracks elapsed time per annotation layer.

    Each call to ``change(layer)`` flushes the elapsed time for the
    *previous* layer and starts a new timer for *layer*.  This means
    ``change()`` must be called once at image-load time (layer 0) so
    that the very first annotation edit is captured.
    """

    def __init__(self, metadata) -> None:
        self._logger = logging.getLogger(__name__)
        self._metadata = metadata
        self._current_layer: int | None = None
        self._start_time: float | None = None

    def change(self, layer_index: int) -> None:
        """Flush elapsed time to the previous layer; start timing *layer_index*."""
        if self._current_layer is not None:
            delta = time.time() - self._start_time
            layer_name = self._metadata.layer_names[self._current_layer]
            self._metadata.update_time(layer_name, delta)

        self._start_time = time.time()
        self._current_layer = layer_index
        self._metadata.save()

    def tick(self) -> None:
        """Increment elapsed time for the current layer and save."""
        if self._current_layer is None:
            return
        delta = time.time() - self._start_time
        layer_name = self._metadata.layer_names[self._current_layer]
        self._metadata.update_time(layer_name, delta)
        self._start_time = time.time()
        self._metadata.save()

    def reset(self) -> None:
        """Reset all layer times to zero."""
        self._metadata.reset_times()
        self._metadata.save()
        self._current_layer = None
