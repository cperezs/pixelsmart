"""Post-autolabel correction metrics tracking.

Tracks user edits made after a plugin run.  After each edit, the latest
counters are pushed into the shared ``ImageMetadata`` object; persistence
is driven by the ``TimeTracker.change()`` call that follows every
annotation action in main.py.
"""

import time
import logging
import numpy as np


class AutolabelMetrics:
    """Accumulates correction metrics for a single autolabel session.

    A session begins when a plugin is successfully run and ends when the
    user switches images or runs another plugin.
    """

    def __init__(self, plugin_id: str, layer_names: list, metadata):
        self._logger = logging.getLogger("AutolabelMetrics")
        self.plugin_id = plugin_id
        self.layer_names = list(layer_names)
        self._metadata = metadata
        self.start_time = time.time()

        # Global counters
        self.total_corrections = 0
        self.total_modified_pixels = 0

        # Per-layer counters (mirrors the shape stored in ImageMetadata)
        self.per_layer = {
            name: {
                "additions": 0,
                "deletions": 0,
                "pixels_added": 0,
                "pixels_deleted": 0,
            }
            for name in self.layer_names
        }

        self._pre_edit_snapshot = None

    # ------------------------------------------------------------------
    # Edit tracking
    # ------------------------------------------------------------------

    def begin_edit(self, annotations):
        """Snapshot the annotation state before a user edit."""
        self._pre_edit_snapshot = [a.copy() for a in annotations]

    def end_edit(self, annotations):
        """Diff the annotation state after a user edit, update counters,
        and push the latest values into the shared ImageMetadata object.

        Does *not* call ``metadata.save()`` — that is left to the
        ``TimeTracker.change()`` call that follows every annotation action.
        """
        if self._pre_edit_snapshot is None:
            return

        self.total_corrections += 1
        edit_modified = 0

        for i, (old, new) in enumerate(zip(self._pre_edit_snapshot, annotations)):
            if i >= len(self.layer_names):
                break
            layer_name = self.layer_names[i]
            added = int(np.count_nonzero((new > 0) & (old == 0)))
            deleted = int(np.count_nonzero((new == 0) & (old > 0)))

            if added > 0:
                self.per_layer[layer_name]["additions"] += 1
                self.per_layer[layer_name]["pixels_added"] += added
                edit_modified += added
            if deleted > 0:
                self.per_layer[layer_name]["deletions"] += 1
                self.per_layer[layer_name]["pixels_deleted"] += deleted
                edit_modified += deleted

        self.total_modified_pixels += edit_modified
        self._pre_edit_snapshot = None

        # Push into the shared metadata (no file I/O here).
        self._metadata.update_correction_metrics(
            total_operations=self.total_corrections,
            total_modified_pixels=self.total_modified_pixels,
            per_layer=self.per_layer,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_total_time(self) -> float:
        """Return elapsed correction time in seconds."""
        return time.time() - self.start_time

