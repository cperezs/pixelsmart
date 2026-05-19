"""Unified per-image metadata model.

Single source of truth for all annotation metrics:
  - global metrics (time per layer, annotated pixels per layer)
  - optional autolabel execution metadata
  - optional post-autolabel correction metrics

The metadata is persisted as JSON in the existing .metadata file.
Backward-compatible migration from the legacy plain-text format (one
integer per line = seconds per layer) is handled transparently on load.

Schema version 1 shape
----------------------
{
  "schema_version": 1,
  "global_metrics": {
    "total_pixels": <int>,
    "pixels_per_layer": {"<layer>": <int>, ...},
    "time_by_layer": {"<layer>": <float>, ...},
    "total_time_seconds": <float>
  },
  "autolabel_plugin": "<plugin_id>",        // absent when never used
  "correction_metrics": {                   // absent when never used
    "total_operations": <int>,
    "total_modified_pixels": <int>,
    "per_layer": {
      "<layer>": {
        "additions": <int>,
        "deletions": <int>,
        "pixels_added": <int>,
        "pixels_deleted": <int>
      }, ...
    }
  }
}
"""

import os
import json
import logging
import numpy as np

SCHEMA_VERSION = 1


class ImageMetadata:
    """Canonical in-memory metadata model for one annotated image.

    Call ``save()`` to persist the current state.  All save triggers are
    driven externally (from TimeTracker and from main.py action handlers).
    """

    def __init__(self, metadata_path: str, layer_names: list):
        self._logger = logging.getLogger("ImageMetadata")
        self._path = metadata_path
        self.layer_names: list = list(layer_names)

        # Global metrics
        self.time_by_layer: dict = {n: 0.0 for n in layer_names}
        self.pixels_per_layer: dict = {n: 0 for n in layer_names}
        self.total_pixels: int = 0

        # Optional autolabel fields — None means never used.
        self.autolabel_plugin: str | None = None
        self.correction_metrics: dict | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, metadata_path: str, layer_names: list) -> "ImageMetadata":
        """Load metadata from *metadata_path*.

        Handles both the new JSON format and the legacy plain-text format.
        Missing or unreadable files return a fresh (zero-initialised) instance.
        """
        instance = cls(metadata_path, layer_names)
        if not os.path.exists(metadata_path):
            return instance

        logger = logging.getLogger("ImageMetadata")
        try:
            with open(metadata_path, "r") as f:
                content = f.read().strip()
            if content.startswith("{"):
                instance._load_json(content)
            else:
                instance._migrate_legacy(content)
        except Exception as exc:
            logger.error("Failed to load metadata from %s: %s", metadata_path, exc)

        return instance

    def _load_json(self, content: str):
        data = json.loads(content)
        gm = data.get("global_metrics", {})

        self.total_pixels = int(gm.get("total_pixels", 0))

        tbl = gm.get("time_by_layer", {})
        ppl = gm.get("pixels_per_layer", {})
        for name in self.layer_names:
            if name in tbl:
                self.time_by_layer[name] = float(tbl[name])
            if name in ppl:
                self.pixels_per_layer[name] = int(ppl[name])

        self.autolabel_plugin = data.get("autolabel_plugin", None)
        self.correction_metrics = data.get("correction_metrics", None)

    def _migrate_legacy(self, content: str):
        """Parse the old plain-text format (one integer per line = seconds)."""
        lines = [ln for ln in content.splitlines() if ln.strip()]
        for name, line in zip(self.layer_names, lines):
            try:
                self.time_by_layer[name] = float(int(line.strip()))
            except ValueError:
                pass
        self._logger.info("Migrated legacy metadata: %s", self._path)

    # ------------------------------------------------------------------
    # Global metrics mutations
    # ------------------------------------------------------------------

    def update_time(self, layer_name: str, delta: float):
        """Add *delta* seconds to *layer_name*."""
        if layer_name in self.time_by_layer:
            self.time_by_layer[layer_name] += delta

    def reset_times(self):
        """Reset all layer times to zero."""
        for name in self.layer_names:
            self.time_by_layer[name] = 0.0

    def update_pixel_stats(self, annotations, image_size=None):
        """Recompute annotated-pixel counts from annotation arrays.

        Parameters
        ----------
        annotations : sequence of np.ndarray
            Binary masks, one per layer (values 0/255, shape H×W).
            Index order must match ``self.layer_names``.
        image_size : tuple (height, width) | None
            Full image dimensions.  When supplied, ``total_pixels`` is set
            to ``height * width`` (total pixel count, not just annotated).
        """
        if image_size is not None:
            self.total_pixels = image_size[0] * image_size[1]

        for i, name in enumerate(self.layer_names):
            if i < len(annotations):
                self.pixels_per_layer[name] = int(np.count_nonzero(annotations[i]))

    # ------------------------------------------------------------------
    # Autolabel mutations
    # ------------------------------------------------------------------

    def set_autolabel_plugin(self, plugin_id: str):
        """Record that *plugin_id* was successfully run.  Resets correction metrics."""
        self.autolabel_plugin = plugin_id
        self.correction_metrics = {
            "total_operations": 0,
            "total_modified_pixels": 0,
            "per_layer": {
                name: {
                    "additions": 0,
                    "deletions": 0,
                    "pixels_added": 0,
                    "pixels_deleted": 0,
                }
                for name in self.layer_names
            },
        }

    def update_correction_metrics(
        self,
        total_operations: int,
        total_modified_pixels: int,
        per_layer: dict,
    ):
        """Overwrite correction counters with the latest accumulated values."""
        if self.correction_metrics is None:
            return
        self.correction_metrics["total_operations"] = total_operations
        self.correction_metrics["total_modified_pixels"] = total_modified_pixels
        self.correction_metrics["per_layer"] = {
            name: dict(stats) for name, stats in per_layer.items()
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        """Overwrite the .metadata file with the current state (JSON)."""
        parent = os.path.dirname(os.path.abspath(self._path))
        os.makedirs(parent, exist_ok=True)
        try:
            with open(self._path, "w") as f:
                json.dump(self._to_dict(), f, indent=2)
            self._logger.debug("Metadata saved: %s", self._path)
        except Exception as exc:
            self._logger.error("Failed to save metadata to %s: %s", self._path, exc)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _to_dict(self) -> dict:
        d: dict = {
            "schema_version": SCHEMA_VERSION,
            "global_metrics": {
                "total_pixels": self.total_pixels,
                "pixels_per_layer": dict(self.pixels_per_layer),
                "time_by_layer": {
                    name: round(t, 3) for name, t in self.time_by_layer.items()
                },
                "total_time_seconds": round(
                    sum(self.time_by_layer.values()), 3
                ),
            },
        }
        if self.autolabel_plugin is not None:
            d["autolabel_plugin"] = self.autolabel_plugin
        if self.correction_metrics is not None:
            d["correction_metrics"] = self.correction_metrics
        return d
