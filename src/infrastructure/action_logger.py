"""Append-only action logger for research-grade annotation statistics.

Every user action is written as a JSON array to a per-image log file
(``annotations/logs/<stem>.json``).  The log is designed so that all metadata
and statistics (time per layer, pixel counts, correction metrics, etc.) can
be fully reconstructed from the log alone.

Log entry schema
----------------
The file is a JSON array; each element is an object with at least::

    {
      "ts": "2026-04-14T12:34:56.789012",   // ISO-8601 UTC timestamp
      "action": "<ACTION_TYPE>",             // see ACTION_* constants
      ...action-specific fields...
    }

Action types
~~~~~~~~~~~~
- ``image_open``       — image loaded (records dimensions, layer count)
- ``image_close``      — image unloaded (records cumulative time)
- ``pen_stroke``       — pen draw committed
- ``erase_stroke``     — eraser committed
- ``selector_commit``  — selector/magic-wand committed
- ``fill_commit``      — fill tool committed
- ``erase_all``        — clear all annotations
- ``autolabel_start``  — model inference started
- ``autolabel_end``    — model inference finished (includes duration)
- ``session_save``     — periodic metadata flush
"""
from __future__ import annotations

import json
import logging
import os
from collections import deque
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

# Acciones que modifican las anotaciones y pueden eliminarse del log al hacer undo.
# autolabel_end se trata como par con autolabel_start: se eliminan juntos.
_MODIFYING_ACTIONS = frozenset({
    "pen_stroke",
    "erase_stroke",
    "selector_commit",
    "fill_commit",
    "erase_all",
    "autolabel_end",
})


class ActionLogger:
    """Append-only logger that writes one JSON line per user action."""

    def __init__(self, log_path: str, layer_names: list[str]) -> None:
        self._path = log_path
        self._layer_names = list(layer_names)
        self._pre_snapshot: list[np.ndarray] | None = None
        # Pila en memoria para soportar redo del log. Cada elemento es una
        # lista de entradas eliminadas en una sola operación (generalmente
        # un único dict, pero autolabel produce [autolabel_start, autolabel_end]).
        self._redo_log_stack: deque[list[dict]] = deque()

    # ------------------------------------------------------------------
    # Low-level writer
    # ------------------------------------------------------------------

    def _append_serialized(self, line: str) -> None:
        """Append a pre-serialized JSON string as a new element of the log array."""
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            if not os.path.exists(self._path) or os.path.getsize(self._path) == 0:
                with open(self._path, "w") as f:
                    f.write("[\n  " + line + "\n]")
            else:
                # Efficiently append: truncate closing \n] then append new entry
                with open(self._path, "r+b") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    tail_len = min(8, size)
                    f.seek(-tail_len, 2)
                    tail = f.read()
                    idx = tail.rfind(b"\n]")
                    if idx != -1:
                        f.seek(size - tail_len + idx)
                        f.truncate()
                        f.write(b",\n  " + line.encode() + b"\n]")
                    else:
                        # Fallback: rewrite as a fresh single-entry array
                        f.seek(0)
                        f.truncate()
                        f.write(("[\n  " + line + "\n]").encode())
        except OSError as exc:
            logger.warning("ActionLogger: cannot write to %s: %s", self._path, exc)

    def _write(self, entry: dict) -> None:
        entry["ts"] = datetime.now(timezone.utc).isoformat()
        self._append_serialized(json.dumps(entry, separators=(",", ":")))

    # ------------------------------------------------------------------
    # Snapshot helpers (for computing pixel deltas)
    # ------------------------------------------------------------------

    def snapshot_before(self, annotations) -> None:
        """Save a copy of annotation arrays before a mutation."""
        self._pre_snapshot = [a.copy() for a in annotations]

    def compute_delta(self, annotations) -> dict:
        """Compute net pixel change per layer vs. the snapshot.

        Returns a dict with ``per_layer`` (net int per layer, omitted when 0)
        and ``pixels`` (net total: positive = added, negative = removed).
        """
        result: dict = {"per_layer": {}, "pixels": 0}
        if self._pre_snapshot is None:
            return result
        for i, name in enumerate(self._layer_names):
            if i >= len(annotations) or i >= len(self._pre_snapshot):
                break
            old, new = self._pre_snapshot[i], annotations[i]
            added = int(np.count_nonzero((new > 0) & (old == 0)))
            deleted = int(np.count_nonzero((new == 0) & (old > 0)))
            net = added - deleted
            if net:
                result["per_layer"][name] = net
            result["pixels"] += net
        self._pre_snapshot = None
        return result

    # ------------------------------------------------------------------
    # High-level action methods
    # ------------------------------------------------------------------

    def log_image_open(
        self,
        filename: str,
        width: int,
        height: int,
        num_layers: int,
        pixels_per_layer: dict,
    ) -> None:
        self._write({
            "action": "image_open",
            "filename": filename,
            "width": width,
            "height": height,
            "num_layers": num_layers,
            "pixels_per_layer": pixels_per_layer,
        })

    def log_image_close(self, filename: str, time_by_layer: dict) -> None:
        self._write({
            "action": "image_close",
            "filename": filename,
            "cumulative_time_by_layer": {
                k: round(v, 3) for k, v in time_by_layer.items()
            },
        })

    def log_annotation_commit(
        self,
        tool: str,
        layer_index: int,
        layer_name: str,
        delta: dict,
    ) -> None:
        """Log a pen_stroke, erase_stroke, selector_commit, or fill_commit."""
        action_map = {
            "pen": "pen_stroke",
            "eraser": "erase_stroke",
            "selector": "selector_commit",
            "fill": "fill_commit",
        }
        self._write({
            "action": action_map.get(tool, f"{tool}_commit"),
            "layer_index": layer_index,
            "layer_name": layer_name,
            **delta,
        })

    def pop_last_annotation_entry(self) -> dict | None:
        """Elimina la última entrada modificadora del fichero de log y la devuelve.

        Busca desde el final del array la última entrada cuyo campo ``action``
        pertenezca a ``_MODIFYING_ACTIONS``.  Si no existe ninguna, el log no
        se modifica y devuelve ``None``.

        Si la entrada encontrada es ``autolabel_end``, se busca y elimina también
        el ``autolabel_start`` precedente: ambas se guardan juntas en
        ``_redo_log_stack`` como un grupo ``[start, end]``.
        """
        if not os.path.exists(self._path) or os.path.getsize(self._path) == 0:
            return None
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                entries: list[dict] = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("ActionLogger: cannot read %s: %s", self._path, exc)
            return None

        # Localizar la última entrada modificadora.
        last_idx = None
        for i in range(len(entries) - 1, -1, -1):
            if entries[i].get("action") in _MODIFYING_ACTIONS:
                last_idx = i
                break

        if last_idx is None:
            return None

        last_entry = entries[last_idx]

        if last_entry.get("action") == "autolabel_end":
            # Buscar el autolabel_start precedente y eliminar el par.
            start_idx = None
            for i in range(last_idx - 1, -1, -1):
                if entries[i].get("action") == "autolabel_start":
                    start_idx = i
                    break
            # Eliminar en orden descendente para no alterar índices.
            end_entry = entries.pop(last_idx)
            if start_idx is not None:
                start_entry = entries.pop(start_idx)
                group: list[dict] = [start_entry, end_entry]
            else:
                group = [end_entry]
        else:
            group = [entries.pop(last_idx)]

        # Reescribir el fichero sin las entradas eliminadas.
        try:
            if entries:
                lines = ",\n".join(
                    "  " + json.dumps(e, separators=(",", ":")) for e in entries
                )
                content = "[\n" + lines + "\n]"
            else:
                content = "[]\n"
            with open(self._path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as exc:
            logger.warning("ActionLogger: cannot rewrite %s: %s", self._path, exc)
            return None

        self._redo_log_stack.append(group)
        return group[-1]

    def replay_entry(self, group: list[dict]) -> None:
        """Reinserta el grupo de entradas *group* en el log (redo del log).

        Para anotaciones simples *group* tiene un único elemento; para autolabel
        contiene ``[autolabel_start, autolabel_end]``.  Los campos originales
        (incluido ``ts``) se preservan tal cual.
        """
        if self._redo_log_stack:
            self._redo_log_stack.pop()
        for entry in group:
            self._append_serialized(json.dumps(entry, separators=(",", ":")))

    def discard_redo_log_entries(self) -> None:
        """Vacía ``_redo_log_stack``.  Llamar cuando una nueva anotación invalida el redo stack."""
        self._redo_log_stack.clear()

    def log_erase_all(self, delta: dict) -> None:
        self._write({"action": "erase_all", **delta})

    def clear_log(self) -> None:
        """Truncate the log file to an empty array and discard the redo stack."""
        self._redo_log_stack.clear()
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                f.write("[]\n")
        except OSError as exc:
            logger.warning("ActionLogger: cannot clear %s: %s", self._path, exc)

    def log_autolabel_start(
        self,
        plugin_id: str,
        model_config: dict | None = None,
    ) -> None:
        entry: dict = {"action": "autolabel_start", "plugin_id": plugin_id}
        if model_config is not None:
            entry["model_config"] = model_config
        self._write(entry)

    def log_autolabel_end(
        self,
        plugin_id: str,
        duration_seconds: float,
        success: bool,
        error: str | None,
        pixels_per_layer: dict | None = None,
    ) -> None:
        entry: dict = {
            "action": "autolabel_end",
            "plugin_id": plugin_id,
            "duration_seconds": round(duration_seconds, 3),
            "success": success,
        }
        if error:
            entry["error"] = error
        if pixels_per_layer is not None:
            entry["pixels_per_layer"] = pixels_per_layer
        self._write(entry)

    def log_session_save(self, pixels_per_layer: dict, time_by_layer: dict) -> None:
        self._write({
            "action": "session_save",
            "pixels_per_layer": pixels_per_layer,
            "time_by_layer": {
                k: round(v, 3) for k, v in time_by_layer.items()
            },
        })
