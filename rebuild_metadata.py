#!/usr/bin/env python3
"""Rebuild a .metadata file from its corresponding action log.

Usage
-----
    python rebuild_metadata.py <log_file> [--layers-file layers.txt] [--output metadata_file]

If ``--output`` is omitted the script writes to stdout.

The script replays every logged action to reconstruct the canonical
``ImageMetadata`` JSON that the application would have produced.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def _read_layers(layers_path: str) -> list[str]:
    """Read layer names from a layers.txt file."""
    names: list[str] = []
    with open(layers_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            names.append(parts[0])
    return names


def rebuild(log_path: str, layer_names: list[str]) -> dict:
    """Replay *log_path* and return a metadata dict."""
    # Accumulated state
    time_by_layer: dict[str, float] = {n: 0.0 for n in layer_names}
    pixels_per_layer: dict[str, int] = {n: 0 for n in layer_names}
    total_pixels: int = 0
    autolabel_plugin: str | None = None
    correction_metrics: dict | None = None

    # Track cumulative time from image_close and session_save events
    last_time_snapshot: dict[str, float] | None = None

    # Timestamp-based time tracking (mirrors TimeTracker.change())
    # Used when image_close / session_save are absent from the log.
    _ts_current_layer: str | None = None
    _ts_layer_start: datetime | None = None

    def _parse_ts(ts_str: str) -> datetime | None:
        try:
            return datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            return None

    def _flush_ts_timer(ts: datetime) -> None:
        """Add elapsed seconds since the last timer reset to the current layer."""
        nonlocal _ts_current_layer, _ts_layer_start
        if _ts_current_layer is not None and _ts_layer_start is not None:
            elapsed = (ts - _ts_layer_start).total_seconds()
            if elapsed > 0 and _ts_current_layer in time_by_layer:
                time_by_layer[_ts_current_layer] += elapsed

    # Counters for research statistics
    total_pen_strokes = 0
    total_selector_commits = 0
    total_fill_commits = 0
    total_erase_strokes = 0
    total_undos = 0
    total_erase_all = 0
    total_pixels_added = 0
    total_pixels_deleted = 0
    total_autolabel_runs = 0
    total_autolabel_time = 0.0
    tool_changes = 0
    layer_changes = 0
    sessions = 0  # number of image_open events

    # Per-layer correction tracking (after autolabel)
    per_layer_corrections: dict[str, dict] = {
        n: {"additions": 0, "deletions": 0, "pixels_added": 0, "pixels_deleted": 0}
        for n in layer_names
    }
    tracking_corrections = False

    with open(log_path) as f:
        try:
            entries = json.load(f)
        except json.JSONDecodeError as exc:
            print(f"ERROR: cannot parse log file: {exc}", file=sys.stderr)
            sys.exit(1)

    for entry_no, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            print(f"WARNING: skipping non-object entry #{entry_no}", file=sys.stderr)
            continue

        action = entry.get("action", "")

        ts = _parse_ts(entry.get("ts", ""))

        if action == "image_open":
            sessions += 1
            total_pixels = entry.get("width", 0) * entry.get("height", 0)
            ppl = entry.get("pixels_per_layer", {})
            for name in layer_names:
                if name in ppl:
                    pixels_per_layer[name] = ppl[name]
            # Mirror time_tracker.change(0): don't flush (avoids counting idle
            # time between sessions), start timing layer 0.
            _ts_current_layer = layer_names[0] if layer_names else None
            _ts_layer_start = ts

        elif action == "image_close":
            # Authoritative logged times — overwrite timestamp-derived values.
            ct = entry.get("cumulative_time_by_layer", {})
            for name in layer_names:
                if name in ct:
                    time_by_layer[name] = ct[name]
            last_time_snapshot = dict(time_by_layer)
            _ts_current_layer = None
            _ts_layer_start = None

        elif action == "session_save":
            ppl = entry.get("pixels_per_layer", {})
            for name in layer_names:
                if name in ppl:
                    pixels_per_layer[name] = ppl[name]
            ct = entry.get("time_by_layer", {})
            for name in layer_names:
                if name in ct:
                    time_by_layer[name] = ct[name]
            # Restart the timestamp timer from this snapshot.
            _ts_layer_start = ts

        elif action == "tool_select":
            tool_changes += 1

        elif action == "layer_select":
            layer_changes += 1
            layer_index = entry.get("layer_index", 0)
            new_layer_name = (
                layer_names[layer_index]
                if layer_index < len(layer_names)
                else entry.get("layer_name")
            )
            # Mirror TimeTracker.change(): flush elapsed to current layer,
            # then start timing the new layer.
            if ts is not None:
                _flush_ts_timer(ts)
                _ts_current_layer = new_layer_name
                _ts_layer_start = ts

        elif action in ("pen_stroke", "selector_commit", "fill_commit", "erase_stroke"):
            delta_per_layer = entry.get("per_layer", {})
            net = entry.get("pixels", 0)
            if net >= 0:
                total_pixels_added += net
            else:
                total_pixels_deleted += abs(net)

            if action == "pen_stroke":
                total_pen_strokes += 1
            elif action == "selector_commit":
                total_selector_commits += 1
            elif action == "fill_commit":
                total_fill_commits += 1
            elif action == "erase_stroke":
                total_erase_strokes += 1

            # Update pixels_per_layer from net deltas
            for name, net_delta in delta_per_layer.items():
                if name in pixels_per_layer:
                    pixels_per_layer[name] += net_delta

            # Track post-autolabel corrections
            if tracking_corrections:
                for name, net_delta in delta_per_layer.items():
                    if name in per_layer_corrections:
                        if net_delta > 0:
                            per_layer_corrections[name]["additions"] += 1
                            per_layer_corrections[name]["pixels_added"] += net_delta
                        elif net_delta < 0:
                            per_layer_corrections[name]["deletions"] += 1
                            per_layer_corrections[name]["pixels_deleted"] += abs(net_delta)

        elif action == "undo":
            total_undos += 1
            net = entry.get("pixels", 0)
            if net >= 0:
                total_pixels_added += net
            else:
                total_pixels_deleted += abs(net)
            delta_per_layer = entry.get("per_layer", {})
            for name, net_delta in delta_per_layer.items():
                if name in pixels_per_layer:
                    pixels_per_layer[name] += net_delta

        elif action == "erase_all":
            total_erase_all += 1
            net = entry.get("pixels", 0)
            if net >= 0:
                total_pixels_added += net
            else:
                total_pixels_deleted += abs(net)
            for name in layer_names:
                pixels_per_layer[name] = 0

        elif action == "autolabel_start":
            pass

        elif action == "autolabel_end":
            total_autolabel_runs += 1
            total_autolabel_time += entry.get("duration_seconds", 0)
            if entry.get("success"):
                autolabel_plugin = entry.get("plugin_id")
                ppl = entry.get("pixels_per_layer", {})
                for name in layer_names:
                    if name in ppl:
                        pixels_per_layer[name] = ppl[name]
                # Reset correction tracking
                tracking_corrections = True
                per_layer_corrections = {
                    n: {"additions": 0, "deletions": 0, "pixels_added": 0, "pixels_deleted": 0}
                    for n in layer_names
                }
                # Reset time (mirrors TimeTracker.reset())
                time_by_layer = {n: 0.0 for n in layer_names}
                _ts_current_layer = None
                _ts_layer_start = None

    # Build output metadata
    result: dict = {
        "schema_version": 1,
        "global_metrics": {
        "total_pixels": total_pixels,
        "pixels_per_layer": {n: max(0, pixels_per_layer.get(n, 0)) for n in layer_names},
        "time_by_layer": {n: round(time_by_layer.get(n, 0.0), 3) for n in layer_names},
        "total_time_seconds": round(sum(time_by_layer.values()), 3),
        },
    }

    if autolabel_plugin:
        result["autolabel_plugin"] = autolabel_plugin
    if tracking_corrections:
        total_ops = sum(
        v["additions"] + v["deletions"] for v in per_layer_corrections.values()
        )
        total_mod = sum(
        v["pixels_added"] + v["pixels_deleted"]
        for v in per_layer_corrections.values()
        )
        result["correction_metrics"] = {
        "total_operations": total_ops,
        "total_modified_pixels": total_mod,
        "per_layer": per_layer_corrections,
        }

    # Extended  statistics (not in the original schema but useful
    # for analysis — stored under a separate key so they don't interfere)
    result["extra_stats"] = {
        "sessions": sessions,
        "total_pen_strokes": total_pen_strokes,
        "total_selector_commits": total_selector_commits,
        "total_fill_commits": total_fill_commits,
        "total_erase_strokes": total_erase_strokes,
        "total_undos": total_undos,
        "total_erase_all": total_erase_all,
        "total_pixels_added": total_pixels_added,
        "total_pixels_deleted": total_pixels_deleted,
        "total_autolabel_runs": total_autolabel_runs,
        "total_autolabel_time_seconds": round(total_autolabel_time, 3),
        "tool_changes": tool_changes,
        "layer_changes": layer_changes,
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild metadata from an action log file.",
    )
    parser.add_argument("log_file", help="Path to the log (.json) file")
    parser.add_argument(
        "--layers-file",
        default="layers.txt",
        help="Path to layers.txt (default: layers.txt in cwd)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output metadata (.json) file (default: stdout)",
    )
    args = parser.parse_args()

    layer_names = _read_layers(args.layers_file)
    result = rebuild(args.log_file, layer_names)
    output = json.dumps(result, indent=2)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output + "\n")
        print(f"Wrote metadata to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
