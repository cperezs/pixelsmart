"""Layer configuration model.

A ``LayerConfig`` describes one annotation layer (name + colour).
``read_layers_file`` / ``write_layers_file`` handle the ``layers.txt``
file format; ``_DEFAULT_LAYERS`` is used whenever the file is absent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


# ------------------------------------------------------------------
# Value object
# ------------------------------------------------------------------

@dataclass(frozen=True)
class LayerConfig:
    """Immutable configuration for a single annotation layer."""

    name: str
    color_hex: str      # hex string e.g. "#FF0000"

    @property
    def color_rgb(self) -> tuple[int, int, int]:
        """Return the colour as an (R, G, B) tuple of ints 0–255."""
        c = self.color_hex.lstrip("#")
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------

_DEFAULT_LAYERS: list[LayerConfig] = [
    LayerConfig("background", "#2596be"),
    LayerConfig("staff",      "#9925be"),
    LayerConfig("notes",      "#be4d25"),
    LayerConfig("lyrics",     "#49be25"),
]


# ------------------------------------------------------------------
# File I/O
# ------------------------------------------------------------------

def read_layers_file(path: str = "layers.txt") -> list[LayerConfig]:
    """Parse *path* and return a list of :class:`LayerConfig` objects.

    Each non-empty line must follow the format::

        <name> [#rrggbb]

    When the file is missing or empty the built-in defaults are returned.
    """
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        return list(_DEFAULT_LAYERS)

    configs: list[LayerConfig] = []
    with open(path, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            name = parts[0]
            color = parts[1] if len(parts) > 1 else "#FF0000"
            configs.append(LayerConfig(name, color))

    return configs or list(_DEFAULT_LAYERS)


def write_layers_file(layers: list[LayerConfig], path: str = "layers.txt") -> None:
    """Serialise *layers* to *path* in the ``<name> #rrggbb`` format."""
    with open(path, "w") as fh:
        for layer in layers:
            fh.write(f"{layer.name} {layer.color_hex}\n")


# ------------------------------------------------------------------
# Name sanitisation
# ------------------------------------------------------------------

import re


def sanitize_name(name: str) -> str:
    """Convert a layer name to a filesystem-safe suffix.

    Examples: ``'My Layer 1!'`` → ``'My_Layer_1_'``,
              ``'  '`` → ``'layer'``.
    """
    result = re.sub(r'[^a-zA-Z0-9_]', '_', name).strip('_')
    return result if result else "layer"
