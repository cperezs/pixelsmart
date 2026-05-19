"""Pure-function annotation tool computations.

All functions take and return NumPy arrays.  There are no side effects,
no Qt dependencies, and no domain state accessed here.  This makes every
function trivially unit-testable.
"""
from __future__ import annotations
from typing import Optional

import cv2
import numpy as np


# ------------------------------------------------------------------
# Interpolation
# ------------------------------------------------------------------

def interpolate_points(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    step: int,
) -> list[tuple[int, int]]:
    """Return evenly-spaced points along the segment from (x0, y0) to (x1, y1).

    The first point returned is always (x0, y0) and the last is always
    (x1, y1).  Intermediate points are placed every *step* pixels along
    the segment.  This prevents visible gaps in fast brush strokes.
    """
    dx = x1 - x0
    dy = y1 - y0
    dist = (dx * dx + dy * dy) ** 0.5
    if dist == 0 or step <= 0:
        return [(x1, y1)]
    n = max(1, int(dist / step))
    points: list[tuple[int, int]] = []
    for i in range(n):
        t = i / n
        points.append((int(x0 + t * dx), int(y0 + t * dy)))
    points.append((x1, y1))
    return points


# ------------------------------------------------------------------
# Pen tool
# ------------------------------------------------------------------

def compute_pen_mask(
    image_height: int,
    image_width: int,
    pos_x: int,
    pos_y: int,
    size: int,
) -> np.ndarray:
    """Return a circular uint8 mask of *size* centred on *(pos_x, pos_y)*."""
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    y, x = np.ogrid[:image_height, :image_width]
    radius = size // 2
    if size % 2 == 0:
        hit = (x - pos_x + 0.5) ** 2 + (y - pos_y + 0.5) ** 2 <= radius ** 2
    else:
        hit = (x - pos_x) ** 2 + (y - pos_y) ** 2 <= radius ** 2
    mask[hit] = 255
    return mask


# ------------------------------------------------------------------
# Mask manipulation
# ------------------------------------------------------------------

def apply_overwrite_guard(mask: np.ndarray, annotated: np.ndarray) -> np.ndarray:
    """Remove pixels already annotated by other layers from *mask*."""
    return cv2.bitwise_and(mask, cv2.bitwise_not(annotated))


def smooth_mask(mask: np.ndarray) -> np.ndarray:
    """Dilate then erode to smooth jagged edges in a binary mask."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    expanded = cv2.dilate(mask, kernel, iterations=1)
    return cv2.erode(expanded, kernel, iterations=1)


def expand_mask(
    mask: np.ndarray,
    annotated: np.ndarray | None = None,
) -> np.ndarray:
    """Grow the selection mask by one pixel in all eight directions.

    If *annotated* is given, already-annotated pixels are excluded from
    the grown region (overwrite guard).
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    grown = cv2.dilate(mask, kernel, iterations=1)
    if annotated is not None:
        grown = apply_overwrite_guard(grown, annotated)
    return grown


def shrink_mask(mask: np.ndarray) -> np.ndarray:
    """Erode the selection mask by one pixel in all eight directions."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.erode(mask, kernel, iterations=1)


# ------------------------------------------------------------------
# Compositing (for the viewer)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Fill tool
# ------------------------------------------------------------------

def compute_fill_mask(
    annotations: np.ndarray,
    active_layer: int,
    click_x: int,
    click_y: int,
) -> np.ndarray:
    """Compute a flood-fill mask starting from (click_x, click_y).

    The fill expands through pixels that have the same label as the seed pixel.
    A pixel's label is the index of its annotated layer (0..N-1), or -1 if
    unannotated.  The fill stops at pixels with a different label or at the
    image boundary.

    Parameters
    ----------
    annotations : np.ndarray
        Shape (N, H, W), dtype uint8, values 0 or 255.
    active_layer : int
        The layer to fill into.
    click_x, click_y : int
        Seed pixel in image coordinates (x = col, y = row).

    Returns
    -------
    np.ndarray
        Boolean or uint8 mask of shape (H, W) — 255 where the fill should
        be applied.
    """
    h, w = annotations.shape[1], annotations.shape[2]

    # Clamp seed to valid range
    seed_x = max(0, min(w - 1, click_x))
    seed_y = max(0, min(h - 1, click_y))

    # Build a label map: -1 = unannotated, 0..N-1 = layer index
    label_map = np.full((h, w), -1, dtype=np.int16)
    for i in range(annotations.shape[0]):
        label_map[annotations[i] > 0] = i

    # Seed label: the label at the clicked pixel
    seed_label = int(label_map[seed_y, seed_x])

    # Create binary mask: True where same label as seed
    same_label = (label_map == seed_label).astype(np.uint8)

    # Flood-fill within same-label region using cv2.floodFill
    # (operates on a copy with a 1-pixel border)
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:h + 1, 1:w + 1] = same_label
    fill_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)  # cv2 requires this size

    cv2.floodFill(
        padded,
        fill_mask,
        (seed_x + 1, seed_y + 1),   # seed in padded coords
        newVal=2,                    # mark visited pixels with 2
        flags=cv2.FLOODFILL_FIXED_RANGE,
    )

    result = np.zeros((h, w), dtype=np.uint8)
    result[padded[1:h + 1, 1:w + 1] == 2] = 255
    return result


# ------------------------------------------------------------------
# Compositing (for the viewer)
# ------------------------------------------------------------------

def build_annotation_rgba(
    annotations: np.ndarray,
    layer_colors: list[tuple[int, int, int]],
    active_layer: int,
    show_other_layers: bool,
    opacity: int = 128,
    hidden_layers: Optional[set] = None,
) -> np.ndarray:
    """Composite all visible annotation layers into a single RGBA image.

    Parameters
    ----------
    annotations:
        Shape (N, H, W), values 0 or 255.
    layer_colors:
        One (R, G, B) tuple per layer, same order as *annotations*.
    active_layer:
        Index of the currently selected layer.
    show_other_layers:
        When False, only the active layer is rendered.
    opacity:
        Alpha value used for annotated pixels (0–255).
    """
    n, h, w = annotations.shape
    composite = np.zeros((h, w, 4), dtype=np.uint8)

    for i in range(n):
        if hidden_layers and i in hidden_layers:
            continue
        if i != active_layer and not show_other_layers:
            continue
        mask = annotations[i] > 0
        r, g, b = layer_colors[i]
        composite[mask] = (r, g, b, opacity)

    # Boost opacity where the primary channel is saturated (visual feedback).
    composite[:, :, 3] = np.where(composite[:, :, 0] == 255, 128, composite[:, :, 3])
    composite[:, :, 3] = np.where(composite[:, :, 2] == 255, 128, composite[:, :, 3])

    return composite


def build_mask_rgba(
    mask: np.ndarray,
    color_rgb: tuple[int, int, int],
    opacity: int = 64,
) -> np.ndarray:
    """Convert a binary mask to a coloured RGBA overlay."""
    h, w = mask.shape
    result = np.zeros((h, w, 4), dtype=np.uint8)
    result[mask > 0] = color_rgb + (opacity,)
    return result
