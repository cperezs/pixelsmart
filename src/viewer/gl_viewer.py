"""OpenGL-based implementation of IImageAnnotationViewer.

Drop-in replacement for ``QtImageAnnotationViewer`` that renders with a
``QOpenGLWidget`` backend.  Swapping viewers requires only changing the
import in the presentation layer; no controller or domain changes needed.

Architecture
------------
``_Viewport``
    Zoom, pan, and coordinate transforms (screen ↔ image ↔ NDC).

``_LayerData``
    Per-layer CPU data (numpy array), GPU texture id, visibility, and
    style parameters (color, opacity).

``_GLCanvas(QOpenGLWidget)``
    All OpenGL objects (shaders, VAO/VBO, textures) and the full render
    pipeline.  Communicates with the outer viewer via shared references
    to ``_Viewport`` and the layer dict.

``GLImageAnnotationViewer(QWidget)``
    Public interface.  Wraps the canvas, implements all
    ``IImageAnnotationViewer`` methods, and translates Qt input events
    into semantic callbacks expressed in image-pixel coordinates.

Performance notes
-----------------
* Only the visible image region is drawn (viewport-culled quad per layer).
* Textures are uploaded lazily in ``paintGL``; data set calls just mark
  layers as dirty and schedule a repaint.
* Partial annotation updates can be added later via ``glTexSubImage2D``.
* The pixel grid is drawn entirely on the GPU via a fragment shader.
* TODO: HiDPI — use ``self.devicePixelRatio()`` for physical-pixel
  ``glViewport``; currently uses logical pixels for simplicity.
"""
from __future__ import annotations

import ctypes
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

import time as _time_module

from PyQt6.QtCore import Qt, QEvent, QObject, QTimer
from PyQt6.QtGui import QColor, QCursor, QImage, QMouseEvent, QPixmap, QSurfaceFormat
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QVBoxLayout, QWidget

import OpenGL.GL as GL

from viewer.interface import IImageAnnotationViewer  # type-annotation only

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RESOURCES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "resources")
_TOOL_CURSOR_FILES = {"pen": "pen.png", "selector": "wand.png", "fill": "fill.png"}

_GRID_MIN_ZOOM = 3   # show pixel grid when zoom >= this value
_STRIDE = 4 * 4      # bytes per vertex: 4 float32 components (pos.xy, uv.xy)

# Layer draw-order keys (lowest first)
_L_BASE    = "base"
_L_ANN     = "ann"
_L_MISSING = "missing"
_L_SEL     = "sel"
_L_TOOL    = "tool"
_LAYER_ORDER = (_L_BASE, _L_ANN, _L_MISSING, _L_SEL, _L_TOOL)

# ---------------------------------------------------------------------------
# GLSL shaders
# ---------------------------------------------------------------------------

_VERT_SRC = """
#version 330 core
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_uv = a_uv;
}
"""

# RGBA texture with configurable opacity (used for image and RGBA overlays).
_FRAG_RGBA_SRC = """
#version 330 core
uniform sampler2D u_tex;
uniform float     u_opacity;
in  vec2 v_uv;
out vec4 frag;
void main() {
    vec4 c = texture(u_tex, v_uv);
    frag = vec4(c.rgb, c.a * u_opacity);
}
"""

# Single-channel (R8) mask → flat coloured overlay.
_FRAG_MASK_SRC = """
#version 330 core
uniform sampler2D u_tex;
uniform vec3      u_color;
uniform float     u_opacity;
in  vec2 v_uv;
out vec4 frag;
void main() {
    float v = texture(u_tex, v_uv).r;
    if (v < 0.01) discard;
    frag = vec4(u_color, u_opacity);
}
"""

# Pixel-grid overlay computed entirely on the GPU.
# v_uv carries normalised screen coordinates [0..1] (0 = top-left).
_FRAG_GRID_SRC = """
#version 330 core
uniform float u_zoom;
uniform vec2  u_pan;
uniform vec2  u_screen;
in  vec2 v_uv;
out vec4 frag;
void main() {
    // Reconstruct screen-space fragment centre (logical pixels, y-down).
    vec2 s = v_uv * u_screen;

    // Draw a 1-screen-pixel-wide grid at every image-pixel boundary.
    //
    // A fragment is on a grid line when its screen-pixel footprint [s-0.5, s+0.5)
    // straddles an image-pixel boundary (integer image coordinate).
    // Equivalent condition: floor(img_hi) != floor(img_lo)
    //   where img_lo / img_hi are the image-space coords of the pixel's two edges.
    //
    // A tiny negative bias (-eps) on both edges breaks the tie that occurs when
    // a grid line falls exactly halfway between two fragment centres (which happens
    // whenever zoom is a whole number and pan is integer-aligned).  Without it,
    // such lines are either doubled or skipped entirely.
    float eps = 1.0 / (u_zoom * 8.0);
    vec2 img_lo = (s - 0.5 - u_pan) / u_zoom - eps;
    vec2 img_hi = (s + 0.5 - u_pan) / u_zoom - eps;

    if (floor(img_hi.x) == floor(img_lo.x) &&
        floor(img_hi.y) == floor(img_lo.y)) discard;

    frag = vec4(0.5, 0.5, 0.5, 0.7);
}
"""

# RGBA annotation overlay — semi-transparent fill with a 1-screen-pixel inner border.
#
# The border width is always exactly 1 screen pixel regardless of zoom.
# It is strictly INSIDE the annotated area (inner-only): no outer ring.
# Border color = same hue as the annotation, near-full opacity.
#
# Key idea for zoom-invariant width:
#   fract(v_uv / u_texel) gives the normalised position within the current
#   texel in [0, 1).  Multiplying by u_zoom converts to screen pixels.
#   A fragment is on the inner border when its screen-pixel distance to at
#   least one adjacent empty edge is < 1.0.
_FRAG_ANN_SRC = """
#version 330 core
uniform sampler2D u_tex;
uniform float     u_opacity;
uniform vec2      u_texel;   // 1.0 / vec2(img_w, img_h)
uniform float     u_zoom;    // pixels per texel on screen
in  vec2 v_uv;
out vec4 frag;
void main() {
    vec4 c = texture(u_tex, v_uv);
    if (c.a < 0.01) discard;

    // Sample 4-connected neighbors (UV.y increases downward in our convention)
    float n = texture(u_tex, v_uv - vec2(0.0,    u_texel.y)).a;  // row above
    float s = texture(u_tex, v_uv + vec2(0.0,    u_texel.y)).a;  // row below
    float e = texture(u_tex, v_uv + vec2(u_texel.x,    0.0)).a;  // col right
    float w = texture(u_tex, v_uv - vec2(u_texel.x,    0.0)).a;  // col left

    // Normalised position within the current texel [0, 1) for both axes.
    vec2 frac_uv = fract(v_uv / u_texel);

    // Screen-pixel distance to the adjacent edge in each direction.
    float dn = (n < 0.01) ? frac_uv.y         * u_zoom : 1e9;
    float ds = (s < 0.01) ? (1.0 - frac_uv.y) * u_zoom : 1e9;
    float de = (e < 0.01) ? (1.0 - frac_uv.x) * u_zoom : 1e9;
    float dw = (w < 0.01) ? frac_uv.x         * u_zoom : 1e9;
    float min_dist = min(min(dn, ds), min(de, dw));

    if (min_dist < 1.0) {
        // Inner border: more opaque than fill, scaled with global opacity
        frag = vec4(c.rgb, clamp(u_opacity * 1.5, 0.0, 0.95));
    } else {
        // Fill: u_opacity controls directly (0=invisible, 1=solid)
        frag = vec4(c.rgb, u_opacity);
    }
}
"""

# Grayscale preliminary selection — semi-transparent fill + animated 1-screen-pixel
# inner marching-ants border.  The dash pattern is computed in screen space so
# it always appears the same size regardless of zoom.
_FRAG_SEL_SRC = """
#version 330 core
uniform sampler2D u_tex;
uniform vec3      u_color;
uniform float     u_opacity;
uniform float     u_time;
uniform vec2      u_texel;
uniform float     u_zoom;
in  vec2 v_uv;
out vec4 frag;
void main() {
    float c = texture(u_tex, v_uv).r;
    if (c < 0.5) discard;

    float n = texture(u_tex, v_uv - vec2(0.0,    u_texel.y)).r;
    float s = texture(u_tex, v_uv + vec2(0.0,    u_texel.y)).r;
    float e = texture(u_tex, v_uv + vec2(u_texel.x,    0.0)).r;
    float w = texture(u_tex, v_uv - vec2(u_texel.x,    0.0)).r;

    vec2 frac_uv = fract(v_uv / u_texel);
    float dn = (n < 0.5) ? frac_uv.y         * u_zoom : 1e9;
    float ds = (s < 0.5) ? (1.0 - frac_uv.y) * u_zoom : 1e9;
    float de = (e < 0.5) ? (1.0 - frac_uv.x) * u_zoom : 1e9;
    float dw = (w < 0.5) ? frac_uv.x         * u_zoom : 1e9;
    float min_dist = min(min(dn, ds), min(de, dw));

    if (min_dist < 1.0) {
        // Marching ants: compute dash in screen-pixel space (independent of zoom)
        vec2 screen_pos = (v_uv / u_texel) * u_zoom;
        float dash = mod(screen_pos.x + screen_pos.y - u_time * 60.0, 10.0);
        frag = (dash < 5.0) ? vec4(1.0, 1.0, 1.0, 0.95)
                             : vec4(0.05, 0.05, 0.05, 0.95);
    } else {
        frag = vec4(u_color, u_opacity);
    }
}
"""

# Pulsing white glow overlay for unannotated (missing) pixels.
# Opacity oscillates between ~0.1 and ~0.75 using a sine wave (~2.5 s period).
_FRAG_GLOW_SRC = """
#version 330 core
uniform sampler2D u_tex;
uniform float     u_time;
in  vec2 v_uv;
out vec4 frag;
void main() {
    float v = texture(u_tex, v_uv).r;
    if (v < 0.01) discard;
    float pulse = 0.10 + 0.65 * (0.5 + 0.5 * sin(u_time * 2.5));
    frag = vec4(1.0, 1.0, 1.0, pulse);
}
"""

# ---------------------------------------------------------------------------
# Shader helpers
# ---------------------------------------------------------------------------

def _compile_shader(kind: int, source: str) -> int:
    sid = GL.glCreateShader(kind)
    GL.glShaderSource(sid, source)
    GL.glCompileShader(sid)
    if not GL.glGetShaderiv(sid, GL.GL_COMPILE_STATUS):
        log = GL.glGetShaderInfoLog(sid)
        if isinstance(log, bytes):
            log = log.decode(errors="replace")
        GL.glDeleteShader(sid)
        raise RuntimeError(f"Shader compile error:\n{log}")
    return sid


def _link_program(vert_src: str, frag_src: str) -> int:
    v = _compile_shader(GL.GL_VERTEX_SHADER, vert_src)
    f = _compile_shader(GL.GL_FRAGMENT_SHADER, frag_src)
    prog = GL.glCreateProgram()
    GL.glAttachShader(prog, v)
    GL.glAttachShader(prog, f)
    GL.glLinkProgram(prog)
    GL.glDeleteShader(v)
    GL.glDeleteShader(f)
    if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
        log = GL.glGetProgramInfoLog(prog)
        if isinstance(log, bytes):
            log = log.decode(errors="replace")
        GL.glDeleteProgram(prog)
        raise RuntimeError(f"Program link error:\n{log}")
    return prog


# ---------------------------------------------------------------------------
# Viewport — zoom / pan / coordinate conversions
# ---------------------------------------------------------------------------

class _Viewport:
    """Stores zoom and pan; converts between screen, image, and NDC spaces.

    ``pan_x``, ``pan_y``
        Position (in logical screen pixels) of the image's top-left
        corner relative to the widget's top-left corner.

    ``screen_w``, ``screen_h``
        Current widget dimensions in logical pixels.
    """

    def __init__(self) -> None:
        self.zoom: float = 5.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self.screen_w: int = 1
        self.screen_h: int = 1

    def clamp_pan(self, img_w: int, img_h: int) -> None:
        """Clamp pan so the image cannot be scrolled beyond its edges.

        Ensures at least a 1-pixel margin of the image stays on screen.
        """
        min_pan_x = min(0.0, self.screen_w - img_w * self.zoom)
        min_pan_y = min(0.0, self.screen_h - img_h * self.zoom)
        self.pan_x = max(min_pan_x, min(0.0, self.pan_x))
        self.pan_y = max(min_pan_y, min(0.0, self.pan_y))

    def screen_to_image(self, sx: float, sy: float) -> tuple[int, int]:
        """Screen pixel → image pixel coordinates (integer, clamped)."""
        return (
            int((sx - self.pan_x) / self.zoom),
            int((sy - self.pan_y) / self.zoom),
        )

    def image_center(self) -> tuple[float, float]:
        """Current viewport centre in image-pixel coordinates."""
        cx = (self.screen_w / 2.0 - self.pan_x) / self.zoom
        cy = (self.screen_h / 2.0 - self.pan_y) / self.zoom
        return float(cx), float(cy)

    def visible_image_rect(
        self, img_w: int, img_h: int
    ) -> tuple[float, float, float, float]:
        """Visible portion of the image in image coords: (x0, y0, x1, y1)."""
        x0 = max(0.0, -self.pan_x / self.zoom)
        y0 = max(0.0, -self.pan_y / self.zoom)
        x1 = min(float(img_w), (self.screen_w - self.pan_x) / self.zoom)
        y1 = min(float(img_h), (self.screen_h - self.pan_y) / self.zoom)
        return x0, y0, x1, y1

    def _to_ndc(self, sx: float, sy: float) -> tuple[float, float]:
        """Logical screen pixel → OpenGL NDC (Y-flipped)."""
        return (
            2.0 * sx / self.screen_w - 1.0,
            1.0 - 2.0 * sy / self.screen_h,
        )

    def quad_for_image(self, img_w: int, img_h: int) -> np.ndarray:
        """Return (6, 4) float32 vertex data for the visible image region.

        Each row: ``[ndc_x, ndc_y, uv_x, uv_y]``.
        Returns an empty array when the image is fully off-screen.
        """
        x0, y0, x1, y1 = self.visible_image_rect(img_w, img_h)
        if x0 >= x1 or y0 >= y1:
            return np.empty((0, 4), dtype=np.float32)

        sx0 = x0 * self.zoom + self.pan_x
        sy0 = y0 * self.zoom + self.pan_y
        sx1 = x1 * self.zoom + self.pan_x
        sy1 = y1 * self.zoom + self.pan_y

        nx0, ny0 = self._to_ndc(sx0, sy0)
        nx1, ny1 = self._to_ndc(sx1, sy1)

        # UV X maps directly.
        # UV Y: glTexImage2D stores data[0] at UV.y=0 (OpenGL bottom-left origin),
        # which is the TOP row of the numpy array.  So UV.y=0 → image top,
        # UV.y=1 → image bottom — no flip needed.
        ux0, ux1 = x0 / img_w, x1 / img_w
        uy_top = y0 / img_h   # screen top    → image row y0  → UV y0/h
        uy_bot = y1 / img_h   # screen bottom → image row y1  → UV y1/h

        # Two CCW triangles: TL, BL, TR, TR, BL, BR
        return np.array([
            [nx0, ny0, ux0, uy_top],  # top-left
            [nx0, ny1, ux0, uy_bot],  # bottom-left
            [nx1, ny0, ux1, uy_top],  # top-right
            [nx1, ny0, ux1, uy_top],  # top-right  (triangle 2)
            [nx0, ny1, ux0, uy_bot],  # bottom-left
            [nx1, ny1, ux1, uy_bot],  # bottom-right
        ], dtype=np.float32)

    def fullscreen_quad(self) -> np.ndarray:
        """Full-screen quad; v_uv carries normalised screen coords (0 = top-left)."""
        return np.array([
            [-1.0,  1.0, 0.0, 0.0],
            [-1.0, -1.0, 0.0, 1.0],
            [ 1.0,  1.0, 1.0, 0.0],
            [ 1.0,  1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0, 1.0],
            [ 1.0, -1.0, 1.0, 1.0],
        ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Layer data
# ---------------------------------------------------------------------------

@dataclass
class _LayerData:
    """CPU-side image data and GPU texture state for one render layer."""

    data: Optional[np.ndarray] = None
    # "rgb" | "rgba" | "gray" — matches the numpy array's channel layout
    fmt: str = "rgba"
    # "rgba" | "mask" | "ann_border" | "sel_ants" — selects the shader program
    shader: str = "rgba"
    dirty: bool = False       # True → needs upload in next paintGL
    tex_id: int = 0           # 0 = texture not yet allocated
    visible: bool = True
    opacity: float = 1.0
    # For grayscale ("gray") layers only — rendered with the mask shader
    color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Gesture filter (trackpad pinch-to-zoom)
# ---------------------------------------------------------------------------

class _GestureFilter(QObject):
    """Forwards QNativeGestureEvent (trackpad pinch) to a callback."""

    def __init__(self, callback: Callable, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._cb = callback

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.NativeGesture:
            self._cb(event)
            return True
        return super().eventFilter(obj, event)


# ---------------------------------------------------------------------------
# OpenGL canvas
# ---------------------------------------------------------------------------

class _GLCanvas(QOpenGLWidget):
    """Low-level OpenGL rendering widget.

    Owns all GL objects (shader programs, VAO, VBO, textures).
    Driven exclusively by the shared ``_Viewport`` and layer dict;
    contains no application or domain logic.
    """

    def __init__(
        self,
        vp: _Viewport,
        layers: dict[str, _LayerData],
        parent: Optional[QWidget] = None,
    ) -> None:
        # Request OpenGL 3.3 Core Profile on this widget explicitly.
        # This overrides any format already set via QSurfaceFormat.setDefaultFormat()
        # and ensures macOS does not fall back to a legacy 2.1 context.
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(0)
        fmt.setStencilBufferSize(0)

        super().__init__(parent)
        self.setFormat(fmt)

        self._vp = vp
        self._layers = layers
        self._img_w: int = 0
        self._img_h: int = 0
        self._grid_visible: bool = False
        self._global_ann_opacity: float = 0.7

        # GL objects — allocated in initializeGL()
        self._prog_rgba: int = 0
        self._prog_mask: int = 0
        self._prog_grid: int = 0
        self._prog_ann:  int = 0
        self._prog_sel:  int = 0
        self._prog_glow: int = 0
        self._vao: int = 0
        self._vbo: int = 0

        # Textures queued for deletion (must happen in GL context)
        self._to_delete: list[int] = []

        # Animation clock — drives the marching-ants selection border.
        self._anim_start: float = _time_module.monotonic()
        self._anim_time:  float = 0.0
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(150)  # ~6 fps — drives both marching-ants and glow
        self._anim_timer.timeout.connect(self._tick_animation)

        self.setMouseTracking(True)

    # ── QOpenGLWidget lifecycle ──────────────────────────────────────────

    def initializeGL(self) -> None:  # noqa: N802
        # Clear any stale GL errors accumulated during context creation.
        # PyOpenGL's error-checking wrapper fires on the *next* call if there
        # is a pending error, even if that next call would succeed.
        _count = 0
        while GL.glGetError() != GL.GL_NO_ERROR:
            _count += 1
            if _count > 10:
                break
        if _count:
            logger.debug("Cleared %d stale GL error(s) before init", _count)

        ver = GL.glGetString(GL.GL_VERSION)
        logger.info("OpenGL version: %s", ver)

        try:
            self._prog_rgba = _link_program(_VERT_SRC, _FRAG_RGBA_SRC)
            self._prog_mask = _link_program(_VERT_SRC, _FRAG_MASK_SRC)
            self._prog_grid = _link_program(_VERT_SRC, _FRAG_GRID_SRC)
            self._prog_ann  = _link_program(_VERT_SRC, _FRAG_ANN_SRC)
            self._prog_sel  = _link_program(_VERT_SRC, _FRAG_SEL_SRC)
            self._prog_glow = _link_program(_VERT_SRC, _FRAG_GLOW_SRC)
        except RuntimeError:
            logger.exception("GL shader initialisation failed")
            return

        self._vao = int(GL.glGenVertexArrays(1))
        self._vbo = int(GL.glGenBuffers(1))

        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        # Reserve space for 6 vertices (2 triangles)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 6 * _STRIDE, None, GL.GL_DYNAMIC_DRAW)
        # a_pos: 2 floats at offset 0
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, _STRIDE,
                                 ctypes.c_void_p(0))
        # a_uv: 2 floats at offset 8
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, _STRIDE,
                                 ctypes.c_void_p(8))
        GL.glBindVertexArray(0)

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glClearColor(0.15, 0.15, 0.15, 1.0)

    def resizeGL(self, w: int, h: int) -> None:  # noqa: N802
        # Use logical pixel dimensions to stay consistent with mouse-event coords.
        self._vp.screen_w = self.width()
        self._vp.screen_h = self.height()
        GL.glViewport(0, 0, self.width(), self.height())

    def paintGL(self) -> None:  # noqa: N802
        # Guard: if initializeGL failed (e.g. unsupported GL version), skip drawing
        # to avoid cascading GL errors from uninitialised resources.
        if self._vbo == 0:
            return

        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        # Delete any textures scheduled for removal
        if self._to_delete:
            GL.glDeleteTextures(len(self._to_delete), self._to_delete)
            self._to_delete.clear()

        if self._img_w == 0:
            return

        # Upload any layers whose data has changed since the last paint
        for layer in self._layers.values():
            if layer.dirty and layer.data is not None:
                self._upload_texture(layer)
                layer.dirty = False

        # Draw in fixed z-order
        for name in _LAYER_ORDER:
            layer = self._layers.get(name)
            if layer is None or not layer.visible or layer.tex_id == 0:
                continue
            if layer.shader == "sel_ants":
                self._draw_sel_layer(layer)
            elif layer.shader == "ann_border":
                self._draw_ann_layer(layer)
            elif layer.shader == "glow":
                self._draw_glow_layer(layer)
            elif layer.fmt == "gray":
                self._draw_mask_layer(layer)
            else:
                self._draw_rgba_layer(layer)

        if self._grid_visible and self._vp.zoom >= _GRID_MIN_ZOOM:
            self._draw_grid()

    # ── Texture management ───────────────────────────────────────────────

    def _upload_texture(self, layer: _LayerData) -> None:
        """Upload (or re-upload) a layer's data to its GPU texture."""
        data = layer.data
        h, w = data.shape[:2]

        if layer.tex_id == 0:
            tid = GL.glGenTextures(1)
            layer.tex_id = int(tid) if not hasattr(tid, "__len__") else int(tid[0])

        GL.glBindTexture(GL.GL_TEXTURE_2D, layer.tex_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)

        buf = np.ascontiguousarray(data)

        # OpenGL's default GL_UNPACK_ALIGNMENT is 4, meaning each row must start
        # on a 4-byte boundary.  For RGB (3 bytes/pixel) or grayscale (1 byte/pixel)
        # images whose width is not a multiple of 4, this causes OpenGL to read
        # past the end of the buffer → segmentation fault on large images.
        # Set alignment to 1 (byte-tight) for non-RGBA formats.
        if layer.fmt in ("rgb", "gray"):
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)

        if layer.fmt == "rgb":
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8,
                            w, h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, buf)
        elif layer.fmt == "rgba":
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8,
                            w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, buf)
        elif layer.fmt == "gray":
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R8,
                            w, h, 0, GL.GL_RED, GL.GL_UNSIGNED_BYTE, buf)

        # Restore default alignment
        if layer.fmt in ("rgb", "gray"):
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def partial_update_texture(
        self,
        layer: _LayerData,
        x: int, y: int, patch: np.ndarray,
    ) -> None:
        """Upload a sub-rectangle of a layer without re-uploading the full texture.

        ``patch`` must have the same channel layout as ``layer.fmt`` and be
        contiguous.  Call ``update()`` afterwards to trigger a repaint.
        """
        if layer.tex_id == 0:
            return
        self.makeCurrent()
        pw = patch.shape[1]
        ph = patch.shape[0]
        buf = np.ascontiguousarray(patch)
        GL.glBindTexture(GL.GL_TEXTURE_2D, layer.tex_id)
        fmt_map = {"rgb": GL.GL_RGB, "rgba": GL.GL_RGBA, "gray": GL.GL_RED}
        gl_fmt = fmt_map.get(layer.fmt, GL.GL_RGBA)
        if layer.fmt in ("rgb", "gray"):
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, x, y, pw, ph,
                           gl_fmt, GL.GL_UNSIGNED_BYTE, buf)
        if layer.fmt in ("rgb", "gray"):
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self.doneCurrent()

    # ── Draw helpers ─────────────────────────────────────────────────────

    def _upload_quad(self, verts: np.ndarray) -> None:
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes,
                        np.ascontiguousarray(verts), GL.GL_DYNAMIC_DRAW)

    def _use_rgba_prog(self, layer: _LayerData) -> None:
        GL.glUseProgram(self._prog_rgba)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, layer.tex_id)
        GL.glUniform1i(GL.glGetUniformLocation(self._prog_rgba, b"u_tex"), 0)
        GL.glUniform1f(GL.glGetUniformLocation(self._prog_rgba, b"u_opacity"),
                       layer.opacity)

    def _use_mask_prog(self, layer: _LayerData) -> None:
        GL.glUseProgram(self._prog_mask)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, layer.tex_id)
        GL.glUniform1i(GL.glGetUniformLocation(self._prog_mask, b"u_tex"), 0)
        r, g, b = layer.color_rgb
        GL.glUniform3f(GL.glGetUniformLocation(self._prog_mask, b"u_color"), r, g, b)
        GL.glUniform1f(GL.glGetUniformLocation(self._prog_mask, b"u_opacity"),
                       layer.opacity)

    def _draw_image_quad(self) -> bool:
        """Upload quad vertices and bind VAO.  Returns False if nothing visible."""
        verts = self._vp.quad_for_image(self._img_w, self._img_h)
        if verts.shape[0] == 0:
            return False
        self._upload_quad(verts)
        GL.glBindVertexArray(self._vao)
        return True

    def _draw_rgba_layer(self, layer: _LayerData) -> None:
        if not self._draw_image_quad():
            return
        self._use_rgba_prog(layer)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindVertexArray(0)

    def _draw_mask_layer(self, layer: _LayerData) -> None:
        if not self._draw_image_quad():
            return
        self._use_mask_prog(layer)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindVertexArray(0)

    def _draw_ann_layer(self, layer: _LayerData) -> None:
        """RGBA annotation overlay with 1-screen-pixel inner border."""
        if not self._draw_image_quad():
            return
        GL.glUseProgram(self._prog_ann)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, layer.tex_id)
        GL.glUniform1i(GL.glGetUniformLocation(self._prog_ann, b"u_tex"), 0)
        GL.glUniform1f(GL.glGetUniformLocation(self._prog_ann, b"u_opacity"), self._global_ann_opacity)
        GL.glUniform1f(GL.glGetUniformLocation(self._prog_ann, b"u_zoom"), self._vp.zoom)
        if self._img_w > 0 and self._img_h > 0:
            GL.glUniform2f(GL.glGetUniformLocation(self._prog_ann, b"u_texel"),
                           1.0 / self._img_w, 1.0 / self._img_h)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindVertexArray(0)

    def _draw_sel_layer(self, layer: _LayerData) -> None:
        """Grayscale selection mask with 1-screen-pixel animated marching-ants border."""
        if not self._draw_image_quad():
            return
        GL.glUseProgram(self._prog_sel)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, layer.tex_id)
        GL.glUniform1i(GL.glGetUniformLocation(self._prog_sel, b"u_tex"), 0)
        r, g, b = layer.color_rgb
        GL.glUniform3f(GL.glGetUniformLocation(self._prog_sel, b"u_color"), r, g, b)
        GL.glUniform1f(GL.glGetUniformLocation(self._prog_sel, b"u_opacity"), layer.opacity)
        GL.glUniform1f(GL.glGetUniformLocation(self._prog_sel, b"u_time"), self._anim_time)
        GL.glUniform1f(GL.glGetUniformLocation(self._prog_sel, b"u_zoom"), self._vp.zoom)
        if self._img_w > 0 and self._img_h > 0:
            GL.glUniform2f(GL.glGetUniformLocation(self._prog_sel, b"u_texel"),
                           1.0 / self._img_w, 1.0 / self._img_h)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindVertexArray(0)

    def _draw_glow_layer(self, layer: _LayerData) -> None:
        """Pulsing white glow over unannotated pixels."""
        if not self._draw_image_quad():
            return
        GL.glUseProgram(self._prog_glow)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, layer.tex_id)
        GL.glUniform1i(GL.glGetUniformLocation(self._prog_glow, b"u_tex"), 0)
        GL.glUniform1f(GL.glGetUniformLocation(self._prog_glow, b"u_time"), self._anim_time)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindVertexArray(0)

    def _has_animated_content(self) -> bool:
        """Return True if any animated layer is visible and has data."""
        sel = self._layers.get(_L_SEL)
        miss = self._layers.get(_L_MISSING)
        return (
            (sel is not None and sel.data is not None and sel.visible) or
            (miss is not None and miss.data is not None and miss.visible)
        )

    def _tick_animation(self) -> None:
        """Advance animation clock and request a repaint."""
        if not self._has_animated_content():
            self._anim_timer.stop()
            return
        self._anim_time = _time_module.monotonic() - self._anim_start
        self.update()

    def start_animation(self) -> None:
        """Start animation timer (drives marching-ants selection and missing-pixels glow)."""
        if not self._anim_timer.isActive():
            self._anim_timer.start()

    def stop_animation(self) -> None:
        """Stop animation timer only when no animated layer needs it."""
        if not self._has_animated_content():
            self._anim_timer.stop()

    def set_global_layer_opacity(self, opacity: float) -> None:
        """Set global opacity multiplier for annotation layers."""
        self._global_ann_opacity = max(0.0, min(1.0, opacity))
        self.update()

    def _draw_grid(self) -> None:
        verts = self._vp.fullscreen_quad()
        self._upload_quad(verts)
        GL.glUseProgram(self._prog_grid)
        GL.glUniform1f(GL.glGetUniformLocation(self._prog_grid, b"u_zoom"),
                       float(self._vp.zoom))
        GL.glUniform2f(GL.glGetUniformLocation(self._prog_grid, b"u_pan"),
                       self._vp.pan_x, self._vp.pan_y)
        GL.glUniform2f(GL.glGetUniformLocation(self._prog_grid, b"u_screen"),
                       float(self._vp.screen_w), float(self._vp.screen_h))
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindVertexArray(0)

    # ── Resource cleanup ─────────────────────────────────────────────────

    def schedule_delete_texture(self, tex_id: int) -> None:
        if tex_id != 0:
            self._to_delete.append(tex_id)

    def cleanup(self) -> None:
        """Release all GL resources.  Must be called with a current context."""
        self.makeCurrent()
        for layer in self._layers.values():
            if layer.tex_id != 0:
                GL.glDeleteTextures(1, [layer.tex_id])
                layer.tex_id = 0
        self.doneCurrent()


# ---------------------------------------------------------------------------
# Public viewer
# ---------------------------------------------------------------------------

class GLImageAnnotationViewer(QWidget):
    """OpenGL-based annotation viewer implementing ``IImageAnnotationViewer``.

    Usage (drop-in swap with ``QtImageAnnotationViewer``)::

        from viewer.gl_viewer import GLImageAnnotationViewer as Viewer
        # was: from viewer.qt_viewer import QtImageAnnotationViewer as Viewer
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._vp = _Viewport()
        self._layers: dict[str, _LayerData] = {
            _L_BASE:    _LayerData(fmt="rgb",  shader="rgba",       opacity=1.0),
            _L_ANN:     _LayerData(fmt="rgba", shader="ann_border", opacity=0.5),
            _L_MISSING: _LayerData(fmt="gray", shader="glow",       opacity=1.0,
                                   color_rgb=(1.0, 1.0, 1.0), visible=False),
            _L_SEL:     _LayerData(fmt="gray", shader="sel_ants",   opacity=0.35,
                                   color_rgb=(1.0, 1.0, 1.0), visible=False),
            _L_TOOL:    _LayerData(fmt="gray", shader="mask",       opacity=0.4,
                                   color_rgb=(1.0, 1.0, 1.0), visible=False),
        }

        self._canvas = _GLCanvas(self._vp, self._layers, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        # Cursor cache: {(tool_name, (r,g,b)): QCursor}
        self._cursor_cache: dict[tuple, QCursor] = {}

        # Registered callbacks
        self._cb_mouse_press:   list[Callable] = []
        self._cb_mouse_release: list[Callable] = []
        self._cb_mouse_move:    list[Callable] = []
        self._cb_scroll:        list[Callable] = []

        # Pinch-to-zoom state
        self._pinch_accum: float = 0.0

        # Pan state (middle mouse button)
        self._pan_active: bool = False
        self._pan_last: Optional[tuple[int, int]] = None
        self._active_cursor: Optional[QCursor] = None

        # Wire canvas input events to our handlers
        self._canvas.mousePressEvent   = self._on_mouse_press
        self._canvas.mouseMoveEvent    = self._on_mouse_move
        self._canvas.mouseReleaseEvent = self._on_mouse_release
        self._canvas.wheelEvent        = self._on_wheel

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Prevent the inner GL canvas from stealing keyboard focus; all key
        # events must flow through the container so ShortcutManager sees them.
        self._canvas.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._gesture_filter = _GestureFilter(self._on_native_gesture, self)
        self._canvas.installEventFilter(self._gesture_filter)

    # ── IImageAnnotationViewer — display API ────────────────────────────

    def set_base_image(self, bgr: np.ndarray) -> None:
        self._canvas._anim_timer.stop()
        # Convert BGR → RGB for upload
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Schedule deletion of existing textures
        for layer in self._layers.values():
            self._canvas.schedule_delete_texture(layer.tex_id)
            layer.tex_id = 0
            layer.data = None
            layer.dirty = False

        h, w = bgr.shape[:2]
        self._canvas._img_w = w
        self._canvas._img_h = h

        self._layers[_L_BASE].data = rgb
        self._layers[_L_BASE].dirty = True
        self._layers[_L_BASE].visible = True

        # Reset viewport pan (keep zoom)
        self._vp.pan_x = 0.0
        self._vp.pan_y = 0.0

        self._canvas.update()

    def set_base_image_visible(self, visible: bool) -> None:
        self._layers[_L_BASE].visible = visible
        self._canvas.update()

    def set_annotation_overlay(self, rgba: np.ndarray) -> None:
        layer = self._layers[_L_ANN]
        layer.data = rgba
        layer.dirty = True
        self._canvas.update()

    def set_annotations_visible(self, visible: bool) -> None:
        self._layers[_L_ANN].visible = visible
        self._canvas.update()

    def set_selection_mask(
        self,
        mask: Optional[np.ndarray],
        color_rgb: tuple[int, int, int],
    ) -> None:
        layer = self._layers[_L_SEL]
        if mask is None:
            layer.visible = False
            self._canvas.stop_animation()
            self._canvas.update()
            return
        layer.data = np.ascontiguousarray(mask)
        layer.fmt = "gray"
        layer.color_rgb = (color_rgb[0] / 255.0,
                           color_rgb[1] / 255.0,
                           color_rgb[2] / 255.0)
        layer.dirty = True
        layer.visible = True
        self._canvas.start_animation()
        self._canvas.update()

    def set_tool_preview(
        self,
        mask: Optional[np.ndarray],
        color_rgb: tuple[int, int, int],
    ) -> None:
        layer = self._layers[_L_TOOL]
        if mask is None:
            layer.visible = False
            self._canvas.update()
            return
        layer.data = np.ascontiguousarray(mask)
        layer.fmt = "gray"
        layer.color_rgb = (color_rgb[0] / 255.0,
                           color_rgb[1] / 255.0,
                           color_rgb[2] / 255.0)
        layer.dirty = True
        layer.visible = True
        self._canvas.update()

    def set_missing_pixels_visible(
        self,
        mask: Optional[np.ndarray],
        visible: bool,
    ) -> None:
        layer = self._layers[_L_MISSING]
        if not visible or mask is None:
            layer.visible = False
            self._canvas.update()
            return
        # mask already contains unannotated pixels = 255 (from get_missing_annotations_mask)
        layer.data = np.ascontiguousarray(mask)
        layer.fmt = "gray"
        layer.dirty = True
        layer.visible = True
        self._canvas.start_animation()
        self._canvas.update()

    def set_grid_visible(self, visible: bool) -> None:
        self._canvas._grid_visible = visible
        self._canvas.update()

    def set_global_layer_opacity(self, opacity: float) -> None:
        self._canvas.set_global_layer_opacity(opacity)

    # ── IImageAnnotationViewer — zoom / viewport ─────────────────────────

    def set_zoom(
        self,
        zoom: int,
        center: Optional[tuple[float, float]] = None,
    ) -> None:
        old_zoom = self._vp.zoom
        if center is not None:
            px, py = center
            # Keep the anchor image pixel at the same screen position
            screen_x = px * old_zoom + self._vp.pan_x
            screen_y = py * old_zoom + self._vp.pan_y
            self._vp.zoom = float(zoom)
            self._vp.pan_x = screen_x - px * zoom
            self._vp.pan_y = screen_y - py * zoom
        else:
            cx, cy = self._vp.image_center()
            self._vp.zoom = float(zoom)
            self._vp.pan_x = self._vp.screen_w / 2.0 - cx * zoom
            self._vp.pan_y = self._vp.screen_h / 2.0 - cy * zoom

        self._vp.clamp_pan(self._canvas._img_w, self._canvas._img_h)
        self._canvas.update()

    def get_zoom(self) -> int:
        return int(self._vp.zoom)

    def zoom_reset(self) -> None:
        self._vp.zoom = 1.0
        self._vp.pan_x = 0.0
        self._vp.pan_y = 0.0
        self._canvas.update()

    def get_view_center(self) -> tuple[float, float]:
        return self._vp.image_center()

    # ── IImageAnnotationViewer — cursor ──────────────────────────────────

    def update_cursor(
        self,
        tool: str,
        layer_color: tuple[int, int, int],
    ) -> None:
        key = (tool, layer_color)
        cursor = self._cursor_cache.get(key)
        if cursor is None:
            cursor = _build_cursor(tool, layer_color)
            self._cursor_cache[key] = cursor
        if cursor:
            self._canvas.setCursor(cursor)
            self._active_cursor = cursor
        else:
            self._canvas.setCursor(Qt.CursorShape.CrossCursor)
            self._active_cursor = QCursor(Qt.CursorShape.CrossCursor)

    # ── IImageAnnotationViewer — widget ──────────────────────────────────

    @property
    def widget(self) -> QWidget:
        return self

    # ── IImageAnnotationViewer — input registration ───────────────────────

    def register_mouse_press(self, cb: Callable[[int, int, str], None]) -> None:
        self._cb_mouse_press.append(cb)

    def register_mouse_release(self, cb: Callable[[int, int, str], None]) -> None:
        self._cb_mouse_release.append(cb)

    def register_mouse_move(self, cb: Callable[[int, int], None]) -> None:
        self._cb_mouse_move.append(cb)

    def register_scroll(self, cb: Callable) -> None:
        self._cb_scroll.append(cb)

    # ── Qt event handlers ────────────────────────────────────────────────

    def _on_mouse_press(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_active = True
            self._pan_last = (int(event.position().x()), int(event.position().y()))
            self._canvas.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        px, py = self._vp.screen_to_image(event.pos().x(), event.pos().y())
        btn = _button_name(event.button())
        for cb in self._cb_mouse_press:
            cb(px, py, btn)

    def _on_mouse_release(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_active = False
            self._pan_last = None
            if self._active_cursor is not None:
                self._canvas.setCursor(self._active_cursor)
            else:
                self._canvas.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        px, py = self._vp.screen_to_image(event.pos().x(), event.pos().y())
        btn = _button_name(event.button())
        for cb in self._cb_mouse_release:
            cb(px, py, btn)

    def _on_mouse_move(self, event: QMouseEvent) -> None:
        if self._pan_active and self._pan_last is not None:
            x, y = int(event.position().x()), int(event.position().y())
            dx, dy = x - self._pan_last[0], y - self._pan_last[1]
            self._pan_last = (x, y)
            self._vp.pan_x += dx
            self._vp.pan_y += dy
            self._vp.clamp_pan(self._canvas._img_w, self._canvas._img_h)
            self._canvas.update()
            event.accept()
            return
        px, py = self._vp.screen_to_image(event.pos().x(), event.pos().y())
        for cb in self._cb_mouse_move:
            cb(px, py)

    def _on_wheel(self, event) -> None:
        mods = _modifiers_frozenset(event.modifiers())
        dy = event.angleDelta().y()
        dx = event.angleDelta().x()

        if "ctrl" in mods:
            # Let the controller handle zoom via the scroll callback
            scene_x, scene_y = event.position().x(), event.position().y()
            px, py = self._vp.screen_to_image(scene_x, scene_y)
            for cb in self._cb_scroll:
                cb(dy, dx, px, py, mods)
        elif "shift" in mods:
            # Horizontal pan only
            self._vp.pan_x += dy
            self._vp.clamp_pan(self._canvas._img_w, self._canvas._img_h)
            self._canvas.update()
        else:
            # Pan both axes
            self._vp.pan_y += dy
            self._vp.pan_x += dx
            self._vp.clamp_pan(self._canvas._img_w, self._canvas._img_h)
            self._canvas.update()

    def _on_native_gesture(self, event) -> None:
        """Handle trackpad pinch events."""
        try:
            gesture_type = event.gestureType()
        except AttributeError:
            return

        if gesture_type == Qt.NativeGestureType.BeginNativeGesture:
            self._pinch_accum = 0.0
            return

        if gesture_type != Qt.NativeGestureType.ZoomNativeGesture:
            return

        self._pinch_accum += event.value()
        _PINCH_STEP = 0.1
        steps = int(self._pinch_accum / _PINCH_STEP)
        if steps == 0:
            return
        self._pinch_accum -= steps * _PINCH_STEP

        pos = event.position().toPoint()
        px, py = self._vp.screen_to_image(pos.x(), pos.y())
        dy = 120 if steps > 0 else -120
        for _ in range(abs(steps)):
            for cb in self._cb_scroll:
                cb(dy, 0, px, py, frozenset({"ctrl"}))

    def closeEvent(self, event) -> None:  # noqa: N802
        self._canvas.cleanup()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Qt → semantic string helpers (module-level, stateless)
# ---------------------------------------------------------------------------

def _button_name(button) -> str:
    if button == Qt.MouseButton.LeftButton:
        return "left"
    if button == Qt.MouseButton.RightButton:
        return "right"
    if button == Qt.MouseButton.MiddleButton:
        return "middle"
    return "unknown"


def _key_name(key) -> str:
    _map = {
        Qt.Key.Key_P:      "P",
        Qt.Key.Key_S:      "S",
        Qt.Key.Key_F:      "F",
        Qt.Key.Key_Z:      "Z",
        Qt.Key.Key_E:      "E",
        Qt.Key.Key_R:      "R",
        Qt.Key.Key_I:      "I",
        Qt.Key.Key_M:      "M",
        Qt.Key.Key_G:      "G",
        Qt.Key.Key_Plus:   "Plus",
        Qt.Key.Key_Equal:  "Plus",
        Qt.Key.Key_Minus:  "Minus",
        Qt.Key.Key_Space:  "Space",
        Qt.Key.Key_Escape: "Escape",
        Qt.Key.Key_Return: "Return",
        Qt.Key.Key_Enter:  "Return",
    }
    if key in _map:
        return _map[key]
    for i in range(1, 10):
        if key == getattr(Qt.Key, f"Key_{i}"):
            return str(i)
    return ""


def _modifiers_frozenset(mods) -> frozenset:
    result = set()
    if mods & Qt.KeyboardModifier.ControlModifier:
        result.add("ctrl")
    if mods & Qt.KeyboardModifier.ShiftModifier:
        result.add("shift")
    if mods & Qt.KeyboardModifier.AltModifier:
        result.add("alt")
    return frozenset(result)


def _build_cursor(
    tool: str,
    layer_color: tuple[int, int, int],
) -> Optional[QCursor]:
    cursor_file = _TOOL_CURSOR_FILES.get(tool)
    if not cursor_file:
        return None
    cursor_path = os.path.join(_RESOURCES_DIR, cursor_file)
    if not os.path.exists(cursor_path):
        return None
    image = QImage(cursor_path)
    target = QColor(layer_color[0], layer_color[1], layer_color[2])
    for y in range(image.height()):
        for x in range(image.width()):
            if image.pixelColor(x, y) == Qt.GlobalColor.white:
                image.setPixelColor(x, y, target)
    return QCursor(QPixmap.fromImage(image), 0, 0)
