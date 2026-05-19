"""Centralized design tokens and Qt stylesheet for the PixelSmart UI.

Translates the design system from design/DESIGN.md and design/code.html
into Python constants and a single QSS stylesheet string.  Every
presentation module imports colours from here — no magic strings scattered
across widgets.
"""
from __future__ import annotations

# ------------------------------------------------------------------
# Design tokens (from design/DESIGN.md colour palette)
# ------------------------------------------------------------------

# Surfaces
SURFACE             = "#0e0e0e"
SURFACE_DIM         = "#0e0e0e"
SURFACE_CONTAINER_LOWEST  = "#000000"
SURFACE_CONTAINER_LOW     = "#131313"
SURFACE_CONTAINER         = "#191a1a"
SURFACE_CONTAINER_HIGH    = "#1f2020"
SURFACE_CONTAINER_HIGHEST = "#252626"
SURFACE_VARIANT           = "#252626"
SURFACE_BRIGHT            = "#2c2c2c"

# Primary (teal / cyan)
PRIMARY             = "#a1faff"
PRIMARY_DIM         = "#00e5ee"
PRIMARY_CONTAINER   = "#00f4fe"
ON_PRIMARY          = "#006165"

# Secondary (purple)
SECONDARY           = "#fbabff"
SECONDARY_DIM       = "#ec63ff"
ON_SECONDARY        = "#710082"

# Tertiary (coral / orange)
TERTIARY            = "#ffb59c"
TERTIARY_DIM        = "#f7794b"

# Error
ERROR               = "#ff716c"
ERROR_DIM           = "#c94947"

# Text / foreground
ON_SURFACE          = "#e7e5e4"
ON_SURFACE_VARIANT  = "#acabaa"
ON_BACKGROUND       = "#e7e5e4"

# Borders
OUTLINE             = "#767575"
OUTLINE_VARIANT     = "#484848"

# ------------------------------------------------------------------
# Typography sizes
# ------------------------------------------------------------------
FONT_FAMILY   = "Inter"
FONT_SIZE_XS  = 10   # px  — label-sm helper text
FONT_SIZE_SM  = 11   # px  — labels, tool names
FONT_SIZE_MD  = 13   # px  — body text
FONT_SIZE_LG  = 15   # px  — panel headers

# ------------------------------------------------------------------
# Spacing / geometry
# ------------------------------------------------------------------
SIDEBAR_WIDTH    = 256   # px
TOPBAR_HEIGHT    = 44    # px
STATUSBAR_HEIGHT = 28    # px
BORDER_RADIUS_SM = 2     # px
BORDER_RADIUS_MD = 6     # px
BORDER_RADIUS_LG = 8     # px
BORDER_RADIUS_XL = 12    # px

# ------------------------------------------------------------------
# Layer accent colours (cycled when layer_config colours are used)
# ------------------------------------------------------------------
LAYER_ACCENTS = [
    SECONDARY,   # purple
    PRIMARY,     # teal
    TERTIARY,    # coral
    OUTLINE_VARIANT,  # grey for background-like layers
]


def _build_global_stylesheet() -> str:
    """Build the application-wide QSS stylesheet string."""
    return f"""
/* ================================================================
   Global application stylesheet — "The Chromatic Precisionist"
   ================================================================ */

* {{
    font-family: "{FONT_FAMILY}", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: {FONT_SIZE_MD}px;
    color: {ON_SURFACE};
    outline: none;
}}

QMainWindow {{
    background-color: {SURFACE};
}}

/* ------ Scroll bars ------------------------------------------- */
QScrollBar:vertical {{
    background: transparent;
    width: 5px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {OUTLINE_VARIANT};
    border-radius: 2px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    height: 0; background: transparent;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 5px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {OUTLINE_VARIANT};
    border-radius: 2px;
    min-width: 20px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    width: 0; background: transparent;
}}

/* ------ Buttons (base) ---------------------------------------- */
QPushButton {{
    background-color: {SURFACE_CONTAINER_HIGHEST};
    border: none;
    border-radius: {BORDER_RADIUS_MD}px;
    padding: 6px 12px;
    color: {ON_SURFACE_VARIANT};
    font-weight: 600;
    font-size: {FONT_SIZE_SM}px;
    letter-spacing: 0.5px;
}}
QPushButton:hover {{
    background-color: {SURFACE_VARIANT};
    color: {ON_SURFACE};
}}
QPushButton:pressed {{
    background-color: {SURFACE_BRIGHT};
}}
QPushButton:checked {{
    background-color: {SURFACE_BRIGHT};
    color: {PRIMARY};
}}
QPushButton:disabled {{
    color: {OUTLINE_VARIANT};
}}

/* ------ Check boxes ------------------------------------------- */
QCheckBox {{
    spacing: 6px;
    font-size: {FONT_SIZE_XS}px;
    color: {ON_SURFACE_VARIANT};
    font-weight: 500;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border-radius: 3px;
    background: {SURFACE_CONTAINER_LOWEST};
    border: 1px solid {OUTLINE_VARIANT};
}}
QCheckBox::indicator:checked {{
    background: {PRIMARY};
    border-color: {PRIMARY};
}}

/* ------ Sliders ----------------------------------------------- */
QSlider::groove:horizontal {{
    border: none;
    height: 4px;
    background: {SURFACE_CONTAINER_LOWEST};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {PRIMARY};
    border: none;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}}
QSlider::sub-page:horizontal {{
    background: {PRIMARY_DIM};
    border-radius: 2px;
}}

/* ------ Spin boxes -------------------------------------------- */
QSpinBox {{
    background: {SURFACE_CONTAINER_LOWEST};
    border: none;
    border-radius: {BORDER_RADIUS_SM}px;
    padding: 2px 4px;
    color: {PRIMARY};
    font-size: {FONT_SIZE_XS}px;
    font-weight: 700;
    min-width: 40px;
}}
QSpinBox::up-button, QSpinBox::down-button {{
    width: 0;
    border: none;
}}

/* ------ Combo boxes ------------------------------------------- */
QComboBox {{
    background: {SURFACE_CONTAINER_LOWEST};
    border: 1px solid rgba(72, 72, 72, 0.2);
    border-radius: {BORDER_RADIUS_LG}px;
    padding: 5px 10px;
    color: {ON_SURFACE};
    font-size: {FONT_SIZE_SM}px;
}}
QComboBox:focus {{
    border: 1px solid {PRIMARY};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background: {SURFACE_CONTAINER_HIGHEST};
    border: 1px solid rgba(72, 72, 72, 0.2);
    selection-background-color: {SURFACE_BRIGHT};
    color: {ON_SURFACE};
    outline: none;
}}

/* ------ Labels ------------------------------------------------ */
QLabel {{
    background: transparent;
    border: none;
}}

/* ------ Tool tips --------------------------------------------- */
QToolTip {{
    background-color: {SURFACE_CONTAINER_HIGHEST};
    color: {ON_SURFACE};
    border: 1px solid rgba(72, 72, 72, 0.3);
    padding: 4px 8px;
    border-radius: {BORDER_RADIUS_MD}px;
    font-size: {FONT_SIZE_SM}px;
}}
"""


GLOBAL_STYLESHEET = _build_global_stylesheet()
