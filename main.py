"""Entry point for the Pixel-Level Annotator.

Package layout
--------------
  src/domain/          — ImageDocument, LayerConfig  (pure domain, no I/O)
  src/infrastructure/  — ImageRepository, ImageMetadata, TimeTracker, WebService
  src/application/     — AppState, AnnotatorController, AutolabelService, tools
  src/viewer/          — IImageAnnotationViewer contract + concrete backends
  src/presentation/    — ToolbarPanel, MainWindow

Runtime configuration
---------------------
Viewer backend and other tunables are read from ``config.toml`` at startup.
"""
import os
import sys
import logging

# ---------------------------------------------------------------------------
# Ensure the src/ package root is resolvable before any internal imports.
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Return the parsed contents of config.toml, or an empty dict on failure."""
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    config_path = os.path.join(_ROOT, "config.toml")
    try:
        with open(config_path, "rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        logger.warning("config.toml not found — using default settings.")
        return {}


# ---------------------------------------------------------------------------
# Viewer selection
# ---------------------------------------------------------------------------

def _setup_opengl_format() -> None:
    """Request an OpenGL 3.3 Core Profile context before the QApplication starts.

    On macOS, Qt defaults to a legacy (2.1) OpenGL context which does not
    support GLSL #version 330.  Setting the default surface format here
    ensures all subsequent QOpenGLWidget instances receive a Core Profile
    3.3 context.  This call is a no-op when the Qt (non-GL) backend is used.
    """
    from PyQt6.QtGui import QSurfaceFormat
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(0)
    fmt.setStencilBufferSize(0)
    QSurfaceFormat.setDefaultFormat(fmt)


def _resolve_viewer(backend: str):
    """Return the viewer class that corresponds to *backend*.

    Parameters
    ----------
    backend:
        ``"gl"``  — OpenGL 3.3 Core Profile (GPU-accelerated).
        ``"qt"``  — Pure Qt ``QGraphicsView`` (software-rendered).

    Raises
    ------
    ValueError
        If *backend* is not one of the supported identifiers.
    """
    if backend == "gl":
        _setup_opengl_format()
        from viewer.gl_viewer import GLImageAnnotationViewer
        return GLImageAnnotationViewer
    if backend == "qt":
        from viewer.qt_viewer import QtImageAnnotationViewer
        return QtImageAnnotationViewer
    raise ValueError(
        f"Unknown viewer backend {backend!r} in config.toml. "
        "Supported values: 'gl', 'qt'."
    )


# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------

def _ensure_layers_file() -> None:
    """Write a default layers.txt when the file is absent or empty."""
    from domain.layer_config import _DEFAULT_LAYERS, write_layers_file
    path = os.path.join(_ROOT, "layers.txt")
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        write_layers_file(list(_DEFAULT_LAYERS))


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication

    config = _load_config()
    backend = config.get("viewer", {}).get("backend", "gl")
    logger.info("Viewer backend: %s", backend)

    viewer_class = _resolve_viewer(backend)

    from presentation.main_window import MainWindow
    from presentation.style import GLOBAL_STYLESHEET
    from infrastructure.project_manager import ProjectManager

    ProjectManager.set_app_root(_ROOT)

    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_STYLESHEET)
    window = MainWindow(viewer_class=viewer_class)
    window.show()

    # Auto-open last project if available
    last_project = ProjectManager.get_last_project_path()
    if last_project:
        window.open_project(last_project)

    sys.exit(app.exec())

