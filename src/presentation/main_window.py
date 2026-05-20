"""Main application window — redesigned layout.

Layout (top-to-bottom, left-to-right):
    ┌──────────────────────────────────────────────────────┐
    │                   Top Navigation Bar                  │
    ├─────────┬──────────┬──────────────────┬──────────────┤
    │  Left   │ Gallery  │                  │    Right     │
    │ Toolbar │ (toggle) │     Canvas       │    Panel     │
    │         │          │                  │              │
    ├─────────┴──────────┴──────────────────┴──────────────┤
    │                     Status Bar                        │
    └──────────────────────────────────────────────────────┘

This module is intentionally thin.  Its only responsibilities are:
- Construct the widget tree.
- Create infrastructure and controller objects.
- Wire callbacks between panels and controller.
- React to controller output callbacks.
"""
from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from domain.layer_config import LayerConfig, read_layers_file
from domain.project import ProjectConfig
from infrastructure.image_repository import ImageRepository
from infrastructure.project_manager import ProjectManager
from application.annotator_controller import AnnotatorController
from application.app_state import PluginConfig
from presentation.toolbar_panel import ToolbarPanel
from presentation.right_panel import RightPanel
from presentation.model_config_dialog import LayerMappingDialog
from presentation.gallery_panel import GalleryPanel
from presentation.stats_panel import StatsPanel
from presentation.status_bar import StatusBar
from presentation.welcome_screen import WelcomeScreen
from presentation.shortcut_manager import ShortcutManager
from presentation.help_window import HelpWindow
from presentation.style import (
    GLOBAL_STYLESHEET,
    PRIMARY,
    ON_SURFACE,
    ON_SURFACE_VARIANT,
    OUTLINE_VARIANT,
    SURFACE,
    SURFACE_CONTAINER_HIGH,
    SURFACE_CONTAINER_HIGHEST,
    SURFACE_CONTAINER_LOW,
    SURFACE_BRIGHT,
    FONT_SIZE_SM,
    FONT_SIZE_XS,
    TOPBAR_HEIGHT,
)
from infrastructure.webservice import WebService

logger = logging.getLogger(__name__)




class _AutolabelWorker(QThread):
    """Runs the autolabel plugin on a background thread."""
    finished = pyqtSignal(object)  # emits error string or None

    def __init__(self, controller, plugin_id: str, parent=None):
        super().__init__(parent)
        self._controller = controller
        self._plugin_id = plugin_id

    def run(self):
        error = self._controller.run_autolabel(self._plugin_id)
        self.finished.emit(error)


class _FineTuneWorker(QThread):
    """Ejecuta fine_tune() del plugin en un hilo de fondo."""
    finished = pyqtSignal(object)  # emite None en éxito o str con mensaje de error

    def __init__(self, controller, plugin_id: str, images_and_annotations: list, parent=None):
        super().__init__(parent)
        self._controller = controller
        self._plugin_id = plugin_id
        self._images_and_annotations = images_and_annotations

    def run(self):
        try:
            self._controller.autolabel_service.run_fine_tune(
                self._plugin_id, self._images_and_annotations
            )
            self.finished.emit(None)
        except Exception as e:
            self.finished.emit(str(e))


class MainWindow(QMainWindow):
    """Top-level Qt window for the Pixel Annotation Tool."""

    def __init__(self, viewer_class=None) -> None:
        super().__init__()
        self.setWindowTitle("PixelSmart — Professional Annotation Suite")
        self.setGeometry(100, 100, 1440, 900)

        # Apply global stylesheet
        self.setStyleSheet(GLOBAL_STYLESHEET)

        self._viewer_class = viewer_class
        self._project_folder: Optional[str] = None
        self._project_config: Optional[ProjectConfig] = None
        self._controller: Optional[AnnotatorController] = None
        self._current_web_request = None
        self._save_timer: Optional[QTimer] = None
        self._help_window: Optional[HelpWindow] = None
        self._fine_tune_worker: Optional[_FineTuneWorker] = None
        self._fine_tune_plugin_display_name: str = ""

        # Stacked widget: 0 = welcome screen, 1 = annotator
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # Welcome screen
        self._welcome = WelcomeScreen()
        self._welcome.on_open_folder(self._ask_open_project)
        self._stack.addWidget(self._welcome)

        # Placeholder for annotator widget (built on project open)
        self._annotator_widget: Optional[QWidget] = None

        # Install a global event filter so keyboard shortcuts reach the
        # controller even when a sidebar widget temporarily holds focus.
        app = QApplication.instance()
        if app is not None:
            self._shortcut_filter = ShortcutManager(self)
            app.installEventFilter(self._shortcut_filter)
            app.focusChanged.connect(self._on_focus_changed)

    # ------------------------------------------------------------------
    # Project lifecycle
    # ------------------------------------------------------------------

    def open_project(self, folder: str) -> None:
        """Open *folder* as the current project."""
        import os

        # Save previous project before switching
        self._save_project_config()
        if self._controller:
            self._controller.close()

        self._project_folder = folder
        self._project_config = ProjectManager.load_project(folder)
        ProjectManager.set_last_project_path(folder)

        # Change working directory to the project folder so relative paths
        # (images/, annotations/) resolve correctly.
        os.chdir(folder)

        # Ensure layers.txt exists in the project folder
        self._ensure_layers_file(folder)

        layer_configs: list[LayerConfig] = read_layers_file(
            os.path.join(folder, "layers.txt")
        )

        # Infrastructure — images can live in {folder}/images/ or directly in {folder}
        candidate_images_dir = os.path.join(folder, "images")
        images_dir = candidate_images_dir if os.path.isdir(candidate_images_dir) else folder
        annotations_dir = os.path.join(folder, "annotations")
        image_repo = ImageRepository(images_dir, annotations_dir)

        # Migrar archivos de anotación del esquema antiguo (índice) al nuevo (nombre de capa)
        layer_names_for_migration = [lc.name for lc in layer_configs]
        for img_filename in image_repo.list_images():
            image_repo.migrate_index_to_name(img_filename, layer_names_for_migration)

        # Ensure project subdirectories exist
        os.makedirs(os.path.join(annotations_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(annotations_dir, "metadata"), exist_ok=True)

        # Viewer
        viewer_class = self._viewer_class
        if viewer_class is None:
            from viewer.gl_viewer import GLImageAnnotationViewer
            viewer_class = GLImageAnnotationViewer
        viewer = viewer_class()

        # Application controller
        controller = AnnotatorController(viewer, layer_configs, image_repo)
        controller.on_progress_changed(self._on_progress_changed)
        controller.on_web_service_image_ready(self._on_web_service_image_ready)
        controller.on_status_changed(self._on_status_changed)
        self._controller = controller
        self._shortcut_filter.set_controller(controller)
        self._shortcut_filter.set_help_callback(self._show_help)
        self._image_repo = image_repo
        self._layer_configs = layer_configs

        # Apply saved project state to controller
        self._apply_project_config(controller)

        # Build (or rebuild) the annotator UI
        self._build_annotator_ui(layer_configs, viewer)

        # Wire callbacks
        self._wire_toolbar()
        self._wire_right_panel()
        self._wire_gallery()

        # Restore stats panel visibility from saved project config
        if self._project_config and self._project_config.show_stats_panel:
            self._stats_panel.setVisible(True)
            self._toolbar.set_stats_panel_open(True)

        # Set project name in the right panel (always the folder name)
        self._right_panel.set_project_name(self._project_config.name)

        # Subscribe to state changes for auto-save
        self._setup_auto_save(controller)

        # Web service
        self._web_service = WebService(self)

        # Populate gallery and load first image
        self._populate_gallery()

        # Resolve target image BEFORE refresh_autolabel_plugins, which can
        # trigger _cb_autolabel_plugin_changed → _save_project_config() and
        # overwrite last_image with None (current_filename not set yet).
        filenames = image_repo.list_images()
        target = self._project_config.last_image

        plugins = controller.autolabel_service.get_compatible_plugins()
        saved_plugin = self._project_config.selected_plugin_id if self._project_config else None
        self._toolbar.refresh_autolabel_plugins(plugins, initial_plugin_id=saved_plugin)

        # Load last image or first available
        if target and target not in filenames:
            target = None
        if not target and filenames:
            target = filenames[0]
        if target:
            controller.load_image(target)
            self._gallery.set_current_filename(target)
            self._update_status()
            self._save_project_config()

        # Show annotator
        self._stack.setCurrentIndex(1)
        self.setWindowTitle(f"PixelSmart — {self._project_config.name}")
        QTimer.singleShot(0, viewer.setFocus)

    def _show_help(self) -> None:
        if self._help_window and self._help_window.isVisible():
            self._help_window.raise_()
            return
        self._help_window = HelpWindow.show_help(
            ShortcutManager.SHORTCUTS,
            parent=self,
        )

    def _ensure_layers_file(self, folder: str) -> None:
        """Write a default layers.txt in *folder* when absent or empty."""
        import os
        from domain.layer_config import _DEFAULT_LAYERS, write_layers_file
        path = os.path.join(folder, "layers.txt")
        if not os.path.exists(path) or os.stat(path).st_size == 0:
            write_layers_file(list(_DEFAULT_LAYERS), path)

    def _apply_project_config(self, controller: AnnotatorController) -> None:
        """Restore saved project state into the controller's AppState."""
        cfg = self._project_config
        if cfg is None:
            return
        s = controller.state
        s.tool.pen_size = cfg.pen_size
        s.tool.eraser_size = cfg.eraser_size
        s.tool.selector_threshold = cfg.selector_threshold
        s.tool.selector_auto_smooth = cfg.selector_auto_smooth
        s.tool.fill_all = cfg.fill_all
        s.view.show_image = cfg.show_image
        s.view.show_other_layers = cfg.show_other_layers
        s.view.show_missing_pixels = cfg.show_missing_pixels
        s.view.show_grid = cfg.show_grid
        s.tool.active = cfg.active_tool
        s.session.active_layer = cfg.active_layer
        s.session.locked_layers = set(cfg.locked_layers)
        s.session.hidden_layers = set(cfg.hidden_layers)
        # Restore global layer opacity
        controller.set_global_layer_opacity(cfg.global_layer_opacity)
        # Restore plugin configs
        for pid, pcfg in cfg.plugin_configs.items():
            s.plugin_configs[pid] = PluginConfig(
                layer_mapping=pcfg.get("layer_mapping", {}),
                conflict_strategy=pcfg.get("conflict_strategy", "argmax"),
                layer_priorities=pcfg.get("layer_priorities", {}),
                selected_version=pcfg.get("selected_version", "v1"),
            )

    def _build_annotator_ui(self, layer_configs, viewer) -> None:
        """Build the full annotator layout and add it to the stack."""
        # Remove old annotator widget if present
        if self._annotator_widget is not None:
            self._stack.removeWidget(self._annotator_widget)
            self._annotator_widget.deleteLater()

        self._viewer = viewer
        self._toolbar = ToolbarPanel(layer_configs)
        self._right_panel = RightPanel(layer_configs)
        self._gallery = GalleryPanel()
        self._gallery.setVisible(False)
        self._stats_panel = StatsPanel()
        self._stats_panel.setVisible(False)
        self._status_bar = StatusBar()
        self._progress_value = 0

        main_widget = QWidget()
        root = QVBoxLayout(main_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Top bar
        root.addWidget(self._build_top_bar())

        # Middle area
        middle = QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(0)

        middle.addWidget(self._toolbar, 0)

        canvas_wrapper = QWidget()
        canvas_wrapper.setStyleSheet(f"background-color: {SURFACE_CONTAINER_LOW};")
        canvas_layout = QVBoxLayout(canvas_wrapper)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(viewer, 1)
        middle.addWidget(self._stats_panel, 0)
        middle.addWidget(canvas_wrapper, 1)
        self._canvas_wrapper = canvas_wrapper

        middle.addWidget(self._gallery, 0)
        middle.addWidget(self._right_panel, 0)

        root.addLayout(middle, 1)

        # Status bar
        root.addWidget(self._status_bar)

        self._stats_panel.close_requested.connect(self._on_stats_panel_close)
        self._stats_panel.refresh_requested.connect(self._refresh_stats)

        self._annotator_widget = main_widget
        self._stack.addWidget(main_widget)

    def _setup_auto_save(self, controller: AnnotatorController) -> None:
        """Subscribe to state changes and debounce project config saves."""
        # Use a single-shot timer to debounce rapid-fire changes.
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)  # ms
        self._save_timer.timeout.connect(self._save_project_config)

        def _schedule_save():
            if self._save_timer is not None:
                self._save_timer.start()

        controller.state.subscribe("tool", _schedule_save)
        controller.state.subscribe("session", _schedule_save)
        controller.state.subscribe("view", _schedule_save)

        def _on_annotations_changed(_progress: int = 0):
            if self._controller and self._controller.document:
                self._gallery.schedule_thumbnail_update(
                    self._controller.current_filename,
                    self._controller.document,
                    self._layer_configs,
                )

        controller.on_progress_changed(_on_annotations_changed)
        controller.on_progress_changed(self._refresh_stats)

    def _save_project_config(self) -> None:
        """Snapshot current state into ProjectConfig and save to disk."""
        if self._project_folder is None or self._project_config is None:
            return
        if self._controller is None:
            return
        cfg = self._project_config
        s = self._controller.state
        cfg.pen_size = s.tool.pen_size
        cfg.eraser_size = s.tool.eraser_size
        cfg.selector_threshold = s.tool.selector_threshold
        cfg.selector_auto_smooth = s.tool.selector_auto_smooth
        cfg.fill_all = s.tool.fill_all
        cfg.show_image = s.view.show_image
        cfg.show_other_layers = s.view.show_other_layers
        cfg.show_missing_pixels = s.view.show_missing_pixels
        cfg.show_grid = s.view.show_grid
        cfg.active_layer = s.session.active_layer
        cfg.locked_layers = list(s.session.locked_layers)
        cfg.hidden_layers = list(s.session.hidden_layers)
        cfg.global_layer_opacity = s.view.global_layer_opacity
        cfg.last_image = self._controller.current_filename
        cfg.active_tool = self._controller.state.tool.active
        cfg.selected_plugin_id = self._toolbar.get_selected_plugin_id()
        cfg.show_stats_panel = self._stats_panel.isVisible()
        # Serialize plugin configs
        cfg.plugin_configs = {}
        for pid, pc in s.plugin_configs.items():
            entry: dict = {
                "layer_mapping": pc.layer_mapping,
                "conflict_strategy": pc.conflict_strategy,
                "selected_version": pc.selected_version,
            }
            if pc.conflict_strategy != "argmax":
                entry["layer_priorities"] = pc.layer_priorities
            cfg.plugin_configs[pid] = entry
        ProjectManager.save_project(self._project_folder, cfg)

    def _ask_open_project(self) -> None:
        """Show a folder picker and open the selected folder as a project."""
        folder = QFileDialog.getExistingDirectory(self, "Open Project Folder")
        if folder:
            self.open_project(folder)

    # ------------------------------------------------------------------
    # Web service integration
    # ------------------------------------------------------------------

    def load_web_service_image(self, image_path: str, request) -> None:
        self._current_web_request = request
        self._controller.load_web_service_image(image_path, request)

    # ------------------------------------------------------------------
    # Qt lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self._fine_tune_worker is not None and self._fine_tune_worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Fine-tuning in progress",
                "A fine-tuning process is currently running. "
                "If you close the application, the results will be lost.\n\n"
                "Do you want to close anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self._fine_tune_worker.terminate()
            self._fine_tune_worker.wait(2000)
        self._save_project_config()
        if self._controller:
            self._controller.close()
        event.accept()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_top_bar(self) -> QWidget:
        """Build the top navigation bar."""
        bar = QWidget()
        bar.setFixedHeight(TOPBAR_HEIGHT)
        bar.setStyleSheet(f"background-color: {SURFACE};")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(0)

        # Left: Brand + Progress
        left = QHBoxLayout()
        left.setSpacing(16)

        brand = QLabel("PixelSmart")
        brand.setStyleSheet(
            f"color: {PRIMARY}; font-size: 16px; font-weight: 700; "
            f"letter-spacing: -0.5px;"
        )
        left.addWidget(brand)

        # Progress bar
        progress_container = QWidget()
        progress_container.setFixedWidth(180)
        p_layout = QVBoxLayout(progress_container)
        p_layout.setContentsMargins(0, 8, 0, 8)
        p_layout.setSpacing(2)

        p_header = QHBoxLayout()
        p_label = QLabel("ANNOTATED")
        p_label.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; font-size: {FONT_SIZE_XS}px; "
            f"font-weight: 600; letter-spacing: 1px;"
        )
        self._q_progress_pct = QLabel("0%")
        self._q_progress_pct.setStyleSheet(
            f"color: {PRIMARY}; font-size: {FONT_SIZE_XS}px; font-weight: 700;"
        )
        p_header.addWidget(p_label)
        p_header.addStretch()
        p_header.addWidget(self._q_progress_pct)
        p_layout.addLayout(p_header)

        # Progress bar track
        self._q_progress_track = QWidget()
        self._q_progress_track.setFixedHeight(6)
        self._q_progress_track.setStyleSheet(
            f"background-color: {SURFACE_CONTAINER_HIGHEST}; border-radius: 3px;"
        )

        self._q_progress_fill = QWidget(self._q_progress_track)
        self._q_progress_fill.setFixedHeight(6)
        self._q_progress_fill.setStyleSheet(
            f"background-color: {PRIMARY}; border-radius: 3px;"
        )
        self._q_progress_fill.setGeometry(0, 0, 0, 6)

        p_layout.addWidget(self._q_progress_track)
        left.addWidget(progress_container)

        layout.addLayout(left)
        layout.addStretch()

        # Right: Undo, Redo, Zoom controls
        right = QHBoxLayout()
        right.setSpacing(2)

        self._q_undo_btn = self._topbar_button("↶", "Undo (Ctrl+Z)")
        right.addWidget(self._q_undo_btn)

        self._q_redo_btn = self._topbar_button("↷", "Redo (Ctrl+Y)")
        self._q_redo_btn.setEnabled(False)
        right.addWidget(self._q_redo_btn)

        sep = QLabel()
        sep.setFixedSize(1, 16)
        sep.setStyleSheet(f"background-color: rgba(72, 72, 72, 0.2);")
        right.addWidget(sep)

        self._q_zoom_label = QLabel("100%")
        self._q_zoom_label.setStyleSheet(
            f"color: {PRIMARY}; font-size: {FONT_SIZE_SM}px; "
            f"font-weight: 700; padding: 0 8px; letter-spacing: -0.5px;"
        )
        self._q_zoom_label.setMinimumWidth(48)
        self._q_zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self._q_zoom_label)

        self._q_zoom_in_btn = self._topbar_button("+", "Zoom In")
        self._q_zoom_in_btn.setShortcut("Ctrl++")
        right.addWidget(self._q_zoom_in_btn)

        self._q_zoom_out_btn = self._topbar_button("−", "Zoom Out")
        self._q_zoom_out_btn.setShortcut("Ctrl+-")
        right.addWidget(self._q_zoom_out_btn)

        sep2 = QLabel()
        sep2.setFixedSize(1, 16)
        sep2.setStyleSheet("background-color: rgba(72, 72, 72, 0.2);")
        right.addWidget(sep2)

        self._q_help_btn = self._topbar_button("?", "Help (H)")
        self._q_help_btn.clicked.connect(self._show_help)
        right.addWidget(self._q_help_btn)

        layout.addLayout(right)
        return bar

    @staticmethod
    def _topbar_button(text: str, tooltip: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setFixedSize(32, 32)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; border: none;
                color: {ON_SURFACE_VARIANT}; font-size: 16px;
                border-radius: 4px;
            }}
            QPushButton:hover {{ background: {SURFACE_CONTAINER_HIGHEST}; color: {ON_SURFACE}; }}
            QPushButton:pressed {{ background: {SURFACE_BRIGHT}; }}
        """)
        return btn

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def _wire_toolbar(self) -> None:
        ctrl = self._controller
        tb = self._toolbar

        tb.on_tool_selected(ctrl.select_tool)
        tb.on_pen_size_changed(ctrl.set_pen_size)
        tb.on_eraser_size_changed(ctrl.set_eraser_size)
        tb.on_threshold_changed(ctrl.set_selector_threshold)
        tb.on_auto_smooth_changed(ctrl.set_selector_auto_smooth)
        tb.on_fill_all_changed(ctrl.set_fill_all)
        tb.on_erase_all_clicked(self._on_erase_all)
        tb.on_web_service_mode_changed(self._toggle_web_service_mode)
        tb.on_submit_annotations(self._cb_submit_annotations)
        tb.on_cancel_annotations(self._cb_cancel_annotations)
        tb.on_autolabel_run(self._cb_run_autolabel)
        tb.on_autolabel_configure(self._cb_configure_autolabel)
        tb.on_autolabel_plugin_changed(self._cb_autolabel_plugin_changed)
        tb.on_stats_toggled(self._on_stats_toggled)

        # Top bar buttons
        self._q_undo_btn.clicked.connect(ctrl.undo)
        self._q_redo_btn.clicked.connect(ctrl.redo)
        self._q_zoom_in_btn.clicked.connect(lambda: ctrl.zoom_in())
        self._q_zoom_out_btn.clicked.connect(lambda: ctrl.zoom_out())

        ctrl.on_toolbar_state_changed(self._sync_toolbar)

    def _wire_right_panel(self) -> None:
        ctrl = self._controller
        rp = self._right_panel

        rp.on_layer_selected(ctrl.set_active_layer)
        rp.on_layer_visibility_changed(ctrl.toggle_layer_visibility)
        rp.on_toggle_all_visibility(ctrl.toggle_all_visibility)
        rp.on_toggle_all_lock(ctrl.toggle_all_lock)
        rp.on_show_image_changed(ctrl.toggle_show_image)
        rp.on_show_missing_pixels_changed(ctrl.toggle_show_missing_pixels)
        rp.on_show_grid_changed(ctrl.toggle_show_grid)
        rp.on_layer_lock_toggled(ctrl.toggle_layer_lock)
        rp.on_open_project(self._ask_open_project)
        rp.on_gallery_clicked(self._toggle_gallery)
        rp.on_opacity_changed(ctrl.set_global_layer_opacity)

    def _wire_gallery(self) -> None:
        self._gallery.on_image_selected(self._on_gallery_image_selected)

    # ------------------------------------------------------------------
    # Sync & updates
    # ------------------------------------------------------------------

    def _sync_toolbar(self) -> None:
        state = self._controller.toolbar_state
        self._toolbar.sync(state)
        self._right_panel.sync(state)
        self._q_zoom_label.setText(f"{self._controller.get_zoom_percent()}%")
        self._q_redo_btn.setEnabled(self._controller.can_redo)

    def _on_progress_changed(self, progress: int) -> None:
        self._progress_value = progress
        self._q_progress_pct.setText(f"{progress}%")

        # Update progress fill bar width
        track_w = self._q_progress_track.width()
        fill_w = int(track_w * progress / 100) if track_w > 0 else 0
        self._q_progress_fill.setGeometry(0, 0, fill_w, 6)

        if self._current_web_request:
            self._toolbar.set_web_service_ui(
                mode_active=True,
                progress_complete=(progress == 100),
            )

    def _on_web_service_image_ready(self, filename: str, request) -> None:
        self._current_web_request = request
        self._toolbar.set_web_service_ui(mode_active=True, progress_complete=False)

    def _on_status_changed(self) -> None:
        self._update_status()

    def _update_status(self) -> None:
        ctrl = self._controller
        self._status_bar.set_filename(ctrl.current_filename)
        w, h = ctrl.image_dimensions
        if w > 0 and h > 0:
            self._status_bar.set_dimensions(w, h)
        pos = ctrl.mouse_position
        if pos:
            self._status_bar.set_coordinates(pos[0], pos[1])
        else:
            self._status_bar.set_coordinates(None, None)
        self._status_bar.set_active_tool(ctrl.active_tool_name)
        has_image = bool(ctrl.current_filename)
        self._toolbar.set_image_loaded(has_image)
        self._right_panel.set_image_loaded(has_image)

    # ------------------------------------------------------------------
    # Gallery
    # ------------------------------------------------------------------

    def _on_erase_all(self) -> None:
        """Ask for confirmation, then erase all annotations on the current image."""
        if not self._controller.current_filename:
            return
        answer = QMessageBox.question(
            self,
            "Erase All Annotations",
            "This will permanently erase <b>all annotations</b> on the current image.<br>"
            "The edit log and all statistics (pixel counts, time) will also be "
            "<b>permanently cleared</b> and cannot be recovered.<br><br>"
            "The annotation pixels can be recovered with Ctrl+Z, "
            "but statistics and edit history cannot.<br><br>"
            "Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._controller.erase_all()

    def _toggle_gallery(self) -> None:
        visible = not self._gallery.isVisible()
        self._gallery.setVisible(visible)
        if visible:
            # populate() is a no-op when the file list hasn't changed, so it's
            # safe to call every time — it only does real work on the first open
            # or when a new image is added to the repo.
            self._populate_gallery()

    def _on_stats_toggled(self, open: bool) -> None:
        # showEvent on the panel fires refresh_requested → _refresh_stats automatically
        self._stats_panel.setVisible(open)
        self._toolbar.set_stats_panel_open(open)

    def _on_stats_panel_close(self) -> None:
        self._on_stats_toggled(False)

    def _refresh_stats(self, _progress: int = 0) -> None:
        if self._stats_panel.isVisible() and self._controller:
            self._stats_panel.update_stats(self._controller.get_display_stats())
            ai_metrics = self._controller.autolabel_service.last_ai_metrics
            doc = self._controller.document
            hcr = None
            if doc is not None:
                hcr = self._controller.autolabel_service.get_hcr(doc.annotations)
            if ai_metrics is not None or hcr is not None:
                combined = dict(ai_metrics) if ai_metrics is not None else {}
                if hcr is not None:
                    combined["Correction Rate (HCR)"] = f"{hcr:.2f} %"
                self._stats_panel.update_ai_metrics(combined)
            else:
                self._stats_panel.clear_ai_metrics()

    def _populate_gallery(self) -> None:
        filenames = self._image_repo.list_images()
        self._gallery.populate(
            filenames,
            self._image_repo._images_dir,
            self._image_repo._annotations_dir,
            self._layer_configs,
        )
        # Always refresh the current-image highlight (cheap operation).
        if self._controller.current_filename:
            self._gallery.set_current_filename(self._controller.current_filename)

    def _on_gallery_image_selected(self, filename: str) -> None:
        self._stats_panel.reset()
        self._controller.autolabel_service.last_ai_metrics = None
        self._controller.autolabel_service.last_confidence_map = None
        self._controller.load_image(filename)
        self._gallery.set_current_filename(filename)
        self._update_status()

    # ------------------------------------------------------------------
    # Toolbar action handlers
    # ------------------------------------------------------------------

    def _toggle_web_service_mode(self, enabled: bool) -> None:
        if enabled:
            self._web_service.start_server()
            self._gallery.setVisible(False)
        else:
            self._web_service.stop_server()
            self._toolbar.set_web_service_ui(mode_active=False)
            self._current_web_request = None

    def _cb_run_autolabel(self) -> None:
        plugin_id = self._toolbar.get_selected_plugin_id()
        if not plugin_id:
            return

        # Warn if any model layers are not assigned to a workspace layer.
        plugin_config = self._controller.state.plugin_configs.get(plugin_id)
        if plugin_config is not None:
            unassigned = [
                layer
                for layer, target in plugin_config.layer_mapping.items()
                if target is None
            ]
        else:
            plugin = self._controller.autolabel_service.get_plugin_by_id(plugin_id)
            app_layer_names_lower = {
                lc.name.lower() for lc in self._controller.layer_configs
            }
            unassigned = [
                pl for pl in (plugin.supported_layers if plugin else [])
                if pl.lower() not in app_layer_names_lower
            ]
        if unassigned:
            names = ", ".join(unassigned)
            answer = QMessageBox.warning(
                self,
                "Unassigned Model Layers",
                f"The following model layer(s) are not assigned to any workspace "
                f"layer and will be ignored:\n\n  {names}\n\nProceed anyway?",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Ok:
                return

        self._show_busy_overlay("Running AI model…")

        self._autolabel_worker = _AutolabelWorker(self._controller, plugin_id, self)
        self._autolabel_worker.finished.connect(self._on_autolabel_finished)
        self._autolabel_worker.start()

    def _cb_configure_autolabel(self, plugin_id: str) -> None:
        """Open the model configuration dialog for the selected plugin."""
        plugin = self._controller.autolabel_service.get_plugin_by_id(plugin_id)
        if plugin is None:
            return
        app_layer_names = [lc.name for lc in self._controller.layer_configs]
        current_config = self._controller.state.plugin_configs.get(plugin_id)
        current_mapping = current_config.layer_mapping if current_config else {}
        current_strategy = current_config.conflict_strategy if current_config else "argmax"
        current_priorities = current_config.layer_priorities if current_config else {}
        current_version = current_config.selected_version if current_config else "v1"

        can_ft, versions = self._controller.autolabel_service.get_plugin_fine_tune_info(plugin_id)
        is_training = (
            self._fine_tune_worker is not None and self._fine_tune_worker.isRunning()
        )
        training_name = (
            getattr(self, "_fine_tune_plugin_display_name", "") if is_training else ""
        )

        dialog_ref: list = [None]
        dialog = LayerMappingDialog(
            plugin.display_name,
            plugin.supported_layers,
            app_layer_names,
            current_mapping,
            current_strategy,
            current_priorities,
            can_fine_tune=can_ft,
            available_versions=versions,
            current_version=current_version,
            is_training=is_training,
            training_plugin_name=training_name,
            on_fine_tune=lambda: self._start_fine_tune(plugin_id, dialog_ref[0]),
            parent=self,
        )
        dialog_ref[0] = dialog

        if self._fine_tune_worker is not None:
            self._fine_tune_worker.finished.connect(
                lambda _err: dialog.set_training_state(False) if dialog.isVisible() else None
            )

        if dialog.exec() == QDialog.DialogCode.Accepted:
            cfg = dialog.get_config()
            new_pc = PluginConfig(
                layer_mapping=cfg["layer_mapping"],
                conflict_strategy=cfg["conflict_strategy"],
                layer_priorities=cfg["layer_priorities"],
                selected_version=dialog.selected_version if can_ft else current_version,
            )
            self._controller.state.plugin_configs[plugin_id] = new_pc
            if can_ft:
                self._controller.autolabel_service.set_active_version(
                    plugin_id, new_pc.selected_version
                )
            self._toolbar.update_mapping_indicator(has_mapping=True)
            self._save_project_config()

    def _cb_autolabel_plugin_changed(self, plugin_id: Optional[str]) -> None:
        """Refresh the mapping indicator when the selected plugin changes.

        When a plugin with no saved configuration is selected, an initial
        layer mapping is created automatically by matching plugin layer
        names to workspace layer names (case-insensitive).
        """
        if plugin_id and plugin_id not in self._controller.state.plugin_configs:
            plugin = self._controller.autolabel_service.get_plugin_by_id(plugin_id)
            if plugin is not None:
                app_layer_names = [lc.name for lc in self._controller.layer_configs]
                app_layers_lower = {n.lower(): n for n in app_layer_names}
                auto_mapping = {
                    pl: app_layers_lower.get(pl.lower())
                    for pl in plugin.supported_layers
                }
                if any(v is not None for v in auto_mapping.values()):
                    self._controller.state.plugin_configs[plugin_id] = PluginConfig(
                        layer_mapping=auto_mapping,
                    )

        if self._project_folder is None:
            return
        has_mapping = bool(
            plugin_id and plugin_id in self._controller.state.plugin_configs
        )
        self._toolbar.update_mapping_indicator(has_mapping)
        self._save_timer.start()

    def _on_autolabel_finished(self, error) -> None:
        self._hide_busy_overlay()
        if error:
            QMessageBox.warning(self, "Autolabel Error", f"Plugin failed:\n{error}")
        else:
            self._controller.finalize_autolabel()
            plugins = self._controller.autolabel_service.get_compatible_plugins()
            self._toolbar.refresh_autolabel_plugins(plugins)
        # FEATURE-018: return keyboard focus to the canvas after the model finishes.
        if hasattr(self, "_viewer"):
            QTimer.singleShot(0, self._viewer.setFocus)

    def _start_fine_tune(self, plugin_id: str, config_dialog=None) -> None:
        """Abre el diálogo de selección de imágenes y lanza el worker de fine-tuning."""
        import numpy as np
        from presentation.fine_tune_selection_dialog import FineTuneSelectionDialog

        if self._controller is None:
            return
        plugin = self._controller.autolabel_service.get_plugin_by_id(plugin_id)
        if plugin is None:
            return

        # Abrir diálogo de selección de imágenes (solo muestra las anotadas al 100%)
        sel_dialog = FineTuneSelectionDialog(
            self._image_repo.images_dir,
            self._image_repo.annotations_dir,
            self._layer_configs,
            parent=config_dialog or self,
        )
        if sel_dialog.exec() != QDialog.DialogCode.Accepted:
            return

        selected_filenames = sel_dialog.get_selected_filenames()
        if not selected_filenames:
            return

        # Construir images_and_annotations en orden de capas del plugin
        plugin_config = self._controller.state.plugin_configs.get(plugin_id)
        layer_mapping = plugin_config.layer_mapping if plugin_config else {}
        app_layer_names = [lc.name for lc in self._layer_configs]
        app_layer_index = {name: i for i, name in enumerate(app_layer_names)}
        plugin_layers = plugin.supported_layers
        nlayers = len(app_layer_names)

        images_and_annotations: list[dict] = []
        for fname in selected_filenames:
            try:
                doc = self._image_repo.load(fname, nlayers, app_layer_names)
            except Exception:
                logger.exception("fine_tune: no se pudo cargar %s", fname)
                continue
            anns_for_plugin = []
            for pl_name in plugin_layers:
                app_name = layer_mapping.get(pl_name, pl_name)
                app_idx = app_layer_index.get(app_name, -1)
                if 0 <= app_idx < len(doc.annotations):
                    anns_for_plugin.append(doc.annotations[app_idx].copy())
                else:
                    anns_for_plugin.append(
                        np.zeros(doc.image.shape[:2], dtype=np.uint8)
                    )
            images_and_annotations.append(
                {"image": doc.image, "annotations": anns_for_plugin}
            )

        if not images_and_annotations:
            return

        self._fine_tune_plugin_display_name = plugin.display_name

        # Informar al usuario que el entrenamiento se realiza en segundo plano
        QMessageBox.information(
            self,
            "Fine-tuning started",
            f"Fine-tuning of {plugin.display_name} has started in the background.\n"
            "You will be notified when it completes.",
        )

        # Cerrar el diálogo de configuración del modelo
        if config_dialog is not None:
            config_dialog.accept()

        # Lanzar worker
        self._fine_tune_worker = _FineTuneWorker(
            self._controller, plugin_id, images_and_annotations, self
        )
        self._fine_tune_worker.finished.connect(self._on_fine_tune_finished)
        self._fine_tune_worker.start()

    def _on_fine_tune_finished(self, error: object) -> None:
        display_name = self._fine_tune_plugin_display_name
        self._fine_tune_worker = None
        if error is None:
            QMessageBox.information(
                self,
                "Fine-tuning complete",
                f"Fine-tuning of {display_name} completed successfully.",
            )
        else:
            QMessageBox.warning(
                self,
                "Fine-tuning Error",
                f"Fine-tuning of {display_name} failed:\n{error}",
            )

    def _show_busy_overlay(self, message: str = "Processing…") -> None:
        """Cover the entire window with a translucent blocking overlay."""
        overlay = QWidget(self)
        overlay.setObjectName("BusyOverlay")
        overlay.setStyleSheet("""
            QWidget#BusyOverlay {
                background-color: rgba(0, 0, 0, 0.65);
            }
        """)
        overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        overlay.setGeometry(self.rect())
        overlay.raise_()

        lbl = QLabel(message, overlay)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            "color: #a1faff; font-size: 16px; font-weight: 700;"
            " background: transparent;"
        )
        lbl.setGeometry(0, 0, overlay.width(), overlay.height())
        lbl.raise_()

        overlay.show()
        self._busy_overlay = overlay
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def _hide_busy_overlay(self) -> None:
        if hasattr(self, "_busy_overlay") and self._busy_overlay:
            self._busy_overlay.deleteLater()
            self._busy_overlay = None
        QApplication.restoreOverrideCursor()

    def _cb_submit_annotations(self) -> None:
        if not self._current_web_request:
            return
        doc = self._controller.document
        if not doc:
            return
        import numpy as np
        annotation_images = {
            lc.name: (
                doc.annotations[i]
                if i < doc.num_layers
                else np.zeros((doc.height, doc.width), dtype=np.uint8)
            )
            for i, lc in enumerate(self._controller.layer_configs)
        }
        success = self._web_service.submit_annotations(annotation_images)
        if success:
            self._toolbar.set_web_service_ui(mode_active=True, progress_complete=False)
            self._current_web_request = None
        else:
            logger.error("Failed to submit annotations")

    def _cb_cancel_annotations(self) -> None:
        if not self._current_web_request:
            return
        self._web_service.cancel_current_request()
        self._toolbar.set_web_service_ui(mode_active=True, progress_complete=False)
        self._current_web_request = None

    # ------------------------------------------------------------------
    # Focus management
    # ------------------------------------------------------------------

    def _on_focus_changed(self, old, new) -> None:
        """Redirect focus to the viewer whenever a sidebar panel widget gains it.

        This ensures keyboard shortcuts (zoom, tool keys, etc.) keep working
        immediately after the user clicks any toolbar or right-panel control.
        Widgets inside the panels have ``NoFocus`` where possible, but scroll
        areas and container widgets can still receive focus; this handles all
        such cases in one place.
        """
        if new is None or not hasattr(self, '_toolbar'):
            return
        w = new
        while w is not None:
            if w is self._toolbar or w is self._right_panel or w is self._stats_panel:
                QTimer.singleShot(0, self._viewer.setFocus)
                return
            # Also redirect if focus lands on the viewer's internal canvas
            # widget (e.g. QOpenGLWidget or QGraphicsView child) rather than
            # the viewer container itself.
            if w is getattr(self._viewer, '_canvas', None) or \
               w is getattr(self._viewer, '_view', None):
                QTimer.singleShot(0, self._viewer.setFocus)
                return
            w = w.parent()

    # ------------------------------------------------------------------
    # Resize event — update progress bar fill width
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, '_progress_value'):
            QTimer.singleShot(0, lambda: self._on_progress_changed(self._progress_value))
