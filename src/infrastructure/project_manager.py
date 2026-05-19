"""Project persistence — loading, saving, and remembering projects.

Handles two distinct configuration files:

1. **Application-level** dot-config (``~/.pixelsmart/app.toml``) that
   remembers the last opened project folder.
2. **Project-level** config (``.pixelsmart/project.toml``) inside each
   project folder that stores the project settings.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import Optional

from domain.project import ProjectConfig

logger = logging.getLogger(__name__)

# Resolved lazily to the application root by set_app_root().
_app_config_file: Optional[str] = None

_PROJECT_DIR_NAME = ".pixelsmart"
_PROJECT_CONFIG_NAME = "project.json"


class ProjectManager:
    """Manages project lifecycle: open, save, remember last project."""

    @staticmethod
    def set_app_root(root: str) -> None:
        """Set the application root directory for the app-level config file."""
        global _app_config_file
        _app_config_file = os.path.join(root, ".pixelsmart_app.json")

    # ------------------------------------------------------------------
    # Application-level config (last opened folder)
    # ------------------------------------------------------------------

    @staticmethod
    def get_last_project_path() -> Optional[str]:
        """Return the path to the last opened project folder, or *None*."""
        if _app_config_file is None or not os.path.isfile(_app_config_file):
            return None
        try:
            with open(_app_config_file, "r") as fh:
                data = json.load(fh)
            path = data.get("last_project")
            if path and os.path.isdir(path):
                return path
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read app config: %s", exc)
        return None

    @staticmethod
    def set_last_project_path(path: str) -> None:
        """Persist *path* as the last opened project folder."""
        if _app_config_file is None:
            logger.warning("App root not set — cannot persist last project path.")
            return
        try:
            with open(_app_config_file, "w") as fh:
                json.dump({"last_project": path}, fh)
        except OSError as exc:
            logger.warning("Could not write app config: %s", exc)

    # ------------------------------------------------------------------
    # Project-level config
    # ------------------------------------------------------------------

    @staticmethod
    def _project_config_path(folder: str) -> str:
        return os.path.join(folder, _PROJECT_DIR_NAME, _PROJECT_CONFIG_NAME)

    @staticmethod
    def load_project(folder: str) -> ProjectConfig:
        """Load the project config from *folder*, or return defaults."""
        config_path = ProjectManager._project_config_path(folder)
        name = os.path.basename(os.path.normpath(folder))
        config = ProjectConfig(name=name)
        if not os.path.isfile(config_path):
            return config
        try:
            with open(config_path, "r") as fh:
                data = json.load(fh)
            config.viewer_backend = data.get("viewer_backend", config.viewer_backend)
            config.pen_size = data.get("pen_size", config.pen_size)
            config.eraser_size = data.get("eraser_size", config.eraser_size)
            config.selector_threshold = data.get("selector_threshold", config.selector_threshold)
            config.selector_auto_smooth = data.get("selector_auto_smooth", config.selector_auto_smooth)
            config.fill_all = data.get("fill_all", config.fill_all)
            config.show_image = data.get("show_image", config.show_image)
            config.show_other_layers = data.get("show_other_layers", config.show_other_layers)
            config.show_missing_pixels = data.get("show_missing_pixels", config.show_missing_pixels)
            config.show_grid = data.get("show_grid", config.show_grid)
            config.active_layer = data.get("active_layer", config.active_layer)
            config.locked_layers = data.get("locked_layers", config.locked_layers)
            config.hidden_layers = data.get("hidden_layers", config.hidden_layers)
            config.last_image = data.get("last_image", config.last_image)
            config.active_tool = data.get("active_tool", config.active_tool)
            config.selected_plugin_id = data.get("selected_plugin_id", config.selected_plugin_id)
            config.plugin_configs = data.get("plugin_configs", config.plugin_configs)
            config.global_layer_opacity = data.get("global_layer_opacity", config.global_layer_opacity)
            config.show_stats_panel = data.get("show_stats_panel", config.show_stats_panel)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read project config from %s: %s", config_path, exc)
        return config

    @staticmethod
    def save_project(folder: str, config: ProjectConfig) -> None:
        """Persist the project config to *folder*."""
        config_dir = os.path.join(folder, _PROJECT_DIR_NAME)
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, _PROJECT_CONFIG_NAME)
        try:
            with open(config_path, "w") as fh:
                json.dump(asdict(config), fh, indent=2)
        except OSError as exc:
            logger.warning("Could not write project config to %s: %s", config_path, exc)

    @staticmethod
    def is_project(folder: str) -> bool:
        """Return *True* if *folder* already contains a project config."""
        return os.path.isfile(ProjectManager._project_config_path(folder))
