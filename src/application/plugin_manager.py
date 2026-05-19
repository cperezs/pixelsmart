"""Plugin discovery, compatibility filtering, execution, and output validation."""

import os
import sys
import logging
import importlib
import numpy as np

from application.plugin_base import AutolabelPlugin


class PluginManager:
    """Discovers, validates, and runs autolabeling plugins."""

    def __init__(self, plugins_dir=None):
        self._logger = logging.getLogger("PluginManager")
        if plugins_dir is None:
            # Resolve relative to this file: src/application/ → src/plugins/
            plugins_dir = os.path.join(os.path.dirname(__file__), "..", "plugins")
        self._plugins_dir = os.path.abspath(plugins_dir)
        self._plugins: list[AutolabelPlugin] = []
        self._compatible_plugins: list[AutolabelPlugin] = []
        self._current_layers: list[str] = []
        self.discover_plugins()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_plugins(self):
        """Scan the plugins directory and load all valid plugins."""
        self._plugins = []

        if not os.path.isdir(self._plugins_dir):
            self._logger.warning("Plugins directory not found: %s", self._plugins_dir)
            return

        # Ensure the *parent* of plugins_dir is on sys.path so that
        # ``import plugins.<name>`` works.
        parent_dir = os.path.abspath(os.path.dirname(self._plugins_dir))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        for entry in sorted(os.listdir(self._plugins_dir)):
            plugin_path = os.path.join(self._plugins_dir, entry)
            if not os.path.isdir(plugin_path):
                continue
            try:
                self._load_plugin(entry, plugin_path)
            except Exception as e:
                self._logger.error("Failed to load plugin '%s': %s", entry, e)

    def _load_plugin(self, plugin_id: str, plugin_path: str):
        """Import and register a single plugin from *plugin_path*."""
        init_file = os.path.join(plugin_path, "__init__.py")
        if not os.path.isfile(init_file):
            self._logger.warning(
                "Plugin '%s' has no __init__.py, skipping.", plugin_id
            )
            return

        module_name = f"plugins.{plugin_id}"

        # Drop cached module so re-discovery works after hot-reload.
        if module_name in sys.modules:
            del sys.modules[module_name]

        module = importlib.import_module(module_name)

        # Multi-plugin factory: if the module exports get_plugins(), use it.
        # This allows one plugin directory to register multiple plugin instances
        # with different ids/layers (e.g. one per model subfolder).
        get_plugins_fn = getattr(module, "get_plugins", None)
        if callable(get_plugins_fn):
            instances = get_plugins_fn()
            for plugin in instances:
                self._validate_and_register(plugin, plugin_id)
            return

        # Single-plugin fallback: find the first concrete AutolabelPlugin subclass.
        plugin_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, AutolabelPlugin)
                and attr is not AutolabelPlugin
            ):
                plugin_class = attr
                break

        if plugin_class is None:
            self._logger.warning(
                "Plugin '%s' has no AutolabelPlugin subclass, skipping.",
                plugin_id,
            )
            return

        self._validate_and_register(plugin_class(), plugin_id)

    def _validate_and_register(self, plugin, source_id: str):
        """Validate a plugin instance and add it to the registry."""
        if not isinstance(plugin.id, str) or not plugin.id:
            raise ValueError(f"Plugin from '{source_id}' has invalid id")
        if not isinstance(plugin.display_name, str) or not plugin.display_name:
            raise ValueError(f"Plugin '{plugin.id}' has invalid display_name")
        if not isinstance(plugin.supported_layers, list) or not plugin.supported_layers:
            raise ValueError(f"Plugin '{plugin.id}' has invalid supported_layers")

        self._plugins.append(plugin)
        self._logger.info("Loaded plugin: %s (%s)", plugin.id, plugin.display_name)

    # ------------------------------------------------------------------
    # Compatibility
    # ------------------------------------------------------------------

    def update_layers(self, layer_names: list[str]):
        """Update the current layer context.

        All discovered plugins are made available regardless of their
        declared layer names — compatibility is now determined entirely
        by the user-configured layer mapping in the UI.
        """
        self._current_layers = list(layer_names)
        self._compatible_plugins = list(self._plugins)
        for plugin in self._plugins:
            self._logger.info("Plugin '%s' available.", plugin.id)

    def get_compatible_plugins(self) -> list[AutolabelPlugin]:
        """Return the list of currently compatible plugins."""
        return list(self._compatible_plugins)

    def get_plugin_by_id(self, plugin_id: str):
        """Return a compatible plugin by *plugin_id*, or ``None``."""
        for plugin in self._compatible_plugins:
            if plugin.id == plugin_id:
                return plugin
        return None

    # ------------------------------------------------------------------
    # Execution & validation
    # ------------------------------------------------------------------

    def run_plugin(
        self,
        plugin: AutolabelPlugin,
        image_array: np.ndarray,
        plugin_config=None,
    ):
        """Execute *plugin* on *image_array* and validate the output.

        Parameters
        ----------
        plugin_config
            Optional ``PluginConfig`` object (duck-typed).  When provided,
            ``plugin_config.layer_mapping`` overrides automatic name-based
            layer matching and ``plugin_config.conflict_strategy`` /
            ``plugin_config.layer_priorities`` control how per-pixel
            predictions are resolved.

        Returns
        -------
        (label_map, None) on success — *label_map* uses **app** layer indices.
        (None, error_message) on failure.
        """
        h, w = image_array.shape[:2]

        # --- run ----------------------------------------------------------
        try:
            if plugin_config is not None and hasattr(plugin, 'run_with_config'):
                strategy = getattr(plugin_config, 'conflict_strategy', 'argmax')
                priorities = getattr(plugin_config, 'layer_priorities', {})
                result = plugin.run_with_config(image_array, strategy, priorities)
            else:
                result = plugin.run(image_array)
        except Exception as e:
            self._logger.error("Plugin '%s' execution failed: %s", plugin.id, e)
            return None, f"Plugin execution failed: {e}"

        # --- validate -----------------------------------------------------
        error = self._validate_output(result, h, w, plugin)
        if error:
            self._logger.error(
                "Plugin '%s' output invalid: %s", plugin.id, error
            )
            return None, error

        # --- map plugin indices → app indices -----------------------------
        user_mapping = getattr(plugin_config, 'layer_mapping', None) if plugin_config else None
        mapped = self._map_layers(result, plugin, user_mapping)
        if mapped is None:
            msg = "Failed to map plugin layers to application layers."
            self._logger.error(msg)
            return None, msg

        return mapped, None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_output(result, h, w, plugin):
        """Return an error string if *result* is invalid, else ``None``."""
        if not isinstance(result, np.ndarray):
            return f"Expected numpy array, got {type(result).__name__}"
        if result.ndim != 2:
            return f"Expected 2D array (H, W), got shape {result.shape}"
        if result.shape[0] != h or result.shape[1] != w:
            return (
                f"Dimensions mismatch: expected ({h}, {w}), "
                f"got ({result.shape[0]}, {result.shape[1]})"
            )
        unique_vals = np.unique(result)
        n_layers = len(plugin.supported_layers)
        invalid = unique_vals[(unique_vals < 0) | (unique_vals >= n_layers)]
        if len(invalid) > 0:
            return f"Invalid layer indices in output: {invalid.tolist()}"
        return None

    def _map_layers(self, result, plugin, user_mapping: dict | None = None):
        """Map plugin layer indices to app layer indices.

        When *user_mapping* is ``None`` (no configuration saved) every layer
        is considered unassigned and the output is all-zero.
        When a layer maps to ``None`` (explicit \"-- No assignment --\") those
        pixels are skipped and stay at 0 in the output.
        """
        # Sentinel: a value that doesn't match any app layer index so that
        # unassigned pixels produce no annotation in set_from_labelmap.
        app_lookup = {name: idx for idx, name in enumerate(self._current_layers)}
        _UNASSIGNED = len(self._current_layers)
        mapped = np.full_like(result, fill_value=_UNASSIGNED)

        if user_mapping is None:
            # No mapping configured at all — nothing is assigned.
            return mapped

        for plugin_idx, plugin_layer_name in enumerate(plugin.supported_layers):
            app_layer_name = user_mapping.get(plugin_layer_name)
            if app_layer_name is None:
                # Explicitly unassigned — keep sentinel (no annotation).
                continue
            if app_layer_name not in app_lookup:
                self._logger.error(
                    "Layer '%s' not found in app layers.", app_layer_name
                )
                return None
            mapped[result == plugin_idx] = app_lookup[app_layer_name]

        return mapped
