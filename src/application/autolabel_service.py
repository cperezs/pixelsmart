"""Autolabeling application service.

Orchestrates plugin execution, annotation application, and correction-
metrics initialisation.  The service owns the PluginManager and is the
single point through which autolabeling flows.
"""
from __future__ import annotations

import logging
from typing import Optional

from domain.image_document import ImageDocument
from infrastructure.metadata import ImageMetadata
from application.autolabel_metrics import AutolabelMetrics
from application.plugin_manager import PluginManager

logger = logging.getLogger(__name__)


class AutolabelService:
    """Application service for running autolabeling plugins.

    Responsibilities:
    - Own and configure the PluginManager.
    - Execute a plugin on a document and update metadata.
    - Begin / end correction-metric sessions.
    """

    def __init__(self, layer_names: list[str]) -> None:
        self._layer_names: list[str] = list(layer_names)
        self._plugin_manager = PluginManager()
        self._plugin_manager.update_layers(layer_names)
        self._active_metrics: Optional[AutolabelMetrics] = None

    # ------------------------------------------------------------------
    # Plugin discovery
    # ------------------------------------------------------------------

    def refresh_plugins(self, layer_names: list[str]) -> None:
        """Update the layer context and recalculate compatible plugins."""
        self._layer_names = list(layer_names)
        self._plugin_manager.update_layers(layer_names)

    def get_compatible_plugins(self):
        return self._plugin_manager.get_compatible_plugins()

    def get_plugin_by_id(self, plugin_id: str):
        return self._plugin_manager.get_plugin_by_id(plugin_id)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        plugin_id: str,
        document: ImageDocument,
        metadata: ImageMetadata,
        plugin_config=None,
    ) -> tuple[bool, Optional[str]]:
        """Run plugin *plugin_id* on *document* and record results in *metadata*.

        Parameters
        ----------
        plugin_config
            Optional ``PluginConfig`` object with layer mapping and conflict
            resolution strategy.  When ``None`` the defaults are used.

        Returns
        -------
        ``(True, None)``
            Success — *document* annotations have been replaced.
        ``(False, message)``
            Failure — the document is unchanged.
        """
        plugin = self._plugin_manager.get_plugin_by_id(plugin_id)
        if plugin is None:
            return False, f"Unknown or incompatible plugin: {plugin_id}"

        label_map, error = self._plugin_manager.run_plugin(plugin, document.image, plugin_config)
        if error:
            return False, error

        document.set_from_labelmap(label_map)
        metadata.update_pixel_stats(
            document.annotations,
            image_size=(document.height, document.width),
        )
        metadata.set_autolabel_plugin(plugin.id)

        self._active_metrics = AutolabelMetrics(
            plugin_id=plugin.id,
            layer_names=self._layer_names,
            metadata=metadata,
        )
        logger.info("Autolabel run: plugin=%s", plugin.id)
        return True, None

    # ------------------------------------------------------------------
    # Correction metrics
    # ------------------------------------------------------------------

    def begin_correction(self, annotations) -> None:
        """Snapshot annotation state before a manual correction edit."""
        if self._active_metrics:
            self._active_metrics.begin_edit(annotations)

    def end_correction(self, annotations) -> None:
        """Diff and accumulate counters after a manual correction edit."""
        if self._active_metrics:
            self._active_metrics.end_edit(annotations)

    def finalize_session(self) -> None:
        """End the current correction session (no-op if none is active)."""
        self._active_metrics = None

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    def get_plugin_fine_tune_info(self, plugin_id: str) -> tuple[bool, list[dict]]:
        """Devuelve (can_fine_tune, versions) para el plugin dado."""
        plugin = self.get_plugin_by_id(plugin_id)
        if plugin is None:
            return False, []
        return plugin.can_fine_tune, plugin.list_versions()

    def run_fine_tune(self, plugin_id: str, images_and_annotations: list[dict]) -> None:
        """Delega fine_tune() en el plugin. Llamar desde un hilo de fondo."""
        plugin = self.get_plugin_by_id(plugin_id)
        if plugin is None:
            raise ValueError(f"Plugin '{plugin_id}' no encontrado.")
        plugin.fine_tune(images_and_annotations)

    def set_active_version(self, plugin_id: str, version: str) -> None:
        """Indica al plugin la versión a usar en la próxima inferencia."""
        plugin = self.get_plugin_by_id(plugin_id)
        if plugin is not None and hasattr(plugin, "set_active_version"):
            plugin.set_active_version(version)
