"""Base class for autolabeling plugins.

Each plugin must subclass AutolabelPlugin and implement all abstract
properties and methods.  The plugin's ``run`` method receives the raw
BGR image (as loaded by OpenCV) and must return a tuple
``(label_map, confidence_map)`` where *label_map* is a 2-D NumPy array
whose pixel values are indices into ``supported_layers`` and
*confidence_map* is an optional 2-D ``np.ndarray`` of float32 values
in ``[0, 1]`` representing the per-pixel prediction confidence, or
``None`` if the plugin cannot produce one.
"""

from abc import ABC, abstractmethod
import numpy as np


class AutolabelPlugin(ABC):
    """Abstract base class that every autolabeling plugin must implement."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique plugin identifier (matches the subdirectory name)."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name shown in the UI dropdown."""
        ...

    @property
    @abstractmethod
    def supported_layers(self) -> list:
        """Ordered list of layer names this plugin can produce.

        The index of each name in this list corresponds to the integer
        value used in the label map returned by ``run()``.
        """
        ...

    @abstractmethod
    def run(self, image: np.ndarray) -> tuple[np.ndarray, "np.ndarray | None"]:
        """Run the plugin on the given image.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format (OpenCV convention), shape ``(H, W, C)``.

        Returns
        -------
        (label_map, confidence_map)
            *label_map* — 2-D ``np.ndarray`` of shape ``(H, W)`` whose pixel
            values are valid indices into ``supported_layers``.
            *confidence_map* — optional 2-D ``np.ndarray`` of float32 values
            in ``[0, 1]`` representing per-pixel prediction confidence,
            or ``None`` if the plugin cannot provide one.
        """
        ...

    # ------------------------------------------------------------------
    # Fine-tuning interface (optional — default: not supported)
    # ------------------------------------------------------------------

    @property
    def can_fine_tune(self) -> bool:
        """True si el plugin soporta reentrenamiento. Default: False."""
        return False

    def list_versions(self) -> list[dict]:
        """Devuelve las versiones disponibles del modelo.

        Cada elemento es un dict con las claves:
          - "label": str        — etiqueta de la versión (e.g. "v1", "v2")
          - "date": str         — fecha en formato ISO-8601 (e.g. "2026-05-19")
          - "is_original": bool — True solo para v1

        Default: lista vacía.
        """
        return []

    def fine_tune(self, images_and_annotations: list[dict]) -> None:
        """Ejecuta el reentrenamiento con las imágenes proporcionadas.

        Cada elemento de images_and_annotations es un dict con:
          - "image": np.ndarray        — imagen BGR
          - "annotations": list[np.ndarray]  — máscaras por capa

        Default: lanza NotImplementedError.
        Nota: este método es síncrono y se llamará desde un hilo de fondo.
        """
        raise NotImplementedError(f"{self.__class__.__name__} no soporta fine-tuning.")
