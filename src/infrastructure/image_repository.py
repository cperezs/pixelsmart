"""File-system repository for images and annotation masks.

All path-building and format details are encapsulated here so that
the domain layer (``ImageDocument``) remains free of I/O concerns.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from domain.image_document import ImageDocument

logger = logging.getLogger(__name__)

_IMAGES_DIR = "images"
_ANNOTATIONS_DIR = "annotations"


class ImageRepository:
    """Loads and saves :class:`~domain.image_document.ImageDocument` objects.

    The repository owns all knowledge of:
    - which directories images and annotations live in,
    - how annotation layer files are named (``<stem>_<i>.png``),
    - how to convert between NumPy arrays and image files.
    """

    def __init__(
        self,
        images_dir: str = _IMAGES_DIR,
        annotations_dir: str = _ANNOTATIONS_DIR,
    ) -> None:
        self._images_dir = images_dir
        self._annotations_dir = annotations_dir
        self._migrate_metadata_files()

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def _migrate_metadata_files(self) -> None:
        """Move legacy .metadata files from annotations/ to annotations/metadata/."""
        if not os.path.isdir(self._annotations_dir):
            return
        metadata_dir = os.path.join(self._annotations_dir, "metadata")
        for fname in os.listdir(self._annotations_dir):
            if fname.endswith(".metadata"):
                old_path = os.path.join(self._annotations_dir, fname)
                os.makedirs(metadata_dir, exist_ok=True)
                new_path = os.path.join(metadata_dir, fname)
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    logger.info("Migrated metadata: %s -> %s", old_path, new_path)

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    @property
    def images_dir(self) -> str:
        """Absolute or relative path to the images directory."""
        return self._images_dir

    @property
    def annotations_dir(self) -> str:
        """Absolute or relative path to the annotations directory."""
        return self._annotations_dir

    def list_images(self) -> list[str]:
        """Return sorted base-filenames of every image in *images_dir*."""
        exts = {".jpg", ".jpeg", ".png", ".gif"}
        if not os.path.isdir(self._images_dir):
            return []
        names = [
            f for f in os.listdir(self._images_dir)
            if os.path.splitext(f.lower())[1] in exts
        ]
        return sorted(names)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        filename: str,
        nlayers: int,
        layer_names: Optional[list[str]] = None,
    ) -> ImageDocument:
        """Load *filename* and its annotation masks from disk.

        Parameters
        ----------
        filename:
            Base filename (e.g. ``"img01.png"``), relative to *images_dir*.
        nlayers:
            Expected number of annotation layers.
        layer_names:
            Optional list of layer names used to resolve annotation files
            by name scheme (falls back to index scheme if not found).

        Returns
        -------
        ImageDocument with annotations loaded from disk, or initialised
        to zero-filled masks when no annotation files are found.
        """
        image_path = os.path.join(self._images_dir, filename)
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        h, w = bgr.shape[:2]
        annotations = self._load_annotations(filename, nlayers, h, w, layer_names)
        doc = ImageDocument(bgr, annotations, image_path)

        if np.all(annotations == 0):
            self.save_annotations(doc, filename, layer_names or [])

        logger.info("Loaded image: %s", image_path)
        return doc

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_annotations(
        self,
        document: ImageDocument,
        filename: str,
        layer_names: list[str],
    ) -> None:
        """Persist all annotation layers of *document* to disk."""
        from domain.layer_config import sanitize_name
        os.makedirs(self._annotations_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(filename))[0]
        used_safe_names: set[str] = set()
        for i, mask in enumerate(document.annotations):
            if i < len(layer_names):
                safe = sanitize_name(layer_names[i])
                if safe in used_safe_names:
                    safe = f"{safe}_{i}"
                used_safe_names.add(safe)
                path = os.path.join(self._annotations_dir, f"{stem}_{safe}.png")
            else:
                path = os.path.join(self._annotations_dir, f"{stem}_{i}.png")
            Image.fromarray(mask).convert("1").save(path)
        logger.debug("Saved annotations for: %s", filename)

    def rename_annotation_files(
        self,
        filename: str,
        old_name: str,
        new_name: str,
    ) -> None:
        """Rename annotation files for a single image when a layer is renamed."""
        from domain.layer_config import sanitize_name
        stem = os.path.splitext(os.path.basename(filename))[0]
        old_safe = sanitize_name(old_name)
        new_safe = sanitize_name(new_name)
        old_path = os.path.join(self._annotations_dir, f"{stem}_{old_safe}.png")
        new_path = os.path.join(self._annotations_dir, f"{stem}_{new_safe}.png")
        if os.path.isfile(old_path) and not os.path.isfile(new_path):
            os.rename(old_path, new_path)
            logger.debug("Renamed annotation: %s -> %s", old_path, new_path)

    def delete_annotation_files(self, filename: str, layer_name: str) -> None:
        """Delete the annotation file for a specific layer and image."""
        from domain.layer_config import sanitize_name
        stem = os.path.splitext(os.path.basename(filename))[0]
        safe = sanitize_name(layer_name)
        path = os.path.join(self._annotations_dir, f"{stem}_{safe}.png")
        if os.path.isfile(path):
            os.remove(path)
            logger.debug("Deleted annotation: %s", path)

    def migrate_index_to_name(
        self,
        filename: str,
        layer_names: list[str],
    ) -> None:
        """Rename annotation files from index scheme to name scheme for a single image.

        Skips if destination already exists (already migrated or name collision).
        """
        from domain.layer_config import sanitize_name
        stem = os.path.splitext(os.path.basename(filename))[0]
        for i, name in enumerate(layer_names):
            old_path = os.path.join(self._annotations_dir, f"{stem}_{i}.png")
            safe = sanitize_name(name)
            new_path = os.path.join(self._annotations_dir, f"{stem}_{safe}.png")
            if os.path.isfile(old_path) and not os.path.isfile(new_path):
                os.rename(old_path, new_path)
                logger.info("Migrated annotation: %s -> %s", old_path, new_path)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def metadata_path(self, filename: str) -> str:
        """Return the ``metadata/<stem>.metadata`` path for *filename*."""
        stem = os.path.splitext(os.path.basename(filename))[0]
        metadata_dir = os.path.join(self._annotations_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        return os.path.join(metadata_dir, f"{stem}.metadata")

    def log_path(self, filename: str) -> str:
        """Return the ``logs/<stem>.json`` path for *filename*."""
        stem = os.path.splitext(os.path.basename(filename))[0]
        logs_dir = os.path.join(self._annotations_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        return os.path.join(logs_dir, f"{stem}.json")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_annotations(
        self,
        filename: str,
        nlayers: int,
        h: int,
        w: int,
        layer_names: Optional[list[str]] = None,
    ) -> np.ndarray:
        from domain.layer_config import sanitize_name
        stem = os.path.splitext(os.path.basename(filename))[0]
        result = np.zeros((nlayers, h, w), dtype=np.uint8)
        any_found = False
        for i in range(nlayers):
            # Try new scheme (layer name) first
            if layer_names and i < len(layer_names):
                safe = sanitize_name(layer_names[i])
                new_path = os.path.join(self._annotations_dir, f"{stem}_{safe}.png")
                if os.path.isfile(new_path):
                    img = cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        result[i] = img
                        any_found = True
                        logger.debug("Loaded annotation (name): %s", new_path)
                        continue
            # Fallback: old scheme (index)
            old_path = os.path.join(self._annotations_dir, f"{stem}_{i}.png")
            if os.path.isfile(old_path):
                img = cv2.imread(old_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    result[i] = img
                    any_found = True
                    logger.debug("Loaded annotation (index): %s", old_path)

        if not any_found:
            logger.debug(
                "No annotation files found for %s — initialising %d blank layers.",
                filename, nlayers,
            )
        return result
