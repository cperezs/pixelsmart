"""ONNX tiled segmentation — multi-model plugin directory.

Layout
------
plugins/onnx_tiled/
    onnx/
        <model_name>/          ← one subfolder per model
            <layer_a>.onnx
            <layer_b>.onnx
            ...

Each subfolder yields one AutolabelPlugin instance.  The plugin's
supported_layers are the sorted stems of the .onnx files it contains.
Different subfolders may declare different layer sets and will therefore
only be compatible with applications whose layers.txt matches exactly.

Inference parameters (shared by all models):
  patch_size  = 256
  margin      = 5
  class_index = 1

For every pixel the layer whose model produced the highest probability
is selected (argmax), guaranteeing full-image coverage with no conflicts.
"""

import os
import logging
import shutil
import time

import cv2
import numpy as np
import onnxruntime as ort

from application.plugin_base import AutolabelPlugin

_PATCH_SIZE = 256
_MARGIN = 5
_CLASS_INDEX = 1
_THRESHOLD = 0.2  # used for debug logging only


# ──────────────────────────────────────────────────────────────────────
# Inference helpers (shared across all OnnxTiledPlugin instances)
# ──────────────────────────────────────────────────────────────────────

def infer_layout_and_channels(input_shape):
    if len(input_shape) != 4:
        raise ValueError(f"Se esperaba una entrada 4D, pero el modelo usa: {input_shape}")

    second = input_shape[1]
    last = input_shape[-1]

    second_is_channel = second in (1, 3)
    last_is_channel = last in (1, 3)

    if second_is_channel and not last_is_channel:
        return "NCHW", int(second)

    if last_is_channel and not second_is_channel:
        return "NHWC", int(last)

    if last_is_channel:
        return "NHWC", int(last)

    if second_is_channel:
        return "NCHW", int(second)

    raise ValueError(
        f"No puedo inferir el layout del modelo con shape {input_shape}. "
        f"Esperaba 1 o 3 canales en la posición 1 o en la última."
    )


def normalize_patch(patch_np):
    return (255.0 - patch_np.astype(np.float32)) / 255.0


def preprocess_patch(patch_np, layout, channels):
    patch_np = normalize_patch(patch_np)

    if channels == 1:
        if patch_np.ndim != 2:
            raise ValueError(f"Se esperaba parche 2D y llegó {patch_np.shape}")
        if layout == "NCHW":
            patch_np = patch_np[None, None, :, :]   # 1,1,H,W
        else:
            patch_np = patch_np[None, :, :, None]   # 1,H,W,1

    elif channels == 3:
        if patch_np.ndim != 3 or patch_np.shape[-1] != 3:
            raise ValueError(f"Se esperaba parche HxWx3 y llegó {patch_np.shape}")
        # cv2.imread devuelve BGR; no convertir a RGB.
        if layout == "NCHW":
            patch_np = np.transpose(patch_np, (2, 0, 1))[None, :, :, :]  # 1,3,H,W
        else:
            patch_np = patch_np[None, :, :, :]                           # 1,H,W,3

    return patch_np.astype(np.float32)


def standardize_output(pred):
    pred = np.array(pred)

    if pred.ndim == 4 and pred.shape[0] == 1:
        pred = pred[0]

    if pred.ndim == 3 and pred.shape[0] in (1, 2, 3) and pred.shape[-1] not in (1, 2, 3):
        pred = np.transpose(pred, (1, 2, 0))

    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred[..., 0]

    if pred.ndim not in (2, 3):
        raise ValueError(f"Se esperaba salida 2D o 3D y llegó {pred.shape}")

    return pred.astype(np.float32)


def select_output_channel(pred, class_index=1):
    if pred.ndim == 2:
        return pred
    if pred.ndim == 3:
        if class_index < 0 or class_index >= pred.shape[-1]:
            raise ValueError(
                f"class_index={class_index} fuera de rango para salida con shape {pred.shape}"
            )
        return pred[..., class_index]
    raise ValueError(f"Forma de salida no soportada: {pred.shape}")


def compute_starts(length, patch_size, stride):
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def pad_to_min_size(img_np, patch_size):
    h, w = img_np.shape[:2]
    new_h = max(h, patch_size)
    new_w = max(w, patch_size)
    pad_h = new_h - h
    pad_w = new_w - w
    if pad_h == 0 and pad_w == 0:
        return img_np, h, w
    if img_np.ndim == 2:
        padded = np.pad(img_np, ((0, pad_h), (0, pad_w)), mode="edge")
    else:
        padded = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return padded, h, w


# ──────────────────────────────────────────────────────────────────────
# Plugin class
# ──────────────────────────────────────────────────────────────────────

class OnnxTiledPlugin(AutolabelPlugin):
    """One ONNX-tiled segmentation model exposed as an AutolabelPlugin.

    Each instance is built from a single subfolder under ``onnx/``.
    The subfolder name becomes the plugin id; the .onnx file stems
    become the layer names.
    """

    def __init__(self, model_name: str, model_dir: str):
        self._logger = logging.getLogger(f"OnnxTiledPlugin.{model_name}")
        self._model_name = model_name
        self._model_dir = model_dir

        onnx_files = sorted(
            f for f in os.listdir(model_dir) if f.endswith(".onnx")
        )
        if not onnx_files:
            raise RuntimeError(
                f"No .onnx files found in model directory: {model_dir}"
            )

        self._layer_names: list = [os.path.splitext(f)[0] for f in onnx_files]

        self._sessions: list = []
        for fname in onnx_files:
            path = os.path.join(model_dir, fname)
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self._logger.info("Loaded ONNX model: %s", fname)
            self._sessions.append(session)

    # AutolabelPlugin interface ----------------------------------------

    @property
    def id(self) -> str:
        return f"onnx_tiled_{self._model_name}"

    @property
    def display_name(self) -> str:
        return f"{self._model_name} (ONNX tiled)"

    @property
    def supported_layers(self) -> list:
        return list(self._layer_names)

    # Fine-tuning interface --------------------------------------------

    @property
    def can_fine_tune(self) -> bool:
        return True

    def list_versions(self) -> list[dict]:
        """Devuelve las versiones disponibles: v1 (original) + fine_tuned/vN."""
        import datetime

        versions = []

        # v1: ficheros .onnx en la raíz del modelo
        root_onnx = [
            os.path.join(self._model_dir, f)
            for f in os.listdir(self._model_dir)
            if f.endswith(".onnx")
        ]
        if root_onnx:
            max_mtime = max(os.path.getmtime(p) for p in root_onnx)
            date_str = datetime.datetime.fromtimestamp(max_mtime).strftime("%Y-%m-%d %H:%M")
        else:
            date_str = ""
        versions.append({"label": "v1", "date": date_str, "is_original": True})

        # vN: subcarpetas fine_tuned/vN/
        fine_tuned_dir = os.path.join(self._model_dir, "fine_tuned")
        if os.path.isdir(fine_tuned_dir):
            for entry in os.listdir(fine_tuned_dir):
                sub = os.path.join(fine_tuned_dir, entry)
                if not os.path.isdir(sub) or not entry.startswith("v"):
                    continue
                try:
                    int(entry[1:])
                except ValueError:
                    continue
                sub_onnx = [
                    os.path.join(sub, f)
                    for f in os.listdir(sub)
                    if f.endswith(".onnx")
                ]
                if sub_onnx:
                    sub_mtime = max(os.path.getmtime(p) for p in sub_onnx)
                    sub_date = datetime.datetime.fromtimestamp(sub_mtime).strftime("%Y-%m-%d %H:%M")
                else:
                    sub_date = ""
                versions.append({"label": entry, "date": sub_date, "is_original": False})

        # Ordenar: mayor número primero, v1 al final
        def _version_key(v):
            try:
                return int(v["label"][1:])
            except (ValueError, IndexError):
                return 0

        versions.sort(key=_version_key, reverse=True)
        return versions

    def fine_tune(self, images_and_annotations: list[dict]) -> None:
        """Stub de fine-tuning: espera 30 s y copia .onnx originales a fine_tuned/vN/."""
        fine_tuned_dir = os.path.join(self._model_dir, "fine_tuned")
        os.makedirs(fine_tuned_dir, exist_ok=True)

        # Determinar siguiente versión
        max_n = 1
        if os.path.isdir(fine_tuned_dir):
            for entry in os.listdir(fine_tuned_dir):
                if entry.startswith("v"):
                    try:
                        n = int(entry[1:])
                        if n > max_n:
                            max_n = n
                    except ValueError:
                        pass
        next_n = max_n + 1

        new_version_dir = os.path.join(fine_tuned_dir, f"v{next_n}")
        os.makedirs(new_version_dir, exist_ok=True)
        self._logger.info("fine_tune: creando versión v%d en %s", next_n, new_version_dir)

        # Guardar imágenes y anotaciones recibidas para depuración
        for i, item in enumerate(images_and_annotations):
            img = item.get("image")
            anns = item.get("annotations", [])
            if img is not None:
                cv2.imwrite(os.path.join(new_version_dir, f"{i:04d}.png"), img)
            for j, mask in enumerate(anns):
                layer_name = self._layer_names[j] if j < len(self._layer_names) else str(j)
                if mask is not None:
                    cv2.imwrite(
                        os.path.join(new_version_dir, f"{i:04d}_{layer_name}.png"), mask
                    )

        time.sleep(30)

        for fname in os.listdir(self._model_dir):
            if fname.endswith(".onnx"):
                src = os.path.join(self._model_dir, fname)
                dst = os.path.join(new_version_dir, fname)
                shutil.copy2(src, dst)
                self._logger.info("fine_tune: copiado %s -> %s", src, dst)

        self._logger.info("fine_tune: versión v%d lista en %s", next_n, new_version_dir)

    def set_active_version(self, version: str) -> None:
        """Carga las sesiones ONNX de la versión indicada.

        Si ``version == "v1"`` usa los ficheros .onnx de la raíz del modelo.
        Para versiones posteriores busca en ``fine_tuned/{version}/``.
        Si la versión no existe en disco registra un warning y no cambia nada.
        """
        if version == "v1":
            target_dir = self._model_dir
        else:
            target_dir = os.path.join(self._model_dir, "fine_tuned", version)

        if not os.path.isdir(target_dir):
            self._logger.warning(
                "set_active_version: versión '%s' no encontrada en %s", version, target_dir
            )
            return

        onnx_files = sorted(f for f in os.listdir(target_dir) if f.endswith(".onnx"))
        if not onnx_files:
            self._logger.warning(
                "set_active_version: no hay ficheros .onnx en %s", target_dir
            )
            return

        new_sessions = []
        for fname in onnx_files:
            path = os.path.join(target_dir, fname)
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self._logger.info("set_active_version: cargado %s", path)
            new_sessions.append(session)

        self._sessions = new_sessions
        self._logger.info("set_active_version: activa versión '%s'", version)

    def run(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prob_maps = []
        for session, layer_name in zip(self._sessions, self._layer_names):
            prob = self._run_model(session, image, layer_name)
            prob_maps.append(prob)

        prob_stack = np.stack(prob_maps, axis=-1)
        label_map = np.argmax(prob_stack, axis=-1).astype(np.uint8)
        H, W = prob_stack.shape[:2]
        winner_prob = prob_stack[np.arange(H)[:, None], np.arange(W)[None, :], label_map]
        return label_map, winner_prob


    def run_with_config(
        self,
        image: np.ndarray,
        strategy: str,
        layer_priorities: dict,
    ) -> tuple[np.ndarray, dict]:
        """Run inference with a specific conflict resolution strategy.

        Parameters
        ----------
        strategy:
            ``"argmax"`` — standard argmax (same as ``run()``).
            ``"layer_priority"`` — assign each pixel to the highest-priority
            layer whose probability ≥ 0.5; falls back to argmax.
        layer_priorities:
            ``{layer_name: priority_int}`` dict (1 = highest priority).
            Used only when *strategy* is ``"layer_priority"``.
        """
        prob_maps = []
        for session, layer_name in zip(self._sessions, self._layer_names):
            prob = self._run_model(session, image, layer_name)
            prob_maps.append(prob)

        prob_stack = np.stack(prob_maps, axis=-1)  # H × W × N_layers

        if strategy == "layer_priority" and layer_priorities:
            label_map = self._apply_layer_priority(prob_stack, layer_priorities)
        else:
            label_map = np.argmax(prob_stack, axis=-1).astype(np.uint8)
        H, W = prob_stack.shape[:2]
        winner_prob = prob_stack[np.arange(H)[:, None], np.arange(W)[None, :], label_map]
        return label_map, winner_prob

    # Conflict resolution ----------------------------------------------

    _PRIORITY_THRESHOLD = 0.5  # minimum probability for a layer to "claim" a pixel

    def _apply_layer_priority(
        self,
        prob_stack: np.ndarray,
        layer_priorities: dict,
    ) -> np.ndarray:
        """Assign pixels using layer priority instead of argmax.

        Processing order: from the lowest-priority layer to the highest so
        that the highest-priority layer wins by overwriting lower-priority
        assignments.  Pixels where no layer reaches ``_PRIORITY_THRESHOLD``
        are assigned by argmax (fallback).
        """
        # Fallback: argmax over all layers
        label_map = np.argmax(prob_stack, axis=-1).astype(np.uint8)

        # Sort layers from lowest priority (highest number) to highest (1),
        # so that we overwrite in ascending priority \u2192 highest priority wins.
        priority_order = sorted(
            (
                (layer_priorities.get(name, idx + 1), idx)
                for idx, name in enumerate(self._layer_names)
            ),
            key=lambda x: x[0],
            reverse=True,          # lowest priority first
        )

        for _priority, layer_idx in priority_order:
            mask = prob_stack[..., layer_idx] >= self._PRIORITY_THRESHOLD
            label_map[mask] = layer_idx

        return label_map

    # Internal helpers -------------------------------------------------

    def _run_model(self, session, image: np.ndarray, layer_name: str) -> np.ndarray:
        input_info = session.get_inputs()[0]
        input_name = input_info.name
        input_shape = input_info.shape

        layout, channels = infer_layout_and_channels(input_shape)
        img_np = self._prepare_channels(image, channels)
        img_np, orig_h, orig_w = pad_to_min_size(img_np, _PATCH_SIZE)
        H, W = img_np.shape[:2]

        stride = _PATCH_SIZE - 2 * _MARGIN
        ys = compute_starts(H, _PATCH_SIZE, stride)
        xs = compute_starts(W, _PATCH_SIZE, stride)

        output_sum = np.zeros((H, W), dtype=np.float32)
        output_count = np.zeros((H, W), dtype=np.float32)

        for y in ys:
            for x in xs:
                patch = img_np[y : y + _PATCH_SIZE, x : x + _PATCH_SIZE]
                inp = preprocess_patch(patch, layout, channels)
                pred = session.run(None, {input_name: inp})[0]
                pred = standardize_output(pred)
                pred = select_output_channel(pred, class_index=_CLASS_INDEX)

                if pred.shape != (_PATCH_SIZE, _PATCH_SIZE):
                    raise ValueError(
                        f"Model '{layer_name}': expected patch output "
                        f"{(_PATCH_SIZE, _PATCH_SIZE)}, got {pred.shape}"
                    )

                top_crop    = 0 if y == ys[0] else _MARGIN
                left_crop   = 0 if x == xs[0] else _MARGIN
                bottom_crop = 0 if y == ys[-1] else _MARGIN
                right_crop  = 0 if x == xs[-1] else _MARGIN

                pred_valid = pred[
                    top_crop : _PATCH_SIZE - bottom_crop,
                    left_crop : _PATCH_SIZE - right_crop,
                ]

                y0 = y + top_crop
                y1 = y + _PATCH_SIZE - bottom_crop
                x0 = x + left_crop
                x1 = x + _PATCH_SIZE - right_crop

                output_sum[y0:y1, x0:x1] += pred_valid
                output_count[y0:y1, x0:x1] += 1.0

        output_count[output_count == 0] = 1.0
        result = (output_sum / output_count)[:orig_h, :orig_w]

        self._logger.debug(
            "Model '%s': mean prob=%.3f, pixels>threshold=%d",
            layer_name,
            float(result.mean()),
            int((result >= _THRESHOLD).sum()),
        )
        return result

    @staticmethod
    def _prepare_channels(image: np.ndarray, channels: int) -> np.ndarray:
        if channels == 1:
            if image.ndim == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image.copy()
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image.copy()


# ──────────────────────────────────────────────────────────────────────
# Multi-plugin factory — called by PluginManager
# ──────────────────────────────────────────────────────────────────────

def get_plugins() -> list:
    """Return one OnnxTiledPlugin instance per subfolder inside onnx/.

    Subfolders with no .onnx files are skipped with a warning.
    Any subfolder that fails to load is skipped without crashing.
    """
    logger = logging.getLogger("onnx_tiled.get_plugins")
    onnx_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx")

    plugins = []
    if not os.path.isdir(onnx_root):
        logger.warning("onnx_tiled: 'onnx/' directory not found at %s", onnx_root)
        return plugins

    for entry in sorted(os.listdir(onnx_root)):
        model_dir = os.path.join(onnx_root, entry)
        if not os.path.isdir(model_dir):
            continue
        has_onnx = any(f.endswith(".onnx") for f in os.listdir(model_dir))
        if not has_onnx:
            logger.warning("onnx_tiled: skipping '%s' — no .onnx files found", entry)
            continue
        try:
            plugin = OnnxTiledPlugin(model_name=entry, model_dir=model_dir)
            plugins.append(plugin)
            logger.info(
                "onnx_tiled: registered model '%s' with layers %s",
                entry,
                plugin.supported_layers,
            )
        except Exception as exc:
            logger.error("onnx_tiled: failed to load model '%s': %s", entry, exc)

    return plugins
