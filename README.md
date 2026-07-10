# PixelSmart

An open-source desktop application for pixel-level semantic annotation of images, designed for dense segmentation workflows in the active learning loop. Built with Python, PyQt6, and OpenGL.

## Citation

If you use this tool in academic work, please cite:

> *Citation pending*

---

## Installation

```bash
pip install -r requirements.txt
```

> **Python 3.11+** is recommended. On Python 3.10, the `tomli` back-port is installed automatically.

### Linux: Qt platform plugin errors

If the application fails with `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"`:

```bash
sudo apt install --reinstall libxcb-xinerama0 libxcb-cursor0
sudo apt install --reinstall libxcb1 libx11-xcb1 libxrender1 libxkbcommon-x11-0
```

---

## Launching

```bash
./run.sh          # recommended — resolves the project root from any cwd
python main.py    # equivalent direct invocation
```

---

## Project-Based Workflow

PixelSmart is organized around **projects**. A project is a directory containing images to annotate. On first open, a `.pixelsmart/project.json` file is created inside the directory to store all project-specific settings.

The application remembers the last opened project in `~/.pixelsmart/app.json`. When launched with no active project, a welcome screen is shown.

### What is persisted per project

- Layer definitions, visibility, and lock state
- Last opened image
- Active annotation tool and brush/eraser sizes
- Selector threshold
- ML plugin selection and configuration
- Global layer opacity
- Rendering backend

---

## Layer Configuration

Annotation layers are defined in a `layers.txt` file at the project root. Each line specifies a layer name and an optional hex color:

```
background  #2596be
staff       #9925be
notes       #be4d25
lyrics      #49be25
```

- Layers can be **locked** (protected from modification) or **hidden** individually.
- Layer overlays are composited over the source image with configurable global opacity.
- Annotations for each layer are stored as independent 1-bit binary PNG masks:
  `annotations/<image_stem>_<layer_name>.png` — pixel `0` = unlabeled, `> 0` = annotated.
- Per-image metadata (time, pixel counts, correction metrics) is stored as JSON in `annotations/<image_stem>.metadata`.

---

## Annotation Tools

| Tool | Key | Description |
|------|-----|-------------|
| **Pen** | `P` | Freehand drawing with a configurable circular brush. |
| **Eraser** | `E` | Removes pixels from the active layer. |
| **Selector** (magic wand) | `S` | Flood-fills connected regions by color similarity. The selection threshold is adjustable with real-time preview before committing. |
| **Fill** | `F` | Fills a connected region or the entire active layer. |

All tools support **undo/redo** (`Ctrl+Z` / `Ctrl+Y`) with a history of up to 20 states per image.

### Selector detail

Click to seed a region; adjust the threshold with `+` / `-` while observing the preview; press `Enter` to commit or `Esc` to cancel. Expand or contract the selection boundary with `E` / `R`.

---

## Keyboard Shortcuts

### Tools

| Key | Action |
|-----|--------|
| `P` | Pen tool |
| `E` | Eraser tool |
| `S` | Selector tool |
| `F` | Fill tool |
| `+` / `-` | Increase / decrease brush or threshold size |

### Edit

| Key | Action |
|-----|--------|
| `Ctrl+Z` | Undo |
| `Ctrl+Y` / `Ctrl+Shift+Z` | Redo |

### View

| Key | Action |
|-----|--------|
| `Space` (hold) | Hide all annotation overlays |
| `I` | Toggle source image visibility |
| `M` | Toggle missing-pixels overlay (highlights unannotated pixels) |
| `G` | Toggle pixel grid |
| `H` | Show help window |

### Layers

| Key | Action |
|-----|--------|
| `1`–`9` | Select annotation layer |
| `V` | Toggle all layers visibility |
| `L` | Toggle all layers lock |
| `Ctrl+1`–`9` | Toggle visibility of layer N |
| `Alt+1`–`9` | Toggle lock of layer N |

### Navigation

| Input | Action |
|-------|--------|
| Scroll wheel | Vertical scroll |
| `Shift` + scroll | Horizontal scroll |
| `Ctrl` + scroll | Zoom in / out |

---

## Visualization Features

- **Coverage indicator**: real-time percentage of annotated pixels per layer and globally.
- **Missing-pixels overlay**: highlights pixels unannotated in any layer (`M`).
- **Image gallery**: thumbnail panel showing all project images with per-image coverage.
- **Statistics panel**: session metrics including time per layer and pixel counts.
- **Grid overlay**: configurable pixel grid for precise editing (`G`).

---

## Logging and Statistics

Every image has an append-only JSON log at `annotations/logs/<image_stem>.json`. The log records every user action with a UTC timestamp, enabling full reconstruction of annotation sessions:

| Event | Recorded fields |
|-------|----------------|
| `image_open` / `image_close` | image dimensions, layer count |
| `pen_stroke` / `erase_stroke` | active layer, pixels modified |
| `selector_commit` / `fill_commit` | active layer, pixels added |
| `autolabel_start` / `autolabel_end` | plugin id, duration |

Metadata files additionally store cumulative **time per layer**, **pixel addition/deletion counts per layer**, and **post-autolabel correction metrics** (operations, pixels changed, per-layer breakdown).

### AI statistics

When a plugin runs, two metrics quantify prediction quality and the human effort spent correcting it ([src/application/autolabel_service.py](src/application/autolabel_service.py), [src/application/autolabel_metrics.py](src/application/autolabel_metrics.py)):

- **Prediction Confidence Ratio (PCR)** — derived from the plugin's confidence map. The per-pixel confidences in `[0, 1]` are split into four equal bins; `PCR = (H1 + H4) / (H2 + H3 + ε)`, so a higher value means predictions are more decisive (concentrated near 0 or 1) rather than uncertain. Only produced when the plugin returns a confidence map.
- **Correction Rate (HCR)** — the fraction of pixels changed relative to the plugin's post-autolabel snapshot: `HCR = |snapshot ⊕ current| / N × 100`. `0%` means the prediction was accepted unchanged; `100%` means every pixel was corrected.

---

## ML Plugin System

PixelSmart integrates ML-based pre-annotation through an extensible plugin architecture. All plugins subclass the `AutolabelPlugin` abstract base class ([src/application/plugin_base.py](src/application/plugin_base.py)):

```python
class AutolabelPlugin(ABC):
    @property @abstractmethod
    def id(self) -> str: ...

    @property @abstractmethod
    def display_name(self) -> str: ...

    @property @abstractmethod
    def supported_layers(self) -> list:
        """Ordered layer names; each name's index is its value in the label map."""

    @abstractmethod
    def run(self, image: np.ndarray) -> tuple[np.ndarray, "np.ndarray | None"]:
        """Given a BGR image (H, W, C), return (label_map, confidence_map).

        label_map      — (H, W) array of indices into supported_layers.
        confidence_map — optional (H, W) float32 array in [0, 1], or None.
        """

    # Optional fine-tuning interface (default: not supported)
    @property
    def can_fine_tune(self) -> bool: return False

    def list_versions(self) -> list[dict]: return []  # {"label", "date", "is_original"}

    def fine_tune(self, images_and_annotations: list[dict]) -> None:
        """Each item: {"image": BGR ndarray, "annotations": list[ndarray] masks}.

        Default raises NotImplementedError. Called synchronously from a background thread.
        """
```

The input `image` is in **BGR** order (OpenCV convention). `run()` returns a **tuple**: an `(H, W)` label map plus an optional per-pixel confidence map (used for the AI statistics below).

Plugins are **auto-discovered** at startup by scanning `src/plugins/` ([src/application/plugin_manager.py](src/application/plugin_manager.py)). A plugin directory needs only an `__init__.py`. Discovery checks for a module-level `get_plugins()` factory first (for registering multiple instances); if absent, it falls back to the first concrete `AutolabelPlugin` subclass it finds. All discovered plugins are offered regardless of their layer names — compatibility is determined by the user-configured layer mapping, not by name matching.

### Included plugin: `onnx_tiled`

Loads one binary ONNX segmentation model per layer (`<layer>.onnx`) from a subfolder under `src/plugins/onnx_tiled/onnx/<model_name>/`. Each model subfolder yields a separate plugin instance via `get_plugins()`. Inference runs on 256×256 patches with a `_MARGIN = 5` px border discarded from each interior patch edge, giving a stride of `256 − 2·5 = 246` (10 px overlap between adjacent patches) before the per-patch probabilities are averaged. The final label for each pixel is the argmax across all per-layer probability maps; the winning probability is returned as the confidence map.

### Layer mapping and conflict resolution

The plugin configuration dialog ([src/presentation/model_config_dialog.py](src/presentation/model_config_dialog.py)) maps model output layers to application layers. Two conflict resolution strategies are available:

- **Argmax** ("Highest model confidence"): the layer with maximum model confidence wins.
- **Layer priority**: the highest-priority layer exceeding a 0.5 probability threshold is selected; pixels below threshold fall back to argmax. Priorities are assigned uniquely per layer.

---

## Fine-Tuning in the Active Learning Loop

The `fine_tune()` / versioning interface lets a plugin participate in an active-learning loop where newly corrected annotations feed back into the model. The application provides the workflow and threading; a plugin that sets `can_fine_tune = True` implements the training itself.

1. A dedicated modal dialog ([src/presentation/fine_tune_selection_dialog.py](src/presentation/fine_tune_selection_dialog.py)) shows a thumbnail grid of the project's **fully annotated images** (100% coverage — every pixel labeled in at least one layer). Only these are selectable, and the *Fine-tune* button stays disabled until at least one image is chosen.
2. The selected images and their per-layer masks are remapped into the plugin's `supported_layers` order and passed as a `list[dict]` to the plugin's `fine_tune()` method on a **background `QThread`** (`_FineTuneWorker`, [src/presentation/main_window.py](src/presentation/main_window.py)), keeping the UI responsive. Completion (or failure) is reported back via a signal.
3. **Versioning contract**: a plugin may create new versioned model directories (`fine_tuned/v2/`, `fine_tuned/v3/`, …). `v1` refers to the original model files in the model root and is always preserved.
4. `list_versions()` reports each version's label and timestamp; all versions are selectable from the plugin configuration dialog and applied via `set_active_version()` for subsequent inference.

---

## Rendering Backend

| Backend | `config.toml` value | Description |
|---------|--------------------|-|
| OpenGL 3.3 Core Profile | `"gl"` | GPU-accelerated via `QOpenGLWidget`. GLSL shader compositing, lazy texture uploads, GPU-side grid rendering. Recommended for large images. |
| Qt software renderer | `"qt"` | CPU-based via `QGraphicsView`. No GPU requirement; suitable for virtualized environments. |

```toml
# config.toml
[viewer]
backend = "gl"   # or "qt"
```

The backend is also configurable per project and persisted in the project configuration.

---

## Output Format Summary

| Path | Content |
|------|---------|
| `annotations/<stem>_<layer>.png` | 1-bit binary mask for one layer (0 = unlabeled) |
| `annotations/<stem>.metadata` | JSON: time per layer, pixel counts, correction metrics |
| `annotations/logs/<stem>.json` | Append-only JSON action log |
| `.pixelsmart/project.json` | Project configuration (layers, tool state, plugin config) |
| `~/.pixelsmart/app.json` | Global app configuration (last project, preferences) |
