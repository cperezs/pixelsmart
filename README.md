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
- AI plugin selection and configuration
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
- Annotations for each layer are stored as independent 8-bit grayscale PNG masks:
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

---

## AI Plugin System

PixelSmart integrates AI-based pre-annotation through an extensible plugin architecture. All plugins subclass the `AutolabelPlugin` abstract base class:

```python
class AutolabelPlugin(ABC):
    @property @abstractmethod
    def id(self) -> str: ...

    @property @abstractmethod
    def display_name(self) -> str: ...

    @property @abstractmethod
    def supported_layers(self) -> list[str]: ...

    @abstractmethod
    def run(self, image: np.ndarray) -> np.ndarray:
        """Return an (H, W) label map with integer layer indices."""

    # Optional fine-tuning interface
    @property
    def can_fine_tune(self) -> bool: return False
    def list_versions(self) -> list[dict]: return []
    def fine_tune(self, data: list[dict]) -> None: ...
```

Plugins are **auto-discovered** at startup by scanning `src/plugins/`. A plugin directory needs only an `__init__.py` exporting a concrete subclass (or a `get_plugins()` factory for multiple instances).

### Included plugin: `onnx_tiled`

Loads one binary ONNX segmentation model per layer from a subfolder under `src/plugins/onnx_tiled/onnx/<model_name>/`. Inference is performed on 256×256 patches with a 5-pixel overlap margin. The final label for each pixel is determined by argmax across all per-layer probability maps.

### Layer mapping and conflict resolution

The plugin configuration dialog allows mapping model output layers to application layers. Two conflict resolution strategies are available:

- **Argmax**: the layer with maximum model confidence wins.
- **Layer priority**: the highest-priority layer exceeding a 0.5 probability threshold is selected; pixels below threshold fall back to argmax.

---

## Fine-Tuning in the Active Learning Loop

1. Open the fine-tuning dialog and select **fully annotated images** (100% coverage) from the project gallery.
2. The selected images and masks are passed to the active plugin's `fine_tune()` method in a **background thread**, keeping the UI responsive.
3. The plugin saves training data and notifies the application on completion.
4. Each run creates a new versioned model directory (`fine_tuned/v2/`, `fine_tuned/v3/`, …). The original model is always preserved as `v1`.
5. All versions are selectable from the plugin configuration dialog, showing version label and creation timestamp.

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

## Web Service Mode

An optional FastAPI server accepts programmatic annotation requests over HTTP. See [WEB_SERVICE_MODE.md](WEB_SERVICE_MODE.md) for full documentation.

---

## Output Format Summary

| Path | Content |
|------|---------|
| `annotations/<stem>_<layer>.png` | 8-bit grayscale mask for one layer (0 = unlabeled) |
| `annotations/<stem>.metadata` | JSON: time per layer, pixel counts, correction metrics |
| `annotations/logs/<stem>.json` | Append-only JSON action log |
| `.pixelsmart/project.json` | Project configuration (layers, tool state, plugin config) |
| `~/.pixelsmart/app.json` | Global app configuration (last project, preferences) |
