# `run_sam2.ipynb` — Pipeline Documentation

## Overview

`run_sam2.ipynb` is the **main inference and optimisation notebook** for the SAM 3D multi-person body reconstruction pipeline. It takes an input image, detects all people in it, reconstructs their 3-D body meshes, generates contact constraints between them using an LLM vision model, and refines the meshes through a physics-aware optimisation loop.

---

## Environment Requirements

| Requirement | Details |
|---|---|
| **Conda env** | `sam_3d_body` |
| **Python** | 3.11 |
| **GPU** | CUDA-capable (falls back to CPU) |
| **Key packages** | `torch`, `ultralytics`, `trimesh`, `cv2`, `yacs` |
| **Checkpoints** | `E:\prosepose-main\checkpoints\sam-3d-body-dinov3\` |

---

## Pipeline Diagram

```
Input Image
    │
    ▼
[Cell 1] Imports & Environment Setup
    │
    ▼
[Cell 2] Model Loading (SAM3D + YOLOv8)
    │
    ▼
[Cell 3] Person Detection → 3D Mesh Inference
    │
    ▼
[Cell 4] Save results → humans.pkl
    │
    ▼
[Cell 5] Inline Interactive 3D Viewer (GLB)
    │
    ▼
[Cell 6] LLM Contact Constraint Generation
         (OpenRouter → gpt_constraints.json → gpt_programs.json)
    │
    ▼
[Cell 7] Fast Parametric Optimisation (V4)
         Contact Loss + Repulsion Loss + Regularisation
    │
    ▼
[Cell 8+] Save Refined Meshes → humans_refined.pkl
          Export Comparison GLBs
```

---

## Cell-by-Cell Reference

### Cell 1 — Imports & Environment Setup

**What it does:**
- Imports all required libraries: `os`, `sys`, `cv2`, `torch`, `numpy`, `matplotlib`, `trimesh`, `base64`, `IPython.display`, `ultralytics`, and local SAM3D modules.
- Removes the `PYOPENGL_PLATFORM` environment variable for Windows compatibility.
- Adds the current working directory to `sys.path` so local modules (`sam_3d_body`, `tools`) can be imported.
- Detects and stores the compute device (`cuda` if available, otherwise `cpu`).

**Key variables set:**
```python
device = "cuda"  # or "cpu"
```

---

### Cell 2 — Model Loading (SAM3D + YOLOv8)

**What it does:**
- Loads the SAM3D model from checkpoint files:
  - `model.ckpt` — main backbone (DINOv3-based)
  - `mhr_model.pt` — Multi-Human Reconstruction head weights
- Wraps the model in `SAM3DBodyEstimator` (the inference API).
- Loads `YOLOv8-nano` (`yolov8n.pt`) for fast multi-person detection.

> ⚠️ **Note:** Several `UserWarning` messages about missing state-dict keys are printed during loading. These are expected and do **not** affect inference — the keys in question are not used by this pipeline.

**Key variables set:**
```python
estimator   # SAM3DBodyEstimator instance
yolo_model  # ultralytics YOLO instance
```

---

### Cell 3 — Detection & 3D Inference

**What it does:**
1. Reads the image at `test_image_path`.
2. Runs YOLOv8 to find all `person` bounding boxes with confidence > 0.5.
3. For each detected person, calls `SAM3DBodyEstimator` to produce a full SMPL-X reconstruction containing:

| Key | Description |
|---|---|
| `pred_vertices` | 3D mesh vertex positions (N × 3) |
| `pred_cam_t` | Camera translation vector |
| `body_pose_params` | SMPL-X body pose (joint rotations) |
| `hand_pose_params` | Hand pose parameters |
| `shape_params` | Body shape (beta vector) |
| `scale_params` | Global scale factor |
| `expr_params` | Facial expression parameters |
| `focal_length` | Estimated camera focal length |
| `global_rot` | Global body orientation |

4. Renders a 2-D mesh overlay on the original image for preview.

**Key variables set:**
```python
outputs        # List of per-person inference dicts
faces          # SMPL-X triangle face connectivity (F × 3)
```

---

### Cell 4 — Persistence (Save `humans.pkl`)

**What it does:**
- Converts all tensor values in `outputs` to numpy arrays.
- Serialises the list to `humans.pkl` alongside the `faces` array using Python `pickle`.

**Why this matters:**
- `humans.pkl` is the **data handoff file** shared with `run_sam3d_integrated.ipynb` and the optimisation cells below.
- Saving it allows you to restart the notebook from Cell 5 onwards without re-running expensive inference.

**Output files:**
```
E:\prosepose-main\sam3d\sam-3d-body-main\humans.pkl
```

---

### Cell 5 — Inline 3-D Viewer

**What it does:**
- Builds a `trimesh.Scene` containing one coloured mesh per detected person.
- Applies a **180° X-axis rotation** to convert from OpenCV image coordinates to the OpenGL/GLB viewer convention.
- Exports the scene to GLB (binary glTF) in memory, base64-encodes it, and injects it into a `<model-viewer>` HTML widget — fully interactive (rotate, zoom, pan) inside the notebook.
- Also saves `combined_humans.glb` to disk.

**Colour coding:**
| Person | Colour |
|--------|--------|
| Person 0 | Sky Blue `#4A90FF` |
| Person 1 | Sunset Orange `#FF9D4A` |
| Person 2+ | Mint Green `#64FF96` |

**Output files:**
```
E:\prosepose-main\sam3d\sam-3d-body-main\combined_humans.glb
```

---

### Cell 6 — Contact Constraint Generation (LLM)

**What it does:**
1. Encodes the input image as base64.
2. Calls the **OpenRouter API** using the `meta-llama/llama-3.2-11b-vision-instruct:free` model.
3. The LLM analyses the image and describes physical contact between people (e.g. *"Person 1 is shaking Person 2's right hand"*).
4. The structured response is written to `gpt_constraints.json`.
5. `convert_tables_to_programs.py` (ProsePose utility) translates the JSON into executable PyTorch loss-function snippets and writes them to `gpt_programs.json`.

**Data flow:**
```
input_image
  → OpenRouter API (LLaMA 3.2 Vision)
  → gpt_constraints.json
  → convert_tables_to_programs.py
  → gpt_programs.json  (list of {code, description} dicts)
```

**Configuration:**
```python
OPENROUTER_API_KEY = "sk-or-v1-..."   # Your API key
USE_MOCK = False                        # Set True to skip the API call
```

**Output files:**
```
E:\prosepose-main\sam3d\sam-3d-body-main\gpt_constraints.json
E:\prosepose-main\sam3d\sam-3d-body-main\gpt_programs.json
```

---

### Cell 7 — Fast Parametric Optimisation (V4)

**What it does:**
- Loads `humans.pkl` and `gpt_programs.json`.
- Sets up differentiable SMPL-X parameters (`p0`, `p1`) with `requires_grad=True` for both detected people.
- Runs an **Adam optimiser** for 100 steps with three loss components:

| Loss | Purpose | Weight |
|------|---------|--------|
| **Contact Loss** | Attracts body parts described by the LLM constraints | From `gpt_programs.json` |
| **Repulsion Loss** | Prevents mesh interpenetration (8 cm threshold) | `50.0` |
| **Regularisation** | Keeps pose close to the original SAM3D estimate | `0.1` |

- Displays an interactive 3-D viewer every 30 steps so you can watch the meshes converge.
- After optimisation, regenerates final vertex positions and applies the coordinate flip.

**Key parameters:**
```python
optimizer = Adam([p0['pose'], p0['transl'], p0['rot'],
                  p1['pose'], p1['transl'], p1['rot']], lr=0.01)
steps = 100
```

---

### Cell 8 — Save Refined Results

**What it does:**
- Copies optimised parameters back into `humans_refined` (a deep copy of the original `humans` list).
- Saves `humans_refined.pkl` — the final output for downstream use.
- Exports two comparison GLB files:
  - `initial_human.glb` — meshes before optimisation
  - `refined_human.glb` — meshes after optimisation

**Output files:**
```
E:\prosepose-main\humans_refined.pkl
E:\prosepose-main\outputs\initial_human.glb
E:\prosepose-main\outputs\refined_human.glb
```

---

## Data Files Reference

| File | Role |
|------|------|
| `humans.pkl` | Raw SAM3D inference output (input to optimisation) |
| `humans_refined.pkl` | Optimised mesh parameters (final output) |
| `gpt_constraints.json` | LLM-generated contact description |
| `gpt_programs.json` | Executable PyTorch contact-loss snippets |
| `combined_humans.glb` | Combined 3D scene (pre-optimisation viewer) |
| `outputs/initial_human.glb` | Pre-optimisation comparison mesh |
| `outputs/refined_human.glb` | Post-optimisation comparison mesh |
| `yolov8n.pt` | YOLOv8-nano weights for person detection |

---

## Running the Notebook

1. **Activate the environment:**
   ```bash
   conda activate sam_3d_body
   ```

2. **Start Jupyter:**
   ```bash
   jupyter notebook run_sam2.ipynb
   ```

3. **Set the input image path** in Cell 3:
   ```python
   test_image_path = r"C:\path\to\your\image.png"
   ```

4. **Run all cells top-to-bottom** (Kernel → Restart & Run All).

5. To **skip re-inference**, run only Cells 5–8 after a previous run has saved `humans.pkl`.

---

## Troubleshooting

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `ModuleNotFoundError: openrouter_constraints` | Module not on `sys.path` | Cell 6 auto-adds `prosepose-main` to path — ensure the file exists at `E:\prosepose-main\prosepose-main\openrouter_constraints.py` |
| `NameError: json is not defined` | Cell 6 missing import | Already fixed — `import json` is included in Cell 6 |
| `Invalid magic number; corrupt file?` | `humans.pkl` saved with `torch.save` but loaded with `pickle.load` | Cell 7 now uses `torch.load` with pickle fallback |
| `KeyError: 0` | `humans` is a dict, not a list | Cell 7 normalises dict → list automatically |
| Models appear upside down | Missing or incorrect axis flip | Remove the `rot = rotation_matrix(180°, [1,0,0])` line in the viewer |
