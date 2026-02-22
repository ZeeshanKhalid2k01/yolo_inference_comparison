# YOLO Inference Comparison — Benchmarking Toolkit

Benchmark **Ultralytics YOLO** (e.g. YOLO11n, YOLO26n) across **three inference stacks** on the same video: **PyTorch (Python)**, **OpenVINO (Python)**, and **OpenVINO (C++)**. Same model, same hardware — compare FPS, CPU, and RAM.

---

## Repository structure (full hierarchy)

All paths below are relative to the **project root** (the directory containing this README).

```
<project_root>/
├── README.md                 # This file
├── .gitignore
├── requirements.txt          # Python deps for export + Python runners
├── export_pt_to_openvino.py  # Convert .pt → OpenVINO (openvino_models/)
│
├── yolo_models/              # Put Ultralytics .pt models here (e.g. yolo11n.pt, yolo26n.pt)
│   └── README.txt
├── openvino_models/          # OpenVINO exports (.xml + .bin). Created by export script or copy here.
│   └── README.txt
├── videos/                   # Input videos for detection (optional; any path works)
│   └── README.txt
├── output/                   # Default folder for detection output videos + .json traces
│   └── README.txt
│
├── python_yolo/              # 1) Python + YOLO (.pt) — baseline
│   ├── README.txt
│   └── run_video_detect.py
├── python_openvino/          # 2) Python + OpenVINO
│   ├── README.txt
│   └── run_video_detect.py
└── cpp_openvino/             # 3) C++ + OpenVINO — production-style
    ├── CMakeLists.txt
    ├── README.txt
    └── src/
        └── video_detect_ov.cpp
```

**Conventions**

- **Project root** = directory that contains `yolo_models/`, `openvino_models/`, `output/`, `python_yolo/`, `python_openvino/`, `cpp_openvino/`.
- Scripts and the C++ executable resolve paths relative to project root (e.g. `output/` is `<project_root>/output/`).
- No environment or machine-specific files (e.g. `ov_env.ps1`) are included; set OpenVINO/OpenCV as described below.

---

## Pre-dependencies

### For Python (export + Python YOLO + Python OpenVINO)

- **Python** 3.8+
- **pip** and a **virtual environment** (recommended)
- Packages: see `requirements.txt` (opencv-python, numpy, ultralytics, psutil)

### For OpenVINO export and C++ build/run

- **OpenVINO** (e.g. 2024.x or 2025.x): install and set **OPENVINO_DIR** to the runtime CMake path (e.g. `<OpenVINO_install>/runtime/cmake`). Add OpenVINO runtime `bin` and TBB to **PATH** when building and running the C++ app.
- **OpenCV** (C++): required for `cpp_openvino`. E.g. install via vcpkg: `vcpkg install opencv:x64-windows`.
- **CMake** 3.20+
- **C++17** compiler (e.g. Visual Studio 2022 on Windows, GCC/Clang on Linux)

---

## Getting models and videos

If you don’t have `.pt` models or a video yet, use the steps below so you can run the toolkit without guessing.

### Models (.pt → then export to OpenVINO)

You need `.pt` weights in `yolo_models/`. Two ways:

**Option A — Let Ultralytics download, then copy into `yolo_models/`**

From project root (venv active), run once so the model is downloaded to the Ultralytics cache:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
```

Then copy the downloaded `.pt` file(s) from the Ultralytics cache into `yolo_models/`. Cache location is in [Ultralytics docs](https://docs.ultralytics.com) (e.g. under your user dir). After copying you should have e.g. `yolo_models/yolo11n.pt` and `yolo_models/yolo26n.pt`.

**Option B — Download manually**

- Get pre-trained weights from [Ultralytics assets/releases](https://github.com/ultralytics/assets/releases) or the model hub, then place the `.pt` file(s) into `yolo_models/`.

After the `.pt` files are in `yolo_models/`, run:

```bash
python export_pt_to_openvino.py
# or: python export_pt_to_openvino.py yolo11n.pt
#     python export_pt_to_openvino.py yolo26n.pt
```

That creates `openvino_models/yolo11n_openvino/` and `openvino_models/yolo26n_openvino/` (or one of them if you only exported one).

### Videos (input for detection)

Use any MP4 (or other format supported by OpenCV). For a quick test:

1. **Your own file:** Put it anywhere and pass the path, e.g. `python python_yolo/run_video_detect.py D:/videos/traffic.mp4`. Optionally copy it into `videos/` for short paths: `videos/traffic.mp4`.
2. **Sample videos (free):** Download a short clip from [Pexels Videos](https://www.pexels.com/search/videos/traffic/), [Sample Videos](https://sample-videos.com/), or similar. Save (or move) into `videos/`, e.g. `videos/sample.mp4`.
3. **Command-line download (if you have curl):** e.g.  
   `curl -L -o videos/sample.mp4 "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4"`  
   (Replace with any direct URL to an MP4.)

Then run the scripts with that path:

```bash
python python_yolo/run_video_detect.py videos/sample.mp4 --model yolo11n.pt --show-fps --show-cpu --show-ram
```

---

## Setup

### 1. Clone or download this repo

Use the project root as your working directory for all commands below.

### 2. Python environment (for export and Python runners)

From project root:

```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Export .pt → OpenVINO (needed for Python OpenVINO and C++ OpenVINO)

1. Put your `.pt` model(s) in `yolo_models/` (e.g. `yolo11n.pt`, `yolo26n.pt`).
2. From project root:

   ```bash
   python export_pt_to_openvino.py
   # Or: python export_pt_to_openvino.py yolo26n.pt
   ```

   This creates a folder under `openvino_models/` (e.g. `openvino_models/yolo26n_openvino/` with `.xml` and `.bin`).

### 4. C++ build (optional, for C++ OpenVINO benchmark)

1. Set OpenVINO environment (no env file included):
   - **OPENVINO_DIR** = path to OpenVINO’s runtime CMake (e.g. `<OpenVINO_install>/runtime/cmake`).
   - **PATH** must include OpenVINO runtime `bin` and TBB `bin`.
2. Configure and build from project root (example for Windows with vcpkg):

   ```bash
   cd cpp_openvino
   mkdir build
   cd build
   cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=<path_to_vcpkg>/scripts/buildsystems/vcpkg.cmake ..
   cmake --build . --config Release
   ```

   The executable is typically at `cpp_openvino/build/Release/video_detect_ov.exe` (or equivalent on your platform). Run it from a shell where OpenVINO and OpenCV DLLs are on PATH (or next to the exe).

---

## Usage (generic paths)

Use any input video path; outputs go to `output/` by default unless you override.

### 1) Python + YOLO (.pt)

From project root (with venv active):

```bash
python python_yolo/run_video_detect.py <path/to/input.mp4>
# Optional: --model yolo26n.pt --out output/out_yolo.mp4 --conf 0.35 --imgsz 640
# Overlay (FPS, CPU, RAM, etc.): --show-fps --show-cpu --show-ram --show-mem --show-cores
```

Default output: `output/<video_stem>_<model_stem>_pt_<timestamp>.mp4` (+ `.json`).

### 2) Python + OpenVINO

From project root (with venv active):

```bash
python python_openvino/run_video_detect.py <path/to/input.mp4>
# Optional: --model yolo26n_openvino --out output/out_ov.mp4
# Overlay: --show-fps --show-cpu --show-ram --show-mem --show-cores
```

Default output: `output/<video_stem>_<model_stem>_ov_py_<timestamp>.mp4` (+ `.json`).

### 3) C++ + OpenVINO

From a shell where OpenVINO and OpenCV are on PATH (project root or build dir):

```bash
cpp_openvino/build/Release/video_detect_ov.exe <path/to/input.mp4> openvino_models/yolo26n_openvino/<model>.xml [output_video] [imgsz] [conf] [iou] [max_det] [pre_topk] [overlay]
```

- **output_video**: explicit path, or `-` / `auto` → `output/<video_stem>_<model_stem>_ov_cpp_<timestamp>.mp4`.
- **overlay**: pass `show` or `1` to draw FPS, CPU%, RAM, Mem, cores.

Example (auto output + overlay):

```bash
video_detect_ov.exe path/to/traffic.mp4 openvino_models/yolo26n_openvino/yolo26n.xml - 640 0.35 0.45 50 200 show
```

---

## Comparing the three stacks

1. Export OpenVINO: put `.pt` in `yolo_models/`, run `python export_pt_to_openvino.py`.
2. Run **Python YOLO**: get e.g. `output/<name>_yolo26n_pt_<timestamp>.mp4`.
3. Run **Python OpenVINO** or **C++ OpenVINO** on the same video: get `output/<name>_*_ov_py_*.mp4` or `output/<name>_*_ov_cpp_*.mp4`.

Same video + same model → compare FPS and resource usage (overlay or `.json` trace).

---

## YOLO Object Detection — Performance Scorecard

Benchmark: **YOLO11n vs YOLO26n** | **Python vs C++** | **PyTorch vs OpenVINO** | Sorted by FPS ↓  

(Same traffic video, same hardware; overlay metrics enabled.)

| Rank | Model   | Language | Framework      | FPS   | CPU % | RAM (MB) | Mem (MB) | Cores Used |
|------|--------|----------|----------------|-------|-------|----------|----------|------------|
| 1st  | YOLO26n | C++      | OpenVINO (C++) | **23.2** | 33.2% | 267.7    | 300.3    | 4.0 / 12   |
| 2nd  | YOLO11n | C++      | OpenVINO (C++) | **22.7** | 38.9% | 267.6    | 293.0    | 4.7 / 12   |
| 3rd  | YOLO26n | Python   | OpenVINO (py)  | 12.3  | 61.8% | 659.5    | 1064.5   | 7.4 / 12   |
| 4th  | YOLO11n | Python   | OpenVINO (py)  | 10.8  | 65.3% | 663.6    | 1072.9   | 7.8 / 12   |
| 5th  | YOLO11n | Python   | PyTorch        | 7.0   | 65.1% | 394.6    | 849.1    | 7.8 / 12   |
| 6th  | YOLO26n | Python   | PyTorch        | 5.8   | 65.5% | 397.5    | 850.8    | 7.9 / 12   |

**WINNER: C++ + OpenVINO** — ~3× faster FPS • ~40% less CPU • ~3× less RAM vs Python baseline.

---

## YOLO26 and End-to-End NMS-Free Inference

**YOLO26** (and some newer Ultralytics exports) use **End-to-End NMS-Free Inference**: the OpenVINO graph can output **already-decoded boxes** in a fixed-size tensor (e.g. `[1, N, 6]` or `[1, 6, N]` with `x1, y1, x2, y2, score, class_id`), so **no separate NMS step** is required in the app. Older YOLO exports (e.g. **YOLO11**) often use a **raw** format (e.g. `[1, 4+num_classes, S]`) that requires **decode + NMS** in code.

### How the C++ OpenVINO app handles both

The C++ pipeline in `cpp_openvino` supports **both** export types:

1. **Raw format** (e.g. YOLO11n): output shape `[1, 4+num_classes, S]`.
   - Uses `decode_yolo_raw()` → top-k prefilter → **NMS** → draw.
   - **Pros**: Works with older/smaller models; NMS is under your control (e.g. tune IoU).
   - **Cons**: Extra decode + NMS cost on CPU; more code to maintain.

2. **End-to-end format** (e.g. YOLO26n): output shape `[1, N, 6]` or `[1, 6, N]`.
   - Uses `decode_yolo_end2end()` only (no NMS); boxes are already filtered by the model.
   - **Pros**: Simpler, faster deployment; no NMS tuning; often better for latency.
   - **Cons**: Fixed max detections and behavior defined by the exported graph; less flexibility than custom NMS.

The app **auto-detects** format from the output shape and chooses the correct decode path. So you can benchmark **YOLO11n** (raw + NMS) and **YOLO26n** (end2end) with the same binary; the READMEs in `cpp_openvino/` and `openvino_models/` describe where models live and how to run them with generic paths.

---

## Output naming and tracing

- **output/** holds detection videos and, when generated, a `.json` file with the same base name (backend, model, input path, conf, imgsz, frame_count, overlay flags). Use it to trace which run produced each file.
- Backend suffixes: `_pt_` (Python .pt), `_ov_py_` (Python OpenVINO), `_ov_cpp_` (C++ OpenVINO). See `output/README.txt` for details.
