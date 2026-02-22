Python + OpenVINO — video detection

  Runs YOLO via OpenVINO (model from openvino_models/) on a video and writes a detection video.
  Requires: Python venv from project root with pip install -r requirements.txt.
  OpenVINO models: export from .pt with python export_pt_to_openvino.py (see project root README).

  Project root = the directory that contains python_openvino/, openvino_models/, output/, etc.

  Setup (once)
    From project root:
      python -m venv .venv
      .\.venv\Scripts\Activate.ps1   # Windows
      # source .venv/bin/activate    # Linux/macOS
      pip install -r requirements.txt
    Ensure openvino_models/ has at least one model folder (e.g. openvino_models/yolo26n_openvino/ with .xml + .bin).

  Run (from project root, venv active)
    python python_openvino/run_video_detect.py <path/to/input.mp4>
    python python_openvino/run_video_detect.py <path/to/input.mp4> --model yolo26n_openvino
    # Short name also works: --model yolo26n (script looks for yolo26n_openvino or yolo26n_openvino_model)

    With overlay (FPS, CPU%%, RAM, Mem, cores):
      python python_openvino/run_video_detect.py <path/to/input.mp4> --model yolo26n --show-fps --show-cpu --show-ram --show-mem --show-cores

  Options
    --model NAME       folder name in openvino_models/ or path to .xml (default: first folder with .xml)
    --out PATH         full output video path
    --out-name NAME    output filename only (saved under output/)
    --conf 0.35        confidence threshold
    --imgsz 640        inference size
    --max-det 50       max detections per frame

  Default output: output/<video_stem>_<model_stem>_ov_py_<timestamp>.mp4
  A .json with the same base name is written for tracing (backend, model, input, overlay, frame_count).
