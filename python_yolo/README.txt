Python + YOLO (.pt) — video detection

  Runs Ultralytics YOLO from a .pt model on a video and writes a detection video.
  Requires: Python venv from project root with pip install -r requirements.txt.

  Project root = the directory that contains python_yolo/, yolo_models/, output/, etc.

  Setup (once)
    From project root:
      python -m venv .venv
      .\.venv\Scripts\Activate.ps1   # Windows
      # source .venv/bin/activate    # Linux/macOS
      pip install -r requirements.txt
    Put a .pt model in yolo_models/ (e.g. yolo11n.pt, yolo26n.pt).

  Run (from project root, venv active)
    python python_yolo/run_video_detect.py <path/to/input.mp4>
    python python_yolo/run_video_detect.py <path/to/input.mp4> --model yolo26n.pt

    With overlay (FPS, CPU%%, RAM, Mem, cores):
      python python_yolo/run_video_detect.py <path/to/input.mp4> --model yolo26n.pt --show-fps --show-cpu --show-ram --show-mem --show-cores

  Options
    --model MODEL.pt    .pt name in yolo_models/ or full path (default: first .pt in yolo_models/)
    --out PATH         full output video path
    --out-name NAME    output filename only (saved under output/)
    --conf 0.35        confidence threshold
    --imgsz 640        inference size
    --max-det 50       max detections per frame

  Default output: output/<video_stem>_<model_stem>_pt_<timestamp>.mp4
  A .json with the same base name is written for tracing (backend, model, input, conf, overlay, frame_count).
