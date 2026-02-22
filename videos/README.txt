Input videos (optional)

  You can place input videos here for convenience, but it is not required.
  All scripts and the C++ executable accept any path to a video file (e.g. path/to/input.mp4).

  Example: if you put traffic.mp4 here, from project root you can run:
    python python_yolo/run_video_detect.py videos/traffic.mp4
    python python_openvino/run_video_detect.py videos/traffic.mp4
    cpp_openvino/build/Release/video_detect_ov.exe videos/traffic.mp4 openvino_models/<model>/<name>.xml - 640 0.35 0.45 50 200 show

  Project root = the directory that contains videos/, output/, python_yolo/, etc.
