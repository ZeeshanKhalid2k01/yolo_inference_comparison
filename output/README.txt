Output videos and traces

  Default folder for detection output videos from:
    - Python YOLO (.pt)     -> output/<video_stem>_<model_stem>_pt_<timestamp>.mp4
    - Python OpenVINO       -> output/<video_stem>_<model_stem>_ov_py_<timestamp>.mp4
    - C++ OpenVINO          -> output/<video_stem>_<model_stem>_ov_cpp_<timestamp>.mp4

  Backend suffixes:
    pt    = Python + YOLO (.pt)
    ov_py = Python + OpenVINO
    ov_cpp = C++ + OpenVINO

  Each run can also write a .json with the same base name (backend, model, input_video, output_video,
  conf, imgsz, frame_count, overlay). Use it to trace which model and backend produced each video.

  Project root = the directory that contains output/, python_yolo/, cpp_openvino/, etc.
