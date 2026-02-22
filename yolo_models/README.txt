YOLO .pt models (PyTorch)

  Put Ultralytics YOLO .pt files here (e.g. yolo11n.pt, yolo26n.pt, or your custom model).

  Then from project root run:
    python export_pt_to_openvino.py
    python export_pt_to_openvino.py <model_name>.pt   # to export a specific model

  This exports to openvino_models/<model_stem>_openvino/ for use with Python OpenVINO and C++ OpenVINO.

  Project root = the directory that contains yolo_models/, openvino_models/, python_yolo/, etc.
