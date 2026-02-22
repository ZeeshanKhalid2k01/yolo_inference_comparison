C++ + OpenVINO — video detection

  Runs YOLO via OpenVINO in C++: video in -> detection video out. Supports both raw format
  (e.g. YOLO11n: [1, 84, S] with decode + NMS) and end2end format (e.g. YOLO26n: [1, N, 6] or [1, 6, N],
  NMS-free). Output naming and overlay (FPS, CPU%%, RAM, Mem, cores) match the Python scripts.

  Requirements
    - OpenVINO: set OPENVINO_DIR to <OpenVINO_install>/runtime/cmake; add runtime bin and TBB to PATH.
    - OpenCV (C++): e.g. vcpkg install opencv:x64-windows, or system install.
    - CMake 3.20+, C++17 compiler.

  Project root = the directory that contains cpp_openvino/, openvino_models/, output/, etc.
  The executable infers project root from its path so that output/ is the same as for Python.

  Build (from project root; adjust generator and vcpkg path for your system)
    cd cpp_openvino
    mkdir build
    cd build
    cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=<path_to_vcpkg>/scripts/buildsystems/vcpkg.cmake ..
    cmake --build . --config Release

  Executable location: cpp_openvino/build/Release/video_detect_ov.exe (or your build dir).
  Run from a shell where OpenVINO and OpenCV DLLs are on PATH (or next to the exe).

  Run
    video_detect_ov.exe <input_video> <model.xml> [output_video] [imgsz] [conf] [iou] [max_det] [pre_topk] [overlay]

    output_video: path, or "-" / "auto" -> <project_root>/output/<video_stem>_<model_stem>_ov_cpp_<timestamp>.mp4
    overlay: "1" or "show" to draw FPS, CPU%%, RAM, Mem, cores (omit for max FPS).

  Examples (paths relative to project root or absolute)
    # Auto output path + overlay
    video_detect_ov.exe path/to/traffic.mp4 openvino_models/yolo26n_openvino/yolo26n.xml - 640 0.35 0.45 50 200 show

    # Explicit output path
    video_detect_ov.exe path/to/traffic.mp4 openvino_models/yolo26n_openvino/yolo26n.xml output/out_cpp.mp4

    # Minimal: input, model, output
    video_detect_ov.exe path/to/traffic.mp4 openvino_models/yolo11n_openvino/yolo11n.xml output/run.mp4

  Model: use the .xml from a folder under openvino_models/ (create with python export_pt_to_openvino.py from project root).
  Raw vs end2end: the app detects output shape and uses the correct decode path (see main README for YOLO26 NMS-free).
