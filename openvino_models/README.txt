OpenVINO model folders

  Each subfolder here should contain an OpenVINO export: .xml and .bin (e.g. openvino_models/yolo26n_openvino/yolo26n.xml).

  How to populate:
    1) Export from .pt: put .pt in yolo_models/, then from project root run:
         python export_pt_to_openvino.py [model_name.pt]
       The script writes to openvino_models/<model_stem>_openvino/.
    2) Or copy an existing OpenVINO export folder (with .xml + .bin) into openvino_models/.

  Python OpenVINO: use --model <folder_name> (e.g. yolo26n_openvino or just yolo26n).
  C++ OpenVINO:   pass the path to the .xml, e.g. openvino_models/yolo26n_openvino/<name>.xml
                  (paths can be relative to where you run the executable or absolute).

  Project root = the directory that contains openvino_models/, cpp_openvino/, output/, etc.
