"""
Export a YOLO .pt model from yolo_models/ to OpenVINO in openvino_models/.
Run from project root: python export_pt_to_openvino.py [model_name.pt]
"""
import os
import sys

# project root = this script's directory
ROOT = os.path.dirname(os.path.abspath(__file__))
YOLO_DIR = os.path.join(ROOT, "yolo_models")
OV_DIR = os.path.join(ROOT, "openvino_models")


def main():
    if len(sys.argv) >= 2:
        pt_name = sys.argv[1]
    else:
        # default: sack_person_model_best.pt if present, else first .pt in yolo_models
        default = "sack_person_model_best.pt"
        path_default = os.path.join(YOLO_DIR, default)
        if os.path.isfile(path_default):
            pt_name = default
        else:
            try:
                pt_name = next(
                    f for f in os.listdir(YOLO_DIR)
                    if f.endswith(".pt")
                )
            except StopIteration:
                print("No .pt in yolo_models/. Put a model there or run: python export_pt_to_openvino.py path/to/model.pt")
                sys.exit(1)

    if not pt_name.endswith(".pt"):
        print("Expected a .pt file. Got:", pt_name)
        sys.exit(1)

    pt_path = os.path.join(YOLO_DIR, pt_name) if not os.path.isabs(pt_name) else pt_name
    if not os.path.isfile(pt_path):
        pt_path = pt_name
    if not os.path.isfile(pt_path):
        print("Model not found:", pt_path)
        sys.exit(1)

    from ultralytics import YOLO

    os.makedirs(OV_DIR, exist_ok=True)
    import shutil
    # Ultralytics exports to a folder next to the model; we then copy to openvino_models
    print("Loading", pt_path)
    model = YOLO(pt_path)
    # export creates <model_name>_openvino_model/ in same dir as .pt
    export_path = model.export(format="openvino")
    print("Exported to:", export_path)
    # export_path is the directory path (e.g. .../yolo_models/sack_person_model_best_openvino_model)
    src_dir = export_path if os.path.isdir(export_path) else os.path.dirname(export_path)
    base = os.path.splitext(os.path.basename(pt_path))[0]
    dest_folder = os.path.join(OV_DIR, base + "_openvino")
    if os.path.isdir(dest_folder):
        shutil.rmtree(dest_folder, ignore_errors=True)
    shutil.copytree(src_dir, dest_folder)
    print("Copied to:", dest_folder)


if __name__ == "__main__":
    main()
