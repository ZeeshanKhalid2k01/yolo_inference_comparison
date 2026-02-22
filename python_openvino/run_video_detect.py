"""
Run YOLO OpenVINO detection on a video and save the detection video.
Usage: python run_video_detect.py <input_video> [--model openvino_folder] [--out OUTPUT.mp4] [--out-name NAME] [--conf 0.35] [--imgsz 640]
Output: use --out for full path, or --out-name for filename in output/; default adds timestamp to avoid overwriting.
Optional overlay: --show-fps --show-cpu --show-ram --show-mem --show-cores (CPU: system %%; cores: used/total).
"""
import os
import sys
import argparse
import json
import threading
import time
from datetime import datetime
import cv2
import numpy as np

# project root: parent of python_openvino
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OV_DIR = os.path.join(ROOT, "openvino_models")
OUT_DIR = os.path.join(ROOT, "output")

sys.path.insert(0, ROOT)


class MetricsCollector:
    """Background thread samples process CPU/RAM/mem; main thread reads last value (no blocking)."""

    def __init__(self, interval=0.3):
        self.interval = interval
        self._cpu = 0.0
        self._ram_mb = 0.0
        self._mem_mb = 0.0  # virtual memory
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._process = None

    def _run(self):
        try:
            import psutil
            self._process = psutil.Process()
        except Exception:
            return
        while not self._stop.wait(self.interval):
            try:
                with self._lock:
                    self._cpu = self._process.cpu_percent()
                    mi = self._process.memory_info()
                    self._ram_mb = mi.rss / (1024 * 1024)
                    self._mem_mb = mi.vms / (1024 * 1024)
            except Exception:
                pass

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get(self):
        with self._lock:
            return self._cpu, self._ram_mb, self._mem_mb


def draw_overlay(vis, lines, x=10, y=24, font_scale=0.55, thickness=1):
    """Draw text lines at top-left; semi-transparent background."""
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = 4
    max_w = max(cv2.getTextSize(l, font, font_scale, thickness)[0][0] for l in lines)
    h_line = int(22 * font_scale + 4)
    block = (max_w + 2 * pad, len(lines) * h_line + 2 * pad)
    overlay = vis.copy()
    cv2.rectangle(overlay, (x, y), (x + block[0], y + block[1]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
    for i, line in enumerate(lines):
        cv2.putText(vis, line, (x + pad, y + pad + (i + 1) * h_line - 4),
                    font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="YOLO OpenVINO video detection")
    ap.add_argument("video", nargs="?", help="Input video path")
    ap.add_argument("--model", "-m", default="", help="OpenVINO folder name in openvino_models/ or path to .xml")
    ap.add_argument("--out", "-o", default="", help="Full output video path (overrides --out-name)")
    ap.add_argument("--out-name", default="", help="Output filename only (saved under output/); default uses timestamp to avoid overwriting")
    ap.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference size")
    ap.add_argument("--max-det", type=int, default=50, help="Max detections per frame")
    ap.add_argument("--show-fps", action="store_true", help="Overlay inference FPS on video")
    ap.add_argument("--show-cpu", action="store_true", help="Overlay process CPU%% (background thread)")
    ap.add_argument("--show-ram", action="store_true", help="Overlay process RAM (RSS) in MB")
    ap.add_argument("--show-mem", action="store_true", help="Overlay process virtual memory in MB")
    ap.add_argument("--show-cores", action="store_true", help="Overlay cores used/total")
    args = ap.parse_args()

    if not args.video or not os.path.isfile(args.video):
        print("Provide an input video path.")
        ap.print_help()
        sys.exit(1)

    # resolve OpenVINO model path (folder with .xml or path to .xml)
    if args.model:
        p = args.model
        if os.path.isabs(p):
            if p.endswith(".xml") and os.path.isfile(p):
                model_path = p
            elif os.path.isdir(p):
                model_path = p
                for f in os.listdir(p):
                    if f.endswith(".xml"):
                        model_path = os.path.join(p, f)
                        break
            else:
                model_path = p
        else:
            folder = os.path.join(OV_DIR, p)
            if not os.path.isdir(folder) and not os.path.isfile(folder):
                # Try common suffix so e.g. --model yolo11n finds yolo11n_openvino_model
                for suffix in ("_openvino_model", "_openvino"):
                    alt = os.path.join(OV_DIR, p + suffix)
                    if os.path.isdir(alt):
                        folder = alt
                        break
            if os.path.isdir(folder):
                for f in os.listdir(folder):
                    if f.endswith(".xml"):
                        model_path = os.path.join(folder, f)
                        break
                else:
                    model_path = os.path.join(folder, p + ".xml")
            else:
                model_path = os.path.join(OV_DIR, p)
    else:
        # default: first folder in openvino_models that contains .xml
        candidates = []
        for name in os.listdir(OV_DIR):
            path = os.path.join(OV_DIR, name)
            if os.path.isdir(path) and any(f.endswith(".xml") for f in os.listdir(path)):
                candidates.append(os.path.join(path, next(f for f in os.listdir(path) if f.endswith(".xml"))))
        if not candidates:
            print("No OpenVINO model in openvino_models/. Run export_pt_to_openvino.py first or use --model path.")
            sys.exit(1)
        model_path = candidates[0]
    if not os.path.isfile(model_path):
        print("OpenVINO model .xml not found:", model_path)
        sys.exit(1)
    # Ultralytics YOLO can load OpenVINO by folder path (dir containing .xml)
    model_dir = os.path.dirname(model_path)
    model_folder = os.path.basename(model_dir)
    model_stem = model_folder.replace("_openvino_model", "").replace("_openvino", "") if model_folder else "openvino"

    # output path: --out (full path) > --out-name (filename in output/) > default with timestamp
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.out:
        out_path = args.out
    elif args.out_name:
        name = args.out_name.strip()
        if not name.endswith(".mp4"):
            name += ".mp4"
        out_path = os.path.join(OUT_DIR, name)
    else:
        base = os.path.splitext(os.path.basename(args.video))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(OUT_DIR, f"{base}_{model_stem}_ov_py_{ts}.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    from ultralytics import YOLO

    show_any = args.show_fps or args.show_cpu or args.show_ram or args.show_mem or args.show_cores
    metrics = MetricsCollector(interval=0.3) if show_any else None
    if metrics:
        metrics.start()
    try:
        import psutil
        num_cores = psutil.cpu_count() or 0
    except Exception:
        num_cores = 0

    print("Model:", model_dir)
    print("Input:", args.video)
    print("Output:", out_path)
    if show_any:
        print("Overlay: fps=%s cpu=%s ram=%s mem=%s cores=%s" % (args.show_fps, args.show_cpu, args.show_ram, args.show_mem, args.show_cores))
    # Ultralytics detects OpenVINO only when path is the model dir (name contains _openvino_model), not the .xml file
    model = YOLO(model_dir, task="detect")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Cannot open video:", args.video)
        sys.exit(1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_idx = 0
    fps_window = []
    fps_window_size = 30
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                max_det=args.max_det,
                verbose=False,
            )[0]
            vis = results.plot()
            if vis is None:
                vis = frame.copy()

            if show_any:
                lines = []
                if args.show_fps:
                    t1 = time.perf_counter()
                    fps_window.append(t1)
                    if len(fps_window) > fps_window_size:
                        fps_window.pop(0)
                    if len(fps_window) >= 2:
                        elapsed = fps_window[-1] - fps_window[0]
                        inf_fps = (len(fps_window) - 1) / elapsed if elapsed > 0 else 0
                        lines.append("FPS: %.1f" % inf_fps)
                    else:
                        lines.append("FPS: --")
                if args.show_cpu or args.show_ram or args.show_mem or args.show_cores:
                    cpu, ram_mb, mem_mb = metrics.get()
                    if args.show_cpu:
                        if num_cores and num_cores > 0:
                            cpu_norm = cpu / num_cores  # 0-100% = fraction of total system
                            lines.append("CPU: %.1f%%" % cpu_norm)
                        else:
                            lines.append("CPU: %.1f%%" % cpu)
                    if args.show_ram:
                        lines.append("RAM: %.1f MB" % ram_mb)
                    if args.show_mem:
                        lines.append("Mem: %.1f MB" % mem_mb)
                    if args.show_cores and num_cores and num_cores > 0:
                        cores_used = cpu / 100.0
                        lines.append("cores: %.1f/%d" % (cores_used, num_cores))
                draw_overlay(vis, lines)
            writer.write(vis)
            frame_idx += 1
            if frame_idx % 50 == 0:
                print("Frames written:", frame_idx)
    finally:
        if metrics:
            metrics.stop()
        cap.release()
        writer.release()
    print("Done. Frames:", frame_idx, "->", out_path)
    meta_path = os.path.splitext(out_path)[0] + ".json"
    with open(meta_path, "w") as f:
        json.dump({
            "backend": "openvino_py",
            "model": model_dir,
            "model_stem": model_stem,
            "input_video": args.video,
            "output_video": out_path,
            "conf": args.conf,
            "imgsz": args.imgsz,
            "frame_count": frame_idx,
            "overlay": {"fps": args.show_fps, "cpu": args.show_cpu, "ram": args.show_ram, "mem": args.show_mem, "cores": args.show_cores},
        }, f, indent=2)


if __name__ == "__main__":
    main()
