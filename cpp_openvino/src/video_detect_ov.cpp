/**
 * Video file -> OpenVINO YOLO detection -> output video file.
 * Usage: video_detect_ov.exe <input_video> <model.xml> [output_video] [imgsz] [conf] [iou] [max_det] [pre_topk] [overlay]
 *   output_video: path, or "-" / "auto" -> <project_root>/output/<input_stem>_cpp_openvino_<timestamp>.mp4
 *   overlay: "1" or "show" = draw FPS (inference-only), CPU%%, RAM, Mem, cores; omit for max inference FPS
 * FPS shown is inference-only (letterbox through NMS). CPU/RAM sampled every 10 frames to reduce overhead.
 */
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <deque>
#include <filesystem>
#include <fstream>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

namespace fs = std::filesystem;

// Get project root (parent of cpp_openvino) so output/ matches Python scripts.
static fs::path get_project_root() {
#ifdef _WIN32
    wchar_t buf[4096];
    if (GetModuleFileNameW(NULL, buf, (DWORD)std::size(buf)) == 0) return fs::current_path();
    fs::path p = buf;
#else
    return fs::current_path();
#endif
#ifdef _WIN32
    // exe is in cpp_openvino/build/Release/ -> 4 levels up = project root
    for (int i = 0; i < 4 && !p.empty(); i++) p = p.parent_path();
    return p.empty() ? fs::current_path() : p;
#endif
}

// COCO 80 (used when model has 80 classes)
static const char* COCO80[80] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};
// 2-class (e.g. sack_person model)
static const char* SACK_PERSON[2] = { "sack", "person" };

// -------------------- letterbox --------------------
struct LetterboxMeta {
    float scale{1.f};
    int pad_x{0}, pad_y{0};
    int new_w{0}, new_h{0};
    int resized_w{0}, resized_h{0};
};

static cv::Mat letterbox(const cv::Mat& src, int new_w, int new_h, LetterboxMeta& m) {
    int w = src.cols, h = src.rows;
    m.new_w = new_w; m.new_h = new_h;
    m.scale = std::min((float)new_w / w, (float)new_h / h);
    m.resized_w = (int)std::round(w * m.scale);
    m.resized_h = (int)std::round(h * m.scale);
    m.pad_x = (new_w - m.resized_w) / 2;
    m.pad_y = (new_h - m.resized_h) / 2;
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(m.resized_w, m.resized_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(new_h, new_w, src.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(m.pad_x, m.pad_y, m.resized_w, m.resized_h)));
    return out;
}

static void unletterbox_xyxy(float& x1, float& y1, float& x2, float& y2, const LetterboxMeta& m) {
    x1 = (x1 - m.pad_x) / m.scale;
    y1 = (y1 - m.pad_y) / m.scale;
    x2 = (x2 - m.pad_x) / m.scale;
    y2 = (y2 - m.pad_y) / m.scale;
}

static void clamp_xyxy(float& x1, float& y1, float& x2, float& y2, int w, int h) {
    x1 = std::max(0.f, std::min(x1, (float)(w - 1)));
    y1 = std::max(0.f, std::min(y1, (float)(h - 1)));
    x2 = std::max(0.f, std::min(x2, (float)(w - 1)));
    y2 = std::max(0.f, std::min(y2, (float)(h - 1)));
}

// -------------------- detections + NMS --------------------
struct Det {
    int class_id{-1};
    float conf{0.f};
    float x1{0}, y1{0}, x2{0}, y2{0};
};

static float iou_xyxy(const Det& a, const Det& b) {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);
    float w = std::max(0.f, xx2 - xx1);
    float h = std::max(0.f, yy2 - yy1);
    float inter = w * h;
    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    float uni = areaA + areaB - inter;
    return (uni > 0.f) ? (inter / uni) : 0.f;
}

static std::vector<Det> nms(std::vector<Det>& dets, float iou_thres) {
    std::sort(dets.begin(), dets.end(), [](const Det& a, const Det& b) { return a.conf > b.conf; });
    std::vector<Det> keep;
    std::vector<char> sup(dets.size(), 0);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (sup[i]) continue;
        keep.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (sup[j]) continue;
            if (iou_xyxy(dets[i], dets[j]) > iou_thres) sup[j] = 1;
        }
    }
    return keep;
}

static void topk_prefilter(std::vector<Det>& dets, size_t k) {
    if (dets.size() <= k) return;
    std::nth_element(dets.begin(), dets.begin() + (std::ptrdiff_t)k, dets.end(),
        [](const Det& a, const Det& b) { return a.conf > b.conf; });
    dets.resize(k);
}

// -------------------- YOLO decode: raw [1, 4+num_classes, S] --------------------
static std::vector<Det> decode_yolo_raw(
    const ov::Tensor& out,
    const LetterboxMeta& meta,
    int orig_w, int orig_h,
    int num_classes,
    float conf_thres,
    int max_det
) {
    if (out.get_element_type() != ov::element::f32) return {};
    const size_t S = out.get_shape()[2];
    const float* p = out.data<const float>();
    std::vector<Det> dets;
    dets.reserve(256);
    for (size_t i = 0; i < S; i++) {
        float x = p[0 * S + i];
        float y = p[1 * S + i];
        float w = p[2 * S + i];
        float h = p[3 * S + i];
        int best_c = -1;
        float best_s = -1.f;
        for (int c = 0; c < num_classes; c++) {
            float sc = p[(size_t)(4 + c) * S + i];
            if (sc > best_s) { best_s = sc; best_c = c; }
        }
        if (best_s < conf_thres) continue;
        float x1 = x - 0.5f * w, y1 = y - 0.5f * h, x2 = x + 0.5f * w, y2 = y + 0.5f * h;
        unletterbox_xyxy(x1, y1, x2, y2, meta);
        clamp_xyxy(x1, y1, x2, y2, orig_w, orig_h);
        if ((x2 - x1) < 2.f || (y2 - y1) < 2.f) continue;
        dets.push_back({best_c, best_s, x1, y1, x2, y2});
        if ((int)dets.size() >= max_det * 50) break;
    }
    return dets;
}

// -------------------- YOLO decode: end2end [1, N, 6] or [1, 6, N] (x1, y1, x2, y2, score, class_index) in 640x640 --------------------
// Ultralytics end2end uses decode_bboxes(..., xywh=False) so boxes are already xyxy.
static std::vector<Det> decode_yolo_end2end(
    const ov::Tensor& out,
    const LetterboxMeta& meta,
    int orig_w, int orig_h,
    float conf_thres,
    int max_det
) {
    if (out.get_element_type() != ov::element::f32) return {};
    auto sh = out.get_shape();
    const size_t dim1 = sh[1];
    const size_t dim2 = sh[2];
    const float* p = out.data<const float>();
    const bool layout_n6 = (dim1 == 6 && dim2 >= 1);  // [1, 6, N]: each channel is contiguous
    const size_t N = layout_n6 ? dim2 : dim1;
    const float min_score = (conf_thres > 0.01f) ? conf_thres : 0.01f;  // skip padded slots
    std::vector<Det> dets;
    dets.reserve(std::min((int)N, max_det));
    for (size_t i = 0; i < N; i++) {
        float x1, y1, x2, y2, score;
        int class_id;
        if (layout_n6) {
            x1 = p[0 * dim2 + i]; y1 = p[1 * dim2 + i]; x2 = p[2 * dim2 + i]; y2 = p[3 * dim2 + i];
            score = p[4 * dim2 + i]; class_id = (int)(p[5 * dim2 + i] + 0.5f);
        } else {
            const float* row = p + i * 6;
            x1 = row[0]; y1 = row[1]; x2 = row[2]; y2 = row[3];
            score = row[4]; class_id = (int)(row[5] + 0.5f);
        }
        if (score < min_score) continue;  // skip padding and low-confidence
        if (class_id < 0 || class_id >= 80) continue;
        if (x2 <= x1 || y2 <= y1) continue;  // invalid box
        unletterbox_xyxy(x1, y1, x2, y2, meta);
        clamp_xyxy(x1, y1, x2, y2, orig_w, orig_h);
        if ((x2 - x1) < 2.f || (y2 - y1) < 2.f) continue;
        dets.push_back({class_id, score, x1, y1, x2, y2});
        if ((int)dets.size() >= max_det) break;
    }
    return dets;
}

static const char* class_name(int class_id, int num_classes) {
    if (num_classes == 2 && class_id >= 0 && class_id < 2)
        return SACK_PERSON[class_id];
    if (class_id >= 0 && class_id < 80)
        return COCO80[class_id];
    return "?";
}

static void draw_dets(cv::Mat& vis, const std::vector<Det>& dets, int num_classes) {
    for (const auto& d : dets) {
        cv::rectangle(vis, cv::Point((int)d.x1, (int)d.y1), cv::Point((int)d.x2, (int)d.y2),
            cv::Scalar(0, 255, 0), 2);
        const char* name = class_name(d.class_id, num_classes);
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s %.2f", name, (double)d.conf);
        int base = 0;
        auto sz = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &base);
        int x = std::max(0, (int)d.x1);
        int y = std::max(0, (int)d.y1 - sz.height - 4);
        cv::rectangle(vis, cv::Rect(x, y, sz.width + 6, sz.height + 6), cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(vis, buf, cv::Point(x + 3, y + sz.height + 2),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    }
}

// -------------------- overlay (same style as Python: FPS, CPU%%, RAM, Mem, cores) --------------------
static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else out += c;
    }
    return out;
}

static std::string timestamp_for_output() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y%m%d_%H%M%S");
    return oss.str();
}

static void draw_overlay(cv::Mat& vis, const std::vector<std::string>& lines, int x = 10, int y = 24,
    double font_scale = 0.55, int thickness = 1) {
    if (lines.empty()) return;
    int pad = 4;
    int max_w = 0;
    for (const auto& s : lines) {
        int base = 0;
        auto sz = cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &base);
        if (sz.width > max_w) max_w = sz.width;
    }
    int h_line = (int)(22 * font_scale + 4);
    int block_w = max_w + 2 * pad;
    int block_h = (int)lines.size() * h_line + 2 * pad;
    cv::Mat overlay = vis.clone();
    cv::rectangle(overlay, cv::Point(x, y), cv::Point(x + block_w, y + block_h), cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(overlay, 0.6f, vis, 0.4f, 0, vis);
    for (size_t i = 0; i < lines.size(); i++)
        cv::putText(vis, lines[i], cv::Point(x + pad, y + pad + (int)(i + 1) * h_line - 4),
            cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 255, 0), thickness, cv::LINE_AA);
}

#ifdef _WIN32
// Windows: process CPU (normalized %% of system), RAM MB, Mem (virtual) MB, num cores
static int g_num_cores = 0;
static ULONGLONG g_last_user = 0, g_last_kernel = 0;
static std::chrono::steady_clock::time_point g_last_wall;

static ULONGLONG filetime_to_ulonglong(const FILETIME& ft) {
    ULARGE_INTEGER u;
    u.LowPart = ft.dwLowDateTime;
    u.HighPart = ft.dwHighDateTime;
    return u.QuadPart;
}

static void win_metrics_init() {
    g_num_cores = (int)std::thread::hardware_concurrency();
    if (g_num_cores <= 0) g_num_cores = 1;
    HANDLE h = GetCurrentProcess();
    FILETIME ct, et, kt, ut;
    if (GetProcessTimes(h, &ct, &et, &kt, &ut)) {
        g_last_user = filetime_to_ulonglong(ut);
        g_last_kernel = filetime_to_ulonglong(kt);
    }
    g_last_wall = std::chrono::steady_clock::now();
}

static void win_metrics_sample(double& cpu_pct_norm, double& ram_mb, double& mem_mb) {
    cpu_pct_norm = 0;
    ram_mb = 0;
    mem_mb = 0;
    HANDLE h = GetCurrentProcess();
    FILETIME ct, et, kt, ut;
    if (!GetProcessTimes(h, &ct, &et, &kt, &ut)) return;
    ULONGLONG user = filetime_to_ulonglong(ut);
    ULONGLONG kernel = filetime_to_ulonglong(kt);
    auto now = std::chrono::steady_clock::now();
    double wall_sec = 1e-9 * (double)std::chrono::duration_cast<std::chrono::nanoseconds>(now - g_last_wall).count();
    if (wall_sec > 0.01 && g_last_user != 0) {
        ULONGLONG delta = (user - g_last_user) + (kernel - g_last_kernel);
        double delta_sec = delta / 10000000.0;  // 100ns -> s
        cpu_pct_norm = (delta_sec / (wall_sec * g_num_cores)) * 100.0;
        if (cpu_pct_norm > 100.0) cpu_pct_norm = 100.0;
    }
    g_last_user = user;
    g_last_kernel = kernel;
    g_last_wall = now;

    PROCESS_MEMORY_COUNTERS pmc = {};
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(h, &pmc, sizeof(pmc))) {
        ram_mb = (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
        mem_mb = (double)pmc.PagefileUsage / (1024.0 * 1024.0);
    }
}

// Cached metrics: sample every N frames to avoid per-frame syscall overhead.
static double g_cpu_pct = 0, g_ram_mb = 0, g_mem_mb = 0;
static const int METRICS_SAMPLE_INTERVAL = 10;

static void win_metrics_sample_cached(int frame_idx, double& cpu_pct, double& ram_mb, double& mem_mb) {
    if (frame_idx % METRICS_SAMPLE_INTERVAL == 0)
        win_metrics_sample(cpu_pct, ram_mb, mem_mb);
    else {
        cpu_pct = g_cpu_pct;
        ram_mb = g_ram_mb;
        mem_mb = g_mem_mb;
    }
    g_cpu_pct = cpu_pct;
    g_ram_mb = ram_mb;
    g_mem_mb = mem_mb;
}
#endif

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: video_detect_ov.exe <input_video> <model.xml> [output_video] [imgsz=640] [conf=0.35] [iou=0.45] [max_det=50] [pre_topk=200] [overlay]\n";
        std::cerr << "  output_video: path, or \"-\" / \"auto\" for <project_root>/output/<stem>_<model>_ov_cpp_<timestamp>.mp4\n";
        std::cerr << "  overlay: \"1\" or \"show\" to draw FPS, CPU%%, RAM, Mem, cores (like Python scripts)\n";
        return 1;
    }
    std::string video_path = argv[1];
    std::string model_xml = argv[2];
    fs::path model_path_p(model_xml);
    std::string model_stem = model_path_p.parent_path().stem().string();
    if (model_stem.size() >= 15 && model_stem.substr(model_stem.size() - 15) == "_openvino_model")
        model_stem = model_stem.substr(0, model_stem.size() - 15);
    else if (model_stem.size() >= 8 && model_stem.substr(model_stem.size() - 8) == "_openvino")
        model_stem = model_stem.substr(0, model_stem.size() - 8);
    std::string out_video;
    int arg_offset = 3;
    if (argc > 3 && std::string(argv[3]) != "-" && std::string(argv[3]) != "auto") {
        out_video = argv[3];
        arg_offset = 4;
    } else {
        fs::path root = get_project_root();
        fs::path out_dir = root / "output";
        fs::create_directories(out_dir);
        fs::path p(video_path);
        std::string stem = p.stem().string();
        out_video = (out_dir / (stem + "_" + model_stem + "_ov_cpp_" + timestamp_for_output() + ".mp4")).string();
        if (argc > 3) arg_offset = 4;  // consumed "-" or "auto"
    }
    int imgsz = (argc >= arg_offset + 1) ? std::stoi(argv[arg_offset]) : 640;
    float conf = (argc >= arg_offset + 2) ? std::stof(argv[arg_offset + 1]) : 0.35f;
    float iou = (argc >= arg_offset + 3) ? std::stof(argv[arg_offset + 2]) : 0.45f;
    int max_det = (argc >= arg_offset + 4) ? std::max(1, std::stoi(argv[arg_offset + 3])) : 50;
    int pre_topk = (argc >= arg_offset + 5) ? std::max(1, std::stoi(argv[arg_offset + 4])) : 200;
    bool show_overlay = false;
    if (argc >= arg_offset + 6) {
        std::string o = argv[arg_offset + 5];
        show_overlay = (o == "1" || o == "show" || o == "overlay");
    }

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << video_path << "\n";
        return 2;
    }
    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25.0;

    ov::Core core;
    auto model = core.read_model(model_xml);
    ov::CompiledModel compiled = core.compile_model(model, "CPU");
    ov::InferRequest req = compiled.create_infer_request();
    ov::Tensor input_tensor = req.get_input_tensor();
    auto in_shape = input_tensor.get_shape();
    if (in_shape.size() != 4 || in_shape[0] != 1 || in_shape[1] != 3) {
        std::cerr << "Unexpected input shape\n";
        return 3;
    }
    auto out_shape = compiled.output(0).get_shape();
    if (out_shape.size() != 3 || out_shape[0] != 1) {
        std::cerr << "Unexpected output shape (expected [1, C, S] or [1, N, 6])\n";
        return 4;
    }
    bool is_end2end = (out_shape[2] == 6) || (out_shape[1] == 6 && out_shape[2] >= 100);
    int num_classes = 80;
    if (is_end2end) {
        num_classes = 80;  // for class_name(); COCO
        std::cout << "Model: " << model_xml << "  format=end2end [1," << out_shape[1] << ",6]  imgsz=" << imgsz << "  conf=" << conf << "\n";
    } else {
        num_classes = (int)out_shape[1] - 4;
        if (num_classes <= 0 || num_classes > 1000) {
            std::cerr << "Unexpected output shape [1, " << out_shape[1] << ", S]\n";
            return 4;
        }
        std::cout << "Model: " << model_xml << "  format=raw [1," << (4 + num_classes) << "," << out_shape[2] << "]  imgsz=" << imgsz << "  conf=" << conf << "\n";
    }
    std::cout << "Input: " << video_path << "  Output: " << out_video << "  " << w << "x" << h << " @ " << fps << " fps\n";
    if (show_overlay) std::cout << "Overlay: FPS, CPU%%, RAM, Mem, cores\n";

    cv::VideoWriter writer(out_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(w, h));
    if (!writer.isOpened()) {
        std::cerr << "Cannot create output video: " << out_video << "\n";
        return 5;
    }

#ifdef _WIN32
    int num_cores = (int)std::thread::hardware_concurrency();
    if (num_cores <= 0) num_cores = 1;
    if (show_overlay) win_metrics_init();
#endif

    int frame_idx = 0;
    std::deque<double> fps_times;
    const size_t fps_window_size = 30;
    double inf_fps = 0.0;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;

        auto t_start = std::chrono::steady_clock::now();

        LetterboxMeta meta{};
        cv::Mat lb = letterbox(frame, imgsz, imgsz, meta);
        cv::cvtColor(lb, lb, cv::COLOR_BGR2RGB);
        lb.convertTo(lb, CV_32F, 1.0 / 255.0);
        std::vector<cv::Mat> ch(3);
        cv::split(lb, ch);
        float* dst = input_tensor.data<float>();
        size_t plane = (size_t)imgsz * (size_t)imgsz;
        std::memcpy(dst + 0 * plane, ch[0].data, plane * sizeof(float));
        std::memcpy(dst + 1 * plane, ch[1].data, plane * sizeof(float));
        std::memcpy(dst + 2 * plane, ch[2].data, plane * sizeof(float));
        req.infer();
        ov::Tensor out = req.get_output_tensor();
        std::vector<Det> dets;
        if (is_end2end) {
            dets = decode_yolo_end2end(out, meta, frame.cols, frame.rows, conf, max_det);
        } else {
            std::vector<Det> dets_raw = decode_yolo_raw(out, meta, frame.cols, frame.rows, num_classes, conf, max_det);
            topk_prefilter(dets_raw, (size_t)pre_topk);
            dets = nms(dets_raw, iou);
            if ((int)dets.size() > max_det) dets.resize(max_det);
        }

        auto t_infer_end = std::chrono::steady_clock::now();
        if (show_overlay) {
            double elapsed = 1e-9 * (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t_infer_end - t_start).count();
            fps_times.push_back(elapsed);
            if (fps_times.size() > fps_window_size) fps_times.pop_front();
            if (fps_times.size() >= 2) {
                double total = 0;
                for (double e : fps_times) total += e;
                inf_fps = (fps_times.size() - 1) / total;
            }
        }

        cv::Mat vis = frame.clone();
        draw_dets(vis, dets, num_classes);

        if (show_overlay) {
            std::vector<std::string> lines;
            char buf[128];
            if (fps_times.size() >= 2)
                std::snprintf(buf, sizeof(buf), "FPS: %.1f", inf_fps);
            else
                std::snprintf(buf, sizeof(buf), "FPS: --");
            lines.push_back(buf);
#ifdef _WIN32
            double cpu_pct, ram_mb, mem_mb;
            win_metrics_sample_cached(frame_idx, cpu_pct, ram_mb, mem_mb);
            std::snprintf(buf, sizeof(buf), "CPU: %.1f%%", cpu_pct);
            lines.push_back(buf);
            std::snprintf(buf, sizeof(buf), "RAM: %.1f MB", ram_mb);
            lines.push_back(buf);
            std::snprintf(buf, sizeof(buf), "Mem: %.1f MB", mem_mb);
            lines.push_back(buf);
            double cores_used = (cpu_pct / 100.0) * num_cores;  // approximate
            std::snprintf(buf, sizeof(buf), "cores: %.1f/%d", cores_used, num_cores);
            lines.push_back(buf);
#else
            lines.push_back("CPU/RAM/Mem: (Windows only)");
            lines.push_back("cores: N/A");
#endif
            draw_overlay(vis, lines);
        }

        writer.write(vis);
        frame_idx++;
        if (frame_idx % 50 == 0)
            std::cout << "Frames written: " << frame_idx << "\n";
    }
    cap.release();
    writer.release();
    std::cout << "Done. Frames: " << frame_idx << " -> " << out_video << "\n";
    // Write metadata JSON alongside output for tracing
    std::string meta_path = out_video;
    if (meta_path.size() >= 4 && meta_path.substr(meta_path.size() - 4) == ".mp4")
        meta_path = meta_path.substr(0, meta_path.size() - 4) + ".json";
    std::ofstream meta(meta_path);
    if (meta) {
        meta << "{\n";
        meta << "  \"backend\": \"openvino_cpp\",\n";
        meta << "  \"model\": \"" << json_escape(model_xml) << "\",\n";
        meta << "  \"model_stem\": \"" << json_escape(model_stem) << "\",\n";
        meta << "  \"input_video\": \"" << json_escape(video_path) << "\",\n";
        meta << "  \"output_video\": \"" << json_escape(out_video) << "\",\n";
        meta << "  \"conf\": " << conf << ",\n";
        meta << "  \"imgsz\": " << imgsz << ",\n";
        meta << "  \"frame_count\": " << frame_idx << ",\n";
        meta << "  \"overlay\": " << (show_overlay ? "true" : "false") << "\n";
        meta << "}\n";
    }
    return 0;
}
