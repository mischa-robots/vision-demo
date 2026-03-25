// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>

/**
 * yolo_det.cpp
 * ─────────────
 * YOLO object detection on Jetson Nano using OpenCV DNN + ONNX model.
 * Streams result to browser via MediaMTX.
 *
 * Three-stage pipeline running in separate threads:
 *   Thread 1 (capture)  — CSI camera → resize + normalise → frame queue
 *   Thread 2 (infer)    — OpenCV DNN inference + NMS → result queue
 *   Thread 3 (encode)   — draw boxes → H.264 encoder → RTP → MediaMTX
 *
 * Watch the stream at:
 *   WebRTC (lowest latency) : http://<jetson-ip>:8889/yolo
 *   HLS   (most compatible) : http://<jetson-ip>:8888/yolo
 *   RTSP  (VLC / ffplay)    : rtsp://<jetson-ip>:8554/yolo
 *
 * Usage:
 *   ./build/yolodnn_det --model ./models/yolo.onnx --labels ./models/yolo-labels.txt
 *   ./build/yolodnn_det --help
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include "yolo_dnn_engine.hpp"
#include "yolo_postprocess.hpp"

// ════════════════════════════════════════════════════════════════════════════
// Thread-safe fixed-capacity queue
// Drops the oldest frame when full — keeps latency low on a robot.
// ════════════════════════════════════════════════════════════════════════════
template<typename T>
class FrameQueue {
public:
    explicit FrameQueue(size_t capacity) : _cap(capacity) {}

    void push(T item) {
        std::unique_lock<std::mutex> lock(_mu);
        if (_q.size() >= _cap) _q.pop();
        _q.push(std::move(item));
        _cv.notify_one();
    }

    bool pop(T& item, int timeout_ms = 200) {
        std::unique_lock<std::mutex> lock(_mu);
        if (!_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                          [this]{ return !_q.empty() || _stop.load(); }))
            return false;
        if (_q.empty()) return false;
        item = std::move(_q.front());
        _q.pop();
        return true;
    }

    void stop() { _stop = true; _cv.notify_all(); }

private:
    size_t                   _cap;
    std::queue<T>            _q;
    std::mutex               _mu;
    std::condition_variable  _cv;
    std::atomic<bool>        _stop{false};
};

// ════════════════════════════════════════════════════════════════════════════
// Data passed between threads
// ════════════════════════════════════════════════════════════════════════════
struct CaptureFrame {
    cv::Mat            bgr;
    std::vector<float> input;         // preprocessed NCHW float32
    int64_t            timestamp_us;
};

struct InferResult {
    cv::Mat                 bgr;
    std::vector<Detection>  detections;
    int64_t                 timestamp_us;
};

// ════════════════════════════════════════════════════════════════════════════
// Config
// ════════════════════════════════════════════════════════════════════════════
struct Config {
    std::string model_path    = "./models/yolo.onnx";
    std::string labels_path   = "./models/yolo-labels.txt";
    int         width         = 1280;
    int         height        = 720;
    int         fps           = 30;
    int         sensor_id     = 0;
    int         flip          = 0;
    float       conf          = 0.25f;
    float       iou           = 0.45f;
    std::string udp_host      = "127.0.0.1";
    int         udp_port      = 5602;    // mediamtx_demo.yml yolo path
    std::string stream_name   = "yolo";
    int         bitrate       = 4000000;
    int         iframeinterval = 30;
    int         h264_profile  = 2;       // 0=Baseline 2=Main 4=High (use 0 or 2 for WebRTC)
    int         warmup_runs   = 5;
    int         queue_size    = 2;
};

// ════════════════════════════════════════════════════════════════════════════
// GStreamer pipeline strings
// ════════════════════════════════════════════════════════════════════════════
static std::string gst_capture(const Config& c) {
    std::ostringstream s;
    s << "nvarguscamerasrc sensor-id=" << c.sensor_id << " ! "
      << "video/x-raw(memory:NVMM),width=" << c.width
      << ",height=" << c.height
      << ",framerate=" << c.fps << "/1,format=NV12 ! "
      << "nvvidconv flip-method=" << c.flip << " ! "
      << "video/x-raw,format=BGRx ! "
      << "videoconvert ! "
      << "video/x-raw,format=BGR ! "
      << "appsink drop=true max-buffers=1 sync=false";
    return s.str();
}

static std::string gst_encode(const Config& c) {
    std::ostringstream s;
    s << "appsrc ! "
      << "video/x-raw,format=BGR,width=" << c.width
      << ",height=" << c.height
      << ",framerate=" << c.fps << "/1 ! "
      << "videoconvert ! "
      << "video/x-raw,format=BGRx ! "
      << "nvvidconv ! "
      << "video/x-raw(memory:NVMM),format=NV12 ! "
      << "nvv4l2h264enc maxperf-enable=1 insert-sps-pps=true"
      << " profile=" << c.h264_profile
      << " control-rate=1"
      << " iframeinterval=" << c.iframeinterval
      << " bitrate=" << c.bitrate << " ! "
      << "h264parse ! "
      << "rtph264pay pt=96 config-interval=-1 ! "
      << "udpsink host=" << c.udp_host
      << " port=" << c.udp_port << " sync=false";
    return s.str();
}

// ════════════════════════════════════════════════════════════════════════════
// Preprocess — resize + normalise frame for DNN input
// GPU handles resize and BGR→RGB; normalisation stays on CPU
// (convertTo CUDA binding is broken on this aarch64 build)
// ════════════════════════════════════════════════════════════════════════════
static void preprocess(const cv::Mat& frame_bgr,
                       std::vector<float>& out,
                       cv::cuda::GpuMat& gpu_src,
                       cv::cuda::GpuMat& gpu_resized,
                       cv::cuda::GpuMat& gpu_rgb)
{
    constexpr int SIZE = DNNEngine::INPUT_H;

    gpu_src.upload(frame_bgr);
    cv::cuda::resize(gpu_src, gpu_resized, {SIZE, SIZE});
    cv::cuda::cvtColor(gpu_resized, gpu_rgb, cv::COLOR_BGR2RGB);

    cv::Mat cpu_rgb;
    gpu_rgb.download(cpu_rgb);

    cv::Mat cpu_float;
    cpu_rgb.convertTo(cpu_float, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(cpu_float, channels);

    out.resize(DNNEngine::INPUT_SIZE);
    const int plane = SIZE * SIZE;
    for (int c = 0; c < 3; ++c)
        std::memcpy(out.data() + c * plane,
                    channels[c].ptr<float>(),
                    plane * sizeof(float));
}

// ════════════════════════════════════════════════════════════════════════════
// Global stop flag — set by Ctrl+C
// ════════════════════════════════════════════════════════════════════════════
static std::atomic<bool> g_stop{false};
static void sig_handler(int) { g_stop.store(true); }

// ════════════════════════════════════════════════════════════════════════════
// Thread 1 — Capture + Preprocess
// ════════════════════════════════════════════════════════════════════════════
static void capture_thread(const Config& cfg, FrameQueue<CaptureFrame>& queue)
{
    cv::VideoCapture cap(gst_capture(cfg), cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "[CAPTURE] Failed to open camera\n";
        g_stop = true;
        return;
    }
    std::cout << "[CAPTURE] Camera opened ("
              << cfg.width << "x" << cfg.height << " @ " << cfg.fps << "fps)\n";

    cv::cuda::GpuMat gpu_src, gpu_resized, gpu_rgb;
    cv::Mat frame;

    while (!g_stop) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "[CAPTURE] Dropped frame\n";
            continue;
        }

        CaptureFrame cf;
        cf.bgr = frame.clone();
        cf.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        preprocess(frame, cf.input, gpu_src, gpu_resized, gpu_rgb);
        queue.push(std::move(cf));
    }

    cap.release();
    queue.stop();
    std::cout << "[CAPTURE] Thread exited\n";
}

// ════════════════════════════════════════════════════════════════════════════
// Thread 2 — DNN Inference + NMS
// ════════════════════════════════════════════════════════════════════════════
static void infer_thread(const Config& cfg,
                         DNNEngine& engine,
                         const std::vector<std::string>& labels,
                         FrameQueue<CaptureFrame>& in_queue,
                         FrameQueue<InferResult>&  out_queue)
{
    std::vector<float> output(DNNEngine::OUTPUT_SIZE);

    while (!g_stop) {
        CaptureFrame cf;
        if (!in_queue.pop(cf)) continue;

        engine.infer(cf.input.data(), output.data());

        auto dets = postprocess(output.data(),
                                cfg.width, cfg.height,
                                labels, cfg.conf, cfg.iou);

        InferResult ir;
        ir.bgr          = std::move(cf.bgr);
        ir.detections   = std::move(dets);
        ir.timestamp_us = cf.timestamp_us;
        out_queue.push(std::move(ir));
    }

    out_queue.stop();
    std::cout << "[INFER] Thread exited\n";
}

// ════════════════════════════════════════════════════════════════════════════
// Thread 3 — Draw + Encode + Stream
// ════════════════════════════════════════════════════════════════════════════
static void encode_thread(const Config& cfg, FrameQueue<InferResult>& queue)
{
    cv::VideoWriter writer(gst_encode(cfg), cv::CAP_GSTREAMER, 0,
                           cfg.fps, {cfg.width, cfg.height});
    if (!writer.isOpened()) {
        std::cerr << "[ENCODE] Failed to open RTP output — is MediaMTX running?\n";
        g_stop = true;
        return;
    }
    std::cout << "[ENCODE] Streaming -> http://<jetson-ip>:8889/" << cfg.stream_name << "  (WebRTC)\n";

    int    fps_count   = 0;
    double fps_display = 0.0;
    auto   t_fps       = std::chrono::steady_clock::now();

    while (!g_stop) {
        InferResult ir;
        if (!queue.pop(ir)) continue;

        fps_count++;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t_fps).count();
        if (elapsed >= 1.0) {
            fps_display = fps_count / elapsed;
            fps_count   = 0;
            t_fps       = now;

            int64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                now.time_since_epoch()).count();
            double latency_ms = (now_us - ir.timestamp_us) / 1000.0;

            std::cout << "[INFO] FPS: " << std::fixed << std::setprecision(1)
                      << fps_display
                      << "  dets: "    << ir.detections.size()
                      << "  latency: " << std::setprecision(0)
                      << latency_ms    << "ms\n";
        }

        drawDetections(ir.bgr, ir.detections, fps_display);
        writer.write(ir.bgr);
    }

    writer.release();
    std::cout << "[ENCODE] Thread exited\n";
}

// ════════════════════════════════════════════════════════════════════════════
// CLI argument parser
// ════════════════════════════════════════════════════════════════════════════
static Config parse_args(int argc, char** argv)
{
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << "\n";
                std::exit(1);
            }
            return argv[++i];
        };

        if      (arg == "--model")          cfg.model_path    = next();
        else if (arg == "--labels")         cfg.labels_path   = next();
        else if (arg == "--width")          cfg.width         = std::stoi(next());
        else if (arg == "--height")         cfg.height        = std::stoi(next());
        else if (arg == "--fps")            cfg.fps           = std::stoi(next());
        else if (arg == "--sensor")         cfg.sensor_id     = std::stoi(next());
        else if (arg == "--flip")           cfg.flip          = std::stoi(next());
        else if (arg == "--conf")           cfg.conf          = std::stof(next());
        else if (arg == "--iou")            cfg.iou           = std::stof(next());
        else if (arg == "--udp-host")       cfg.udp_host      = next();
        else if (arg == "--udp-port")       cfg.udp_port      = std::stoi(next());
        else if (arg == "--stream-name")    cfg.stream_name   = next();
        else if (arg == "--bitrate")        cfg.bitrate       = std::stoi(next());
        else if (arg == "--iframeinterval") cfg.iframeinterval = std::stoi(next());
        else if (arg == "--h264-profile")   cfg.h264_profile  = std::stoi(next());
        else if (arg == "--warmup")         cfg.warmup_runs   = std::stoi(next());
        else if (arg == "--help") {
            std::cout <<
                "Usage: ./build/yolodnn_det [options]\n"
                "\n"
                "  --model          PATH   ONNX model file       (default: ./models/yolo.onnx)\n"
                "  --labels         PATH   Labels file           (default: ./models/yolo-labels.txt)\n"
                "  --width          INT    Camera width          (default: 1280)\n"
                "  --height         INT    Camera height         (default: 720)\n"
                "  --fps            INT    Camera FPS            (default: 30)\n"
                "  --sensor         INT    CSI sensor-id         (default: 0)\n"
                "  --flip           INT    nvvidconv flip        (default: 0)\n"
                "  --conf           FLOAT  Confidence threshold  (default: 0.25)\n"
                "  --iou            FLOAT  NMS IoU threshold     (default: 0.45)\n"
                "  --udp-host       STR    MediaMTX UDP host     (default: 127.0.0.1)\n"
                "  --udp-port       INT    MediaMTX UDP port     (default: 5602)\n"
                "  --stream-name    STR    Stream name (banner)  (default: yolo)\n"
                "  --bitrate        INT    Encoder bps           (default: 4000000)\n"
                "  --iframeinterval INT    Keyframe interval     (default: 30)\n"
                "  --h264-profile   INT    H.264 profile         (default: 2=Main)\n"
                "                           0=Baseline 2=Main 4=High\n"
                "                           NOTE: use 0 or 2 for WebRTC/MediaMTX\n"
                "  --warmup         INT    Warmup runs           (default: 5)\n";
            std::exit(0);
        }
        else {
            std::cerr << "Unknown argument: " << arg << " (use --help)\n";
            std::exit(1);
        }
    }
    return cfg;
}

// ════════════════════════════════════════════════════════════════════════════
// Main
// ════════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv)
{
    std::signal(SIGINT,  sig_handler);
    std::signal(SIGTERM, sig_handler);

    Config cfg = parse_args(argc, argv);

    std::vector<std::string> labels = load_labels(cfg.labels_path);

    std::cout << "==================================================\n"
              << "  yolodnn_det -- YOLO DNN Detection\n"
              << "==================================================\n"
              << "  Model    : " << cfg.model_path  << "\n"
              << "  Labels   : " << cfg.labels_path
              <<             " (" << labels.size() << " classes)\n"
              << "  Camera   : " << cfg.width << "x" << cfg.height
              <<             " @ " << cfg.fps << "fps  sensor=" << cfg.sensor_id << "\n"
              << "  Conf     : " << cfg.conf          << "\n"
              << "  IoU      : " << cfg.iou           << "\n"
              << "  Bitrate  : " << cfg.bitrate       << " bps\n"
              << "  UDP ->   : " << cfg.udp_host << ":" << cfg.udp_port << "\n"
              << "  Stream   : http://<host>:8889/" << cfg.stream_name << "\n"
              << "==================================================\n";

    DNNEngine engine(cfg.model_path);
    engine.warmup(cfg.warmup_runs);

    FrameQueue<CaptureFrame> capture_queue(cfg.queue_size);
    FrameQueue<InferResult>  infer_queue(cfg.queue_size);

    std::thread t_cap(capture_thread,
                      std::cref(cfg), std::ref(capture_queue));
    std::thread t_inf(infer_thread,
                      std::cref(cfg), std::ref(engine), std::cref(labels),
                      std::ref(capture_queue), std::ref(infer_queue));
    std::thread t_enc(encode_thread,
                      std::cref(cfg), std::ref(infer_queue));

    t_cap.join();
    t_inf.join();
    t_enc.join();

    std::cout << "[INFO] All threads done. Goodbye.\n";
    return 0;
}
