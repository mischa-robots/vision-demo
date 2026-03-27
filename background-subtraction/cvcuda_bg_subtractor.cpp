// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>

/**
 * cvcuda_bg_subtractor.cpp
 * ------------------------
 * Background subtraction for Jetson Nano using OpenCV CUDA MOG2.
 * Streams a side-by-side canvas to the browser via MediaMTX.
 *
 * Algorithm : cv::cuda::BackgroundSubtractorMOG2 (CUDA)
 *
 * Side-by-side layout (default 1280x360):
 *   +---------------------+---------------------+
 *   |  Original stream    |  Foreground mask     |
 *   |                     |  (COLORMAP_TURBO)    |
 *   +---------------------+---------------------+
 *
 * Watch the stream at:
 *   WebRTC : http://<jetson-ip>:8889/bg_subtraction
 *   HLS    : http://<jetson-ip>:8888/bg_subtraction
 *   RTSP   : rtsp://<jetson-ip>:8554/bg_subtraction
 *
 * Usage:
 *   ./build/cvcuda_bg_subtractor
 *   ./build/cvcuda_bg_subtractor --help
 */

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

// ============================================================================
// Global stop flag
// ============================================================================
static std::atomic<bool> g_stop{false};
static void sig_handler(int) { g_stop.store(true); }

// ============================================================================
// Config
// ============================================================================
struct Config {
    int         width          = 1280;
    int         height         = 720;
    int         fps            = 15;
    int         sensor_id      = 0;
    int         flip_method    = 0;
    bool        lock_ae        = false;
    bool        lock_awb       = false;
    int         bitrate        = 4000000;
    int         iframeinterval = 30;
    std::string udp_host       = "127.0.0.1";
    int         udp_port       = 5603;
    std::string stream_name    = "bg_subtraction";
    bool        show_fps       = true;
    int         stream_w       = 1280;
    int         stream_h       = 360;

    // MOG2 parameters
    int    history       = 500;
    double var_threshold = 16.0;
    bool   detect_shadows = false;
    double learning_rate = 0.01;
};

// ============================================================================
// GStreamer pipelines
// ============================================================================
static std::string capture_pipeline(int sensor_id, int w, int h, int fps, int flip,
                                    bool lock_ae, bool lock_awb)
{
    std::ostringstream ss;
    ss << "nvarguscamerasrc sensor-id=" << sensor_id;
    if (lock_ae)  ss << " aelock=true";
    if (lock_awb) ss << " awblock=true";
    ss << " ! video/x-raw(memory:NVMM)"
          ",width=" << w << ",height=" << h
       << ",framerate=" << fps << "/1,format=NV12"
          " ! nvvidconv flip-method=" << flip
       << " ! video/x-raw,format=BGRx,width=" << w << ",height=" << h
       << " ! videoconvert ! video/x-raw,format=BGR"
          " ! appsink name=appsink max-buffers=1 drop=true sync=false";
    return ss.str();
}

static std::string push_pipeline(int w, int h, int fps,
                                  const std::string& host, int port,
                                  int bitrate, int iframeinterval)
{
    std::ostringstream ss;
    ss << "appsrc name=src is-live=true format=time"
          " caps=video/x-raw,format=BGR"
          ",width=" << w << ",height=" << h
       << ",framerate=" << fps << "/1"
          " ! videoconvert ! video/x-raw,format=BGRx"
          " ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12"
          " ! nvv4l2h264enc maxperf-enable=1 insert-sps-pps=true"
          "   profile=2 control-rate=1"
          "   bitrate=" << bitrate << " iframeinterval=" << iframeinterval
       << " ! h264parse config-interval=-1"
          " ! rtph264pay pt=96 config-interval=-1"
          " ! udpsink host=" << host << " port=" << port << " sync=false";
    return ss.str();
}

// ============================================================================
// CameraBuffer
// ============================================================================
struct CameraBuffer {
    std::mutex              mtx;
    std::condition_variable cv;
    cv::Mat                 buf;
    bool                    valid = false;

    void put(cv::Mat&& frame) {
        { std::lock_guard<std::mutex> lk(mtx); buf = std::move(frame); valid = true; }
        cv.notify_one();
    }
    bool get(cv::Mat& out,
             std::chrono::milliseconds timeout = std::chrono::milliseconds(500)) {
        std::unique_lock<std::mutex> lk(mtx);
        if (!cv.wait_for(lk, timeout, [this]{ return valid; })) return false;
        out = buf.clone(); valid = false; return true;
    }
};

static void camera_thread(const std::string& pipe, CameraBuffer& cam_buf)
{
    cv::VideoCapture cap(pipe, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "[camera] FATAL: " << pipe << "\n";
        g_stop.store(true); return;
    }
    std::cout << "[camera] opened.\n";
    cv::Mat frame;
    while (!g_stop.load()) {
        if (!cap.read(frame) || frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); continue;
        }
        cam_buf.put(std::move(frame));
    }
    cap.release();
}

// ============================================================================
// GstPushPipeline
// ============================================================================
struct GstPushPipeline {
    bool         ok       = false;
    GstElement*  pipeline = nullptr;
    GstElement*  appsrc   = nullptr;
    GstClockTime ts       = 0;
    GstClockTime frame_ns = 0;

    bool start(const std::string& s, int fps) {
        GError* err = nullptr;
        pipeline = gst_parse_launch(s.c_str(), &err);
        if (!pipeline || err) {
            std::cerr << "[gst-push] " << (err ? err->message : "null") << "\n";
            if (err) { g_error_free(err); return false; }
        }
        appsrc   = gst_bin_get_by_name(GST_BIN(pipeline), "src");
        frame_ns = GST_SECOND / fps;
        if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE)
            return false;
        ok = true;
        std::cout << "[gst-push] pipeline started.\n";
        return true;
    }
    bool push(const cv::Mat& frame) {
        if (!ok) return false;
        size_t sz = frame.total() * frame.elemSize();
        GstBuffer* buf = gst_buffer_new_allocate(nullptr, sz, nullptr);
        GstMapInfo map;
        gst_buffer_map(buf, &map, GST_MAP_WRITE);
        std::memcpy(map.data, frame.data, sz);
        gst_buffer_unmap(buf, &map);
        GST_BUFFER_PTS(buf) = ts; GST_BUFFER_DURATION(buf) = frame_ns; ts += frame_ns;
        GstFlowReturn ret;
        g_signal_emit_by_name(appsrc, "push-buffer", buf, &ret);
        gst_buffer_unref(buf);
        return ret == GST_FLOW_OK;
    }
    void stop() {
        if (!ok) return;
        g_signal_emit_by_name(appsrc, "end-of-stream");
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(appsrc); gst_object_unref(pipeline);
        pipeline = nullptr; appsrc = nullptr; ok = false;
        std::cout << "[gst-push] stopped.\n";
    }
    ~GstPushPipeline() { stop(); }
};

// ============================================================================
// OpenCV CUDA MOG2 Background Subtractor
// ============================================================================
struct CvCudaMOG2 {
    cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> mog2;
    cv::cuda::GpuMat d_frame;
    cv::cuda::GpuMat d_fgmask;

    void init(const Config& cfg)
    {
        mog2 = cv::cuda::createBackgroundSubtractorMOG2(
            cfg.history, cfg.var_threshold, cfg.detect_shadows);
    }

    // Upload BGR frame and run MOG2; returns foreground mask as U8 cv::Mat
    cv::Mat process(const cv::Mat& bgr, double learning_rate)
    {
        d_frame.upload(bgr);
        mog2->apply(d_frame, d_fgmask, learning_rate);
        cv::Mat result;
        d_fgmask.download(result);
        return result;
    }
};

// ============================================================================
// Visualisation helper
// ============================================================================
static void tile_label(cv::Mat& tile, const std::string& label,
                       const std::vector<std::string>& stats = {})
{
    auto put = [&](const std::string& s, cv::Point p, double scale) {
        cv::putText(tile, s, p, cv::FONT_HERSHEY_SIMPLEX, scale,
                    cv::Scalar(0,0,0),       3, cv::LINE_AA);
        cv::putText(tile, s, p, cv::FONT_HERSHEY_SIMPLEX, scale,
                    cv::Scalar(255,255,255), 1, cv::LINE_AA);
    };
    put(label, {8, 22}, 0.60);
    int y = 44;
    for (const auto& s : stats) { put(s, {8, y}, 0.48); y += 18; }
}

static cv::Mat build_sidebyside(
    const cv::Mat& bgr,
    const cv::Mat& fgmask_u8,
    double fps,
    int canvas_w, int canvas_h)
{
    const int tw = canvas_w / 2, th = canvas_h;
    char buf[64];

    // Left tile: original
    cv::Mat t0;
    cv::resize(bgr, t0, {tw, th}, 0, 0, cv::INTER_AREA);
    tile_label(t0, "Original");

    // Right tile: foreground mask with COLORMAP_TURBO
    cv::Mat mask_resized;
    cv::resize(fgmask_u8, mask_resized, {tw, th}, 0, 0, cv::INTER_AREA);
    cv::Mat t1;
    cv::applyColorMap(mask_resized, t1, cv::COLORMAP_TURBO);

    int fg_pixels = cv::countNonZero(mask_resized);
    float fg_pct  = 100.f * fg_pixels / (tw * th);
    std::snprintf(buf, sizeof(buf), "fg: %.1f%%", fg_pct);
    tile_label(t1, "Foreground mask  (TURBO)", {buf});

    // Compose canvas
    cv::Mat canvas(canvas_h, canvas_w, CV_8UC3);
    t0.copyTo(canvas(cv::Rect(0,  0, tw, th)));
    t1.copyTo(canvas(cv::Rect(tw, 0, tw, th)));
    cv::line(canvas, {tw, 0}, {tw, canvas_h}, cv::Scalar(200,200,200), 1);

    std::snprintf(buf, sizeof(buf), "%.1f FPS", fps);
    cv::putText(canvas, buf, {canvas_w-110, 22},
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0,0,0),     3, cv::LINE_AA);
    cv::putText(canvas, buf, {canvas_w-110, 22},
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0,255,255), 1, cv::LINE_AA);
    return canvas;
}

// ============================================================================
// CLI argument parser
// ============================================================================
static Config parse_args(int argc, char** argv)
{
    Config c;
    auto get_next = [&](int i) -> std::string {
        if (i+1 >= argc) throw std::runtime_error("Missing value for " + std::string(argv[i]));
        return argv[i+1];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--width")            { c.width           = std::stoi(get_next(i)); ++i; }
        else if (a == "--height")           { c.height          = std::stoi(get_next(i)); ++i; }
        else if (a == "--fps")              { c.fps             = std::stoi(get_next(i)); ++i; }
        else if (a == "--sensor")           { c.sensor_id       = std::stoi(get_next(i)); ++i; }
        else if (a == "--flip")             { c.flip_method     = std::stoi(get_next(i)); ++i; }
        else if (a == "--udp-host")         { c.udp_host        = get_next(i); ++i; }
        else if (a == "--udp-port")         { c.udp_port        = std::stoi(get_next(i)); ++i; }
        else if (a == "--stream-name")      { c.stream_name     = get_next(i); ++i; }
        else if (a == "--bitrate")          { c.bitrate         = std::stoi(get_next(i)); ++i; }
        else if (a == "--iframeinterval")   { c.iframeinterval  = std::stoi(get_next(i)); ++i; }
        else if (a == "--stream-width")     { c.stream_w        = std::stoi(get_next(i)); ++i; }
        else if (a == "--stream-height")    { c.stream_h        = std::stoi(get_next(i)); ++i; }
        else if (a == "--history")          { c.history         = std::stoi(get_next(i)); ++i; }
        else if (a == "--var-threshold")    { c.var_threshold   = std::stod(get_next(i)); ++i; }
        else if (a == "--detect-shadows")   { c.detect_shadows  = true; }
        else if (a == "--learning-rate")    { c.learning_rate   = std::stod(get_next(i)); ++i; }
        else if (a == "--lock-ae")          { c.lock_ae         = true; }
        else if (a == "--lock-awb")         { c.lock_awb        = true; }
        else if (a == "--no-fps")           { c.show_fps        = false; }
        else if (a == "--help" || a == "-h") {
            std::cout <<
                "cvcuda_bg_subtractor -- OpenCV CUDA MOG2 Background Subtractor\n\n"
                "  --width/height/fps/sensor/flip\n"
                "  --udp-host/udp-port/stream-name/bitrate/iframeinterval\n"
                "  --stream-width/stream-height\n"
                "  --history N         [500]  frames used to build background model\n"
                "  --var-threshold F   [16.0] Mahalanobis distance threshold\n"
                "  --detect-shadows           enable shadow detection (grey pixels)\n"
                "  --learning-rate F   [0.01] background model adaptation speed\n"
                "  --lock-ae  --lock-awb  --no-fps\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown: " << a << "\n"; std::exit(1);
        }
    }
    return c;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[])
{
    gst_init(&argc, &argv);
    std::signal(SIGINT,  sig_handler);
    std::signal(SIGTERM, sig_handler);

    Config cfg = parse_args(argc, argv);

    std::cout << "==================================================\n"
              << "  cvcuda_bg_subtractor -- OpenCV CUDA MOG2\n"
              << "==================================================\n"
              << "  Capture        : " << cfg.width << "x" << cfg.height
              <<                     " @ " << cfg.fps << "fps\n"
              << "  Stream out     : " << cfg.stream_w << "x" << cfg.stream_h << "\n"
              << "  History        : " << cfg.history << " frames\n"
              << "  Var threshold  : " << cfg.var_threshold << "\n"
              << "  Detect shadows : " << (cfg.detect_shadows ? "yes" : "no") << "\n"
              << "  Learning rate  : " << cfg.learning_rate << "\n"
              << "  UDP ->         : " << cfg.udp_host << ":" << cfg.udp_port << "\n"
              << "  Stream         : http://<host>:8889/" << cfg.stream_name << "\n"
              << "==================================================\n\n";

    std::string cap_pipe = capture_pipeline(cfg.sensor_id, cfg.width, cfg.height,
                                            cfg.fps, cfg.flip_method,
                                            cfg.lock_ae, cfg.lock_awb);
    std::cout << "[camera] pipeline: " << cap_pipe << "\n\n";

    CameraBuffer cam_buf;
    std::thread  cam_thr(camera_thread, std::cref(cap_pipe), std::ref(cam_buf));

    CvCudaMOG2 bgsub;
    bgsub.init(cfg);

    std::string push_str = push_pipeline(cfg.stream_w, cfg.stream_h, cfg.fps,
                                         cfg.udp_host, cfg.udp_port,
                                         cfg.bitrate, cfg.iframeinterval);
    std::cout << "[gst-push] pipeline: " << push_str << "\n\n";
    GstPushPipeline gst_push;
    gst_push.start(push_str, cfg.fps);

    cv::Mat cur_bgr;
    auto   t_last     = std::chrono::steady_clock::now();
    long   frame_n    = 0;
    double fps_smooth = 0.0;

    std::cout << "[main] entering loop. Ctrl+C to stop.\n";

    while (!g_stop.load()) {
        if (!cam_buf.get(cur_bgr)) continue;

        cv::Mat fgmask;
        try {
            fgmask = bgsub.process(cur_bgr, cfg.learning_rate);
        } catch (const std::exception& e) {
            std::cerr << "[cuda] process ERROR: " << e.what() << "\n"; continue;
        }

        auto   t_now     = std::chrono::steady_clock::now();
        double dt        = std::chrono::duration<double>(t_now - t_last).count();
        t_last           = t_now;
        fps_smooth       = 0.9*fps_smooth + 0.1*(dt > 0 ? 1.0/dt : 0.0);
        ++frame_n;

        cv::Mat vis = build_sidebyside(cur_bgr, fgmask,
                                       fps_smooth,
                                       cfg.stream_w, cfg.stream_h);
        if (gst_push.ok) gst_push.push(vis);

        if (cfg.show_fps && fps_smooth > 0 &&
            frame_n % static_cast<long>(fps_smooth + 0.5) == 0) {
            std::cout << "\r[main] frame=" << std::setw(6) << frame_n
                      << "  fps=" << std::fixed << std::setprecision(1) << fps_smooth
                      << "          " << std::flush;
        }
    }

    std::cout << "\n[main] shutting down...\n";
    gst_push.stop();
    cam_thr.join();
    std::cout << "[main] done.\n";
    return 0;
}
