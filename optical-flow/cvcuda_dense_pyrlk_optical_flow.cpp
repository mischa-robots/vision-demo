// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Mischa Schaefer <https://github.com/mischa-robots/vision-demo>

/**
 * cvcuda_dense_pyrlk_optical_flow.cpp
 * ------------------------------------
 * Dense Pyramidal LK optical flow for Jetson Nano.
 * Streams a quad-view canvas to the browser via MediaMTX.  No VPI dependency.
 *
 * Algorithm : cv::cuda::DensePyrLKOpticalFlow
 *   Applies the Lucas-Kanade brightness-constancy assumption to every pixel
 *   in the image (not just selected corners) using a Gaussian pyramid for
 *   multi-scale robustness.  Denser than SparsePyrLK but typically slower
 *   and less used in practice than Farneback.
 *
 * Quad-view layout (default 1920x1080):
 *   +---------------------+---------------------+
 *   |  Current frame      |  HSV colour wheel   |
 *   +---------------------+---------------------+
 *   |  Speed  (HOT)       |  Magnitude (INFERNO)|
 *   +---------------------+---------------------+
 *
 * Watch the stream at:
 *   WebRTC : http://<jetson-ip>:8889/optical_flow
 *   HLS    : http://<jetson-ip>:8888/optical_flow
 *   RTSP   : rtsp://<jetson-ip>:8554/optical_flow
 *
 * Usage:
 *   ./build/cvcuda_dense_pyrlk
 *   ./build/cvcuda_dense_pyrlk --help
 */

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include <algorithm>
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
#include <vector>

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
    int         udp_port       = 5601;
    std::string stream_name    = "optical_flow";
    bool        show_fps       = true;
    int         stream_w       = 1920;
    int         stream_h       = 1080;

    // DensePyrLK parameters
    int   win_size    = 13;   // search window size per level
    int   pyr_levels  = 3;    // pyramid levels
    int   iterations  = 30;   // iterations per level

    float mag_clip    = 25.0f;
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
// Dense flow visualisation helpers
//
// Quad-view layout:
//   +---------------------+---------------------+
//   |  Current frame      |  HSV colour wheel   |
//   |                     |  (direction+speed)  |
//   +---------------------+---------------------+
//   |  Speed heatmap      |  Magnitude overlay  |
//   |  (HOT colourmap)    |  (INFERNO colourmap)|
//   +---------------------+---------------------+
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

// Build the 2x2 dense quad canvas from a (H,W,2) float32 flow field.
// flow_cpu must be a contiguous CV_32FC2 Mat at capture resolution.
static cv::Mat build_dense_quad(
    const cv::Mat& cur_bgr,
    const cv::Mat& flow_cpu,   // CV_32FC2, capture res — may be empty on first frame
    double fps,
    float  mag_clip,
    int    canvas_w, int canvas_h)
{
    const int tw = canvas_w / 2, th = canvas_h / 2;
    char buf[64];

    // ---- Tile 0: current frame (top-left) -----------------------------------
    cv::Mat t0;
    cv::resize(cur_bgr, t0, {tw, th}, 0, 0, cv::INTER_AREA);
    tile_label(t0, "Current frame");

    // ---- Tiles 1,2,3: derived from flow field --------------------------------
    cv::Mat t1(th, tw, CV_8UC3, cv::Scalar(20, 20, 20));
    cv::Mat t2(th, tw, CV_8UC3, cv::Scalar(20, 20, 20));
    cv::Mat t3(th, tw, CV_8UC3, cv::Scalar(20, 20, 20));

    if (!flow_cpu.empty()) {
        // Split into x/y planes, resize to tile resolution on CPU.
        // cv::cuda::resize does not support CV_32FC2, so we split first.
        std::vector<cv::Mat> planes(2);
        cv::split(flow_cpu, planes);

        cv::Mat fx_tile, fy_tile;
        cv::resize(planes[0], fx_tile, {tw, th}, 0, 0, cv::INTER_AREA);
        cv::resize(planes[1], fy_tile, {tw, th}, 0, 0, cv::INTER_AREA);

        // Compute magnitude and angle once, reuse for all three tiles.
        cv::Mat mag, ang;
        cv::cartToPolar(fx_tile, fy_tile, mag, ang, false);  // angle in radians

        // ---- Tile 1: HSV colour wheel (top-right) ----------------------------
        // Hue encodes direction (0-180), value encodes speed clamped to mag_clip.
        //
        // NOTE: do NOT normalise value by max_mag. When the scene is static,
        // max_mag is a tiny floating-point residual (e.g. 0.003 px). Dividing
        // by it produces a huge scale factor that overflows CV_8U -> colour
        // trash. On the next frame max_mag rounds to 0.0 -> black. This causes
        // the alternating black / correct / trash pattern visible at low motion.
        // Normalising against mag_clip instead is stable and consistent with
        // how the HOT and INFERNO tiles already work.
        {
            cv::Mat hsv(th, tw, CV_8UC3);
            cv::Mat hue(th, tw, CV_8U);
            cv::Mat val(th, tw, CV_8U);
            // hue: angle in [0, 2*pi) -> [0, 180)
            ang.convertTo(hue, CV_8U, 90.0 / CV_PI);
            // value: magnitude clamped to mag_clip -> [0, 255]
            cv::Mat norm;
            cv::min(mag / mag_clip, 1.f, norm);
            norm.convertTo(val, CV_8U, 255.0);
            cv::Mat sat(th, tw, CV_8U, cv::Scalar(255));
            std::vector<cv::Mat> ch = {hue, sat, val};
            cv::merge(ch, hsv);
            cv::cvtColor(hsv, t1, cv::COLOR_HSV2BGR);

            float mean_mag = static_cast<float>(cv::mean(mag)[0]);
            std::snprintf(buf, sizeof(buf), "mean: %.1f px/frame", mean_mag);
            tile_label(t1, "HSV colour wheel  (direction+speed)", {buf});
        }

        // ---- Tile 2: speed heatmap HOT (bottom-left) ------------------------
        {
            cv::Mat norm;
            cv::min(mag / mag_clip, 1.f, norm);
            cv::Mat norm_u8;
            norm.convertTo(norm_u8, CV_8U, 255.0);
            cv::applyColorMap(norm_u8, t2, cv::COLORMAP_HOT);
            float mean_mag = static_cast<float>(cv::mean(mag)[0]);
            std::snprintf(buf, sizeof(buf), "mean: %.1f px/frame", mean_mag);
            tile_label(t2, "Speed  (HOT)", {buf});
        }

        // ---- Tile 3: INFERNO magnitude overlay (bottom-right) ---------------
        {
            cv::Mat dark;
            cv::Mat cur_small;
            cv::resize(cur_bgr, cur_small, {tw, th}, 0, 0, cv::INTER_AREA);
            cur_small.convertTo(dark, CV_8UC3, 0.35);

            cv::Mat norm;
            cv::min(mag / mag_clip, 1.f, norm);
            cv::Mat norm_u8;
            norm.convertTo(norm_u8, CV_8U, 255.0);
            cv::Mat heat;
            cv::applyColorMap(norm_u8, heat, cv::COLORMAP_INFERNO);

            // alpha blend: dark * (1-alpha) + heat * alpha
            cv::Mat alpha3;
            cv::cvtColor(norm_u8, alpha3, cv::COLOR_GRAY2BGR);
            alpha3.convertTo(alpha3, CV_32FC3, 1.0 / 255.0);
            cv::Mat dark32, heat32;
            dark.convertTo(dark32, CV_32FC3);
            heat.convertTo(heat32, CV_32FC3);
            cv::Mat blend32 = dark32.mul(cv::Scalar(1,1,1) - alpha3)
                            + heat32.mul(alpha3);
            blend32.convertTo(t3, CV_8UC3);
            tile_label(t3, "Magnitude overlay  (INFERNO)");
        }
    } else {
        tile_label(t1, "HSV colour wheel",   {"(waiting for frame 2)"});
        tile_label(t2, "Speed  (HOT)",       {"(waiting for frame 2)"});
        tile_label(t3, "Magnitude  (INFERNO)",{"(waiting for frame 2)"});
    }

    // ---- Composite ----------------------------------------------------------
    cv::Mat canvas(canvas_h, canvas_w, CV_8UC3);
    t0.copyTo(canvas(cv::Rect(0,  0,  tw, th)));
    t1.copyTo(canvas(cv::Rect(tw, 0,  tw, th)));
    t2.copyTo(canvas(cv::Rect(0,  th, tw, th)));
    t3.copyTo(canvas(cv::Rect(tw, th, tw, th)));
    cv::line(canvas, {tw,0},     {tw,canvas_h}, cv::Scalar(200,200,200), 1);
    cv::line(canvas, {0,th},     {canvas_w,th}, cv::Scalar(200,200,200), 1);
    std::snprintf(buf, sizeof(buf), "%.1f FPS", fps);
    cv::putText(canvas, buf, {canvas_w-110,22},
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0,0,0),     3, cv::LINE_AA);
    cv::putText(canvas, buf, {canvas_w-110,22},
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
                if      (a == "--width")          { c.width          = std::stoi(get_next(i)); ++i; }
        else if (a == "--height")         { c.height         = std::stoi(get_next(i)); ++i; }
        else if (a == "--fps")            { c.fps            = std::stoi(get_next(i)); ++i; }
        else if (a == "--sensor")         { c.sensor_id      = std::stoi(get_next(i)); ++i; }
        else if (a == "--flip")           { c.flip_method    = std::stoi(get_next(i)); ++i; }
        else if (a == "--udp-host")       { c.udp_host       = get_next(i); ++i; }
        else if (a == "--udp-port")       { c.udp_port       = std::stoi(get_next(i)); ++i; }
        else if (a == "--stream-name")    { c.stream_name    = get_next(i); ++i; }
        else if (a == "--bitrate")        { c.bitrate        = std::stoi(get_next(i)); ++i; }
        else if (a == "--iframeinterval") { c.iframeinterval = std::stoi(get_next(i)); ++i; }
        else if (a == "--stream-width")   { c.stream_w       = std::stoi(get_next(i)); ++i; }
        else if (a == "--stream-height")  { c.stream_h       = std::stoi(get_next(i)); ++i; }
        else if (a == "--mag-clip")       { c.mag_clip       = std::stof(get_next(i)); ++i; }
        else if (a == "--lock-ae")        { c.lock_ae        = true; }
        else if (a == "--lock-awb")       { c.lock_awb       = true; }
        else if (a == "--no-fps")         { c.show_fps       = false; }

        else if (a == "--win-size")   { c.win_size   = std::stoi(get_next(i)); ++i; }
        else if (a == "--pyr-levels") { c.pyr_levels = std::stoi(get_next(i)); ++i; }
        else if (a == "--iterations") { c.iterations = std::stoi(get_next(i)); ++i; }
        else if (a == "--help" || a == "-h") {
            std::cout <<
                "cvcuda_dense_pyrlk -- OpenCV CUDA Dense Pyramidal LK\n\n"
                "  --width/height/fps/sensor/flip\n"
                "  --udp-host/udp-port/stream-name/bitrate/iframeinterval\n"
                "  --stream-width/stream-height\n"
                "  --win-size N     [13]   search window size per pyramid level\n"
                "  --pyr-levels N   [3]    pyramid depth\n"
                "  --iterations N   [30]   iterations per pyramid level\n"
                "  --mag-clip F     [25]   speed saturation for visualisation\n"
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
              << "  cvcuda_dense_pyrlk -- OpenCV CUDA Dense PyrLK\n"
              << "==================================================\n"
              << "  Capture    : " << cfg.width << "x" << cfg.height
              <<                  " @ " << cfg.fps << "fps\n"
              << "  Stream out : " << cfg.stream_w << "x" << cfg.stream_h << "\n"
              << "  Win size   : " << cfg.win_size << "\n"
              << "  Pyr levels : " << cfg.pyr_levels << "\n"
              << "  Iterations : " << cfg.iterations << "\n"
              << "  Mag clip   : " << cfg.mag_clip << " px/frame\n"
              << "  UDP ->     : " << cfg.udp_host << ":" << cfg.udp_port << "\n"
              << "  Stream     : http://<host>:8889/" << cfg.stream_name << "\n"
              << "==================================================\n\n";

    std::string cap_pipe = capture_pipeline(cfg.sensor_id, cfg.width, cfg.height,
                                            cfg.fps, cfg.flip_method,
                                            cfg.lock_ae, cfg.lock_awb);
    std::cout << "[camera] pipeline: " << cap_pipe << "\n\n";

    CameraBuffer cam_buf;
    std::thread  cam_thr(camera_thread, std::cref(cap_pipe), std::ref(cam_buf));

    auto of = cv::cuda::DensePyrLKOpticalFlow::create(
        cv::Size(cfg.win_size, cfg.win_size),
        cfg.pyr_levels,
        cfg.iterations);
    std::cout << "[dense-pyrlk] created.\n";

    std::string push_str = push_pipeline(cfg.stream_w, cfg.stream_h, cfg.fps,
                                         cfg.udp_host, cfg.udp_port,
                                         cfg.bitrate, cfg.iframeinterval);
    std::cout << "[gst-push] pipeline: " << push_str << "\n\n";
    GstPushPipeline gst_push;
    gst_push.start(push_str, cfg.fps);

    cv::Mat cur_bgr;
    cv::cuda::GpuMat prev_gray_gpu, cur_gray_gpu;
    auto   t_last     = std::chrono::steady_clock::now();
    long   frame_n    = 0;
    double fps_smooth = 0.0;
    bool   have_prev  = false;

    std::cout << "[main] entering loop. Ctrl+C to stop.\n";

    while (!g_stop.load()) {
        if (!cam_buf.get(cur_bgr)) continue;

        cv::Mat cur_gray;
        cv::cvtColor(cur_bgr, cur_gray, cv::COLOR_BGR2GRAY);

        cv::Mat flow_cpu;  // CV_32FC2, filled when we have two frames

        if (have_prev) {
            cur_gray_gpu.upload(cur_gray);
            cv::cuda::GpuMat flow_gpu;
            of->calc(prev_gray_gpu, cur_gray_gpu, flow_gpu);
            flow_gpu.download(flow_cpu);
        } else {
            cur_gray_gpu.upload(cur_gray);
            have_prev = true;
        }

        auto t_now  = std::chrono::steady_clock::now();
        double dt   = std::chrono::duration<double>(t_now - t_last).count();
        t_last      = t_now;
        fps_smooth  = 0.9*fps_smooth + 0.1*(dt > 0 ? 1.0/dt : 0.0);
        ++frame_n;

        cv::Mat vis = build_dense_quad(cur_bgr, flow_cpu, fps_smooth,
                                       cfg.mag_clip, cfg.stream_w, cfg.stream_h);
        if (gst_push.ok) gst_push.push(vis);

        if (cfg.show_fps && fps_smooth > 0 &&
            frame_n % static_cast<long>(fps_smooth + 0.5) == 0) {
            std::cout << "\r[main] frame=" << std::setw(6) << frame_n
                      << "  fps=" << std::fixed << std::setprecision(1) << fps_smooth
                      << "          " << std::flush;
        }

        // Advance: cur becomes prev (clone so they are independent allocations)
        prev_gray_gpu = cur_gray_gpu.clone();
    }


    std::cout << "\n[main] shutting down...\n";
    gst_push.stop();
    cam_thr.join();
    std::cout << "[main] done.\n";
    return 0;
}
