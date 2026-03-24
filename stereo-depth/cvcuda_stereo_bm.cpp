// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>

/**
 * cvcuda_stereo_bm.cpp
 * --------------------
 * Stereo depth demo for Jetson Nano using OpenCV CUDA — streams quad-view
 * to browser via MediaMTX. No VPI dependency.
 *
 * Algorithm : OpenCV CUDA StereoBM
 *   Fastest stereo algorithm. Compares fixed-size blocks between left and right images. Works well on textured surfaces; struggles with textureless regions (sky, walls).
 *
 * Quad-view layout (default 1920x1080):
 *   +------------------+------------------+
 *   |  Left (rect)     |  Right (rect)    |
 *   +------------------+------------------+
 *   |  Disparity (TURBO) | Depth (TURBO) |
 *   +------------------+------------------+
 *
 * Watch the stream at:
 *   WebRTC : http://<jetson-ip>:8889/stereo_depth
 *   HLS    : http://<jetson-ip>:8888/stereo_depth
 *   RTSP   : rtsp://<jetson-ip>:8554/stereo_depth
 *
 * Usage:
 *   ./build/cvcuda_stereo_bm --calib stereo_calibration.yaml
 *   ./build/cvcuda_stereo_bm --help
 */

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/cudastereo.hpp>

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
// GStreamer pipeline strings
// ============================================================================
static std::string capture_pipeline(int sensor_id, int w, int h, int fps, int flip)
{
    std::ostringstream ss;
    ss << "nvarguscamerasrc sensor-id=" << sensor_id
       << " ! video/x-raw(memory:NVMM)"
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
// StereoCalib — loads stereo_calibration.yaml and builds rectification maps
// ============================================================================
struct StereoCalib {
    cv::Mat mapLx, mapLy, mapRx, mapRy;
    double  focal_px   = 0.0;
    double  baseline_m = 0.0;

    void load(const std::string& path, int w, int h)
    {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened())
            throw std::runtime_error("Cannot open calibration file: " + path);

        auto read_mat = [&](const std::string& key) -> cv::Mat {
            cv::Mat m; fs[key] >> m;
            if (m.empty())
                throw std::runtime_error("Missing key '" + key + "' in " + path);
            return m;
        };

        cv::Mat mtxL  = read_mat("mtxL");
        cv::Mat distL = read_mat("distL");
        cv::Mat mtxR  = read_mat("mtxR");
        cv::Mat distR = read_mat("distR");
        cv::Mat R1    = read_mat("R1");
        cv::Mat P1    = read_mat("P1");
        cv::Mat R2    = read_mat("R2");
        cv::Mat P2    = read_mat("P2");
        fs.release();

        cv::initUndistortRectifyMap(mtxL, distL, R1, P1,
            cv::Size(w, h), CV_16SC2, mapLx, mapLy);
        cv::initUndistortRectifyMap(mtxR, distR, R2, P2,
            cv::Size(w, h), CV_16SC2, mapRx, mapRy);

        if (!P1.empty() && !P2.empty()) {
            focal_px   = P1.at<double>(0, 0);
            baseline_m = std::abs(P2.at<double>(0, 3) / focal_px);
        }

        std::cout << "[calib] loaded " << path
                  << "  f=" << std::fixed << std::setprecision(0) << focal_px
                  << "px  B=" << std::setprecision(1) << baseline_m * 100.0
                  << "cm\n";
    }

    void rectify(const cv::Mat& left, const cv::Mat& right,
                 cv::Mat& rLeft, cv::Mat& rRight) const
    {
        cv::remap(left,  rLeft,  mapLx, mapLy, cv::INTER_LINEAR);
        cv::remap(right, rRight, mapRx, mapRy, cv::INTER_LINEAR);
    }
};

// ============================================================================
// CameraBuffer — thread-safe double-buffer for stereo frame pairs
// ============================================================================
struct CameraBuffer {
    std::mutex              mtx;
    std::condition_variable cv;
    cv::Mat                 bufL, bufR;
    bool                    valid = false;

    void put(cv::Mat&& l, cv::Mat&& r) {
        { std::lock_guard<std::mutex> lk(mtx);
          bufL = std::move(l); bufR = std::move(r); valid = true; }
        cv.notify_one();
    }
    bool get(cv::Mat& l, cv::Mat& r,
             std::chrono::milliseconds timeout = std::chrono::milliseconds(500)) {
        std::unique_lock<std::mutex> lk(mtx);
        if (!cv.wait_for(lk, timeout, [this]{ return valid; })) return false;
        l = bufL.clone(); r = bufR.clone(); valid = false; return true;
    }
};

static void camera_thread(const std::string& pipeL, const std::string& pipeR,
                           CameraBuffer& cam_buf)
{
    cv::VideoCapture capL(pipeL, cv::CAP_GSTREAMER);
    cv::VideoCapture capR(pipeR, cv::CAP_GSTREAMER);
    if (!capL.isOpened() || !capR.isOpened()) {
        std::cerr << "[camera] FATAL: could not open one or both cameras.\n";
        g_stop.store(true); return;
    }
    std::cout << "[camera] both cameras opened.\n";
    cv::Mat l, r;
    while (!g_stop.load()) {
        if (!capL.read(l) || !capR.read(r) || l.empty() || r.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); continue;
        }
        cam_buf.put(std::move(l), std::move(r));
    }
    capL.release(); capR.release();
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
// Config
// ============================================================================
struct Config {
    std::string calib_file     = "stereo_calibration.yaml";
    int         width          = 1280;
    int         height         = 720;
    int         fps            = 15;
    int         stereo_w       = 640;
    int         stereo_h       = 360;
    int         max_disp       = 64;
    int         flip_method    = 0;
    int         bitrate        = 4000000;
    int         iframeinterval = 30;
    std::string udp_host       = "127.0.0.1";
    int         udp_port       = 5600;
    std::string stream_name    = "stereo_depth";
    bool        swap_cameras   = false;
    bool        no_rectify     = false;
    bool        show_fps       = true;
    int         stream_w       = 1920;
    int         stream_h       = 1080;
    // Bilateral filter post-processing (0 = disabled)
    // Smooths disparity while preserving depth edges using the rectified
    // left image as a guide. Uses cv::ximgproc::jointBilateralFilter.
    bool   bilateral_filter = false;
    int    bilateral_d      = 9;      // neighbourhood diameter
    double bilateral_sc     = 75.0;   // sigma in colour space
    double bilateral_ss     = 75.0;   // sigma in coordinate space
    // StereoBM parameters
    int block_size  = 15;   // block size for matching (odd, >= 5)
};

// ============================================================================
// StereoBM — block matching, fastest CUDA stereo algorithm
//
// Input must be grayscale CV_8UC1. Disparity output is CV_16SC1 (fixed-point,
// divide by 16 to get pixel values). numDisparities must be a multiple of 8,
// blockSize must be odd.
// ============================================================================
static cv::Ptr<cv::cuda::StereoBM> g_bm;

static void init_stereo(const Config& cfg)
{
    // numDisparities must be multiple of 8, block_size must be odd
    int nd = ((cfg.max_disp + 7) / 8) * 8;
    g_bm = cv::cuda::createStereoBM(nd, cfg.block_size);
    std::cout << "[stereo-bm] created  numDisp=" << nd
              << "  blockSize=" << cfg.block_size << "\n";
}

static cv::Mat compute_disparity(cv::Mat& left_rect, cv::Mat& right_rect,
                                  const Config& cfg)
{
    cv::Mat gL, gR;
    cv::cvtColor(left_rect,  gL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_rect, gR, cv::COLOR_BGR2GRAY);

    cv::cuda::GpuMat gpuL(gL), gpuR(gR), gpuDisp;
    g_bm->compute(gpuL, gpuR, gpuDisp);

    cv::Mat disp16; gpuDisp.download(disp16);
    cv::Mat disp_f32;
    disp16.convertTo(disp_f32, CV_32F, 1.0);

    // BM marks invalid pixels as -1 (negative) — zero them out
    cv::threshold(disp_f32, disp_f32, 0.0, 0.0, cv::THRESH_TOZERO);

    // Resize to full stereo resolution if needed
    if (disp_f32.size() != cv::Size(cfg.stereo_w, cfg.stereo_h))
        cv::resize(disp_f32, disp_f32, {cfg.stereo_w, cfg.stereo_h},
                   0, 0, cv::INTER_NEAREST);
    return disp_f32;
}

// ============================================================================
// Visualisation helpers — shared by all stereo demos
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

// Green horizontal epipolar lines — after good rectification all features
// in both cameras should lie exactly on these lines
static void draw_epipolar(cv::Mat& tile, int step = 40)
{
    for (int y = step; y < tile.rows; y += step)
        cv::line(tile, {0, y}, {tile.cols, y}, cv::Scalar(0,200,0), 1);
}

// Disparity float32 -> TURBO colourmap (black = no data)
static cv::Mat disp_to_colour(const cv::Mat& disp_f32, int tw, int th, int max_disp)
{
    cv::Mat vis8;
    disp_f32.convertTo(vis8, CV_8UC1, 255.0 / static_cast<double>(max_disp));
    cv::Mat colour;
    cv::applyColorMap(vis8, colour, cv::COLORMAP_TURBO);
    colour.setTo(cv::Scalar(0,0,0), (vis8 == 0));
    if (colour.cols != tw || colour.rows != th)
        cv::resize(colour, colour, {tw,th}, 0, 0, cv::INTER_NEAREST);
    return colour;
}

// Depth (metres) -> TURBO colourmap, clamped 0.1-10 m, warm=near
static cv::Mat depth_to_colour(const cv::Mat& disp_f32, int tw, int th,
                                double focal_px, double baseline_m)
{
    cv::Mat depth(disp_f32.size(), CV_32F, cv::Scalar(0));
    if (focal_px > 0 && baseline_m > 0) {
        cv::Mat valid = (disp_f32 > 0.5f);
        cv::divide(focal_px * baseline_m, disp_f32, depth, 1.0, CV_32F);
        depth.setTo(0, ~valid);
        cv::threshold(depth, depth, 10.0, 0, cv::THRESH_TOZERO_INV);
        cv::threshold(depth, depth, 0.1,  0, cv::THRESH_TOZERO);
    }
    cv::Mat valid_mask = (depth > 0.0f);
    cv::Mat norm;
    depth.convertTo(norm, CV_32F);
    norm = 1.0f - cv::min(cv::max((norm - 0.1f) / 9.9f, 0.0f), 1.0f);
    cv::Mat u8; norm.convertTo(u8, CV_8UC1, 255.0);
    cv::Mat colour; cv::applyColorMap(u8, colour, cv::COLORMAP_TURBO);
    colour.setTo(cv::Scalar(0,0,0), ~valid_mask);
    if (colour.cols != tw || colour.rows != th)
        cv::resize(colour, colour, {tw,th}, 0, 0, cv::INTER_NEAREST);
    return colour;
}

// Median depth at centre of disparity map -- useful sanity check
static float median_depth(const cv::Mat& disp_f32, double focal_px, double baseline_m)
{
    std::vector<float> vals;
    vals.reserve(disp_f32.total());
    for (int r = 0; r < disp_f32.rows; ++r) {
        const float* row = disp_f32.ptr<float>(r);
        for (int c = 0; c < disp_f32.cols; ++c)
            if (row[c] > 0.5f) vals.push_back(row[c]);
    }
    if (vals.empty()) return 0.0f;
    std::nth_element(vals.begin(), vals.begin() + vals.size()/2, vals.end());
    float med_d = vals[vals.size()/2];
    return static_cast<float>(focal_px * baseline_m / med_d);
}

// Optional joint bilateral filter to smooth disparity while preserving edges.
// Applied on CPU to the float32 disparity map before visualisation.
// Joint bilateral filter using the rectified left image as a spatial guide.
// Smooths the disparity map while preserving edges aligned with image edges.
// Uses cv::bilateralFilter on the disparity directly (guide-free fallback
// since ximgproc is not always built into the Nano's OpenCV).
// For a true joint/cross bilateral filter, replace with
// cv::ximgproc::jointBilateralFilter if your OpenCV build includes ximgproc.
static cv::Mat apply_bilateral(const cv::Mat& disp_f32, const cv::Mat& /*guide*/,
                                int d, double sigma_color, double sigma_space)
{
    double max_val = 0.0;
    cv::minMaxLoc(disp_f32, nullptr, &max_val);
    if (max_val < 1.0) return disp_f32;   // no data, skip

    // bilateralFilter works on CV_8U or CV_32F single-channel
    cv::Mat filtered;
    cv::bilateralFilter(disp_f32, filtered, d,
                        sigma_color * max_val / 255.0,   // scale sigma to float range
                        sigma_space);
    return filtered;
}

static cv::Mat build_quad(
    const cv::Mat& left_rect, const cv::Mat& right_rect,
    const cv::Mat& disp_f32,
    double fps, double focal_px, double baseline_m,
    int max_disp, int canvas_w, int canvas_h,
    const std::string& algo_label)
{
    const int tw = canvas_w/2, th = canvas_h/2;
    const cv::Size tsz(tw, th);
    char buf[64];

    cv::Mat valid_mask = (disp_f32 > 0.5f);
    double valid_pct = 100.0 * cv::countNonZero(valid_mask) /
                       (disp_f32.cols * disp_f32.rows);
    float med_z = median_depth(disp_f32, focal_px, baseline_m);

    cv::Mat t0; cv::resize(left_rect,  t0, tsz, 0, 0, cv::INTER_AREA);
    draw_epipolar(t0);
    tile_label(t0, "Left (rectified)", {algo_label});

    cv::Mat t1; cv::resize(right_rect, t1, tsz, 0, 0, cv::INTER_AREA);
    draw_epipolar(t1);
    tile_label(t1, "Right (rectified)");

    cv::Mat t2 = disp_to_colour(disp_f32, tw, th, max_disp);
    std::snprintf(buf, sizeof(buf), "valid: %.0f%%", valid_pct);
    tile_label(t2, "Disparity  (red=near)", {buf});

    cv::Mat t3 = depth_to_colour(disp_f32, tw, th, focal_px, baseline_m);
    if (med_z > 0) std::snprintf(buf, sizeof(buf), "med: %.2f m", med_z);
    else            std::snprintf(buf, sizeof(buf), "no depth data");
    tile_label(t3, "Depth  (warm=near)", {buf});

    cv::Mat canvas(canvas_h, canvas_w, CV_8UC3);
    t0.copyTo(canvas(cv::Rect(0,  0,  tw, th)));
    t1.copyTo(canvas(cv::Rect(tw, 0,  tw, th)));
    t2.copyTo(canvas(cv::Rect(0,  th, tw, th)));
    t3.copyTo(canvas(cv::Rect(tw, th, tw, th)));
    cv::line(canvas, {tw,0},    {tw,canvas_h}, cv::Scalar(200,200,200), 1);
    cv::line(canvas, {0,th},    {canvas_w,th}, cv::Scalar(200,200,200), 1);
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
                if      (a == "--calib")          { c.calib_file     = get_next(i); ++i; }
        else if (a == "--width")          { c.width          = std::stoi(get_next(i)); ++i; }
        else if (a == "--height")         { c.height         = std::stoi(get_next(i)); ++i; }
        else if (a == "--fps")            { c.fps            = std::stoi(get_next(i)); ++i; }
        else if (a == "--stereo-width")   { c.stereo_w       = std::stoi(get_next(i)); ++i; }
        else if (a == "--stereo-height")  { c.stereo_h       = std::stoi(get_next(i)); ++i; }
        else if (a == "--max-disp")       { c.max_disp       = std::stoi(get_next(i)); ++i; }
        else if (a == "--flip")           { c.flip_method    = std::stoi(get_next(i)); ++i; }
        else if (a == "--udp-host")       { c.udp_host       = get_next(i); ++i; }
        else if (a == "--udp-port")       { c.udp_port       = std::stoi(get_next(i)); ++i; }
        else if (a == "--stream-name")    { c.stream_name    = get_next(i); ++i; }
        else if (a == "--bitrate")        { c.bitrate        = std::stoi(get_next(i)); ++i; }
        else if (a == "--iframeinterval") { c.iframeinterval = std::stoi(get_next(i)); ++i; }
        else if (a == "--stream-width")   { c.stream_w       = std::stoi(get_next(i)); ++i; }
        else if (a == "--stream-height")  { c.stream_h       = std::stoi(get_next(i)); ++i; }
        else if (a == "--swap-cameras")   { c.swap_cameras   = true; }
        else if (a == "--no-rectify")     { c.no_rectify     = true; }
        else if (a == "--no-fps")         { c.show_fps       = false; }
        else if (a == "--bilateral-filter") { c.bilateral_filter = true; }
        else if (a == "--bilateral-d")    { c.bilateral_d    = std::stoi(get_next(i)); ++i; }
        else if (a == "--bilateral-sc")   { c.bilateral_sc   = std::stod(get_next(i)); ++i; }
        else if (a == "--bilateral-ss")   { c.bilateral_ss   = std::stod(get_next(i)); ++i; }
        else if (a == "--block-size")  { c.block_size = std::stoi(get_next(i)); ++i; }
        else if (a == "--help" || a == "-h") {
            std::cout <<
                "cvcuda_stereo_bm -- OpenCV CUDA Block Matching stereo depth\n\n"
                "  --calib FILE          Stereo calibration YAML        [stereo_calibration.yaml]\n"
                "  --width/height/fps/flip\n"
                "  --stereo-width/stereo-height N  Processing res        [640x360]\n"
                "  --max-disp N          Max disparity (must be mult of 16 for BM/SGM) [64]\n"
                "  --block-size N        Matching block size (odd, >=5)  [15]\n"

                "  --bilateral-filter    Enable bilateral post-filter\n"
                "  --bilateral-d N       Filter diameter                [9]\n"
                "  --bilateral-sc F      Sigma colour                   [75]\n"
                "  --bilateral-ss F      Sigma space                    [75]\n"
                "  --udp-host/udp-port/stream-name/bitrate/iframeinterval\n"
                "  --stream-width/stream-height\n"
                "  --swap-cameras  --no-rectify  --no-fps\n";
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
              << "  cvcuda_stereo_bm -- OpenCV CUDA StereoBM\n"
              << "==================================================\n"
                            << "  Calib       : " << cfg.calib_file << "\n"
              << "  Capture     : " << cfg.width << "x" << cfg.height
              <<                   " @ " << cfg.fps << "fps\n"
              << "  Stereo res  : " << cfg.stereo_w << "x" << cfg.stereo_h << "\n"
              << "  Max disp    : " << cfg.max_disp << "\n"
              << "  Bilateral   : " << (cfg.bilateral_filter ? "yes" : "no") << "\n"
              << "  Swap cams   : " << cfg.swap_cameras << "\n"
              << "  Rectify     : " << !cfg.no_rectify << "\n"
              << "  UDP ->      : " << cfg.udp_host << ":" << cfg.udp_port << "\n"
              << "  Stream      : http://<host>:8889/" << cfg.stream_name << "\n"

              << "  Block size  : " << cfg.block_size << "\n"

              << "==================================================\n\n";

    StereoCalib calib;
    if (!cfg.no_rectify) {
        try { calib.load(cfg.calib_file, cfg.width, cfg.height); }
        catch (const std::exception& e) {
            std::cerr << "[calib] ERROR: " << e.what() << "\n"; return 1;
        }
    }

    int sensor_L = cfg.swap_cameras ? 1 : 0;
    int sensor_R = cfg.swap_cameras ? 0 : 1;
    std::string pipe_L = capture_pipeline(sensor_L, cfg.width, cfg.height, cfg.fps, cfg.flip_method);
    std::string pipe_R = capture_pipeline(sensor_R, cfg.width, cfg.height, cfg.fps, cfg.flip_method);
    std::cout << "[camera] left  pipeline: " << pipe_L << "\n";
    std::cout << "[camera] right pipeline: " << pipe_R << "\n\n";

    CameraBuffer cam_buf;
    std::thread  cam_thr(camera_thread, std::cref(pipe_L), std::cref(pipe_R), std::ref(cam_buf));

    init_stereo(cfg);

    std::string push_str = push_pipeline(cfg.stream_w, cfg.stream_h, cfg.fps,
                                         cfg.udp_host, cfg.udp_port,
                                         cfg.bitrate, cfg.iframeinterval);
    std::cout << "[gst-push] pipeline: " << push_str << "\n\n";
    GstPushPipeline gst_push;
    gst_push.start(push_str, cfg.fps);

    cv::Mat left_raw, right_raw, left_rect, right_rect;
    auto   t_last     = std::chrono::steady_clock::now();
    long   frame_n    = 0;
    double fps_smooth = 0.0;

    std::cout << "[main] entering loop. Ctrl+C to stop.\n";

        while (!g_stop.load()) {
        if (!cam_buf.get(left_raw, right_raw)) continue;

        if (!cfg.no_rectify)
            calib.rectify(left_raw, right_raw, left_rect, right_rect);
        else {
            left_rect  = left_raw;
            right_rect = right_raw;
        }

        cv::Mat disp_f32 = compute_disparity(left_rect, right_rect, cfg);

        // Optional joint bilateral filter -- smooths disparity, preserves edges
        if (cfg.bilateral_filter && !disp_f32.empty()) {
            cv::Mat guide_gray;
            cv::cvtColor(left_rect, guide_gray, cv::COLOR_BGR2GRAY);
            disp_f32 = apply_bilateral(disp_f32, guide_gray,
                                       cfg.bilateral_d,
                                       cfg.bilateral_sc,
                                       cfg.bilateral_ss);
        }

        auto t_now  = std::chrono::steady_clock::now();
        double dt   = std::chrono::duration<double>(t_now - t_last).count();
        t_last      = t_now;
        fps_smooth  = 0.9*fps_smooth + 0.1*(dt > 0 ? 1.0/dt : 0.0);
        ++frame_n;

        cv::Mat vis = build_quad(left_rect, right_rect, disp_f32,
                                 fps_smooth, calib.focal_px, calib.baseline_m,
                                 cfg.max_disp, cfg.stream_w, cfg.stream_h,
                                 "OpenCV CUDA StereoBM");
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
