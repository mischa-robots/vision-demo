// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Mischa Schaefer <https://github.com/mischa-robots/vision-demo>

/**
 * vpi_optical_flow.cpp
 * ---------------------
 * Sparse Pyramidal LK optical flow for Jetson Nano using NVIDIA VPI.
 * Streams a quad-view canvas to the browser via MediaMTX.
 *
 * Algorithm : VPI Harris corner detection (CPU) + OpticalFlowPyrLK (CUDA)
 *
 * WARNING: vpiSubmitOpticalFlowPyrLK has a known NVIDIA bug on JetPack 4.6.6 /
 * VPI 1.2.x / CUDA 10.2 where tracked points are not updated after the first
 * frame. This demo is kept as a VPI pipeline reference only. For correct
 * optical flow use cvcuda_sparse_pyrlk instead.
 *
 * Quad-view layout (default 1920x1080):
 *   +---------------------+---------------------+
 *   |  Current + trails   |  Previous + dots    |
 *   +---------------------+---------------------+
 *   |  Flow arrows        |  Speed heatmap      |
 *   +---------------------+---------------------+
 *
 * Watch the stream at:
 *   WebRTC : http://<jetson-ip>:8889/optical_flow
 *   HLS    : http://<jetson-ip>:8888/optical_flow
 *   RTSP   : rtsp://<jetson-ip>:8554/optical_flow
 *
 * Usage:
 *   ./build/vpi_optical_flow
 *   ./build/vpi_optical_flow --help
 */

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <vpi/OpenCVInterop.hpp>
#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/HarrisCorners.h>
#include <vpi/algo/OpticalFlowPyrLK.h>

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
#include <numeric>
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
// VPI error helper
// ============================================================================
static void vpi_check(VPIStatus s, const char* file, int line)
{
    if (s == VPI_SUCCESS) return;
    char buf[VPI_MAX_STATUS_MESSAGE_LENGTH];
    vpiGetLastStatusMessage(buf, sizeof(buf));
    std::ostringstream ss;
    ss << "[VPI] " << vpiStatusGetName(s) << ": " << buf
       << "  (" << file << ":" << line << ")";
    throw std::runtime_error(ss.str());
}
#define VPI_CHECK(x) vpi_check((x), __FILE__, __LINE__)

// ============================================================================
// RAII wrappers for VPI handles
// ============================================================================
struct VpiImage {
    VPIImage h = nullptr;
    VpiImage() = default;
    explicit VpiImage(VPIImage h_) : h(h_) {}
    ~VpiImage() { if (h) vpiImageDestroy(h); }
    VpiImage(const VpiImage&) = delete;
    VpiImage& operator=(const VpiImage&) = delete;
    VpiImage(VpiImage&& o) noexcept : h(o.h) { o.h = nullptr; }
    VpiImage& operator=(VpiImage&& o) noexcept {
        if (h) vpiImageDestroy(h);
        h = o.h;
        o.h = nullptr;
        return *this;
    }
    operator VPIImage() const { return h; }
};
struct VpiStream {
    VPIStream h = nullptr;
    VpiStream() = default;
    explicit VpiStream(VPIStream h_) : h(h_) {}
    ~VpiStream() { if (h) vpiStreamDestroy(h); }
    VpiStream(const VpiStream&) = delete;
    VpiStream& operator=(const VpiStream&) = delete;
    VpiStream(VpiStream&& o) noexcept : h(o.h) { o.h = nullptr; }
    VpiStream& operator=(VpiStream&& o) noexcept {
        if (h) vpiStreamDestroy(h);
        h = o.h;
        o.h = nullptr;
        return *this;
    }
    operator VPIStream() const { return h; }
};
struct VpiPayload {
    VPIPayload h = nullptr;
    VpiPayload() = default;
    explicit VpiPayload(VPIPayload h_) : h(h_) {}
    ~VpiPayload() { if (h) vpiPayloadDestroy(h); }
    VpiPayload(const VpiPayload&) = delete;
    VpiPayload& operator=(const VpiPayload&) = delete;
    VpiPayload(VpiPayload&& o) noexcept : h(o.h) { o.h = nullptr; }
    VpiPayload& operator=(VpiPayload&& o) noexcept {
        if (h) vpiPayloadDestroy(h);
        h = o.h;
        o.h = nullptr;
        return *this;
    }
    operator VPIPayload() const { return h; }
};
struct VpiArray {
    VPIArray h = nullptr;
    VpiArray() = default;
    explicit VpiArray(VPIArray h_) : h(h_) {}
    ~VpiArray() { if (h) vpiArrayDestroy(h); }
    VpiArray(const VpiArray&) = delete;
    VpiArray& operator=(const VpiArray&) = delete;
    VpiArray(VpiArray&& o) noexcept : h(o.h) { o.h = nullptr; }
    VpiArray& operator=(VpiArray&& o) noexcept {
        if (h) vpiArrayDestroy(h);
        h = o.h;
        o.h = nullptr;
        return *this;
    }
    operator VPIArray() const { return h; }
};
struct VpiPyramid {
    VPIPyramid h = nullptr;
    VpiPyramid() = default;
    explicit VpiPyramid(VPIPyramid h_) : h(h_) {}
    ~VpiPyramid() { if (h) vpiPyramidDestroy(h); }
    VpiPyramid(const VpiPyramid&) = delete;
    VpiPyramid& operator=(const VpiPyramid&) = delete;
    VpiPyramid(VpiPyramid&& o) noexcept : h(o.h) { o.h = nullptr; }
    VpiPyramid& operator=(VpiPyramid&& o) noexcept {
        if (h) vpiPyramidDestroy(h);
        h = o.h;
        o.h = nullptr;
        return *this;
    }
    operator VPIPyramid() const { return h; }
};

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

    int         pyr_levels     = 5;
    int         max_points     = 200;
    int         min_points     = 20;
    int         lk_window      = 15;
    int         lk_iterations  = 20;
    float       lk_epsilon     = 0.01f;

    float       harris_sensitivity = 0.01f;

    float       mag_clip    = 25.0f;
    int         arrow_scale = 4;
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
// TrackPoint + colour helper
// ============================================================================
struct TrackPoint {
    cv::Scalar  colour;
    cv::Point2f prev;
    cv::Point2f cur;
    bool        valid;
};

static cv::Scalar point_colour(float x, float y)
{
    uint8_t hue = static_cast<uint8_t>((static_cast<int>(x) ^ static_cast<int>(y)) % 180);
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 220, 220));
    cv::Mat bgr; cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    auto* p = bgr.ptr<uint8_t>(0);
    return cv::Scalar(p[0], p[1], p[2]);
}

// ============================================================================
// VpiLK tracker
//
// Harris corners (CPU) -> Gaussian pyramid (CUDA) -> OpticalFlowPyrLK (CUDA)
//
// NOTE: vpiSubmitOpticalFlowPyrLK has a known bug on JetPack 4.6.6 / VPI 1.2
// where kpts_cur is not updated after the first frame. The pipeline structure
// is correct and follows the NVIDIA sample exactly -- the bug is in the VPI
// runtime itself.
// ============================================================================
struct VpiLK {
    VpiStream  stream;
    VpiPayload lk_payload;
    VpiPayload harris_payload;

    VpiImage   img_wrap;
    VpiImage   img_u8;
    VpiPyramid pyr_prev;
    VpiPyramid pyr_cur;
    VpiArray   kpts_prev;
    VpiArray   kpts_cur;
    VpiArray   kpts_status;
    VpiArray   kpts_scores;

    std::vector<cv::Scalar> colours;
    int   W = 0, H = 0;
    int   max_points = 200;
    int   min_points = 20;
    float harris_sensitivity = 0.01f;
    int   n_reinit_features  = 0;

    VPIOpticalFlowPyrLKParams lk_params;
    cv::Mat cv_wrap_buf;

    void init(int w, int h, const Config& cfg)
    {
        W = w; H = h;
        max_points         = cfg.max_points;
        min_points         = cfg.min_points;
        harris_sensitivity = cfg.harris_sensitivity;

        { VPIStream s; VPI_CHECK(vpiStreamCreate(0, &s)); stream = VpiStream(s); }
        VPI_CHECK(vpiStreamSync(stream));

        { VPIImage img;
          VPI_CHECK(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_U8, 0, &img));
          img_u8 = VpiImage(img); }

        auto make_pyr = [&]() {
            VPIPyramid p;
            VPI_CHECK(vpiPyramidCreate(W, H, VPI_IMAGE_FORMAT_U8,
                                       cfg.pyr_levels, 0.5f, 0, &p));
            return VpiPyramid(p);
        };
        pyr_prev = make_pyr();
        pyr_cur  = make_pyr();

        constexpr int MAX_H = 8192;
        auto make_arr = [&](int cap, VPIArrayType type) {
            VPIArray arr;
            VPI_CHECK(vpiArrayCreate(cap, type, 0, &arr));
            return VpiArray(arr);
        };
        kpts_prev   = make_arr(MAX_H, VPI_ARRAY_TYPE_KEYPOINT);
        kpts_cur    = make_arr(MAX_H, VPI_ARRAY_TYPE_KEYPOINT);
        kpts_status = make_arr(MAX_H, VPI_ARRAY_TYPE_U8);
        kpts_scores = make_arr(MAX_H, VPI_ARRAY_TYPE_U32);

        { VPIPayload p;
          VPI_CHECK(vpiCreateHarrisCornerDetector(VPI_BACKEND_CPU, W, H, &p));
          harris_payload = VpiPayload(p); }

        { VPIPayload p;
          VPI_CHECK(vpiInitOpticalFlowPyrLKParams(&lk_params));
          lk_params.windowDimension = cfg.lk_window;
          lk_params.numIterations   = cfg.lk_iterations;
          lk_params.epsilon         = cfg.lk_epsilon;
          VPI_CHECK(vpiCreateOpticalFlowPyrLK(VPI_BACKEND_CUDA,
                                              W, H, VPI_IMAGE_FORMAT_U8,
                                              cfg.pyr_levels, 0.5f, &p));
          lk_payload = VpiPayload(p); }

        colours.resize(MAX_H);
        std::cout << "[vpi] initialised  " << W << "x" << H
                  << "  pyr=" << cfg.pyr_levels
                  << "  win=" << cfg.lk_window
                  << "  iters=" << cfg.lk_iterations
                  << "  max_pts=" << max_points << "\n";
    }

    // BGR -> U8 -> pyr_cur
    void upload(cv::Mat& bgr)
    {
        cv_wrap_buf = bgr;
        if (!img_wrap.h) {
            VPIImage h;
            VPI_CHECK(vpiImageCreateOpenCVMatWrapper(cv_wrap_buf, 0, &h));
            img_wrap = VpiImage(h);
        } else {
            VPI_CHECK(vpiImageSetWrappedOpenCVMat(img_wrap, cv_wrap_buf));
        }
        VPI_CHECK(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA,
                                             img_wrap, img_u8, nullptr));
        VPI_CHECK(vpiSubmitGaussianPyramidGenerator(stream, VPI_BACKEND_CUDA,
                                                    img_u8, pyr_cur));
        VPI_CHECK(vpiStreamSync(stream));
    }

    // Harris -> kpts_cur; sort/cull on CPU; swap so kpts_cur becomes kpts_prev
    void detect_features()
    {
        VPIHarrisCornerDetectorParams hparams;
        VPI_CHECK(vpiInitHarrisCornerDetectorParams(&hparams));
        hparams.strengthThresh = 0;
        hparams.sensitivity    = harris_sensitivity;
        VPI_CHECK(vpiSubmitHarrisCornerDetector(stream, VPI_BACKEND_CPU,
                                                harris_payload, img_u8,
                                                kpts_cur, kpts_scores,
                                                &hparams));
        VPI_CHECK(vpiStreamSync(stream));

        // Sort by score, cull to max_points, seed colours
        {
            VPIArrayData pts_d, scr_d;
            VPI_CHECK(vpiArrayLock(kpts_cur,    VPI_LOCK_READ_WRITE, &pts_d));
            VPI_CHECK(vpiArrayLock(kpts_scores, VPI_LOCK_READ_WRITE, &scr_d));
            int n = *pts_d.sizePointer;
            std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
            const uint32_t* sc = reinterpret_cast<const uint32_t*>(scr_d.data);
            std::stable_sort(idx.begin(), idx.end(),
                [sc](int a, int b){ return sc[a] >= sc[b]; });
            n = std::min(n, max_points);
            VPIKeypoint* kp = reinterpret_cast<VPIKeypoint*>(pts_d.data);
            std::vector<VPIKeypoint> sorted(n);
            for (int i = 0; i < n; ++i) sorted[i] = kp[idx[i]];
            std::copy(sorted.begin(), sorted.end(), kp);
            *pts_d.sizePointer = n;
            for (int i = 0; i < n; ++i)
                colours[i] = point_colour(sorted[i].x, sorted[i].y);
            vpiArrayUnlock(kpts_scores);
            vpiArrayUnlock(kpts_cur);
            n_reinit_features = n;
        }

        std::swap(kpts_prev.h, kpts_cur.h);
        std::swap(pyr_prev.h,  pyr_cur.h);
    }

    // LK: reads kpts_prev, writes kpts_cur (pure GPU)
    int track()
    {
        // Mirror prev size into cur before each call
        {
            VPIArrayData pd, cd;
            VPI_CHECK(vpiArrayLock(kpts_prev, VPI_LOCK_READ,       &pd));
            VPI_CHECK(vpiArrayLock(kpts_cur,  VPI_LOCK_READ_WRITE, &cd));
            *cd.sizePointer = *pd.sizePointer;
            vpiArrayUnlock(kpts_cur);
            vpiArrayUnlock(kpts_prev);
        }
        VPI_CHECK(vpiSubmitOpticalFlowPyrLK(stream, 0, lk_payload,
                                            pyr_prev, pyr_cur,
                                            kpts_prev, kpts_cur, kpts_status,
                                            &lk_params));
        VPI_CHECK(vpiStreamSync(stream));
        return count_live();
    }

    void advance()
    {
        std::swap(kpts_prev.h, kpts_cur.h);
        std::swap(pyr_prev.h,  pyr_cur.h);
    }

    std::vector<TrackPoint> read_tracks() const
    {
        VPIArrayData prev_d, cur_d, sts_d;
        VPI_CHECK(vpiArrayLock(kpts_prev,   VPI_LOCK_READ, &prev_d));
        VPI_CHECK(vpiArrayLock(kpts_cur,    VPI_LOCK_READ, &cur_d));
        VPI_CHECK(vpiArrayLock(kpts_status, VPI_LOCK_READ, &sts_d));
        const VPIKeypoint* pp = reinterpret_cast<const VPIKeypoint*>(prev_d.data);
        const VPIKeypoint* cp = reinterpret_cast<const VPIKeypoint*>(cur_d.data);
        const uint8_t*     sp = reinterpret_cast<const uint8_t*>(sts_d.data);
        int n = *sts_d.sizePointer;
        const float max_disp2 = (W*0.5f)*(W*0.5f) + (H*0.5f)*(H*0.5f);
        std::vector<TrackPoint> tracks(n);
        for (int i = 0; i < n; ++i) {
            tracks[i].colour = colours[i];
            tracks[i].prev   = {pp[i].x, pp[i].y};
            tracks[i].cur    = {cp[i].x, cp[i].y};
            float dx = cp[i].x - pp[i].x, dy = cp[i].y - pp[i].y;
            bool in_bounds = (cp[i].x >= 0.f && cp[i].x < W &&
                              cp[i].y >= 0.f && cp[i].y < H);
            bool plausible = (dx*dx + dy*dy) <= max_disp2;
            tracks[i].valid = (sp[i] == 0) && in_bounds && plausible;
        }
        vpiArrayUnlock(kpts_status);
        vpiArrayUnlock(kpts_cur);
        vpiArrayUnlock(kpts_prev);
        return tracks;
    }

private:
    int count_live() const
    {
        const float max_disp2 = (W*0.5f)*(W*0.5f) + (H*0.5f)*(H*0.5f);
        VPIArrayData pd, cd, sd;
        VPI_CHECK(vpiArrayLock(kpts_prev,   VPI_LOCK_READ, &pd));
        VPI_CHECK(vpiArrayLock(kpts_cur,    VPI_LOCK_READ, &cd));
        VPI_CHECK(vpiArrayLock(kpts_status, VPI_LOCK_READ, &sd));
        int n = *sd.sizePointer;
        const VPIKeypoint* pp = reinterpret_cast<const VPIKeypoint*>(pd.data);
        const VPIKeypoint* cp = reinterpret_cast<const VPIKeypoint*>(cd.data);
        const uint8_t*     sp = reinterpret_cast<const uint8_t*>(sd.data);
        int live = 0;
        for (int i = 0; i < n; ++i) {
            if (sp[i] != 0) continue;
            if (cp[i].x < 0.f || cp[i].x >= W || cp[i].y < 0.f || cp[i].y >= H) continue;
            float dx = cp[i].x - pp[i].x, dy = cp[i].y - pp[i].y;
            if (dx*dx + dy*dy > max_disp2) continue;
            ++live;
        }
        vpiArrayUnlock(kpts_status);
        vpiArrayUnlock(kpts_cur);
        vpiArrayUnlock(kpts_prev);
        return live;
    }
};

// ============================================================================
// Visualisation helpers (identical to cvcuda_sparse_pyrlk)
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

static void draw_arrow(cv::Mat& img, cv::Point2f from, cv::Point2f to,
                       const cv::Scalar& colour, int thickness = 2)
{
    float dx = to.x-from.x, dy = to.y-from.y;
    if (dx*dx + dy*dy < 1.f) return;
    cv::arrowedLine(img, from, to, colour, thickness, cv::LINE_AA, 0, 0.25);
}

static cv::Mat build_quad(
    const cv::Mat&                 cur_bgr,
    const cv::Mat&                 prev_bgr,
    const std::vector<TrackPoint>& tracks,
    cv::Mat&                       trail,
    double fps, int n_live, int n_total,
    float mag_clip, int arrow_scale,
    int canvas_w, int canvas_h)
{
    const int tw = canvas_w/2, th = canvas_h/2;
    const float sx = static_cast<float>(tw)/cur_bgr.cols;
    const float sy = static_cast<float>(th)/cur_bgr.rows;
    char buf[64];

    cv::Mat t0; cv::resize(cur_bgr, t0, {tw,th}, 0, 0, cv::INTER_AREA);
    trail *= 0.82f;
    for (const auto& tp : tracks) {
        if (!tp.valid) continue;
        cv::Point2f p(tp.cur.x*sx,  tp.cur.y*sy);
        cv::Point2f q(tp.prev.x*sx, tp.prev.y*sy);
        cv::line(trail, q, p, tp.colour, 2, cv::LINE_AA);
        cv::circle(trail, p, 4, tp.colour, -1, cv::LINE_AA);
    }
    cv::add(t0, trail, t0);
    std::snprintf(buf, sizeof(buf), "tracks: %d / %d", n_live, n_total);
    tile_label(t0, "Current  +  trails", {buf});

    cv::Mat t1;
    if (prev_bgr.empty()) {
        t1 = cv::Mat::zeros(th, tw, CV_8UC3);
        tile_label(t1, "Previous", {"(waiting for frame 2)"});
    } else {
        cv::resize(prev_bgr, t1, {tw,th}, 0, 0, cv::INTER_AREA);
        for (const auto& tp : tracks) {
            if (!tp.valid) continue;
            cv::circle(t1, cv::Point2f(tp.prev.x*sx, tp.prev.y*sy),
                       4, tp.colour, -1, cv::LINE_AA);
        }
        tile_label(t1, "Previous  +  detections");
    }

    cv::Mat t2(th, tw, CV_8UC3, cv::Scalar(20,20,20));
    for (const auto& tp : tracks) {
        if (!tp.valid) continue;
        cv::Point2f from(tp.prev.x*sx, tp.prev.y*sy);
        draw_arrow(t2, from,
                   {from.x + (tp.cur.x - tp.prev.x)*sx*arrow_scale,
                    from.y + (tp.cur.y - tp.prev.y)*sy*arrow_scale},
                   tp.colour);
    }
    std::snprintf(buf, sizeof(buf), "scale: x%d", arrow_scale);
    tile_label(t2, "Flow vectors", {buf});

    cv::Mat heat(th, tw, CV_32F, cv::Scalar(0));
    float mean_mag = 0.f; int cnt = 0;
    for (const auto& tp : tracks) {
        if (!tp.valid) continue;
        float dx = tp.cur.x - tp.prev.x, dy = tp.cur.y - tp.prev.y;
        float mag = std::sqrt(dx*dx + dy*dy);
        mean_mag += mag; ++cnt;
        float val = std::min(mag/mag_clip, 1.f);
        cv::circle(heat, cv::Point2f(tp.cur.x*sx, tp.cur.y*sy),
                   static_cast<int>(val*20+4), cv::Scalar(val), -1, cv::LINE_AA);
    }
    cv::GaussianBlur(heat, heat, {15,15}, 5.0);
    cv::Mat heat_u8; heat.convertTo(heat_u8, CV_8U, 255.0);
    cv::Mat t3; cv::applyColorMap(heat_u8, t3, cv::COLORMAP_HOT);
    if (cnt > 0) mean_mag /= cnt;
    std::snprintf(buf, sizeof(buf), "mean: %.1f px/frame", mean_mag);
    tile_label(t3, "Speed  (hot)", {buf});

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
        if      (a == "--width")            { c.width              = std::stoi(get_next(i)); ++i; }
        else if (a == "--height")           { c.height             = std::stoi(get_next(i)); ++i; }
        else if (a == "--fps")              { c.fps                = std::stoi(get_next(i)); ++i; }
        else if (a == "--sensor")           { c.sensor_id          = std::stoi(get_next(i)); ++i; }
        else if (a == "--flip")             { c.flip_method        = std::stoi(get_next(i)); ++i; }
        else if (a == "--udp-host")         { c.udp_host           = get_next(i); ++i; }
        else if (a == "--udp-port")         { c.udp_port           = std::stoi(get_next(i)); ++i; }
        else if (a == "--stream-name")      { c.stream_name        = get_next(i); ++i; }
        else if (a == "--bitrate")          { c.bitrate            = std::stoi(get_next(i)); ++i; }
        else if (a == "--iframeinterval")   { c.iframeinterval     = std::stoi(get_next(i)); ++i; }
        else if (a == "--stream-width")     { c.stream_w           = std::stoi(get_next(i)); ++i; }
        else if (a == "--stream-height")    { c.stream_h           = std::stoi(get_next(i)); ++i; }
        else if (a == "--pyr-levels")       { c.pyr_levels         = std::stoi(get_next(i)); ++i; }
        else if (a == "--max-points")       { c.max_points         = std::stoi(get_next(i)); ++i; }
        else if (a == "--min-points")       { c.min_points         = std::stoi(get_next(i)); ++i; }
        else if (a == "--lk-window")        { c.lk_window          = std::stoi(get_next(i)); ++i; }
        else if (a == "--lk-iter")          { c.lk_iterations      = std::stoi(get_next(i)); ++i; }
        else if (a == "--lk-eps")           { c.lk_epsilon         = std::stof(get_next(i)); ++i; }
        else if (a == "--harris-sens")      { c.harris_sensitivity = std::stof(get_next(i)); ++i; }
        else if (a == "--mag-clip")         { c.mag_clip           = std::stof(get_next(i)); ++i; }
        else if (a == "--arrow-scale")      { c.arrow_scale        = std::stoi(get_next(i)); ++i; }
        else if (a == "--lock-ae")          { c.lock_ae            = true; }
        else if (a == "--lock-awb")         { c.lock_awb           = true; }
        else if (a == "--no-fps")           { c.show_fps           = false; }
        else if (a == "--help" || a == "-h") {
            std::cout <<
                "vpi_optical_flow -- VPI Harris + PyrLK sparse optical flow\n\n"
                "  WARNING: vpiSubmitOpticalFlowPyrLK has a known NVIDIA bug on\n"
                "  JetPack 4.6.6 -- points are not updated after the first frame.\n"
                "  This is a pipeline reference demo. Use cvcuda_sparse_pyrlk for\n"
                "  correct output.\n\n"
                "  --width/height/fps/sensor/flip\n"
                "  --udp-host/udp-port/stream-name/bitrate/iframeinterval\n"
                "  --stream-width/stream-height\n"
                "  --pyr-levels N      [5]    pyramid depth\n"
                "  --max-points N      [200]  max tracked keypoints\n"
                "  --min-points N      [20]   reinit Harris below this count\n"
                "  --lk-window N       [15]   LK search window (odd px)\n"
                "  --lk-iter N         [20]   LK max iterations per level\n"
                "  --lk-eps F          [0.01] LK convergence epsilon\n"
                "  --harris-sens F     [0.01] Harris sensitivity\n"
                "  --mag-clip F        [25]   speed saturation for heatmap\n"
                "  --arrow-scale N     [4]    arrow display multiplier\n"
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
              << "  vpi_optical_flow -- VPI Harris + PyrLK\n"
              << "  WARNING: known NVIDIA bug -- demo only\n"
              << "==================================================\n"
              << "  Capture    : " << cfg.width << "x" << cfg.height
              <<                  " @ " << cfg.fps << "fps\n"
              << "  Stream out : " << cfg.stream_w << "x" << cfg.stream_h << "\n"
              << "  Pyr levels : " << cfg.pyr_levels << "\n"
              << "  Max points : " << cfg.max_points << "\n"
              << "  Min points : " << cfg.min_points << "\n"
              << "  LK window  : " << cfg.lk_window << "px\n"
              << "  LK iters   : " << cfg.lk_iterations << "\n"
              << "  Harris sens: " << cfg.harris_sensitivity << "\n"
              << "  UDP ->     : " << cfg.udp_host << ":" << cfg.udp_port << "\n"
              << "  Stream     : http://<host>:8889/" << cfg.stream_name << "\n"
              << "==================================================\n\n";

    std::string cap_pipe = capture_pipeline(cfg.sensor_id, cfg.width, cfg.height,
                                            cfg.fps, cfg.flip_method,
                                            cfg.lock_ae, cfg.lock_awb);
    std::cout << "[camera] pipeline: " << cap_pipe << "\n\n";

    CameraBuffer cam_buf;
    std::thread  cam_thr(camera_thread, std::cref(cap_pipe), std::ref(cam_buf));

    VpiLK tracker;
    tracker.init(cfg.width, cfg.height, cfg);

    std::string push_str = push_pipeline(cfg.stream_w, cfg.stream_h, cfg.fps,
                                         cfg.udp_host, cfg.udp_port,
                                         cfg.bitrate, cfg.iframeinterval);
    std::cout << "[gst-push] pipeline: " << push_str << "\n\n";
    GstPushPipeline gst_push;
    gst_push.start(push_str, cfg.fps);

    cv::Mat cur_bgr, prev_bgr;
    const int tw = cfg.stream_w/2, th = cfg.stream_h/2;
    cv::Mat trail = cv::Mat::zeros(th, tw, CV_8UC3);

    auto   t_last     = std::chrono::steady_clock::now();
    long   frame_n    = 0;
    double fps_smooth = 0.0;
    bool   have_prev  = false;

    std::cout << "[main] entering loop. Ctrl+C to stop.\n";

    while (!g_stop.load()) {
        if (!cam_buf.get(cur_bgr)) continue;

        try { tracker.upload(cur_bgr); }
        catch (const std::exception& e) {
            std::cerr << "[vpi] upload ERROR: " << e.what() << "\n"; continue;
        }

        std::vector<TrackPoint> tracks;
        int n_live = 0, n_total = 0;
        bool just_reinit = false;

        if (!have_prev) {
            tracker.detect_features();
            have_prev   = true;
            just_reinit = true;
        } else {
            try {
                n_live  = tracker.track();
                tracks  = tracker.read_tracks();
                n_total = static_cast<int>(tracks.size());
            } catch (const std::exception& e) {
                std::cerr << "[vpi] track ERROR: " << e.what() << "\n";
            }

            if (!just_reinit && n_live < cfg.min_points) {
                try {
                    tracker.detect_features();
                    trail       = cv::Mat::zeros(th, tw, CV_8UC3);
                    just_reinit = true;
                    n_live      = tracker.n_reinit_features;
                } catch (const std::exception& e) {
                    std::cerr << "[vpi] redetect ERROR: " << e.what() << "\n";
                }
            }
        }

        auto t_now  = std::chrono::steady_clock::now();
        double dt   = std::chrono::duration<double>(t_now - t_last).count();
        t_last      = t_now;
        fps_smooth  = 0.9*fps_smooth + 0.1*(dt > 0 ? 1.0/dt : 0.0);
        ++frame_n;

        cv::Mat vis = build_quad(cur_bgr, prev_bgr, tracks, trail,
                                 fps_smooth, n_live, n_total,
                                 cfg.mag_clip, cfg.arrow_scale,
                                 cfg.stream_w, cfg.stream_h);
        if (gst_push.ok) gst_push.push(vis);

        if (cfg.show_fps && fps_smooth > 0 &&
            frame_n % static_cast<long>(fps_smooth + 0.5) == 0) {
            std::cout << "\r[main] frame=" << std::setw(6) << frame_n
                      << "  fps=" << std::fixed << std::setprecision(1) << fps_smooth
                      << "  tracks=" << n_live << "/" << n_total;
            if (just_reinit)
                std::cout << "  [reinit -> " << tracker.n_reinit_features << " pts]";
            std::cout << "          " << std::flush;
        }

        if (have_prev && !just_reinit)
            tracker.advance();

        prev_bgr = cur_bgr.clone();
    }

    std::cout << "\n[main] shutting down...\n";
    gst_push.stop();
    cam_thr.join();
    std::cout << "[main] done.\n";
    return 0;
}
