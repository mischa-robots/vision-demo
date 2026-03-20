// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Mischa Schaefer <https://github.com/mischa-robots/vision-demo>

/**
 * cvcuda_sparse_pyrlk_optical_flow.cpp
 * ---------------------------------
 * Sparse LK optical flow for Jetson Nano -- streams quad-view to browser via MediaMTX.
 *
 * Detection : cv::cuda::GoodFeaturesToTrackDetector (Shi-Tomasi, CUDA, blockSize=7)
 * Tracking  : cv::cuda::SparsePyrLKOpticalFlow (CUDA) + forward-backward filter (CPU)
 *
 * No VPI dependency -- pure OpenCV CUDA pipeline.
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
 *   ./build/cvcuda_sparse_pyrlk
 *   ./build/cvcuda_sparse_pyrlk --help
 */

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>

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

    // LK parameters -- matched to working Python cv_optical_flow_cuda.py:
    //   win=15, levels=4, iters=20, min_points=20
    int         pyr_levels     = 4;
    int         max_points     = 200;
    int         min_points     = 20;
    int         lk_window      = 15;
    int         lk_iterations  = 20;
    float       lk_epsilon     = 0.01f;

    // BUG 5: blockSize 3->7 to match Python goodFeaturesToTrack(blockSize=7).
    // Larger block averages gradients over a wider neighbourhood -- far more
    // selective, avoids picking "corners" on flat/textured-noise surfaces.
    int         block_size     = 7;

    // Shi-Tomasi quality / spacing
    double      quality_level  = 0.01;
    double      min_distance   = 10.0;

    // BUG 5: forward-backward consistency check.
    // After forward track (prev->cur) we re-track backward (cur->prev) and
    // reject points whose round-trip error > fb_thresh px.
    // False tracks on smooth surfaces rarely survive the round-trip.
    // Set to 0 to disable.
    float       fb_thresh      = 1.0f;

    // BUG 4: per-frame displacement cap (was ~735 px).
    float       max_disp_px    = 80.0f;

    float       mag_clip       = 25.0f;
    int         arrow_scale    = 4;
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
// CvLK_CvHarris  --  Option B tracker
//
// Detection : cv::cuda::GoodFeaturesToTrackDetector (Shi-Tomasi, blockSize=7)
// Tracking  : cv::cuda::SparsePyrLKOpticalFlow + manual forward-backward check
// ============================================================================
struct CvLK_CvHarris {
    cv::Ptr<cv::cuda::CornersDetector>          detector;
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow>   lk;

    // Gray frames -- always independently allocated via .clone() (BUG 1 fix)
    cv::cuda::GpuMat prev_gray_gpu;
    cv::cuda::GpuMat cur_gray_gpu;

    // Points stay on GPU for their lifetime; downloaded only for visualisation
    cv::cuda::GpuMat prev_pts_gpu;
    cv::cuda::GpuMat cur_pts_gpu;
    cv::cuda::GpuMat status_gpu;

    // CPU caches -- populated inside track(), reused by read_tracks/count_live
    cv::Mat  prev_pts_cpu;
    cv::Mat  cur_pts_cpu;
    cv::Mat  status_cpu;

    std::vector<cv::Scalar> colours;
    int   W = 0, H = 0;
    int   max_points    = 200;
    int   min_points    = 20;
    int   n_reinit_features = 0;
    float max_disp_px   = 80.0f;
    float fb_thresh     = 1.0f;

    void init(int w, int h, const Config& cfg)
    {
        W = w; H = h;
        max_points  = cfg.max_points;
        min_points  = cfg.min_points;
        max_disp_px = cfg.max_disp_px;
        fb_thresh   = cfg.fb_thresh;

        // BUG 5: blockSize 7 matches Python goodFeaturesToTrack(blockSize=7).
        detector = cv::cuda::createGoodFeaturesToTrackDetector(
            CV_8UC1,
            cfg.max_points,
            cfg.quality_level,
            cfg.min_distance,
            cfg.block_size);

        // BUG 3 fix: create(winSize, maxLevel, iters) -- parameters now correct.
        lk = cv::cuda::SparsePyrLKOpticalFlow::create(
            cv::Size(cfg.lk_window, cfg.lk_window),
            cfg.pyr_levels,
            cfg.lk_iterations);

        colours.resize(cfg.max_points + 16);
        std::cout << "[tracker] init  " << W << "x" << H
                  << "  pyr=" << cfg.pyr_levels
                  << "  win=" << cfg.lk_window
                  << "  iters=" << cfg.lk_iterations
                  << "  block=" << cfg.block_size
                  << "  fb=" << cfg.fb_thresh << "px"
                  << "  max_disp=" << cfg.max_disp_px << "px"
                  << "  max_pts=" << max_points << "\n";
    }

    cv::Mat to_gray(const cv::Mat& bgr)
    {
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }

    void detect_features(const cv::Mat& gray)
    {
        cur_gray_gpu.upload(gray);
        detector->detect(cur_gray_gpu, prev_pts_gpu);

        // Normalise to (1, N, CV_32FC2) row-vector
        if (!prev_pts_gpu.empty())
            prev_pts_gpu = prev_pts_gpu.reshape(2, 1);

        // Download once to seed colours
        prev_pts_gpu.download(prev_pts_cpu);
        int n = prev_pts_cpu.cols;
        for (int i = 0; i < n && i < static_cast<int>(colours.size()); ++i) {
            auto pt = prev_pts_cpu.at<cv::Point2f>(0, i);
            colours[i] = point_colour(pt.x, pt.y);
        }
        n_reinit_features = n;

        // BUG 1 fix: .clone() for independent device allocation
        prev_gray_gpu = cur_gray_gpu.clone();

        // Invalidate CPU caches
        prev_pts_cpu.release();
        cur_pts_cpu.release();
        status_cpu.release();
    }

    // Track prev -> cur with optional forward-backward consistency filter.
    //
    // Forward-backward check (BUG 5 fix):
    //   After the forward pass (prev_gray -> cur_gray), we download all arrays
    //   to CPU, then run a backward pass using a freshly uploaded cur_pts as
    //   source. This avoids shape ambiguity from the forward output GpuMat.
    //   Any point whose round-trip error > fb_thresh is zeroed in status_cpu
    //   before count_live() is called.
    int track(const cv::Mat& gray)
    {
        if (prev_pts_gpu.empty()) return 0;

        cur_gray_gpu.upload(gray);

        // ---- Forward pass: prev_gray -> cur_gray -------------------------
        cv::cuda::GpuMat fwd_cur_gpu, fwd_st_gpu;
        lk->calc(prev_gray_gpu, cur_gray_gpu,
                 prev_pts_gpu, fwd_cur_gpu, fwd_st_gpu);

        // Download to CPU immediately -- we need clean copies for the FB pass
        // and count_live()/read_tracks() will reuse them.
        prev_pts_gpu.download(prev_pts_cpu);
        fwd_cur_gpu.download(cur_pts_cpu);
        fwd_st_gpu.download(status_cpu);

        // Flatten to (1, N, *) regardless of kernel output shape
        auto flat2 = [](cv::Mat& m) {
            cv::Mat c = m.isContinuous() ? m : m.clone();
            // int total = static_cast<int>(c.total()) * c.channels() / 2;
            return c.reshape(2, 1);
        };
        auto flat1 = [](cv::Mat& m) {
            cv::Mat c = m.isContinuous() ? m : m.clone();
            return c.reshape(1, 1);
        };

        prev_pts_cpu = flat2(prev_pts_cpu);
        cur_pts_cpu  = flat2(cur_pts_cpu);
        status_cpu   = flat1(status_cpu);

        int n = std::min({ prev_pts_cpu.cols, cur_pts_cpu.cols, status_cpu.cols });

        // ---- Forward-backward check (all on CPU) -------------------------
        if (fb_thresh > 0.f && n > 0) {
            // Upload cur_pts as a fresh, known-shape GpuMat for the bwd pass.
            // Do NOT reuse fwd_cur_gpu -- its internal shape may differ from
            // what the calc() kernel expects as input.
            cv::cuda::GpuMat bwd_src_gpu;
            bwd_src_gpu.upload(cur_pts_cpu);   // (1, N, CV_32FC2) -- clean

            cv::cuda::GpuMat bwd_back_gpu, bwd_st_gpu;
            lk->calc(cur_gray_gpu, prev_gray_gpu,
                     bwd_src_gpu, bwd_back_gpu, bwd_st_gpu);

            cv::Mat back_cpu, back_st_cpu;
            bwd_back_gpu.download(back_cpu);
            bwd_st_gpu.download(back_st_cpu);
            back_cpu    = flat2(back_cpu);
            back_st_cpu = flat1(back_st_cpu);

            int nb = std::min({ n, back_cpu.cols, back_st_cpu.cols });

            const auto*    pp  = prev_pts_cpu.ptr<cv::Point2f>(0);
            const auto*    bp  = back_cpu.ptr<cv::Point2f>(0);
            auto*          sp  = status_cpu.ptr<uint8_t>(0);
            const uint8_t* bsp = back_st_cpu.ptr<uint8_t>(0);

            const float t2 = fb_thresh * fb_thresh;
            for (int i = 0; i < nb; ++i) {
                if (sp[i] != 1 || bsp[i] != 1) { sp[i] = 0; continue; }
                float dx = bp[i].x - pp[i].x;
                float dy = bp[i].y - pp[i].y;
                if (dx*dx + dy*dy > t2) sp[i] = 0;
            }
            for (int i = nb; i < n; ++i) sp[i] = 0;
        }

        // Keep GPU status in sync with the filtered CPU version
        cur_pts_gpu = fwd_cur_gpu;
        status_gpu.upload(status_cpu);

        return count_live();
    }

    void advance()
    {
        // BUG 1 fix: .clone() so prev and cur are independent allocations
        prev_gray_gpu = cur_gray_gpu.clone();
        prev_pts_gpu  = cur_pts_gpu;   // shallow OK -- overwritten next calc()
    }

    std::vector<TrackPoint> read_tracks()
    {
        if (prev_pts_cpu.empty() || cur_pts_cpu.empty() || status_cpu.empty())
            return {};

        const float max_disp2 = max_disp_px * max_disp_px;
        int n = std::min({ prev_pts_cpu.cols, cur_pts_cpu.cols, status_cpu.cols });

        std::vector<TrackPoint> tracks(n);
        const auto*    ppf = prev_pts_cpu.ptr<cv::Point2f>(0);
        const auto*    cpf = cur_pts_cpu.ptr<cv::Point2f>(0);
        const uint8_t* spf = status_cpu.ptr<uint8_t>(0);

        for (int i = 0; i < n; ++i) {
            tracks[i].colour = colours[i];
            tracks[i].prev   = ppf[i];
            tracks[i].cur    = cpf[i];
            float dx = cpf[i].x - ppf[i].x;
            float dy = cpf[i].y - ppf[i].y;
            bool in_bounds = (cpf[i].x >= 0.f && cpf[i].x < W &&
                              cpf[i].y >= 0.f && cpf[i].y < H);
            bool plausible = (dx*dx + dy*dy) <= max_disp2;
            tracks[i].valid = (spf[i] == 1) && in_bounds && plausible;
        }
        return tracks;
    }

private:
    int count_live()
    {
        if (cur_pts_cpu.empty() || status_cpu.empty() || prev_pts_cpu.empty())
            return 0;

        const float max_disp2 = max_disp_px * max_disp_px;
        int n = std::min({ prev_pts_cpu.cols, cur_pts_cpu.cols, status_cpu.cols });

        const auto*    ppf = prev_pts_cpu.ptr<cv::Point2f>(0);
        const auto*    cpf = cur_pts_cpu.ptr<cv::Point2f>(0);
        const uint8_t* spf = status_cpu.ptr<uint8_t>(0);

        int live = 0;
        for (int i = 0; i < n; ++i) {
            if (spf[i] != 1) continue;
            if (cpf[i].x < 0.f || cpf[i].x >= W) continue;
            if (cpf[i].y < 0.f || cpf[i].y >= H) continue;
            float dx = cpf[i].x - ppf[i].x;
            float dy = cpf[i].y - ppf[i].y;
            if (dx*dx + dy*dy > max_disp2) continue;
            ++live;
        }
        return live;
    }
};

// ============================================================================
// Visualisation helpers
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

    // Tile 0: current frame + fading trails
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

    // Tile 1: previous frame + detection dots
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

    // Tile 2: flow arrows
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

    // Tile 3: speed heatmap
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

    // Composite
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
        else if (a == "--pyr-levels")     { c.pyr_levels     = std::stoi(get_next(i)); ++i; }
        else if (a == "--max-points")     { c.max_points     = std::stoi(get_next(i)); ++i; }
        else if (a == "--min-points")     { c.min_points     = std::stoi(get_next(i)); ++i; }
        else if (a == "--lk-window")      { c.lk_window      = std::stoi(get_next(i)); ++i; }
        else if (a == "--lk-iter")        { c.lk_iterations  = std::stoi(get_next(i)); ++i; }
        else if (a == "--lk-eps")         { c.lk_epsilon     = std::stof(get_next(i)); ++i; }
        else if (a == "--block-size")     { c.block_size     = std::stoi(get_next(i)); ++i; }
        else if (a == "--quality-level")  { c.quality_level  = std::stod(get_next(i)); ++i; }
        else if (a == "--min-distance")   { c.min_distance   = std::stod(get_next(i)); ++i; }
        else if (a == "--fb-thresh")      { c.fb_thresh      = std::stof(get_next(i)); ++i; }
        else if (a == "--max-disp")       { c.max_disp_px    = std::stof(get_next(i)); ++i; }
        else if (a == "--mag-clip")       { c.mag_clip       = std::stof(get_next(i)); ++i; }
        else if (a == "--arrow-scale")    { c.arrow_scale    = std::stoi(get_next(i)); ++i; }
        else if (a == "--lock-ae")        { c.lock_ae        = true; }
        else if (a == "--lock-awb")       { c.lock_awb       = true; }
        else if (a == "--no-fps")         { c.show_fps       = false; }
        else if (a == "--help" || a == "-h") {
            std::cout <<
                "cvcuda_sparse_pyrlk_optical_flow -- OpenCV CUDA Shi-Tomasi + LK (no VPI)\n\n"
                "  --width/height/fps/sensor/flip\n"
                "  --udp-host/udp-port/stream-name/bitrate/iframeinterval\n"
                "  --stream-width/stream-height\n"
                "  --pyr-levels N      [4]    --max-points N    [200] --min-points N [20]\n"
                "  --lk-window N       [15]   --lk-iter N       [20]\n"
                "  --block-size N      [7]    corner detector block (3=noisy, 7=selective)\n"
                "  --quality-level F   [0.01] --min-distance F  [10]\n"
                "  --fb-thresh F       [1.0]  forward-backward px error (0=disable)\n"
                "  --max-disp F        [80]   max px/frame before track rejected\n"
                "  --mag-clip F        [25]   --arrow-scale N   [4]\n"
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
              << "  cvcuda_sparse_pyrlk_optical_flow -- OpenCV CUDA Shi-Tomasi + LK\n"
              << "==================================================\n"
              << "  Capture    : " << cfg.width << "x" << cfg.height
              <<                  " @ " << cfg.fps << "fps\n"
              << "  Stream out : " << cfg.stream_w << "x" << cfg.stream_h << "\n"
              << "  Pyr levels : " << cfg.pyr_levels << "\n"
              << "  Max points : " << cfg.max_points << "\n"
              << "  Min points : " << cfg.min_points << "\n"
              << "  LK window  : " << cfg.lk_window << "px\n"
              << "  LK iters   : " << cfg.lk_iterations << "\n"
              << "  Block size : " << cfg.block_size << "\n"
              << "  FB thresh  : " << cfg.fb_thresh << "px\n"
              << "  Max disp   : " << cfg.max_disp_px << "px/frame\n"
              << "  Quality    : " << cfg.quality_level << "\n"
              << "  Min dist   : " << cfg.min_distance << "px\n"
              << "  UDP ->     : " << cfg.udp_host << ":" << cfg.udp_port << "\n"
              << "  Stream     : http://<host>:8889/" << cfg.stream_name << "\n"
              << "==================================================\n\n";

    std::string cap_pipe = capture_pipeline(cfg.sensor_id, cfg.width, cfg.height,
                                            cfg.fps, cfg.flip_method,
                                            cfg.lock_ae, cfg.lock_awb);
    std::cout << "[camera] pipeline: " << cap_pipe << "\n\n";

    CameraBuffer cam_buf;
    std::thread  cam_thr(camera_thread, std::cref(cap_pipe), std::ref(cam_buf));

    CvLK_CvHarris tracker;
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

        cv::Mat cur_gray = tracker.to_gray(cur_bgr);

        std::vector<TrackPoint> tracks;
        int n_live = 0, n_total = 0;
        bool just_reinit = false;

        if (!have_prev) {
            tracker.detect_features(cur_gray);
            have_prev   = true;
            just_reinit = true;
        } else {
            n_live  = tracker.track(cur_gray);
            tracks  = tracker.read_tracks();
            n_total = static_cast<int>(tracks.size());

            if (n_live < cfg.min_points) {
                tracker.detect_features(cur_gray);
                trail       = cv::Mat::zeros(th, tw, CV_8UC3);
                just_reinit = true;
                n_live      = tracker.n_reinit_features;
                n_total     = static_cast<int>(tracks.size());
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
