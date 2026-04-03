// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>

/**
 * cv_face_landmarks.cpp
 * ---------------------
 * Real-time face landmark detection for Jetson Nano using OpenCV's
 * Facemark API (FacemarkLBF). Streams a side-by-side canvas to the
 * browser via MediaMTX.
 *
 * Pipeline:
 *   Haar cascade (CPU) → face ROIs
 *   FacemarkLBF (CPU)  → 68 landmarks per face
 *
 * Performance note: Both the Haar cascade and FacemarkLBF run entirely
 * on CPU. Detection is run on a downscaled copy of the frame (controlled
 * by --detect-scale) and the resulting landmarks are mapped back to full
 * resolution. This significantly reduces CPU load without affecting the
 * quality of the overlay drawn at full resolution.
 *
 * Side-by-side layout (default 1280x360):
 *   +---------------------+---------------------+
 *   |  Original stream    |  Landmark overlay   |
 *   +---------------------+---------------------+
 *
 * Landmark overlay:
 *   - 68 landmark dots colour-coded by facial region
 *   - Facial region contours (jaw, eyebrows, nose, eyes, mouth)
 *   - Face bounding box
 *   - Per-face landmark count
 *
 * Required model files (pass paths with --cascade and --model):
 *   haarcascade_frontalface_alt2.xml  — ships with OpenCV
 *     Typically at: /usr/share/opencv4/haarcascades/
 *   lbfmodel.yaml                     — ~54 MB, download once:
 *     wget https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml
 *
 * Watch the stream at:
 *   WebRTC : http://<jetson-ip>:8889/face
 *   HLS    : http://<jetson-ip>:8888/face
 *   RTSP   : rtsp://<jetson-ip>:8554/face
 
 * Usage:
 *   ./build/cv_face --cascade /path/to/haarcascade_frontalface_alt2.xml \
 *                             --model   /path/to/lbfmodel.yaml
 *   ./build/cv_face --help
 */

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/face.hpp>

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
    int         udp_port       = 5604;
    std::string stream_name    = "face";
    bool        show_fps       = true;
    int         stream_w       = 1280;
    int         stream_h       = 360;

    // Model paths — required
    std::string cascade_path;
    std::string model_path;

    // Detection parameters
    float  detect_scale    = 0.5f;   // downscale factor for detection pass
    double scale_factor    = 1.1;    // Haar cascade scaleFactor
    int    min_neighbours  = 3;      // Haar cascade minNeighbours
    int    min_face_px     = 60;     // minimum face size in detection-scale px
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
// FaceLandmarkDetector
// ============================================================================
struct FaceLandmarkDetector {
    cv::CascadeClassifier         cascade;
    cv::Ptr<cv::face::Facemark>   facemark;

    void init(const Config& cfg)
    {
        if (!cascade.load(cfg.cascade_path))
            throw std::runtime_error("Cannot load cascade: " + cfg.cascade_path);

        facemark = cv::face::FacemarkLBF::create();
        facemark->loadModel(cfg.model_path);

        std::cout << "[facemark] cascade : " << cfg.cascade_path << "\n"
                  << "[facemark] model   : " << cfg.model_path   << "\n";
    }

    struct Result {
        std::vector<cv::Rect>                    faces;
        std::vector<std::vector<cv::Point2f>>    landmarks;  // one entry per face
    };

    // Detect on a downscaled grey image, scale results back to full resolution.
    Result detect(const cv::Mat& bgr_full, const Config& cfg)
    {
        Result r;
        const float inv = 1.f / cfg.detect_scale;

        // Downscale + greyscale for the detection pass
        cv::Mat small;
        cv::resize(bgr_full, small,
                   cv::Size(static_cast<int>(bgr_full.cols * cfg.detect_scale),
                            static_cast<int>(bgr_full.rows * cfg.detect_scale)),
                   0, 0, cv::INTER_AREA);
        cv::Mat gray;
        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        // Haar cascade face detection
        std::vector<cv::Rect> faces_small;
        cascade.detectMultiScale(gray, faces_small,
                                 cfg.scale_factor,
                                 cfg.min_neighbours,
                                 cv::CASCADE_SCALE_IMAGE,
                                 cv::Size(cfg.min_face_px, cfg.min_face_px));

        if (faces_small.empty()) return r;

        // Scale face rects back to full resolution
        for (auto& f : faces_small) {
            r.faces.push_back(cv::Rect(
                static_cast<int>(f.x      * inv),
                static_cast<int>(f.y      * inv),
                static_cast<int>(f.width  * inv),
                static_cast<int>(f.height * inv)));
        }

        // FacemarkLBF fit — runs on the small greyscale image with small rects
        std::vector<std::vector<cv::Point2f>> lm_small;
        bool ok = facemark->fit(gray, faces_small, lm_small);
        if (!ok) return r;

        // Scale landmarks back to full resolution
        for (auto& face_lm : lm_small) {
            std::vector<cv::Point2f> scaled;
            scaled.reserve(face_lm.size());
            for (auto& p : face_lm)
                scaled.emplace_back(p.x * inv, p.y * inv);
            r.landmarks.push_back(std::move(scaled));
        }

        return r;
    }
};

// ============================================================================
// Landmark drawing
//
// 68-point iBUG / dlib layout:
//   0-16   jaw line
//   17-21  left eyebrow
//   22-26  right eyebrow
//   27-30  nose bridge
//   30-35  nose tip
//   36-41  left eye
//   42-47  right eye
//   48-59  outer mouth
//   60-67  inner mouth
// ============================================================================

// Colour scheme per region (BGR)
static const cv::Scalar COL_JAW        {180, 180, 180};
static const cv::Scalar COL_EYEBROW    {  0, 200, 255};
static const cv::Scalar COL_NOSE       {  0, 255, 200};
static const cv::Scalar COL_EYE        { 50,  50, 255};
static const cv::Scalar COL_MOUTH_OUT  {255, 100,  50};
static const cv::Scalar COL_MOUTH_IN   {255, 200,  80};
static const cv::Scalar COL_BOX        {  0, 255, 100};

static cv::Scalar landmark_colour(int idx)
{
    if (idx <= 16)  return COL_JAW;
    if (idx <= 26)  return COL_EYEBROW;
    if (idx <= 35)  return COL_NOSE;
    if (idx <= 47)  return COL_EYE;
    if (idx <= 59)  return COL_MOUTH_OUT;
    return COL_MOUTH_IN;
}

// Draw a polyline through a subset of landmarks (closed or open)
static void draw_contour(cv::Mat& img,
                          const std::vector<cv::Point2f>& lm,
                          int from, int to,
                          const cv::Scalar& col,
                          bool closed = false)
{
    std::vector<cv::Point> pts;
    pts.reserve(to - from + 1);
    for (int i = from; i <= to; ++i)
        pts.emplace_back(static_cast<int>(lm[i].x),
                         static_cast<int>(lm[i].y));
    cv::polylines(img, pts, closed, col, 1, cv::LINE_AA);
}

static void draw_landmarks(cv::Mat& img,
                            const cv::Rect& face,
                            const std::vector<cv::Point2f>& lm)
{
    if (lm.size() < 68) return;

    // Face bounding box
    cv::rectangle(img, face, COL_BOX, 1, cv::LINE_AA);

    // Region contours
    draw_contour(img, lm,  0, 16, COL_JAW,       false); // jaw
    draw_contour(img, lm, 17, 21, COL_EYEBROW,   false); // left brow
    draw_contour(img, lm, 22, 26, COL_EYEBROW,   false); // right brow
    draw_contour(img, lm, 27, 30, COL_NOSE,      false); // nose bridge
    draw_contour(img, lm, 30, 35, COL_NOSE,      true);  // nose tip (closed)
    draw_contour(img, lm, 36, 41, COL_EYE,       true);  // left eye (closed)
    draw_contour(img, lm, 42, 47, COL_EYE,       true);  // right eye (closed)
    draw_contour(img, lm, 48, 59, COL_MOUTH_OUT, true);  // outer mouth (closed)
    draw_contour(img, lm, 60, 67, COL_MOUTH_IN,  true);  // inner mouth (closed)

    // Landmark dots — drawn on top of contours
    for (int i = 0; i < 68; ++i) {
        cv::circle(img,
                   cv::Point(static_cast<int>(lm[i].x), static_cast<int>(lm[i].y)),
                   2, landmark_colour(i), -1, cv::LINE_AA);
    }
}

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
    const FaceLandmarkDetector::Result& det,
    double fps,
    int canvas_w, int canvas_h)
{
    const int tw = canvas_w / 2, th = canvas_h;
    char buf[64];

    // Left tile: original
    cv::Mat t0;
    cv::resize(bgr, t0, {tw, th}, 0, 0, cv::INTER_AREA);
    tile_label(t0, "Original");

    // Right tile: landmark overlay
    cv::Mat t1;
    cv::resize(bgr, t1, {tw, th}, 0, 0, cv::INTER_AREA);

    // Scale factor from full capture resolution to tile resolution
    const float sx = static_cast<float>(tw) / bgr.cols;
    const float sy = static_cast<float>(th) / bgr.rows;

    for (size_t i = 0; i < det.faces.size(); ++i) {
        // Scale face rect to tile
        const cv::Rect& f = det.faces[i];
        cv::Rect tf(static_cast<int>(f.x * sx),
                    static_cast<int>(f.y * sy),
                    static_cast<int>(f.width  * sx),
                    static_cast<int>(f.height * sy));

        // Scale landmarks to tile
        std::vector<cv::Point2f> tlm;
        if (i < det.landmarks.size()) {
            tlm.reserve(det.landmarks[i].size());
            for (const auto& p : det.landmarks[i])
                tlm.emplace_back(p.x * sx, p.y * sy);
        }

        draw_landmarks(t1, tf, tlm);
    }

    int n_faces = static_cast<int>(det.faces.size());
    int n_lm    = static_cast<int>(det.landmarks.size());
    std::snprintf(buf, sizeof(buf), "faces: %d  landmarks: %d",
                  n_faces, n_lm * 68);
    tile_label(t1, "Face landmarks  (LBF)", {buf});

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

    // Try to find the Haar cascade in the default OpenCV data directory
    // so the user only needs to supply --model for lbfmodel.yaml
    const std::string default_cascade =
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml";

    c.cascade_path = default_cascade;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--width")            { c.width          = std::stoi(get_next(i)); ++i; }
        else if (a == "--height")           { c.height         = std::stoi(get_next(i)); ++i; }
        else if (a == "--fps")              { c.fps            = std::stoi(get_next(i)); ++i; }
        else if (a == "--sensor")           { c.sensor_id      = std::stoi(get_next(i)); ++i; }
        else if (a == "--flip")             { c.flip_method    = std::stoi(get_next(i)); ++i; }
        else if (a == "--udp-host")         { c.udp_host       = get_next(i); ++i; }
        else if (a == "--udp-port")         { c.udp_port       = std::stoi(get_next(i)); ++i; }
        else if (a == "--stream-name")      { c.stream_name    = get_next(i); ++i; }
        else if (a == "--bitrate")          { c.bitrate        = std::stoi(get_next(i)); ++i; }
        else if (a == "--iframeinterval")   { c.iframeinterval = std::stoi(get_next(i)); ++i; }
        else if (a == "--stream-width")     { c.stream_w       = std::stoi(get_next(i)); ++i; }
        else if (a == "--stream-height")    { c.stream_h       = std::stoi(get_next(i)); ++i; }
        else if (a == "--cascade")          { c.cascade_path   = get_next(i); ++i; }
        else if (a == "--model")            { c.model_path     = get_next(i); ++i; }
        else if (a == "--detect-scale")     { c.detect_scale   = std::stof(get_next(i)); ++i; }
        else if (a == "--scale-factor")     { c.scale_factor   = std::stod(get_next(i)); ++i; }
        else if (a == "--min-neighbours")   { c.min_neighbours = std::stoi(get_next(i)); ++i; }
        else if (a == "--min-face")         { c.min_face_px    = std::stoi(get_next(i)); ++i; }
        else if (a == "--lock-ae")          { c.lock_ae        = true; }
        else if (a == "--lock-awb")         { c.lock_awb       = true; }
        else if (a == "--no-fps")           { c.show_fps       = false; }
        else if (a == "--help" || a == "-h") {
            std::cout <<
                "cv_face -- OpenCV FacemarkLBF 68-point landmark detection\n\n"
                "  REQUIRED:\n"
                "  --model PATH        path to lbfmodel.yaml (~54 MB)\n"
                "                      download: wget https://raw.githubusercontent.com/\n"
                "                        kurnianggoro/GSOC2017/master/data/lbfmodel.yaml\n\n"
                "  OPTIONAL:\n"
                "  --cascade PATH      Haar cascade xml  [" + default_cascade + "]\n"
                "  --width/height/fps/sensor/flip\n"
                "  --udp-host/udp-port/stream-name/bitrate/iframeinterval\n"
                "  --stream-width/stream-height\n"
                "  --detect-scale F    [0.5]  downscale factor for detection (0.25-1.0)\n"
                "  --scale-factor F    [1.1]  Haar cascade scaleFactor\n"
                "  --min-neighbours N  [3]    Haar cascade minNeighbours\n"
                "  --min-face N        [60]   minimum face size in px (detection scale)\n"
                "  --lock-ae  --lock-awb  --no-fps\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown: " << a << "\n"; std::exit(1);
        }
    }

    if (c.model_path.empty()) {
        std::cerr << "[error] --model is required. Run with --help for details.\n";
        std::exit(1);
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
              << "  cv_face -- FacemarkLBF 68-point\n"
              << "==================================================\n"
              << "  Capture        : " << cfg.width << "x" << cfg.height
              <<                     " @ " << cfg.fps << "fps\n"
              << "  Stream out     : " << cfg.stream_w << "x" << cfg.stream_h << "\n"
              << "  Detect scale   : " << cfg.detect_scale
              <<   "  ("
              <<   static_cast<int>(cfg.width  * cfg.detect_scale) << "x"
              <<   static_cast<int>(cfg.height * cfg.detect_scale) << ")\n"
              << "  Scale factor   : " << cfg.scale_factor << "\n"
              << "  Min neighbours : " << cfg.min_neighbours << "\n"
              << "  Min face       : " << cfg.min_face_px << "px\n"
              << "  Cascade        : " << cfg.cascade_path << "\n"
              << "  Model          : " << cfg.model_path << "\n"
              << "  UDP ->         : " << cfg.udp_host << ":" << cfg.udp_port << "\n"
              << "  Stream         : http://<host>:8889/" << cfg.stream_name << "\n"
              << "==================================================\n\n";

    FaceLandmarkDetector detector;
    try {
        detector.init(cfg);
    } catch (const std::exception& e) {
        std::cerr << "[facemark] init FAILED: " << e.what() << "\n";
        return 1;
    }

    std::string cap_pipe = capture_pipeline(cfg.sensor_id, cfg.width, cfg.height,
                                            cfg.fps, cfg.flip_method,
                                            cfg.lock_ae, cfg.lock_awb);
    std::cout << "[camera] pipeline: " << cap_pipe << "\n\n";

    CameraBuffer cam_buf;
    std::thread  cam_thr(camera_thread, std::cref(cap_pipe), std::ref(cam_buf));

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

        FaceLandmarkDetector::Result det;
        try {
            det = detector.detect(cur_bgr, cfg);
        } catch (const std::exception& e) {
            std::cerr << "[facemark] detect ERROR: " << e.what() << "\n"; continue;
        }

        auto   t_now     = std::chrono::steady_clock::now();
        double dt        = std::chrono::duration<double>(t_now - t_last).count();
        t_last           = t_now;
        fps_smooth       = 0.9*fps_smooth + 0.1*(dt > 0 ? 1.0/dt : 0.0);
        ++frame_n;

        cv::Mat vis = build_sidebyside(cur_bgr, det,
                                       fps_smooth,
                                       cfg.stream_w, cfg.stream_h);
        if (gst_push.ok) gst_push.push(vis);

        if (cfg.show_fps && fps_smooth > 0 &&
            frame_n % static_cast<long>(fps_smooth + 0.5) == 0) {
            std::cout << "\r[main] frame=" << std::setw(6) << frame_n
                      << "  fps=" << std::fixed << std::setprecision(1) << fps_smooth
                      << "  faces=" << det.faces.size()
                      << "          " << std::flush;
        }
    }

    std::cout << "\n[main] shutting down...\n";
    gst_push.stop();
    cam_thr.join();
    std::cout << "[main] done.\n";
    return 0;
}
