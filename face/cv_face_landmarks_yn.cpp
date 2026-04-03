// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>

/**
 * cv_face_landmarks_yn.cpp
 * ------------------------
 * Real-time face landmark detection for Jetson Nano using OpenCV's
 * YuNet DNN face detector + FacemarkLBF. Streams a side-by-side canvas
 * to the browser via MediaMTX.
 *
 * Pipeline:
 *   YuNet / FaceDetectorYN (CPU, DNN) → face ROIs + 5 key-point landmarks
 *   FacemarkLBF            (CPU)      → 68 landmarks per face
 *
 * Improvements over cv_face_landmarks (Haar cascade version):
 *   - YuNet is a CNN trained on WIDER Face; it detects side faces,
 *     partially occluded faces, and small faces that the Haar cascade misses.
 *   - ~5 ms vs ~25 ms per 320×320 frame on the same hardware.
 *   - Only 337 KB ONNX model file (vs 908 KB cascade XML).
 *   - No scaleFactor / minNeighbours tuning needed.
 *   - The 5 YuNet key-points (eyes, nose, mouth corners) are drawn on the
 *     overlay in addition to the full 68-point LBF mesh.
 *
 * YuNet output row format (15 values per face, CV_32F):
 *   [x, y, w, h,  re_x, re_y,  le_x, le_y,  nt_x, nt_y,
 *    rcm_x, rcm_y,  lcm_x, lcm_y,  score]
 *   re=right eye, le=left eye, nt=nose tip, rcm/lcm=mouth corners
 *
 * Side-by-side layout (default 1280x360):
 *   +---------------------+---------------------+
 *   |  Original stream    |  Landmark overlay   |
 *   +---------------------+---------------------+
 *
 * Landmark overlay:
 *   - 68 landmark dots colour-coded by facial region
 *   - Facial region contours (jaw, eyebrows, nose, eyes, mouth)
 *   - Face bounding box + confidence score
 *   - 5 YuNet key-points (coloured circles)
 *
 * Required model files:
 *   face_detection_yunet_2023mar.onnx  — ~337 KB, download once:
 *     wget -P ./models https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
 *   lbfmodel.yaml                      — ~54 MB, download once:
 *     wget -P ./models https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml
 *
 * Watch the stream at:
 *   WebRTC : http://<jetson-ip>:8889/face
 *   HLS    : http://<jetson-ip>:8888/face
 *   RTSP   : rtsp://<jetson-ip>:8554/face
 *
 * Usage:
 *   ./build/cv_face_yn --yunet  /path/to/face_detection_yunet_2023mar.onnx \
 *                                --model  /path/to/lbfmodel.yaml
 *   ./build/cv_face_yn --help
 */

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/face.hpp>
#include <opencv2/objdetect.hpp>   // cv::FaceDetectorYN

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

    // Model paths — both required
    std::string yunet_path;   // face_detection_yunet_2023mar.onnx
    std::string model_path;   // lbfmodel.yaml

    // YuNet detection parameters
    float  detect_scale     = 0.5f;  // downscale factor for YuNet detection pass
    float  score_threshold  = 0.9f;  // YuNet confidence threshold
    float  nms_threshold    = 0.3f;  // YuNet NMS IoU threshold
    int    top_k            = 100;   // YuNet max candidates before NMS
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
// FaceLandmarkDetector  (YuNet + FacemarkLBF)
// ============================================================================
struct FaceLandmarkDetector {
    cv::Ptr<cv::FaceDetectorYN>   yunet;
    cv::Ptr<cv::face::Facemark>   facemark;

    void init(const Config& cfg)
    {
        // YuNet is initialised with the downscaled detection resolution.
        // setInputSize() is called each frame when the image size is known,
        // so this initial size is just a placeholder.
        const int dw = static_cast<int>(cfg.width  * cfg.detect_scale);
        const int dh = static_cast<int>(cfg.height * cfg.detect_scale);

        yunet = cv::FaceDetectorYN::create(
            cfg.yunet_path, "",
            cv::Size(dw, dh),
            cfg.score_threshold,
            cfg.nms_threshold,
            cfg.top_k);

        facemark = cv::face::FacemarkLBF::create();
        facemark->loadModel(cfg.model_path);

        std::cout << "[facemark] yunet  : " << cfg.yunet_path << "\n"
                  << "[facemark] model  : " << cfg.model_path << "\n";
    }

    // YuNet output row: [x, y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y,
    //                    rcm_x, rcm_y, lcm_x, lcm_y, score]  (15 float values)
    struct Face {
        cv::Rect              box;
        float                 score;
        cv::Point2f           kp[5];   // re, le, nt, rcm, lcm
    };

    struct Result {
        std::vector<Face>                        faces;
        std::vector<std::vector<cv::Point2f>>    landmarks;  // one entry per face
    };

    // Detect on a downscaled BGR image, scale results back to full resolution.
    Result detect(const cv::Mat& bgr_full, const Config& cfg)
    {
        Result r;
        const float inv = 1.f / cfg.detect_scale;

        // Downscale for the detection pass
        cv::Mat small;
        cv::resize(bgr_full, small,
                   cv::Size(static_cast<int>(bgr_full.cols * cfg.detect_scale),
                            static_cast<int>(bgr_full.rows * cfg.detect_scale)),
                   0, 0, cv::INTER_AREA);

        // Update YuNet input size to match the actual downscaled frame size
        yunet->setInputSize(small.size());

        cv::Mat faces_mat;
        yunet->detect(small, faces_mat);

        if (faces_mat.empty() || faces_mat.rows == 0) return r;

        // Parse YuNet output rows and scale back to full resolution.
        // Collect cv::Rect list in parallel for FacemarkLBF.
        std::vector<cv::Rect> rects_small;

        for (int i = 0; i < faces_mat.rows; ++i) {
            const float* d = faces_mat.ptr<float>(i);
            // d[0..3] = x, y, w, h  (at detect-scale resolution)
            // d[4..13] = 5 key-points (x,y pairs)
            // d[14]   = confidence score

            Face f;
            f.score = d[14];
            f.box   = cv::Rect(
                static_cast<int>(d[0] * inv),
                static_cast<int>(d[1] * inv),
                static_cast<int>(d[2] * inv),
                static_cast<int>(d[3] * inv));

            for (int k = 0; k < 5; ++k)
                f.kp[k] = cv::Point2f(d[4 + 2*k] * inv, d[5 + 2*k] * inv);

            r.faces.push_back(f);

            // Keep a small-resolution rect for LBF fitting
            rects_small.emplace_back(
                static_cast<int>(d[0]),
                static_cast<int>(d[1]),
                static_cast<int>(d[2]),
                static_cast<int>(d[3]));
        }

        // FacemarkLBF fit — runs on the small BGR image with small rects.
        // LBF accepts BGR directly; no greyscale conversion needed here.
        std::vector<std::vector<cv::Point2f>> lm_small;
        bool ok = facemark->fit(small, rects_small, lm_small);
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

// YuNet 5 key-point colours: re, le, nt, rcm, lcm (BGR)
static const cv::Scalar COL_KP[5] = {
    {255,  50, 255},   // right eye  — magenta
    { 50, 255, 255},   // left eye   — yellow
    {  0, 200, 255},   // nose tip   — amber
    { 50, 255,  50},   // right mouth corner — green
    { 50, 255,  50},   // left mouth corner  — green
};

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

static void draw_face(cv::Mat& img,
                      const FaceLandmarkDetector::Face& face,
                      const std::vector<cv::Point2f>& lm)
{
    // Bounding box + confidence score
    cv::rectangle(img, face.box, COL_BOX, 1, cv::LINE_AA);
    char score_buf[16];
    std::snprintf(score_buf, sizeof(score_buf), "%.2f", face.score);
    cv::putText(img, score_buf,
                cv::Point(face.box.x, face.box.y - 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.40,
                cv::Scalar(0,0,0),   2, cv::LINE_AA);
    cv::putText(img, score_buf,
                cv::Point(face.box.x, face.box.y - 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.40,
                COL_BOX,             1, cv::LINE_AA);

    // YuNet 5 key-points
    for (int k = 0; k < 5; ++k)
        cv::circle(img,
                   cv::Point(static_cast<int>(face.kp[k].x),
                             static_cast<int>(face.kp[k].y)),
                   4, COL_KP[k], -1, cv::LINE_AA);

    // 68-point LBF mesh
    if (lm.size() < 68) return;

    draw_contour(img, lm,  0, 16, COL_JAW,       false);
    draw_contour(img, lm, 17, 21, COL_EYEBROW,   false);
    draw_contour(img, lm, 22, 26, COL_EYEBROW,   false);
    draw_contour(img, lm, 27, 30, COL_NOSE,      false);
    draw_contour(img, lm, 30, 35, COL_NOSE,      true);
    draw_contour(img, lm, 36, 41, COL_EYE,       true);
    draw_contour(img, lm, 42, 47, COL_EYE,       true);
    draw_contour(img, lm, 48, 59, COL_MOUTH_OUT, true);
    draw_contour(img, lm, 60, 67, COL_MOUTH_IN,  true);

    for (int i = 0; i < 68; ++i)
        cv::circle(img,
                   cv::Point(static_cast<int>(lm[i].x), static_cast<int>(lm[i].y)),
                   2, landmark_colour(i), -1, cv::LINE_AA);
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
        // Scale face to tile
        const auto& f = det.faces[i];
        FaceLandmarkDetector::Face tf;
        tf.score = f.score;
        tf.box   = cv::Rect(static_cast<int>(f.box.x      * sx),
                            static_cast<int>(f.box.y      * sy),
                            static_cast<int>(f.box.width  * sx),
                            static_cast<int>(f.box.height * sy));
        for (int k = 0; k < 5; ++k)
            tf.kp[k] = cv::Point2f(f.kp[k].x * sx, f.kp[k].y * sy);

        // Scale landmarks to tile
        std::vector<cv::Point2f> tlm;
        if (i < det.landmarks.size()) {
            tlm.reserve(det.landmarks[i].size());
            for (const auto& p : det.landmarks[i])
                tlm.emplace_back(p.x * sx, p.y * sy);
        }

        draw_face(t1, tf, tlm);
    }

    int n_faces = static_cast<int>(det.faces.size());
    int n_lm    = static_cast<int>(det.landmarks.size());
    std::snprintf(buf, sizeof(buf), "faces: %d  landmarks: %d",
                  n_faces, n_lm * 68);
    tile_label(t1, "Face landmarks  (YuNet + LBF)", {buf});

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
        else if (a == "--yunet")            { c.yunet_path      = get_next(i); ++i; }
        else if (a == "--model")            { c.model_path      = get_next(i); ++i; }
        else if (a == "--detect-scale")     { c.detect_scale    = std::stof(get_next(i)); ++i; }
        else if (a == "--score-threshold")  { c.score_threshold = std::stof(get_next(i)); ++i; }
        else if (a == "--nms-threshold")    { c.nms_threshold   = std::stof(get_next(i)); ++i; }
        else if (a == "--top-k")            { c.top_k           = std::stoi(get_next(i)); ++i; }
        else if (a == "--lock-ae")          { c.lock_ae         = true; }
        else if (a == "--lock-awb")         { c.lock_awb        = true; }
        else if (a == "--no-fps")           { c.show_fps        = false; }
        else if (a == "--help" || a == "-h") {
            std::cout <<
                "cv_face_yn -- YuNet DNN + FacemarkLBF 68-point\n\n"
                "  REQUIRED:\n"
                "  --yunet PATH        face_detection_yunet_2023mar.onnx (~337 KB)\n"
                "                      wget -P ./models https://github.com/opencv/opencv_zoo/\n"
                "                        raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx\n"
                "  --model PATH        lbfmodel.yaml (~54 MB)\n"
                "                      wget -P ./models https://raw.githubusercontent.com/\n"
                "                        kurnianggoro/GSOC2017/master/data/lbfmodel.yaml\n\n"
                "  OPTIONAL:\n"
                "  --width/height/fps/sensor/flip\n"
                "  --udp-host/udp-port/stream-name/bitrate/iframeinterval\n"
                "  --stream-width/stream-height\n"
                "  --detect-scale F    [0.5]  downscale factor for detection (0.25-1.0)\n"
                "  --score-threshold F [0.9]  YuNet confidence threshold\n"
                "  --nms-threshold F   [0.3]  YuNet NMS IoU threshold\n"
                "  --top-k N           [100]  YuNet max candidates before NMS\n"
                "  --lock-ae  --lock-awb  --no-fps\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown: " << a << "\n"; std::exit(1);
        }
    }

    if (c.yunet_path.empty()) {
        std::cerr << "[error] --yunet is required. Run with --help for details.\n";
        std::exit(1);
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
              << "  cv_face_yn -- YuNet + FacemarkLBF\n"
              << "==================================================\n"
              << "  Capture        : " << cfg.width << "x" << cfg.height
              <<                     " @ " << cfg.fps << "fps\n"
              << "  Stream out     : " << cfg.stream_w << "x" << cfg.stream_h << "\n"
              << "  Detect scale   : " << cfg.detect_scale
              <<   "  ("
              <<   static_cast<int>(cfg.width  * cfg.detect_scale) << "x"
              <<   static_cast<int>(cfg.height * cfg.detect_scale) << ")\n"
              << "  Score thresh   : " << cfg.score_threshold << "\n"
              << "  NMS thresh     : " << cfg.nms_threshold << "\n"
              << "  Top-k          : " << cfg.top_k << "\n"
              << "  YuNet model    : " << cfg.yunet_path << "\n"
              << "  LBF model      : " << cfg.model_path << "\n"
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
