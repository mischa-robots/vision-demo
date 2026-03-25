// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>

/**
 * yolo_postprocess.hpp
 * ─────────────────────
 * Postprocessing for YOLO detection output.
 *
 * Steps:
 *   1. Parse raw (84, 8400) output tensor
 *   2. Filter anchors by confidence score
 *   3. Non-Maximum Suppression (NMS)
 *   4. Scale boxes back to original frame size
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

// ════════════════════════════════════════════════════════════════════════════
// Label loader — reads one class name per line from a text file
// ════════════════════════════════════════════════════════════════════════════
inline std::vector<std::string> load_labels(const std::string& path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Cannot open labels file: " + path);

    std::vector<std::string> names;
    std::string line;
    while (std::getline(f, line)) {
        // Strip trailing CR/spaces (Windows line endings)
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        if (!line.empty())
            names.push_back(line);
    }
    if (names.empty())
        throw std::runtime_error("Labels file is empty: " + path);

    std::cout << "[LABELS] Loaded " << names.size()
              << " labels from " << path << "\n";
    return names;
}

// ════════════════════════════════════════════════════════════════════════════
// Detection result — one detected object
// ════════════════════════════════════════════════════════════════════════════
struct Detection {
    int   x1, y1, x2, y2;  // bounding box in original frame pixels
    float score;
    int   class_id;
    std::string label;
};

// ── Deterministic colour per class (same colour every run) ───────────────────
inline cv::Scalar classColor(int class_id) {
    int h = (class_id * 137 + 42) % 180;  // spread colours evenly around hue wheel
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(h, 200, 220));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    auto* p = bgr.ptr<uint8_t>(0);
    return cv::Scalar(p[0], p[1], p[2]);
}

// ── Intersection-over-Union between two boxes ─────────────────────────────────
inline float iou(float ax1, float ay1, float ax2, float ay2,
                 float bx1, float by1, float bx2, float by2)
{
    float ix1  = std::max(ax1, bx1);
    float iy1  = std::max(ay1, by1);
    float ix2  = std::min(ax2, bx2);
    float iy2  = std::min(ay2, by2);
    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float area_a = (ax2 - ax1) * (ay2 - ay1);
    float area_b = (bx2 - bx1) * (by2 - by1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

// ════════════════════════════════════════════════════════════════════════════
// postprocess — parse TRT output and return filtered detections
//
// output     — raw float array (OUTPUT_ROWS * OUTPUT_COLS = 84 * 8400)
//              layout: [cx, cy, w, h, class_0..class_79] × 8400 anchors
//              stored row-major as (84, 8400)
// orig_w/h   — original frame size (for scaling boxes back)
// labels     — class name strings
// conf_thresh — minimum confidence to keep a detection
// iou_thresh  — NMS overlap threshold
// ════════════════════════════════════════════════════════════════════════════
inline std::vector<Detection> postprocess(
    const float*                    output,
    int                             orig_w,
    int                             orig_h,
    const std::vector<std::string>& labels,
    float conf_thresh = 0.25f,
    float iou_thresh  = 0.45f,
    int   model_size  = 640)
{
    const int NUM_CLASSES = static_cast<int>(labels.size());
    constexpr int COLS = 8400;

    float sx = static_cast<float>(orig_w) / model_size;
    float sy = static_cast<float>(orig_h) / model_size;

    // ── 1. Parse all anchors, keep those above confidence threshold ───────────
    struct Candidate {
        float cx, cy, w, h, score;
        int   class_id;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(512);

    for (int a = 0; a < COLS; ++a) {
        float best_score = 0.0f;
        int   best_cls   = 0;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float s = output[(4 + c) * COLS + a];
            if (s > best_score) { best_score = s; best_cls = c; }
        }
        if (best_score < conf_thresh) continue;

        candidates.push_back({
            output[0 * COLS + a], output[1 * COLS + a],
            output[2 * COLS + a], output[3 * COLS + a],
            best_score, best_cls
        });
    }

    if (candidates.empty()) return {};

    // ── 2. Sort highest score first ───────────────────────────────────────────
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b){
                  return a.score > b.score; });

    // ── 3. NMS — suppress overlapping boxes of the same class ─────────────────
    std::vector<bool> suppressed(candidates.size(), false);
    std::vector<Detection> results;

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        const auto& ci = candidates[i];

        float bx1 = ci.cx - ci.w * 0.5f;
        float by1 = ci.cy - ci.h * 0.5f;
        float bx2 = ci.cx + ci.w * 0.5f;
        float by2 = ci.cy + ci.h * 0.5f;

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j] || candidates[j].class_id != ci.class_id) continue;
            const auto& cj = candidates[j];
            if (iou(bx1, by1, bx2, by2,
                    cj.cx - cj.w * 0.5f, cj.cy - cj.h * 0.5f,
                    cj.cx + cj.w * 0.5f, cj.cy + cj.h * 0.5f) > iou_thresh)
                suppressed[j] = true;
        }

        // Scale box to original frame size and clamp to image bounds
        Detection det;
        det.x1       = static_cast<int>(std::max(0.0f, bx1 * sx));
        det.y1       = static_cast<int>(std::max(0.0f, by1 * sy));
        det.x2       = static_cast<int>(std::min(static_cast<float>(orig_w), bx2 * sx));
        det.y2       = static_cast<int>(std::min(static_cast<float>(orig_h), by2 * sy));
        det.score    = ci.score;
        det.class_id = ci.class_id;
        det.label    = (ci.class_id < NUM_CLASSES) ? labels[ci.class_id]
                                                    : std::to_string(ci.class_id);
        results.push_back(det);
    }

    return results;
}

// ════════════════════════════════════════════════════════════════════════════
// drawDetections — draw boxes and labels onto a frame
// ════════════════════════════════════════════════════════════════════════════
inline void drawDetections(cv::Mat&                       frame,
                           const std::vector<Detection>&  dets,
                           double                         fps = -1.0)
{
    for (const auto& d : dets) {
        cv::Scalar color = classColor(d.class_id);

        // Bounding box
        cv::rectangle(frame, {d.x1, d.y1}, {d.x2, d.y2}, color, 2);

        // Label: "person 87%"
        std::string text = d.label + " " +
                           std::to_string(static_cast<int>(d.score * 100)) + "%";
        int baseline = 0;
        cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);

        // Filled background behind label
        cv::rectangle(frame,
                      {d.x1, d.y1 - ts.height - 6},
                      {d.x1 + ts.width + 4, d.y1},
                      color, cv::FILLED);
        cv::putText(frame, text, {d.x1 + 2, d.y1 - 4},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    // FPS counter top-left
    if (fps > 0) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "FPS: %.1f", fps);
        cv::putText(frame, buf, {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }
}
