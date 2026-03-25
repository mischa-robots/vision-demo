// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>

/**
 * yoloe_seg_postprocess.hpp
 * ──────────────────────────
 * Postprocessing for YOLOE segmentation output.
 * num_classes is runtime-dynamic (unlike fixed-80 YOLO-seg).
 *
 * Mask decode uses a simple CPU sigmoid loop — no CUDA dependency.
 * Identical interface to the yolo-trt version so yoloe_seg.cpp needs
 * only its #includes and engine type changed.
 *
 * Output bindings:
 *   output0: (1, 4+num_classes+32, 8400)
 *   output1: (1, 32, 160, 160)
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

// ── Fixed sizes (proto tensor shape never changes) ────────────────────────────
static constexpr int YOLOE_SEG_COLS     = 8400;
static constexpr int YOLOE_PROTO_MASKS  = 32;
static constexpr int YOLOE_PROTO_H      = 160;
static constexpr int YOLOE_PROTO_W      = 160;
static constexpr int YOLOE_PROTO_PIXELS = YOLOE_PROTO_H * YOLOE_PROTO_W; // 25,600

// output1 size is fixed regardless of num_classes
static constexpr int YOLOE_OUT1_SIZE =
    YOLOE_PROTO_MASKS * YOLOE_PROTO_H * YOLOE_PROTO_W; // 819,200

// output0 size depends on num_classes — use this helper
inline int yoloe_out0_size(int num_classes) {
    return (4 + num_classes + YOLOE_PROTO_MASKS) * YOLOE_SEG_COLS;
}

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
// Detection + mask result — one detected object with its segmentation mask
// ════════════════════════════════════════════════════════════════════════════
struct YoloeSegDetection {
    int   x1, y1, x2, y2;
    float score;
    int   class_id;
    std::string label;
    cv::Mat mask;   // CV_8UC1, same size as original frame, values 0 or 255
};

// ── Deterministic colour per class ───────────────────────────────────────────
inline cv::Scalar classColor(int class_id) {
    int h = (class_id * 137 + 42) % 180;
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(h, 200, 220));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    auto* p = bgr.ptr<uint8_t>(0);
    return cv::Scalar(p[0], p[1], p[2]);
}

// ── Sigmoid activation ────────────────────────────────────────────────────────
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// ── Intersection-over-Union between two boxes ─────────────────────────────────
inline float yoloe_iou(float ax1, float ay1, float ax2, float ay2,
                       float bx1, float by1, float bx2, float by2)
{
    float inter = std::max(0.0f, std::min(ax2, bx2) - std::max(ax1, bx1)) *
                  std::max(0.0f, std::min(ay2, by2) - std::max(ay1, by1));
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-6f);
}

// ── Decode one mask: dot product of 32 coefficients with 32 proto masks ───────
// Returns CV_32FC1 (160×160) after sigmoid — CPU version, no CUDA required
inline cv::Mat yoloe_decode_mask(const float* coeffs, const float* protos)
{
    cv::Mat mask(YOLOE_PROTO_H, YOLOE_PROTO_W, CV_32FC1, 0.0f);
    float* mp = mask.ptr<float>(0);

    for (int k = 0; k < YOLOE_PROTO_MASKS; ++k) {
        const float* proto_k = protos + k * YOLOE_PROTO_PIXELS;
        float c = coeffs[k];
        for (int i = 0; i < YOLOE_PROTO_PIXELS; ++i)
            mp[i] += c * proto_k[i];
    }
    for (int i = 0; i < YOLOE_PROTO_PIXELS; ++i)
        mp[i] = sigmoid(mp[i]);

    return mask;
}

// ════════════════════════════════════════════════════════════════════════════
// yoloe_seg_postprocess — parse DNN outputs and return filtered detections
//
// output0  — float* — (4+num_classes+32) × 8400
// output1  — float* — 32 × 160 × 160
// labels   — class name strings (size = num_classes)
// ════════════════════════════════════════════════════════════════════════════
inline std::vector<YoloeSegDetection> yoloe_seg_postprocess(
    const float*                    output0,
    const float*                    output1,
    const std::vector<std::string>& labels,
    int orig_w, int orig_h,
    float conf_thresh,
    float iou_thresh,
    float mask_thresh,
    int   model_size = 640)
{
    const int num_classes = static_cast<int>(labels.size());
    float sx = static_cast<float>(orig_w) / model_size;
    float sy = static_cast<float>(orig_h) / model_size;

    // ── 1. Parse anchors, filter by confidence ────────────────────────────────
    struct Candidate {
        float cx, cy, w, h, score;
        int   class_id;
        float coeffs[YOLOE_PROTO_MASKS];
    };
    std::vector<Candidate> candidates;
    candidates.reserve(256);

    for (int a = 0; a < YOLOE_SEG_COLS; ++a) {
        float best_score = 0.0f;
        int   best_cls   = 0;
        for (int c = 0; c < num_classes; ++c) {
            float s = output0[(4 + c) * YOLOE_SEG_COLS + a];
            if (s > best_score) { best_score = s; best_cls = c; }
        }
        if (best_score < conf_thresh) continue;

        Candidate cand;
        cand.cx       = output0[0 * YOLOE_SEG_COLS + a];
        cand.cy       = output0[1 * YOLOE_SEG_COLS + a];
        cand.w        = output0[2 * YOLOE_SEG_COLS + a];
        cand.h        = output0[3 * YOLOE_SEG_COLS + a];
        cand.score    = best_score;
        cand.class_id = best_cls;
        for (int k = 0; k < YOLOE_PROTO_MASKS; ++k)
            cand.coeffs[k] = output0[(4 + num_classes + k) * YOLOE_SEG_COLS + a];
        candidates.push_back(cand);
    }

    if (candidates.empty()) return {};

    // ── 2. Sort highest score first ───────────────────────────────────────────
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b){
                  return a.score > b.score; });

    // ── 3. NMS ────────────────────────────────────────────────────────────────
    std::vector<bool> suppressed(candidates.size(), false);
    std::vector<int>  kept;

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(static_cast<int>(i));
        const auto& ci = candidates[i];
        float bx1 = ci.cx - ci.w * 0.5f, by1 = ci.cy - ci.h * 0.5f;
        float bx2 = ci.cx + ci.w * 0.5f, by2 = ci.cy + ci.h * 0.5f;

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j] || candidates[j].class_id != ci.class_id) continue;
            const auto& cj = candidates[j];
            if (yoloe_iou(bx1, by1, bx2, by2,
                          cj.cx - cj.w * 0.5f, cj.cy - cj.h * 0.5f,
                          cj.cx + cj.w * 0.5f, cj.cy + cj.h * 0.5f) > iou_thresh)
                suppressed[j] = true;
        }
    }

    // ── 4. Decode masks per detection (CPU sigmoid loop) ──────────────────────
    std::vector<YoloeSegDetection> results;
    results.reserve(kept.size());

    for (int idx : kept) {
        const auto& c = candidates[idx];
        float bx1 = c.cx - c.w * 0.5f, by1 = c.cy - c.h * 0.5f;
        float bx2 = c.cx + c.w * 0.5f, by2 = c.cy + c.h * 0.5f;

        YoloeSegDetection det;
        det.x1       = static_cast<int>(std::max(0.0f, bx1 * sx));
        det.y1       = static_cast<int>(std::max(0.0f, by1 * sy));
        det.x2       = static_cast<int>(std::min(static_cast<float>(orig_w), bx2 * sx));
        det.y2       = static_cast<int>(std::min(static_cast<float>(orig_h), by2 * sy));
        det.score    = c.score;
        det.class_id = c.class_id;
        det.label    = (c.class_id < static_cast<int>(labels.size()))
                       ? labels[c.class_id] : std::to_string(c.class_id);

        // CPU decode: dot product + sigmoid at proto resolution, then resize
        cv::Mat proto_mask = yoloe_decode_mask(c.coeffs, output1);
        cv::Mat binary;
        cv::threshold(proto_mask, binary, mask_thresh, 1.0f, cv::THRESH_BINARY);

        cv::Mat mask_full;
        cv::resize(binary, mask_full, {orig_w, orig_h}, 0, 0, cv::INTER_NEAREST);
        mask_full.convertTo(mask_full, CV_8UC1, 255.0);

        // Crop mask to bounding box
        cv::Mat bbox_mask = cv::Mat::zeros(orig_h, orig_w, CV_8UC1);
        cv::rectangle(bbox_mask, {det.x1, det.y1}, {det.x2, det.y2},
                      cv::Scalar(255), cv::FILLED);
        cv::bitwise_and(mask_full, bbox_mask, det.mask);

        results.push_back(std::move(det));
    }

    return results;
}

// ════════════════════════════════════════════════════════════════════════════
// drawYoloeSegDetections — filled overlay + contour + label
// ════════════════════════════════════════════════════════════════════════════
inline void drawYoloeSegDetections(cv::Mat&                             frame,
                                   const std::vector<YoloeSegDetection>& dets,
                                   double fps   = -1.0,
                                   float  alpha = 0.4f)
{
    cv::Mat overlay = frame.clone();
    for (const auto& d : dets) {
        if (!d.mask.empty())
            overlay.setTo(classColor(d.class_id), d.mask);
    }
    cv::addWeighted(overlay, alpha, frame, 1.0f - alpha, 0, frame);

    for (const auto& d : dets) {
        cv::Scalar color = classColor(d.class_id);

        if (!d.mask.empty()) {
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(d.mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::drawContours(frame, contours, -1, color, 2);
        }

        cv::rectangle(frame, {d.x1, d.y1}, {d.x2, d.y2}, color, 1);

        std::string text = d.label + " " +
                           std::to_string(static_cast<int>(d.score * 100)) + "%";
        int baseline = 0;
        cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
        cv::rectangle(frame,
                      {d.x1, d.y1 - ts.height - 6},
                      {d.x1 + ts.width + 4, d.y1},
                      color, cv::FILLED);
        cv::putText(frame, text, {d.x1 + 2, d.y1 - 4},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    if (fps > 0) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "FPS: %.1f", fps);
        cv::putText(frame, buf, {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }
}
