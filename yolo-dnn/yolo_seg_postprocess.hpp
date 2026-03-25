// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>

/**
 * yolo_seg_postprocess.hpp
 * ─────────────────────────
 * Postprocessing for YOLO segmentation output.
 *
 * Steps:
 *   1. Parse output0 (116, 8400) — boxes + scores + 32 mask coefficients
 *   2. Parse output1 (32, 160, 160) — prototype masks
 *   3. Filter by confidence + NMS
 *   4. Decode masks: coefficients · protos → sigmoid → threshold
 *   5. Resize masks to original frame size, crop to bounding box
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

// ── Output tensor sizes (fixed for 640×640 YOLO-seg) ─────────────────────────
static constexpr int SEG_COLS        = 8400;   // anchor candidates
static constexpr int SEG_CLASSES     = 80;     // COCO classes
static constexpr int SEG_MASK_COEFFS = 32;     // mask coefficients per anchor
static constexpr int SEG_ROWS        = 4 + SEG_CLASSES + SEG_MASK_COEFFS; // 116

static constexpr int PROTO_MASKS     = 32;
static constexpr int PROTO_H         = 160;
static constexpr int PROTO_W         = 160;
static constexpr int OUTPUT0_SIZE    = SEG_ROWS  * SEG_COLS;              // 974,400
static constexpr int OUTPUT1_SIZE    = PROTO_MASKS * PROTO_H * PROTO_W;   // 819,200

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
struct SegDetection {
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
inline float iou(float ax1, float ay1, float ax2, float ay2,
                 float bx1, float by1, float bx2, float by2)
{
    float inter = std::max(0.0f, std::min(ax2, bx2) - std::max(ax1, bx1)) *
                  std::max(0.0f, std::min(ay2, by2) - std::max(ay1, by1));
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-6f);
}

// ── Decode one mask: dot product of 32 coefficients with 32 proto masks ───────
// Returns CV_32FC1 (160×160) after sigmoid
inline cv::Mat decodeMask(const float* coeffs, const float* protos)
{
    cv::Mat mask(PROTO_H, PROTO_W, CV_32FC1, 0.0f);
    float* mp = mask.ptr<float>(0);

    for (int k = 0; k < PROTO_MASKS; ++k) {
        const float* proto_k = protos + k * PROTO_H * PROTO_W;
        float c = coeffs[k];
        for (int i = 0; i < PROTO_H * PROTO_W; ++i)
            mp[i] += c * proto_k[i];
    }
    for (int i = 0; i < PROTO_H * PROTO_W; ++i)
        mp[i] = sigmoid(mp[i]);

    return mask;
}

// ════════════════════════════════════════════════════════════════════════════
// seg_postprocess — parse TRT outputs and return filtered detections with masks
//
// output0    — (116, 8400) boxes + scores + mask coefficients
// output1    — (32, 160, 160) prototype masks
// labels     — class name strings
// orig_w/h   — original frame size (for scaling)
// ════════════════════════════════════════════════════════════════════════════
inline std::vector<SegDetection> seg_postprocess(
    const float*                    output0,
    const float*                    output1,
    int                             orig_w,
    int                             orig_h,
    const std::vector<std::string>& labels,
    float conf_thresh = 0.25f,
    float iou_thresh  = 0.45f,
    float mask_thresh = 0.5f,
    int   model_size  = 640)
{
    const int NUM_CLASSES = static_cast<int>(labels.size());
    float sx = static_cast<float>(orig_w) / model_size;
    float sy = static_cast<float>(orig_h) / model_size;

    // ── 1. Parse anchors, filter by confidence ────────────────────────────────
    struct Candidate {
        float cx, cy, w, h, score;
        int   class_id;
        float coeffs[SEG_MASK_COEFFS];
    };
    std::vector<Candidate> candidates;
    candidates.reserve(256);

    for (int a = 0; a < SEG_COLS; ++a) {
        float best_score = 0.0f;
        int   best_cls   = 0;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float s = output0[(4 + c) * SEG_COLS + a];
            if (s > best_score) { best_score = s; best_cls = c; }
        }
        if (best_score < conf_thresh) continue;

        Candidate cand;
        cand.cx       = output0[0 * SEG_COLS + a];
        cand.cy       = output0[1 * SEG_COLS + a];
        cand.w        = output0[2 * SEG_COLS + a];
        cand.h        = output0[3 * SEG_COLS + a];
        cand.score    = best_score;
        cand.class_id = best_cls;
        for (int k = 0; k < SEG_MASK_COEFFS; ++k)
            cand.coeffs[k] = output0[(4 + NUM_CLASSES + k) * SEG_COLS + a];
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
            if (iou(bx1, by1, bx2, by2,
                    cj.cx - cj.w * 0.5f, cj.cy - cj.h * 0.5f,
                    cj.cx + cj.w * 0.5f, cj.cy + cj.h * 0.5f) > iou_thresh)
                suppressed[j] = true;
        }
    }

    // ── 4. Decode masks and build results ─────────────────────────────────────
    std::vector<SegDetection> results;
    results.reserve(kept.size());

    for (int idx : kept) {
        const auto& c = candidates[idx];
        float bx1 = c.cx - c.w * 0.5f, by1 = c.cy - c.h * 0.5f;
        float bx2 = c.cx + c.w * 0.5f, by2 = c.cy + c.h * 0.5f;

        SegDetection det;
        det.x1       = static_cast<int>(std::max(0.0f, bx1 * sx));
        det.y1       = static_cast<int>(std::max(0.0f, by1 * sy));
        det.x2       = static_cast<int>(std::min(static_cast<float>(orig_w), bx2 * sx));
        det.y2       = static_cast<int>(std::min(static_cast<float>(orig_h), by2 * sy));
        det.score    = c.score;
        det.class_id = c.class_id;
        det.label    = (c.class_id < static_cast<int>(labels.size()))
                       ? labels[c.class_id] : std::to_string(c.class_id);

        // Decode mask at proto resolution (160×160), threshold to binary
        cv::Mat proto_mask = decodeMask(c.coeffs, output1);
        cv::Mat binary;
        cv::threshold(proto_mask, binary, mask_thresh, 1.0f, cv::THRESH_BINARY);

        // Resize to full frame size
        cv::Mat mask_full;
        cv::resize(binary, mask_full, {orig_w, orig_h}, 0, 0, cv::INTER_NEAREST);
        mask_full.convertTo(mask_full, CV_8UC1, 255.0);

        // Zero out pixels outside the bounding box
        cv::Mat bbox_mask = cv::Mat::zeros(orig_h, orig_w, CV_8UC1);
        cv::rectangle(bbox_mask, {det.x1, det.y1}, {det.x2, det.y2},
                      cv::Scalar(255), cv::FILLED);
        cv::bitwise_and(mask_full, bbox_mask, det.mask);

        results.push_back(std::move(det));
    }

    return results;
}

// ════════════════════════════════════════════════════════════════════════════
// drawSegDetections — filled semi-transparent overlay + contour + label
// ════════════════════════════════════════════════════════════════════════════
inline void drawSegDetections(cv::Mat&                        frame,
                              const std::vector<SegDetection>& dets,
                              double fps   = -1.0,
                              float  alpha = 0.4f)
{
    // Draw all filled mask overlays on a clone, then blend once — cheaper
    // than blending per-detection
    cv::Mat overlay = frame.clone();
    for (const auto& d : dets) {
        if (!d.mask.empty())
            overlay.setTo(classColor(d.class_id), d.mask);
    }
    cv::addWeighted(overlay, alpha, frame, 1.0f - alpha, 0, frame);

    // Contour outline + bounding box + label on top of the blended frame
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

    // FPS counter top-left
    if (fps > 0) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "FPS: %.1f", fps);
        cv::putText(frame, buf, {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }
}
