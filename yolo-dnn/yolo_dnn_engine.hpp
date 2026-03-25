// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>

/**
 * yolo_dnn_engine.hpp
 * ────────────────────
 * OpenCV DNN engine wrapper for YOLO detection models (ONNX).
 *
 * Output shape: (1, 84, 8400)
 *   84 = 4 box coords + 80 class scores
 *   8400 = number of anchor candidates
 *
 * OpenCV DNN returns output as (1, 84, 8400) — we strip the batch dim and
 * hand the caller a flat (84 * 8400) float array, identical to TRT layout.
 */

#pragma once

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

// ════════════════════════════════════════════════════════════════════════════
// DNNEngine
// ════════════════════════════════════════════════════════════════════════════
class DNNEngine {
public:
    // Fixed input/output sizes — identical to TRTEngine constants so
    // yolo_postprocess.hpp and the .cpp files need no changes.
    static constexpr int INPUT_C    = 3;
    static constexpr int INPUT_H    = 640;
    static constexpr int INPUT_W    = 640;
    static constexpr int INPUT_SIZE = INPUT_C * INPUT_H * INPUT_W; // 1,228,800

    static constexpr int OUTPUT_ROWS = 84;    // 4 + 80 classes
    static constexpr int OUTPUT_COLS = 8400;  // anchor candidates
    static constexpr int OUTPUT_SIZE = OUTPUT_ROWS * OUTPUT_COLS; // 705,600

    explicit DNNEngine(const std::string& model_path) {
        _net = cv::dnn::readNetFromONNX(model_path);
        if (_net.empty())
            throw std::runtime_error("Failed to load ONNX model: " + model_path);

        // Use OpenCV CUDA backend if available — falls back to CPU silently
        _net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        _net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

        _net.enableFusion(false);

        std::cout << "[DNN] Model loaded: " << model_path << "\n";
    }

    DNNEngine(const DNNEngine&)            = delete;
    DNNEngine& operator=(const DNNEngine&) = delete;

    // Run one forward pass.
    // input  — float32 NCHW array, INPUT_SIZE floats, values 0..1
    //          (same format produced by the preprocess() function in the .cpp)
    // output — float32 array, OUTPUT_SIZE floats, filled on return
    //          layout: (OUTPUT_ROWS, OUTPUT_COLS) row-major — identical to TRT
    void infer(const float* input, float* output) {
        // Wrap the pre-normalised NCHW float buffer as a cv::Mat blob
        // (no copy — cv::Mat header over existing memory)
        cv::Mat blob(1, INPUT_SIZE, CV_32F, const_cast<float*>(input));
        blob = blob.reshape(1, {1, INPUT_C, INPUT_H, INPUT_W});

        _net.setInput(blob);

        // forward() returns (1, OUTPUT_ROWS, OUTPUT_COLS)
        cv::Mat out = _net.forward();

        // Strip batch dim → (OUTPUT_ROWS, OUTPUT_COLS), copy to caller buffer
        cv::Mat out2d = out.reshape(1, OUTPUT_ROWS);
        std::memcpy(output, out2d.ptr<float>(), OUTPUT_SIZE * sizeof(float));
    }

    // No-op warmup — kept for API compatibility with TRTEngine.
    // OpenCV DNN does its own lazy init on the first forward() call.
    void warmup(int runs = 5) {
        std::cout << "[DNN] Warming up (" << runs << " runs)...\n";
        std::vector<float> dummy_in(INPUT_SIZE,   0.0f);
        std::vector<float> dummy_out(OUTPUT_SIZE, 0.0f);
        for (int i = 0; i < runs; ++i)
            infer(dummy_in.data(), dummy_out.data());
        std::cout << "[DNN] Warmup done\n";
    }

private:
    cv::dnn::Net _net;
};
