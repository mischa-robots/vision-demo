// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>

/**
 * yoloe_seg_dnn_engine.hpp
 * ─────────────────────────
 * OpenCV DNN engine wrapper for YOLOE segmentation models (ONNX).
 *
 * num_classes is runtime-dynamic — output0 size is computed at construction,
 * identical to the TRT version.
 *
 * Two outputs:
 *   output0: (1, 4+num_classes+32, 8400) — boxes + class scores + mask coefficients
 *   output1: (1, 32, 160, 160)           — prototype masks
 */

#pragma once

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "yoloe_seg_postprocess.hpp"   // for YOLOE_* constants and yoloe_out0_size()

// ════════════════════════════════════════════════════════════════════════════
// DNNYoloeSegEngine
// ════════════════════════════════════════════════════════════════════════════
class DNNYoloeSegEngine {
public:
    static constexpr int INPUT_C    = 3;
    static constexpr int INPUT_H    = 640;
    static constexpr int INPUT_W    = 640;
    static constexpr int INPUT_SIZE = INPUT_C * INPUT_H * INPUT_W; // 1,228,800

    static constexpr int OUT1_SIZE  = YOLOE_OUT1_SIZE; // 32 × 160 × 160 = 819,200

    // out0 size depends on num_classes — (4 + num_classes + 32) × 8400
    const int OUT0_SIZE;

    // model_path  — path to .onnx file
    // num_classes — must match the model (loaded from labels file before calling this)
    explicit DNNYoloeSegEngine(const std::string& model_path, int num_classes)
        : OUT0_SIZE(yoloe_out0_size(num_classes))
    {
        _net = cv::dnn::readNetFromONNX(model_path);
        if (_net.empty())
            throw std::runtime_error("Failed to load ONNX model: " + model_path);

        _net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        _net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

        _net.enableFusion(false);

        _out_names = { "output0", "output1" };

        std::cout << "[DNN] YOLOE-seg model loaded: " << model_path
                  << "  num_classes=" << num_classes
                  << "  out0=" << OUT0_SIZE * 4 / 1024 << "KB"
                  << "  out1=" << OUT1_SIZE * 4 / 1024 << "KB\n";
    }

    DNNYoloeSegEngine(const DNNYoloeSegEngine&)            = delete;
    DNNYoloeSegEngine& operator=(const DNNYoloeSegEngine&) = delete;

    // Run one forward pass.
    // input — float32 NCHW array, INPUT_SIZE floats, values 0..1
    // out0  — float32 array, OUT0_SIZE floats
    // out1  — float32 array, OUT1_SIZE floats
    void infer(const float* input, float* out0, float* out1) {
        cv::Mat blob(1, INPUT_SIZE, CV_32F, const_cast<float*>(input));
        blob = blob.reshape(1, {1, INPUT_C, INPUT_H, INPUT_W});

        _net.setInput(blob);

        std::vector<cv::Mat> outs;
        _net.forward(outs, _out_names);

        // outs[0]: (1, 4+num_classes+32, 8400) → flatten
        cv::Mat o0 = outs[0].reshape(1, 1);
        std::memcpy(out0, o0.ptr<float>(), OUT0_SIZE * sizeof(float));

        // outs[1]: (1, 32, 160, 160) → flatten
        cv::Mat o1 = outs[1].reshape(1, 1);
        std::memcpy(out1, o1.ptr<float>(), OUT1_SIZE * sizeof(float));
    }

    void warmup(int runs = 5) {
        std::cout << "[DNN] Warming up (" << runs << " runs)...\n";
        std::vector<float> di(INPUT_SIZE, 0.0f);
        std::vector<float> d0(OUT0_SIZE,  0.0f);
        std::vector<float> d1(OUT1_SIZE,  0.0f);
        for (int i = 0; i < runs; ++i)
            infer(di.data(), d0.data(), d1.data());
        std::cout << "[DNN] Warmup done\n";
    }

private:
    cv::dnn::Net             _net;
    std::vector<std::string> _out_names;
};
