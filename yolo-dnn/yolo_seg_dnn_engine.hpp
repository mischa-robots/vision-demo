// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>

/**
 * yolo_seg_dnn_engine.hpp
 * ────────────────────────
 * OpenCV DNN engine wrapper for YOLO segmentation models (ONNX).
 *
 * Two outputs:
 *   output0: (1, 116, 8400) — boxes + class scores + mask coefficients
 *   output1: (1, 32, 160, 160) — prototype masks
 *
 * Both are returned as flat float arrays in the same row-major layout as TRT.
 */

#pragma once

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "yolo_seg_postprocess.hpp"   // for OUTPUT0_SIZE, OUTPUT1_SIZE constants

// ════════════════════════════════════════════════════════════════════════════
// DNNSegEngine
// ════════════════════════════════════════════════════════════════════════════
class DNNSegEngine {
public:
    static constexpr int INPUT_C    = 3;
    static constexpr int INPUT_H    = 640;
    static constexpr int INPUT_W    = 640;
    static constexpr int INPUT_SIZE = INPUT_C * INPUT_H * INPUT_W; // 1,228,800

    // Output sizes from yolo_seg_postprocess.hpp — identical to TRTSegEngine
    static constexpr int OUT0_SIZE  = OUTPUT0_SIZE;   // 116 × 8400 = 974,400
    static constexpr int OUT1_SIZE  = OUTPUT1_SIZE;   //  32 × 160 × 160 = 819,200

    explicit DNNSegEngine(const std::string& model_path) {
        _net = cv::dnn::readNetFromONNX(model_path);
        if (_net.empty())
            throw std::runtime_error("Failed to load ONNX model: " + model_path);

        _net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        _net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

        _net.enableFusion(false);

        // Cache output layer names — YOLO-seg ONNX exports as "output0" / "output1"
        _out_names = { "output0", "output1" };

        std::cout << "[DNN] Seg model loaded: " << model_path << "\n";
    }

    DNNSegEngine(const DNNSegEngine&)            = delete;
    DNNSegEngine& operator=(const DNNSegEngine&) = delete;

    // Run one forward pass.
    // input — float32 NCHW array, INPUT_SIZE floats, values 0..1
    // out0  — float32 array, OUT0_SIZE floats — boxes + scores + mask coeffs
    // out1  — float32 array, OUT1_SIZE floats — prototype masks
    void infer(const float* input, float* out0, float* out1) {
        cv::Mat blob(1, INPUT_SIZE, CV_32F, const_cast<float*>(input));
        blob = blob.reshape(1, {1, INPUT_C, INPUT_H, INPUT_W});

        _net.setInput(blob);

        std::vector<cv::Mat> outs;
        _net.forward(outs, _out_names);

        // outs[0]: (1, 116, 8400) → flatten to (116 * 8400)
        cv::Mat o0 = outs[0].reshape(1, SEG_ROWS);
        std::memcpy(out0, o0.ptr<float>(), OUT0_SIZE * sizeof(float));

        // outs[1]: (1, 32, 160, 160) → flatten to (32 * 160 * 160)
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
