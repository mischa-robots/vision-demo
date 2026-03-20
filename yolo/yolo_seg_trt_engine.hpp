// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Mischa Schaefer <https://github.com/mischa-robots/vision-demo>

/**
 * yolo_seg_trt_engine.hpp
 * ────────────────────────
 * TensorRT 8.2 engine wrapper for YOLO segmentation models.
 * Handles two output bindings:
 *   output0: (1, 116, 8400) — boxes + class scores + mask coefficients
 *   output1: (1, 32, 160, 160) — prototype masks
 */

#pragma once

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "yolo_seg_postprocess.hpp"   // for size constants

// ── Suppress TRT warnings below WARNING level ─────────────────────────────────
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << "\n";
    }
};

// ── Die loudly on any CUDA error ──────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess)                                                  \
            throw std::runtime_error(std::string("[CUDA] ") +                  \
                cudaGetErrorString(_e) + " at " __FILE__ +                     \
                ":" + std::to_string(__LINE__));                                \
    } while (0)

// ════════════════════════════════════════════════════════════════════════════
// TRTSegEngine
// ════════════════════════════════════════════════════════════════════════════
class TRTSegEngine {
public:
    static constexpr int INPUT_C    = 3;
    static constexpr int INPUT_H    = 640;
    static constexpr int INPUT_W    = 640;
    static constexpr int INPUT_SIZE = INPUT_C * INPUT_H * INPUT_W; // 1,228,800

    // Output sizes come from yolo_seg_postprocess.hpp constants
    static constexpr int OUT0_SIZE  = OUTPUT0_SIZE;   // 116 × 8400 = 974,400
    static constexpr int OUT1_SIZE  = OUTPUT1_SIZE;   //  32 × 160 × 160 = 819,200

    explicit TRTSegEngine(const std::string& engine_path) {
        std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
        if (!file)
            throw std::runtime_error("Cannot open engine: " + engine_path);
        size_t size = file.tellg();
        file.seekg(0);
        std::vector<char> buf(size);
        file.read(buf.data(), size);

        _runtime = nvinfer1::createInferRuntime(_logger);
        _engine  = _runtime->deserializeCudaEngine(buf.data(), size);
        if (!_engine)
            throw std::runtime_error("Failed to deserialise engine: " + engine_path);
        _context = _engine->createExecutionContext();

        // Pinned host buffers — faster transfers on Tegra unified memory
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&_h_input),
                                  INPUT_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&_h_out0),
                                  OUT0_SIZE  * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&_h_out1),
                                  OUT1_SIZE  * sizeof(float)));

        // GPU buffers
        CUDA_CHECK(cudaMalloc(&_d_input, INPUT_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&_d_out0,  OUT0_SIZE  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&_d_out1,  OUT1_SIZE  * sizeof(float)));

        CUDA_CHECK(cudaStreamCreate(&_stream));

        std::cout << "[TRT] Seg engine loaded: " << engine_path << "\n";
    }

    ~TRTSegEngine() {
        cudaFreeHost(_h_input);
        cudaFreeHost(_h_out0);
        cudaFreeHost(_h_out1);
        cudaFree(_d_input);
        cudaFree(_d_out0);
        cudaFree(_d_out1);
        cudaStreamDestroy(_stream);
        delete _context;
        delete _engine;
        delete _runtime;
    }

    TRTSegEngine(const TRTSegEngine&)            = delete;
    TRTSegEngine& operator=(const TRTSegEngine&) = delete;

    // Run one forward pass.
    // input — float32 NCHW (INPUT_SIZE floats, values 0..1)
    // out0  — float32 (OUT0_SIZE floats) — boxes + scores + mask coeffs
    // out1  — float32 (OUT1_SIZE floats) — prototype masks
    void infer(const float* input, float* out0, float* out1) {
        std::memcpy(_h_input, input, INPUT_SIZE * sizeof(float));

        CUDA_CHECK(cudaMemcpyAsync(_d_input, _h_input,
            INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, _stream));

        // 3 bindings: input, output0, output1
        void* bindings[3] = { _d_input, _d_out0, _d_out1 };
        if (!_context->enqueueV2(bindings, _stream, nullptr))
            throw std::runtime_error("TRT enqueueV2 failed");

        CUDA_CHECK(cudaMemcpyAsync(_h_out0, _d_out0,
            OUT0_SIZE * sizeof(float), cudaMemcpyDeviceToHost, _stream));
        CUDA_CHECK(cudaMemcpyAsync(_h_out1, _d_out1,
            OUT1_SIZE * sizeof(float), cudaMemcpyDeviceToHost, _stream));
        CUDA_CHECK(cudaStreamSynchronize(_stream));

        std::memcpy(out0, _h_out0, OUT0_SIZE * sizeof(float));
        std::memcpy(out1, _h_out1, OUT1_SIZE * sizeof(float));
    }

    // Run a few dummy passes to warm up GPU clocks before real inference.
    void warmup(int runs = 5) {
        std::cout << "[TRT] Warming up (" << runs << " runs)...\n";
        std::vector<float> di(INPUT_SIZE, 0.0f);
        std::vector<float> d0(OUT0_SIZE,  0.0f);
        std::vector<float> d1(OUT1_SIZE,  0.0f);
        for (int i = 0; i < runs; ++i)
            infer(di.data(), d0.data(), d1.data());
        std::cout << "[TRT] Warmup done\n";
    }

private:
    TRTLogger                        _logger;
    nvinfer1::IRuntime*              _runtime = nullptr;
    nvinfer1::ICudaEngine*           _engine  = nullptr;
    nvinfer1::IExecutionContext*     _context = nullptr;

    float*       _h_input = nullptr;
    float*       _h_out0  = nullptr;
    float*       _h_out1  = nullptr;
    void*        _d_input = nullptr;
    void*        _d_out0  = nullptr;
    void*        _d_out1  = nullptr;
    cudaStream_t _stream  = nullptr;
};
