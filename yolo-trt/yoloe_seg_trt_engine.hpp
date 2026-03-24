// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>

/**
 * yoloe_seg_trt_engine.hpp
 * ─────────────────────────
 * TensorRT 8.2 engine wrapper for YOLOE segmentation models.
 *
 * Two output bindings:
 *   output0: (1, 4+num_classes+32, 8400) — boxes + class scores + mask coefficients
 *   output1: (1, 32, 160, 160)           — prototype masks
 *
 * num_classes is runtime-dynamic (YOLOE supports custom label sets),
 * so output0 size is computed at construction time.
 *
 * GPU optimisation — zero-copy output (Option C):
 *   Output buffers are cudaMallocHost pinned memory allocated ONCE in the
 *   engine. TRT writes directly into them (DtoH copy lands in pinned memory).
 *   infer_direct() gives those pointers to the caller — no staging copy.
 *   On Tegra's unified memory architecture this saves one full memcpy
 *   per frame for both output tensors.
 */

#pragma once

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

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
// TRTYoloeSegEngine
// ════════════════════════════════════════════════════════════════════════════
class TRTYoloeSegEngine {
public:
    static constexpr int INPUT_C    = 3;
    static constexpr int INPUT_H    = 640;
    static constexpr int INPUT_W    = 640;
    static constexpr int INPUT_SIZE = INPUT_C * INPUT_H * INPUT_W; // 1,228,800

    static constexpr int PROTO_MASKS   = 32;
    static constexpr int PROTO_H       = 160;
    static constexpr int PROTO_W       = 160;
    static constexpr int OUT1_SIZE     = PROTO_MASKS * PROTO_H * PROTO_W; // 819,200

    // output0 size depends on num_classes — (4 + num_classes + 32) × 8400
    const int OUT0_SIZE;

    // engine_path  — path to .trt file
    // num_classes  — must match the model (loaded from labels file before calling this)
    explicit TRTYoloeSegEngine(const std::string& engine_path, int num_classes)
        : OUT0_SIZE((4 + num_classes + PROTO_MASKS) * 8400)
    {
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

        // Pinned host input buffer
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&_h_input),
                                  INPUT_SIZE * sizeof(float)));

        // GPU input buffer
        CUDA_CHECK(cudaMalloc(&_d_input, INPUT_SIZE * sizeof(float)));

        // Zero-copy output: pinned host buffers that TRT writes into directly
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&_h_out0),
                                  OUT0_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&_h_out1),
                                  OUT1_SIZE * sizeof(float)));

        // GPU output buffers — TRT writes here, then async DtoH into pinned above
        CUDA_CHECK(cudaMalloc(&_d_out0, OUT0_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&_d_out1, OUT1_SIZE * sizeof(float)));

        CUDA_CHECK(cudaStreamCreate(&_stream));

        std::cout << "[TRT] YOLOE-seg engine loaded: " << engine_path
                  << "  num_classes=" << num_classes
                  << "  out0=" << OUT0_SIZE * 4 / 1024 << "KB"
                  << "  out1=" << OUT1_SIZE * 4 / 1024 << "KB\n";
    }

    ~TRTYoloeSegEngine() {
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

    TRTYoloeSegEngine(const TRTYoloeSegEngine&)            = delete;
    TRTYoloeSegEngine& operator=(const TRTYoloeSegEngine&) = delete;

    // Zero-copy inference — returns pinned pointers valid until next infer_direct() call.
    // Caller must NOT free or hold these pointers across calls.
    // out0 — (4+num_classes+32) × 8400 floats
    // out1 — 32 × 160 × 160 floats
    void infer_direct(const float* input, const float*& out0, const float*& out1) {
        std::memcpy(_h_input, input, INPUT_SIZE * sizeof(float));

        CUDA_CHECK(cudaMemcpyAsync(_d_input, _h_input,
            INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, _stream));

        void* bindings[3] = { _d_input, _d_out0, _d_out1 };
        if (!_context->enqueueV2(bindings, _stream, nullptr))
            throw std::runtime_error("TRT enqueueV2 failed");

        // DtoH directly into pinned memory — caller reads from _h_out0/1
        CUDA_CHECK(cudaMemcpyAsync(_h_out0, _d_out0,
            OUT0_SIZE * sizeof(float), cudaMemcpyDeviceToHost, _stream));
        CUDA_CHECK(cudaMemcpyAsync(_h_out1, _d_out1,
            OUT1_SIZE * sizeof(float), cudaMemcpyDeviceToHost, _stream));
        CUDA_CHECK(cudaStreamSynchronize(_stream));

        out0 = _h_out0;
        out1 = _h_out1;
    }

    // Run dummy passes to warm up GPU frequency before real inference.
    // Uses a temporary stack-allocated input — no per-run heap allocation.
    void warmup(int runs = 5) {
        std::cout << "[TRT] Warming up (" << runs << " runs)...\n";
        std::vector<float> dummy(INPUT_SIZE, 0.0f);
        const float* o0;
        const float* o1;
        for (int i = 0; i < runs; ++i)
            infer_direct(dummy.data(), o0, o1);
        std::cout << "[TRT] Warmup done\n";
    }

private:
    TRTLogger                        _logger;
    nvinfer1::IRuntime*              _runtime = nullptr;
    nvinfer1::ICudaEngine*           _engine  = nullptr;
    nvinfer1::IExecutionContext*     _context = nullptr;

    float*       _h_input = nullptr;  // pinned host input
    float*       _h_out0  = nullptr;  // pinned host output0 (zero-copy target)
    float*       _h_out1  = nullptr;  // pinned host output1 (zero-copy target)
    void*        _d_input = nullptr;  // GPU input
    void*        _d_out0  = nullptr;  // GPU output0
    void*        _d_out1  = nullptr;  // GPU output1
    cudaStream_t _stream  = nullptr;
};
