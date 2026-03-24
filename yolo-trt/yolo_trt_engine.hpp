// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>

/**
 * yolo_trt_engine.hpp
 * ────────────────────
 * TensorRT 8.2 engine wrapper for YOLO detection models.
 * Loads a .trt engine file, manages GPU buffers, and runs inference.
 *
 * Output shape: (1, 84, 8400)
 *   84 = 4 box coords + 80 class scores
 *   8400 = number of anchor candidates
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
// TRTEngine
// ════════════════════════════════════════════════════════════════════════════
class TRTEngine {
public:
    // Fixed input/output sizes for 640×640 YOLO detection
    static constexpr int INPUT_C    = 3;
    static constexpr int INPUT_H    = 640;
    static constexpr int INPUT_W    = 640;
    static constexpr int INPUT_SIZE = INPUT_C * INPUT_H * INPUT_W; // 1,228,800

    static constexpr int OUTPUT_ROWS = 84;    // 4 + 80 classes
    static constexpr int OUTPUT_COLS = 8400;  // anchor candidates
    static constexpr int OUTPUT_SIZE = OUTPUT_ROWS * OUTPUT_COLS; // 705,600

    explicit TRTEngine(const std::string& engine_path) {
        // Load serialised engine from disk
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

        // Pinned host memory — faster transfers on Tegra unified memory
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&_h_input),
                                  INPUT_SIZE  * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&_h_output),
                                  OUTPUT_SIZE * sizeof(float)));

        // GPU buffers
        CUDA_CHECK(cudaMalloc(&_d_input,  INPUT_SIZE  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&_d_output, OUTPUT_SIZE * sizeof(float)));

        CUDA_CHECK(cudaStreamCreate(&_stream));

        std::cout << "[TRT] Engine loaded: " << engine_path << "\n";
    }

    ~TRTEngine() {
        cudaFreeHost(_h_input);
        cudaFreeHost(_h_output);
        cudaFree(_d_input);
        cudaFree(_d_output);
        cudaStreamDestroy(_stream);
        delete _context;
        delete _engine;
        delete _runtime;
    }

    // Not copyable
    TRTEngine(const TRTEngine&)            = delete;
    TRTEngine& operator=(const TRTEngine&) = delete;

    // Run one forward pass.
    // input  — float32 NCHW array, INPUT_SIZE floats, values 0..1
    // output — float32 array, OUTPUT_SIZE floats, filled on return
    void infer(const float* input, float* output) {
        std::memcpy(_h_input, input, INPUT_SIZE * sizeof(float));

        CUDA_CHECK(cudaMemcpyAsync(_d_input, _h_input,
            INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, _stream));

        void* bindings[2] = { _d_input, _d_output };
        if (!_context->enqueueV2(bindings, _stream, nullptr))
            throw std::runtime_error("TRT enqueueV2 failed");

        CUDA_CHECK(cudaMemcpyAsync(_h_output, _d_output,
            OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, _stream));
        CUDA_CHECK(cudaStreamSynchronize(_stream));

        std::memcpy(output, _h_output, OUTPUT_SIZE * sizeof(float));
    }

    // Run a few dummy passes to warm up GPU clocks before real inference.
    void warmup(int runs = 5) {
        std::cout << "[TRT] Warming up (" << runs << " runs)...\n";
        std::vector<float> dummy_in(INPUT_SIZE,   0.0f);
        std::vector<float> dummy_out(OUTPUT_SIZE, 0.0f);
        for (int i = 0; i < runs; ++i)
            infer(dummy_in.data(), dummy_out.data());
        std::cout << "[TRT] Warmup done\n";
    }

private:
    TRTLogger                        _logger;
    nvinfer1::IRuntime*              _runtime = nullptr;
    nvinfer1::ICudaEngine*           _engine  = nullptr;
    nvinfer1::IExecutionContext*     _context = nullptr;

    float*       _h_input  = nullptr;  // pinned host input
    float*       _h_output = nullptr;  // pinned host output
    void*        _d_input  = nullptr;  // GPU input
    void*        _d_output = nullptr;  // GPU output
    cudaStream_t _stream   = nullptr;
};
