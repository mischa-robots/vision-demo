// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Mischa Schaefer <https://github.com/mischa-robots/vision-demo>

/**
 * trt_infer.cpp
 * ─────────────
 * Minimal TensorRT 8.2 inference wrapper exposed as a C shared library.
 * Called from Python via ctypes — no Python TRT bindings needed.
 *
 * Build:
 *   g++ -O2 -shared -fPIC -o trt_infer.so trt_infer.cpp \
 *       -I/usr/include/aarch64-linux-gnu \
 *       -lnvinfer -lnvonnxparser -lcudart \
 *       -L/usr/lib/aarch64-linux-gnu
 *
 * What this exposes to Python (all via ctypes):
 *   trt_create()          — load engine, allocate GPU buffers
 *   trt_infer()           — run one forward pass
 *   trt_destroy()         — free everything
 */

#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

// ── Tiny TRT logger (warnings + errors only, goes to stderr) ────────────────
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << "\n";
    }
} gLogger;

// ── Internal engine state (one per loaded engine) ───────────────────────────
struct TRTEngine {
    nvinfer1::IRuntime*            runtime  = nullptr;
    nvinfer1::ICudaEngine*         engine   = nullptr;
    nvinfer1::IExecutionContext*   context  = nullptr;

    void*  d_input   = nullptr;   // GPU buffer — input
    void*  d_output  = nullptr;   // GPU buffer — output
    float* h_input   = nullptr;   // host pinned — input
    float* h_output  = nullptr;   // host pinned — output

    int input_size   = 0;         // number of floats
    int output_size  = 0;         // number of floats

    cudaStream_t stream = nullptr;
};

// ════════════════════════════════════════════════════════════════════════════
// C API (extern "C" so Python ctypes can find the symbols easily)
// ════════════════════════════════════════════════════════════════════════════
extern "C" {

/**
 * trt_create — deserialise engine file, allocate buffers.
 *
 * @param engine_path  path to .trt file
 * @param input_size   total floats in input  tensor (1*3*640*640 = 1228800)
 * @param output_size  total floats in output tensor (1*84*8400  =  705600)
 * @return opaque handle (pointer cast to long long), or 0 on failure
 */
long long trt_create(const char* engine_path,
                     int         input_size,
                     int         output_size)
{
    // Read engine file into buffer
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "[TRT] Cannot open engine: " << engine_path << "\n";
        return 0;
    }
    size_t size = file.tellg();
    file.seekg(0);
    std::vector<char> buf(size);
    file.read(buf.data(), size);
    file.close();

    auto* e = new TRTEngine();

    e->runtime = nvinfer1::createInferRuntime(gLogger);
    e->engine  = e->runtime->deserializeCudaEngine(buf.data(), size);
    if (!e->engine) {
        std::cerr << "[TRT] Failed to deserialise engine\n";
        delete e; return 0;
    }
    e->context = e->engine->createExecutionContext();

    e->input_size  = input_size;
    e->output_size = output_size;

    // Allocate pinned host memory (faster H2D/D2H transfers on Nano)
    cudaMallocHost(reinterpret_cast<void**>(&e->h_input),  input_size  * sizeof(float));
    cudaMallocHost(reinterpret_cast<void**>(&e->h_output), output_size * sizeof(float));

    // Allocate GPU memory
    cudaMalloc(&e->d_input,  input_size  * sizeof(float));
    cudaMalloc(&e->d_output, output_size * sizeof(float));

    cudaStreamCreate(&e->stream);

    std::cout << "[TRT] Engine ready — input=" << input_size
              << " output=" << output_size << "\n";

    return reinterpret_cast<long long>(e);
}

/**
 * trt_infer — run one inference pass.
 *
 * @param handle       value returned by trt_create
 * @param input_data   pointer to (input_size) floats — NCHW, 0..1
 * @param output_data  pointer to (output_size) floats — filled by this call
 * @return 0 on success, -1 on error
 */
int trt_infer(long long handle,
              const float* input_data,
              float*       output_data)
{
    if (!handle) return -1;
    auto* e = reinterpret_cast<TRTEngine*>(handle);

    // Copy input to pinned buffer then to GPU
    memcpy(e->h_input, input_data, e->input_size * sizeof(float));
    cudaMemcpyAsync(e->d_input, e->h_input,
                    e->input_size * sizeof(float),
                    cudaMemcpyHostToDevice, e->stream);

    // Run inference
    void* bindings[2] = { e->d_input, e->d_output };
    bool ok = e->context->enqueueV2(bindings, e->stream, nullptr);
    if (!ok) {
        std::cerr << "[TRT] enqueueV2 failed\n";
        return -1;
    }

    // Copy output back to pinned buffer then to caller
    cudaMemcpyAsync(e->h_output, e->d_output,
                    e->output_size * sizeof(float),
                    cudaMemcpyDeviceToHost, e->stream);
    cudaStreamSynchronize(e->stream);

    memcpy(output_data, e->h_output, e->output_size * sizeof(float));
    return 0;
}

/**
 * trt_destroy — release all resources.
 */
void trt_destroy(long long handle)
{
    if (!handle) return;
    auto* e = reinterpret_cast<TRTEngine*>(handle);

    cudaFreeHost(e->h_input);
    cudaFreeHost(e->h_output);
    cudaFree(e->d_input);
    cudaFree(e->d_output);
    cudaStreamDestroy(e->stream);

    delete e->context;
    delete e->engine;
    delete e->runtime;

    delete e;
    std::cout << "[TRT] Engine destroyed\n";
}

} // extern "C"
