# Vision Demo

Computer Vision demos for the **Jetson Nano** — stereo depth perception,
optical flow, and YOLO object detection and segmentation — streaming results live
to a browser via MediaMTX. No desktop required, runs fully headless.

All demos share a single build step and a single MediaMTX instance. Results
are accessible from any browser on the same network.

The purpose of this repository is not to achieve best performance but to find out the possibilities and limitations 
of the Jetson Nano for Computer Vision tasks.

> **Hardware note:** All demos target the Jetson Nano B01 with JetPack 4.6.6
> (CUDA 10.2, Maxwell GPU, 128 CUDA cores). Some OpenCV CUDA algorithms
> require Pascal or newer and are documented as non-functional on this hardware
> — they are kept in the repo for reference and for users on newer boards (i.e. Orin Nano).

---

## Demos

*HINT:* run only one binary or python script at a time, or expect camera capture errors.

### Stereo Depth

Two synchronised CSI cameras → stereo rectification → disparity map →
depth colourmap → streamed to browser.

| Binary | Algorithm | Nano FPS | Notes |
|---|---|---|---|
| `vpi_stereo_depth` | VPI SGM | x fps | Recommended. Hardware-optimised SGM. |
| `cvcuda_stereo_bm` | OpenCV CUDA StereoBM | x fps | Fastest, lower quality. |
| `cvcuda_stereo_sgm` | OpenCV CUDA StereoSGM | ❌ | Requires Pascal+ GPU. Empty output on Nano. |
| `cvcuda_stereo_csbp` | OpenCV CUDA CSBP | ~0.8 fps | Good quality, not real-time. |
| `cvcuda_stereo_bp` | OpenCV CUDA full BP | ❌ | Too slow even at minimum settings. |
| `cv_stereo_sgbm.py` | CPU SGBM + WLS filter | ~2-5 fps | Best quality. Browser UI for parameter tuning. |

Video Stream: `http://<jetson-ip>:8889/stereo_depth`

See [stereo-depth/README.md](stereo-depth/README.md) for details.

### Optical Flow

Single CSI camera → corner detection or dense flow → motion visualisation →
streamed to browser. All demos stream to the same path so you can switch
algorithms without changing the browser URL.

| Binary | Algorithm | Type | Nano FPS | Notes |
|---|---|---|---|---|
| `vpi_optical_flow` | VPI PyrLK | Sparse | ~15 fps | ⚠️ Known NVIDIA bug — points not updated after frame 1. Demo only. |
| `cvcuda_sparse_pyrlk` | SparsePyrLK + FB filter | Sparse | ~14 fps | Recommended baseline for ego-motion. |
| `cvcuda_farneback` | Farneback | Dense | ~3 fps | Best dense option on Nano. |
| `cvcuda_dense_pyrlk` | DensePyrLK | Dense | x fps | Use `--width 640 --height 360` to avoid GPU timeout. |
| `cvcuda_brox` | Brox variational | Dense | x fps | Very slow — not viable for real-time. |
| `cvcuda_dual_tvl1` | TV-L1 variational | Dense | ~2 fps | Use `--iterations 20 --scales 2 --warps 1`. |
| `cv_optical_flow.py` | CUDA LK + CPU Farneback | Sparse+Dense | ~2 fps | Python reference. |
| `cv_optical_flow_cuda.py` | CUDA LK + CUDA Farneback | Sparse+Dense | ~2 fps | Python reference, optimised. |

Video Stream: `http://<jetson-ip>:8889/optical_flow`

See [optical-flow/README.md](optical-flow/README.md) for details.

### YOLO DNN

Single CSI or USB camera → OpenCV DNN inference (ONNX) → detection/segmentation overlay → streamed to browser. No TensorRT conversion required.

| Binary | Description | Nano FPS | Notes |
|---|---|---|---|
| `yolodnn_det` | Object detection with any YOLO ONNX model | - | - |
| `yolodnn_seg` | Instance segmentation with YOLO-seg ONNX model | - | - |
| `yoloednn_seg` | Instance segmentation with YOLOE-seg ONNX model | - | - |

Video Stream: `http://<jetson-ip>:8889/yolo`

See [yolo-dnn/README.md](yolo-dnn/README.md) for details.

### YOLO TRT

Single CSI or USB camera → TensorRT inference → detection/segmentation overlay → streamed to browser.

| Binary | Description | Nano FPS | Notes |
|---|---|---|---|
| `yolotrt_det` | Object detection with any YOLO TensorRT engine | - | - |
| `yolotrt_seg` | Instance segmentation with YOLO-seg TensorRT engine | - | - |
| `yoloetrt_seg` | Instance segmentation with YOLOE-seg TensorRT engine | - | - |

Video Stream: `http://<jetson-ip>:8889/yolo`

See [yolo-trt/README.md](yolo-trt/README.md) for details.


### Performance

The measured performance (fps) of all demos is lower than what is actually possible on the Jetson Nano 
due to simultaneous rendering of the results into the video stream.

---

## Hardware & Software Requirements

**Hardware**
- Jetson Nano B01 (4 GB recommended)
- Two CSI cameras (e.g. IMX219) for stereo depth demos
- Single CSI or USB camera for optical flow and YOLO demos
- USB stick or fast SD card for TensorRT model files

**Software**
- L4T Ubuntu 22.04 (headless)
- JetPack 4.6.6
- CUDA 10.2
- TensorRT 8.2
- OpenCV 4.8.1 (built from source with CUDA + GStreamer + contrib)
- VPI 1.2.x
- GStreamer 1.20
- CMake 3.12+
- Python 3.10 (for Python demos and stereo calibration)

OpenCV must be built from source with `-DWITH_CUDA=ON`, `-DWITH_GSTREAMER=ON`,
and the `cudaoptflow`, `cudafeatures2d`, `cudastereo`, and `cudawarping` contrib
modules enabled. The system OpenCV package is not sufficient.

*HINT:* You can also use my [Jetson Nano Ubuntu 22.04](https://github.com/mischa-robots/jetson-nano-ubuntu22) image that comes with OpenCV + CUDA precompiled.

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/mischa-robots/vision-demo.git
cd vision-demo
```

### 2. One-time setup — lock GPU/CPU clocks

```bash
sudo jetson_clocks
```

This locks all clocks to their maximum frequency. Without it the GPU starts
at 76 MHz and performance is unpredictable.

Run once after every reboot.

### 3. Build

```bash
make
```

Binaries are placed in `./build/`. CMake configure runs automatically on the
first call; subsequent calls only rebuild changed files.

For a clean rebuild:

```bash
make rebuild
```

### 4. Start MediaMTX

> **Important:** edit `mediamtx/mediamtx_demo.yml` and set
> `webrtcAdditionalHosts` to your Jetson Nano's actual LAN IP (find out with `ip a`).
> Without this, WebRTC ICE negotiation falls back to a slow peer-reflexive path and the
> stream takes ~20 seconds to appear in the browser.

Then start mediamtx with the demo config:

```bash
cd mediamtx
./mediamtx mediamtx_demo.yml
```

*IMPORTANT:* This repository does not contain the mediamtx binary, see [mediamtx/README.md](mediamtx/README.md) how to download it.

### 5. Run a demo

**Stereo depth:**
```bash
./build/vpi_stereo_depth --calib ./stereo-depth/stereo_calibration.yaml
```

See [stereo-depth/README.md](stereo-depth/README.md) for all Stereo Depth demos.

**Optical flow:**
```bash
./build/cvcuda_sparse_pyrlk
```

See [optical-flow/README.md](optical-flow/README.md) for all Optical Flow demos.

**YOLO DNN detection:**
```bash
./build/yolodnn_det --model ./models/yolo.onnx --labels ./models/yolo-labels.txt
```

See [yolo-dnn/README.md](yolo-dnn/README.md) for all YOLO DNN demos.

**YOLO TRT detection:**
```bash
./build/yolotrt_det --model ./models/yolo.trt --labels ./models/yolo-labels.txt
```

See [yolo-trt/README.md](yolo-trt/README.md) for all YOLO TRT demos.

**YOLO models**

*IMPORTANT:* This repository does not contain the yolo model files (`.onnx` , `.trt`) !

See [models/README.md](models/README.md) for model export and conversion instructions.

**HELP**

Run any binary with `--help` to see all available options.

### 6. Watch in browser

```
http://<jetson-ip>:8889/stereo_depth
http://<jetson-ip>:8889/optical_flow
http://<jetson-ip>:8889/yolo
```

All streams are served simultaneously by the same MediaMTX instance — no
restart needed when switching between demos.

Mediamtx serves WebRTC by default, RTSP and HLS can be simply activated in the `mediamtx_demo.yml` file.

---

## Streaming Architecture

```
CSI cameras
  │
  ├─► stereo-depth/ ───────────────────────────┐
  │     rectification → disparity → colourmap  │
  │     vpi_stereo_depth, cvcuda_stereo_*      │
  │                                            │
  ├─► optical-flow/ ───────────────────────────┤
  │     corner detection → LK / dense flow     │
  │     cvcuda_sparse_pyrlk, cvcuda_farneback  │
  │     cvcuda_dense_pyrlk, cvcuda_brox        │
  │     cvcuda_dual_tvl1, vpi_optical_flow     │
  │                                            │
  ├─► yolo-dnn/ ───────────────────────────────┤
  │     OpenCV DNN inference → overlay         │
  │     yolodnn_det, yolodnn_seg, yoloednn_seg │
  │                                            │
  ├─► yolo-trt/ ───────────────────────────────┤
  │     TensorRT inference → overlay           │
  │     yolotrt_det, yolotrt_seg, yoloetrt_seg │
  │                                            │
  └────────────────────────────────────────────┘
                                               │
                                            RTP/UDP
                                               │
                                               ▼
                                            MediaMTX
                                               │
                         ┌─────────────────────│─────────────────────┐
                         │                     │                     │
                         ▼                     ▼                     ▼
                      WebRTC                  HLS                   RTSP
                  http://<ip>:8889/...   http://<ip>:8888/...    rtsp://<ip>:8554
                         │                     │                     │
                         ▼                     ▼                     ▼
                     Any browser on the same network          VLC, ffplay, gstreamer
```

Each binary encodes the rendered frames as H.264 and pushes over RTP/UDP into MediaMTX.

MediaMTX re-serves all streams as WebRTC (and optionally as HLS or RTSP) simultaneously without re-encoding.

---

## Repository Structure

```
vision-demo/
├── CMakeLists.txt                  # Single build for all C++ targets
├── Makefile                        # Thin wrapper: make / make clean / make rebuild
├── .gitignore
│
├── stereo-depth/                   # Stereo depth demos
│   ├── vpi_stereo_depth.cpp        # VPI SGM — recommended
│   ├── cvcuda_stereo_bm.cpp        # OpenCV CUDA Block Matching
│   ├── cvcuda_stereo_sgm.cpp       # OpenCV CUDA SGM (❌ Maxwell not supported)
│   ├── cvcuda_stereo_csbp.cpp      # OpenCV CUDA Constant Space BP
│   ├── cvcuda_stereo_bp.cpp        # OpenCV CUDA full BP (❌ too slow on Nano)
│   ├── cv_stereo_sgbm.py           # CPU SGBM + WLS — best quality, browser UI
│   ├── stereo_calibration.yaml     # Generated — not committed
│   ├── calibration/                # Calibration scripts
│   └── README.md
│
├── optical-flow/                   # Optical flow demos
│   ├── cvcuda_sparse_pyrlk_optical_flow.cpp
│   ├── cvcuda_farneback_optical_flow.cpp
│   ├── cvcuda_dense_pyrlk_optical_flow.cpp
│   ├── cvcuda_brox_optical_flow.cpp
│   ├── cvcuda_dual_tvl1_optical_flow.cpp
│   ├── vpi_optical_flow.cpp        # ⚠️ VPI bug — demo only
│   ├── cv_optical_flow.py
│   ├── cv_optical_flow_cuda.py
│   └── README.md
│
├── yolo-dnn/                       # YOLO with OpenCV DNN (ONNX)
│   ├── yolo_det.cpp                # binary: yolodnn_det
│   ├── yolo_seg.cpp                # binary: yolodnn_seg
│   ├── yoloe_seg.cpp               # binary: yoloednn_seg
│   ├── ...                         # engine and postprocess headers
│   └── README.md
│
├── yolo-trt/                       # YOLO with TensorRT
│   ├── yolo_det.cpp                # binary: yolotrt_det
│   ├── yolo_seg.cpp                # binary: yolotrt_seg
│   ├── yoloe_seg.cpp               # binary: yoloetrt_seg
│   ├── trt_infer.cpp               # shared lib: trt_infer.so
│   ├── ...                         # engine and postprocess headers
│   └── README.md
│
├── models/                         # ONNX + TRT model files and labels — not committed
│   └── README.md                   # Model export and conversion instructions
│
├── mediamtx/                       # MediaMTX config
│   ├── mediamtx_demo.yml
│   └── README.md
│
└── build/                          # Compiled binaries — not committed
```

---

## Known Issues & Hardware Limitations

| Issue | Affected | Workaround |
|---|---|---|
| `vpiSubmitOpticalFlowPyrLK` does not update points after frame 1 | `vpi_optical_flow` | Use `cvcuda_sparse_pyrlk` instead |
| `cv::cuda::StereoSGM` requires Pascal+ GPU (compute 6.0+) | `cvcuda_stereo_sgm` | Use `vpi_stereo_depth` or `cv_stereo_sgbm.py` |
| `cv::cuda::StereoBeliefPropagation` too slow on 128 CUDA cores | `cvcuda_stereo_bp` | Use `cvcuda_stereo_csbp` at reduced settings |
| Dense optical flow GPU kernel timeout at 720p | `cvcuda_dense_pyrlk` | Use `--width 640 --height 360` |
| TV-L1 hangs at default OpenCV parameters | `cvcuda_dual_tvl1` | Use `--iterations 20 --scales 2 --warps 1` |
| WebRTC stream takes ~20 s to appear | MediaMTX | Set `webrtcAdditionalHosts` to Jetson LAN IP in `mediamtx_demo.yml` |
| Any binary: endless error output like `[CAPTURE] Dropped frame` | all binaries | restart Jetson Nano `sudo reboot` |

---

## License

MIT

Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>
