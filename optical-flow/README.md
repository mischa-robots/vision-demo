# VPI / OpenCV CUDA Optical Flow

Optical flow demos for the Jetson Nano, all streaming a quad-view canvas
to the browser via MediaMTX.  No desktop required — runs fully headless.

| Binary | Language | Algorithm | Type | Typical FPS | NOTE |
|---|---|---|---|---|---|
| `vpi_optical_flow` | C++ | VPI PyrLK | Sparse | 10 - 15 fps | Don't use ⚠️ |
| `cvcuda_sparse_pyrlk` | C++ | SparsePyrLKOpticalFlow | Sparse | ~ 13 fps | Recommended baseline |
| `cvcuda_dense_pyrlk` | C++ | DensePyrLKOpticalFlow | Dense | ~ 0.7 fps | Half resolution required ⚠️ |
| `cvcuda_farneback` | C++ | FarnebackOpticalFlow | Dense | ~ 3 fps | Best dense option on Nano |
| `cvcuda_brox` | C++ | BroxOpticalFlow | Dense | ~ 0.5 fps | Likely unusable at 720p |
| `cvcuda_dual_tvl1` | C++ | OpticalFlowDual_TVL1 | Dense | ~ 2.8 fps | Unusable with default params ⚠️ |
| `cv_optical_flow.py` | Python | CUDA LK + CPU Farneback | Sparse + Dense | ~ 2 fps | - |
| `cv_optical_flow_cuda.py` | Python | CUDA LK + CUDA Farneback | Sparse + Dense | ~ 2 fps | - |

⚠️ `vpi_optical_flow` has a known NVIDIA bug — tracked points are not updated
after the first frame. It is kept as a pipeline reference only. See [Bugs](#bugs).

⚠️ `cvcuda_dense_pyrlk` will almost always terminate with this error: `(-217:Gpu API call) the launch timed out ...` , to prevent it, run with half resolution:
`./build/cvcuda_dense_pyrlk --width 640 --height 360` but expect very bad results.

⚠️ `cvcuda_dual_tvl1` with the defautl parameters will take minutes to run 1 frame! set this params for 2 fps:
`cvcuda_dual_tvl1 --iterations 20 --scales 2 --warps 1`

---

## Algorithm comparison

| Algorithm | Type | Tracks | Compute cost | Best used for |
|---|---|---|---|---|
| **SparsePyrLK** | Sparse | Selected corners | Low | Ego-motion, feature tracking, robotics baseline |
| **DensePyrLK** | Dense | Every pixel | Medium-high | Full motion maps, rarely used over Farneback |
| **Farneback** | Dense | Every pixel | Medium | Full motion maps, moving object pre-filter |
| **Brox** | Dense | Every pixel | Very high | Highest accuracy, large displacements — not real-time on Nano |
| **Dual TV-L1** | Dense | Every pixel | High | Better motion boundaries than Farneback, 3-5x slower |

**Practical shortlist for Jetson Nano:** `cvcuda_sparse_pyrlk` for ego-motion
(fast, robust), `cvcuda_farneback` if you need a dense motion map.
Everything below Farneback is too slow for real-time on 128 CUDA cores.

---

## Quad-view layout

All C++ demos stream a 2×2 canvas. Sparse demos:

```
┌─────────────────────┬─────────────────────┐
│  Current + trails   │  Previous + dots    │
├─────────────────────┼─────────────────────┤
│  Flow vectors       │  Speed heatmap      │
│                     │  (HOT colourmap)    │
└─────────────────────┴─────────────────────┘
```

Dense demos replace the right column with a flow visualisation:

```
┌─────────────────────┬─────────────────────┐
│  Current frame      │  HSV colour wheel   │
│                     │  (direction+speed)  │
├─────────────────────┼─────────────────────┤
│  Speed heatmap      │  Magnitude overlay  │
│  (HOT colourmap)    │  (INFERNO colourmap)│
└─────────────────────┴─────────────────────┘
```

---

## How it works — background and concepts

### What is optical flow?

When a camera moves, or when objects in front of it move, the pixels in the
image shift from one frame to the next.  **Optical flow** is the name for
measuring those shifts — for every point in the image it asks:
*where did this point go in the next frame?*  The result is a map of motion
vectors, one per tracked point (sparse) or one per pixel (dense).

Think of it like watching rain on a car window.  Each raindrop leaves a streak
that tells you how fast and in which direction it moved.  Optical flow does the
same thing mathematically for every interesting point in the camera image.

### Sparse optical flow — Pyramidal Lucas-Kanade (SparsePyrLK)

Instead of tracking every single pixel (which is slow), the **sparse** approach
first picks a small set of *interesting* points — typically corners and edges
where there is enough texture to track reliably.  This step is called
**feature detection**; `cvcuda_sparse_pyrlk` uses the Shi-Tomasi corner
detector (`GoodFeaturesToTrack`, CUDA, `blockSize=7`).

Once we have those points, **Lucas-Kanade (LK)** tracks each one from frame to
frame.  The core idea is simple: a small patch around a point should look
roughly the same in the next frame, just shifted slightly.  LK finds the shift
that makes the two patches match as closely as possible.

To handle faster motion, the image is processed at several resolutions at once
(called a *pyramid*) — coarse levels catch large movements, fine levels refine
the result.  This is called **pyramidal LK**.

`cvcuda_sparse_pyrlk` also runs a **forward-backward consistency check**: after
tracking forward (prev→cur), it re-tracks backward (cur→prev) and rejects any
point whose round-trip error exceeds 1 px.  This eliminates false tracks on
textureless surfaces (smooth walls, car doors) that pass the LK convergence
check but are wrong matches.

### Dense optical flow — Farneback

The **dense** approach computes a motion vector for *every pixel* in the image,
not just selected corners.  This gives a much richer picture of what is moving
and where, at the cost of more computation.

**Gunnar Farneback's algorithm** works by fitting a smooth polynomial
approximation to the brightness values around each pixel, then comparing those
approximations between frames to find the best-fitting motion vector.  Like LK,
it uses a pyramid to handle larger movements.

### Dense optical flow — Brox

Energy-minimisation variational method.  Formulates optical flow as a global
optimisation problem that enforces spatial smoothness across the whole frame.
Handles large displacements well and produces the most accurate dense flow, but
the iterative solver is very expensive — not viable for real-time on the Nano.

### Dense optical flow — Dual TV-L1

TV-L1 (Total Variation L1) is another variational method.  Compared to
Farneback it handles motion boundaries and occlusions more cleanly because the
L1 norm preserves sharp edges instead of blurring them.  Runs 3–5× slower than
Farneback on CUDA; a reasonable quality/speed trade-off if Farneback misses
edge detail that matters for your application.

### Sparse vs Dense — the trade-off

| | Sparse (SparsePyrLK) | Dense (Farneback etc.) |
|---|---|---|
| What it tracks | Selected corner points | Every pixel |
| New objects | Only detected after reinit | Detected immediately |
| Speed | Fast | Slow–very slow |
| Ego-motion | Excellent | Overkill |
| Moving object detection | Indirect (point clustering) | Direct (magnitude map) |

A moving object (person, car) that enters the frame is **invisible to sparse LK
until** existing tracked points are lost and redetection lands on it.  Dense
flow catches it immediately because every pixel is evaluated every frame.

### The middle ground

For production robotics, a common pattern is:

- **Sparse PyrLK** for fast ego-motion estimation (which direction is the robot moving)
- **Dense Farneback** at reduced resolution as a cheap pre-filter to find active
  regions before running a heavier detector
- **YOLO** seeded with bounding boxes from detected objects, with additional
  tracking points placed inside those boxes

See `yolo-trt/` for the object detection and segmentation demos that complete this
pipeline.

### Why does a robot need optical flow?

**Am I moving?**  If the camera is mounted on the robot and all tracked points
drift in the same direction, the robot is moving that way — even without wheel
encoders or GPS.  This is called *ego-motion estimation* and is especially
useful indoors or on slippery surfaces where odometry is unreliable.

**Is something else moving?**  If most of the scene is stationary but a cluster
of points has large flow vectors, those points belong to a moving object.
Detecting moving obstacles early is critical for safe navigation.

**Where should I look next?**  Dense flow magnitude is a cheap way to find
active regions before running heavier algorithms.  Running a detector only on
regions where something is actually moving saves compute — which matters a lot
on a Jetson Nano.

**How fast is that object moving?**  Combined with a depth estimate from the
stereo camera (see `vpi-stereo-depth/`), flow vectors can be turned into
real-world velocities, letting the robot predict where an obstacle will be a
moment from now.

---

## Bugs

### VPI `vpiSubmitOpticalFlowPyrLK` — tracked points not updated (NVIDIA bug)

`vpi_optical_flow` uses `vpiSubmitOpticalFlowPyrLK` from VPI 1.2.x on
JetPack 4.6.6.  After the first frame, the output point array is not written
back — tracked points remain frozen at their initial positions regardless of
camera or scene motion.  This is a confirmed bug in the VPI 1.2 runtime on
CUDA 10.2 / Jetson Nano that NVIDIA has not patched.

The demo is kept as a reference for VPI pipeline structure.  For correct optical
flow use any of the `cvcuda_*` binaries instead.

### OpenCV CUDA `SparsePyrLKOpticalFlow` — implementation bugs found and fixed

Several non-obvious bugs were found and fixed during development of
`cvcuda_sparse_pyrlk`.  Documented here so others building on this code
understand what went wrong and why the fixes are necessary.

**BUG 1 — GpuMat shallow copy aliases device memory**

```cpp
// WRONG — both names now point to the same device allocation
prev_gray_gpu = cur_gray_gpu;

// CORRECT — independent allocation
prev_gray_gpu = cur_gray_gpu.clone();
```

`GpuMat` assignment is reference-counted like `cv::Mat` — it does not copy
device memory.  Without `.clone()`, `prev_gray_gpu` and `cur_gray_gpu` share
the same device pointer.  A subsequent `upload()` into `cur_gray_gpu` silently
overwrites `prev_gray_gpu` too.  LK then compares the same image against
itself → zero flow → all points appear frozen.

**BUG 2 — Points round-tripped CPU↔GPU every frame**

The original code downloaded tracked points from the GPU after each `calc()`
call and re-uploaded them as the source for the next call.  Fix: keep
`prev_pts_gpu` and `cur_pts_gpu` as `GpuMat` throughout; download to CPU only
inside `read_tracks()` for visualisation.

**BUG 3 — `create()` argument order misread**

```cpp
// WRONG — lk_iterations (20) passed as maxLevel → 20-level pyramid on 720p
SparsePyrLKOpticalFlow::create(winSize, lk_iterations, pyr_levels);

// CORRECT — signature is create(winSize, maxLevel, iters)
SparsePyrLKOpticalFlow::create(winSize, pyr_levels, lk_iterations);
```

A 20-level pyramid on 1280×720 collapses to 1×1 px at the coarse end.
LK cannot track anything meaningful there and the coarse-level garbage
estimates pollute all finer levels → chaotic flow vectors on every frame.

**BUG 4 — Displacement plausibility filter too loose**

```cpp
// WRONG — allows up to ~735 px/frame (almost the full image diagonal)
const float max_disp2 = (W*0.5f)*(W*0.5f) + (H*0.5f)*(H*0.5f);

// CORRECT — 80 px/frame covers fast hand motion at 15 fps
const float max_disp2 = max_disp_px * max_disp_px;  // default 80 px
```

**BUG 5 — CUDA corner detector `blockSize` too small; no quality filter on LK output**

`cv::cuda::createGoodFeaturesToTrackDetector` defaults to `blockSize=3`.
The CPU `cv::goodFeaturesToTrack` (used in the working Python reference)
uses `blockSize=7`.  A 3-pixel block detects noise and faint texture on smooth
surfaces (walls, car doors, skin) as "corners".  These points get tracked by LK
which always returns a result even for textureless patches — `status=1` only
means the iteration converged, not that the match is correct.

Fix 1: pass `blockSize=7` to the CUDA detector.

Fix 2: manual forward-backward consistency check.  After the forward pass
(prev→cur), re-track backward (cur→prev) using a freshly uploaded `GpuMat`
as source (not the reshaped forward output, which some kernel versions reject
as input).  All filtering is done on CPU after download.  Points whose
round-trip error exceeds `fb_thresh` (default 1.0 px) are rejected.

---

## Hardware

- Jetson Nano B01
- One CSI camera (IMX219 or compatible)

---

## Dependencies

| Dependency | Version | Source |
|---|---|---|
| JetPack | 4.6.6 | NVIDIA |
| VPI | 1.2.x | JetPack (vpi_optical_flow only) |
| CUDA | 10.2 | JetPack |
| OpenCV | 4.8.1 | Built from source with CUDA + GStreamer |
| GStreamer | 1.20.3 | JetPack |
| Python | 3.10 | System (Python demos only) |

OpenCV must be built with `-DWITH_CUDA=ON`, `-DWITH_GSTREAMER=ON`, and the
`cudaoptflow`, `cudawarping`, and `cudafeatures2d` contrib modules enabled.
The system OpenCV package is not sufficient.

---

## Run

Start MediaMTX first (from the project root):

```bash
sudo jetson_clocks   # lock clocks to max — important for consistent FPS
./mediamtx/mediamtx ./mediamtx/mediamtx.yml
```

### C++ demos

```bash
./build/cvcuda_sparse_pyrlk
./build/cvcuda_dense_pyrlk
./build/cvcuda_farneback
./build/cvcuda_brox
./build/cvcuda_dual_tvl1

# Legacy VPI demo (buggy — demo only)
./build/vpi_optical_flow
```

All C++ demos share the same CLI flags. Run any binary with `--help` for the
full list. Common options:

```
Camera / capture:
  --width N           Capture width              [1280]
  --height N          Capture height             [720]
  --fps N             Capture framerate          [15]
  --sensor N          CSI sensor-id              [0]
  --flip N            nvvidconv flip-method      [0]
  --lock-ae           Lock camera auto-exposure
  --lock-awb          Lock camera auto-white-balance

Encoder / stream:
  --bitrate N         H.264 bitrate bps          [4000000]
  --udp-port N        MediaMTX UDP ingest port   [5601]
  --stream-width N    Output canvas width        [1920]
  --stream-height N   Output canvas height       [1080]

Sparse LK (cvcuda_sparse_pyrlk only):
  --max-points N      Max tracked keypoints      [200]
  --min-points N      Re-detect below this count [20]
  --lk-window N       LK search window (odd px)  [15]
  --pyr-levels N      Pyramid levels             [4]
  --lk-iter N         Iterations per level       [20]
  --block-size N      Corner detector block size [7]
  --fb-thresh F       Forward-backward threshold [1.0]
  --max-disp F        Max displacement px/frame  [80.0]
```

### Python demos

```bash
python3 vpi-optical-flow/cv_optical_flow.py         # CPU Farneback
python3 vpi-optical-flow/cv_optical_flow_cuda.py    # CUDA Farneback
```

### Stream URLs

Each binary pushes to its own MediaMTX stream path (matching the binary name):

```
WebRTC (lowest latency) : http://<jetson-ip>:8889/<binary-name>
HLS    (most compatible) : http://<jetson-ip>:8888/<binary-name>
RTSP   (VLC / ffplay)   : rtsp://<jetson-ip>:8554/<binary-name>
```

Example: `http://<jetson-ip>:8889/cvcuda_sparse_pyrlk`

> **Tip:** run `sudo jetson_clocks` before starting any demo to lock CPU and
> GPU clocks to their maximum frequency.  Without this the GPU governor starts
> at 76 MHz and the first few seconds of output are noticeably slow.

---

## Files

```
vpi-optical-flow/
├── vpi_optical_flow.cpp                    # VPI PyrLK — buggy, reference only
├── cvcuda_sparse_pyrlk_optical_flow.cpp    # Sparse PyrLK — recommended baseline
├── cvcuda_dense_pyrlk_optical_flow.cpp     # Dense PyrLK
├── cvcuda_farneback_optical_flow.cpp       # Farneback dense
├── cvcuda_brox_optical_flow.cpp            # Brox dense — slow
├── cvcuda_dual_tvl1_optical_flow.cpp       # TV-L1 dense
├── cv_optical_flow.py                      # Python: CUDA LK + CPU Farneback
└── cv_optical_flow_cuda.py                 # Python: CUDA LK + CUDA Farneback
```

---

## Relation to the navigation pipeline

These demos are visualisation tools, not production components.  In the actual
robot pipeline the relevant outputs are:

- **Sparse PyrLK** — ego-motion estimation: if the camera is nominally static
  and most tracks move in the same direction, the robot is drifting.
- **Dense flow magnitude** — cheap pre-filter before heavier inference: regions
  with near-zero flow contain no moving obstacles and can be skipped.
- **YOLO + optical flow** — the long-term plan: seed additional tracking points
  inside YOLO bounding boxes so that detected objects are always tracked even
  when they lack strong corners.

For depth and obstacle avoidance see `vpi-stereo-depth/`.
For object detection and segmentation see `yolo-trt/`.


## License

MIT

Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>
