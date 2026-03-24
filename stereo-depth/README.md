# Stereo Depth

Real-time stereo depth perception for the Jetson Nano. Two synchronised CSI
cameras capture a stereo pair, disparity is computed on the GPU, and the result
is colour-mapped and streamed to the browser via MediaMTX. No desktop required
— runs fully headless.

| Binary | Algorithm | Type | Typical FPS | Note |
|---|---|---|---|---|
| `vpi_stereo_depth` | VPI SGM | Semi-Global Matching | ~ 7.5 fps | |
| `cvcuda_stereo_bm` | cv::cuda::StereoBM | Block Matching | ~ 6.5 fps | |
| `cvcuda_stereo_sgm` | cv::cuda::StereoSGM | Semi-Global Matching | - | ⚠️ requires Pascal+ |
| `cvcuda_stereo_csbp` | cv::cuda::StereoConstantSpaceBP | Belief Propagation (CSBP) | ~ 0.8 fps | |
| `cvcuda_stereo_bp` | cv::cuda::StereoBeliefPropagation | Full Belief Propagation | - | ⚠️ too slow even at minimum settings |
| `cv_stereo_sgbm.py` | OpenCV SGBM | Semi-Global Matching | ~ 2.5 fps | CPU only |

All demos share the same calibration file.
The C++ demos share the same `stereo_depth` MediaMTX stream path, and the same `--bilateral-filter` post-processing option.
The Python demo streams directly to the browser via built-in web server (no Mediamtx required)

---

## Algorithm comparison

| Algorithm | Quality | Speed | Notes |
|---|---|---|---|
| **VPI SGM** | Good | Fast | Recommended. Hardware-accelerated SGM on CUDA backend. VPI 1.2 only exposes `windowSize` at runtime — p1/p2 tuning requires VPI 2.x (Orin). |
| **StereoBM** | Low | Fastest | Simple block matching. Works well on textured surfaces; fails on textureless regions (sky, walls, skin). Good as a speed baseline. |
| **StereoSGM** | Good | Fast | Same SGM algorithm as VPI but via OpenCV. Exposes `p1`/`p2` penalties and 4 or 8 path directions. Direct comparison target for VPI SGM. |
| **StereoConstantSpaceBP** | Good | Moderate | Memory-efficient belief propagation. Limits active disparity hypotheses per pixel (`nr_plane`) — more practical on Nano than full BP. Better than BM on textureless surfaces. |
| **StereoBeliefPropagation** | Highest | Slow | Full global BP. Propagates confidence across the whole image. Highest quality of the OpenCV CUDA set. Likely too slow for real-time at 720p on the Nano — reduce `iterations` and `levels`. |

**Practical shortlist for Jetson Nano:** `vpi_stereo_depth` or `cvcuda_stereo_sgm`
for best quality, `cvcuda_stereo_bm` if you need maximum FPS.

---

## Quad-view layout (all demos)

```
+---------------------+---------------------+
|  Left (rectified)   |  Right (rectified)  |
|  + epipolar lines   |  + epipolar lines   |
+---------------------+---------------------+
|  Disparity (TURBO)  |  Depth (TURBO)      |
|  valid: xx%         |  med: x.xx m        |
+---------------------+---------------------+
```

The green horizontal epipolar lines in the camera tiles are a built-in
rectification check — corresponding features in both cameras should lie
exactly on the same horizontal line after good calibration. If they don't,
recollect calibration images.

---

## How it works — background and concepts

### What is stereo depth?

A stereo camera rig — two cameras separated by a known baseline — can measure
depth the same way human eyes do. A point that is close to the cameras appears
at very different horizontal positions in the left and right images (large
*disparity*). A point that is far away appears at nearly the same position in
both images (small disparity). Measuring that pixel offset and combining it
with the known baseline and focal length gives real-world depth in metres:

```
depth = focal_length_px * baseline_m / disparity_px
```

### Calibration and rectification

Raw stereo cameras are never perfectly aligned — there is always some rotation,
tilt and lens distortion between them. *Calibration* measures these errors using
a chessboard target and produces a set of rectification maps. *Rectification*
applies those maps to each frame so that corresponding points in both cameras
always lie on the same horizontal scan line. This is required by all stereo
matching algorithms, which search for correspondences only along horizontal rows.

### Block Matching (StereoBM)

The simplest approach: for each pixel in the left image, slide a fixed-size
block along the same row in the right image and find the position that minimises
the pixel difference. Fast and predictable, but unreliable on textureless
regions where many positions look equally similar — the algorithm has no way to
distinguish between them.

### Semi-Global Matching (SGM)

Instead of matching one block at a time, SGM enforces *smoothness* — nearby
pixels should have similar depths. It does this by aggregating matching costs
along multiple scan-line paths (4 or 8 directions) across the whole image before
picking the best disparity at each pixel. The result is dramatically better on
textureless surfaces and at depth boundaries. This is the standard algorithm for
embedded stereo vision and what both `vpi_stereo_depth` and `cvcuda_stereo_sgm`
implement.

### Belief Propagation (StereoBeliefPropagation / StereoConstantSpaceBP)

A global optimisation approach. BP propagates confidence messages between
neighbouring pixels iteratively until the whole disparity map converges to a
globally consistent solution. Higher quality than SGM at boundaries and
occlusions, but the iterative solver is expensive. `StereoConstantSpaceBP`
reduces cost by limiting the number of active disparity hypotheses per pixel
(`nr_plane`) — a good middle ground between quality and speed.

### Bilateral filter post-processing (`--bilateral-filter`)

All demos support an optional post-processing step that smooths the raw
disparity map using `cv::bilateralFilter`. A bilateral filter is
edge-preserving — it blurs smooth regions while leaving sharp depth boundaries
intact, unlike a plain Gaussian blur. Useful for cleaning up noise in
`StereoBM` output without eroding depth edges.

For a true *joint* bilateral filter (guided by the left camera image rather
than the disparity itself), replace `cv::bilateralFilter` with
`cv::ximgproc::jointBilateralFilter` if your OpenCV build includes the
`ximgproc` contrib module.

### Why does a robot need stereo depth?

**Obstacle avoidance.** A depth map gives the robot a real-world distance to
every visible surface. Combined with optical flow (see `optical-flow/`), it can
distinguish between a stationary wall at 0.5 m and a person at 2 m moving
towards it.

**Scale for optical flow.** Flow vectors alone tell you that something moved
20 pixels — but not whether that was a hand at 0.3 m or a car at 30 m. Depth
converts flow vectors into real-world velocities in m/s.

**3D mapping.** The `Q` matrix from stereo calibration allows
`cv::reprojectImageTo3D` to convert the disparity map into a 3D point cloud,
which can feed into SLAM or obstacle mapping pipelines.

---

## Hardware

- Jetson Nano B01 (dual CSI camera connectors required)
- Two compatible CSI cameras (e.g. IMX219), mounted on a rigid baseline
- A printed chessboard calibration target (see Calibration section)

---

## Dependencies

VPI 1.2.x, CUDA 10.2, OpenCV 4.8.1 (with `cudastereo` module) and GStreamer
are all part of JetPack 4.6.6. The calibration scripts additionally need:

```bash
pip3 install flask opencv-python numpy
```

OpenCV must be built from source with `-DWITH_CUDA=ON`, `-DWITH_GSTREAMER=ON`,
and the `cudastereo` contrib module enabled. The system OpenCV package does not
include CUDA stereo support.

---

## Calibration

Calibration must be done once per camera rig. The result is saved to
`stereo_calibration.yaml` which all depth demos load at runtime.

### Step 1 — Print a chessboard

Print a chessboard calibration target. The default configuration expects a
**10×4 inner corners** board (11×5 squares) with a **53 mm square size**. Both
values can be overridden via CLI — see Step 3.

### Step 2 — Capture stereo pairs

```bash
cd stereo-depth/
python3 calibration/stereo_web_capture.py
```

Open `http://<jetson-ip>:8000` in your browser. You will see a live feed from
both cameras with chessboard detection overlay.

- The tool requires the chessboard to be detected in **both** cameras before
  saving a pair
- Move the board to different positions and angles — cover corners, edges and
  centre
- Aim for **at least 20–30 pairs** for reliable calibration
- Pairs are saved to `calibration/captures/` as `left_000001.png` / `right_000001.png`

### Step 3 — Run calibration

```bash
python3 calibration/stereo_calibrate.py \
  --captures calibration/captures \
  --square 0.053 \
  --board 10x4
```

| Argument | Default | Description |
|---|---|---|
| `--captures` | `captures/` | Directory with captured PNG pairs |
| `--square` | `0.053` | Square size in metres |
| `--board` | `10x4` | Inner corners (cols×rows) |
| `--out` | `stereo_calibration.yaml` | Output file |

The script prints a calibration report including stereo RMS, baseline, and a
suggested `numDisparities` value. A rectification preview is saved as
`rectification_preview.png`.

**Target quality:**

| Stereo RMS | Quality |
|---|---|
| < 0.5 px | Good |
| 0.5 – 1.0 px | Acceptable |
| > 1.0 px | Poor — recollect images |

### Step 4 — Verify rectification

```bash
python3 calibration/rectify_web_check.py
```

Opens a browser view of the rectified stereo pair with epipolar lines overlaid.
Corresponding points should sit on the same horizontal line. If they do not,
recollect calibration images.

> `stereo_calibration.yaml` is specific to your camera pair and is not committed
> to the repository. Keep a backup.

---

## Run

Start MediaMTX first (from project root):

```bash
sudo jetson_clocks   # lock clocks to max — important for consistent FPS
./mediamtx/mediamtx ./mediamtx/mediamtx.yml
```

Then run any depth demo (from project root):

```bash
# VPI SGM — recommended
./build/vpi_stereo_depth --calib ./stereo-depth/stereo_calibration.yaml

# OpenCV CUDA Block Matching — fastest
./build/cvcuda_stereo_bm --calib ./stereo-depth/stereo_calibration.yaml

# OpenCV CUDA SGM — tunable p1/p2
./build/cvcuda_stereo_sgm --calib ./stereo-depth/stereo_calibration.yaml

# OpenCV CUDA Constant Space BP
./build/cvcuda_stereo_csbp --calib ./stereo-depth/stereo_calibration.yaml

# OpenCV CUDA full Belief Propagation — slow
./build/cvcuda_stereo_bp --calib ./stereo-depth/stereo_calibration.yaml
```

All demos stream to the same path:

```
WebRTC (lowest latency) : http://<jetson-ip>:8889/stereo_depth
HLS    (most compatible) : http://<jetson-ip>:8888/stereo_depth
RTSP   (VLC / ffplay)   : rtsp://<jetson-ip>:8554/stereo_depth
```

> **Tip:** run `sudo jetson_clocks` before starting any demo to lock CPU and GPU
> clocks to their maximum frequency. Without this the GPU governor starts at
> 76 MHz and the first few seconds of output are noticeably slow.

---

## CLI options (all demos)

All demos share a common set of flags. Run any binary with `--help` for
algorithm-specific options in addition to these.

```
Camera / capture:
  --calib FILE          Stereo calibration YAML  [stereo_calibration.yaml]
  --width N             Capture width            [1280]
  --height N            Capture height           [720]
  --fps N               Capture framerate        [15]
  --flip N              nvvidconv flip-method    [0]
  --swap-cameras        Swap left / right sensor IDs
  --no-rectify          Skip rectification (debug raw cameras)

Processing:
  --stereo-width N      Disparity computation width  [640]
  --stereo-height N     Disparity computation height [360]
  --max-disp N          Maximum disparity in pixels  [64]
                        Must be a multiple of 8 for BM/SGM.
                        Calibration script suggests a good starting value.

Post-processing:
  --bilateral-filter    Enable edge-preserving bilateral filter on disparity
  --bilateral-d N       Filter neighbourhood diameter  [9]
  --bilateral-sc F      Sigma in colour space          [75]
  --bilateral-ss F      Sigma in coordinate space      [75]

Encoder / stream:
  --bitrate N           H.264 bitrate bps         [4000000]
  --udp-port N          MediaMTX UDP ingest port  [5600]
  --stream-width N      Output canvas width       [1920]
  --stream-height N     Output canvas height      [1080]
```

---

## Bugs

### VPI SGM — `p1` / `p2` penalties not tunable on JetPack 4.6.6

VPI 1.2.x on the Nano only exposes `windowSize` and `maxDisparity` as runtime
parameters. The SGM path penalty parameters `p1` and `p2`, which control
smoothness, are hardcoded in the VPI 1.2 runtime and cannot be changed. These
parameters become tunable in VPI 2.x, which requires Jetson Orin or newer.

For tunable SGM with p1/p2 control on the Nano, use `cvcuda_stereo_sgm`
instead.

---

## Files

```
stereo-depth/
├── vpi_stereo_depth.cpp            # VPI SGM — recommended baseline
├── cvcuda_stereo_bm.cpp            # OpenCV CUDA Block Matching
├── cvcuda_stereo_sgm.cpp           # OpenCV CUDA SGM (tunable p1/p2)
├── cvcuda_stereo_csbp.cpp          # OpenCV CUDA Constant Space BP
├── cvcuda_stereo_bp.cpp            # OpenCV CUDA full Belief Propagation
├── stereo_calibration.yaml         # Generated by calibration — not committed
└── calibration/
    ├── stereo_web_capture.py       # Browser-based stereo pair capture tool
    ├── stereo_calibrate.py         # Calibration from captured pairs
    └── rectify_web_check.py        # Rectification verification tool
```

---

## Relation to the navigation pipeline

These demos are visualisation tools, not production components. In the actual
robot pipeline the relevant outputs are:

- **Disparity map** — raw input for obstacle distance estimation. Pixels with
  zero disparity have no depth data (textureless surface or occluded region).
- **Depth in metres** — `depth = focal_px * baseline_m / disparity_px`. Used
  to scale optical flow vectors from pixels/frame into real-world m/s.
- **3D point cloud** — `cv::reprojectImageTo3D(disparity, points3D, Q)` using
  the `Q` matrix from stereo calibration. Feeds SLAM or occupancy grid mapping.

For optical flow and ego-motion estimation see `optical-flow/`.
For object detection and segmentation see `yolo-trt/`.


## License

MIT

Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>
