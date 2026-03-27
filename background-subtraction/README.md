# VPI / OpenCV CUDA Background Subtraction

Background subtraction demos for the Jetson Nano, streaming a side-by-side
canvas to the browser via MediaMTX. No desktop required — runs fully headless.

| Binary | Language | Algorithm | Typical FPS | Notes |
|---|---|---|---|---|
| `vpi_bg_subtractor` | C++ | VPI BackgroundSubtractor (CUDA) | ~ 14 fps | VPI native implementation. |
| `cvcuda_bg_subtractor` | C++ | OpenCV CUDA MOG2 | ~ 15 fps | Gaussian mixture model. Recommended. |

---

## Algorithm comparison

| Algorithm | Model | Shadow detection | Tunable params | Best used for |
|---|---|---|---|---|
| **VPI BackgroundSubtractor** | Internal (CUDA) | No | `learning_rate` | Simple, low-overhead pipeline integrated with other VPI stages |
| **OpenCV CUDA MOG2** | Gaussian mixture (K=5) | Optional | `history`, `varThreshold`, `learning_rate` | General-purpose; more control over sensitivity and adaptation |

**Practical recommendation:** use `cvcuda_bg_subtractor` (MOG2). It is more
widely documented, exposes more tuning parameters, and produces consistent
results. Use `vpi_bg_subtractor` if you are already running other VPI
algorithms in the same pipeline and want to avoid a mixed dependency.

---

## Side-by-side layout

Both demos stream a 1×2 canvas at the configured stream resolution (default 1280×360):

```
┌─────────────────────┬─────────────────────┐
│  Original stream    │  Foreground mask    │
│                     │  (COLORMAP_TURBO)   │
└─────────────────────┴─────────────────────┘
```

The foreground mask is rendered with `COLORMAP_TURBO`: black pixels are
confirmed background, the colour scale from deep blue through green to red
indicates increasing foreground probability / pixel intensity. A foreground
percentage stat is shown in the top-left of the mask tile.

---

## How it works — background and concepts

### What is background subtraction?

Background subtraction separates the static parts of a scene (walls, floor,
furniture) from anything that moves or recently changed. The algorithm
maintains an internal **background model** — a statistical description of what
each pixel looks like when nothing is happening — and classifies any pixel that
deviates significantly from that model as **foreground**.

The result is a binary (or near-binary) mask: white pixels are foreground
(moving or changed), black pixels are background. Because the classification
runs per-pixel on every frame, moving objects are detected the moment they
enter the field of view, with no need for corner detection or feature matching.

### MOG2 — Gaussian Mixture Model

`cv::cuda::BackgroundSubtractorMOG2` models each pixel's history as a mixture
of up to K=5 Gaussian distributions. Each Gaussian captures one stable
appearance state (e.g. a pixel that alternates between shadow and sunlight
might settle into two Gaussians). The algorithm assigns each new pixel value to
the closest matching Gaussian and updates its mean and variance; pixel values
that don't fit any Gaussian are classified as foreground.

Key parameters:

* **`--history N`** — how many past frames each Gaussian is fitted to. Longer
  history = slower adaptation, more stable background model. Short history =
  faster adaptation, more susceptible to gradual lighting changes being
  classified as background. Default: `500`.
* **`--var-threshold F`** — Mahalanobis distance threshold for a pixel to be
  classified as background. Lower values are stricter (more foreground
  detections, more noise). Higher values are more permissive (cleaner mask,
  may miss slow-moving objects). Default: `16.0`.
* **`--learning-rate F`** — explicit override for the background model update
  rate per frame (0 = frozen model, 1 = single-frame model, -1 = automatic).
  Default: `0.01`.
* **`--detect-shadows`** — when enabled, MOG2 runs a secondary check to
  distinguish shadows from true foreground. Shadow pixels are marked grey
  (intensity 127) instead of white. Disabled by default because shadow
  classification adds compute cost and the grey values need extra handling if
  you threshold the mask downstream.

### VPI BackgroundSubtractor

NVIDIA VPI's implementation wraps the same conceptual model (statistical
per-pixel background modelling) but runs entirely inside the VPI CUDA backend.
The internal model is not publicly documented by NVIDIA beyond the API. Tuning
is limited to `learningRate`, making it simpler but less flexible than MOG2.

### Limitations on the Jetson Nano

Background subtraction is a per-pixel algorithm and scales with resolution.
At 1280×720 both implementations run comfortably in real-time on the Nano's
128 CUDA cores. At 1080p you may see a meaningful FPS drop; reduce capture
resolution with `--width 1280 --height 720` (the default) if needed.

Neither algorithm handles sudden global illumination changes (lights switching
on/off, direct sunlight moving across the scene) gracefully — the entire scene
is temporarily classified as foreground until the background model re-adapts.
Increase `--history` to make the model more resistant, or decrease
`--learning-rate` to slow adaptation for static cameras.

### Relation to optical flow and YOLO

Background subtraction answers *where* something is moving (a pixel-level
mask), but not *what* it is. A useful production pipeline combines all three:

1. **Background subtraction** — cheap per-frame mask of active regions.
2. **Optical flow** (see `optical-flow/`) — motion vectors inside active
   regions to estimate trajectory and speed.
3. **YOLO** (see `yolo-trt/`) — classification of detected objects, seeded
   with bounding boxes derived from connected components of the foreground mask.

Running YOLO only on foreground regions rather than the full frame reduces
inference cost significantly — important on a resource-constrained device.

---

## Hardware

* Jetson Nano B01
* One CSI camera (IMX219 or compatible)

---

## Dependencies

| Dependency | Version | Source |
|---|---|---|
| JetPack | 4.6.6 | NVIDIA |
| VPI | 1.2.x | JetPack (`vpi_bg_subtractor` only) |
| CUDA | 10.2 | JetPack |
| OpenCV | 4.8.1 | Built from source with CUDA + GStreamer + contrib |
| GStreamer | 1.20.3 | JetPack |

OpenCV must be built with `-DWITH_CUDA=ON`, `-DWITH_GSTREAMER=ON`, and the
`cudabgsegm` and `cudaimgproc` contrib modules enabled.
The system OpenCV package is not sufficient.

---

## Run

Start MediaMTX first (from the project root):

```bash
sudo jetson_clocks   # lock clocks to max — important for consistent FPS
./mediamtx/mediamtx ./mediamtx/mediamtx_demo.yml
```

### Recommended (OpenCV CUDA MOG2)

```bash
./build/cvcuda_bg_subtractor
```

### VPI implementation

```bash
./build/vpi_bg_subtractor
```

Both binaries stream to the same `bg_subtraction` MediaMTX path — switch
between them without changing the browser URL.

### Common options

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
  --udp-port N        MediaMTX UDP ingest port   [5603]
  --stream-width N    Output canvas width        [1280]
  --stream-height N   Output canvas height       [360]

Background model (cvcuda_bg_subtractor):
  --history N         Background model history   [500]
  --var-threshold F   Mahalanobis threshold      [16.0]
  --detect-shadows    Enable shadow detection
  --learning-rate F   Model adaptation rate      [0.01]

Background model (vpi_bg_subtractor):
  --learning-rate F   Model adaptation rate      [0.01]
```

Run any binary with `--help` for the full option list.

### Tuning tips

**Too much noise / flickering mask** — increase `--var-threshold` (e.g. `25`)
or increase `--history` (e.g. `1000`). Alternatively decrease `--learning-rate`
so the model is less sensitive to instantaneous changes.

**Slow-moving objects not detected** — decrease `--var-threshold` (e.g. `10`)
or decrease `--history` so the model adapts faster and the object's position
is never absorbed into the background.

**Camera is static but the whole scene goes white after a lighting change** —
this is expected behaviour. The model will re-adapt within `--history` frames.
For a static camera with stable lighting, try `--learning-rate 0` to freeze
the background model after the first few seconds, then restart to re-learn.

### Stream URLs

```
WebRTC (lowest latency) : http://<jetson-ip>:8889/bg_subtraction
HLS    (most compatible) : http://<jetson-ip>:8888/bg_subtraction
RTSP   (VLC / ffplay)   : rtsp://<jetson-ip>:8554/bg_subtraction
```

---

## Files

```
bg-subtraction/
├── vpi_bg_subtractor.cpp       # VPI CUDA BackgroundSubtractor
├── cvcuda_bg_subtractor.cpp    # OpenCV CUDA MOG2 — recommended
└── README.md
```

---

## License

MIT

Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>
