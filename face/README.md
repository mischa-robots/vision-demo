# OpenCV Face Landmark Detection

Face landmark detection demos for the Jetson Nano, streaming a side-by-side
canvas to the browser via MediaMTX. No desktop required — runs fully headless.

| Binary | Language | Algorithm | Typical FPS | Notes |
|---|---|---|---|---|
| `cv_face` | C++ | Haar cascade + FacemarkLBF | x fps | CPU-only. Frontal faces only. |
| `cv_face_yn` | C++ | YuNet DNN + FacemarkLBF | x fps | CPU-only. Recommended — detects side faces. |

Both binaries stream to the same `face` MediaMTX path so you can switch
between them without changing the browser URL.

> **Performance note:** Both the face detector and FacemarkLBF run entirely on
> CPU. Detection is performed on a downscaled frame (default 0.5×) to reduce
> load; landmarks are mapped back to full resolution before drawing. FPS will
> depend on the number of faces in frame and the `--detect-scale` setting.

---

## Side-by-side layout

Both demos stream a 1×2 canvas at the configured stream resolution (default 1280×360):

```
┌─────────────────────┬─────────────────────┐
│  Original stream    │  Landmark overlay   │
└─────────────────────┴─────────────────────┘
```

The landmark overlay draws for each detected face:
- **Face bounding box** (green), with confidence score for `cv_face_yn`
- **68 landmark dots**, colour-coded by facial region
- **Facial region contours** connecting the dots (jaw, eyebrows, nose bridge, nose tip, eyes, outer mouth, inner mouth)
- **5 YuNet coarse keypoints** (yellow) — `cv_face_yn` only

Colour legend:

| Region | Points | Colour |
|---|---|---|
| Jaw line | 0–16 | Grey |
| Eyebrows | 17–26 | Amber |
| Nose bridge | 27–30 | Teal |
| Nose tip | 30–35 | Teal |
| Eyes | 36–47 | Red |
| Outer mouth | 48–59 | Orange |
| Inner mouth | 60–67 | Yellow |
| YuNet keypoints | 5 pts | Yellow (`cv_face_yn` only) |

---

## How it works — background and concepts

### What is facial landmark detection?

A **facial landmark** is a specific anatomical point on a face: the corner of an
eye, the tip of the nose, a point along the jaw line. Locating these points
automatically is called **facial landmark detection** (also known as face
alignment or face fitting).

The standard model used by both demos predicts **68 landmarks** following the
iBUG / 300-W annotation scheme — a set of points agreed upon in the computer
vision research community that covers all major facial features.

Landmarks are the foundation of many practical applications: face morphing,
driver drowsiness detection (eye aspect ratio), expression recognition, head
pose estimation, virtual try-on, and gaze tracking. On a robot, they can be used
to determine where a person is looking, whether they are alert, or as a trigger
for interaction ("approach when a face is detected and smiling").

### Two-stage pipeline

**Stage 1 — Face detection**

Before landmarks can be fitted, we need to know *where the face is*. Both demos
share the same FacemarkLBF landmark fitter but use different face detectors:

`cv_face` uses OpenCV's **Haar cascade** classifier
(`haarcascade_frontalface_alt2.xml`), a classic sliding-window detector trained
with boosted decision trees on Haar-like features. It is fast on CPU but is
trained primarily on frontal faces and struggles at yaw angles beyond ~30°.

`cv_face_yn` uses **YuNet** (`cv::FaceDetectorYN`), a CNN-based face detector
from the OpenCV model zoo. It handles side faces, tilted heads, and partially
occluded faces that the Haar cascade misses entirely. It is also ~5× faster
than the Haar cascade on the same input image and outputs 5 coarse keypoints
(eyes, nose tip, mouth corners) for free alongside each bounding box.

**Stage 2 — Landmark fitting (FacemarkLBF)**

Given a face bounding box, `FacemarkLBF` (Local Binary Features) regresses the
position of all 68 landmark points directly from pixel data. The algorithm works
in a coarse-to-fine cascade of regressors: each stage refines the landmark
positions from the previous stage by fitting a random forest to local binary
feature descriptors computed around the current estimate.

LBF was chosen because it is fast (the original paper reports 3000 fps on a
desktop CPU), requires only a single ~54 MB YAML model file with no training
step, and is natively available in OpenCV contrib's `face` module — no external
dependencies.

### Limitations on the Jetson Nano

* **CPU-only:** Neither the face detectors nor FacemarkLBF have a CUDA
  implementation in OpenCV. All processing runs on the Nano's 4 ARM Cortex-A57
  cores.
* **Haar frontal bias (`cv_face`):** Faces at more than ~30° yaw or pitch are
  frequently missed. Use `cv_face_yn` if side-face detection matters.
* **LBF profile degradation:** FacemarkLBF degrades on faces beyond ~45° yaw
  regardless of which detector found them — the 68-point model was trained
  predominantly on frontal images. The box will be drawn but landmark quality
  will suffer on extreme angles.

---

## Required model files

Both binaries share `lbfmodel.yaml`. `cv_face_yn` additionally needs the YuNet
ONNX model. Download both once from the project root:

### `lbfmodel.yaml` — landmark model (~54 MB, both binaries)

```bash
wget -P ./models https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml
```

### `face_detection_yunet_2023mar.onnx` — YuNet detector (~337 KB, `cv_face_yn` only)

```bash
wget -P ./models https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

### `haarcascade_frontalface_alt2.xml` — Haar detector (`cv_face` only)

Ships with OpenCV. Typically located at:

```
/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml
```

This path is the default. Override with `--cascade /your/path/...xml` if your
installation differs.

---

## Hardware

* Jetson Nano B01
* One CSI camera (IMX219 or compatible)

---

## Dependencies

| Dependency | Version | Source |
|---|---|---|
| JetPack | 4.6.6 | NVIDIA |
| CUDA | 10.2 | JetPack (camera pipeline only) |
| OpenCV | 4.8.1 | Built from source with CUDA + GStreamer + contrib |
| GStreamer | 1.20.3 | JetPack |

OpenCV must be built with `-DWITH_CUDA=ON`, `-DWITH_GSTREAMER=ON`, and the
`face` contrib module enabled (`opencv_contrib`). The system OpenCV package is
not sufficient.

---

## Run

Start MediaMTX first (from the project root):

```bash
sudo jetson_clocks
./mediamtx/mediamtx ./mediamtx/mediamtx_demo.yml
```

### Recommended — YuNet detector (side faces, faster)

```bash
./build/cv_face_yn \
  --yunet ./models/face_detection_yunet_2023mar.onnx \
  --model ./models/lbfmodel.yaml
```

### Haar cascade detector (frontal only)

```bash
./build/cv_face --model ./models/lbfmodel.yaml
```

With explicit cascade path:

```bash
./build/cv_face \
  --model   ./models/lbfmodel.yaml \
  --cascade /usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml
```

### Common options (both binaries)

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
  --udp-port N        MediaMTX UDP ingest port   [5604]
  --stream-width N    Output canvas width        [1280]
  --stream-height N   Output canvas height       [360]

Detection scale (both):
  --detect-scale F    [0.5]  downscale factor for detection (0.25=faster, 1.0=full res)

cv_face specific:
  --cascade PATH      Haar cascade xml           [system default]
  --scale-factor F    [1.1]  Haar scaleFactor
  --min-neighbours N  [3]    Haar minNeighbours
  --min-face N        [60]   minimum face size in px (at detect-scale resolution)

cv_face_yn specific:
  --yunet PATH        YuNet ONNX model           (REQUIRED)
  --score-threshold F [0.9]  min confidence to keep a detection
  --nms-threshold F   [0.3]  NMS IoU threshold
  --top-k N           [5000] max candidates before NMS
```

### Tuning tips

**No faces detected (`cv_face`)** — reduce `--min-neighbours 2`, reduce
`--min-face 40`, or increase `--detect-scale 0.75`. Check lighting; strong
backlight causes cascade failures.

**No faces detected (`cv_face_yn`)** — lower `--score-threshold 0.7`. YuNet
is much more robust than the Haar cascade by default.

**FPS too low** — reduce `--detect-scale 0.25`. Landmark quality is unaffected
since fitting maps results back to full resolution. Also try `--width 640 --height 360`.

**Too many false positives (`cv_face`)** — increase `--min-neighbours 5` or
`--min-face 80`.

### Stream URLs

```
WebRTC (lowest latency) : http://<jetson-ip>:8889/face
HLS    (most compatible) : http://<jetson-ip>:8888/face
RTSP   (VLC / ffplay)   : rtsp://<jetson-ip>:8554/face
```

---

## Files

```
face/
├── cv_face_landmarks.cpp       # Haar cascade + FacemarkLBF    → binary: cv_face
├── cv_face_landmarks_yn.cpp    # YuNet DNN + FacemarkLBF       → binary: cv_face_yn
└── README.md

models/                         # not committed — download separately (see above)
├── lbfmodel.yaml               # ~54 MB, shared by both binaries
└── face_detection_yunet_2023mar.onnx  # ~337 KB, cv_face_yn only
```

---

## References

**OpenCV FacemarkLBF tutorial**
https://docs.opencv.org/4.x/d2/d42/tutorial_face_landmark_detection_in_an_image.html

**YuNet face detector — OpenCV Zoo**
https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

**OpenCV DNN-based Face Detection tutorial**
https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html

**Original LBF model (GSOC 2017)**
https://github.com/kurnianggoro/GSOC2017

**OpenCV face module API**
https://docs.opencv.org/4.8.0/d3/dd0/classcv_1_1face_1_1Facemark.html

---

## License

MIT

Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>
