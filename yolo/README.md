# YOLO

Real-time object detection and segmentation on the Jetson Nano using TensorRT. Three binaries cover detection, YOLO segmentation, and YOLOE segmentation. A Python-accessible shared library (`trt_infer.so`) is also built for running TRT inference from Python scripts without requiring TensorRT Python bindings.

---

## Models

Model files are not included in the repository. You need to export and convert them on a host PC before copying them to the Nano.

Place converted `.trt` files in `yolo/models/`. The directory is `.gitignored`.

---

## Model Preparation (on host PC)

### Step 1 — Export to ONNX

```bash
pip install ultralytics onnx onnxsim

python3 << 'EOF'
from ultralytics import YOLO
model = YOLO("yolov8n.pt")        # or any other YOLO model
model.export(
    format="onnx",
    opset=11,       # opset 11 avoids the Mod op unsupported by TRT 8.2
    simplify=True,
    imgsz=640,
    dynamic=False,  # static shapes required for trtexec
)
EOF
```

> **Note on the Mod op:** TensorRT 8.2 on the Nano does not support the `Mod` operator that appears in the NMS-free head of newer YOLO models. Using `opset=11` avoids it in most cases. If you still see a `Mod` error from `trtexec`, run `onnxsim` on the exported file:
>
> ```bash
> python3 -c "
> import onnx; from onnxsim import simplify
> m, ok = simplify(onnx.load('model.onnx'))
> assert ok; onnx.save(m, 'model_simplified.onnx')
> "
> ```
> Then use `model_simplified.onnx` for the next step.

### Step 2 — Convert to TensorRT on the Nano

Copy the `.onnx` file to the Nano and run:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolov8n.onnx \
  --saveEngine=yolo/models/yolo.trt \
  --fp16
```

For segmentation models:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolov8n-seg.onnx \
  --saveEngine=yolo/models/yolo-seg.trt \
  --fp16
```

Conversion takes several minutes on the Nano. The resulting `.trt` file is specific to the Nano's GPU and TRT version — it cannot be transferred from another machine.

---

## Run

Start MediaMTX first (from project root):

```bash
./mediamtx/mediamtx ./mediamtx/mediamtx.yml
```

**Detection:**
```bash
./build/yolo_det \
  --model ./yolo/models/yolo.trt \
  --labels ./yolo/labels.txt
```

**YOLO Segmentation:**
```bash
./build/yolo_seg \
  --model ./yolo/models/yolo-seg.trt \
  --labels ./yolo/labels.txt
```

**YOLOE Segmentation:**
```bash
./build/yoloe_seg \
  --model ./yolo/models/yoloe-seg.trt \
  --labels ./yolo/labels.txt
```

Watch streams at:

| Demo | URL |
|------|-----|
| Detection | `http://<jetson-ip>:8889/yolo_det` |
| Segmentation | `http://<jetson-ip>:8889/yolo_seg` |
| YOLOE Seg | `http://<jetson-ip>:8889/yoloe_seg` |

---

## Python Scripts

The Python scripts use `trt_infer.so` for TensorRT inference via `ctypes` — this avoids the TensorRT Python binding version conflict on Ubuntu 22.04 with JetPack 4.6.6.

`trt_infer.so` is built automatically by `make` and placed in `build/`. The scripts load it by path relative to the project root:

```python
import ctypes, pathlib
root = pathlib.Path(__file__).parent.parent
lib = ctypes.CDLL(str(root / "build" / "trt_infer.so"))
```

Run from the project root:

```bash
python3 yolo/scripts/yolo_nano_stream.py
```

---

## Files

```
yolo/
├── trt_infer.cpp              # TRT inference shared library (model-agnostic)
├── trt_infer.hpp
├── yolo_trt_engine.hpp        # Detection engine wrapper
├── yolo_postprocess.hpp       # Detection postprocessing
├── yolo_det.cpp               # Detection binary
├── yolo_seg_trt_engine.hpp    # Segmentation engine wrapper
├── yolo_seg_postprocess.hpp   # Segmentation postprocessing
├── yolo_seg.cpp               # YOLO segmentation binary
├── yoloe_seg_trt_engine.hpp   # YOLOE engine wrapper
├── yoloe_seg_postprocess.hpp  # YOLOE postprocessing
├── yoloe_seg.cpp              # YOLOE segmentation binary
├── labels.txt                 # COCO class labels
├── scripts/
│   ├── yolo_nano_stream.py    # Python detection stream
│   └── yolo_nano_stream2.py  # Python detection stream (alternative)
└── models/                    # Place .trt files here — not committed
    └── .gitkeep
```

## License

MIT

Copyright (c) 2026 Mischa Schaefer <https://github.com/mischa-robots/vision-demo>
