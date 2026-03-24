# YOLO TRT

Real-time object detection and segmentation with the YOLO models on the Jetson Nano using TensorRT.

Three binaries cover YOLO detection, YOLO segmentation, and YOLOE segmentation.

A Python-accessible shared library (`trt_infer.so`) is also built for running TRT inference from Python scripts without requiring TensorRT Python bindings.

More information about [YOLO](https://docs.ultralytics.com/) - please read the docs.

All tests were run with [YOLO26](https://docs.ultralytics.com/models/yolo26/) and [YOLOE26](https://docs.ultralytics.com/models/yoloe/) nano models.

---

## Models

Model files are not included in the repository. You need to export and convert them on a host PC before copying them to the Nano.

Place converted `.trt` files in `yolo-trt/models/`. The directory is `.gitignored`.

---

## Model Preparation (on host PC)

### Step 1 — Export to ONNX

```bash
pip install ultralytics onnx onnxsim

python3 << 'EOF'
from ultralytics import YOLO
model = YOLO("yolo26n.pt")        # or any other YOLO model
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
  --onnx=yolo26n.onnx \
  --saveEngine=yolo-trt/models/yolo26n.trt \
  --fp16
```

For segmentation models:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolo26n-seg.onnx \
  --saveEngine=yolo-trt/models/yolo26n-seg.trt \
  --fp16
```

Conversion takes several minutes on the Nano. The resulting `.trt` file is specific to the Nano's GPU and TRT version — it cannot be transferred from another machine different to the Jetson Nano.

---

## Run

Start MediaMTX first (from mediamtx directory):

```bash
./mediamtx mediamtx_demo.yml
```

Then run the demos in a separate terminal (from root directory):

**YOLO Detection:**
```bash
./build/yolotrt_det --model ./yolo-trt/models/yolo.trt --labels ./yolo-trt/models/yolo-labels.txt
```

**YOLO Segmentation:**
```bash
./build/yolotrt_seg --model ./yolo-trt/models/yolo-seg.trt --labels ./yolo-trt/models/yolo-seg-labels.txt
```

**YOLOE Segmentation:**
```bash
./build/yoloetrt_seg --model ./yolo-trt/models/yoloe-seg.trt --labels ./yolo-trt/models/yoloe-seg-labels.txt
```

Watch streams at:

| Demo | URL |
|------|-----|
| Detection | `http://<jetson-ip>:8889/yolo` |
| Segmentation | `http://<jetson-ip>:8889/yolo` |
| YOLOE Seg | `http://<jetson-ip>:8889/yolo` |

---

## Python Scripts (WIP !)

The Python scripts use `trt_infer.so` for TensorRT inference via `ctypes` — this avoids the TensorRT Python binding version conflict on Ubuntu 22.04 with JetPack 4.6.6.

`trt_infer.so` is built automatically by `make` and placed in `build/`. The scripts load it by path relative to the project root:

```python
import ctypes, pathlib
root = pathlib.Path(__file__).parent.parent
lib = ctypes.CDLL(str(root / "build" / "trt_infer.so"))
```

Run from the project root:

```bash
python3 yolo-trt/scripts/yolo_det.py
```

```bash
python3 yolo-trt/scripts/yolo_seg.py
```

```bash
python3 yolo-trt/scripts/yoloe_seg.py
```

The python examples will stream the video via their own local webserver.

---

## Files

```
yolo-trt/
├── trt_infer.cpp              # TRT inference shared library (model-agnostic)
├── yolo_trt_engine.hpp        # engine wrapper
├── yolo_postprocess.hpp       # postprocessing
├── yolo_det.cpp               # YOLO Detection binary: yolotrt_det
├── yolo_seg_trt_engine.hpp    # engine wrapper
├── yolo_seg_postprocess.hpp   # postprocessing
├── yolo_seg.cpp               # YOLO Segmentation binary: yolotrt_seg
├── yoloe_seg_trt_engine.hpp   # engine wrapper
├── yoloe_seg_postprocess.hpp  # postprocessing
├── yoloe_seg.cpp              # YOLOE Segmentation binary: yoloetrt_seg
├── scripts/
│   ├── yolo_det.py            # (WIP) YOLO detection Python example
│   ├── yolo_seg.py            # (WIP) YOLO segmentation Python example
│   ├── yoloe_seg.py           # (WIP) YOLOE segmentation Python example
└── models/                    # Place .trt model files label files here — not committed
    └── .gitkeep
```

## License

MIT

Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>
