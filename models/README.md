# Models

YOLO model files for both inference backends. Not committed to the repository — place your exported files here.

| Extension | Used by | Type |
|---|---|---|
| `.onnx`   | `yolo-dnn` | OpenCV DNN |
| `.trt`    | `yolo-trt` | TensorRT |

All tests were run with [YOLO26](https://docs.ultralytics.com/models/yolo26/) and [YOLOE 26](https://docs.ultralytics.com/models/yoloe/) nano models.

More information about YOLO models: [Ultralytics Docs](https://docs.ultralytics.com/)

*IMPORTANT:* do not use NMS-free models or prompt-free models.

For more information about [YOLO](https://docs.ultralytics.com/) — please read the docs.

All tests were run with [YOLO26](https://docs.ultralytics.com/models/yolo26/) and [YOLOE26](https://docs.ultralytics.com/models/yoloe/) nano models.

Bigger models might work but were not tested.

---

## Step 1 — Export to ONNX (on host PC)

Both backends start from an PT to ONNX export.

Run this on your host PC (or a cload instance) first, not the Jetson Nano.

Then copy the `.onnx` files into the `models/` directory on the Jetson Nano.

The `.onnx` files can then be used directly by the `yolo-dnn` binaries.

It is required to have this packages installed on the host PC:

```bash
pip install ultralytics onnx onnxsim
```

The python script `export_yolo.py` will download the models from ultralytics if they don't exist on the host PC and then export them to ONNX:

**Yolo 26 nano detection model:**
```bash
./export_yolo.py --model yolo26n.pt
```

**Yolo 26 nano segmentation model:**
```bash
./export_yolo.py --model yolo26n-seg.pt
```

**Yoloe 26 nano segmentation model:**
```bash
./export_yolo.py --model yoloe-26n-seg.pt --labels yoloe-seg-labels.txt
```

`--labels` is required for yoloe-seg models to prevent the OOM error during trt export.


> *Note on the Mod OP error:* TensorRT 8.2 on the Nano does not support the `Mod` operator that appears in the NMS-free head of newer YOLO models.
> Also prompt-free models are not supported by TensorRT 8.2
> Both *might* work with the OpenCV DNN binaries (ONNX) but is not tested.

> *Note on the OOM error with YOLOE-SEG:* yoloe-seg is trained on the 4500+ labels RAM classes, that will cause an OOM (out of memory) error during trt export.
> Extending the SWAP memory does not solve the issue. Only reducing the classes is a solution on the Jetson Nano.
> The example `yoloe-seg-labels.txt` file has 205 classes.


See the [yolo-dnn/README.md](../yolo-dnn/README.md) to run the demo programs with the ONNX models.

---

## Step 2 — Convert to TensorRT (on the Nano, for yolo-trt only)

Copy the `.onnx` files to the Nano and run `trtexec`. Conversion takes several
minutes. The resulting `.trt` file is tied to this Nano's GPU and TRT version —
it cannot be copied from another machine different than a Jetson Nano.

**Detection:**
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolo26n.onnx \
  --saveEngine=models/yolo26n.trt \
  --fp16
```

**Segmentation:**
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolo26n-seg.onnx \
  --saveEngine=models/yolo26n-seg.trt \
  --fp16
```

**YOLOE segmentation:**
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=yoloe-26n-seg.onnx \
  --saveEngine=models/yoloe-26n-seg.trt \
  --fp16
```

See the [yolo-trt/README.md](../yolo-trt/README.md) to run the demo programs with the TRT models.

---

## Files

Label files: One `.txt` file per model, one class name per line.

The default YOLO models (object detection and segmentation) are trained on the COCO 80-class data set, 
the file (`yolo-labels.txt`) works for default standard detection and segmentation models.

If you fine-tuned your yolo model with a different set of labels, you must use them with the export script and when running the binaries.

YOLOE requires its own label file (`yoloe-seg-labels.txt` is an example of 205 classes), matching the custom class set it was exported with as ONNX.

```
models/
├── yolo26n.onnx              # detection — yolo-dnn
├── yolo26n.trt               # detection — yolo-trt
├── yolo26n-seg.onnx          # segmentation — yolo-dnn
├── yolo26n-seg.trt           # segmentation — yolo-trt
├── yoloe-26n-seg.onnx        # YOLOE segmentation — yolo-dnn
├── yoloe-26n-seg.trt         # YOLOE segmentation — yolo-trt
├── yolo-labels.txt           # COCO 80 classes
└── yoloe-labels.txt          # custom YOLOE class list
```

## License

MIT

Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>
