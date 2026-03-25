# YOLO DNN

Real-time object detection and segmentation on the Jetson Nano using OpenCV DNN with ONNX models.

Three binaries cover YOLO detection, YOLO segmentation, and YOLOE segmentation.

For more information about [YOLO](https://docs.ultralytics.com/) — please read the docs.

All tests were run with [YOLO26](https://docs.ultralytics.com/models/yolo26/) and [YOLOE26](https://docs.ultralytics.com/models/yoloe/) nano models.

---

## Models

Model files are not included in the repository. See [models/README.md](../models/README.md) for export instructions.

Place `.onnx` model files and label files in `models/` at the project root.

No conversion step is needed — the ONNX files exported in Step 1 are used directly.

---

## Run

Start MediaMTX first (from the `mediamtx/` directory):

```bash
./mediamtx mediamtx_demo.yml
```

Then run the demos from the project root:

**YOLO Detection:**
```bash
./build/yolodnn_det --model ./models/yolo.onnx --labels ./models/yolo-labels.txt
```

**YOLO Segmentation:**
```bash
./build/yolodnn_seg --model ./models/yolo-seg.onnx --labels ./models/yolo-labels.txt
```

**YOLOE Segmentation:**
```bash
./build/yoloednn_seg --model ./models/yoloe-seg.onnx --labels ./models/yoloe-seg-labels.txt
```

Run any binary with `--help` to see all available options.

Watch streams at:

| Demo | URL |
|------|-----|
| Detection | `http://<jetson-ip>:8889/yolo` |
| Segmentation | `http://<jetson-ip>:8889/yolo` |
| YOLOE Seg | `http://<jetson-ip>:8889/yolo` |

All three stream to the same path — run only one at a time.

---

## Files

```
yolo-dnn/
├── yolo_dnn_engine.hpp        # engine wrapper
├── yolo_postprocess.hpp       # postprocessing
├── yolo_det.cpp               # binary: yolodnn_det
├── yolo_seg_dnn_engine.hpp    # engine wrapper
├── yolo_seg_postprocess.hpp   # postprocessing
├── yolo_seg.cpp               # binary: yolodnn_seg
├── yoloe_seg_dnn_engine.hpp   # engine wrapper
├── yoloe_seg_postprocess.hpp  # postprocessing
└── yoloe_seg.cpp              # binary: yoloednn_seg
```

## License

MIT

Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>
