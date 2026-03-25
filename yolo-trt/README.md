# YOLO TRT

Real-time object detection and segmentation on the Jetson Nano using TensorRT.

Three binaries cover YOLO detection, YOLO segmentation, and YOLOE segmentation.

A Python-accessible shared library (`trt_infer.so`) is also built for running TRT inference from Python scripts without requiring TensorRT Python bindings.

For more information about [YOLO](https://docs.ultralytics.com/) — please read the docs.

All tests were run with [YOLO26](https://docs.ultralytics.com/models/yolo26/) and [YOLOE26](https://docs.ultralytics.com/models/yoloe/) nano models.

---

## Models

Model files are not included in the repository. See [models/README.md](../models/README.md) for export and conversion instructions.

Place `.trt` engine files and label files in `models/` at the project root.

---

## Run

Start MediaMTX first (from the `mediamtx/` directory):

```bash
./mediamtx mediamtx_demo.yml
```

Then run the demos from the project root:

**YOLO Detection:**
```bash
./build/yolotrt_det --model ./models/yolo.trt --labels ./models/yolo-labels.txt
```

**YOLO Segmentation:**
```bash
./build/yolotrt_seg --model ./models/yolo-seg.trt --labels ./models/yolo-labels.txt
```

**YOLOE Segmentation:**
```bash
./build/yoloetrt_seg --model ./models/yoloe-seg.trt --labels ./models/yoloe-seg-labels.txt
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
yolo-trt/
├── trt_infer.cpp              # TRT inference shared library (model-agnostic)
├── yolo_trt_engine.hpp        # engine wrapper
├── yolo_postprocess.hpp       # postprocessing
├── yolo_det.cpp               # binary: yolotrt_det
├── yolo_seg_trt_engine.hpp    # engine wrapper
├── yolo_seg_postprocess.hpp   # postprocessing
├── yolo_seg.cpp               # binary: yolotrt_seg
├── yoloe_seg_trt_engine.hpp   # engine wrapper
├── yoloe_seg_postprocess.hpp  # postprocessing
├── yoloe_seg.cpp              # binary: yoloetrt_seg
```

## License

MIT

Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>
