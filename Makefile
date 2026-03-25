# Makefile — vision-demo
# Thin wrapper around CMake. All binaries land in ./build/
#
# Usage:
#   make          — configure (if needed) + build everything
#   make clean    — remove build artifacts
#   make rebuild  — clean + full rebuild

BUILD_DIR := build

.PHONY: all clean rebuild

all: $(BUILD_DIR)
	cmake --build $(BUILD_DIR) --parallel
	@echo ""
	@echo "  Build OK — binaries in ./build/"
	@echo ""
	@echo "  ── Stereo Depth ─────────────────────────────────────────────────────"
	@echo "  VPI SGM              : ./build/vpi_stereo_depth"
	@echo "                           --calib ./stereo-depth/stereo_calibration.yaml"
	@echo "  Block Matching       : ./build/cvcuda_stereo_bm"
	@echo "                           --calib ./stereo-depth/stereo_calibration.yaml"
	@echo "  OpenCV SGM           : ./build/cvcuda_stereo_sgm"
	@echo "                           --calib ./stereo-depth/stereo_calibration.yaml"
	@echo "  Const. Space BP      : ./build/cvcuda_stereo_csbp"
	@echo "                           --calib ./stereo-depth/stereo_calibration.yaml"
	@echo "  Belief Propagation   : ./build/cvcuda_stereo_bp"
	@echo "                           --calib ./stereo-depth/stereo_calibration.yaml"
	@echo "                         (too slow on Jetson Nano !)"
	@echo ""
	@echo "  All Stereo Depth demos stream to http://<jetson-ip>:8889/stereo_depth"
	@echo ""
	@echo "  ── Optical Flow ─────────────────────────────────────────────────────"
	@echo "  VPI PyrLK            : ./build/vpi_optical_flow"
	@echo "                         (known NVIDIA bug — demo only)"
	@echo "  Sparse PyrLK         : ./build/cvcuda_sparse_pyrlk"
	@echo "  Dense PyrLK          : ./build/cvcuda_dense_pyrlk"
	@echo "                         (use --width 640 --height 360 to avoid GPU timeout)"
	@echo "  Farneback            : ./build/cvcuda_farneback"
	@echo "  Brox                 : ./build/cvcuda_brox"
	@echo "  Dual TV-L1           : ./build/cvcuda_dual_tvl1"
	@echo "                         (use --iterations 20 --scales 2 --warps 1)"
	@echo ""
	@echo "  All Optical Flow demos stream to http://<jetson-ip>:8889/optical_flow"
	@echo ""
	@echo "  ── YOLO DNN ─────────────────────────────────────────────────────────"
	@echo "  Detection            : ./build/yolodnn_det"
	@echo "                           --model ./models/yolo.onnx"
	@echo "                           --labels ./models/yolo-labels.txt"
	@echo "  Segmentation         : ./build/yolodnn_seg"
	@echo "                           --model ./models/yolo-seg.onnx"
	@echo "                           --labels ./models/yolo-seg-labels.txt"
	@echo "  YOLOE Segmentation   : ./build/yoloednn_seg"
	@echo "                           --model ./models/yoloe-seg.onnx"
	@echo "                           --labels ./models/yoloe-seg-labels.txt"
	@echo ""
	@echo "  ── YOLO TRT ─────────────────────────────────────────────────────────"
	@echo "  Detection            : ./build/yolotrt_det"
	@echo "                           --model ./models/yolo.trt"
	@echo "                           --labels ./models/yolo-labels.txt"
	@echo "  Segmentation         : ./build/yolotrt_seg"
	@echo "                           --model ./models/yolo-seg.trt"
	@echo "                           --labels ./models/yolo-seg-labels.txt"
	@echo "  YOLOE Segmentation   : ./build/yoloetrt_seg"
	@echo "                           --model ./models/yoloe-seg.trt"
	@echo "                           --labels ./models/yoloe-seg-labels.txt"
	@echo ""
	@echo "  All YOLO demos stream to http://<jetson-ip>:8889/yolo"
	@echo ""
	@echo "  ─────────────────────────────────────────────────────────────────────"
	@echo "  Run 'sudo jetson_clocks' before starting for consistent FPS."
	@echo ""

$(BUILD_DIR):
	cmake -B $(BUILD_DIR) -S .

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean all
