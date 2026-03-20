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
	@echo "                         --calib ./stereo-depth/stereo_calibration.yaml"
	@echo "  Block Matching       : ./build/cvcuda_stereo_bm"
	@echo "                         --calib ./stereo-depth/stereo_calibration.yaml"
	@echo "  OpenCV SGM           : ./build/cvcuda_stereo_sgm"
	@echo "                         --calib ./stereo-depth/stereo_calibration.yaml"
	@echo "  Const. Space BP      : ./build/cvcuda_stereo_csbp"
	@echo "                         --calib ./stereo-depth/stereo_calibration.yaml"
	@echo "  Belief Propagation   : ./build/cvcuda_stereo_bp"
	@echo "                         --calib ./stereo-depth/stereo_calibration.yaml"
	@echo "                         (too slow on Jetson Nano !)"
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
	@echo "  ── YOLO ─────────────────────────────────────────────────────────────"
	@echo "  Detection            : ./build/yolo_det"
	@echo "                         --model ./yolo/models/yolo.trt --labels ./yolo/models/yolo-labels.txt"
	@echo "  Segmentation         : ./build/yolo_seg"
	@echo "                         --model ./yolo/models/yolo-seg.trt --labels ./yolo/models/yolo-seg-labels.txt"
	@echo "  YOLOE Segmentation   : ./build/yoloe_seg"
	@echo "                         --model ./yolo/models/yoloe-seg.trt --labels ./yolo/models/yoloe-seg-labels.txt"
	@echo ""
	@echo "  All demos stream to http://<jetson-ip>:8889/<stream-name>"
	@echo "  Run 'sudo jetson_clocks' before starting for consistent FPS."
	@echo ""

$(BUILD_DIR):
	cmake -B $(BUILD_DIR) -S .

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean all
