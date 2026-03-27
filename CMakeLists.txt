cmake_minimum_required(VERSION 3.12)
project(vision-demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Output all binaries and libraries to build/ ───────────────────────────────
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

# ── Common dependencies ───────────────────────────────────────────────────────
set(OpenCV_DIR /usr/lib/aarch64-linux-gnu/cmake/opencv4)
find_package(OpenCV REQUIRED COMPONENTS
    core imgproc calib3d imgcodecs videoio highgui
    cudaimgproc cudawarping cudaarithm
    cudaoptflow cudafeatures2d cudastereo
)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# ── CUDA / TensorRT (yolo) ────────────────────────────────────────────────────
find_package(CUDA REQUIRED)

# TensorRT does not ship CMake config files on JetPack 4.x — locate manually
find_path(TRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/aarch64-linux-gnu
)
find_library(TRT_LIBRARY nvinfer
    PATHS /usr/lib/aarch64-linux-gnu
)
find_library(CUBLAS_LIBRARY cublas
    PATHS /usr/local/cuda-10.2/targets/aarch64-linux/lib
)
if(NOT TRT_INCLUDE_DIR OR NOT TRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found — check NvInfer.h and libnvinfer locations")
endif()

# ── VPI + GStreamer ───────────────────────────────────────────────────────────
find_package(vpi 1.1 REQUIRED)

pkg_check_modules(GST     REQUIRED gstreamer-1.0)
pkg_check_modules(GSTAPP  REQUIRED gstreamer-app-1.0)

separate_arguments(GST_CFLAGS_LIST    UNIX_COMMAND "${GST_CFLAGS}")
separate_arguments(GSTAPP_CFLAGS_LIST UNIX_COMMAND "${GSTAPP_CFLAGS}")


# ── Reusable link sets ────────────────────────────────────────────────────────

# All OpenCV CUDA optical flow targets
set(CVCUDA_OF_COMPILE_OPTS
    ${GST_CFLAGS_LIST} ${GSTAPP_CFLAGS_LIST} -O2 -Wall -Wextra
)
set(CVCUDA_OF_INCLUDE_DIRS
    ${OpenCV_INCLUDE_DIRS} ${GST_INCLUDE_DIRS} ${GSTAPP_INCLUDE_DIRS}
)
set(CVCUDA_OF_LINK_LIBS
    ${OpenCV_LIBS}
    opencv_cudaoptflow
    opencv_cudaimgproc
    opencv_cudafeatures2d
    ${GST_LIBRARIES} ${GSTAPP_LIBRARIES}
    Threads::Threads
)

# All OpenCV CUDA stereo targets (adds cudastereo)
set(CVCUDA_STEREO_COMPILE_OPTS
    ${GST_CFLAGS_LIST} ${GSTAPP_CFLAGS_LIST} -O2 -Wall -Wextra
)
set(CVCUDA_STEREO_INCLUDE_DIRS
    ${OpenCV_INCLUDE_DIRS} ${GST_INCLUDE_DIRS} ${GSTAPP_INCLUDE_DIRS}
)
set(CVCUDA_STEREO_LINK_LIBS
    ${OpenCV_LIBS}
    opencv_cudastereo
    opencv_cudaimgproc
    ${GST_LIBRARIES} ${GSTAPP_LIBRARIES}
    Threads::Threads
)

# VPI stereo + optical flow targets
set(VPI_COMPILE_OPTS
    ${GST_CFLAGS_LIST} ${GSTAPP_CFLAGS_LIST} -O2 -Wall -Wextra
)
set(VPI_INCLUDE_DIRS
    ${OpenCV_INCLUDE_DIRS} ${GST_INCLUDE_DIRS} ${GSTAPP_INCLUDE_DIRS}
)
set(VPI_LINK_LIBS
    vpi
    ${OpenCV_LIBS}
    ${GST_LIBRARIES} ${GSTAPP_LIBRARIES}
    Threads::Threads
)


# ═════════════════════════════════════════════════════════════════════════════
# trt_infer.so  — shared library loaded by Python scripts via ctypes
# ═════════════════════════════════════════════════════════════════════════════
add_library(trt_infer SHARED
    yolo-trt/trt_infer.cpp
)
target_include_directories(trt_infer PRIVATE
    ${TRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(trt_infer PRIVATE
    ${TRT_LIBRARY} ${CUDA_LIBRARIES} ${OpenCV_LIBS} Threads::Threads
)


# ═════════════════════════════════════════════════════════════════════════════
# YOLO DNN targets
# ═════════════════════════════════════════════════════════════════════════════
add_executable(yolodnn_det yolo-dnn/yolo_det.cpp)
target_include_directories(yolodnn_det PRIVATE
    ${OpenCV_INCLUDE_DIRS})
target_compile_options(yolodnn_det PRIVATE -O2 -Wall -Wno-deprecated-declarations)
target_link_libraries(yolodnn_det PRIVATE
    ${OpenCV_LIBS} Threads::Threads)

add_executable(yolodnn_seg yolo-dnn/yolo_seg.cpp)
target_include_directories(yolodnn_seg PRIVATE
    ${OpenCV_INCLUDE_DIRS})
target_compile_options(yolodnn_seg PRIVATE -O2 -Wall -Wno-deprecated-declarations)
target_link_libraries(yolodnn_seg PRIVATE
    ${OpenCV_LIBS} Threads::Threads)

add_executable(yoloednn_seg yolo-dnn/yoloe_seg.cpp)
target_include_directories(yoloednn_seg PRIVATE
    ${OpenCV_INCLUDE_DIRS})
target_compile_options(yoloednn_seg PRIVATE -O2 -Wall -Wno-deprecated-declarations)
target_link_libraries(yoloednn_seg PRIVATE
    ${OpenCV_LIBS} ${CUBLAS_LIBRARY} Threads::Threads)

# ═════════════════════════════════════════════════════════════════════════════
# YOLO TRT targets
# ═════════════════════════════════════════════════════════════════════════════
add_executable(yolotrt_det yolo-trt/yolo_det.cpp)
target_include_directories(yolotrt_det PRIVATE
    ${TRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_compile_options(yolotrt_det PRIVATE -O2 -Wall -Wno-deprecated-declarations)
target_link_libraries(yolotrt_det PRIVATE
    ${TRT_LIBRARY} ${CUDA_LIBRARIES} ${OpenCV_LIBS} Threads::Threads)

add_executable(yolotrt_seg yolo-trt/yolo_seg.cpp)
target_include_directories(yolotrt_seg PRIVATE
    ${TRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_compile_options(yolotrt_seg PRIVATE -O2 -Wall -Wno-deprecated-declarations)
target_link_libraries(yolotrt_seg PRIVATE
    ${TRT_LIBRARY} ${CUDA_LIBRARIES} ${OpenCV_LIBS} Threads::Threads)

add_executable(yoloetrt_seg yolo-trt/yoloe_seg.cpp)
target_include_directories(yoloetrt_seg PRIVATE
    ${TRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_compile_options(yoloetrt_seg PRIVATE -O2 -Wall -Wno-deprecated-declarations)
target_link_libraries(yoloetrt_seg PRIVATE
    ${TRT_LIBRARY} ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${CUBLAS_LIBRARY} Threads::Threads)


# ═════════════════════════════════════════════════════════════════════════════
# Stereo depth demos
#
# VPI:
#   vpi_stereo_depth      VPI SGM — recommended, best quality on Nano
#
# OpenCV CUDA (cudastereo module):
#   cvcuda_stereo_bm      Block Matching — fastest, weakest quality
#   cvcuda_stereo_sgm     Semi-Global Matching — same algo as VPI, tunable p1/p2
#   cvcuda_stereo_csbp    Constant Space Belief Propagation — good quality, moderate speed
#   cvcuda_stereo_bp      Full Belief Propagation — highest quality, very slow
#
# All demos:
#   - accept the same --calib stereo_calibration.yaml
#   - stream to the same stereo_depth MediaMTX path
#   - support --bilateral-filter for disparity post-processing
# ═════════════════════════════════════════════════════════════════════════════

add_executable(vpi_stereo_depth stereo-depth/vpi_stereo_depth.cpp)
target_include_directories(vpi_stereo_depth PRIVATE ${VPI_INCLUDE_DIRS})
target_compile_options(vpi_stereo_depth     PRIVATE ${VPI_COMPILE_OPTS})
target_link_libraries(vpi_stereo_depth      PRIVATE ${VPI_LINK_LIBS})

add_executable(cvcuda_stereo_bm  stereo-depth/cvcuda_stereo_bm.cpp)
target_include_directories(cvcuda_stereo_bm  PRIVATE ${CVCUDA_STEREO_INCLUDE_DIRS})
target_compile_options(cvcuda_stereo_bm      PRIVATE ${CVCUDA_STEREO_COMPILE_OPTS})
target_link_libraries(cvcuda_stereo_bm       PRIVATE ${CVCUDA_STEREO_LINK_LIBS})

add_executable(cvcuda_stereo_sgm stereo-depth/cvcuda_stereo_sgm.cpp)
target_include_directories(cvcuda_stereo_sgm PRIVATE ${CVCUDA_STEREO_INCLUDE_DIRS})
target_compile_options(cvcuda_stereo_sgm     PRIVATE ${CVCUDA_STEREO_COMPILE_OPTS})
target_link_libraries(cvcuda_stereo_sgm      PRIVATE ${CVCUDA_STEREO_LINK_LIBS})

add_executable(cvcuda_stereo_csbp stereo-depth/cvcuda_stereo_csbp.cpp)
target_include_directories(cvcuda_stereo_csbp PRIVATE ${CVCUDA_STEREO_INCLUDE_DIRS})
target_compile_options(cvcuda_stereo_csbp     PRIVATE ${CVCUDA_STEREO_COMPILE_OPTS})
target_link_libraries(cvcuda_stereo_csbp      PRIVATE ${CVCUDA_STEREO_LINK_LIBS})

add_executable(cvcuda_stereo_bp  stereo-depth/cvcuda_stereo_bp.cpp)
target_include_directories(cvcuda_stereo_bp  PRIVATE ${CVCUDA_STEREO_INCLUDE_DIRS})
target_compile_options(cvcuda_stereo_bp      PRIVATE ${CVCUDA_STEREO_COMPILE_OPTS})
target_link_libraries(cvcuda_stereo_bp       PRIVATE ${CVCUDA_STEREO_LINK_LIBS})


# ═════════════════════════════════════════════════════════════════════════════
# Optical flow demos
#
# VPI (requires VPI runtime):
#   vpi_optical_flow      VPI PyrLK — note: known NVIDIA bug, demo only
#
# OpenCV CUDA sparse (track selected corners):
#   cvcuda_sparse_pyrlk   Shi-Tomasi + PyrLK + forward-backward filter
#                         Recommended baseline for robotics ego-motion
#
# OpenCV CUDA dense (compute flow for every pixel):
#   cvcuda_dense_pyrlk    Dense Pyramidal LK — half-resolution recommended
#   cvcuda_farneback      Farneback polynomial expansion — best dense option
#   cvcuda_brox           Brox variational — most accurate, very slow
#   cvcuda_dual_tvl1      TV-L1 variational — good boundaries, slow
# ═════════════════════════════════════════════════════════════════════════════

add_executable(vpi_optical_flow optical-flow/vpi_optical_flow.cpp)
target_include_directories(vpi_optical_flow PRIVATE ${VPI_INCLUDE_DIRS})
target_compile_options(vpi_optical_flow     PRIVATE ${VPI_COMPILE_OPTS})
target_link_libraries(vpi_optical_flow      PRIVATE ${VPI_LINK_LIBS})

add_executable(cvcuda_sparse_pyrlk optical-flow/cvcuda_sparse_pyrlk_optical_flow.cpp)
target_include_directories(cvcuda_sparse_pyrlk PRIVATE ${CVCUDA_OF_INCLUDE_DIRS})
target_compile_options(cvcuda_sparse_pyrlk     PRIVATE ${CVCUDA_OF_COMPILE_OPTS})
target_link_libraries(cvcuda_sparse_pyrlk      PRIVATE ${CVCUDA_OF_LINK_LIBS})

add_executable(cvcuda_dense_pyrlk optical-flow/cvcuda_dense_pyrlk_optical_flow.cpp)
target_include_directories(cvcuda_dense_pyrlk PRIVATE ${CVCUDA_OF_INCLUDE_DIRS})
target_compile_options(cvcuda_dense_pyrlk     PRIVATE ${CVCUDA_OF_COMPILE_OPTS})
target_link_libraries(cvcuda_dense_pyrlk      PRIVATE ${CVCUDA_OF_LINK_LIBS})

add_executable(cvcuda_farneback optical-flow/cvcuda_farneback_optical_flow.cpp)
target_include_directories(cvcuda_farneback PRIVATE ${CVCUDA_OF_INCLUDE_DIRS})
target_compile_options(cvcuda_farneback     PRIVATE ${CVCUDA_OF_COMPILE_OPTS})
target_link_libraries(cvcuda_farneback      PRIVATE ${CVCUDA_OF_LINK_LIBS})

add_executable(cvcuda_brox optical-flow/cvcuda_brox_optical_flow.cpp)
target_include_directories(cvcuda_brox PRIVATE ${CVCUDA_OF_INCLUDE_DIRS})
target_compile_options(cvcuda_brox     PRIVATE ${CVCUDA_OF_COMPILE_OPTS})
target_link_libraries(cvcuda_brox      PRIVATE ${CVCUDA_OF_LINK_LIBS})

add_executable(cvcuda_dual_tvl1 optical-flow/cvcuda_dual_tvl1_optical_flow.cpp)
target_include_directories(cvcuda_dual_tvl1 PRIVATE ${CVCUDA_OF_INCLUDE_DIRS})
target_compile_options(cvcuda_dual_tvl1     PRIVATE ${CVCUDA_OF_COMPILE_OPTS})
target_link_libraries(cvcuda_dual_tvl1      PRIVATE ${CVCUDA_OF_LINK_LIBS})


# ═════════════════════════════════════════════════════════════════════════════
# Background subtractor demos
# ═════════════════════════════════════════════════════════════════════════════

add_executable(vpi_bgsubtract background-subtraction/vpi_bg_subtractor.cpp)
target_include_directories(vpi_bgsubtract PRIVATE ${VPI_INCLUDE_DIRS})
target_compile_options(vpi_bgsubtract     PRIVATE ${VPI_COMPILE_OPTS})
target_link_libraries(vpi_bgsubtract      PRIVATE ${VPI_LINK_LIBS})

add_executable(cvcuda_bgsubtract background-subtraction/cvcuda_bg_subtractor.cpp)
target_include_directories(cvcuda_bgsubtract PRIVATE
    ${OpenCV_INCLUDE_DIRS} ${GST_INCLUDE_DIRS} ${GSTAPP_INCLUDE_DIRS})
target_compile_options(cvcuda_bgsubtract     PRIVATE
    ${GST_CFLAGS_LIST} ${GSTAPP_CFLAGS_LIST} -O2 -Wall -Wextra)
target_link_libraries(cvcuda_bgsubtract      PRIVATE
    ${OpenCV_LIBS}
    opencv_cudabgsegm
    opencv_cudaimgproc
    ${GST_LIBRARIES} ${GSTAPP_LIBRARIES}
    Threads::Threads)
