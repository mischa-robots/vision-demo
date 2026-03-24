#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>

"""
cv_optical_flow.py
──────────────────
OpenCV optical flow demo for Jetson Nano — streams a quad-view canvas to a
browser via MediaMTX.  No desktop required, runs fully headless.

Quad-view layout (default 1920×1080 output):
  ┌─────────────────────┬─────────────────────┐
  │  Sparse LK          │  Dense Farneback     │
  │  trails + arrows    │  HSV colour wheel    │
  ├─────────────────────┼─────────────────────┤
  │  LK speed heatmap   │  Dense magnitude     │
  │  (HOT colourmap)    │  (INFERNO colourmap) │
  └─────────────────────┴─────────────────────┘

Sparse side  — cv2.cuda.SparsePyrLKOpticalFlow (GPU)
              Shi-Tomasi corner detection (CPU, re-run on track loss)
Dense  side  — cv2.calcOpticalFlowFarneback (CPU)
              Colour-mapped with HSV wheel + magnitude overlay

Camera:  CSI via nvarguscamerasrc
Encoder: nvv4l2h264enc (Jetson HW encoder)
Ingest:  udp+rtp → MediaMTX (same pattern as vpi_optical_flow.cpp)

Watch the stream at:
  WebRTC (lowest latency) : http://<jetson-ip>:8889/optical_flow
  HLS    (most compatible) : http://<jetson-ip>:8888/optical_flow
  RTSP   (VLC / ffplay)   : rtsp://<jetson-ip>:8554/optical_flow

Usage:
  python3 cv_optical_flow.py
  python3 cv_optical_flow.py --help
"""

import argparse
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# gi / GLib not needed — push pipeline uses cv2.VideoWriter (GStreamer backend)


# ════════════════════════════════════════════════════════════════════════════
# Config — all runtime parameters in one place
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    # Camera / capture
    width:          int   = 1280
    height:         int   = 720
    fps:            int   = 15
    sensor_id:      int   = 0
    flip_method:    int   = 0
    lock_ae:        bool  = False   # lock auto-exposure — helps LK on brightness changes
    lock_awb:       bool  = False   # lock auto-white-balance

    # Encoder / stream
    bitrate:        int   = 4_000_000
    iframeinterval: int   = 30
    udp_host:       str   = "127.0.0.1"
    udp_port:       int   = 5601
    stream_name:    str   = "optical_flow"
    stream_w:       int   = 1920
    stream_h:       int   = 1080

    # Sparse LK parameters
    max_points:     int   = 200     # max tracked keypoints
    min_points:     int   = 20      # re-detect when live tracks drop below this
    lk_win_size:    int   = 15      # LK search window (odd px)
    lk_max_level:   int   = 4       # pyramid levels
    lk_max_iter:    int   = 20      # max iterations per level
    lk_epsilon:     float = 0.01    # convergence epsilon

    # Dense Farneback parameters
    fb_pyr_scale:   float = 0.5
    fb_levels:      int   = 3
    fb_win_size:    int   = 15
    fb_iterations:  int   = 3
    fb_poly_n:      int   = 5
    fb_poly_sigma:  float = 1.2

    # Visualisation
    mag_clip:       float = 25.0    # speed saturation px/frame (sparse heatmap)
    arrow_scale:    int   = 4       # multiply vector length for display
    show_fps:       bool  = True


# ════════════════════════════════════════════════════════════════════════════
# GStreamer pipeline strings — mirror the C++ versions exactly
# ════════════════════════════════════════════════════════════════════════════

def capture_pipeline(cfg: Config) -> str:
    """nvarguscamerasrc → nvvidconv → BGR → appsink.
    max-buffers=1 drop=true → always the newest frame, never block capture."""
    ae  = " aelock=true"  if cfg.lock_ae  else ""
    awb = " awblock=true" if cfg.lock_awb else ""
    return (
        f"nvarguscamerasrc sensor-id={cfg.sensor_id}{ae}{awb}"
        f" ! video/x-raw(memory:NVMM)"
        f",width={cfg.width},height={cfg.height}"
        f",framerate={cfg.fps}/1,format=NV12"
        f" ! nvvidconv flip-method={cfg.flip_method}"
        f" ! video/x-raw,format=BGRx,width={cfg.width},height={cfg.height}"
        f" ! videoconvert ! video/x-raw,format=BGR"
        f" ! appsink name=appsink max-buffers=1 drop=true sync=false"
    )


def push_pipeline(cfg: Config) -> str:
    """cv2.VideoWriter GStreamer sink pipeline.
    OpenCV's GStreamer backend opens this as:
      appsrc(BGR) → [this string] → udpsink
    VideoWriter handles appsrc caps negotiation internally — reliable on Jetson.
    The string must start at the element *after* appsrc."""
    w, h = cfg.stream_w, cfg.stream_h
    return (
        f"appsrc"
        f" ! video/x-raw,format=BGR,width={w},height={h},framerate={cfg.fps}/1"
        f" ! videoconvert"
        f" ! video/x-raw,format=BGRx"
        f" ! nvvidconv"
        f" ! video/x-raw(memory:NVMM),format=NV12"
        f" ! nvv4l2h264enc maxperf-enable=1 insert-sps-pps=true"
        f"   profile=2 control-rate=1"
        f"   bitrate={cfg.bitrate} iframeinterval={cfg.iframeinterval}"
        f" ! h264parse config-interval=-1"
        f" ! rtph264pay pt=96 config-interval=-1"
        f" ! udpsink host={cfg.udp_host} port={cfg.udp_port} sync=false"
    )


# ════════════════════════════════════════════════════════════════════════════
# GstPushPipeline — thin wrapper around cv2.VideoWriter (GStreamer backend)
#
# Why VideoWriter instead of gi/appsrc directly:
#   OpenCV's GStreamer backend constructs the appsrc element internally and
#   sets its caps from the VideoWriter parameters before linking the pipeline.
#   This guarantees correct caps negotiation on Jetson — the gi/parse_launch
#   approach requires manual appsrc wiring that is fragile across JetPack
#   versions and was causing silent buffer drops (black stream).
# ════════════════════════════════════════════════════════════════════════════
class GstPushPipeline:
    """Push BGR frames into a GStreamer pipeline via cv2.VideoWriter."""

    def __init__(self, pipeline_str: str, width: int, height: int, fps: int) -> None:
        self._ok     = False
        self._writer = cv2.VideoWriter(
            pipeline_str,
            cv2.CAP_GSTREAMER,
            0,          # fourcc ignored by GStreamer backend
            float(fps),
            (width, height),
        )
        if not self._writer.isOpened():
            raise RuntimeError(
                "[gst-push] VideoWriter failed to open pipeline.\n"
                f"  Pipeline: {pipeline_str}"
            )
        self._ok = True
        print("[gst-push] pipeline started.")

    @property
    def ok(self) -> bool:
        return self._ok

    def push(self, frame: np.ndarray) -> bool:
        """Write one BGR frame. Returns False if the writer has gone bad."""
        if not self._ok:
            return False
        self._writer.write(frame)
        return True

    def stop(self) -> None:
        if not self._ok:
            return
        self._writer.release()
        self._ok = False
        print("[gst-push] pipeline stopped.")

    def __del__(self) -> None:
        self.stop()


# ════════════════════════════════════════════════════════════════════════════
# Sparse LK tracker (cv2.cuda.SparsePyrLKOpticalFlow)
# ════════════════════════════════════════════════════════════════════════════
class SparseLKTracker:
    """
    Shi-Tomasi corner detection (CPU) + LK tracking (CUDA).

    State machine mirrors the C++ VpiLK:
      detect()  — detect corners into prev_pts, set prev_gray
      track()   — GPU LK: prev → cur, return live count
      advance() — cur becomes prev for next frame
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg    = cfg
        self._lk     = cv2.cuda.SparsePyrLKOpticalFlow_create(
            winSize    = (cfg.lk_win_size, cfg.lk_win_size),
            maxLevel   = cfg.lk_max_level,
            iters      = cfg.lk_max_iter,
            useInitialFlow = False,
        )
        # Shi-Tomasi params
        self._st_params = dict(
            maxCorners      = cfg.max_points,
            qualityLevel    = 0.01,
            minDistance     = 10,
            blockSize       = 7,
        )
        self._prev_gray_gpu: Optional[cv2.cuda.GpuMat] = None
        self._cur_gray_gpu:  Optional[cv2.cuda.GpuMat] = None
        self._prev_pts:      Optional[np.ndarray]       = None  # (N,1,2) float32
        self._cur_pts:       Optional[np.ndarray]       = None
        self._status:        Optional[np.ndarray]       = None  # (N,1) uint8
        self._colours:       List[Tuple[int,int,int]]   = []
        self.n_reinit = 0

    # Assign a stable BGR colour from detection position (matches C++ point_colour)
    @staticmethod
    def _point_colour(x: float, y: float) -> Tuple[int, int, int]:
        hue = (int(x) ^ int(y)) % 180
        hsv = np.array([[[hue, 220, 220]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    def detect(self, gray: np.ndarray) -> None:
        """Detect Shi-Tomasi corners and seed the tracker."""
        pts = cv2.goodFeaturesToTrack(gray, **self._st_params)
        if pts is None or len(pts) == 0:
            # (1, 0, 2) — the shape cuda::SparsePyrLKOpticalFlow expects
            pts = np.empty((1, 0, 2), dtype=np.float32)
        else:
            # goodFeaturesToTrack returns (N, 1, 2); CUDA LK needs (1, N, 2)
            pts = pts.reshape(1, -1, 2)
        self._prev_pts = pts
        self._colours  = [self._point_colour(pts[0, i, 0], pts[0, i, 1])
                          for i in range(pts.shape[1])]
        self.n_reinit  = pts.shape[1]
        # Upload to GPU
        self._prev_gray_gpu = cv2.cuda.GpuMat()
        self._prev_gray_gpu.upload(gray)

    def track(self, gray: np.ndarray) -> int:
        """GPU LK track from prev → cur. Returns live count."""
        if self._prev_pts is None or self._prev_pts.shape[1] == 0:
            return 0

        self._cur_gray_gpu = cv2.cuda.GpuMat()
        self._cur_gray_gpu.upload(gray)

        prev_gpu = cv2.cuda.GpuMat()
        prev_gpu.upload(self._prev_pts)          # (1, N, 2) float32

        cur_gpu, status_gpu, _ = self._lk.calc(
            self._prev_gray_gpu,
            self._cur_gray_gpu,
            prev_gpu,
            None,
        )
        # Keep (1, N, 2) shape so advance() and tracks() stay consistent
        self._cur_pts = cur_gpu.download()       # (1, N, 2)
        self._status  = status_gpu.download()    # (1, N) or (N,) depending on version
        return int(np.sum(self._status.ravel() == 1))

    def advance(self) -> None:
        """Swap cur → prev for the next frame (zero-copy pointers)."""
        self._prev_gray_gpu = self._cur_gray_gpu
        self._prev_pts      = self._cur_pts

    def tracks(self):
        """Yield (prev_pt, cur_pt, colour, valid) for all points.

        n is capped to the shortest of prev_pts, cur_pts and status so a
        sudden camera move (which can cause LK to return fewer entries than
        were fed in) never causes an IndexError."""
        if self._prev_pts is None or self._cur_pts is None:
            return
        status_flat = self._status.ravel() if self._status is not None else None
        n = min(
            self._prev_pts.shape[1],
            self._cur_pts.shape[1],
            len(status_flat) if status_flat is not None else 0,
            len(self._colours),
        )
        for i in range(n):
            valid = (status_flat is not None and status_flat[i] == 1)
            yield (
                self._prev_pts[0, i],
                self._cur_pts[0, i],
                self._colours[i],
                valid,
            )

    def live_count(self) -> int:
        if self._status is None:
            return 0
        return int(np.sum(self._status.ravel() == 1))


# ════════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ════════════════════════════════════════════════════════════════════════════

def tile_label(tile: np.ndarray, label: str,
               stats: List[str] = ()) -> None:
    """Draw label + optional stat lines in the top-left — shadow + white text."""
    def put(s, pt, scale):
        cv2.putText(tile, s, pt, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0),       3, cv2.LINE_AA)
        cv2.putText(tile, s, pt, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    put(label, (8, 22), 0.60)
    y = 44
    for s in stats:
        put(s, (8, y), 0.48)
        y += 18


def draw_arrow(img: np.ndarray,
               frm: Tuple[float, float],
               to:  Tuple[float, float],
               colour: Tuple[int, int, int],
               thickness: int = 2) -> None:
    """arrowedLine that skips sub-pixel motion to avoid OpenCV assertion."""
    dx = to[0] - frm[0]
    dy = to[1] - frm[1]
    if dx*dx + dy*dy < 1.0:
        return
    cv2.arrowedLine(img,
                    (int(frm[0]), int(frm[1])),
                    (int(to[0]),  int(to[1])),
                    colour, thickness, cv2.LINE_AA, 0, 0.25)


def flow_to_hsv(flow: np.ndarray) -> np.ndarray:
    """Convert a (H,W,2) float32 flow field to a BGR HSV-wheel image.
    Hue encodes direction, value encodes magnitude (normalised to frame max)."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 90.0 / np.pi).astype(np.uint8)   # hue  0-180
    hsv[..., 1] = 255                                       # full saturation
    max_mag = mag.max()
    if max_mag > 0:
        hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def dense_magnitude_overlay(frame_bgr: np.ndarray,
                             flow: np.ndarray,
                             mag_clip: float = 25.0) -> np.ndarray:
    """INFERNO colourmap over a darkened frame, keyed by flow magnitude."""
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    norm   = np.clip(mag / mag_clip, 0.0, 1.0)
    norm_u8 = (norm * 255).astype(np.uint8)
    heat    = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)
    # Blend: show heat where there is motion, darken the rest
    alpha = norm[..., np.newaxis].astype(np.float32)
    dark  = (frame_bgr * 0.35).astype(np.uint8)
    return (dark * (1.0 - alpha) + heat * alpha).astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════════
# build_quad — assemble the 2×2 canvas sent to the encoder every frame
#
#   t0 (top-left)  : sparse LK  — current frame + fading trails + current dots
#   t1 (top-right) : dense flow — HSV colour-wheel field
#   t2 (bot-left)  : sparse LK  — speed heatmap (HOT colourmap)
#   t3 (bot-right) : dense flow — magnitude overlay (INFERNO colourmap)
# ════════════════════════════════════════════════════════════════════════════

def build_quad(
    cur_bgr:   np.ndarray,
    prev_bgr:  Optional[np.ndarray],
    tracker:   SparseLKTracker,
    trail:     np.ndarray,           # persistent BGR tile, modified in-place
    dense_flow: Optional[np.ndarray],
    fps:       float,
    n_live:    int,
    n_total:   int,
    cfg:       Config,
) -> np.ndarray:

    tw = cfg.stream_w // 2
    th = cfg.stream_h // 2
    tsz = (tw, th)

    sx = tw / cur_bgr.shape[1]
    sy = th / cur_bgr.shape[0]

    # ── Tile 0: sparse LK — current frame + fading trails + dots ─────────────
    t0 = cv2.resize(cur_bgr, tsz, interpolation=cv2.INTER_AREA)

    # Multiplicative fade — trail persists ~8 frames at 0.82 decay
    trail *= 0.82

    for prev_pt, cur_pt, colour, valid in tracker.tracks():
        if not valid:
            continue
        p_sc = (cur_pt[0]  * sx, cur_pt[1]  * sy)
        q_sc = (prev_pt[0] * sx, prev_pt[1] * sy)
        cv2.line(trail,
                 (int(q_sc[0]), int(q_sc[1])),
                 (int(p_sc[0]), int(p_sc[1])),
                 colour, 2, cv2.LINE_AA)
        cv2.circle(trail, (int(p_sc[0]), int(p_sc[1])),
                   4, colour, -1, cv2.LINE_AA)

    cv2.add(t0, trail.astype(np.uint8), t0)
    tile_label(t0, "Sparse LK  +  trails",
               [f"tracks: {n_live} / {n_total}"])

    # ── Tile 1: dense flow — HSV colour wheel ─────────────────────────────────
    if dense_flow is not None:
        hsv_img = cv2.resize(flow_to_hsv(dense_flow), tsz,
                             interpolation=cv2.INTER_AREA)
        # Blend lightly with the original frame so scene content is visible
        cur_small = cv2.resize(cur_bgr, tsz, interpolation=cv2.INTER_AREA)
        t1 = cv2.addWeighted(cur_small, 0.30, hsv_img, 0.70, 0)
        # Legend: colour wheel icon (tiny 30×30) top-right corner
        _draw_hsv_legend(t1, (tw - 38, 8), 30)
        tile_label(t1, "Dense Farneback  (HSV)")
    else:
        t1 = np.zeros((th, tw, 3), dtype=np.uint8)
        tile_label(t1, "Dense Farneback  (HSV)", ["(waiting for frame 2)"])

    # ── Tile 2: sparse LK — speed heatmap ────────────────────────────────────
    heat = np.zeros((th, tw), dtype=np.float32)
    mean_mag = 0.0
    cnt      = 0
    for prev_pt, cur_pt, colour, valid in tracker.tracks():
        if not valid:
            continue
        dx  = cur_pt[0]  - prev_pt[0]
        dy  = cur_pt[1]  - prev_pt[1]
        mag = float(np.sqrt(dx*dx + dy*dy))
        mean_mag += mag
        cnt += 1
        val = min(mag / cfg.mag_clip, 1.0)
        p_sc = (int(cur_pt[0] * sx), int(cur_pt[1] * sy))
        r    = int(val * 20 + 4)
        cv2.circle(heat, p_sc, r, float(val), -1, cv2.LINE_AA)

    cv2.GaussianBlur(heat, (15, 15), 5.0, heat)
    heat_u8 = (np.clip(heat, 0.0, 1.0) * 255).astype(np.uint8)
    t2 = cv2.applyColorMap(heat_u8, cv2.COLORMAP_HOT)

    if cnt > 0:
        mean_mag /= cnt
    tile_label(t2, "LK speed  (HOT)",
               [f"mean: {mean_mag:.1f} px/frame"])

    # ── Tile 3: dense flow — magnitude / INFERNO overlay ─────────────────────
    if dense_flow is not None:
        cur_small = cv2.resize(cur_bgr, tsz, interpolation=cv2.INTER_AREA)
        flow_small = cv2.resize(dense_flow, tsz, interpolation=cv2.INTER_AREA)
        t3 = dense_magnitude_overlay(cur_small, flow_small, cfg.mag_clip)
        mag_all, _ = cv2.cartToPolar(dense_flow[..., 0], dense_flow[..., 1])
        mean_dense = float(mag_all.mean())
        tile_label(t3, "Dense magnitude  (INFERNO)",
                   [f"mean: {mean_dense:.1f} px/frame"])
    else:
        t3 = np.zeros((th, tw, 3), dtype=np.uint8)
        tile_label(t3, "Dense magnitude  (INFERNO)", ["(waiting for frame 2)"])

    # ── Assemble canvas ───────────────────────────────────────────────────────
    canvas = np.empty((cfg.stream_h, cfg.stream_w, 3), dtype=np.uint8)
    canvas[0:th,    0:tw]    = t0
    canvas[0:th,    tw:tw*2] = t1
    canvas[th:th*2, 0:tw]    = t2
    canvas[th:th*2, tw:tw*2] = t3

    # Grid lines between tiles
    cv2.line(canvas, (tw, 0),           (tw, cfg.stream_h), (200, 200, 200), 1)
    cv2.line(canvas, (0,  th),          (cfg.stream_w, th), (200, 200, 200), 1)

    # FPS counter top-right — black shadow + yellow text
    fps_str = f"{fps:.1f} FPS"
    cv2.putText(canvas, fps_str, (cfg.stream_w - 110, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),     3, cv2.LINE_AA)
    cv2.putText(canvas, fps_str, (cfg.stream_w - 110, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)

    return canvas


def _draw_hsv_legend(img: np.ndarray, origin: Tuple[int, int], size: int) -> None:
    """Draw a tiny HSV colour-wheel icon at origin (x, y)."""
    cx = origin[0] + size // 2
    cy = origin[1] + size // 2
    r  = size // 2
    for angle_deg in range(0, 360, 4):
        rad = np.deg2rad(angle_deg)
        hue = angle_deg // 2          # 0-180
        hsv = np.array([[[hue, 220, 220]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colour = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        x = int(cx + r * np.cos(rad))
        y = int(cy + r * np.sin(rad))
        cv2.circle(img, (x, y), 2, colour, -1, cv2.LINE_AA)


# ════════════════════════════════════════════════════════════════════════════
# CLI argument parser
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="OpenCV optical flow demo — quad-view streamed via MediaMTX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Stream via MediaMTX (source: udp+rtp://host:port):\n"
            "  WebRTC : http://<host>:8889/optical_flow\n"
            "  HLS    : http://<host>:8888/optical_flow\n"
            "  RTSP   : rtsp://<host>:8554/optical_flow\n"
        ),
    )

    g = p.add_argument_group("Camera / capture")
    g.add_argument("--width",         type=int,   default=1280,
                   help="Capture width  [1280]")
    g.add_argument("--height",        type=int,   default=720,
                   help="Capture height [720]")
    g.add_argument("--fps",           type=int,   default=15,
                   help="Capture framerate [15]")
    g.add_argument("--sensor",        type=int,   default=0,
                   help="CSI sensor-id [0]")
    g.add_argument("--flip",          type=int,   default=0,
                   help="nvvidconv flip-method [0]")
    g.add_argument("--lock-ae",       action="store_true",
                   help="Lock camera auto-exposure")
    g.add_argument("--lock-awb",      action="store_true",
                   help="Lock camera auto-white-balance")

    g = p.add_argument_group("Encoder / stream")
    g.add_argument("--bitrate",       type=int,   default=4_000_000,
                   help="H.264 bitrate bps [4000000]")
    g.add_argument("--iframeinterval",type=int,   default=30,
                   help="Keyframe interval in frames [30]")
    g.add_argument("--udp-host",      type=str,   default="127.0.0.1",
                   help="MediaMTX UDP ingest host [127.0.0.1]")
    g.add_argument("--udp-port",      type=int,   default=5601,
                   help="MediaMTX UDP ingest port [5601]")
    g.add_argument("--stream-name",   type=str,   default="optical_flow",
                   help="Stream name shown in banner [optical_flow]")
    g.add_argument("--stream-width",  type=int,   default=1920,
                   help="Output canvas width  [1920]")
    g.add_argument("--stream-height", type=int,   default=1080,
                   help="Output canvas height [1080]")

    g = p.add_argument_group("Sparse LK parameters")
    g.add_argument("--max-points",    type=int,   default=200,
                   help="Max tracked keypoints [200]")
    g.add_argument("--min-points",    type=int,   default=20,
                   help="Re-detect corners below this count [20]")
    g.add_argument("--lk-win-size",   type=int,   default=15,
                   help="LK search window (odd px) [15]")
    g.add_argument("--lk-max-level",  type=int,   default=4,
                   help="LK pyramid levels [4]")
    g.add_argument("--lk-max-iter",   type=int,   default=20,
                   help="LK max iterations per level [20]")
    g.add_argument("--lk-epsilon",    type=float, default=0.01,
                   help="LK convergence epsilon [0.01]")

    g = p.add_argument_group("Dense Farneback parameters")
    g.add_argument("--fb-pyr-scale",  type=float, default=0.5,
                   help="Farneback pyramid scale [0.5]")
    g.add_argument("--fb-levels",     type=int,   default=3,
                   help="Farneback pyramid levels [3]")
    g.add_argument("--fb-win-size",   type=int,   default=15,
                   help="Farneback averaging window [15]")
    g.add_argument("--fb-iterations", type=int,   default=3,
                   help="Farneback iterations per level [3]")
    g.add_argument("--fb-poly-n",     type=int,   default=5,
                   help="Farneback poly_n [5]")
    g.add_argument("--fb-poly-sigma", type=float, default=1.2,
                   help="Farneback poly_sigma [1.2]")

    g = p.add_argument_group("Visualisation")
    g.add_argument("--mag-clip",      type=float, default=25.0,
                   help="Speed saturation px/frame [25.0]")
    g.add_argument("--arrow-scale",   type=int,   default=4,
                   help="Arrow display multiplier [4]")
    g.add_argument("--no-fps",        action="store_true",
                   help="Suppress FPS console output")

    args = p.parse_args()

    cfg = Config(
        width          = args.width,
        height         = args.height,
        fps            = args.fps,
        sensor_id      = args.sensor,
        flip_method    = args.flip,
        lock_ae        = args.lock_ae,
        lock_awb       = args.lock_awb,
        bitrate        = args.bitrate,
        iframeinterval = args.iframeinterval,
        udp_host       = args.udp_host,
        udp_port       = args.udp_port,
        stream_name    = args.stream_name,
        stream_w       = args.stream_width,
        stream_h       = args.stream_height,
        max_points     = args.max_points,
        min_points     = args.min_points,
        lk_win_size    = args.lk_win_size,
        lk_max_level   = args.lk_max_level,
        lk_max_iter    = args.lk_max_iter,
        lk_epsilon     = args.lk_epsilon,
        fb_pyr_scale   = args.fb_pyr_scale,
        fb_levels      = args.fb_levels,
        fb_win_size    = args.fb_win_size,
        fb_iterations  = args.fb_iterations,
        fb_poly_n      = args.fb_poly_n,
        fb_poly_sigma  = args.fb_poly_sigma,
        mag_clip       = args.mag_clip,
        arrow_scale    = args.arrow_scale,
        show_fps       = not args.no_fps,
    )
    return cfg


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Graceful shutdown on Ctrl+C / SIGTERM ─────────────────────────────────
    stop = [False]
    def _sig(signum, frame):
        stop[0] = True
    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    cfg = parse_args()

    print("══════════════════════════════════════════════")
    print("  Optical Flow (OpenCV) — Jetson Nano")
    print("══════════════════════════════════════════════")
    print(f"  Capture     : {cfg.width}×{cfg.height} @ {cfg.fps}fps")
    print(f"  Stream out  : {cfg.stream_w}×{cfg.stream_h}")
    print(f"  Sparse LK   : win={cfg.lk_win_size}  levels={cfg.lk_max_level}"
          f"  max_pts={cfg.max_points}  min_pts={cfg.min_points}")
    print(f"  Dense FB    : levels={cfg.fb_levels}  win={cfg.fb_win_size}"
          f"  iters={cfg.fb_iterations}")
    print(f"  Sensor      : {cfg.sensor_id}  flip={cfg.flip_method}")
    print(f"  Lock AE/AWB : {cfg.lock_ae} / {cfg.lock_awb}")
    print(f"  Bitrate     : {cfg.bitrate} bps")
    print(f"  UDP →       : {cfg.udp_host}:{cfg.udp_port}")
    print(f"  Stream      : http://<host>:8889/{cfg.stream_name}  (WebRTC)")
    print("══════════════════════════════════════════════\n")

    # ── Camera capture ────────────────────────────────────────────────────────
    cap_pipe = capture_pipeline(cfg)
    print(f"[camera] pipeline: {cap_pipe}\n")

    cap = cv2.VideoCapture(cap_pipe, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[camera] FATAL: could not open camera.", file=sys.stderr)
        print(f"  Pipeline: {cap_pipe}", file=sys.stderr)
        sys.exit(1)
    print("[camera] opened.")

    # ── GStreamer push pipeline ───────────────────────────────────────────────
    push_pipe = push_pipeline(cfg)
    print(f"\n[gst-push] pipeline: {push_pipe}\n")
    gst_push = GstPushPipeline(push_pipe, cfg.stream_w, cfg.stream_h, cfg.fps)

    # ── Sparse LK tracker ────────────────────────────────────────────────────
    tracker  = SparseLKTracker(cfg)

    # Persistent trail overlay — fading motion history, tile-sized
    tw    = cfg.stream_w // 2
    th    = cfg.stream_h // 2
    trail = np.zeros((th, tw, 3), dtype=np.float32)

    # ── State ─────────────────────────────────────────────────────────────────
    cur_bgr:    Optional[np.ndarray] = None
    prev_bgr:   Optional[np.ndarray] = None
    prev_gray:  Optional[np.ndarray] = None
    dense_flow: Optional[np.ndarray] = None

    have_prev   = False
    frame_n     = 0
    fps_smooth  = 0.0
    t_last      = time.monotonic()

    print("[main] entering loop. Ctrl+C to stop.\n")

    while not stop[0]:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[camera] read failed, retrying...", file=sys.stderr)
            time.sleep(0.01)
            continue

        cur_bgr  = frame
        cur_gray = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)

        n_live  = 0
        n_total = 0
        just_reinit = False

        if not have_prev:
            # Frame 0: seed the sparse tracker
            tracker.detect(cur_gray)
            have_prev   = True
            just_reinit = True
        else:
            # ── Sparse LK track ───────────────────────────────────────────────
            n_live  = tracker.track(cur_gray)
            n_total = tracker.n_reinit

            # Re-detect Shi-Tomasi if too many points have been lost
            if n_live < cfg.min_points:
                tracker.detect(cur_gray)
                trail[:]    = 0
                just_reinit = True
                n_live      = tracker.n_reinit
                n_total     = n_live

            # ── Dense Farneback (half-resolution for performance) ─────────────
            # Full 1280×720 costs ~1s/frame on Cortex-A57.
            # Running at 640×360 cuts cost ~4× with negligible visual difference.
            # Flow vectors are scaled ×2 after to represent full-res displacement.
            if prev_gray is not None:
                fb_w = cfg.width  // 2
                fb_h = cfg.height // 2
                pg_small = cv2.resize(prev_gray, (fb_w, fb_h),
                                      interpolation=cv2.INTER_AREA)
                cg_small = cv2.resize(cur_gray,  (fb_w, fb_h),
                                      interpolation=cv2.INTER_AREA)
                flow_small = cv2.calcOpticalFlowFarneback(
                    pg_small, cg_small,
                    None,
                    cfg.fb_pyr_scale,
                    cfg.fb_levels,
                    cfg.fb_win_size,
                    cfg.fb_iterations,
                    cfg.fb_poly_n,
                    cfg.fb_poly_sigma,
                    0,
                )
                # Scale back to capture resolution: resize geometry ×2, vectors ×2
                dense_flow = cv2.resize(flow_small,
                                        (cfg.width, cfg.height),
                                        interpolation=cv2.INTER_LINEAR) * 2.0

        # ── FPS (exponential moving average — same formula as C++) ────────────
        t_now      = time.monotonic()
        dt         = t_now - t_last
        t_last     = t_now
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt if dt > 0 else 0.0)
        frame_n   += 1

        # ── Build quad canvas ─────────────────────────────────────────────────
        canvas = build_quad(
            cur_bgr, prev_bgr, tracker, trail,
            dense_flow, fps_smooth, n_live, n_total, cfg,
        )

        # ── Push to MediaMTX ──────────────────────────────────────────────────
        if gst_push.ok:
            if not gst_push.push(canvas):
                print(f"[gst-push] push error on frame {frame_n}", file=sys.stderr)

        # ── Console stats ─────────────────────────────────────────────────────
        if cfg.show_fps and fps_smooth > 0:
            period = max(1, int(fps_smooth + 0.5))
            if frame_n % period == 0:
                reinit_str = (f"  [reinit → {tracker.n_reinit} pts]"
                              if just_reinit else "")
                print(f"\r[main] frame={frame_n:6d}"
                      f"  fps={fps_smooth:4.1f}"
                      f"  lk={n_live}/{n_total}"
                      f"{reinit_str}          ",
                      end="", flush=True)

        # ── Advance sparse tracker (swap prev ↔ cur) ──────────────────────────
        if have_prev and not just_reinit:
            tracker.advance()

        prev_bgr  = cur_bgr
        prev_gray = cur_gray

    # ── Shutdown ──────────────────────────────────────────────────────────────
    print("\n[main] shutting down...")
    gst_push.stop()
    cap.release()
    print("[main] done.")


if __name__ == "__main__":
    main()
