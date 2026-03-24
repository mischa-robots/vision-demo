#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>

"""
Stereo Disparity Web — Jetson Nano / OpenCV SGBM + WLS filter
Run:  python3 stereo_web.py
Stop: Ctrl+C

Key fix vs previous version:
  Disparity is computed ONCE per frame at a single chosen resolution.
  Both the disparity visualisation and the depth map are derived from
  that single result.  The previous version ran two separate SGBM calls
  (one downscaled for preview, one full-res for depth) which caused
  flickering because the two calls used frames captured at different
  times and produced inconsistent disparity values.

Resolution strategy:
  SGBM runs at  CAPTURE / SGBM_DOWNSCALE.
  Before passing to reprojectImageTo3D the disparity values are scaled
  back up by SGBM_DOWNSCALE so the full-res Q matrix gives correct Z.
  Web MJPEG is always resized to WEB_W x WEB_H for bandwidth.
"""

import os
import time
import threading
import signal

import cv2
import numpy as np
import yaml
from flask import (Flask, Response, request, redirect,
                   url_for, render_template_string, send_file)

# ──────────────────────────────────────────────────────────────
# Shutdown
# ──────────────────────────────────────────────────────────────
STOP = threading.Event()

def _sig(signum, frame):
    if STOP.is_set():
        return
    print("\n[shutdown] signal – waiting for GStreamer …")
    STOP.set()
    def _exit():
        time.sleep(3.5)
        print("[shutdown] force exit.")
        os._exit(0)
    threading.Thread(target=_exit, daemon=True).start()

signal.signal(signal.SIGINT,  _sig)
signal.signal(signal.SIGTERM, _sig)

# ──────────────────────────────────────────────────────────────
# Config  ← edit these to match your setup
# ──────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8020

CALIB_FILE  = "stereo_calibration.yaml"
PARAMS_FILE = "sgbm_params.yaml"

LEFT_ID  = 0
RIGHT_ID = 1

# Capture resolution — must match what you calibrated at
CAPTURE_W = 1280
CAPTURE_H = 720
CAPTURE_FPS = 30

# SGBM downscale factor.  SGBM runs at CAPTURE/SGBM_DOWNSCALE.
# 1 = full res (most accurate, ~1-2 fps on Nano)
# 2 = half res (good balance, ~4-5 fps)  <-- recommended
# Disparity values are scaled back up before depth reprojection
# so Z values are always correct regardless of this setting.
SGBM_DOWNSCALE = 2

# Web stream size (what the browser sees)
WEB_W = 960
WEB_H = 540

COMPUTE_HZ = 5.0
JPEG_Q     = 80
LINE_STEP  = 40

# ──────────────────────────────────────────────────────────────
# GStreamer pipeline — appsink only, no nvenc
# ──────────────────────────────────────────────────────────────
def gst_pipeline(sensor_id, w, h, fps):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} aelock=true awblock=true ! "
        f"video/x-raw(memory:NVMM),width={w},height={h},"
        f"framerate={fps}/1,format=NV12 ! "
        f"nvvidconv ! "
        f"video/x-raw,format=BGRx,width={w},height={h} ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

# ──────────────────────────────────────────────────────────────
# Rectifier
# ──────────────────────────────────────────────────────────────
class Rectifier:
    def __init__(self, path):
        self.path        = path
        self._lock       = threading.Lock()
        self._loaded     = False
        self._maps_ready = False
        self.image_size  = None
        self.mtxL = self.distL = self.mtxR = self.distR = None
        self.R1 = self.R2 = self.P1 = self.P2 = self.Q = None
        self.mapLx = self.mapLy = self.mapRx = self.mapRy = None
        self.calib_info   = "not loaded"
        self.focal_px     = 0.0
        self.baseline_m   = 0.0
        self.suggested_nd = 160

    def _mat(self, fs, name):
        n = fs.getNode(name)
        if n.empty():
            raise RuntimeError(f"Missing '{name}'")
        return n.mat()

    def load(self):
        with self._lock:
            fs = cv2.FileStorage(self.path, cv2.FILE_STORAGE_READ)
            if not fs.isOpened():
                raise RuntimeError(f"Cannot open {self.path}")
            self.mtxL  = self._mat(fs, "mtxL")
            self.distL = self._mat(fs, "distL")
            self.mtxR  = self._mat(fs, "mtxR")
            self.distR = self._mat(fs, "distR")
            self.R1    = self._mat(fs, "R1")
            self.R2    = self._mat(fs, "R2")
            self.P1    = self._mat(fs, "P1")
            self.P2    = self._mat(fs, "P2")
            try:
                self.Q = self._mat(fs, "Q")
            except Exception:
                self.Q = None
            fs.release()
            try:
                f  = float(self.P1[0, 0])
                Tx = float(self.P2[0, 3])
                bl = abs(Tx / f)
                self.focal_px   = f
                self.baseline_m = bl
                nd = int(np.ceil(f * bl / 0.3 / 16) * 16)
                self.suggested_nd = nd
                self.calib_info = (
                    f"f={f:.0f}px  B={bl*100:.1f}cm  "
                    f"disp@1m={f*bl:.0f}px  disp@3m={f*bl/3:.0f}px  "
                    f"→ numDisparities≥{nd}"
                )
            except Exception:
                self.calib_info = "could not parse P matrices"
            self._loaded = True
            self._maps_ready = False

    def ensure_maps(self, w, h):
        with self._lock:
            if not self._loaded or (self._maps_ready and self.image_size == (w, h)):
                return
            self.mapLx, self.mapLy = cv2.initUndistortRectifyMap(
                self.mtxL, self.distL, self.R1, self.P1, (w, h), cv2.CV_16SC2)
            self.mapRx, self.mapRy = cv2.initUndistortRectifyMap(
                self.mtxR, self.distR, self.R2, self.P2, (w, h), cv2.CV_16SC2)
            self.image_size  = (w, h)
            self._maps_ready = True

    def rectify_pair(self, imgL, imgR):
        h, w = imgL.shape[:2]
        self.ensure_maps(w, h)
        with self._lock:
            rL = cv2.remap(imgL, self.mapLx, self.mapLy, cv2.INTER_LINEAR)
            rR = cv2.remap(imgR, self.mapRx, self.mapRy, cv2.INTER_LINEAR)
        return rL, rR


rectifier = Rectifier(CALIB_FILE)
try:
    rectifier.load()
    print(f"[calib] {rectifier.calib_info}")
except Exception as e:
    print(f"[calib] WARNING: {e}")

# ──────────────────────────────────────────────────────────────
# Live capture
# ──────────────────────────────────────────────────────────────
class LiveStereo:
    def __init__(self):
        self.lock  = threading.Lock()
        self.capL = self.capR = None
        self.lastL = self.lastR = None
        self.thread = None

    def start(self):
        print(f"[camera] opening {CAPTURE_W}×{CAPTURE_H}@{CAPTURE_FPS}fps")
        self.capL = cv2.VideoCapture(
            gst_pipeline(LEFT_ID,  CAPTURE_W, CAPTURE_H, CAPTURE_FPS),
            cv2.CAP_GSTREAMER)
        self.capR = cv2.VideoCapture(
            gst_pipeline(RIGHT_ID, CAPTURE_W, CAPTURE_H, CAPTURE_FPS),
            cv2.CAP_GSTREAMER)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not STOP.is_set():
            with self.lock:
                for cap, attr in ((self.capL, "lastL"), (self.capR, "lastR")):
                    if cap and cap.isOpened():
                        ok, f = cap.read()
                        if ok and f is not None:
                            setattr(self, attr, f)
            time.sleep(0.001)
        with self.lock:
            for cap in (self.capL, self.capR):
                try:
                    if cap: cap.release()
                except Exception:
                    pass
        print("[camera] released.")

    def get_frames(self):
        """
        Returns a matched pair: both frames are grabbed under the same lock
        so they correspond to the same capture window.
        """
        with self.lock:
            L = None if self.lastL is None else self.lastL.copy()
            R = None if self.lastR is None else self.lastR.copy()
        return L, R


live = LiveStereo()
live.start()

# ──────────────────────────────────────────────────────────────
# SGBM + WLS params
# ──────────────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    "minDisparity":       0,
    "numDisparities":     160,   # from stereo_calibrate.py report
    "blockSize":          3,
    "disp12MaxDiff":      1,
    "uniquenessRatio":    5,
    "speckleWindowSize":  200,
    "speckleRange":       4,
    "preFilterCap":       63,
    "mode":               "SGBM_3WAY",
    "useWLS":             1,
    "wlsLambda":          8000,
    "wlsSigma":           1.5,
    "claheClipLimit":     2,
    "claheGrid":          8,
    "swapCameras":        0,
}

MODE_MAP = {
    "SGBM":      cv2.STEREO_SGBM_MODE_SGBM,
    "SGBM_3WAY": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
}

params_lock = threading.Lock()
params: dict = {}

def load_params():
    global params
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            data = yaml.safe_load(f) or {}
        params = {**DEFAULT_PARAMS, **data}
    else:
        params = dict(DEFAULT_PARAMS)
        save_params()

def save_params():
    with open(PARAMS_FILE, "w") as f:
        yaml.safe_dump(params, f, sort_keys=False)

def sanitize(p: dict) -> dict:
    out = dict(p)
    nd = max(16, int(out.get("numDisparities", 160)))
    out["numDisparities"] = (nd + 15) // 16 * 16
    bs = max(3, int(out.get("blockSize", 3)))
    out["blockSize"] = bs if bs % 2 == 1 else bs + 1
    m = str(out.get("mode", "SGBM_3WAY"))
    out["mode"] = m if m in MODE_MAP else "SGBM_3WAY"
    def cl(k, lo, hi, df):
        out[k] = int(np.clip(int(out.get(k, df)), lo, hi))
    cl("uniquenessRatio",    0, 50,  5)
    cl("speckleWindowSize",  0, 400, 200)
    cl("speckleRange",       0, 50,  4)
    cl("disp12MaxDiff",     -1, 50,  1)
    cl("preFilterCap",       1, 63,  63)
    cl("minDisparity",    -100, 200, 0)
    cl("useWLS",             0, 1,   1)
    cl("claheClipLimit",     0, 40,  2)
    cl("claheGrid",          1, 32,  8)
    cl("swapCameras",        0, 1,   0)
    out["wlsLambda"] = float(np.clip(float(out.get("wlsLambda", 8000)), 100, 1e5))
    out["wlsSigma"]  = float(np.clip(float(out.get("wlsSigma",  1.5)),  0.1, 5.0))
    return out

def make_matchers(p: dict):
    bs = p["blockSize"]
    lm = cv2.StereoSGBM_create(
        minDisparity      = p["minDisparity"],
        numDisparities    = p["numDisparities"],
        blockSize         = bs,
        P1 = 8  * bs * bs,
        P2 = 32 * bs * bs,
        disp12MaxDiff     = p["disp12MaxDiff"],
        uniquenessRatio   = p["uniquenessRatio"],
        speckleWindowSize = p["speckleWindowSize"],
        speckleRange      = p["speckleRange"],
        preFilterCap      = p["preFilterCap"],
        mode              = MODE_MAP[p["mode"]],
    )
    rm  = cv2.ximgproc.createRightMatcher(lm)
    wls = None
    if int(p.get("useWLS", 1)):
        wls = cv2.ximgproc.createDisparityWLSFilter(lm)
        wls.setLambda(float(p["wlsLambda"]))
        wls.setSigmaColor(float(p["wlsSigma"]))
    return lm, rm, wls

load_params()
params = sanitize(params)
save_params()

matcher_lock = threading.Lock()
left_m, right_m, wls_f = make_matchers(params)

def update_matchers(new_p: dict):
    global left_m, right_m, wls_f, params
    new_p = sanitize(new_p)
    with params_lock:
        params = new_p
        save_params()
    with matcher_lock:
        left_m, right_m, wls_f = make_matchers(params)

# ──────────────────────────────────────────────────────────────
# Image processing
# ──────────────────────────────────────────────────────────────
def apply_clahe(gray, clip, grid):
    if clip <= 0:
        return gray
    return cv2.createCLAHE(clipLimit=clip,
                            tileGridSize=(grid, grid)).apply(gray)

def compute_disparity(rectL, rectR, p):
    """
    Compute disparity once. Returns float32 pixels, NaN = invalid.
    This is the single source of truth for both disp_vis and depth.
    """
    clip = float(p.get("claheClipLimit", 2))
    grid = int(p.get("claheGrid", 8))
    gL = apply_clahe(cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY), clip, grid)
    gR = apply_clahe(cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY), clip, grid)
    with matcher_lock:
        lm, rm, wls = left_m, right_m, wls_f
        dL = lm.compute(gL, gR)
        if wls is not None:
            dR  = rm.compute(gR, gL)
            d16 = wls.filter(dL, rectL, None, dR)
        else:
            d16 = dL
    disp = d16.astype(np.float32) / 16.0
    disp[disp < float(p.get("minDisparity", 0)) + 0.5] = np.nan
    return disp

def disp_to_vis(disp):
    valid = np.isfinite(disp)
    if not np.any(valid):
        return np.zeros((*disp.shape, 3), np.uint8)
    vmin = np.nanpercentile(disp, 2)
    vmax = np.nanpercentile(disp, 98)
    if vmax <= vmin: vmax = vmin + 1.0
    u8  = (np.clip((disp - vmin)/(vmax - vmin), 0, 1)*255).astype(np.uint8)
    out = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    out[~valid] = 0
    return out

def depth_from_disp(disp, Q, sgbm_scale):
    """
    Convert disparity → depth (metres).

    sgbm_scale: SGBM was run at 1/sgbm_scale resolution, so its disparity
    values are smaller by that factor.  Multiply before reprojectImageTo3D
    so the full-resolution Q matrix gives correct metric Z values.

    Example: capture 1280x720, SGBM_DOWNSCALE=2 → SGBM at 640x360 →
    disparity values are half of what full-res SGBM would give →
    multiply by 2 → Q gives correct Z.
    """
    if Q is None:
        return None
    valid = np.isfinite(disp) & (disp > 0.5)
    d_scaled = np.nan_to_num(disp * sgbm_scale, nan=0.0).astype(np.float32)
    pts = cv2.reprojectImageTo3D(d_scaled, Q)
    Z = pts[:, :, 2].astype(np.float32)
    # Auto-fix sign: valid Z should be positive (objects in front of cameras)
    zv = Z[valid]
    if zv.size > 100 and float(np.median(zv)) < 0:
        Z = -Z
    Z[~valid] = np.nan
    Z[Z > 20.0] = np.nan   # clip > 20m (indoor noise)
    Z[Z < 0.05] = np.nan   # clip < 5cm (noise)
    return Z

def depth_to_vis(Z, disp):
    if Z is None:
        return disp_to_vis(np.where(disp > 0.5, 1.0/disp, np.nan))
    valid = np.isfinite(Z)
    if not np.any(valid):
        return np.zeros((*Z.shape, 3), np.uint8)
    zmin = np.nanpercentile(Z, 2)
    zmax = np.nanpercentile(Z, 98)
    if zmax <= zmin: zmax = zmin + 0.1
    norm = 1.0 - np.clip((Z - zmin)/(zmax - zmin), 0, 1)
    out  = cv2.applyColorMap((norm*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[~valid] = 0
    return out

def to_web(img):
    """Resize to web display resolution."""
    if img.shape[1] == WEB_W and img.shape[0] == WEB_H:
        return img
    return cv2.resize(img, (WEB_W, WEB_H), interpolation=cv2.INTER_AREA)

def overlay(img, lines):
    out = img.copy()
    y = 22
    for line in lines:
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (255,255,255), 1, cv2.LINE_AA)
        y += 20
    return out

def draw_epipolar(img, step=LINE_STEP):
    out = img.copy()
    for y in range(0, out.shape[0], step):
        cv2.line(out, (0, y), (out.shape[1], y), (0,255,0), 1)
    return out

def encode_jpeg(img, q=JPEG_Q):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok: raise RuntimeError("JPEG failed")
    return buf.tobytes()

# ──────────────────────────────────────────────────────────────
# Background compute loop
# ──────────────────────────────────────────────────────────────
frame_lock = threading.Lock()
latest_rect_jpg  = None
latest_disp_jpg  = None
latest_depth_jpg = None

diag_lock   = threading.Lock()
latest_diag = {}


def compute_loop():
    global latest_rect_jpg, latest_disp_jpg, latest_depth_jpg
    period = 1.0 / max(0.5, COMPUTE_HZ)

    while not STOP.is_set():
        t0 = time.time()

        # Grab both frames atomically — same lock acquisition
        L, R = live.get_frames()
        if L is None or R is None:
            time.sleep(0.05)
            continue

        with params_lock:
            p = dict(params)
        swap = int(p.get("swapCameras", 0))

        try:
            # ── 1. Rectify at full capture resolution ─────────
            if swap:
                rectL, rectR = rectifier.rectify_pair(R, L)
            else:
                rectL, rectR = rectifier.rectify_pair(L, R)

            # ── 2. Downscale for SGBM ─────────────────────────
            sd = SGBM_DOWNSCALE
            if sd > 1:
                h_s = rectL.shape[0] // sd
                w_s = rectL.shape[1] // sd
                sgbmL = cv2.resize(rectL, (w_s, h_s), interpolation=cv2.INTER_AREA)
                sgbmR = cv2.resize(rectR, (w_s, h_s), interpolation=cv2.INTER_AREA)
            else:
                sgbmL, sgbmR = rectL, rectR

            # ── 3. Compute disparity ONCE ─────────────────────
            # This single result feeds both the disparity visualisation
            # and the depth map — no second SGBM call, no flickering.
            disp = compute_disparity(sgbmL, sgbmR, p)

            # ── 4. Depth — scale disparity back to full-res ───
            # disp values are at 1/sd of full resolution.
            # Multiplying by sd corrects them before passing to Q.
            Z = depth_from_disp(disp, rectifier.Q, sgbm_scale=sd)

            # ── 5. Build visualisations ───────────────────────
            pct      = 100.0 * np.sum(np.isfinite(disp)) / disp.size
            med_disp = float(np.nanmedian(disp))

            # Rectified preview
            rect_web = to_web(np.hstack([rectL, rectR]))
            rect_web = draw_epipolar(rect_web)

            # Disparity vis
            dvis = to_web(disp_to_vis(disp))
            dvis = overlay(dvis, [
                f"valid:{pct:.0f}%  [{p['minDisparity']},{p['minDisparity']+p['numDisparities']}]px  "
                f"scale÷{sd}",
                f"median:{med_disp:.1f}px  swap={swap}",
            ])

            # Depth vis + annotation
            zvis = to_web(depth_to_vis(Z, disp))
            if Z is not None and np.any(np.isfinite(Z)):
                zmed = float(np.nanmedian(Z))
                z5   = float(np.nanpercentile(Z, 5))
                z95  = float(np.nanpercentile(Z, 95))
                # Cross-check: focal length at SGBM resolution × baseline / median disparity
                f_sgbm   = rectifier.focal_px / sd
                z_check  = (f_sgbm * rectifier.baseline_m / med_disp
                            if med_disp > 0 else 0.0)
                zvis = overlay(zvis, [
                    f"depth {z5:.2f}–{z95:.2f}m   median:{zmed:.2f}m",
                    f"check f*B/d={z_check:.2f}m",
                ])
                with diag_lock:
                    latest_diag = {
                        "pct": pct, "med_disp": med_disp,
                        "z_med": zmed, "z_check": z_check,
                        "nd": p["numDisparities"],
                        "suggested_nd": rectifier.suggested_nd,
                        "ok": abs(zmed - z_check) < zmed * 0.35,
                    }
            else:
                zvis = overlay(zvis, [
                    "no depth — check numDisparities",
                    f"suggested: numDisparities≥{rectifier.suggested_nd}",
                ])

            with frame_lock:
                latest_rect_jpg  = encode_jpeg(rect_web)
                latest_disp_jpg  = encode_jpeg(dvis)
                latest_depth_jpg = encode_jpeg(zvis)

        except Exception:
            import traceback; traceback.print_exc()

        slp = period - (time.time() - t0)
        if slp > 0:
            time.sleep(slp)


threading.Thread(target=compute_loop, daemon=True).start()

# ──────────────────────────────────────────────────────────────
# Flask
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)

SGBM_KEYS = ["minDisparity","numDisparities","blockSize","disp12MaxDiff",
             "uniquenessRatio","speckleWindowSize","speckleRange","preFilterCap","mode"]
WLS_KEYS  = ["useWLS","wlsLambda","wlsSigma"]
PRE_KEYS  = ["claheClipLimit","claheGrid","swapCameras"]
ALL_KEYS  = SGBM_KEYS + WLS_KEYS + PRE_KEYS

HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Stereo Depth Web</title>
  <style>
    *,*::before,*::after{box-sizing:border-box}
    body{font-family:system-ui,sans-serif;margin:0;padding:16px;background:#f0f2f5}
    .wrap{max-width:1440px;margin:0 auto}
    h2{margin:0 0 2px}
    .sub{color:#555;font-size:12px;margin:0 0 10px;line-height:1.8;
         background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:8px 12px;
         font-family:monospace}
    code{background:#e8e8e8;padding:2px 5px;border-radius:4px;font-size:11px}
    .card{background:#fff;border:1px solid #ddd;border-radius:12px;padding:14px;margin-bottom:14px}
    img{width:100%;height:auto;border-radius:8px;background:#111;display:block}
    .grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
    @media(max-width:960px){.grid2{grid-template-columns:1fr}}
    h3{margin:0 0 8px;font-size:15px}
    .row{display:flex;flex-wrap:wrap;gap:8px;align-items:flex-end}
    label{display:block;font-size:11px;color:#555;margin-bottom:3px}
    input,select{padding:6px 8px;font-size:13px;border:1px solid #ccc;
                 border-radius:6px;width:120px}
    button{padding:9px 18px;font-size:14px;background:#0052cc;color:#fff;
           border:none;border-radius:8px;cursor:pointer}
    button:hover{background:#0041a3}
    .sec{font-size:11px;font-weight:700;color:#888;text-transform:uppercase;
         letter-spacing:.06em;border-top:1px solid #eee;padding-top:10px;margin:12px 0 6px}
    .hint{color:#888;font-size:11px;margin:10px 0 0;line-height:1.7}
    .badge{background:#e8f0fe;color:#1a73e8;border-radius:4px;font-size:11px;
           padding:2px 6px;margin-left:5px}
    .diag-ok  {background:#f6ffed;border:1px solid #b7eb8f;border-radius:8px;
               padding:10px 14px;font-size:12px;margin-bottom:10px}
    .diag-warn{background:#fff7e6;border:1px solid #ffd591;border-radius:8px;
               padding:10px 14px;font-size:12px;margin-bottom:10px}
    .tip{background:#e6f7ff;border:1px solid #91d5ff;border-radius:8px;
         padding:10px 14px;font-size:12px;margin-bottom:14px;line-height:1.8}
  </style>
</head>
<body>
<div class="wrap">
  <h2>Stereo Depth Web <span class="badge">SGBM+WLS</span></h2>
  <div class="sub">
    📷 {{capture}}  SGBM÷{{sgbm_scale}}={{sgbm_res}}  web={{web_res}}<br>
    🔬 {{calib_info}}
  </div>

  {% if diag %}
  <div class="{{ 'diag-ok' if diag.ok else 'diag-warn' }}">
    {% if diag.ok %}
    ✅ Depth consistent — median <b>{{diag.z_med|round(2)}}m</b>,
       formula check <b>{{diag.z_check|round(2)}}m</b>
       (valid {{diag.pct|round(0)|int}}%, disp median {{diag.med_disp|round(1)}}px)
    {% else %}
    ⚠️  Mismatch — Q depth <b>{{diag.z_med|round(2)}}m</b> vs
       formula <b>{{diag.z_check|round(2)}}m</b>.
       Try re-running <code>stereo_calibrate.py</code>.
       numDisparities={{diag.nd}}, suggested≥{{diag.suggested_nd}}.
    {% endif %}
  </div>
  {% endif %}

  <div class="tip">
    <b>Tuning:</b>
    <b>Black bar left</b> = normal occluded zone (~numDisparities px wide). &nbsp;
    <b>Black bar right / inverted depth</b> → swapCameras=1. &nbsp;
    <b>valid% low</b> → lower uniquenessRatio, raise wlsLambda. &nbsp;
    <b>Still flickering</b> → this version computes disparity once per frame,
    so flickering now means the scene has insufficient texture for stable matching
    (floors, plain walls) — try raising wlsLambda to 15000.
  </div>

  <div class="card">
    <h3>Rectified — green lines must align across both halves</h3>
    <img src="/view/rectified"/>
  </div>
  <div class="grid2">
    <div class="card">
      <h3>Disparity <span class="badge">red=near</span></h3>
      <img src="/view/disparity"/>
    </div>
    <div class="card">
      <h3>Depth <span class="badge">warm=near · cool=far</span></h3>
      <img src="/view/depth"/>
    </div>
  </div>

  <div class="card">
    <h3>Parameters</h3>
    <form action="/apply" method="post">
      <div class="sec">Core SGBM</div>
      <div class="row">
        {% for k in sgbm_keys %}
        <div>
          <label>{{k}}</label>
          {% if k == "mode" %}
          <select name="{{k}}">
            <option value="SGBM"      {% if p[k]=="SGBM"      %}selected{% endif %}>SGBM</option>
            <option value="SGBM_3WAY" {% if p[k]=="SGBM_3WAY" %}selected{% endif %}>SGBM_3WAY</option>
          </select>
          {% else %}
          <input name="{{k}}" value="{{p[k]}}"/>
          {% endif %}
        </div>
        {% endfor %}
      </div>
      <div class="sec">WLS Filter</div>
      <div class="row">
        {% for k in wls_keys %}
        <div>
          <label>{{k}}</label>
          {% if k == "useWLS" %}
          <select name="{{k}}">
            <option value="1" {% if p[k]|int==1 %}selected{% endif %}>On</option>
            <option value="0" {% if p[k]|int==0 %}selected{% endif %}>Off</option>
          </select>
          {% else %}
          <input name="{{k}}" value="{{p[k]}}"/>
          {% endif %}
        </div>
        {% endfor %}
      </div>
      <div class="sec">Preprocessing &amp; Camera</div>
      <div class="row">
        {% for k in pre_keys %}
        <div>
          <label>{{k}}</label>
          {% if k == "swapCameras" %}
          <select name="swapCameras">
            <option value="0" {% if p['swapCameras']|int==0 %}selected{% endif %}>0-normal</option>
            <option value="1" {% if p['swapCameras']|int==1 %}selected{% endif %}>1-swap</option>
          </select>
          {% else %}
          <input name="{{k}}" value="{{p[k]}}"/>
          {% endif %}
        </div>
        {% endfor %}
      </div>
      <div class="row" style="margin-top:14px">
        <button type="submit">▶ Apply + Save</button>
        <a href="/params.yaml" style="font-size:13px;align-self:center">⬇ params.yaml</a>
      </div>
      <p class="hint">
        numDisparities: ×16 rounded, see calib info above for min value &nbsp;·&nbsp;
        blockSize: odd only &nbsp;·&nbsp;
        wlsLambda: 1k–20k (higher=smoother holes) &nbsp;·&nbsp;
        wlsSigma: 0.8–2.5 (edge sensitivity) &nbsp;·&nbsp;
        CLAHE clip: 0=off
      </p>
    </form>
  </div>
</div>
</body>
</html>
"""

@app.route("/")
def index():
    with params_lock:
        p = dict(params)
    with diag_lock:
        diag = dict(latest_diag)
    return render_template_string(HTML,
        calib_info = rectifier.calib_info,
        capture    = f"{CAPTURE_W}×{CAPTURE_H}",
        sgbm_res   = f"{CAPTURE_W//SGBM_DOWNSCALE}×{CAPTURE_H//SGBM_DOWNSCALE}",
        sgbm_scale = SGBM_DOWNSCALE,
        web_res    = f"{WEB_W}×{WEB_H}",
        p=p, diag=diag,
        sgbm_keys=SGBM_KEYS,
        wls_keys=WLS_KEYS,
        pre_keys=PRE_KEYS,
    )

@app.route("/apply", methods=["POST"])
def apply():
    with params_lock:
        newp = dict(params)
    for k in ALL_KEYS:
        v = request.form.get(k)
        if v is None: continue
        if k == "mode":
            newp[k] = v
        else:
            try:
                newp[k] = float(v) if "." in v else int(v)
            except Exception:
                pass
    update_matchers(newp)
    return redirect(url_for("index"))

@app.route("/params.yaml")
def params_yaml():
    if not os.path.exists(PARAMS_FILE):
        return "no params file", 404
    return send_file(PARAMS_FILE, as_attachment=True)

def mjpeg_stream(which):
    while not STOP.is_set():
        with frame_lock:
            data = {"rectified": latest_rect_jpg,
                    "disparity": latest_disp_jpg,
                    "depth":     latest_depth_jpg}.get(which)
        if data is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
               + f"Content-Length: {len(data)}\r\n\r\n".encode()
               + data + b"\r\n")
        time.sleep(0.04)

@app.route("/view/rectified")
def view_rectified():
    return Response(mjpeg_stream("rectified"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
@app.route("/view/disparity")
def view_disparity():
    return Response(mjpeg_stream("disparity"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
@app.route("/view/depth")
def view_depth():
    return Response(mjpeg_stream("depth"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print(f"[server] http://{HOST}:{PORT}   (Ctrl+C to stop)")
    app.run(host=HOST, port=PORT, threaded=True,
            use_reloader=False, debug=False)
    STOP.set()
