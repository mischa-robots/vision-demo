#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mischa Schaefer <https://github.com/mischa-robots/vision-demo>

"""
rectify_web_check.py  (headless)
- Serves a small web UI to inspect stereo rectification results in a browser.
- Loads stereo_calibration.yaml produced by your stereo_calibrate.py
- Lets you pick any captured pair (captures/left_*.png + right_*.png)
- Generates a rectified side-by-side image with horizontal epipolar lines
- Optional: "live rectified" MJPEG streams (left/right and combined) from CSI cams

Run:
  sudo apt-get install -y python3-flask python3-opencv
  python3 rectify_web_check.py

Open:
  http://<JETSON_IP>:8010
"""

import os
import io
import time
import glob
import threading
from dataclasses import dataclass

import cv2
import numpy as np
from flask import Flask, Response, request, redirect, url_for, send_file, abort, render_template_string

# ===================== Config =====================
HOST = "0.0.0.0"
PORT = 8010

CAPTURE_DIR = "captures"
CALIB_FILE = "stereo_calibration.yaml"

# Camera (optional live mode)
ENABLE_LIVE = True
LEFT_ID = 0
RIGHT_ID = 1
WIDTH = 1280
HEIGHT = 720
FPS = 30

# Rectify overlay
LINE_STEP = 40
JPEG_Q = 85

# ===================== GStreamer pipeline (CSI/Argus) =====================
def gst_pipeline(sensor_id: int, width: int, height: int, fps: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1,format=NV12 ! "
        f"nvvidconv ! video/x-raw,format=BGRx,width={width},height={height} ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

# ===================== Calibration loading + maps =====================
class Rectifier:
    def __init__(self, calib_path: str):
        self.calib_path = calib_path
        self._loaded = False
        self._maps_ready = False
        self._lock = threading.Lock()

        self.mtxL = self.distL = self.mtxR = self.distR = None
        self.R1 = self.R2 = self.P1 = self.P2 = None
        self.mapLx = self.mapLy = self.mapRx = self.mapRy = None
        self.image_size = None  # (w,h)

    def _read_mat(self, fs, name: str):
        node = fs.getNode(name)
        if node.empty():
            raise RuntimeError(f"Missing '{name}' in {self.calib_path}")
        return node.mat()

    def load(self):
        with self._lock:
            fs = cv2.FileStorage(self.calib_path, cv2.FILE_STORAGE_READ)
            if not fs.isOpened():
                raise RuntimeError(f"Could not open {self.calib_path}")

            self.mtxL = self._read_mat(fs, "mtxL")
            self.distL = self._read_mat(fs, "distL")
            self.mtxR = self._read_mat(fs, "mtxR")
            self.distR = self._read_mat(fs, "distR")
            self.R1 = self._read_mat(fs, "R1")
            self.R2 = self._read_mat(fs, "R2")
            self.P1 = self._read_mat(fs, "P1")
            self.P2 = self._read_mat(fs, "P2")
            fs.release()
            self._loaded = True
            self._maps_ready = False  # will be created when image size known

    def ensure_maps(self, w: int, h: int):
        with self._lock:
            if not self._loaded:
                self.load()

            if self._maps_ready and self.image_size == (w, h):
                return

            # initUndistortRectifyMap wants (w,h)
            self.mapLx, self.mapLy = cv2.initUndistortRectifyMap(
                self.mtxL, self.distL, self.R1, self.P1, (w, h), cv2.CV_16SC2
            )
            self.mapRx, self.mapRy = cv2.initUndistortRectifyMap(
                self.mtxR, self.distR, self.R2, self.P2, (w, h), cv2.CV_16SC2
            )
            self.image_size = (w, h)
            self._maps_ready = True

    def rectify_pair(self, imgL: np.ndarray, imgR: np.ndarray):
        h, w = imgL.shape[:2]
        self.ensure_maps(w, h)
        rectL = cv2.remap(imgL, self.mapLx, self.mapLy, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, self.mapRx, self.mapRy, cv2.INTER_LINEAR)
        return rectL, rectR

rectifier = Rectifier(CALIB_FILE)

# ===================== File pair handling =====================
def list_pairs():
    lefts = sorted(glob.glob(os.path.join(CAPTURE_DIR, "left_*.png")))
    pairs = []
    for lp in lefts:
        base = os.path.basename(lp).replace("left_", "").replace(".png", "")
        rp = os.path.join(CAPTURE_DIR, f"right_{base}.png")
        if os.path.exists(rp):
            pairs.append((base, lp, rp))
    return pairs[::-1]  # newest first

def draw_epipolar_lines(side_by_side_bgr: np.ndarray, step=LINE_STEP):
    out = side_by_side_bgr.copy()
    h = out.shape[0]
    for y in range(0, h, step):
        cv2.line(out, (0, y), (out.shape[1], y), (0, 255, 0), 1)
    return out

def encode_jpeg(img_bgr: np.ndarray, quality=JPEG_Q) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()

# ===================== Optional live capture =====================
@dataclass
class CamState:
    cap: cv2.VideoCapture
    ok: bool = False
    last_bgr: np.ndarray | None = None
    last_ts: float = 0.0

class LiveStereo:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.left = self.right = None
        self.thread = None

    def start(self):
        if not ENABLE_LIVE:
            return
        with self.lock:
            if self.running:
                return
            capL = cv2.VideoCapture(gst_pipeline(LEFT_ID, WIDTH, HEIGHT, FPS), cv2.CAP_GSTREAMER)
            capR = cv2.VideoCapture(gst_pipeline(RIGHT_ID, WIDTH, HEIGHT, FPS), cv2.CAP_GSTREAMER)
            self.left = CamState(cap=capL, ok=capL.isOpened())
            self.right = CamState(cap=capR, ok=capR.isOpened())
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _read_one(self, cs: CamState):
        if not cs.ok:
            return
        ok, frame = cs.cap.read()
        if ok and frame is not None:
            cs.last_bgr = frame
            cs.last_ts = time.time()

    def _loop(self):
        while True:
            with self.lock:
                if not self.running:
                    break
                self._read_one(self.left)
                self._read_one(self.right)
            time.sleep(0.001)

    def get_frames(self):
        with self.lock:
            if not self.running:
                return None, None
            L = None if self.left.last_bgr is None else self.left.last_bgr.copy()
            R = None if self.right.last_bgr is None else self.right.last_bgr.copy()
            return L, R

    def stop(self):
        with self.lock:
            self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        for cs in (self.left, self.right):
            if cs and cs.cap:
                try:
                    cs.cap.release()
                except Exception:
                    pass

live = LiveStereo()
if ENABLE_LIVE:
    # Start automatically; if cameras are busy, it will just show "not available".
    live.start()

# ===================== Flask app =====================
app = Flask(__name__)

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Rectify Web Check</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 18px; }
    .row { display:flex; gap:14px; flex-wrap:wrap; align-items:flex-start; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px; }
    img { max-width: 100%; height: auto; border-radius: 8px; background:#111; }
    select, button { padding: 8px 10px; font-size: 14px; }
    .small { color:#666; font-size: 13px; }
    code { background:#f6f6f6; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <h2>Rectify Web Check</h2>
  <p class="small">
    Uses <code>{{calib}}</code> and pairs from <code>{{dir}}</code>.
    {% if live_enabled %}
      Live mode: <b>enabled</b>.
    {% else %}
      Live mode: <b>disabled</b>.
    {% endif %}
  </p>

  {% if msg %}
    <p><b>{{msg}}</b></p>
  {% endif %}

  <div class="card" style="margin-bottom:14px;">
    <form action="/set" method="post" class="row">
      <div>
        <b>Select captured pair</b><br/>
        <select name="pair" style="min-width:260px;">
          {% for base in bases %}
            <option value="{{base}}" {% if base == selected %}selected{% endif %}>{{base}}</option>
          {% endfor %}
        </select>
      </div>
      <div style="align-self:flex-end;">
        <button type="submit">Show rectified</button>
      </div>
      <div style="align-self:flex-end;">
        <a href="/rectified.jpg?pair={{selected}}">Open rectified.jpg</a>
      </div>
    </form>
    <p class="small" style="margin-top:10px;">
      Tip: in the rectified image, the same features should lie on the same green horizontal lines in left & right.
    </p>
  </div>

  <div class="row">
    <div class="card" style="flex: 1 1 560px;">
      <h3>Rectified (pair {{selected}})</h3>
      <img src="/rectified.jpg?pair={{selected}}&t={{ts}}" />
    </div>
  </div>

  {% if live_enabled %}
  <div class="row" style="margin-top:14px;">
    <div class="card">
      <h3>Live rectified (left/right)</h3>
      <p class="small">This is MJPEG. If cameras are busy, it will not update.</p>
      <div class="row">
        <div>
          <b>Left rectified</b><br/>
          <img src="/live/left" style="width:520px;" />
        </div>
        <div>
          <b>Right rectified</b><br/>
          <img src="/live/right" style="width:520px;" />
        </div>
      </div>
    </div>
    <div class="card" style="flex: 1 1 560px;">
      <h3>Live rectified (combined)</h3>
      <img src="/live/combined" style="width:1060px; max-width:100%;" />
    </div>
  </div>
  {% endif %}

  <div class="card" style="margin-top:14px;">
    <h3>Downloads</h3>
    <ul>
      <li><a href="/download_pair.zip?pair={{selected}}">Download selected pair (raw) as ZIP</a></li>
      <li><a href="/download_rectified.png?pair={{selected}}">Download rectified image as PNG</a></li>
    </ul>
  </div>
</body>
</html>
"""

_selected_pair = None

@app.route("/")
def index():
    global _selected_pair
    pairs = list_pairs()
    bases = [b for (b, _, _) in pairs]
    if not bases:
        return render_template_string(
            "<h3>No pairs found in {{dir}}</h3><p>Expected left_*.png and right_*.png</p>",
            dir=CAPTURE_DIR,
        )
    if _selected_pair not in bases:
        _selected_pair = bases[0]

    return render_template_string(
        HTML,
        calib=CALIB_FILE,
        dir=CAPTURE_DIR,
        bases=bases,
        selected=_selected_pair,
        msg=request.args.get("msg", ""),
        ts=int(time.time()),
        live_enabled=ENABLE_LIVE,
    )

@app.route("/set", methods=["POST"])
def set_pair():
    global _selected_pair
    pair = request.form.get("pair", "")
    if not pair or not pair.isdigit():
        return redirect(url_for("index", msg="Bad pair id"))
    _selected_pair = pair
    return redirect(url_for("index"))

def load_pair_images(base: str):
    lp = os.path.join(CAPTURE_DIR, f"left_{base}.png")
    rp = os.path.join(CAPTURE_DIR, f"right_{base}.png")
    if not os.path.exists(lp) or not os.path.exists(rp):
        raise FileNotFoundError("pair not found")
    imgL = cv2.imread(lp, cv2.IMREAD_COLOR)
    imgR = cv2.imread(rp, cv2.IMREAD_COLOR)
    if imgL is None or imgR is None:
        raise RuntimeError("failed to read images")
    return imgL, imgR, lp, rp

@app.route("/rectified.jpg")
def rectified_jpg():
    base = request.args.get("pair", "")
    if not base.isdigit():
        abort(400)
    try:
        imgL, imgR, _, _ = load_pair_images(base)
        rectL, rectR = rectifier.rectify_pair(imgL, imgR)
        out = np.hstack([rectL, rectR])
        out = draw_epipolar_lines(out, LINE_STEP)
        data = encode_jpeg(out, quality=JPEG_Q)
        return Response(data, mimetype="image/jpeg")
    except Exception as e:
        return Response(f"rectify failed: {e}\n", mimetype="text/plain", status=500)

@app.route("/download_rectified.png")
def download_rectified_png():
    base = request.args.get("pair", "")
    if not base.isdigit():
        abort(400)
    imgL, imgR, _, _ = load_pair_images(base)
    rectL, rectR = rectifier.rectify_pair(imgL, imgR)
    out = np.hstack([rectL, rectR])
    out = draw_epipolar_lines(out, LINE_STEP)
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        abort(500)
    return send_file(
        io.BytesIO(buf.tobytes()),
        mimetype="image/png",
        as_attachment=True,
        download_name=f"rectified_{base}.png",
    )

@app.route("/download_pair.zip")
def download_pair_zip():
    base = request.args.get("pair", "")
    if not base.isdigit():
        abort(400)
    imgL_path = os.path.join(CAPTURE_DIR, f"left_{base}.png")
    imgR_path = os.path.join(CAPTURE_DIR, f"right_{base}.png")
    if not os.path.exists(imgL_path) or not os.path.exists(imgR_path):
        abort(404)

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(imgL_path, arcname=os.path.basename(imgL_path))
        z.write(imgR_path, arcname=os.path.basename(imgR_path))
    mem.seek(0)
    return send_file(
        mem,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"pair_{base}.zip",
    )

# ---------------- Live rectified MJPEG ----------------
def mjpeg(boundary=b"--frame"):
    return boundary

def stream_mjpeg(get_frame_bytes_fn):
    boundary = b"--frame"
    while True:
        try:
            data = get_frame_bytes_fn()
            if data is None:
                time.sleep(0.05)
                continue
            yield boundary + b"\r\n"
            yield b"Content-Type: image/jpeg\r\n"
            yield f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
            yield data + b"\r\n"
            time.sleep(0.02)
        except GeneratorExit:
            return
        except Exception:
            time.sleep(0.1)

def live_rectified(which: str) -> bytes | None:
    L, R = live.get_frames()
    if L is None or R is None:
        return None
    rectL, rectR = rectifier.rectify_pair(L, R)

    if which == "left":
        img = rectL
    elif which == "right":
        img = rectR
    elif which == "combined":
        img = draw_epipolar_lines(np.hstack([rectL, rectR]), LINE_STEP)
    else:
        return None

    return encode_jpeg(img, quality=JPEG_Q)

@app.route("/live/<which>")
def live_stream(which):
    if not ENABLE_LIVE:
        return "live mode disabled", 404
    if which not in ("left", "right", "combined"):
        return "bad stream", 400
    return Response(
        stream_mjpeg(lambda: live_rectified(which)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

# ---------------- Simple file route (optional) ----------------
@app.route("/file/<name>")
def file(name):
    path = os.path.join(CAPTURE_DIR, name)
    if not os.path.isfile(path):
        return "not found", 404
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    try:
        # Preload calibration to fail fast (optional)
        rectifier.load()
        print(f"Loaded calibration: {CALIB_FILE}")
    except Exception as e:
        print(f"WARNING: calibration not loaded yet: {e}")

    try:
        app.run(host=HOST, port=PORT, threaded=True)
    finally:
        try:
            live.stop()
        except Exception:
            pass
