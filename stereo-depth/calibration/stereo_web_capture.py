#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mischa Schaefer <https://github.com/mischa-robots/vision-demo>

import os
import io
import time
import glob
import zipfile
import threading
from dataclasses import dataclass

import cv2
import numpy as np
from flask import Flask, Response, request, redirect, url_for, send_file, abort

# ===== Config =====
HOST = "0.0.0.0"
PORT = 8000

WIDTH = 1280
HEIGHT = 720
FPS = 30

CAPTURE_DIR = "captures"
LEFT_ID = 0
RIGHT_ID = 1

EXT = "png"          # calibration-friendly
THUMB_W = 240        # gallery thumbs
JPEG_Q = 80          # preview stream

# ===== GStreamer pipeline for CSI (Argus) =====
def gst_pipeline(sensor_id: int, width: int, height: int, fps: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1,format=NV12 ! "
        f"nvvidconv ! video/x-raw,format=BGRx,width={width},height={height} ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

@dataclass
class CamState:
    cap: cv2.VideoCapture
    last_bgr: np.ndarray | None = None
    last_ts: float = 0.0
    ok: bool = False
    cb_found: bool = False
    cb_corners: np.ndarray | None = None  # Nx1x2

class StereoServer:
    def __init__(self):
        self.lock = threading.Lock()
        self.left = self._open_cam(LEFT_ID)
        self.right = self._open_cam(RIGHT_ID)

        # Default chessboard inner corners (adjust in UI)
        self.cb_cols = 9
        self.cb_rows = 6
        self.require_cb_for_capture = True

        self.running = True
        self.thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.thread.start()

        os.makedirs(CAPTURE_DIR, exist_ok=True)

    def _open_cam(self, sid: int) -> CamState:
        cap = cv2.VideoCapture(gst_pipeline(sid, WIDTH, HEIGHT, FPS), cv2.CAP_GSTREAMER)
        return CamState(cap=cap, ok=cap.isOpened())

    def set_cb(self, cols: int, rows: int, require: bool):
        with self.lock:
            self.cb_cols = max(2, int(cols))
            self.cb_rows = max(2, int(rows))
            self.require_cb_for_capture = bool(require)

    def _grab_loop(self):
        while self.running:
            with self.lock:
                self._read_one(self.left)
                self._read_one(self.right)
            time.sleep(0.001)

    def _detect_checkerboard(self, bgr: np.ndarray) -> tuple[bool, np.ndarray | None]:
        # Downscale for speed, detect, then scale corners back.
        h, w = bgr.shape[:2]
        scale = 0.5
        small = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        pattern = (self.cb_cols, self.cb_rows)  # (cols, rows) inner corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern, flags)

        if not found or corners is None:
            return False, None

        # Refine corners
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)

        # Scale back to full-res coordinates
        corners_full = corners / scale
        return True, corners_full

    def _read_one(self, cs: CamState):
        if not cs.ok:
            return
        ok, frame = cs.cap.read()
        if ok and frame is not None:
            cs.last_bgr = frame
            cs.last_ts = time.time()
            found, corners = self._detect_checkerboard(frame)
            cs.cb_found = found
            cs.cb_corners = corners

    def _overlay_status(self, bgr: np.ndarray, found: bool, corners: np.ndarray | None) -> np.ndarray:
        out = bgr.copy()
        color = (0, 200, 0) if found else (0, 0, 220)

        label = f"CHECKERBOARD: {'FOUND' if found else 'NOT FOUND'}  (inner corners {self.cb_cols}x{self.cb_rows})"
        cv2.rectangle(out, (10, 10), (10 + 760, 10 + 40), (0, 0, 0), -1)
        cv2.putText(out, label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

        if found and corners is not None:
            pattern = (self.cb_cols, self.cb_rows)
            cv2.drawChessboardCorners(out, pattern, corners, found)

        return out

    def get_jpeg(self, which: str) -> bytes | None:
        with self.lock:
            cs = self.left if which == "left" else self.right
            if cs.last_bgr is None:
                return None
            img = self._overlay_status(cs.last_bgr, cs.cb_found, cs.cb_corners)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
            if not ok:
                return None
            return buf.tobytes()

    def next_index(self) -> int:
        files = sorted(glob.glob(os.path.join(CAPTURE_DIR, f"left_*.{EXT}")))
        if not files:
            return 1
        last = os.path.basename(files[-1])
        num = int(last.split("_")[1].split(".")[0])
        return num + 1

    def capture_pair(self) -> tuple[str, str, float]:
        with self.lock:
            L = None if self.left.last_bgr is None else self.left.last_bgr.copy()
            R = None if self.right.last_bgr is None else self.right.last_bgr.copy()
            tL = self.left.last_ts
            tR = self.right.last_ts
            require = self.require_cb_for_capture
            okL = self.left.cb_found
            okR = self.right.cb_found

        if L is None or R is None:
            raise RuntimeError("No frames available yet (wait 1–2 seconds after start).")

        if require and (not okL or not okR):
            raise RuntimeError("Checkerboard not detected in BOTH cameras (disable 'require' to force capture).")

        idx = self.next_index()
        left_path = os.path.join(CAPTURE_DIR, f"left_{idx:06d}.{EXT}")
        right_path = os.path.join(CAPTURE_DIR, f"right_{idx:06d}.{EXT}")

        if not cv2.imwrite(left_path, L):
            raise RuntimeError("Failed to write left image.")
        if not cv2.imwrite(right_path, R):
            raise RuntimeError("Failed to write right image.")

        dt_ms = abs(tL - tR) * 1000.0
        return left_path, right_path, dt_ms

    def list_pairs(self):
        lefts = sorted(glob.glob(os.path.join(CAPTURE_DIR, f"left_*.{EXT}")))
        pairs = []
        for lp in lefts:
            base = os.path.basename(lp).replace("left_", "").replace(f".{EXT}", "")
            rp = os.path.join(CAPTURE_DIR, f"right_{base}.{EXT}")
            if os.path.exists(rp):
                pairs.append((base, lp, rp))
        return pairs[::-1]  # newest first

    def delete_pair(self, base: str):
        lp = os.path.join(CAPTURE_DIR, f"left_{base}.{EXT}")
        rp = os.path.join(CAPTURE_DIR, f"right_{base}.{EXT}")
        for p in (lp, rp):
            if os.path.exists(p):
                os.remove(p)

    def zip_all(self) -> bytes:
        pairs = self.list_pairs()
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for _, lp, rp in pairs:
                z.write(lp, arcname=os.path.basename(lp))
                z.write(rp, arcname=os.path.basename(rp))
        mem.seek(0)
        return mem.read()

    def make_thumb_jpeg(self, path: str) -> bytes:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        h, w = img.shape[:2]
        scale = THUMB_W / float(w)
        nh = max(1, int(h * scale))
        small = cv2.resize(img, (THUMB_W, nh), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            raise RuntimeError("thumb encode failed")
        return buf.tobytes()

    def shutdown(self):
        self.running = False
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        for cs in (self.left, self.right):
            try:
                cs.cap.release()
            except Exception:
                pass


app = Flask(__name__)
srv = StereoServer()

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Jetson Stereo Capture</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 18px; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px; }
    img.live { width: 560px; max-width: 100%; height: auto; border-radius: 8px; background:#111; }
    button { padding: 10px 14px; font-size: 16px; }
    input { padding: 6px 8px; font-size: 14px; width: 72px; }
    label { margin-right: 10px; }
    .small { color: #666; font-size: 13px; }
    table { border-collapse: collapse; width: 100%; }
    td, th { border-bottom: 1px solid #eee; padding: 8px; text-align: left; vertical-align: middle; }
    a { text-decoration: none; }
    img.thumb { width: 160px; border-radius: 8px; border: 1px solid #eee; }
    .danger { background: #fff0f0; border: 1px solid #f2bcbc; border-radius: 8px; padding: 8px 10px; cursor: pointer; }
  </style>
</head>
<body>
  <h2>Jetson Stereo Capture</h2>
  <p class="small">
    Resolution: {{w}}×{{h}} @ {{fps}} — saving PNG pairs to <code>{{dir}}</code>
  </p>

  <div class="card" style="margin-bottom:12px;">
    <form action="/settings" method="post" class="row" style="gap:16px;">
      <div>
        <b>Checkerboard inner corners</b><br/>
        <label>Cols <input name="cols" value="{{cols}}" /></label>
        <label>Rows <input name="rows" value="{{rows}}" /></label>
        <label>
          <input type="checkbox" name="require" value="1" {% if require %}checked{% endif %}/>
          Require detection to capture
        </label>
      </div>
      <div style="align-self:flex-end;">
        <button type="submit">Apply</button>
      </div>
    </form>
    <p class="small" style="margin-top:8px;">
      Typical boards: 9×6 (10×7 squares) or 7×6 (8×7 squares). Use the number of <i>inner</i> corners.
    </p>
  </div>

  <div class="row">
    <div class="card">
      <h3>Left (sensor-id=0)</h3>
      <img class="live" src="/stream/left" />
    </div>
    <div class="card">
      <h3>Right (sensor-id=1)</h3>
      <img class="live" src="/stream/right" />
    </div>
  </div>

  <div style="margin: 14px 0;">
    <form action="/capture" method="post" style="display:inline;">
      <button type="submit">📸 Capture stereo pair</button>
    </form>
    <a href="/download.zip" style="margin-left: 10px;">⬇️ Download all as ZIP</a>
  </div>

  {% if msg %}
    <p><b>{{msg}}</b></p>
  {% endif %}

  <h3>Captured pairs</h3>
  <table>
    <tr><th>ID</th><th>Preview</th><th>Files</th><th>Action</th></tr>
    {% for base, lp, rp in pairs %}
      <tr>
        <td><code>{{base}}</code></td>
        <td>
          <a href="/file/{{lp|basename}}"><img class="thumb" src="/thumb/{{lp|basename}}" /></a>
          <a href="/file/{{rp|basename}}" style="margin-left:8px;"><img class="thumb" src="/thumb/{{rp|basename}}" /></a>
        </td>
        <td>
          <div><a href="/file/{{lp|basename}}">{{lp|basename}}</a></div>
          <div><a href="/file/{{rp|basename}}">{{rp|basename}}</a></div>
        </td>
        <td>
          <form action="/delete/{{base}}" method="post" onsubmit="return confirm('Delete pair {{base}}?');">
            <button class="danger" type="submit">Delete</button>
          </form>
        </td>
      </tr>
    {% endfor %}
  </table>
</body>
</html>
"""

@app.template_filter("basename")
def _basename(p): return os.path.basename(p)

@app.route("/")
def index():
    msg = request.args.get("msg", "")
    pairs = srv.list_pairs()
    from flask import render_template_string
    return render_template_string(
        HTML,
        w=WIDTH, h=HEIGHT, fps=FPS, dir=CAPTURE_DIR,
        pairs=pairs, msg=msg,
        cols=srv.cb_cols, rows=srv.cb_rows, require=srv.require_cb_for_capture
    )

@app.route("/settings", methods=["POST"])
def settings():
    try:
        cols = int(request.form.get("cols", "9"))
        rows = int(request.form.get("rows", "6"))
    except ValueError:
        cols, rows = 9, 6
    require = request.form.get("require", "") == "1"
    srv.set_cb(cols, rows, require)
    return redirect(url_for("index", msg=f"Settings applied: corners {cols}x{rows}, require={require}"))

def mjpeg_generator(which: str):
    boundary = b"--frame"
    while True:
        frame = srv.get_jpeg(which)
        if frame is None:
            time.sleep(0.05)
            continue
        yield boundary + b"\r\n"
        yield b"Content-Type: image/jpeg\r\n"
        yield f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
        yield frame + b"\r\n"
        time.sleep(0.02)

@app.route("/stream/<which>")
def stream(which):
    if which not in ("left", "right"):
        return "bad stream", 400
    return Response(
        mjpeg_generator(which),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/capture", methods=["POST"])
def capture():
    try:
        lp, rp, dt_ms = srv.capture_pair()
        msg = f"Saved {os.path.basename(lp)} + {os.path.basename(rp)} (Δt≈{dt_ms:.1f} ms)"
        return redirect(url_for("index", msg=msg))
    except Exception as e:
        return redirect(url_for("index", msg=f"Capture failed: {e}"))

@app.route("/delete/<base>", methods=["POST"])
def delete(base):
    if not base.isdigit():
        abort(400)
    srv.delete_pair(base)
    return redirect(url_for("index", msg=f"Deleted pair {base}"))

@app.route("/file/<name>")
def file(name):
    path = os.path.join(CAPTURE_DIR, name)
    if not os.path.isfile(path):
        return "not found", 404
    return send_file(path, as_attachment=True)

@app.route("/thumb/<name>")
def thumb(name):
    path = os.path.join(CAPTURE_DIR, name)
    if not os.path.isfile(path):
        return "not found", 404
    data = srv.make_thumb_jpeg(path)
    return Response(data, mimetype="image/jpeg")

@app.route("/download.zip")
def download_zip():
    data = srv.zip_all()
    return send_file(
        io.BytesIO(data),
        mimetype="application/zip",
        as_attachment=True,
        download_name="stereo_captures.zip"
    )

if __name__ == "__main__":
    try:
        app.run(host=HOST, port=PORT, threaded=True)
    finally:
        srv.shutdown()
