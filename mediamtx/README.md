# MediaMTX

MediaMTX is a lightweight media server that receives video streams from the demo binaries and serves them to the browser via WebRTC, HLS, or RTSP. One MediaMTX instance handles all demos simultaneously.

---

## Download

MediaMTX is not included in the repository. Download the ARM64 binary for the Nano:

```bash
cd mediamtx/
wget https://github.com/bluenviron/mediamtx/releases/download/v1.16.2/mediamtx_v1.16.2_linux_arm64v8.tar.gz
tar -xzf mediamtx_v1.16.2_linux_arm64v8.tar.gz
rm mediamtx_v1.16.2_linux_arm64v8.tar.gz
```

The tarball includes its own `mediamtx.yml` — this is fine since our config is named `mediamtx_demo.yml` and won't be overwritten.

---

## Run

```bash
cd mediamtx/
./mediamtx mediamtx_demo.yml
```

Start MediaMTX before running any demo binary. It listens for incoming streams and serves them as soon as a binary connects — no restart needed when switching demos.

---

## Streams

Each demo pushes to a dedicated UDP port when running. MediaMTX exposes all of them simultaneously:

| Demo | WebRTC | HLS | RTSP |
|------|--------|-----|------|
| Stereo Depth | `http://<jetson-ip>:8889/stereo_depth` | `http://<jetson-ip>:8888/stereo_depth` | `rtsp://<jetson-ip>:8554/stereo_depth` |
| Optical Flow | `http://<jetson-ip>:8889/optical_flow` | `http://<jetson-ip>:8888/optical_flow` | `rtsp://<jetson-ip>:8554/optical_flow` |
| YOLO Detection | `http://<jetson-ip>:8889/yolo` | `http://<jetson-ip>:8888/yolo` | `rtsp://<jetson-ip>:8554/yolo` |

**WebRTC** gives the lowest latency and works directly in Chrome and Firefox — recommended for live demos.
**HLS** is the most compatible fallback for other browsers and devices but has a streaming delay. It's deactivated by default, to activate set `hls: true`
**RTSP** is for VLC, ffplay, gstreamer or other video tools. It's deactivated by default, to activate set `rtsp: true`

---

## Ports

| Port | Protocol |
|------|----------|
| 8554 | RTSP |
| 8888 | HLS |
| 8889 | WebRTC |
| 8189 | WebRTC UDP |
| 9997 | API |

Make sure these ports are reachable on your network. On a typical home/lab network no firewall configuration is needed.

---

## Files

```
mediamtx/
├── mediamtx          # Binary — not committed, download above
└── mediamtx_demo.yml # Unified config for all demos
```


## License

MIT

Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>
