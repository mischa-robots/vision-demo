#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mischa Schaefer <https://github.com/mischa-robots/vision-demo>

"""
stereo_calibrate.py — Robust stereo calibration for Jetson Nano dual-camera rig.

IMPORTANT: all calibration steps use the same distortion model (5-coefficient
standard model).  Mixing CALIB_RATIONAL_MODEL (8 coefficients) in individual
calibration with CALIB_FIX_INTRINSIC in stereo calibration causes OpenCV's
stereo solver to diverge completely — producing RMS in the hundreds of millions
and a baseline that is 20x the real value.

Usage:
    python3 stereo_calibrate.py
    python3 stereo_calibrate.py --captures captures/ --square 0.053 --board 10x4
"""

import cv2
import numpy as np
import glob
import os
import argparse
import sys

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
CHECKERBOARD      = (10, 4)     # inner corners (cols, rows)
SQUARE_SIZE       = 0.053       # metres — measure the physical board
CAPTURE_DIR       = "captures"
OUTPUT_FILE       = "stereo_calibration.yaml"
MAX_PER_IMAGE_RMS = 0.6         # drop pairs above this threshold (pixels)
KNOWN_BASELINE_M  = 0.064       # your measured camera separation — for sanity check only

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
STEREO_CRITERIA = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 200, 1e-7)

# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--captures", default=CAPTURE_DIR)
    p.add_argument("--out",      default=OUTPUT_FILE)
    p.add_argument("--square",   type=float, default=SQUARE_SIZE,
                   help="Square size in metres")
    p.add_argument("--board",    default=f"{CHECKERBOARD[0]}x{CHECKERBOARD[1]}",
                   help="Inner corners WxH, e.g. 10x4")
    return p.parse_args()


def make_objp(board, sq):
    objp = np.zeros((board[0] * board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2)
    objp *= sq
    return objp


def find_corners(gray, board):
    """
    Detect and sub-pixel refine checkerboard corners.
    cornerSubPix is mandatory — integer corners cause depth errors of 2–5×.
    """
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)
    ret, corners = cv2.findChessboardCorners(gray, board, flags)
    if not ret:
        return False, None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
    return True, corners


def per_image_rms(objp, imgp, rvec, tvec, mtx, dist):
    proj, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
    return float(np.sqrt(np.mean((imgp - proj) ** 2)))


def main():
    args = parse_args()
    try:
        bw, bh = map(int, args.board.lower().split("x"))
        board = (bw, bh)
    except Exception:
        print(f"ERROR: cannot parse board '{args.board}'")
        sys.exit(1)

    sq   = args.square
    objp = make_objp(board, sq)

    print(f"Board  : {board[0]}×{board[1]} inner corners")
    print(f"Square : {sq*100:.1f} cm  ({sq} m)")
    print(f"Dir    : {args.captures}")

    # ── Collect image pairs ────────────────────────────────────
    left_images = sorted(glob.glob(os.path.join(args.captures, "left_*.png")))
    if not left_images:
        print(f"ERROR: no left_*.png in '{args.captures}'")
        sys.exit(1)

    print(f"\nFound {len(left_images)} image pairs — detecting corners …")

    objpoints = []
    imgpts_L  = []
    imgpts_R  = []
    names     = []
    image_size = None

    for lp in left_images:
        base = os.path.basename(lp).replace("left_", "")
        rp   = os.path.join(args.captures, "right_" + base)
        if not os.path.exists(rp):
            print(f"  SKIP {base}: no right image")
            continue

        imgL = cv2.imread(lp)
        imgR = cv2.imread(rp)
        if imgL is None or imgR is None:
            print(f"  SKIP {base}: read failed")
            continue

        gL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gL.shape[::-1]

        retL, cL = find_corners(gL, board)
        retR, cR = find_corners(gR, board)

        if retL and retR:
            objpoints.append(objp)
            imgpts_L.append(cL)
            imgpts_R.append(cR)
            names.append(base)
            print(f"  OK   {base}")
        else:
            print(f"  FAIL {base}  (L={retL} R={retR})")

    n = len(objpoints)
    print(f"\n{n} valid pairs detected.")
    if n < 10:
        print("ERROR: need at least 10 pairs.")
        sys.exit(1)

    # ── Individual calibrations ────────────────────────────────
    # IMPORTANT: use the DEFAULT flag (0) — standard 5-coefficient model.
    # Do NOT use CALIB_RATIONAL_MODEL here if you will pass the result into
    # stereoCalibrate with CALIB_FIX_INTRINSIC — the coefficient vector
    # lengths must match exactly, or the stereo solve will diverge.
    print("\n── Individual calibrations (standard 5-coeff model) ──")

    rmsL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
        objpoints, imgpts_L, image_size, None, None)[:5]
    rmsR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
        objpoints, imgpts_R, image_size, None, None)[:5]

    print(f"  Left  RMS: {rmsL:.4f} px")
    print(f"  Right RMS: {rmsR:.4f} px")

    if max(rmsL, rmsR) > 1.5:
        print("  WARNING: individual RMS > 1.5 px — calibration may be poor.")
        print("  Consider: more images, better coverage, no blur.")

    # ── Per-image filtering ────────────────────────────────────
    errsL = [per_image_rms(objpoints[i], imgpts_L[i], rvecsL[i], tvecsL[i], mtxL, distL)
             for i in range(n)]
    errsR = [per_image_rms(objpoints[i], imgpts_R[i], rvecsR[i], tvecsR[i], mtxR, distR)
             for i in range(n)]
    pair_err  = [max(errsL[i], errsR[i]) for i in range(n)]
    keep      = [e <= MAX_PER_IMAGE_RMS for e in pair_err]
    n_dropped = keep.count(False)

    if n_dropped:
        print(f"\nDropping {n_dropped} high-error pairs (>{MAX_PER_IMAGE_RMS}px):")
        for i, (nm, e, k) in enumerate(zip(names, pair_err, keep)):
            if not k:
                print(f"  {nm}  rms={e:.3f}px")
        objpoints = [objpoints[i] for i, k in enumerate(keep) if k]
        imgpts_L  = [imgpts_L[i]  for i, k in enumerate(keep) if k]
        imgpts_R  = [imgpts_R[i]  for i, k in enumerate(keep) if k]
        names     = [names[i]     for i, k in enumerate(keep) if k]
        print(f"  Remaining: {len(objpoints)} pairs")

    if len(objpoints) < 10:
        print("ERROR: fewer than 10 pairs after filtering — collect more images.")
        sys.exit(1)

    # Re-calibrate on filtered set with same standard model
    print("\n── Re-calibrating on filtered set ──")
    rmsL, mtxL, distL, _, _ = cv2.calibrateCamera(
        objpoints, imgpts_L, image_size, None, None)
    rmsR, mtxR, distR, _, _ = cv2.calibrateCamera(
        objpoints, imgpts_R, image_size, None, None)
    print(f"  Left  RMS: {rmsL:.4f} px")
    print(f"  Right RMS: {rmsR:.4f} px")

    # Sanity check: distortion vector length must be 5 for CALIB_FIX_INTRINSIC
    assert distL.shape[1] == 5, f"distL shape {distL.shape} — expected 5 coefficients"
    assert distR.shape[1] == 5, f"distR shape {distR.shape} — expected 5 coefficients"

    # ── Stereo calibration ────────────────────────────────────
    print("\n── Stereo calibration ──")
    (retStereo, mtxL, distL, mtxR, distR,
     R, T, E, F) = cv2.stereoCalibrate(
        objpoints, imgpts_L, imgpts_R,
        mtxL, distL, mtxR, distR,
        image_size,
        criteria = STEREO_CRITERIA,
        # CALIB_FIX_INTRINSIC keeps the individually-calibrated K and D
        # matrices fixed and only solves for R and T.
        # This requires identical distortion model (same coefficient count).
        flags    = cv2.CALIB_FIX_INTRINSIC,
    )

    baseline_m = float(np.linalg.norm(T))
    print(f"  Stereo RMS : {retStereo:.4f} px")
    print(f"  Baseline   : {baseline_m*100:.2f} cm  (measured: {KNOWN_BASELINE_M*100:.1f} cm)")

    # Validate the result before going further
    if retStereo > 5.0:
        print("\n  ERROR: stereo RMS > 5 px — the solve diverged.")
        print("  Most likely cause: distortion model mismatch.")
        print("  Check that distL and distR both have exactly 5 coefficients (see above).")
        sys.exit(1)

    if abs(baseline_m - KNOWN_BASELINE_M) > 0.020:
        print(f"\n  WARNING: baseline {baseline_m*100:.1f}cm differs from measured "
              f"{KNOWN_BASELINE_M*100:.1f}cm by more than 2cm.")
        print("  Check SQUARE_SIZE is correct in metres and matches the physical board.")

    # ── Stereo rectification ──────────────────────────────────
    print("\n── Stereo rectification ──")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtxL, distL, mtxR, distR, image_size, R, T,
        alpha = 0,                         # crop to valid pixels (no black borders)
        flags = cv2.CALIB_ZERO_DISPARITY,  # align principal points
    )

    f_px       = float(P1[0, 0])
    Tx         = float(P2[0, 3])
    baseline_Q = abs(Tx / f_px) if abs(f_px) > 1e-6 else 0.0

    print(f"  Focal length (P1)   : {f_px:.1f} px")
    print(f"  Baseline from Q     : {baseline_Q*100:.2f} cm")
    print(f"  cx diff (P2-P1)     : {float(P2[0,2]-P1[0,2]):.2f} px  (should be ~0)")

    # ── Epipolar check on last pair ───────────────────────────
    last   = names[-1]
    imgL_t = cv2.imread(os.path.join(args.captures, "left_"  + last))
    imgR_t = cv2.imread(os.path.join(args.captures, "right_" + last))
    mapLx, mapLy = cv2.initUndistortRectifyMap(
        mtxL, distL, R1, P1, image_size, cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(
        mtxR, distR, R2, P2, image_size, cv2.CV_32FC1)
    rectL = cv2.remap(imgL_t, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR_t, mapRx, mapRy, cv2.INTER_LINEAR)

    gL2 = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    gR2 = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    flags_cb = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE |
                cv2.CALIB_CB_FAST_CHECK)
    retL2, cL2 = cv2.findChessboardCorners(gL2, board, flags_cb)
    retR2, cR2 = cv2.findChessboardCorners(gR2, board, flags_cb)
    if retL2 and retR2:
        cL2 = cv2.cornerSubPix(gL2, cL2, (11,11), (-1,-1), SUBPIX_CRITERIA)
        cR2 = cv2.cornerSubPix(gR2, cR2, (11,11), (-1,-1), SUBPIX_CRITERIA)
        vert_err = float(np.mean(np.abs(cL2[:,0,1] - cR2[:,0,1])))
        horiz_err = float(np.mean(np.abs(cL2[:,0,0] - cR2[:,0,0])))
        print(f"  Epipolar vertical error : {vert_err:.3f} px  "
              f"({'OK' if vert_err < 0.5 else 'HIGH — recollect images'})")
        print(f"  Epipolar horizontal err : {horiz_err:.3f} px  "
              f"(expected ~{f_px*baseline_Q:.0f} px at board distance)")
    else:
        print("  Epipolar check: board not visible in rectified image")

    # Save rectification preview
    preview = np.hstack([rectL, rectR])
    for y in range(0, preview.shape[0], 40):
        cv2.line(preview, (0,y), (preview.shape[1],y), (0,255,0), 1)
    cv2.imwrite("rectification_preview.png", preview)
    print("  Saved: rectification_preview.png")

    # ── Save YAML ─────────────────────────────────────────────
    print(f"\nSaving to {args.out} …")
    fs = cv2.FileStorage(args.out, cv2.FILE_STORAGE_WRITE)
    fs.write("mtxL",          mtxL)
    fs.write("distL",         distL)
    fs.write("mtxR",          mtxR)
    fs.write("distR",         distR)
    fs.write("R",             R)
    fs.write("T",             T)
    fs.write("R1",            R1)
    fs.write("R2",            R2)
    fs.write("P1",            P1)
    fs.write("P2",            P2)
    fs.write("Q",             Q)
    fs.write("stereo_rms",    retStereo)
    fs.write("baseline_m",    baseline_m)
    fs.write("square_size_m", sq)
    fs.write("image_width",   image_size[0])
    fs.write("image_height",  image_size[1])
    fs.release()

    # ── Final report ──────────────────────────────────────────
    disp_at_1m  = f_px * baseline_m / 1.0
    nd_suggested = int(np.ceil(disp_at_1m / 16) * 16 * 2)

    print("\n" + "="*46)
    print("  CALIBRATION REPORT")
    print("="*46)
    print(f"  Pairs used        : {len(objpoints)}")
    print(f"  Stereo RMS        : {retStereo:.4f} px  "
          f"({'good' if retStereo < 0.5 else 'ok' if retStereo < 1.0 else 'poor'})")
    print(f"  Baseline (calib)  : {baseline_m*100:.2f} cm")
    print(f"  Baseline (Q)      : {baseline_Q*100:.2f} cm")
    print(f"  Focal length (P1) : {f_px:.1f} px")
    print(f"  Disparity at 1m   : {disp_at_1m:.1f} px")
    print(f"  Disparity at 3m   : {f_px*baseline_m/3.0:.1f} px")
    print()
    print(f"  --> numDisparities >= {nd_suggested} to cover 0.5m–inf")
    print(f"      (set this in stereo_web.py DEFAULT_PARAMS)")
    print("="*46)


if __name__ == "__main__":
    main()
