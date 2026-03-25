#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Michael Schaefer <https://github.com/mischa-robots/vision-demo>

"""
export_yolo.py
───────────────
Export YOLO and YOLOE models to ONNX, then simplify with onnxsim.

Supports:
  - YOLO detection      (yolo11n.pt, yolo26n.pt, ...)
  - YOLO segmentation   (yolo11n-seg.pt, yolo26n-seg.pt, ...)
  - YOLOE segmentation  (yoloe-11n-seg.pt, yoloe-26n-seg.pt, ...)
    └─ requires --labels with a SMALL class list (≤205 classes for TRT on Nano)

--labels behaviour:
  YOLOE  — required. Passed to set_classes() before export, shapes the output tensor.
  YOLO   — optional. Not used during export (class count is baked into the weights),
            but listed in the summary as a reminder to copy it to models/ on the Nano.
  No --labels + standard YOLO — Ultralytics downloads the weights if not present
            and uses the COCO classes already embedded in the model.
  IF you FINE-TUNED a yolo model with a specific set of classes,
            add them as a my-labels.txt file and include them with the --labels parameter.

Key export settings (non-negotiable for TRT 8.2 on Jetson Nano):
  end2end=False  — one-to-many head, avoids the Mod op that TRT 8.2 cannot handle
                   opset=11 and onnxsim do NOT fix this — only end2end=False does
  dynamic=False  — static shapes required by trtexec
  opset=12       — works for all model types

Usage:
  # Standard YOLO — downloads weights automatically if not present, COCO classes
  python3 export_yolo.py --model yolo26n.pt

  # Fine-tuned YOLO with custom classes
  python3 export_yolo.py --model my_yolo26n.pt --labels my-labels.txt

  # YOLO segmentation — downloads weights automatically if not present, COCO classes
  python3 export_yolo.py --model yolo26n-seg.pt

  # YOLOE segmentation (labels required — keep list short for TRT on Nano)
  python3 export_yolo.py --model yoloe-26n-seg.pt --labels yoloe-seg-labels.txt

  # Skip simplification
  python3 export_yolo.py --model yolo26n.pt --no-simplify
"""

import argparse
import os
import sys


# ── Dependency check ──────────────────────────────────────────────────────────

def check_deps(need_simplify: bool) -> None:
    """Check required packages are installed and print actionable errors."""
    missing = []

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        missing.append("ultralytics")

    try:
        import onnx  # noqa: F401
    except ImportError:
        missing.append("onnx")

    if need_simplify:
        try:
            import onnxsim  # noqa: F401
        except ImportError:
            missing.append("onnxsim")

    if missing:
        pkgs = " ".join(missing)
        print(f"[ERROR] Missing packages: {pkgs}")
        print(f"        Install with:  pip install {pkgs}")
        sys.exit(1)


# ── Model type detection ──────────────────────────────────────────────────────

def is_yoloe(model_pt: str) -> bool:
    return os.path.basename(model_pt).lower().startswith("yoloe")

def is_seg(model_pt: str) -> bool:
    return "-seg" in os.path.basename(model_pt).lower()


# ── Label loader ──────────────────────────────────────────────────────────────

def load_labels(labels_path: str) -> list[str]:
    if not os.path.exists(labels_path):
        print(f"[ERROR] Labels file not found: {labels_path}")
        sys.exit(1)

    with open(labels_path) as f:
        names = [line.strip() for line in f if line.strip()]

    if not names:
        print(f"[ERROR] Labels file is empty: {labels_path}")
        sys.exit(1)

    print(f"[INFO]  Loaded {len(names)} classes from {labels_path}")

    if len(names) > 205:
        print(f"[WARN]  {len(names)} classes — TRT export on Jetson Nano may fail")
        print( "        The Nano TRT engine runs out of memory beyond ~205 classes.")
        print( "        Reduce your label list if you plan to run trtexec on the Nano.")

    return names


# ── Export ────────────────────────────────────────────────────────────────────

def export_onnx(model_pt: str, opset: int, imgsz: int,
                labels: list[str] | None) -> str:
    """Export model to ONNX. Returns path to the exported .onnx file."""

    yoloe = is_yoloe(model_pt)
    seg   = is_seg(model_pt)

    kind = "YOLOE-seg" if yoloe else ("YOLO-seg" if seg else "YOLO-det")
    print(f"\n[1/2] Exporting {model_pt}  [{kind}]")
    print(f"      opset={opset}  imgsz={imgsz}  end2end=False  dynamic=False")

    if not os.path.exists(model_pt):
        print(f"[ERROR] Model file not found: {model_pt}")
        sys.exit(1)

    if yoloe:
        # ── YOLOE: open-vocabulary model, must set classes before export ──────
        if labels is None:
            print("[ERROR] YOLOE models require --labels <file>")
            print("        The default 4500+ class list will cause trtexec to OOM")
            print("        on the Jetson Nano. Provide a short class list (≤205).")
            sys.exit(1)

        from ultralytics import YOLOE
        model = YOLOE(model_pt)

        print(f"      Setting {len(labels)} classes on YOLOE model...")
        model.set_classes(labels, model.get_text_pe(labels))

    else:
        # ── Standard YOLO (det or seg) ────────────────────────────────────────
        # Labels are not consumed during export — class count is baked into the
        # weights. If provided, the file is noted in the summary as a reminder
        # to copy it to models/ alongside the .onnx on the Nano.
        from ultralytics import YOLO
        model = YOLO(model_pt)

    result = model.export(
        format="onnx",
        opset=opset,
        simplify=False,   # we run onnxsim ourselves below for better control
        imgsz=imgsz,
        dynamic=False,    # static shapes — required for trtexec
        end2end=False,    # one-to-many head — CRITICAL for TRT 8.2 (avoids Mod op)
    )

    onnx_path = str(result)

    if not os.path.exists(onnx_path):
        print(f"[ERROR] Export reported success but file not found: {onnx_path}")
        sys.exit(1)

    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"[OK]   Exported → {onnx_path}  ({size_mb:.1f} MB)")
    return onnx_path


# ── Simplify ──────────────────────────────────────────────────────────────────

def simplify_onnx(onnx_path: str) -> str:
    """Simplify with onnxsim. Returns path to simplified file."""
    import onnx
    from onnxsim import simplify

    print(f"\n[2/2] Simplifying {onnx_path} with onnxsim...")

    print("      Loading and validating original model...")
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"[ERROR] Original ONNX model is invalid: {e}")
        sys.exit(1)

    print("      Running simplification...")
    try:
        simplified, ok = simplify(model)
    except Exception as e:
        print(f"[ERROR] onnxsim failed: {e}")
        sys.exit(1)

    if not ok:
        print("[ERROR] onnxsim reported simplification failed")
        sys.exit(1)

    print("      Validating simplified model...")
    try:
        onnx.checker.check_model(simplified)
    except Exception as e:
        print(f"[ERROR] Simplified model failed validation: {e}")
        sys.exit(1)

    base, ext = os.path.splitext(onnx_path)
    out_path = f"{base}_simplified{ext}"
    onnx.save(simplified, out_path)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[OK]   Simplified → {out_path}  ({size_mb:.1f} MB)")
    return out_path


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(final_path: str, labels_path: str | None, yoloe: bool) -> None:
    fname     = os.path.basename(final_path)
    trt_fname = fname.replace(".onnx", ".trt")

    print("\n" + "=" * 58)
    print("  Done!")
    print(f"  Output: {final_path}")
    print()
    print("  Copy to Jetson Nano:")
    print(f"    {fname}")
    if labels_path:
        print(f"    {labels_path}  ← copy this too, needed by the binary")
    print()
    print("  Then build TRT engine on the Nano:")
    print(f"    trtexec \\")
    print(f"      --onnx={fname} \\")
    print(f"      --saveEngine={trt_fname} \\")
    print( "      --fp16 --workspace=3072")
    print()
    print("  Note: trtexec must be run ON the Nano — .trt files are")
    print("        GPU- and TRT-version-specific.")
    if not labels_path and not yoloe:
        print()
        print("  No --labels provided. If this is a fine-tuned model, pass")
        print("  --labels <file> so the summary reminds you to copy it to the Nano.")
    print("=" * 58)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export YOLO / YOLOE model to ONNX (TRT 8.2 compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model",       default="yolo26n.pt",
                    help="Model weights file (default: yolo26n.pt)")
    ap.add_argument("--labels",      default=None,
                    help="Labels .txt file — required for YOLOE; optional for fine-tuned YOLO (not used during export, but listed in summary as a copy reminder)")
    ap.add_argument("--opset",       type=int, default=12,
                    help="ONNX opset version (default: 12)")
    ap.add_argument("--imgsz",       type=int, default=640,
                    help="Input image size (default: 640)")
    ap.add_argument("--no-simplify", action="store_true",
                    help="Skip onnxsim simplification step")
    args = ap.parse_args()

    need_simplify = not args.no_simplify
    check_deps(need_simplify)

    print("=" * 58)
    print("  YOLO ONNX Export")
    print("=" * 58)
    print(f"  Model      : {args.model}")
    print(f"  Labels     : {args.labels or '(none)'}")
    print(f"  Opset      : {args.opset}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Simplify   : {'yes' if need_simplify else 'no'}")

    labels = load_labels(args.labels) if args.labels else None

    onnx_path = export_onnx(args.model, args.opset, args.imgsz, labels)

    if need_simplify:
        final_path = simplify_onnx(onnx_path)
    else:
        final_path = onnx_path

    print_summary(final_path, args.labels, is_yoloe(args.model))


if __name__ == "__main__":
    main()
