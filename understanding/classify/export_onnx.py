#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export a trained classifier checkpoint to ONNX for in-browser use.

Produces model.onnx (+ a .classes.json sidecar) that onnxruntime-web / transformers.js
can run in the live-overlay demo, and VERIFIES the ONNX outputs match torch within
tolerance so a browser gets the same predictions as training.

Example:
  python export_onnx.py --run runs/books
  python export_onnx.py --run runs/books --out web/model.onnx --imgsz 224
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import common


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="training output dir (has model.pt)")
    ap.add_argument("--out", type=Path, default=None, help="output .onnx (default: <run>/model.onnx)")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--tol", type=float, default=1e-3, help="max |torch-onnx| tolerance")
    args = ap.parse_args()

    # torch's ONNX exporter prints status with emoji; avoid a cp1252 crash on Windows.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    device = torch.device("cpu")
    model, classes = common.load_for_inference(args.run / "model.pt", device)
    model.eval()
    out = args.out or (args.run / "model.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, args.imgsz, args.imgsz)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        verbose=False,
    )
    out.with_suffix(".classes.json").write_text(json.dumps(classes, indent=2))

    # Consolidate any external-data tensors into the single .onnx (browser-friendly).
    import onnx
    onnx.save_model(onnx.load(str(out)), str(out), save_as_external_data=False)
    side = out.with_name(out.name + ".data")
    if side.exists():
        side.unlink()

    # Verify parity with onnxruntime (CPU) so the browser gets identical results.
    import onnxruntime as ort
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["logits"], {"input": dummy.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    ok = max_diff < args.tol

    print(f"Exported {out}  ({len(classes)} classes: {classes})")
    print(f"Sidecar:  {out.with_suffix('.classes.json')}")
    print(f"torch-vs-onnx max |diff| = {max_diff:.2e}  ->  {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print("MISMATCH: the ONNX graph diverges from torch beyond tolerance.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
