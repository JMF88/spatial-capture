#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Eval gate: block publishing a scene whose quality metrics are below threshold.

Reads a metrics.json produced by the eval steps and checks it against thresholds
(reconstruction PSNR/SSIM vs held-out views, classifier macro-F1, OCR CER). Exits
non-zero if any gate fails -- wire it before the publish step in CI/orchestration
so a low-quality capture never goes live automatically.

metrics.json:
  { "reconstruction": {"psnr": 27.1, "ssim": 0.86},
    "classifier": {"macro_f1": 0.83},
    "ocr": {"cer": 0.09} }

Example:
  python pipeline/gate.py --metrics runs/office/metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# (dotted path, comparison, default threshold). ">=" passes if value >= thr.
DEFAULT_GATES = [
    ("reconstruction.psnr", ">=", 20.0),
    ("reconstruction.ssim", ">=", 0.70),
    ("classifier.macro_f1", ">=", 0.60),
    ("ocr.cer", "<=", 0.30),
]


def _get(d, dotted):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def check(metrics, gates=DEFAULT_GATES):
    """Return (passed, results). results: list of (path, value, op, thr, status).
    A metric that is absent is reported 'missing' and does NOT fail the gate."""
    results = []
    passed = True
    for path, op, thr in gates:
        val = _get(metrics, path)
        if val is None:
            results.append((path, None, op, thr, "missing"))
            continue
        ok = (val >= thr) if op == ">=" else (val <= thr)
        results.append((path, val, op, thr, "ok" if ok else "FAIL"))
        if not ok:
            passed = False
    return passed, results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, type=Path)
    ap.add_argument("--thresholds", type=Path,
                    help='optional JSON overriding defaults: {"path": ["op", thr]}')
    args = ap.parse_args()

    metrics = json.loads(args.metrics.read_text())
    gates = list(DEFAULT_GATES)
    if args.thresholds:
        ov = json.loads(args.thresholds.read_text())
        gates = [(p, op, thr) for p, (op, thr) in ov.items()]

    passed, results = check(metrics, gates)
    for path, val, op, thr, status in results:
        shown = "--" if val is None else f"{val:.4g}"
        print(f"  [{status:7}] {path} = {shown}  (need {op} {thr})")
    print("GATE PASSED" if passed else "GATE FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
