#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Orchestrator: build a queryable scene from one config.

Runs the automatable stages -- frame extraction -> detection -> semantic fusion ->
publish -- from a YAML config, skipping stages whose outputs already exist
(resumable; use --force to re-run). Splat training itself (Brush, or VGGT+gsplat)
is a separate step: point the config at its outputs (sparse poses + trained splat).

config.yaml:
  scene: office
  video: data/office/office.mp4
  workdir: data/office
  frames:  { fps: 3, long_edge: 1600, keep: 0.85 }
  sparse:  data/office/sparse/0
  splat:   data/office/office.ply
  detect:  { classes: ["book", "chair", "potted plant", "lamp"], weights: yoloe-11l-seg.pt }
  fusion:  { dist_thresh: 0.5, min_points: 4, min_frames: 2 }
  publish: docs/viewer/assets

Examples:
  python pipeline/run.py --config configs/office.yaml
  python pipeline/run.py --config configs/office.yaml --dry-run
  python pipeline/run.py --config configs/office.yaml --from detect
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

STAGES = ["frames", "detect", "fusion", "publish"]


def plan_stages(cfg, repo, python_exe="python", from_stage=None, to_stage=None, only=None):
    """Pure planning: return [{name, argv, out}] for the selected stages.
    argv[0] == '@copy' marks a file-copy pseudo-stage (src, dst)."""
    repo = Path(repo)
    work = Path(cfg.get("workdir", "."))
    frames = work / "frames"
    det_out = work / "detect"
    detections = det_out / "detections.json"
    publish = Path(cfg.get("publish", "docs/viewer/assets"))
    fr, dt, fu = cfg.get("frames", {}), cfg.get("detect", {}), cfg.get("fusion", {})

    catalog = {
        "frames": {
            "argv": [python_exe, str(repo / "pipeline" / "01_extract_frames.py"),
                     "--input", str(cfg.get("video", "")), "--out", str(frames),
                     "--fps", str(fr.get("fps", 3)),
                     "--long-edge", str(fr.get("long_edge", 1600)),
                     "--keep", str(fr.get("keep", 0.85))],
            "out": frames},
        "detect": {
            "argv": [python_exe, str(repo / "understanding" / "detect.py"),
                     "--frames", str(frames),
                     "--classes", ",".join(dt.get("classes", [])),
                     "--out", str(det_out),
                     "--weights", dt.get("weights", "yoloe-11l-seg.pt")],
            "out": detections},
        "fusion": {
            "argv": [python_exe, str(repo / "understanding" / "fusion" / "build_scene.py"),
                     "--sparse", str(cfg.get("sparse", "")),
                     "--detections", str(detections),
                     "--splat", "./assets/scene.ply",
                     "--out", str(publish / "scene.json"),
                     "--dist-thresh", str(fu.get("dist_thresh", 0.5)),
                     "--min-points", str(fu.get("min_points", 4)),
                     "--min-frames", str(fu.get("min_frames", 2))],
            "out": publish / "scene.json"},
        "publish": {
            "argv": ["@copy", str(cfg.get("splat", "")), str(publish / "scene.ply")],
            "out": publish / "scene.ply"},
    }

    if only:
        names = [only]
    else:
        i0 = STAGES.index(from_stage) if from_stage else 0
        i1 = STAGES.index(to_stage) + 1 if to_stage else len(STAGES)
        names = STAGES[i0:i1]
    return [{"name": n, **catalog[n]} for n in names]


def load_config(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--from", dest="from_stage", choices=STAGES)
    ap.add_argument("--to", dest="to_stage", choices=STAGES)
    ap.add_argument("--only", choices=STAGES)
    ap.add_argument("--force", action="store_true", help="re-run even if outputs exist")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    plan = plan_stages(cfg, repo, sys.executable, args.from_stage, args.to_stage, args.only)

    for st in plan:
        out = Path(st["out"])
        if out.exists() and not args.force:
            print(f"[skip] {st['name']} (exists: {out})")
            continue
        if args.dry_run:
            print(f"[plan] {st['name']}: " + " ".join(st["argv"]))
            continue
        print(f"[run ] {st['name']}")
        if st["argv"][0] == "@copy":
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(st["argv"][1], st["argv"][2])
        else:
            subprocess.run(st["argv"], check=True)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
