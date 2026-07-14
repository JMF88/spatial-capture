#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Open lane, stage A - camera poses via VGGT (feed-forward transformer SfM).

The "AI-aligned" reproducible pose step: instead of classical iterative
structure-from-motion, VGGT (a transformer, CVPR 2025) infers camera poses +
geometry in one forward pass, then exports COLMAP-format poses for the trainer.
We use the low-VRAM fork so ~150 images fit in 8 GB.

This lane is OPTIONAL and OFF the demo critical path. The reliable default is
Jawset Postshot (GUI, no CUDA build) - see pipeline/README.md. VGGT is best on the
bounded bookshelf; for the whole-office enclosure prefer Postshot or COLMAP, since
VGGT poses get noisy on large divergent scenes.

Prereqs (install once; heavy, needs a working CUDA torch):
  git clone https://github.com/harry7557558/vggt-low-vram.git
  cd vggt-low-vram && pip install -r requirements.txt -r requirements_demo.txt
Model facebook/VGGT-1B (~5 GB, research / non-commercial checkpoint) auto-downloads.

Input layout (required):  SCENE_DIR/images/  containing ONLY image files.
Extract at ~1024 px long edge so VGGT intrinsics match the pixels the trainer reads:
  python pipeline/01_extract_frames.py --input shelf.mp4 --out data/shelf/images --long-edge 1024

Example:
  python pipeline/02_poses_vggt.py --scene-dir data/shelf --vggt-repo ../vggt-low-vram
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin")


def normalize_sparse(scene_dir: Path) -> Path:
    """VGGT writes sparse/ flat; gsplat/COLMAP want sparse/0/. Normalize + verify."""
    flat, nested = scene_dir / "sparse", scene_dir / "sparse" / "0"
    if (flat / "cameras.bin").exists() and not (nested / "cameras.bin").exists():
        nested.mkdir(parents=True, exist_ok=True)
        for f in SPARSE_FILES:
            if (flat / f).exists():
                shutil.move(str(flat / f), str(nested / f))
    missing = [f for f in SPARSE_FILES if not (nested / f).exists()]
    if missing:
        raise SystemExit(f"VGGT did not emit {missing} in {nested}.")
    return nested


def run_vggt(scene_dir: Path, vggt_repo: Path, use_ba: bool) -> Path:
    imgs = scene_dir / "images"
    if not imgs.is_dir() or not any(imgs.iterdir()):
        raise SystemExit(f"Put frames in {imgs} first (images ONLY - no other files).")
    demo = vggt_repo / "demo_colmap.py"
    if not demo.exists():
        raise SystemExit(f"demo_colmap.py not in {vggt_repo} "
                         "(git clone https://github.com/harry7557558/vggt-low-vram).")
    # Resolve to absolute: the child runs with cwd=vggt_repo, so a relative
    # --scene-dir would resolve against the wrong directory.
    cmd = [sys.executable, str(demo.resolve()), f"--scene_dir={scene_dir.resolve()}"]
    if use_ba:  # bundle-adjust refine: better, more VRAM; throttled for 8 GB
        cmd += ["--use_ba", "--max_query_pts=2048", "--query_frame_num=5"]
    print("- VGGT:", " ".join(cmd))
    subprocess.run(cmd, cwd=vggt_repo, check=True)
    return normalize_sparse(scene_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, type=Path,
                    help="scene dir; must contain images/ (see 01_extract_frames)")
    ap.add_argument("--vggt-repo", required=True, type=Path,
                    help="path to a cloned harry7557558/vggt-low-vram")
    ap.add_argument("--use-ba", action="store_true",
                    help="bundle-adjust refine (better, more VRAM; skip for big scenes)")
    args = ap.parse_args()
    out = run_vggt(args.scene_dir, args.vggt_repo, args.use_ba)
    print(f"\nPoses ready (COLMAP format): {out}")
    print(f"Next: python pipeline/03_train_gsplat.py --data-dir {args.scene_dir} "
          "--gsplat-examples <path> --result-dir out/<scene>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
