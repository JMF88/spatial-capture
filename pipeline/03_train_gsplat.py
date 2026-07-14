#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Open lane, stage B - train a 3D Gaussian splat with gsplat (Apache-2.0).

Consumes COLMAP-format poses (from 02_poses_vggt.py, or COLMAP/GLOMAP) and trains
a splat, exporting a standard 3DGS .ply the web viewer loads.

This is the ONE step that needs a compiled CUDA extension (gsplat's rasterizer).
Try a prebuilt wheel first to avoid a compiler:
  python -c "import torch; print(torch.__version__, torch.version.cuda)"
  # match the tag, e.g. torch 2.5 + CUDA 12.4 -> pt25cu124:
  pip install gsplat --index-url https://docs.gsplat.studio/whl/pt25cu124
  git clone https://github.com/nerfstudio-project/gsplat
  pip install -r gsplat/examples/requirements.txt
If no wheel matches your torch+CUDA, you need VS 2022 Build Tools + a CUDA Toolkit
to build from source. If that fights you, don't get blocked on the CUDA build -
train in Postshot instead. This lane is the reproducible/open proof, not the demo path.

Data dir must contain:  images/  and  sparse/0/{cameras,images,points3D}.bin

Example:
  python pipeline/03_train_gsplat.py --data-dir data/shelf \
      --gsplat-examples ../gsplat/examples --result-dir out/shelf --steps 30000
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--gsplat-examples", required=True, type=Path,
                    help="path to gsplat/examples (contains simple_trainer.py)")
    ap.add_argument("--result-dir", type=Path, default=Path("out/gsplat"))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--strategy", default="default", choices=["default", "mcmc"],
                    help="'mcmc' (or a gaussian cap) helps fit 8 GB")
    ap.add_argument("--data-factor", type=int, default=1,
                    help="1 for VGGT/Postshot-sized frames; default 4 downscales")
    args = ap.parse_args()

    cams = args.data_dir / "sparse" / "0" / "cameras.bin"
    if not cams.exists():
        raise SystemExit(f"{cams} missing - run 02_poses_vggt.py (or COLMAP) first.")
    trainer = args.gsplat_examples / "simple_trainer.py"
    if not trainer.exists():
        raise SystemExit(f"simple_trainer.py not in {args.gsplat_examples} "
                         "(git clone https://github.com/nerfstudio-project/gsplat).")

    cmd = [sys.executable, "simple_trainer.py", args.strategy,
           "--data_dir", str(args.data_dir.resolve()),
           "--data_factor", str(args.data_factor),
           "--result_dir", str(args.result_dir.resolve()),
           "--init_type", "sfm",
           "--save_ply",
           "--max_steps", str(args.steps)]
    print("- gsplat:", " ".join(cmd))
    subprocess.run(cmd, cwd=args.gsplat_examples, check=True)

    # gsplat names the PLY by internal 0-indexed step (e.g. point_cloud_29999.ply
    # for --steps 30000), and the exact name is version-dependent, so glob it.
    ply_dir = args.result_dir / "ply"
    plys = sorted(ply_dir.glob("point_cloud_*.ply")) if ply_dir.exists() else []
    if plys:
        ply = plys[-1]
        print(f"\nSplat PLY: {ply}")
        print(f"Validate before hosting:  python pipeline/validate_splat_ply.py {ply}")
    else:
        print(f"\nTraining finished but no PLY under {ply_dir} - check the result dir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
