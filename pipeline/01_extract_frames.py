#!/usr/bin/env python3
"""
Stage 1 - frame extraction.

Pull sharp, well-spaced still frames from a phone capture video, ready for the
structure-from-motion / Gaussian-splat stage.

Why this stage matters: reconstruction quality is set at capture and here, not
in the trainer. Fewer sharp, well-overlapped frames beat thousands of soft ones.
We sample at a fixed rate, downscale to a VRAM-friendly size, then drop the
blurriest frames by variance-of-Laplacian (a standard focus measure).

Usage:
    python 01_extract_frames.py --input data/office/office.mp4 \
        --out data/office/frames --fps 3 --long-edge 1600 --keep 0.85

Dependencies: ffmpeg on PATH, plus numpy + pillow (see requirements.txt).
Intentionally light: no OpenCV, no GPU, no ML deps for this stage.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def run_ffmpeg(video: Path, tmp: Path, fps: float, long_edge: int) -> None:
    """Sample frames at `fps`, downscaling the long edge to `long_edge` px
    (never upscaling), writing high-quality JPEGs into `tmp`."""
    tmp.mkdir(parents=True, exist_ok=True)
    # If landscape (iw>ih): width=min(long_edge,iw), height auto (-2, keeps aspect,
    # divisible by 2). If portrait: height=min(long_edge,ih), width auto.
    scale = (
        f"scale='if(gt(iw,ih),min({long_edge},iw),-2)':"
        f"'if(gt(iw,ih),-2,min({long_edge},ih))'"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video),
        "-vf", f"fps={fps},{scale}",
        "-qscale:v", "2",
        str(tmp / "raw_%05d.jpg"),
    ]
    print("- ffmpeg:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def laplacian_variance(path: Path) -> float:
    """Variance of the Laplacian (4-neighbour kernel). Higher = sharper.
    Computed on grayscale with pure numpy slicing (no scipy/OpenCV)."""
    img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    # kernel [[0,1,0],[1,-4,1],[0,1,0]] over the valid interior
    lap = (
        img[:-2, 1:-1] + img[1:-1, :-2]
        - 4.0 * img[1:-1, 1:-1]
        + img[1:-1, 2:] + img[2:, 1:-1]
    )
    return float(lap.var())


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract sharp, downscaled frames from a capture video."
    )
    ap.add_argument("--input", required=True, type=Path, help="capture video (mp4/mov)")
    ap.add_argument("--out", required=True, type=Path, help="output frames directory")
    ap.add_argument("--fps", type=float, default=3.0, help="frames/sec to sample")
    ap.add_argument("--long-edge", type=int, default=1600,
                    help="downscale the long edge to N px (never upscales)")
    ap.add_argument("--keep", type=float, default=0.85,
                    help="fraction of sharpest frames to keep, 0-1")
    args = ap.parse_args()

    if not 0.0 < args.keep <= 1.0:
        print("ERROR: --keep must be in (0, 1].", file=sys.stderr)
        return 2
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found on PATH.", file=sys.stderr)
        return 2
    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 2

    tmp = args.out.parent / f"{args.out.name}_raw"
    run_ffmpeg(args.input, tmp, args.fps, args.long_edge)
    raw = sorted(tmp.glob("raw_*.jpg"))
    if not raw:
        print("ERROR: ffmpeg produced no frames (check the video).", file=sys.stderr)
        shutil.rmtree(tmp, ignore_errors=True)
        return 1

    scored = [(p, laplacian_variance(p)) for p in raw]
    scored.sort(key=lambda t: t[1], reverse=True)          # sharpest first
    n_keep = max(1, int(round(len(scored) * args.keep)))
    keep = sorted(p for p, _ in scored[:n_keep])           # restore time order

    args.out.mkdir(parents=True, exist_ok=True)
    for existing in args.out.glob("frame_*.jpg"):
        existing.unlink()
    for i, p in enumerate(keep):
        shutil.copy2(p, args.out / f"frame_{i:05d}.jpg")
    shutil.rmtree(tmp, ignore_errors=True)

    sharp = [s for _, s in scored]
    lo, mid, hi = sharp[n_keep - 1], sharp[len(sharp) // 2], sharp[0]
    print(
        f"\nSampled {len(raw)} frames, kept {len(keep)} sharpest ({args.keep:.0%}).\n"
        f"Sharpness (var-of-Laplacian) kept-min / all-median / max: "
        f"{lo:.0f} / {mid:.0f} / {hi:.0f}\n"
        f"Frames ready: {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
