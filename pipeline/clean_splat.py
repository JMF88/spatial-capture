#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Clean a trained splat: cull floaters, needles and dead Gaussians.

Why this stage exists, measured on the first real capture: a 1.34M-Gaussian splat that
reconstructed beautifully (354/354 frames registered, 0.81 px reprojection error) rendered
as unreadable streaks. Nothing was wrong with the scene. Two things were wrong with 0.3%
of it:

  floaters   4,243 Gaussians sat far outside the room. The scene's real extent is
             7.7 x 9.8 x 2.1 units; the full bounding box was 74.8 x 62.0 x 67.9 -- up to
             32x larger. Any viewer that frames on min/max therefore points the camera at
             nothing. A tenth of a percent of the data owned the entire framing.
  needles    a Gaussian stretched into a splinter draws a streak across the whole view.
             These are an artifact of optimisation, not a feature of the room.

             Measure them by longest/MIDDLE axis, not longest/shortest. 3DGS Gaussians are
             SUPPOSED to flatten into disks -- that is how they represent a surface -- and a
             disk (1 : 1 : 0.01) has a longest/shortest ratio of 100:1 while being exactly
             what you want. A needle (1 : 0.01 : 0.01) is the pathology. The first draft of
             this file used longest/shortest and would have culled 33.7% of a good scene,
             almost all of it legitimate surface disks; longest/middle culls 3.0% and hits
             the actual splinters. Median longest/middle here is 3.6 -- elongation is normal
             and healthy. The tail is not: p99.9 reaches 2,816:1.

Both are normal 3DGS outputs, not a training failure -- which is the point. Every real
splat needs this, so it belongs in the pipeline rather than in a manual GUI pass.

Culling is by percentile and ratio, never by a hard-coded world size, so it transfers to a
scene of any scale. Everything it removes is counted and printed: a cleaner that silently
eats 40% of your scene is worse than the streaks.

Usage:
    python pipeline/clean_splat.py in.ply --out clean.ply
    python pipeline/clean_splat.py in.ply --out clean.ply --keep 99.0 --max-aniso 40

Dependencies: numpy. The PLY is read with memmap, so a 300 MB splat costs no RAM.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def read_ply(path: Path):
    """Return (properties, memmapped structured array, header_text)."""
    with open(path, "rb") as f:
        hdr = b""
        while b"end_header\n" not in hdr:
            chunk = f.read(4096)
            if not chunk:
                raise ValueError("no end_header found; not a PLY?")
            hdr += chunk
    txt = hdr[: hdr.find(b"end_header\n") + 11].decode("ascii", "replace")
    if "binary_little_endian" not in txt:
        raise ValueError("only binary_little_endian PLY is supported")
    props = [ln.split()[-1] for ln in txt.splitlines() if ln.startswith("property float")]
    n = int(next(ln for ln in txt.splitlines() if ln.startswith("element vertex")).split()[-1])
    dt = np.dtype([(p, "<f4") for p in props])
    arr = np.memmap(path, dtype=dt, mode="r", offset=len(txt), shape=(n,))
    return props, arr, txt


def write_ply(path: Path, props: list[str], rows: np.ndarray, source_header: str) -> None:
    keep_comments = [ln for ln in source_header.splitlines() if ln.startswith("comment")]
    head = ["ply", "format binary_little_endian 1.0", *keep_comments,
            "comment cleaned by pipeline/clean_splat.py",
            f"element vertex {len(rows)}",
            *[f"property float {p}" for p in props], "end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        rows.tofile(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cull floaters and needles from a 3DGS PLY.")
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep", type=float, default=99.0,
                    help="central percentile of positions to keep per axis (default 99 => trim 0.5%% each end)")
    ap.add_argument("--max-aniso", type=float, default=30.0,
                    help="drop splinters: longest axis / MIDDLE axis above this. Not longest/shortest "
                         "-- flat disks are the good case (see module docstring)")
    ap.add_argument("--max-scale-pct", type=float, default=99.9,
                    help="drop Gaussians larger than this percentile of size")
    ap.add_argument("--min-opacity", type=float, default=0.02,
                    help="drop Gaussians fainter than this (post-sigmoid)")
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"error: {args.input} not found", file=sys.stderr)
        return 2

    props, v, hdr = read_ply(args.input)
    need = {"x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity"}
    missing = need - set(props)
    if missing:
        print(f"error: not a 3DGS splat; missing {sorted(missing)}", file=sys.stderr)
        return 2

    n = len(v)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    lin = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float32))
    opacity = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))

    lo = np.percentile(xyz, (100 - args.keep) / 2, axis=0)
    hi = np.percentile(xyz, 100 - (100 - args.keep) / 2, axis=0)
    in_box = np.all((xyz >= lo) & (xyz <= hi), axis=1)

    srt = np.sort(lin, axis=1)                       # [shortest, middle, longest]
    longest = srt[:, 2]
    # longest/MIDDLE: ~1 for a surface disk (good), huge for a splinter (bad). See docstring.
    not_needle = (srt[:, 2] / np.maximum(srt[:, 1], 1e-12)) <= args.max_aniso
    not_huge = longest <= np.percentile(longest, args.max_scale_pct)
    visible = opacity >= args.min_opacity

    keep = in_box & not_needle & not_huge & visible

    def pct(mask):
        return f"{(~mask).sum():>9,}  ({100 * (~mask).mean():5.2f}%)"

    print(f"  input: {n:,} gaussians")
    print(f"  culled floaters (outside p{args.keep} box)   : {pct(in_box)}")
    print(f"  culled needles  (long/mid > {args.max_aniso:g}:1)      : {pct(not_needle)}")
    print(f"  culled giants   (> p{args.max_scale_pct} size)          : {pct(not_huge)}")
    print(f"  culled faint    (opacity < {args.min_opacity})        : {pct(visible)}")
    print(f"  KEPT: {keep.sum():,}  ({100 * keep.mean():.1f}%)")

    if keep.sum() < 0.5 * n:
        print("  !! culling more than half the scene -- thresholds are too aggressive", file=sys.stderr)

    before = xyz.max(0) - xyz.min(0)
    after = xyz[keep].max(0) - xyz[keep].min(0)
    print(f"\n  bbox before: {np.round(before, 1)}")
    print(f"  bbox after : {np.round(after, 2)}   <- what a viewer will now frame on")
    print(f"  shrunk by  : {np.round(before / np.maximum(after, 1e-9), 1)}x per axis")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_ply(args.out, props, np.ascontiguousarray(v[keep]), hdr)
    mb_in = args.input.stat().st_size / 1048576
    mb_out = args.out.stat().st_size / 1048576
    print(f"\n  wrote {args.out}  {mb_out:.0f} MB (was {mb_in:.0f} MB, {100 * mb_out / mb_in:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
