#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Compress a cleaned splat into something a phone will actually load.

Brush exports float32 PLY. The first real capture came out at 259 MB cleaned, against a
budget of ~15 MB for a page someone opens on cellular. This stage closes that gap by
shelling out to PlayCanvas splat-transform, and exists mostly to record which of its many
knobs are the right ones -- all three answers below are counter-intuitive and were measured,
not assumed.

MEASURED, on the 1.15M-Gaussian cleaned shelf capture (see docs/PRACTICE.md):

  format      SOG beats SPZ decisively at equal quality: 19 MB vs 27 MB, both with full
              spherical harmonics. Spark loads either. Use SOG.

                  PLY (source)             259 MB
                  SPZ, all SH               27 MB
                  SOG, all SH               19 MB     <- 13.6x, no quality given up
                  SOG, 50% decimated        11 MB
                  SOG, veil-culled + 50%    10 MB     <- ships

  harmonics   KEEP THEM. This is the trap. In the PLY, the 45 f_rest_* properties are 76%
              of the bytes, so dropping spherical harmonics looks like the obvious first
              move -- it is how you would halve any other asset. But SOG stores SH
              clustered and palettised, so in SOG they cost only 21%: dropping every band
              saves 4 MB (19 -> 15) and gives up ALL view-dependent colour. Terrible trade.
              Compress first, then decide about SH -- never the other way round, or you pay
              full price in appearance for a discount SOG was already giving you.

  decimation  50% is close to free here and is where the budget lands. splat-transform's
              -d is merge-based, not random sampling: it fuses neighbours rather than
              throwing them away, which is why halving the count does not visibly soften
              the shelf. Below ~35% the spines start to smear, which is exactly the detail
              the whole capture is for.

  Do NOT reach for -F (voxel-occupancy floater filter) expecting it to replace
  pipeline/clean_splat.py. Measured on this scene it removed nothing: the haze here is
  big+faint Gaussians connected to and in front of the scene, not disconnected floaters.
  Clean first with clean_splat.py, compress second.

Usage:
    python pipeline/compress_splat.py clean.ply --out docs/viewer/assets/scene.sog
    python pipeline/compress_splat.py clean.ply --out scene.sog --decimate 35%

Requires Node (npx fetches @playcanvas/splat-transform on first run). Everything it shells
out to is a build-time tool -- nothing here is served to the browser.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PKG = "@playcanvas/splat-transform"
BUDGET_MB = 15.0


def find_npx() -> str | None:
    """Resolve npx to a real executable.

    On Windows npx is npx.cmd, and subprocess without shell=True cannot spawn a bare "npx"
    -- CreateProcess wants the extension. shutil.which applies PATHEXT and hands back the
    full path, so resolve once here rather than letting it fail at spawn time.
    """
    for name in ("npx", "npx.cmd"):
        found = shutil.which(name)
        if found:
            return found
    return None


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        raise RuntimeError(f"{' '.join(cmd[:3])} failed:\n  " + "\n  ".join(tail))


def main() -> int:
    ap = argparse.ArgumentParser(description="Compress a 3DGS PLY to SOG for the web viewer.")
    ap.add_argument("input", type=Path, help="cleaned .ply (run clean_splat.py first)")
    ap.add_argument("--out", type=Path, required=True, help="output .sog (or .spz, but see docstring)")
    ap.add_argument("--decimate", default="50%",
                    help="merge-based decimation, n or n%% (default 50%%; below ~35%% detail smears). "
                         "'none' to skip")
    ap.add_argument("--drop-harmonics", type=int, default=None, choices=[0, 1, 2, 3],
                    help="remove SH bands above n. Default: keep all. In SOG they cost ~21%%, "
                         "not the 76%% they cost in PLY -- rarely worth it (see docstring)")
    ap.add_argument("--budget-mb", type=float, default=BUDGET_MB,
                    help=f"warn if the result exceeds this (default {BUDGET_MB})")
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"error: {args.input} not found", file=sys.stderr)
        return 2
    npx = find_npx()
    if npx is None:
        print("error: npx not found; Node is required for splat-transform", file=sys.stderr)
        return 2

    src_mb = args.input.stat().st_size / 1048576
    args.out.parent.mkdir(parents=True, exist_ok=True)
    base = [npx, "-y", PKG]
    pre: list[str] = []
    if args.drop_harmonics is not None:
        pre += ["-H", str(args.drop_harmonics)]

    with tempfile.TemporaryDirectory() as td:
        # -d must be the final action and must write .ply, so decimation is its own pass.
        if args.decimate.lower() == "none":
            stage = args.input
            extra = pre
        else:
            stage = Path(td) / "decimated.ply"
            print(f"  decimating {args.decimate} (merge-based -- fuses neighbours, not random sampling)")
            run(base + [str(args.input), *pre, "-d", args.decimate, str(stage)])
            extra = []
        print(f"  encoding {args.out.suffix.lstrip('.').upper()}"
              + ("" if args.drop_harmonics is None else f" (SH bands <= {args.drop_harmonics})"))
        run(base + [str(stage), *extra, str(args.out)])

    out_mb = args.out.stat().st_size / 1048576
    print(f"\n  {args.input.name} {src_mb:.0f} MB -> {args.out.name} {out_mb:.1f} MB "
          f"({src_mb / max(out_mb, 1e-9):.1f}x smaller)")
    if out_mb > args.budget_mb:
        print(f"  !! over the {args.budget_mb:.0f} MB budget -- decimate harder before dropping "
              f"harmonics; SH is only ~21% of a SOG", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1) from None
