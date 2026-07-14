#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate that a .ply is a real 3D Gaussian SPLAT, not just a point cloud.

Some exporters (notably Postshot with the wrong profile) emit a position+color PLY
that a splat viewer loads but renders as flat dots. A real 3DGS PLY has per-Gaussian
scale, rotation, opacity, and spherical-harmonic color. This parses only the PLY
header and asserts those exist - cheap, no dependencies.

Example:
  python pipeline/validate_splat_ply.py out/shelf/ply/point_cloud_30000.ply
"""
from __future__ import annotations

import argparse
from pathlib import Path

REQUIRED = ["scale_0", "rot_0", "opacity", "f_dc_0"]


def read_header_props(path: Path) -> list[str]:
    props: list[str] = []
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise SystemExit(f"{path} is not a PLY file.")
        while True:
            line = f.readline()
            if not line:
                raise SystemExit("Unterminated PLY header (no end_header).")
            s = line.decode("ascii", "replace").strip()
            if s == "end_header":
                break
            if s.startswith("property"):
                props.append(s.split()[-1])
    return props


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", type=Path)
    args = ap.parse_args()
    if not args.ply.exists():
        raise SystemExit(f"Not found: {args.ply}")

    props = read_header_props(args.ply)
    missing = [p for p in REQUIRED if p not in props]
    n_sh = sum(1 for p in props if p.startswith("f_rest_"))
    if missing:
        print(f"NOT A SPLAT PLY. Missing splat attributes: {missing}")
        print("This looks like a point cloud, not Gaussians. Re-export a *Splat* "
              "profile as PLY (not the point-cloud export).")
        return 1
    print(f"OK: valid 3DGS splat PLY. Has scale/rot/opacity/f_dc + {n_sh} SH "
          "(f_rest_*) coeffs. Safe to compress and host.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
