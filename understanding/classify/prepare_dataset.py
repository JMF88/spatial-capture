#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the classifier dataset.

Two modes:
  --from-detections : crop a class (e.g. 'book') out of detect.py's output into a
                      flat pool you then hand-sort into <class>/ subfolders.
  --synthetic       : generate a tiny labelled dataset (colored, textured, shaped
                      tiles) to smoke-test train.py / eval.py before real data.

Examples:
  python prepare_dataset.py --from-detections out/office/detections.json \
      --class-name book --out data/books_unlabeled
  python prepare_dataset.py --synthetic --out data/_synthetic --classes 3 --per-class 40
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


def from_detections(det_json: Path, class_name: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    dets = json.loads(det_json.read_text())
    n = 0
    for d in dets:
        if d.get("class") != class_name:
            continue
        img = Image.open(d["image"]).convert("RGB")
        x1, y1, x2, y2 = (int(v) for v in d["box_xyxy"])
        img.crop((x1, y1, x2, y2)).save(out / f"{Path(d['image']).stem}_{n:03d}.jpg", quality=92)
        n += 1
    print(f"Cropped {n} '{class_name}' regions -> {out}")
    print("Next: hand-sort these into out/<class>/ subfolders, then run train.py.")


def synthetic(out: Path, n_classes: int, per_class: int, seed: int = 0) -> None:
    rng = random.Random(seed)
    base = [(200, 60, 60), (60, 160, 200), (80, 190, 90),
            (200, 170, 60), (160, 90, 200), (90, 90, 90)]
    for c in range(n_classes):
        d = out / f"class_{c}"
        d.mkdir(parents=True, exist_ok=True)
        r0, g0, b0 = base[c % len(base)]
        for i in range(per_class):
            jitter = lambda v: max(0, min(255, v + rng.randint(-25, 25)))  # noqa: E731
            img = Image.new("RGB", (160, 160), (jitter(r0), jitter(g0), jitter(b0)))
            dr = ImageDraw.Draw(img)
            for _ in range(8):  # pixel noise so it's not a flat color
                dr.point((rng.randint(0, 159), rng.randint(0, 159)),
                         fill=(rng.randint(0, 255),) * 3)
            # a class-specific shape so a pretrained net can separate the classes
            if c % 3 == 0:
                dr.ellipse([40, 40, 120, 120], outline=(255, 255, 255), width=4)
            elif c % 3 == 1:
                dr.rectangle([40, 40, 120, 120], outline=(0, 0, 0), width=4)
            else:
                dr.line([20, 20, 140, 140], fill=(255, 255, 0), width=6)
            img.save(d / f"img_{i:03d}.png")
    print(f"Synthetic dataset: {n_classes} classes x {per_class} imgs -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-detections", type=Path)
    ap.add_argument("--class-name", default="book")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--classes", type=int, default=3, help="synthetic: number of classes")
    ap.add_argument("--per-class", type=int, default=40, help="synthetic: images per class")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.synthetic:
        synthetic(args.out, args.classes, args.per_class, args.seed)
    elif args.from_detections:
        from_detections(args.from_detections, args.class_name, args.out)
    else:
        raise SystemExit("Choose --synthetic or --from-detections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
