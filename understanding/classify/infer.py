#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the trained classifier on new image(s).

Example:
  python infer.py --run runs/books --image some_book.jpg
  python infer.py --run runs/books --dir data/new_crops --topk 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import common


def predict(model, tf, classes, path, device, topk):
    x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0]
    k = min(topk, len(classes))
    conf, idx = probs.topk(k)
    return [(classes[i], float(c)) for c, i in zip(conf.tolist(), idx.tolist())]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="training output dir")
    ap.add_argument("--image", type=Path)
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    if not args.image and not args.dir:
        raise SystemExit("Give --image or --dir.")
    device = common.pick_device(args.device)
    model, classes = common.load_for_inference(args.run / "model.pt", device)
    tf = common.build_transforms(model, training=False)

    if args.image:
        paths = [args.image]
    else:
        paths = sorted(p for p in args.dir.iterdir() if p.suffix.lower() in common.IMG_EXTS)
    for p in paths:
        preds = predict(model, tf, classes, p, device, args.topk)
        print(f"{p.name}: " + ", ".join(f"{n}={c:.2f}" for n, c in preds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
