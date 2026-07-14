#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# LICENSE NOTE (deliberate): this file imports Ultralytics YOLOE, which is
# AGPL-3.0. Under AGPL, code importing it forms a combined work, so THIS FILE is
# AGPL-3.0 -- not MIT like the rest of the repo. That boundary is intentional and
# isolated: nothing else in the repo imports it. For a closed product or a hosted
# service, either buy an Ultralytics Enterprise license, or swap this stage for an
# Apache-2.0 detector (OWLv2 via HF transformers, or GroundingDINO + SAM2).
# See ARCHITECTURE.md ("License hygiene").
"""
Open-vocabulary detection + segmentation over capture frames (zero-shot).

Type the class names you care about -- no training. YOLOE finds and masks them,
writing per-object mask PNGs and a single detections.json that the classifier
and OCR stages consume.

Example:
  python detect.py --frames data/office/frames \
      --classes "book,chair,potted plant,laptop,lamp" --out out/office
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, type=Path, help="folder of frames")
    ap.add_argument("--classes", required=True,
                    help="comma-separated open-vocab class names")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--weights", default="yoloe-11l-seg.pt",
                    help="yoloe-11l-seg.pt (best) / yoloe-11s-seg.pt (8GB/CPU-friendly)")
    ap.add_argument("--imgsz", type=int, default=640, help="multiple of 32")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="auto", help="'0' for GPU, 'cpu', or 'auto'")
    ap.add_argument("--no-masks", action="store_true", help="boxes only, skip masks")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLOE

    if args.device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if str(device) != "cpu" and not torch.cuda.is_available():
        print("WARNING: GPU requested but torch.cuda.is_available()==False "
              "(CPU-only torch). Running on CPU. Install CUDA torch for the RTX 4070.")
        device = "cpu"

    names = [c.strip() for c in args.classes.split(",") if c.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    mask_dir = args.out / "masks"
    want_masks = not args.no_masks
    if want_masks:
        mask_dir.mkdir(exist_ok=True)

    model = YOLOE(args.weights)
    # YOLOE needs names + text embeddings (2-arg form). get_text_pe() pulls
    # MobileCLIP on first run (needs internet + git once).
    model.set_classes(names, model.get_text_pe(names))

    results = model.predict(
        source=str(args.frames), imgsz=args.imgsz, conf=args.conf,
        retina_masks=want_masks, device=device, stream=True, verbose=False,
    )

    import cv2
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        stem = Path(r.path).stem
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        masks = None if (not want_masks or r.masks is None) else r.masks.data.cpu().numpy()
        for i in range(len(boxes)):
            cname = r.names[cls[i]]
            rec = {
                "image": r.path,
                "class": cname,
                "class_id": int(cls[i]),
                "confidence": round(float(conf[i]), 4),
                "box_xyxy": [round(float(v), 1) for v in xyxy[i]],
            }
            if masks is not None:
                binm = (masks[i] > 0.5).astype(np.uint8) * 255
                mp = mask_dir / f"{stem}_{i:02d}_{cname.replace(' ', '_')}.png"
                cv2.imwrite(str(mp), binm)
                rec["mask_png"] = str(mp)
            detections.append(rec)

    (args.out / "detections.json").write_text(json.dumps(detections, indent=2))
    by_class: dict[str, int] = {}
    for d in detections:
        by_class[d["class"]] = by_class.get(d["class"], 0) + 1
    print(f"{len(detections)} detections across classes {by_class}")
    print(f"-> {args.out / 'detections.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
