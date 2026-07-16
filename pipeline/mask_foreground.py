#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0
# NOTE: imports Ultralytics YOLOE + SAM2 (AGPL-3.0), like understanding/detect.py. Isolated
# here so the rest of the pipeline stays MIT. Swap for a printed alpha matte or an
# Apache-2.0 segmenter to relicense.
"""
Mask the foreground object out of every frame, so a splat trainer that honours input
transparency will not reconstruct the room around it.

Why this exists, measured on the shelf capture: an object against a wall in a tight room
forces every camera to stare at the front within a ~16 deg cone. The far background (wall,
floor) and the near-field air are therefore weakly triangulated, and 3DGS fills them with
faint, stretched Gaussians that render as fog from any novel angle. No post-hoc position
cull removes them (they are anchored to the surface and stretch outward). The fix is upstream:
delete the background from the input. Brush then "forces the final splat to match the
transparency of the input" (its README) -- rays through a transparent pixel are penalised for
any opacity, so the haze cannot form. Removing the background also removes whatever was on the
walls from the reconstruction -- frequently a privacy win.

The silhouette must be ACCURATE, and this was learned the hard way. A first version used a
loose YOLOE mask solidified by contour fill; that filled the CONCAVE background (the gap
between two shelf units, the wall above) as if it were foreground, and Brush dutifully
rebuilt the wall as a brown slab -- worse than no mask. The fix is a true silhouette:

  1. YOLOE (open-vocab) proposes boxes for the shelf CONTENTS. The vocabulary matters: plain
     "toy" misses a Lego R2-D2; "robot"/"action figure"/"figure"/"statue" catch the top
     pieces. "picture frame"/"sign" are deliberately OMITTED so wall art is never grabbed.
  2. SAM2 is prompted with EACH box (not one union box -- one box segments only the dominant
     object, leaving a second shelf unit ragged) and the per-box masks are unioned. This
     follows the real outline and leaves concave background as background.
  3. A small morphological close bridges adjacent objects; a small dilate avoids clipping
     edges. No contour fill -- that was the bug.

Two guardrails, because a wrongly-transparent frame actively teaches the splat to be empty
there: a frame with no boxes, or a near-empty SAM result, is kept FULLY OPAQUE rather than
blanked; a frame whose matte already covers almost everything is a close-up of the subject
and is likewise left opaque (no background to remove).

Output is RGBA PNG (original RGB + matte alpha), same stem, ready for a transparency-aware
trainer such as Brush (`--match-alpha-weight`, default 0.1, is the right strength; higher
trades away photometric detail).

Usage:
    python pipeline/mask_foreground.py --images data/shelf/recon/images \
        --out data/shelf/recon/images_masked
Dependencies: ultralytics (YOLOE + SAM2), opencv-python, numpy. CPU is fine
(~0.3 s/frame YOLOE + ~2 s/frame SAM2; ~18 min for 354 frames). SAM2 weights auto-download.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np

# shelf CONTENTS only. no "picture frame"/"sign" -> never grab wall art.
# "robot"/"action figure"/"figure"/"statue" catch top toys that plain "toy" misses.
DEFAULT_CLASSES = [
    "bookcase", "shelf", "book", "toy", "figurine", "box", "bottle", "vase",
    "lego", "model", "sculpture", "robot", "action figure", "figure", "statue", "bird",
]


def matte_from_masks(masks, h, w, *, close_px, dilate_px):
    import cv2
    matte = np.zeros((h, w), np.uint8)
    if masks is not None:
        for m in masks:
            matte |= cv2.resize((m > 0.5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) * 255
    matte = cv2.morphologyEx(matte, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    return cv2.dilate(matte, np.ones((dilate_px, dilate_px), np.uint8))


def main() -> int:
    ap = argparse.ArgumentParser(description="Matte the foreground for transparency-aware splat training.")
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--yolo-weights", default="yoloe-11s-seg.pt")
    ap.add_argument("--sam-weights", default="sam2_b.pt")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--min-box-frac", type=float, default=0.004, help="drop YOLOE boxes smaller than this frac of frame")
    ap.add_argument("--fail-cov", type=float, default=0.06, help="SAM matte below this -> keep frame opaque (never blank)")
    ap.add_argument("--closeup-cov", type=float, default=0.92, help="matte above this -> a close-up; leave opaque")
    ap.add_argument("--close-px", type=int, default=25)
    ap.add_argument("--dilate-px", type=int, default=7)
    ap.add_argument("--qa-every", type=int, default=30)
    args = ap.parse_args()

    if not args.images.is_dir():
        print(f"error: {args.images} not found", file=sys.stderr)
        return 2
    import cv2
    from ultralytics import SAM, YOLOE

    frames = sorted([p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not frames:
        print(f"error: no frames in {args.images}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "_qa"
    qa.mkdir(exist_ok=True)

    yolo = YOLOE(args.yolo_weights)
    yolo.set_classes(args.classes, yolo.get_text_pe(args.classes))
    sam = SAM(args.sam_weights)

    counts: dict[str, int] = {}
    for k, f in enumerate(frames):
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        r = yolo.predict(str(f), imgsz=args.imgsz, conf=args.conf, retina_masks=False,
                         device=args.device, verbose=False)[0]
        mode = "masked"
        if r.boxes is None or len(r.boxes) == 0:
            alpha = np.full((h, w), 255, np.uint8)
            mode = "fail-opaque"
        else:
            xy = r.boxes.xyxy.cpu().numpy()
            areas = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
            boxes = xy[areas > args.min_box_frac * h * w].tolist()
            if not boxes:
                boxes = [[xy[:, 0].min(), xy[:, 1].min(), xy[:, 2].max(), xy[:, 3].max()]]
            s = sam(str(f), bboxes=boxes, device=args.device, verbose=False)[0]
            masks = None if s.masks is None else s.masks.data.cpu().numpy()
            alpha = matte_from_masks(masks, h, w, close_px=args.close_px, dilate_px=args.dilate_px)
            cov = alpha.mean() / 255
            if cov < args.fail_cov:
                alpha = np.full((h, w), 255, np.uint8)
                mode = "fail-opaque"
            elif cov > args.closeup_cov:
                mode = "closeup-opaque"
        counts[mode] = counts.get(mode, 0) + 1
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha
        # Zero the background RGB, do not leave the real wall there. Brush composites the
        # rendered splat over black and compares to the input, so a transparent pixel that
        # still shows wall makes the photometric loss fight the alpha loss -- the compromise
        # is a blurred shelf. Measured: leaving the wall RGB softened the whole reconstruction;
        # zeroing it is the standard alpha-matte convention and restores sharpness.
        rgba[alpha == 0, :3] = 0
        cv2.imwrite(str(args.out / f"{f.stem}.png"), rgba)
        if k % args.qa_every == 0:
            ov = img.copy()
            ov[alpha == 0] = (ov[alpha == 0] * 0.2).astype(np.uint8)
            cv2.imwrite(str(qa / f"{f.stem}_ov.jpg"), ov)
        if k % 50 == 0:
            print(f"  {k:>4}/{len(frames)}  {f.name}  [{mode}]", flush=True)

    print("\nsummary:")
    for m, c in sorted(counts.items()):
        print(f"  {m:<16}: {c:>4}  ({c/len(frames):.0%})")
    print(f"wrote {len(frames)} RGBA PNGs to {args.out}  (QA overlays in {qa})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
