#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Semantic fusion: lift 2D detections into 3D objects in the splat's frame.

Reads COLMAP poses + sparse points and the detector's detections.json, estimates
a 3D anchor per detection, clusters multi-view anchors of the same class into
unique scene objects, and writes scene.json (consumed by the queryable viewer).

Anchor estimation without dense depth: a detection's 3D position is the median of
the COLMAP sparse points that reproject inside its box (and sit in front of the
camera). Robust, dependency-free, and good enough to place a marker on the object.

scene.json schema (matches docs/viewer/index.html):
  { "version":1, "splat":"./assets/scene.ply",
    "objects":[ {"id","label","category","keywords","position":[x,y,z],
                 "aabb":{"min","max"},"confidence","source":{"frames","support"}} ] }
Optional, additive (only when --titles is given and a member spine matched):
  "title" (matched book title), "text" (the OCR read behind the match); the
  title's tokens are also appended to "keywords" so search finds the book.

Example:
  python understanding/fusion/build_scene.py \
    --sparse data/office/sparse/0 --detections out/office/detections.json \
    --splat ./assets/scene.ply --out docs/viewer/assets/scene.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import colmap_io


def _points_array(points3D):
    ids = list(points3D.keys())
    if not ids:
        return ids, np.zeros((0, 3), dtype=np.float64)
    return ids, np.array([points3D[i].xyz for i in ids], dtype=np.float64)


def lift_detection(camera, image, xyz, box, min_points=4):
    """Median 3D of sparse points reprojecting inside `box` (x1,y1,x2,y2), in front.
    Returns (xyz | None, support_count)."""
    if len(xyz) == 0:
        return None, 0
    uv, z = colmap_io.project(camera, image, xyz)
    x1, y1, x2, y2 = box
    inside = (
        (z > 1e-6)
        & (uv[:, 0] >= x1) & (uv[:, 0] <= x2)
        & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
    )
    n = int(inside.sum())
    if n < min_points:
        return None, n
    return np.median(xyz[inside], axis=0), n


def attach_titles(detections, title_records, min_score=0.45, label="book"):
    """Annotate detections in place with d['title_match'] from ocr_titles.py output.

    titles.json keys crops as '<image stem>#<idx>' where idx counts only the
    label-matched ('book') boxes of that image, in file order (see
    ocr_titles.iter_crops_detections). Reproduce that count here to join.
    Caveat: ocr_titles also drops crops that clamp below 4 px, which would shift
    later indices for that image; that needs image dims we don't load here, so
    verify counts line up (they do for every capture so far: one record per box).
    Only matches with score >= min_score attach. Returns #attached.
    """
    by_source = {}
    for r in title_records:
        m = r.get("match")
        if m and m.get("title") and float(m.get("score") or 0) >= min_score:
            by_source[r["source"]] = {
                "title": m["title"],
                "text": (r.get("ocr") or {}).get("text") or "",
                "score": float(m["score"]),
            }
    counters, attached = {}, 0
    for d in detections:
        cls = (d.get("class") or d.get("category") or "").lower()
        if label not in cls:
            continue
        stem = Path(d["image"]).stem
        idx = counters.get(stem, 0)
        counters[stem] = idx + 1
        rec = by_source.get(f"{stem}#{idx}")
        if rec:
            d["title_match"] = rec
            attached += 1
    return attached


def cluster_objects(anchors, dist_thresh):
    """Greedy single-link clustering per class over 3D anchors."""
    objs = []
    for a in anchors:
        placed = False
        for o in objs:
            same_class = o["category"] == a["category"]
            near = np.linalg.norm(o["_sum"] / o["_n"] - a["xyz"]) <= dist_thresh
            if same_class and near:
                o["_sum"] = o["_sum"] + a["xyz"]
                o["_n"] += 1
                o["confidence"] = max(o["confidence"], a["confidence"])
                o["frames"].add(a["frame"])
                if a.get("ocr_text"):
                    o["ocr_texts"].add(a["ocr_text"])
                tm = a.get("title_match")
                if tm and (o["title_match"] is None
                           or tm["score"] > o["title_match"]["score"]):
                    o["title_match"] = tm
                o["_members"].append(a["xyz"])
                placed = True
                break
        if not placed:
            objs.append({
                "category": a["category"], "_sum": a["xyz"].copy(), "_n": 1,
                "confidence": a["confidence"], "frames": {a["frame"]},
                "ocr_texts": set([a["ocr_text"]] if a.get("ocr_text") else []),
                "title_match": a.get("title_match"),
                "_members": [a["xyz"]],
            })
    return objs


def fuse(cameras, images, points3D, detections, dist_thresh=0.5, min_points=4, min_frames=1):
    """Return a list of scene-object dicts. detections: list of
    {image, class, box_xyxy, confidence?, mask_png?, ocr_text?}."""
    _ids, xyz = _points_array(points3D)
    by_name = {Path(im.name).name: im for im in images.values()}

    anchors = []
    for d in detections:
        im = by_name.get(Path(d["image"]).name)
        if im is None:
            continue
        cam = cameras.get(im.camera_id)
        if cam is None:
            continue
        pos, _n = lift_detection(cam, im, xyz, d["box_xyxy"], min_points)
        if pos is None:
            continue
        anchors.append({
            "category": d.get("class") or d.get("category"),
            "xyz": np.asarray(pos, dtype=np.float64),
            "confidence": float(d.get("confidence", 1.0)),
            "frame": Path(d["image"]).name,
            "ocr_text": d.get("ocr_text"),
            "title_match": d.get("title_match"),
        })

    objects = []
    clustered = sorted(cluster_objects(anchors, dist_thresh), key=lambda c: -c["confidence"])
    for i, o in enumerate(clustered):
        if len(o["frames"]) < min_frames:
            continue
        members = np.array(o["_members"])
        centroid = members.mean(axis=0)
        label = sorted(o["ocr_texts"])[0] if o["ocr_texts"] else o["category"]
        kws = {o["category"]}
        for t in o["ocr_texts"]:
            kws.update(w.lower() for w in t.split() if len(w) > 2)
        tm = o.get("title_match")
        if tm:
            kws.update(w.lower() for w in tm["title"].split() if len(w) > 2)
        obj = {
            "id": f"obj_{i:04d}",
            "label": label,
            "category": o["category"],
            "keywords": sorted(kws),
            "position": [round(float(v), 5) for v in centroid],
            "aabb": {
                "min": [round(float(v), 5) for v in members.min(axis=0)],
                "max": [round(float(v), 5) for v in members.max(axis=0)],
            },
            "confidence": round(float(o["confidence"]), 4),
            "source": {"frames": sorted(o["frames"]), "support": int(o["_n"])},
        }
        if tm:  # additive, optional: present only when a member spine matched
            obj["title"] = tm["title"]
            obj["text"] = tm["text"]
        objects.append(obj)
    return objects


def build_scene_json(objects, splat_path):
    return {"version": 1, "splat": splat_path, "objects": objects}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", required=True, type=Path, help="COLMAP sparse/0 dir")
    ap.add_argument("--detections", required=True, type=Path, help="detections.json")
    ap.add_argument("--splat", default="./assets/scene.ply", help="splat path for the viewer")
    ap.add_argument("--out", type=Path, default=Path("docs/viewer/assets/scene.json"))
    ap.add_argument("--dist-thresh", type=float, default=0.5, help="merge radius (scene units)")
    ap.add_argument("--min-points", type=int, default=4, help="min sparse points to place a detection")
    ap.add_argument("--min-frames", type=int, default=1, help="min frames to keep an object")
    ap.add_argument("--titles", type=Path, default=None,
                    help="titles.json from ocr_titles.py; attaches matched book "
                         "titles to fused objects (optional 'title'/'text' fields)")
    ap.add_argument("--min-title-score", type=float, default=0.45,
                    help="min fuzzy match score for a title to attach")
    args = ap.parse_args()

    cameras, images, points3D = colmap_io.read_model(args.sparse)
    detections = json.loads(args.detections.read_text())
    if args.titles:
        n = attach_titles(detections, json.loads(args.titles.read_text(encoding="utf-8")),
                          args.min_title_score)
        print(f"Attached {n} titled spines from {args.titles}")
    objects = fuse(cameras, images, points3D, detections,
                   args.dist_thresh, args.min_points, args.min_frames)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(build_scene_json(objects, args.splat), indent=2))

    print(f"Fused {len(detections)} detections -> {len(objects)} objects -> {args.out}")
    for o in objects:
        print(f"  {o['id']} [{o['category']}] '{o['label']}' @ {o['position']} "
              f"(conf {o['confidence']}, {o['source']['support']} pts, "
              f"{len(o['source']['frames'])} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
