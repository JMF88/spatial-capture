#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Enrich a scene.json's per-object `keywords` with category synonyms so the
viewer's existing keyword search generalizes -- "toy" finds figures + Lego,
"photo" finds picture frames -- without any viewer-code change. The expansion
lives in the data. Pure stdlib; run after fusion (with or without titles).

    python understanding/enrich_keywords.py docs/viewer/assets/scene.public.json
    python understanding/enrich_keywords.py in.json --out out.json   # non-destructive
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# category -> extra search terms. Kept deliberately conservative and honest:
# only synonyms a person would actually type for that object class.
SYNONYMS = {
    "book": ["book", "novel", "hardcover", "read", "reading"],
    "figurine": ["figurine", "figure", "toy", "collectible", "character"],
    "lego model": ["lego", "model", "build", "toy", "brick", "set"],
    "box": ["box", "container", "case"],
    "statue": ["statue", "figure", "sculpture", "bust"],
    "owl": ["owl", "bird", "figurine", "toy"],
    "picture frame": ["picture frame", "photo", "picture", "frame", "portrait"],
    "shelf": ["shelf", "ledge", "board"],
}


def enrich(scene: dict) -> int:
    """Add synonyms to each object's keywords (dedup, lowercase). Returns count changed."""
    changed = 0
    for o in scene.get("objects", []):
        kw = [str(k).lower() for k in o.get("keywords", [])]
        extra = SYNONYMS.get(o.get("category", ""), [])
        merged = list(dict.fromkeys(kw + [e.lower() for e in extra]))  # order-preserving dedup
        if merged != kw:
            o["keywords"] = merged
            changed += 1
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--out", type=Path, help="default: in place")
    args = ap.parse_args()
    scene = json.loads(args.path.read_text(encoding="utf-8"))
    n = enrich(scene)
    out = args.out or args.path
    out.write_text(json.dumps(scene, indent=1), encoding="utf-8")
    print(f"enriched {n} objects -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
