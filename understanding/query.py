#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Query a fused scene.json from the terminal (or as a machine interface for an agent).

Mirrors the viewer's structured scoring so terminal and browser agree: the query
is tokenized and each object scored by category/label/keyword matches, then ranked.
Dependency-free (stdlib only). With --json it emits a structured result an LLM or
another tool can consume ("find every book on the shelf" -> ids + 3D positions).

Examples:
  python understanding/query.py scene.json "book"
  python understanding/query.py scene.json "potted plant" --top 3 --json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _haystack(o):
    kws = [str(k).lower() for k in o.get("keywords", [])]
    parts = [o.get("label", ""), o.get("category", ""), *kws]
    return " ".join(p for p in parts if p).lower()


def score_object(o, tokens):
    """Same rubric as the viewer: exact category/label 3, label-prefix 2.2,
    exact keyword 2, substring anywhere 1; averaged over tokens."""
    if not tokens:
        return 0.0
    cat = (o.get("category") or "").lower()
    label = (o.get("label") or "").lower()
    kws = [str(k).lower() for k in o.get("keywords", [])]
    hay = _haystack(o)
    s = 0.0
    for tk in tokens:
        if not tk:
            continue
        if cat and cat == tk:
            s += 3
        elif label == tk:
            s += 3
        elif label.startswith(tk):
            s += 2.2
        elif tk in kws:
            s += 2
        elif tk in hay:
            s += 1
    return s / len(tokens)


def query(objects, q, top=None, min_score=1e-9):
    tokens = q.lower().split()
    hits = []
    for o in objects:
        s = score_object(o, tokens)
        if s > min_score:
            hits.append({"id": o.get("id"), "label": o.get("label"),
                         "category": o.get("category"), "position": o.get("position"),
                         "score": round(s, 3)})
    hits.sort(key=lambda h: -h["score"])
    return hits[:top] if top else hits


def load_scene(path):
    # utf-8-sig tolerates a BOM (some editors add one).
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return data if isinstance(data, list) else data.get("objects", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path, help="scene.json")
    ap.add_argument("query", help="search text, e.g. 'book' or 'potted plant'")
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--json", action="store_true", help="emit JSON (for an agent/pipe)")
    args = ap.parse_args()

    objects = load_scene(args.scene)
    hits = query(objects, args.query, top=args.top)
    if args.json:
        print(json.dumps(hits, indent=2))
        return 0
    if not hits:
        print(f"no matches for {args.query!r} among {len(objects)} objects")
    for h in hits:
        pos = ", ".join(f"{v:.2f}" for v in (h["position"] or []))
        print(f"  {h['score']:.1f}  [{h['category']}] {h['label']}  @ ({pos})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
