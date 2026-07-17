#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Feature B — read BOOK SPINE titles, then look up real book metadata.

Pipeline for one spine crop:

    crop ->  OCR at 3 orientations (0, +90, -90)  ->  keep the highest-
             confidence read  ->  clean to a query  ->  free book-metadata
             API (Open Library, Google Books fallback)  ->  fuzzy-match the
             returned title back to the OCR text  ->  titles.json

Why this shape: spine text is almost always printed vertically, and which way
(90 vs 270) is not knowable up front. Rather than trust one auto-rotation, we
run the OCR on the crop rotated *both* ways plus upright, score each read by
confidence x characters, and keep the winner. That single trick does most of
the work for spines and needs no orientation classifier.

OCR ENGINE CHOICE (Windows, CPU or an 8 GB RTX 4070 — verified 2026-07):
  * EasyOCR 1.7.2  <- chosen. `pip install easyocr` and it just runs; models
    auto-download once (~100 MB). Has `rotation_info` built in, but we drive
    rotation ourselves so we get a comparable confidence per orientation.
    Note: EasyOCR declares torch/torchvision as deps and will PULL ITS OWN
    torch if none is present — on Windows that means the CPU-only wheel and an
    idle GPU. This venv already has torch, so install with `--no-deps` or just
    let pip see the existing torch (see install notes below).
  * PaddleOCR 3.x (PP-OCRv5) — highest accuracy, but on Windows it wants the
    paddlepaddle(-gpu) runtime, which is a heavier/fussier install than torch.
    Great choice if you already live in the Paddle stack; overkill here.
  * docTR (python-doctr[torch]) — clean PyTorch OCR, reuses our torch, but it
    is tuned for flat documents; rotated single-word spines are not its sweet
    spot and it needs more coaxing than EasyOCR for this job.
  * A small VLM (e.g. a 2-3B vision model via Ollama/transformers) can read a
    spine from a plain prompt and is the most robust to weird typography, but
    it is the heaviest option for 8 GB VRAM and the least deterministic. Kept
    as a future swap, not the default.
All four are drop-in behind `ocr_spine()`; only that one function is engine-
specific.

METADATA APIS (both FREE, NO API KEY):
  * Open Library  https://openlibrary.org/search.json?q=...   (default)
  * Google Books  https://www.googleapis.com/books/v1/volumes?q=...  (fallback;
    keyless queries are rate-limited but fine for a demo)

INPUT (one of):
  --crops   DIR            a folder of pre-cropped spine images (one book each)
  --detections FILE.json   detector output; we crop every 'book' box ourselves
                           (pair with --frames-root so image paths resolve)
  --from-titles FILE.json  an earlier titles.json; reuses its OCR reads and
                           re-runs only the metadata lookup (no OCR, no GPU --
                           the cheap way to iterate on retrieval/matching)

OUTPUT:
  titles.json  — one record per spine: the OCR read + the matched book.

Examples:
  # A) straight from a folder of spine crops
  python understanding/ocr_titles.py --crops data/spines --out titles.json

  # B) from detector boxes (understanding/detect.py output), cropping 'book'
  python understanding/ocr_titles.py --detections data/office/detections.json \\
      --frames-root data/office/frames --out data/office/titles.json --provider both

Deps (already in understanding/requirements.txt): easyocr, requests, opencv-python, numpy.
Install torch FIRST so EasyOCR does not drag in the CPU-only wheel:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  pip install easyocr requests            # sees the torch already present
First run needs internet once (EasyOCR model download + the book APIs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import cv2
import numpy as np
import requests

# Run as a script (python understanding/ocr_titles.py), so this dir isn't importable
# as a package; matching.py sits beside this file. Kept torch/cv2-free on purpose so
# the precision policy is unit-testable without the ML stack — same reason splits.py
# is split out of the classifier's common.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from matching import is_queryable, match_score, walk_ladder  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
USER_AGENT = "spatial-capture/1.0 (book-spine OCR demo; https://github.com/jmf88/spatial-capture)"

# Orientations to try, as (label, cv2 rotate code or None). Upright first, then
# the two vertical reading directions ("both ways", per the spine requirement).
_ORIENTATIONS = [
    ("0", None),
    ("90cw", cv2.ROTATE_90_CLOCKWISE),
    ("90ccw", cv2.ROTATE_90_COUNTERCLOCKWISE),
]


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
def want_gpu(device: str) -> bool:
    """Map --device to EasyOCR's gpu flag, warning on a CPU-only torch."""
    if device == "cpu":
        return False
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    if device == "cuda" and not cuda:
        print("WARNING: --device cuda but torch.cuda.is_available() is False "
              "(CPU-only torch?). Running EasyOCR on CPU.", file=sys.stderr)
    return cuda  # auto/cuda -> use GPU iff really available


def load_reader(langs, gpu: bool):
    """Build the EasyOCR reader once (heavy: imports torch, may download models)."""
    import easyocr  # lazy so --help / lookup-only paths stay light
    print(f"- loading EasyOCR (langs={langs}, gpu={gpu}) ...", file=sys.stderr)
    return easyocr.Reader(langs, gpu=gpu)


def _upscale_min_side(img, min_side: int):
    """EasyOCR's detector struggles on tiny crops; gently upscale small ones."""
    h, w = img.shape[:2]
    s = min(h, w)
    if s and s < min_side:
        f = min_side / s
        img = cv2.resize(img, (int(round(w * f)), int(round(h * f))),
                         interpolation=cv2.INTER_CUBIC)
    return img


def _order_reads(reads):
    """Reading order for the winning orientation: top-to-bottom by lines,
    left-to-right within a line. `reads` is EasyOCR's [(bbox, text, conf), ...]."""
    items = []
    for bbox, text, conf in reads:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        items.append((min(ys), (max(ys) - min(ys)) or 1, min(xs), text))
    if not items:
        return ""
    line_h = np.median([h for _, h, _, _ in items])
    tol = 0.6 * line_h
    items.sort(key=lambda t: (round(t[0] / max(tol, 1)), t[2]))
    return " ".join(t[3].strip() for t in items if t[3].strip())


def ocr_spine(reader, img_bgr, min_side: int):
    """Read a spine crop at 3 orientations; keep the best (conf x #chars).

    Returns (text, mean_confidence, angle_label). Swap the body of this one
    function to switch OCR engines; nothing else in the file is engine-specific.
    """
    best = ("", 0.0, "0", -1.0)  # text, conf, angle, score
    for label, code in _ORIENTATIONS:
        rot = img_bgr if code is None else cv2.rotate(img_bgr, code)
        rot = _upscale_min_side(rot, min_side)
        rgb = cv2.cvtColor(rot, cv2.COLOR_BGR2RGB)  # EasyOCR expects RGB
        reads = reader.readtext(rgb, detail=1, paragraph=False)
        if not reads:
            continue
        # Score rewards a long, confident read over a short lucky one.
        score = sum(float(c) * len(t.strip()) for _, t, c in reads)
        text = _order_reads(reads)
        mean_conf = float(np.mean([float(c) for _, _, c in reads]))
        if score > best[3] and text:
            best = (text, mean_conf, label, score)
    return best[0], best[1], best[2]


# --------------------------------------------------------------------------- #
# Query cleaning + metadata lookup (free, keyless)
# --------------------------------------------------------------------------- #
def clean_query(text: str) -> str:
    """Normalize an OCR read into a search query: collapse noise/whitespace,
    keep letters/digits/spaces, drop 1-char junk tokens."""
    t = re.sub(r"[^0-9A-Za-z&'\-\s]", " ", text)
    toks = [w for w in t.split() if len(w) > 1 or w.isdigit()]
    return " ".join(toks).strip()



class Lookup:
    """Book-metadata lookups with an on-disk cache and polite pacing."""

    def __init__(self, cache_dir: Path, sleep: float, timeout: float):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep = sleep
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT})
        self.calls_live = 0
        self.calls_cached = 0

    def _get_json(self, url: str):
        key = hashlib.sha1(url.encode("utf-8")).hexdigest()
        cf = self.cache_dir / f"{key}.json"
        if cf.exists():
            try:
                data = json.loads(cf.read_text(encoding="utf-8"))
                self.calls_cached += 1
                return data
            except Exception:
                pass
        self.calls_live += 1
        try:
            r = self.s.get(url, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as e:  # network/HTTP/JSON — treat as "no match"
            print(f"  ! lookup failed ({e}) for {url}", file=sys.stderr)
            if self.sleep:
                time.sleep(self.sleep)
            return None  # don't cache transient failures (e.g. a 429)
        cf.write_text(json.dumps(data), encoding="utf-8")
        if self.sleep:
            time.sleep(self.sleep)  # politeness on a live (uncached) fetch
        return data

    def openlibrary_docs(self, kind: str, query: str):
        """One Open Library search attempt -> candidate payloads (unscored).

        `kind` comes from matching.build_query_ladder: "q" is the plain AND
        keyword search (URL identical to earlier versions of this tool, so the
        on-disk cache keeps paying), "q_or" is the Solr OR-union (wider net, so
        ask for more docs and let scoring pick), "title" restricts AND matching
        to the title field.
        """
        fields = "title,author_name,first_publish_year,isbn,key,cover_i,edition_count"
        if kind == "title":
            params = {"title": query, "limit": 5, "fields": fields}
        elif kind == "q_or":
            # The union is wide by construction; ask for more docs so the true
            # title is IN the candidate set -- scoring, not rank, then decides.
            params = {"q": query, "limit": 20, "fields": fields}
        else:
            params = {"q": query, "limit": 5, "fields": fields}
        url = "https://openlibrary.org/search.json?" + urlencode(params)
        data = self._get_json(url)
        out = []
        for d in (data or {}).get("docs", []):
            cover = d.get("cover_i")
            out.append({
                "provider": "openlibrary",
                "title": d.get("title"),
                "authors": d.get("author_name") or [],
                "first_publish_year": d.get("first_publish_year"),
                "isbn": (d.get("isbn") or [None])[0],
                "work_key": d.get("key"),
                "url": ("https://openlibrary.org" + d["key"]) if d.get("key") else None,
                "cover_url": (f"https://covers.openlibrary.org/b/id/{cover}-M.jpg"
                              if cover else None),
            })
        return out

    def googlebooks(self, query: str):
        url = "https://www.googleapis.com/books/v1/volumes?" + urlencode(
            {"q": query, "maxResults": 5, "country": "US"})
        data = self._get_json(url)
        best, best_sc = None, -1.0
        for it in (data or {}).get("items", []):
            vi = it.get("volumeInfo", {})
            sc = match_score(query, vi.get("title", ""))
            if sc > best_sc:
                ids = {x.get("type"): x.get("identifier")
                       for x in vi.get("industryIdentifiers", [])}
                best_sc, best = sc, {
                    "provider": "googlebooks",
                    "title": vi.get("title"),
                    "authors": vi.get("authors") or [],
                    "first_publish_year": (vi.get("publishedDate") or "")[:4] or None,
                    "isbn": ids.get("ISBN_13") or ids.get("ISBN_10"),
                    "work_key": it.get("id"),
                    "url": vi.get("infoLink"),
                    "cover_url": (vi.get("imageLinks") or {}).get("thumbnail"),
                    "score": sc,
                }
        return best

    def best(self, query: str, provider: str, min_match: float):
        """Resolve one read per --provider; None unless a match clears min_match.

        Open Library runs the bounded degradation ladder (matching.walk_ladder):
        full query -> OR-union of content tokens -> title-field query -> clean
        token pairs, stopping at the first attempt whose best doc matches the
        FULL read at min_match. Google Books stays a single full-query call
        (keyless quota is tiny; it is a fallback, not a peer).
        """
        if provider in ("openlibrary", "both"):
            m, _tried = walk_ladder(query, self.openlibrary_docs, min_match)
            if m:
                return m
        if provider in ("googlebooks", "both"):
            g = self.googlebooks(query)
            if g and g["score"] >= min_match:
                return g
        return None


# --------------------------------------------------------------------------- #
# Inputs: a folder of crops, or 'book' boxes from detections.json
# --------------------------------------------------------------------------- #
def _clamp_box(box, w, h, pad=0.02):
    x1, y1, x2, y2 = box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1 = int(max(0, x1 - px))
    y1 = int(max(0, y1 - py))
    x2 = int(min(w, x2 + px))
    y2 = int(min(h, y2 + py))
    return x1, y1, x2, y2


def iter_crops_folder(folder: Path):
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  ! could not read {p}", file=sys.stderr)
            continue
        yield p.stem, img


def _box_of(det):
    return det.get("box") or det.get("bbox") or det.get("xyxy") or det.get("box_xyxy")


def _label_of(det):
    return (det.get("label") or det.get("name") or det.get("class") or "").lower()


def _normalize_detections(data):
    """Group detections by image, tolerating several JSON shapes:

      * FLAT (this repo's detect.py): [{"image": p, "class": "book",
        "box_xyxy": [x1,y1,x2,y2], "confidence": ..}, ...] -- one record per box.
      * GROUPED list: [{"image": p, "detections": [{"box":[..],"label":..}, ..]}, ..]
      * MAPPING: {"<image path>": [{"box":[..],"label":..}, ..], ..}

    Box key may be box / bbox / xyxy / box_xyxy; label may be label / name / class.
    Yields (image_path_str, [(box, label, conf), ...]).
    """
    if isinstance(data, dict) and "images" in data:
        data = data["images"]
    grouped: dict = {}

    def add(img_path, det):
        box = _box_of(det)
        if img_path and box and len(box) == 4:
            grouped.setdefault(img_path, []).append(
                (box, _label_of(det), det.get("conf") or det.get("confidence")))

    if isinstance(data, dict):
        for img_path, dets in data.items():
            for det in dets:
                add(img_path, det)
    else:
        for d in data:
            nested = d.get("detections") or d.get("boxes")
            img_path = d.get("image") or d.get("path")
            if isinstance(nested, list):
                for det in nested:
                    add(img_path, det)
            else:  # flat: the record itself is a single detection
                add(img_path, d)

    for img_path, boxes in grouped.items():
        yield img_path, boxes


def iter_crops_detections(det_file: Path, frames_root: Path | None, label_filter: str):
    data = json.loads(det_file.read_text(encoding="utf-8"))
    lf = label_filter.lower()
    for img_path, boxes in _normalize_detections(data):
        if not img_path:
            continue
        p = Path(img_path)
        for cand in ([p] if p.is_absolute() else
                     [p, det_file.parent / p] + ([frames_root / p.name, frames_root / p]
                                                 if frames_root else [])):
            if cand.exists():
                p = cand
                break
        img = cv2.imread(str(p))
        if img is None:
            print(f"  ! could not read image for detection: {img_path}", file=sys.stderr)
            continue
        h, w = img.shape[:2]
        idx = 0
        for box, label, _conf in boxes:
            if lf and lf not in label:  # keep only 'book' (substring match)
                continue
            x1, y1, x2, y2 = _clamp_box(box, w, h)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            yield f"{Path(img_path).stem}#{idx}", img[y1:y2, x1:x2]
            idx += 1


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="OCR book-spine titles and match them to free book metadata.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--crops", type=Path, help="folder of pre-cropped spine images")
    src.add_argument("--detections", type=Path,
                     help="detections.json; 'book' boxes are cropped for OCR")
    src.add_argument("--from-titles", type=Path,
                     help="re-run only the metadata lookup over an existing "
                          "titles.json (reuses its OCR reads; no OCR, no GPU)")
    ap.add_argument("--frames-root", type=Path, default=None,
                    help="dir the detections.json image paths live in (box mode)")
    ap.add_argument("--label", default="book",
                    help="detection label to OCR (substring match); box mode")
    ap.add_argument("--out", type=Path, default=Path("titles.json"))
    ap.add_argument("--checkpoint", type=int, default=100,
                    help="write titles.json every N spines so a killed run keeps "
                         "its progress (0 = write only at the end)")
    ap.add_argument("--provider", choices=["openlibrary", "googlebooks", "both"],
                    default="openlibrary")
    ap.add_argument("--langs", nargs="+", default=["en"], help="EasyOCR languages")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--min-conf", type=float, default=0.30,
                    help="skip a spine whose best OCR mean-confidence is below this")
    ap.add_argument("--min-match", type=float, default=0.45,
                    help="min fuzzy title-match [0-1] to accept a book")
    ap.add_argument("--min-side", type=int, default=64,
                    help="upscale crops whose shorter side is below this (px)")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/ocr-cache"),
                    help="on-disk cache for API responses (data/ is gitignored)")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="seconds between live API calls (politeness)")
    ap.add_argument("--debug-crops", type=Path, default=None,
                    help="optional dir to dump each crop for inspection")
    args = ap.parse_args()

    if args.detections and not args.detections.exists():
        print(f"ERROR: detections file not found: {args.detections}", file=sys.stderr)
        return 2
    if args.crops and not args.crops.exists():
        print(f"ERROR: crops folder not found: {args.crops}", file=sys.stderr)
        return 2
    if args.from_titles and not args.from_titles.exists():
        print(f"ERROR: titles file not found: {args.from_titles}", file=sys.stderr)
        return 2

    lookup = Lookup(args.cache_dir, args.sleep, timeout=15.0)
    if args.debug_crops:
        args.debug_crops.mkdir(parents=True, exist_ok=True)

    if args.from_titles:
        # Lookup-only replay: the OCR pass is the expensive step (GPU, ~minutes);
        # retrieval policy changes should not repeat it.
        prev = json.loads(args.from_titles.read_text(encoding="utf-8"))

        def reads():
            for r in prev:
                o = r.get("ocr") or {}
                yield (r.get("source", ""), o.get("text", ""),
                       float(o.get("confidence") or 0.0), o.get("angle", "0"),
                       o.get("engine", "easyocr"))
    else:
        reader = load_reader(args.langs, want_gpu(args.device))
        if args.crops:
            source = iter_crops_folder(args.crops)
        else:
            source = iter_crops_detections(args.detections, args.frames_root,
                                           args.label)

        def reads():
            for sid, crop in source:
                if args.debug_crops is not None:
                    cv2.imwrite(str(args.debug_crops / f"{sid.replace('#', '_')}.png"),
                                crop)
                text, conf, angle = ocr_spine(reader, crop, args.min_side)
                yield sid, text, conf, angle, "easyocr"

    args.out.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        # Atomic checkpoint: write to a temp then replace, so an interrupted run
        # never leaves a half-written titles.json and never loses everything. The
        # OCR pass is the expensive step (GPU, minutes-to-hours); writing only at
        # the very end means one kill throws all of it away.
        tmp = args.out.with_name(args.out.name + ".tmp")
        tmp.write_text(json.dumps(records, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(args.out)

    records, matched = [], 0
    for i, (sid, text, conf, angle, engine) in enumerate(reads()):
        rec = {"source": sid,
               "ocr": {"text": text, "confidence": round(conf, 3),
                       "angle": angle, "engine": engine},
               "query": "", "match": None}
        if text and conf >= args.min_conf:
            q = clean_query(text)
            rec["query"] = q
            if q and is_queryable(q):
                m = lookup.best(q, args.provider, args.min_match)
                rec["match"] = m
                if m:
                    matched += 1
            elif q:
                rec["skipped"] = "too little alphabetic signal to look up"
        m = rec["match"]
        tag = (f"-> {m['title']} ({m['provider']} {m['score']})" if m
               else "-> (no confident match)")
        print(f"[{i:03d}] {sid}: '{text}' conf={conf:.2f} @{angle} {tag}")
        records.append(rec)
        if args.checkpoint and (i + 1) % args.checkpoint == 0:
            flush()

    flush()
    print(f"\n{matched}/{len(records)} spines matched. Wrote {args.out} "
          f"(API: {lookup.calls_live} live, {lookup.calls_cached} cached)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
