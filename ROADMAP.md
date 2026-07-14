# Roadmap

The goal is open-ended: push each stage as far as it profitably goes, and let the
demo be whatever the best push produces. This file tracks what's built, what's next,
and where it heads. It doubles as the honest scope statement for the repo.

## Status snapshot

| Stage | State |
|---|---|
| Frame extraction (`pipeline/01`) | **done, validated** on synthetic video |
| Classifier train/eval/infer (`understanding/classify`) | **done, validated** end-to-end on the real torch/timm stack |
| Open-vocab detector (`understanding/detect.py`) | **done**; verified API; smoke-tested on a sample image |
| OCR + book lookup (`understanding/ocr_titles.py`) | implemented; not yet run on real crops |
| Splat — Postshot lane | doc'd; ready to run on your capture |
| Splat — open lane (VGGT→gsplat) | scripted + documented; CUDA build, run after the demo is safe |
| Queryable web viewer (`docs/viewer`) | **done, verified headless** — orbit + measure + search→3D highlight |
| Semantic fusion (`understanding/fusion`) | **done, tested** — lifts 2D detections into a 3D scene graph (scene.json) |
| Query CLI (`understanding/query.py`) | **done, tested** — terminal + `--json` for an agent |
| Orchestrator + eval gate (`pipeline/run.py`, `gate.py`) | **done, tested** — config-driven, threshold-gated |
| Test suite + CI (`tests/`, `.github`) | **done, green** — 20 tests, ruff |
| Classifier → ONNX (`export_onnx.py`) | **done** — torch↔onnx parity verified (browser-ready) |
| Docs (README / ARCHITECTURE / ROADMAP / CAPTURE_GUIDE) | authored |

_(Kept current as milestones land.)_

## Milestones

- **M0 — repo + pipeline code (no capture needed).** Everything runnable, documented,
  validated with synthetic/sample data. ← we are here.
- **M1 — first real capture.** Bookshelf (object). Frames → Postshot → `.ply` →
  SuperSplat → `docs/` → live URL. This is the reliable first result.
- **M2 — the enclosure.** Whole office, inside-out. The hard case that matters for real walkthroughs.
- **M3 — understanding on real frames.** Detect books/objects → train the classifier
  on real spine crops → OCR reads titles → results surfaced beside the splat.
- **M4 — open lane reproduced.** VGGT→gsplat produces a second splat, in-repo, fully
  open-source — the "I can productionize this without the proprietary app" proof.
- **M5 — polish + measurement.** Measurement overlay tuned; README/demo tightened;
  full review before sharing.

## Push-log (fill as each push lands)

| Push | How far it got | Notes |
|---|---|---|
| Bookshelf capture | | |
| Office enclosure | | |
| Classifier on real books | | |
| OCR titles | | |
| VGGT→gsplat open lane | | |

## Horizon

- **AR headset overlay.** The same pipeline made real-time and head-mounted:
  capture → reconstruct → detect/classify → overlay in view. This repo is the honest
  step 1. Not a 2-day build; a roadmap conversation.
- **Digital twins / measurement.** Metric geometry on top of the splat (pair 3DGS with
  a mesh lane, e.g. 2DGS/SuGaR) → dimensioned walkthroughs. Plays to the civil-engineer
  and "enclosure" angles; the natural client productization.
- **Language-queryable splats.** LangSplat-style features baked into the Gaussians so
  the 3D scene itself is text-searchable ("find the fire extinguisher") — the bridge
  between the two branches.

## If time collapses (cut order)

Cut from the top; the bottom line still lands:
1. Cut VGGT→gsplat open lane → keep Postshot only.
2. Cut the OCR title-reading garnish.
3. Cut the trained classifier → keep zero-shot detection only.
4. Cut self-hosted Pages → SuperSplat one-click `Publish` URL.

**Irreducible minimum:** one clean bookshelf/office capture rendering from a URL on
any phone, plus an honest README naming the pipeline and the ML framing. That single
fast-loading link is the demo; everything above it is upside.
