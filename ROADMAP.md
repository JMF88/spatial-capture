# Roadmap

The goal is open-ended: push each stage as far as it profitably goes, and let the
demo be whatever the best push produces. This file tracks what's built, what's next,
and where it heads. It doubles as the honest scope statement for the repo.

## Status snapshot

| Stage | State |
|---|---|
| LAN import (`pipeline/00`) | **done, tested** — streams phone → `data/`; upload, type-reject and traversal-containment verified |
| Frame extraction (`pipeline/01`) | **done, validated** on synthetic video |
| Classifier train/eval/infer (`understanding/classify`) | **done, validated** end-to-end on the real torch/timm stack |
| Open-vocab detector (`understanding/detect.py`) | **done**; verified API; smoke-tested on a sample image |
| OCR + book lookup (`understanding/ocr_titles.py`) | **run on real crops** — reads spines legibly; resolves **0/32** to a book. Lookup is exact on clean text (1.000), but noisy reads off a gate-rejected capture retrieve nothing. Unproven at shelf scale until a capture the gate accepts. See ARCHITECTURE §8. |
| Spine→title match policy (`understanding/matching.py`) | **done, tested** — coverage-scaled containment; killed a real false positive (0.900 → 0.256) with recall intact |
| Splat — Postshot lane | doc'd; ready to run on your capture |
| Splat — open lane (VGGT→gsplat) | scripted + documented; CUDA build, run after the demo is safe |
| Queryable web viewer (`docs/viewer`) | **done, verified headless** — orbit + measure + search→3D highlight + passphrase gate |
| Metric calibration (`docs/viewer/scale.js`) | **done, tested** — one known length → real units scene-wide; shareable via `?scale=`; reports its own error |
| Semantic fusion (`understanding/fusion`) | **done, tested** — lifts 2D detections into a 3D scene graph (scene.json) |
| Query CLI (`understanding/query.py`) | **done, tested** — terminal + `--json` for an agent |
| Orchestrator + eval gate (`pipeline/run.py`, `gate.py`) | **done, tested** — config-driven, threshold-gated |
| Test suite + CI (`tests/`, `.github`) | **done, green** — 51 tests, ruff. Several are built from real failures rather than invented ones. |
| Classifier → ONNX (`export_onnx.py`) | **done** — torch↔onnx parity verified (browser-ready) |
| Capture app (`docs/app`) | **done, verified headless** — Trove: capture coach + WiFi import + collection + object catalog (installable PWA). Not a camera, by platform necessity — see README. |
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
| Bookshelf capture | **rejected by the gate** | Two 4K takes, handheld. Framing and pacing were fine — 92-94% frame overlap, only 4-11% soft frames. Killed by the camera, not the operator: auto-exposure re-metered continuously while panning between a window and dark shelves, swinging scene brightness **92% and 106%**, with white balance drifting 18-23% alongside. `rate_capture.py` returned RESHOOT before any GPU time was spent. Reshoot needs locked exposure/WB/focus, which stock iOS will not give you (AE/AF Lock holds exposure and focus but not white balance). |
| Office enclosure | not started | |
| Classifier on real books | not started | needs a capture the gate accepts |
| OCR titles | **run, honest result** | 38 book detections → 32 spine reads, human-legible, **0 resolved**. Isolated the chain: lookup is exact on clean text, retrieval is the wall on noisy reads. Fixed a false-positive bug it exposed (see ARCHITECTURE §8). Rerun on the reshoot: if clean frames still don't resolve, the OCR stage is the problem, not the capture. |
| VGGT→gsplat open lane | blocked | this machine has no CUDA toolkit, no MSVC and no COLMAP, so gsplat cannot compile — the open lane needs a toolchain install, not just a GPU |

## Horizon

- **AR headset overlay.** The same pipeline made real-time and head-mounted:
  capture → reconstruct → detect/classify → overlay in view. This repo is the honest
  step 1. Not a 2-day build; a roadmap conversation.
- **Digital twins / measurement.** Reference-based calibration has landed (`scale.js`):
  one known length in the shot makes the whole scene dimensioned, and the viewer states
  its own error rather than implying precision. What is still ahead is *geometry* you can
  trust without a reference — pair 3DGS with a mesh lane (2DGS/SuGaR) for surfaces and
  true metric extraction. Plays to the civil-engineer and "enclosure" angles; the natural
  client productization.
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
