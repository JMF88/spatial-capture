# spatial-capture

Capture a real enclosed space with a phone, reconstruct it as a **browser-viewable 3D Gaussian splat**, and **understand what's in it**. A reproducible, Windows-native pipeline that runs on a single consumer GPU.

> **Status:** pipeline and tooling complete; first real capture in progress. The live demo link below goes up once the office capture is trained. Everything here was built and tested on Windows 11 with an RTX 4070 Laptop (8 GB VRAM).

**Live demo:** _(coming — a GitHub Pages URL you open on any phone or laptop and orbit around)_

---

## Why enclosures

Most real reality-capture jobs — training simulations, workspace walkthroughs, digital twins of venues and facilities — are **enclosures**: you stand *inside* the space and the camera looks *outward*. That "inside-out" capture is the harder case than orbiting a single object (divergent views give the solver less overlap to lock onto), which is exactly why it's the valuable one. This repo proves the enclosure case end to end, and uses a bookshelf as the de-risking first capture inside the same room.

## Two branches, two kinds of AI

One capture feeds two branches. Being able to name the difference between them matters:

```
   CAPTURE  (phone video: bookshelf, then the office as an enclosure)
      |
   FRAMES   (ffmpeg -> sharp, well-overlapped stills)     [pipeline/01_extract_frames.py]
      |
      |-----------------  RECONSTRUCTION  (AI as OPTIMIZATION)
      |   POSES         where were the cameras?     Postshot SfM  (open lane: VGGT)
      |   SPLAT TRAIN   fit millions of 3D Gaussians Postshot 3DGS (open lane: gsplat)
      |   CLEAN/EXPORT  crop, compress              SuperSplat -> .ply/.compressed
      |   VIEW          a URL renders your space     docs/ on GitHub Pages
      |   MEASURE       in-viewer dimensions         (the civil-engineer touch)
      |
      \-----------------  UNDERSTANDING  (AI as SUPERVISED / TRAINED MODELS)
          DETECT        find things, zero-shot        open-vocab detector [understanding/detect.py]
          CLASSIFY      a model *I trained*           transfer learning   [understanding/classify/]
          READ          OCR spine titles -> lookup    [understanding/ocr_titles.py]
```

- **Reconstruction is optimization.** A Gaussian splat is not "modeled" — it is *optimized*: differentiable rendering plus gradient descent fit a cloud of 3D Gaussians to your photos until the rendered views match the real ones. No labels, no training set.
- **Understanding is supervised learning.** The classifier is a model trained on labeled examples — the full lifecycle (collect → label → split → transfer-learn → evaluate → deploy).

Together they are a small **spatial-AI** pipeline: capture → reconstruct → detect → classify → read. That is also the honest first rung of a live AR overlay (the same pipeline, made real-time and head-mounted) — see [`ROADMAP.md`](ROADMAP.md).

## Pipeline at a glance

| Stage | Tool (default) | Open/reproducible lane | License |
|---|---|---|---|
| Frame extraction | ffmpeg + `01_extract_frames.py` | — | LGPL / MIT (this code) |
| Camera poses (SfM) | Jawset Postshot | VGGT transformer → COLMAP format | Postshot EULA / Apache-2.0 |
| Splat training | Jawset Postshot (3DGS) | gsplat | Postshot EULA / **Apache-2.0** |
| Clean + compress | PlayCanvas SuperSplat | — | MIT |
| Web viewer | `docs/` static page on GitHub Pages | — | MIT |
| Detection | open-vocab detector (YOLOE) | — | **AGPL-3.0** (see note) |
| Classification | transfer-learned model (this repo) | — | MIT (this code) |
| OCR + lookup | OCR + Open Library API | — | permissive |

**Deliberate license hygiene:** the reproducible splat lane uses **gsplat (Apache-2.0)**, *not* the original INRIA reference implementation (non-commercial research license) — the wrong base for anything company-facing. The open-vocab detector is **AGPL-3.0**: perfect for a public OSS demo, but a closed product or hosted service on top of it needs a commercial license. Called out on purpose.

## Quickstart

```bash
# 1. extract sharp frames from a capture video
python pipeline/01_extract_frames.py --input data/office/office.mp4 \
    --out data/office/frames --fps 3 --long-edge 1600 --keep 0.85

# 2a. RELIABLE lane: open the frames (or the video) in Jawset Postshot,
#     let it solve poses + train the splat, export .ply.  (see capture/CAPTURE_GUIDE.md)
# 2b. OPEN lane (reproducible): poses via VGGT, train via gsplat
#     (see pipeline/README + ARCHITECTURE.md — CUDA build, run after the demo is safe)

# 3. clean/compress the .ply in SuperSplat (superspl.at/editor), drop it in docs/assets/,
#    then enable GitHub Pages on /docs.  That URL is the demo.

# 4. understanding branch (own venv):
python understanding/detect.py --frames data/office/frames --classes "book,chair,plant,monitor,lamp"
python understanding/classify/train.py  --data data/books --epochs 20
python understanding/ocr_titles.py --frames data/office/frames
```

See [`capture/CAPTURE_GUIDE.md`](capture/CAPTURE_GUIDE.md) for how to shoot a capture that actually reconstructs, [`ARCHITECTURE.md`](ARCHITECTURE.md) for why every choice was made (and how each stage scales to a team), and [`ROADMAP.md`](ROADMAP.md) for scope and where this goes next.

## Hardware

Built and tested on Windows 11, Intel i7-13620H, 16 GB RAM, NVIDIA RTX 4070 Laptop (8 GB VRAM). The 8 GB VRAM budget shapes real choices (image count, resolution, iteration count) — noted throughout so the constraints are explicit, not hidden.

---

_Personal project. Not affiliated with, and containing no code or data from, any employer or client._
