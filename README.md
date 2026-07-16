# spatial-capture

<!-- screenshot: hero image of the trained splat in the repo viewer goes here. TODO: publishing a real-room screenshot is a pending decision — leave commented out until decided. -->

Capture a real enclosed space with a phone, reconstruct it as a **browser-viewable 3D Gaussian splat**, **understand what's in it**, and **ask it questions**. A reproducible, Windows-native pipeline that runs on a single consumer GPU.

> **Status:** the pipeline is built and tested end to end — reconstruction, understanding, semantic fusion, a queryable viewer with metric calibration, an orchestrator + two eval gates, a test suite/CI, and a mobile capture app. **The first real capture has been shot, passed the QA gate with zero blockers** (nine passes, 4K, locked exposure — all nine MARGINAL, advisory warnings only), **and trained to a splat**: 1,342,519 Gaussians, cleaned to 1,019,159, compressed to a 9.5 MB SOG that renders in the repo's own viewer. No live demo link yet — publishing the asset is a pending decision, so the collection ships empty. Nothing in this repo is illustrative — when a scene appears, it's a real one. Built and tested on Windows 11 with an RTX 4070 Laptop (8 GB VRAM).

**Live demo:** _(pending the publish decision — a GitHub Pages URL you open on any phone or laptop and orbit around)_

---

## Why enclosures

Most real reality-capture jobs — training simulations, workspace walkthroughs, digital twins of venues and facilities — are **enclosures**: you stand *inside* the space and the camera looks *outward*. That "inside-out" capture is the harder case than orbiting a single object (divergent views give the solver less overlap to lock onto), which is exactly why it's the valuable one. This repo is built for the enclosure case and hasn't proven it yet: the bookshelf — the de-risking subject in the same room — has been shot, passed the capture gate with zero blockers (all nine passes MARGINAL, advisory warnings only), and reconstructed; the enclosure is next.

## Two branches, two kinds of AI — then fused

One capture feeds two branches, which are then **fused into one queryable scene**:

```
   CAPTURE  (phone video: bookshelf, then the office as an enclosure)
      |
   FRAMES   (ffmpeg -> sharp, well-overlapped stills)     [pipeline/01_extract_frames.py]
      |
   GRADE    exposure/WB drift, blur, overlap, flicker     [pipeline/rate_capture.py]
            -> GO / RESHOOT *before* the GPU bill
      |
      |-----------------  RECONSTRUCTION  (AI as OPTIMIZATION)
      |   POSES         where were the cameras?      COLMAP        (open lane: VGGT)
      |   SPLAT TRAIN   fit millions of 3D Gaussians Brush/wgpu    (open lane: gsplat)
      |   CLEAN/EXPORT  crop, compress               clean_splat.py -> compress_splat.py -> .sog
      |
      |-----------------  UNDERSTANDING  (AI as SUPERVISED / TRAINED MODELS)
      |   DETECT        find things, zero-shot        open-vocab detector [understanding/detect.py]
      |   CLASSIFY      a model *I trained*           transfer learning   [understanding/classify/]
      |   READ          OCR spine titles -> lookup    [understanding/ocr_titles.py]
      |
      \-----------------  FUSE  ->  scene.json        [understanding/fusion/]
          lift the 2D detections into the splat's 3D frame -> a scene graph
             |
          VIEW + QUERY   a URL renders the space; type "book" and it lights up in 3D
                         [docs/viewer]  ·  terminal/agent: [understanding/query.py]
          MEASURE        one known length -> real units, scene-wide  [docs/viewer/scale.js]
```

- **Reconstruction is optimization.** A Gaussian splat is not "modeled" — it is *optimized*: differentiable rendering plus gradient descent fit a cloud of 3D Gaussians to your photos until the rendered views match the real ones. No labels, no training set.
- **Understanding is supervised learning.** The classifier is a model trained on labeled examples — the full lifecycle (collect → label → split → transfer-learn → evaluate → deploy).
- **Fusion is geometry.** The camera poses let you project the sparse 3D points into each detection and recover where an object actually *is* — turning pixels-with-labels into a scene graph.

Together: capture → reconstruct → detect → classify → read → fuse → query. That is the honest first rung of a live AR overlay (the same pipeline, made real-time and head-mounted) — see [`ROADMAP.md`](ROADMAP.md).

## Pipeline at a glance

| Stage | Tool (default) | Open/reproducible lane | License |
|---|---|---|---|
| **LAN import (phone → workstation)** | `pipeline/00_import_server.py` | — | MIT |
| Frame extraction | ffmpeg + `pipeline/01_extract_frames.py` | — | LGPL / MIT (this code) |
| **Capture QA gate** | `pipeline/rate_capture.py` | — | MIT |
| Camera poses (SfM) | **COLMAP** → `sparse/0/*.bin` | VGGT transformer → COLMAP format | BSD / Apache-2.0 |
| Splat training | **Brush** (Rust/wgpu — no CUDA toolkit) | gsplat | Apache-2.0 / **Apache-2.0** |
| Splat validation | `pipeline/validate_splat_ply.py` | — | MIT |
| Clean + compress | `pipeline/clean_splat.py` → `pipeline/compress_splat.py` (scripted, no GUI) | — | MIT |
| Detection | open-vocab detector (YOLOE) | — | **AGPL-3.0** (see note) |
| Classification | transfer-learned model (this repo) | — | MIT |
| Classifier → ONNX | `understanding/classify/export_onnx.py` | — | MIT |
| OCR + lookup | OCR + Open Library API | — | permissive |
| Spine→title matching | `understanding/matching.py` | — | MIT |
| **Semantic fusion** | `understanding/fusion/build_scene.py` → `scene.json` | — | MIT |
| **Queryable viewer** | `docs/viewer` (Spark + three.js) on GitHub Pages | — | MIT |
| **Query (terminal/agent)** | `understanding/query.py --json` | — | MIT |
| **Orchestrator + eval gate** | `pipeline/run.py`, `pipeline/gate.py` | — | MIT |
| **Capture app** | `docs/app` — Trove (installable PWA) | — | MIT |

**Deliberate license hygiene:** the reproducible splat lane uses **gsplat (Apache-2.0)**, *not* the original INRIA reference implementation (non-commercial research license) — the wrong base for anything company-facing. The open-vocab detector is **AGPL-3.0** and is isolated to two SPDX-tagged files (`understanding/detect.py`, `pipeline/mask_foreground.py`); a permissive swap (OWLv2 / GroundingDINO, Apache-2.0) is documented. Called out on purpose.

## Quickstart

Everything below is one command if you have a config:

```bash
cp configs/office.example.yaml configs/office.yaml              # then edit the paths
python pipeline/run.py --config configs/office.yaml --dry-run   # see the plan
python pipeline/run.py --config configs/office.yaml             # frames -> detect -> fuse -> publish
```

Or stage by stage:

```bash
# 1. sharp frames from the capture video
python pipeline/01_extract_frames.py --input data/office/office.mp4 \
    --out data/office/frames --fps 3 --long-edge 1600 --keep 0.85

# 2. poses + splat -- free and headless, no CUDA toolkit and no MSVC required:
colmap feature_extractor --database_path recon/database.db --image_path recon/images \
    --ImageReader.single_camera 1 --FeatureExtraction.use_gpu 0
colmap exhaustive_matcher --database_path recon/database.db --FeatureMatching.use_gpu 0
colmap mapper --database_path recon/database.db --image_path recon/images --output_path recon/sparse
brush_app recon --total-steps 15000 --export-path recon/splat
#    then verify it really is a splat (not a point cloud):
python pipeline/validate_splat_ply.py data/office/office.ply
#    (open lane, reproducible: pipeline/02_poses_vggt.py -> pipeline/03_train_gsplat.py)

# 3. understanding
python understanding/detect.py --frames data/office/frames --classes "book,chair,potted plant,lamp"
python understanding/classify/train.py --data data/books --epochs 30      # your trained model
python understanding/ocr_titles.py --detections out/office/detections.json --frames-root data/office/frames

# 4. fuse the 2D detections into a 3D scene graph the viewer can query
python understanding/fusion/build_scene.py --sparse data/office/sparse/0 \
    --detections out/office/detections.json --out docs/viewer/assets/scene.json

# 5. ask the scene a question (terminal, or --json for an agent)
python understanding/query.py docs/viewer/assets/scene.json "book"

# 6. clean + compress -- scripted, no GUI needed
python pipeline/clean_splat.py data/office/office.ply --out data/office/office_clean.ply
python pipeline/compress_splat.py data/office/office_clean.ply --out docs/viewer/assets/office.sog
#    then enable GitHub Pages on /docs. That URL is the demo.
```

## Measuring a splat (scene units → real units)

Structure-from-motion recovers geometry but not size. Move every camera twice as far apart and the photos come out identical, so the solve is only ever correct **up to one unknown scalar** — which is why photoreal capture gets written off as "not measurable" and why the viewer says *scene units* until you tell it otherwise.

One known length fixes that scalar for the whole scene. Put something you know the size of in the shot — a tape measure costs nothing and is the surveyor's habit — then in the viewer: **Measure** it, press **Set scale**, and type what it really is. Every measurement after that is in real units, and the calibration rides in the URL (`?scale=`) so a shared link arrives already calibrated.

The accuracy story, plainly: error is set by how precisely you can click the two reference points, and that click error is a fixed fraction of the **reference** length. So a longer reference is proportionally better — the same slop over a 72″ shelf is a third of the error it is over a 24″ tape (0.19% vs 0.57%, modelled from pick error; not yet validated against the tape ground truth). Reach for the longest thing in the room you know the size of. The viewer shows the resulting figure next to the scale badge rather than quietly implying that three decimals means three decimals of truth.

This is up-to-scale geometry made metric by a reference, not a survey instrument. It does not make a splat survey-grade; it makes it dimensioned.

## The capture app

[`docs/app`](docs/app) is **Trove**, an installable mobile PWA and the front door to the whole thing: a guided capture coach (Object vs. Enclosure) that makes a capture actually reconstructable, a one-tap **import** that sends the video to the workstation over WiFi, a **Collection** of your captured scenes, and an **Objects** catalog of everything found across them. Scene cards open the live viewer.

**Trove is deliberately not a camera.** On iOS a web page can't open the Camera app or lock a look for it — WebKit exposes only `whiteBalanceMode`, `zoom` and `torch` to web code, while `exposureMode`, `focusMode` and `iso` are unimplemented. Recording in-page via `MediaRecorder` works but runs exposure/focus/white-balance in continuous auto, which is precisely the pumping and hunting that wrecks feature matching. So Trove coaches the shot, you take it in an app that has real manual control, and Trove carries the file to the GPU:

```bash
python pipeline/00_import_server.py     # prints a LAN URL; open it on your phone
```

The GitHub Pages copy can't import — a secure page may not POST to a plain-HTTP LAN address — so the app detects that and says so instead of failing quietly.

**Shoot to Files, import from Files.** The iOS Photos picker does not hand a web page the file on disk; it hands over a "compatible" *representation*. Measured on a real capture here: a 4K60 HEVC take arrived as **4K30 H.264 at ~24.5 Mbps** — half the frames discarded and a weaker codec at the same bitrate, before reconstruction ever started. Record to a Files location (Blackmagic Camera can) and import from Files, and the original bytes survive. The import server ffprobes whatever lands and warns when it looks re-encoded, because this is invisible otherwise.

## Quality

`pytest` unit tests cover the pure logic (frame sharpness, splat-PLY validation, COLMAP IO + projection, semantic fusion on a synthetic scene, dataset splitting, query scoring, the orchestrator's stage planning, the eval gate, capture QA, asset encryption, and spine→title matching). `ruff` lints. Both run on every push via GitHub Actions once published; verified locally (59/59 tests, ruff clean).

Two gates, at opposite ends. `pipeline/rate_capture.py` grades the capture *before* reconstruction — a capture that won't reconstruct is cheap to reject and expensive to discover after 90 minutes of GPU time. `pipeline/gate.py` blocks *publishing* a scene whose reconstruction/classifier/OCR metrics miss threshold.

Several tests are built from real failures rather than invented ones — including the capture gate's own. It rejected a real capture for a 92% exposure swing, and every downstream stage independently confirmed the call. Then it **failed nine good captures**, because frame-mean brightness cannot distinguish a meter re-metering from a locked camera walking past a lamp — and locking the exposure is the advice, so the ramp it flagged was guaranteed. It now measures the *shape* of the change rather than the size, and colour was demoted to advisory outright because no threshold separates a hunting white balance from a subject that is simply a different colour. A gate that fails good work is worse than no gate; the tests encode both directions. See [`ARCHITECTURE.md`](ARCHITECTURE.md) §2b.

```bash
pytest -m "not heavy" -q     # light, no torch
ruff check .
```

See [`docs/PRACTICE.md`](docs/PRACTICE.md) for the practice written down properly — what was measured here, what the literature says, and where the two disagree (SIFT is invariant to affine illumination change, so locking exposure does *not* help the way folklore says it does; it helps because auto-exposure clips highlights and varies motion blur). See [`capture/CAPTURE_GUIDE.md`](capture/CAPTURE_GUIDE.md) for how to shoot a capture that actually reconstructs, [`ARCHITECTURE.md`](ARCHITECTURE.md) for why every choice was made (and how each stage scales to a team), [`pipeline/README.md`](pipeline/README.md) for the pose/train lanes, and [`ROADMAP.md`](ROADMAP.md) for scope and where this goes next.

## Hardware

Built and tested on Windows 11, Intel i7-13620H, 16 GB RAM, NVIDIA RTX 4070 Laptop (8 GB VRAM). The 8 GB VRAM budget shapes real choices (image count, resolution, iteration count) — noted throughout so the constraints are explicit, not hidden.

---

_Personal project. Not affiliated with, and containing no code or data from, any employer or client._
