# Architecture & rationale

The purpose of this document is fluency. For every stage it states *what it does,
why this choice over the alternatives, how it fails, and how it would scale to a
team and to production.* If you can speak to those four things per stage, you can
lead this work, not just demo it.

Altitude is deliberately "pipeline architecture," not paper-level math. Terms you
can drop confidently are in the glossary at the end.

---

## The system in one paragraph

A phone captures a real enclosed space. We extract sharp frames, recover where the
camera was for each frame (structure-from-motion), and optimize a **3D Gaussian
splat** — a photorealistic, browser-renderable reconstruction. In parallel, an
**understanding branch** runs an open-vocabulary detector over the same frames,
feeds the crops to a **classifier we trained**, and reads book spines with OCR.
The reconstruction becomes a URL; the understanding becomes structured data laid
over it. Together that is a small **spatial-AI** pipeline — and the honest first
rung of a live AR overlay (the same pipeline made real-time and head-mounted).

## Two kinds of AI here — name the difference

This distinction is the most sophisticated thing to say out loud, because most
people conflate them:

- **Reconstruction is optimization.** A Gaussian splat is not "modeled" and it is
  not a neural network you run inference on. It is *fit*: differentiable rendering
  + gradient descent tune millions of 3D Gaussians until the rendered views match
  the real photos. There are no labels and no training set — the photos are the
  only supervision, and the "model" is the scene itself.
- **Understanding is supervised learning.** The classifier is a model trained on
  labeled examples — the classic lifecycle (collect → label → split → transfer-learn
  → evaluate → deploy). The detector is a pretrained vision-language model queried
  by text.

Same repo, two fundamentally different machine-learning paradigms. Being precise
about which is which signals you actually understand the field.

---

## Stage-by-stage

### 1. Capture (phone)
- **What:** phone video (or stills) of the subject — bookshelf first (object,
  outside-in), then the office (enclosure, inside-out).
- **Why this way:** reconstruction quality is set here, not in the trainer.
  ~80% of the outcome is capture technique. Starting on a texture-rich object
  de-risks; the enclosure is the valuable, harder case (see `capture/CAPTURE_GUIDE.md`).
- **Failure modes:** mirrors/TVs/windows (reflective + transparent surfaces break
  the geometry), blank textureless walls (no features to match), motion blur,
  and pure in-place rotation (no parallax → degenerate pose solve).
- **Scale:** for a client/team you'd standardize a capture SOP, a capture-quality
  checklist, and ideally an on-device capture app that flags coverage gaps live.
  The single highest-leverage quality control in the whole pipeline is here.

### 2. Frame extraction — `pipeline/01_extract_frames.py`
- **What:** sample frames at a fixed rate, downscale the long edge (VRAM budget),
  drop the blurriest by variance-of-Laplacian.
- **Why this way:** fewer sharp, well-overlapped frames beat thousands of soft
  ones — extra blurry frames actively poison feature matching. Kept deliberately
  dependency-light (numpy + Pillow, no OpenCV/GPU) so this stage always runs.
- **Failure modes:** too-high fps → near-duplicate frames and slow SfM; too-low →
  insufficient overlap; over-aggressive blur filter → coverage holes.
- **Scale:** add per-frame coverage/overlap scoring and adaptive sampling; this is
  a cheap, deterministic, testable stage — a good candidate for CI.

### 3. Camera poses — structure-from-motion (SfM)
- **What:** recover each camera's intrinsics + pose and a sparse point cloud from
  the frames. This is the classic photogrammetry step.
- **Choices & tradeoffs:**
  - **Jawset Postshot (default engine).** Built-in SfM + splat training in one
    Windows app — no COLMAP/CUDA-build. Reliable; the demo runs on it.
  - **VGGT (open lane).** A feed-forward *transformer* (CVPR 2025) that infers
    poses + geometry in seconds instead of iterative solving — the "AI-aligned"
    headline: we replaced 20-year-old geometric SfM with a learned model. Exports
    COLMAP format for the trainer. (`pipeline/02_poses_vggt.py`.)
  - **COLMAP / GLOMAP (classical fallback).** The gold-standard incremental /
    global solvers; slower but rock-solid — proves the fundamentals.
- **Failure modes:** low-texture or repetitive scenes fail to register; VGGT can
  drift or mis-scale on room-scale scenes (keep COLMAP as the always-works path).
- **Scale:** pose is the pipeline's reliability fulcrum. In production you'd gate
  on a registration-quality metric and auto-fall-back classical when the learned
  path under-registers.

### 4. Splat training — 3D Gaussian Splatting (3DGS)
- **What (at altitude):** the scene is represented as a cloud of 3D Gaussians.
  Each Gaussian has a **position**, a **covariance** (its ellipsoidal shape/orientation,
  parameterized by a scale + a rotation), an **opacity**, and **view-dependent color**
  stored as **spherical-harmonic (SH)** coefficients. A differentiable, tile-based
  rasterizer projects them to the image and alpha-composites front-to-back. Training
  = render a known view, compare to the real photo (photometric loss), backpropagate,
  repeat for thousands of iterations. **Adaptive density control** periodically
  clones/splits Gaussians where detail is missing ("densify") and prunes weak/oversized
  ones — so the "model" literally grows into the scene.
- **Choices & tradeoffs:**
  - **gsplat (Apache-2.0)** for the open lane — permissive, scriptable, the correct
    base for anything company-facing.
  - **Not** the INRIA reference implementation — its license is non-commercial
    research only, wrong for a portfolio/company repo. (Deliberate; see below.)
  - **vs NeRF:** NeRF is an implicit MLP rendered by slow volumetric ray-marching;
    3DGS is explicit primitives, rasterized in real time — which is why it goes to
    the web. **vs mesh (classic MVS):** a mesh gives clean measurable geometry (good
    for CAD/measurement) but weaker photorealism and view-dependent light; 3DGS is
    the opposite. They're complementary — see the measurement note.
- **Failure modes:** floaters/haze in under-observed regions; VRAM blow-ups (the
  8 GB budget caps image resolution / iteration count); over-densification.
- **Scale:** standardize iteration budgets and quality metrics (PSNR/SSIM vs held-out
  views), track training cost per scene, and export at multiple compression levels
  for different delivery targets.

### 5. Web viewer — `docs/` on GitHub Pages
- **What:** a static page that loads the trained splat and renders it in-browser
  with orbit controls — the URL you open on any laptop or phone.
- **Why this way:** a splat renders on a plain laptop/phone GPU; hosting is free
  and static (nothing to break live). A specific trap is avoided: viewers that need
  `SharedArrayBuffer` require COOP/COEP response headers that **GitHub Pages cannot
  set** — so we pick a viewer that doesn't need them.
- **The signature touch:** an in-viewer 2-point **measurement/annotation** overlay.
  A civil engineer adding metric dimensions to a phone capture is unique-to-me and
  points at the digital-twin roadmap.
- **Scale:** for clients, wrap as an embeddable component, add compression tiers,
  and a CDN. Delivery is a solved commodity — spend effort on capture/pose, not here.

### 6. Open-vocabulary detection — `understanding/detect.py`
- **What:** query arbitrary class names in plain English ("book", "potted plant")
  with **no training** — a vision-language model matches CLIP-style text embeddings
  to image regions and returns boxes + masks.
- **Why this way:** zero-shot means the "vocabulary" is just a prompt; it feeds the
  classifier (crops) and OCR (spine regions). Uses YOLOE for detect+segment in one pass.
- **Failure modes:** prompt wording matters ("potted plant" ≫ "plant"); fine-grained
  labels ("book spine") are weak — detect the whole book, then crop. Masks are at
  model resolution unless `retina_masks=True`.
- **Scale:** cache text embeddings; batch/stream frames for constant memory; for a
  fixed client vocabulary, distill to a smaller closed-set detector for speed.

### 7. Classification — `understanding/classify/` (the model I trained)
- **What:** a supervised image classifier via **transfer learning** — start from an
  ImageNet-pretrained backbone (timm), replace the head, **freeze the backbone** and
  warm up the head, then **unfreeze and fine-tune** at a low LR. Stratified
  train/val/test split; early stopping on validation macro-F1; evaluation with a
  confusion matrix on a held-out set never seen in training.
- **Why this way:** with only ~50–200 images, learning features from scratch would
  overfit instantly. Transfer learning reuses general visual features and only adapts
  the last layers — the standard, correct move for small data. Label smoothing and
  strong augmentation further fight overfitting.
- **Failure modes:** tiny/imbalanced classes, train/test leakage (avoided via the
  persisted split), over-fine-tuning (mitigated by warmup + low LR + early stop).
- **Scale (the lead-a-team story):** this is the full lifecycle. To industrialize:
  a labeling workflow + guidelines, an **active-learning** loop (label the images the
  model is least sure about), a **data flywheel** (production captures → new labels →
  retrain), versioned datasets + models, an eval gate in CI, and monitoring for drift.
  That narrative — *"here's how I'd turn this 90-image demo into a maintained model a
  team owns"* — is the point of building it by hand.

### 8. OCR + lookup — `understanding/ocr_titles.py` (feature B)
- **What:** read book-spine text (usually vertical/rotated) with OCR, then match the
  title to metadata via a free API (Open Library / Google Books).
- **Why this way:** turns "there's a book here" into "*which* book" — a legible,
  delightful capstone. It's mostly pipeline/integration, so it garnishes the trained
  classifier rather than replacing it.
- **Failure modes:** rotated text (rotate the crop both ways, keep the higher-confidence
  read), stylized fonts, glare. Lookup needs a decent title string.
- **Scale:** confidence thresholds + human review for low-confidence reads; cache
  lookups; batch API calls politely.

---

## License hygiene (a feature, said out loud)

The repo is **MIT**. Deliberate, not incidental:
- The reproducible splat lane uses **gsplat (Apache-2.0)**, *not* the INRIA reference
  (non-commercial research license) — the wrong base for anything a company would ship.
- The open-vocab detector (`detect.py`) is **AGPL-3.0** (Ultralytics/YOLOE). AGPL treats
  hosted/network use as distribution, so it's isolated in a single file with an SPDX
  header; the documented permissive swap is **OWLv2** or **GroundingDINO + SAM2**
  (Apache-2.0). Nothing else imports it.

Saying this unprompted demonstrates exactly the cost/license/security discipline a
security-conscious team wants to see — it's free credibility, not a footnote.

## Cost / eval / security discipline (productionization view)

If I led a team shipping this: bound GPU cost per scene and per training run; gate
merges on reconstruction and classifier eval metrics; keep humans in the loop on
low-confidence understanding outputs; treat every dependency's license as a
first-class decision; and never let a learned component (VGGT, the classifier) be a
single point of failure without a classical fallback.

---

## Glossary (terms to use confidently)

- **SfM (structure-from-motion):** recovering camera poses + sparse 3D from images.
- **Radiance field:** a representation of how light leaves every point/direction in a
  scene; NeRF and 3DGS are two ways to fit one.
- **3DGS / Gaussian splat:** explicit cloud of 3D Gaussians, rasterized in real time.
- **Spherical harmonics (SH):** compact basis for view-dependent color (why the splat
  has believable specular/lighting as you orbit).
- **Differentiable rendering:** a renderer you can backpropagate through, so image
  error can update the 3D representation.
- **Densify / prune:** adaptive growth/removal of Gaussians during training.
- **Transfer learning:** adapt a pretrained backbone to a new small dataset.
- **Open-vocabulary detection:** detect arbitrary text-named classes with no retraining.
- **Confusion matrix / precision / recall:** per-class correctness of a classifier.
