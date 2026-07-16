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

### 2b. Capture QA — `pipeline/rate_capture.py` (measure first, spend later)
- **What:** grade the extracted frames — sharpness, frame-to-frame overlap, exposure,
  colour, mains flicker, clipping — and return **GO / MARGINAL / RESHOOT** *before* the
  30–90 minutes of splat training.
- **Why this way:** reconstruction is the expensive step and it fails for reasons already
  visible in the frames. Every one of them is cheap to measure and impossible to fix
  afterwards, when the trainer's time and the chance to reshoot are both gone. This is the
  cost argument in one file: the cheapest GPU-hour is the one you don't spend.
- **It has earned its keep twice, in opposite directions.** It rejected a real capture for
  92% exposure swing before any GPU time — and every downstream stage independently
  confirmed the call (see §8). Then it **failed a good capture**, which taught it more.

#### The lesson: measure the camera, not the room
Of nine correctly-locked takes, six came back RESHOOT and three MARGINAL for 13.1–58.9% "exposure drift". They were fine.
Frame-mean brightness moves for two unrelated reasons:

| | behaves like |
|---|---|
| a meter re-metering | **oscillates** — chases content, overshoots, corrects |
| a locked camera walking past a lamp | **ramps** — smooth, because a room's brightness is a smooth function of where you stand |

Spread cannot tell them apart, and since the advice is to *lock* exposure, the ramp is
**guaranteed**. The gate was flagging the behaviour it asks for — the worst failure a gate
can have, because it sends someone to reshoot correct work. **Shape separates what size
cannot:** fit a smooth trend, measure the residual. On one shelf, same subject:

    auto-exposure  : 18.0%, 16.1%      <- chatter
    locked exposure: 2.2% - 6.2%       <- scene only

- **Failure modes, named:** sharpness is inflated by sensor noise (grain reads as detail),
  so it flatters a high-ISO take. Overlap saturates below 50% and over-reports exactly
  where a capture is worst. A perfectly *graceful* AE drift hides in the trend and reads
  clean — this catches meters that chatter, which is what meters do.
- **What it refuses to do:** colour is reported and **never blocks**. A subject's colour
  does not vary smoothly with position (white robot → wood → beige wall), so detrending
  isolates nothing, and on real data the distributions overlap outright — auto-WB takes at
  3.5%/4.5% against *locked* takes at 3.7% and 4.2%. No threshold exists. Reporting a
  number you cannot act on is honest; gating on a coin flip is not.
- **Scale:** thresholds here are calibrated on one room and two lighting states. That is a
  data point, not a calibration — widen it before trusting it on a client's site. The
  productised version of this is not a better metric, it's the **capture SOP and operator
  training** the metric implies (see the Horizon note in `ROADMAP.md`).

### 3. Camera poses — structure-from-motion (SfM)
- **What:** recover each camera's intrinsics + pose and a sparse point cloud from
  the frames. This is the classic photogrammetry step.
- **Choices & tradeoffs:**
  - **COLMAP (default).** The gold-standard incremental solver. On the gate-passing
    capture it registered **354/354 frames** at **0.810 px mean reprojection error**.
    The default trainer (Brush, stage 4) does no SfM of its own, so COLMAP is
    mandatory upstream — and it emits the binary sparse model that semantic fusion
    (stage 9) consumes natively. GLOMAP's global solver ships inside COLMAP ≥ 4.1.
  - **VGGT (open lane).** A feed-forward *transformer* (CVPR 2025) that infers
    poses + geometry in seconds instead of iterative solving — the "AI-aligned"
    headline: we replaced 20-year-old geometric SfM with a learned model. Exports
    COLMAP format for the trainer. (`pipeline/02_poses_vggt.py`.)
  - **Jawset Postshot (GUI alternative).** SfM + splat training in one Windows app,
    but the free tier cannot export the trained splat, and its pose export is COLMAP
    *text* (needs a bin conversion before fusion). Kept only as a poses fallback.
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
  - **Brush (Apache-2.0, default trainer).** Rust/wgpu — no CUDA toolkit to build.
    On the gate-passing capture: **1,342,519 Gaussians in 8.8 min**, cleaned to
    1,019,159 (75.9% kept), compressed to a **9.5 MB SOG** that renders in the
    repo's own viewer. Consumes COLMAP output; does no SfM of its own.
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
  with orbit controls — the URL you open on any laptop or phone. It also loads
  `scene.json` (stage 9) and overlays the scene objects: 3D markers, labels, and a
  search box that highlights matches in the capture.
- **Why this way:** a splat renders on a plain laptop/phone GPU; hosting is free
  and static (nothing to break live). A specific trap is avoided: viewers that need
  `SharedArrayBuffer` require COOP/COEP response headers that **GitHub Pages cannot
  set** — so we pick a viewer that doesn't need them.
- **The signature touch:** a 2-point measurement overlay **that knows what its numbers
  mean**. SfM recovers geometry up to an unknown scalar, so a splat measures in scene
  units and photoreal capture gets written off as "not measurable". Measure one known
  length — a tape measure in the shot, the surveyor's habit — and `docs/viewer/scale.js`
  fixes the scalar for the whole scene; the calibration rides in `?scale=` so a shared
  link stays calibrated. Uncalibrated readings say "scene units" rather than implying a
  precision that isn't there, and the viewer reports its own error: accuracy is bounded
  by the pick precision on the *reference*, so the same slop is 0.57% over a 24″ tape and
  0.19% over a 72″ shelf — figures modelled from pick error, not yet validated against a
  real measurement; reach for the longest reference in the room. Dimensioned, not
  survey-grade, and it says so.
- **No runtime third-party dependency.** three + spark are vendored into
  `docs/viewer/vendor/` at recorded hashes rather than pulled from a CDN. This was not
  hygiene theatre: with the CDN blocked the page hung on `Loading...` indefinitely with
  no JS error, because an unresolved importmap means the module script never runs and
  nothing survives to report it. A portfolio link that dies when someone else's host has
  a bad afternoon is not a link. Verified by hard-aborting every non-localhost request
  and confirming it still renders.
- **Scale:** for clients, wrap as an embeddable component, add compression tiers, and a
  CDN for *assets*. Delivery is a solved commodity — spend effort on capture/pose, not
  here. Note the asymmetry: a CDN for a big static asset is a cache miss, a CDN for the
  code that renders it is a hard dependency.

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

### 7. Classification — `understanding/classify/` (the model we trained)
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
  That narrative — *"here's how we'd turn this 90-image demo into a maintained model a
  team owns"* — is the point of building it by hand.

### 8. OCR + lookup — `understanding/ocr_titles.py` (feature B)
- **What:** read book-spine text (usually vertical/rotated) with OCR, then match the
  title to metadata via a free API (Open Library / Google Books).
- **Why this way:** aims to turn "there's a book here" into "*which* book". It's mostly
  pipeline/integration, so it garnishes the trained classifier rather than replacing it.
- **Honest status — measured, not assumed.** Run against a real shelf (8 frames, 38 book
  detections): OCR produced 32 reads that a human recognises instantly — `MASQUERADE Ea
  DINNIMAN`, `THIS INEVIIABLE Dinmhan RUIN` — and the lookup resolved **none** of them.
  The chain was isolated stage by stage: on *clean* text the lookup is exact (a correct
  title scores 1.000; title+author 0.731), but the noisy reads retrieve no candidates at
  all, because Open Library has no fuzzy search and `HBVTCHERS` is unrecoverable. The
  frames came from a capture our own QA gate had already rejected for 92% exposure drift.
  So the causal chain is: drifting auto-exposure → soft frames → OCR character errors →
  zero retrieval → zero matches. Every downstream stage independently confirmed the gate.
  **This stage is proven on clean input and unproven at shelf scale.** It is not claimed
  to resolve titles until it does so on a capture the gate accepts. On the capture the
  gate did accept, reads are legible and retrieval matched 4 titles (2 verified real,
  best score 0.870), with the remaining misses dominated by query construction — one
  garbled token zeroes the keyword search — not read legibility; per-title identity is
  still work in progress.
- **Failure modes:** rotated text (rotate the crop both ways, keep the higher-confidence
  read), stylized fonts, glare, and — dominant in practice — capture quality upstream.
  Lookup needs a decent title string, and retrieval is the wall long before scoring is.
- **Precision over recall, deliberately.** `understanding/matching.py` scores a read
  against a candidate title, and credits containment only in proportion to coverage. A
  flat containment credit once matched a spine reading `Jonan` to *"Jonan & evolusi
  kereta api Indonesia"* — a book about Indonesian railways — because five letters sat
  inside a thirty-four-letter title. A confident wrong answer is worse than none: one bad
  match discredits every good one. Reads without enough alphabetic signal are never
  queried. The policy is kept free of cv2/easyocr/torch so it is unit-testable against
  the real reads, for the same reason `splits.py` sits apart from `common.py`.
- **Scale:** confidence thresholds + human review for low-confidence reads; cache
  lookups; batch API calls politely; and a fuzzy retrieval tier (or a local title index)
  if noisy reads must be resolved rather than rejected.

---

### 9. Semantic fusion — `understanding/fusion/` (the bridge between the branches)
- **What:** lift the 2D detections into the splat's 3D frame. For each detection, project the
  COLMAP sparse points into that camera, keep the ones landing inside the box and in front of
  the lens, and take their **median** as the object's 3D anchor; then cluster anchors of the
  same class across frames into unique scene objects and emit `scene.json`.
- **Why this way:** it needs no dense depth and no extra model — just the poses you already
  have. The median is robust to a few stray points. It's the cheapest honest way to turn
  "pixels with labels" into "things at coordinates."
- **Failure modes:** floaters between the camera and the object drag the median (the documented
  next step is a depth-MAD filter); textureless objects get too few sparse points to place, so
  they're skipped rather than mis-placed; the merge radius is in *scene units*, so it wants
  scaling per capture.
- **Scale:** the next investment — mask-membership instead of box, DBSCAN instead of greedy
  clustering, per-object embeddings so query is semantic rather than lexical.

### 10. Query — `understanding/query.py` + the viewer's search
- **What:** score every scene object against a text query (exact category/label, prefix,
  keyword, substring) and return ranked hits with their 3D anchors. The viewer runs the same
  rubric in JS and highlights matches in 3D; the CLI emits `--json` for a tool or an agent.
- **Why this way:** the structured tier is dependency-free, instant, and unit-testable — and
  it's what makes a capture *useful* rather than merely pretty. One rubric in both places means
  the terminal and the browser never disagree.
- **Failure modes:** lexical matching misses synonyms ("couch" vs "sofa"); the documented next
  step is a CLIP-embedding tier behind the same API.
- **Scale:** this is the natural seam for an LLM — detection + OCR already emit structured
  output, so an agent can ask the scene questions and get back anchors.

## The operator layer — `pipeline/run.py`, `pipeline/gate.py`, `tests/`

- **Orchestrator:** one config drives frames → detect → fuse → publish, skipping stages whose
  outputs already exist (resumable), with `--dry-run` to print the plan. Splat training stays
  external on purpose; the config points at its outputs.
- **Eval gate:** publishing is blocked when reconstruction PSNR/SSIM, classifier macro-F1, or
  OCR CER miss threshold. A missing metric is reported, never silently passed.
- **Tests + CI:** the pure logic is unit-tested and linted — green locally, and on every push
  once the repo is published. The suite is deliberately pure-logic (no torch/GPU in CI); the
  ML stages are hand-validated on real runs rather than unit-tested. That's why `splits.py`
  exists apart from `common.py` — so the split logic is testable without the ML stack.
- **Why it matters:** the demo proves the capability; this layer answers "how would a team run
  this without you." And the thing that actually needs to scale isn't compute — it's capture,
  which is why the app below exists.

## Trove — the capture app (`docs/app`)

An installable mobile PWA: a guided capture coach (Object vs. Enclosure) that turns the capture
guide into a pre-flight checklist beside the record button, a **Collection** of captured scenes,
and an **Objects** catalog of everything found across them. Capture quality is the pipeline's
bottleneck, so the highest-leverage interface in the whole system is the one standing between a
person and a bad capture.

## License hygiene (a feature, said out loud)

The repo is **MIT**. Deliberate, not incidental:
- The splat lane uses **Brush and gsplat (both Apache-2.0)**, *not* the INRIA reference
  (non-commercial research license) — the wrong base for anything a company would ship.
- The Ultralytics stack (YOLOE, its SAM2 wrapper) is **AGPL-3.0**. AGPL treats
  hosted/network use as distribution, so it is confined to exactly two files, each
  carrying an SPDX header: `understanding/detect.py` (open-vocab detection) and
  `pipeline/mask_foreground.py` (foreground matting). Nothing else imports it; the
  documented permissive swap is **OWLv2** or **GroundingDINO + SAM2** (Apache-2.0).

Saying this unprompted demonstrates exactly the cost/license/security discipline a
security-conscious team wants to see — it's free credibility, not a footnote.

## Cost / eval / security discipline (productionization view)

If we led a team shipping this: bound GPU cost per scene and per training run; gate
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
