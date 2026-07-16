# Pipeline — capture to hosted splat

Frames -> camera poses -> splat training -> clean -> compress -> web. Every stage is
scripted; none of it needs a GUI or a subscription.

**The default is COLMAP -> Brush, and it is the default because it was measured, not
because it was assumed.** On the first real capture (9 passes, ~450 frames) it ran end
to end with no CUDA toolkit and no human step:

| stage | result |
|---|---|
| COLMAP 4.1.0 (CPU) | **354/354 frames registered**, 0.810 px reprojection error, 128,877 points |
| Brush (Rust/wgpu, no CUDA toolkit) | 1,342,519 Gaussians in 8.8 min |
| `clean_splat.py` | 1,019,159 kept (75.9%); bbox 74.8x62.0x67.9 -> 9.22x10.15x2.53 |
| `compress_splat.py` | **9.5 MB SOG**, 31.8x smaller than the raw 302 MB export, visually indistinguishable |

A 100% registration rate is what a disciplined capture buys you (locked exposure/focus/WB
— see `docs/PRACTICE.md`), and it is the number to check first when a scene looks wrong.

Alternative lanes below: Postshot (Windows GUI; splat export is paywalled) and
VGGT -> gsplat (open/reproducible, needs a CUDA build). All lanes end at a standard 3DGS `.ply`.

## Stage 1 — frames (always)

```
python 01_extract_frames.py --input <video> --out data/<scene>/images --long-edge 1600
```
For the VGGT open lane use `--long-edge 1024` (VGGT exports intrinsics at 1024 px;
a mismatch silently corrupts the reconstruction).

## Stage 1b — foreground masking (optional; wall-flush / low-parallax captures)

When every camera stares at an object from a narrow cone (an object against a wall in a
tight room), the background and near-field air never triangulate and 3DGS fills them
with fog no post-hoc cull removes. `mask_foreground.py` deletes the background from the
input instead: YOLOE proposes boxes for the subject, SAM2 segments each box, the union
becomes an alpha matte (RGBA PNG out, background zeroed). Train with Brush
`--match-alpha-weight 0.1`. Measured trade: kills the haze, softens the subject
slightly — compare against an unmasked train before adopting. AGPL-3.0 (same isolation
rule as `understanding/detect.py`).

## Pose + train

### Default lane — COLMAP -> Brush (proven on real data; no GUI, no CUDA toolkit, free)
COLMAP for poses (see the COLMAP section below), then Brush to train. Brush is Rust/wgpu
/Burn, so it runs on the GPU through Vulkan with **no CUDA toolkit installed** — which is
the whole reason this lane beat the GUI one. It consumes COLMAP or Nerfstudio data and
does no SfM of its own, so COLMAP is mandatory upstream.
```
brush_app data/<scene> --total-steps 15000 --export-path data/<scene>/recon/splat   # writes export_<step>.ply
python validate_splat_ply.py data/<scene>/recon/splat/export_15000.ply
```
Expect `OK: valid 3DGS splat PLY`. Then clean and compress (below) — do not skip either.

### Alternative — Jawset Postshot (Windows GUI)
Built-in SfM + 3DGS training, no CUDA build. Drag in the video/frames, train a **Splat**
profile, export **PLY** (a Splat profile — *not* the point-cloud export), then
`validate_splat_ply.py`. Worth knowing: **splat export is paywalled**, and the free tier
emits COLMAP *text*, needing a `model_converter --output_type BIN` pass before fusion can
read it. The free lane above produces better pose output than the paid one.

### Open lane — VGGT -> gsplat (reproducible; the AI-aligned STORY)
Best on the bounded bookshelf. Needs a working CUDA torch; gsplat's rasterizer is
the one compiled dependency (try a prebuilt wheel first — see `03` header).
```
python 02_poses_vggt.py  --scene-dir data/shelf --vggt-repo ../vggt-low-vram
python 03_train_gsplat.py --data-dir data/shelf --gsplat-examples ../gsplat/examples --result-dir out/shelf
```
Keep it OFF the demo critical path: if the CUDA build fights you, train with Brush
(the default lane) and present this one as a recorded-run comparison.

### COLMAP / GLOMAP — not a fallback. The path.

Measured 2026-07-15, and it inverts the framing this file used to have: **COLMAP is
mandatory upstream of any trainer that does not do its own SfM** (Brush takes COLMAP or
Nerfstudio data and does no SfM), and it writes `sparse/0/{cameras,images,points3D}.bin`
**natively** — exactly what semantic fusion needs. Postshot's free tier exports COLMAP
*text* and needs a `model_converter --output_type BIN` step, and its splat export is
paywalled entirely. So the free lane produces better pose output than the paid one, as a
first-class artifact. "Fallback" was the wrong word.

**Its CUDA build may not run on your machine, and that is survivable.** Here, COLMAP 4.1.0
dies at `CUDA error at cuda.cc:56` about two seconds in: the build wants a newer CUDA than
the driver's ceiling (566.07 caps at 12.7) and ships no CUDA runtime of its own. CPU is
fine — **354 frames of feature extraction took 49 seconds**. Pass
`--FeatureExtraction.use_gpu 0` and `--FeatureMatching.use_gpu 0`. Exhaustive matching is
the slow part (~62k pairs from 354 frames ran at a few hundred pairs/minute on CPU); use
`sequential_matcher` when the frames really are one continuous take.

Note the 4.1 option rename: `FeatureExtraction.*` and `FeatureMatching.*`, **not** the old
`SiftExtraction.*` / `SiftMatching.*`. The old names fail with "unrecognised option".
In 2026 GLOMAP folded into COLMAP (>=4.1) as `--Mapper.mapper_type global`; the
standalone `glomap` binary still works. Prebuilt COLMAP Windows binaries, no compile:
```
colmap feature_extractor --database_path SCENE/database.db --image_path SCENE/images \
  --ImageReader.camera_model OPENCV --ImageReader.single_camera 1
colmap sequential_matcher --database_path SCENE/database.db      # video = sequential
colmap mapper --database_path SCENE/database.db --image_path SCENE/images --output_path SCENE/sparse
```
Produces `SCENE/sparse/0/` -> feed to `03_train_gsplat.py`. Prefer this (or Postshot)
for the inside-out office; VGGT poses get noisy on large divergent scenes.

## Clean (always) — the stage nobody warns you about

A splat that reconstructs perfectly can still render as unreadable fog, and the cause is a
fraction of a percent of the Gaussians. This is normal 3DGS output, not a training failure,
so it is a scripted stage rather than a manual GUI cleanup.
```
python clean_splat.py data/<scene>/recon/splat/export_15000.ply --out data/<scene>/recon/splat/clean.ply
```
It prints every cull so you can see what it ate. Three things it knows that are worth
knowing yourself (all measured — full reasoning in the script's docstring):

- **Never cull "faint" Gaussians.** Median opacity is ~0.09 and half are below 0.1: a 3DGS
  surface IS a pile of near-transparent blobs. The pathology is the conjunction — **big AND
  faint** fogs a wide area while contributing nothing. Culling 14.5% on that rule took the
  shelf from fog to legible book spines *and made the file smaller*.
- **Measure elongation by longest/MIDDLE, never longest/shortest.** Gaussians are *supposed*
  to flatten into disks; longest/shortest punishes the healthy case and culled 33.7% of a
  good scene where the right metric culls 3.0%.
- **Floaters own the framing.** 0.3% of Gaussians outside the room made the bounding box 32x
  too big, so any viewer that auto-frames on min/max aims at empty space.

## Compress + host (web)

Raw 3DGS `.ply` is huge (250-300 MB) — **never ship it to the web.** No GUI needed:
```
python compress_splat.py data/<scene>/recon/splat/clean.ply --out ../docs/viewer/assets/scene.sog
```
**SOG, not SPZ** (19 MB vs 27 MB at equal quality), and **keep the spherical harmonics**:
they are 76% of a PLY but only 21% of a SOG, so dropping them saves ~4 MB and costs all
view-dependent colour. Compress first, then consider SH — never the reverse.

Drop the result at `docs/viewer/assets/scene.sog` (the viewer also loads
`.ply`/`.splat`/`.spz`) and enable GitHub Pages on `/docs`. Target **< 10-15 MB** and
**< ~1.5 M Gaussians** for smooth mobile. URL:
`https://<user>.github.io/spatial-capture/viewer/`.

3DGS `.ply` files often load upside down; the viewer's `?flip=z` (or the Flip button) fixes
it. Check this before concluding a reconstruction failed.

## Notes

- The VGGT default checkpoint (`facebook/VGGT-1B`) is a research / non-commercial
  checkpoint; the open lane's poses use it for a portfolio demo — don't imply a
  commercial license you don't hold.
- Always run `validate_splat_ply.py` before hosting: some exports silently drop
  opacity/scale/rotation/SH and render as flat dots instead of a splat.
