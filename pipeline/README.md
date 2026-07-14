# Pipeline — capture to hosted splat

Frames -> camera poses -> splat training -> export -> web. Two lanes for the
pose+train middle: a reliable default (Postshot) and an open, reproducible one
(VGGT -> gsplat), with a classical fallback (COLMAP). All three end at a standard
3DGS `.ply`.

## Stage 1 — frames (always)

```
python 01_extract_frames.py --input <video> --out data/<scene>/images --long-edge 1600
```
For the VGGT open lane use `--long-edge 1024` (VGGT exports intrinsics at 1024 px;
a mismatch silently corrupts the reconstruction).

## Pose + train

### Default lane — Jawset Postshot (reliable; the DEMO path)
Windows GUI, built-in SfM + 3DGS training, no CUDA build. Drag in the video/frames,
train a **Splat** profile, export **PLY** (a Splat profile — *not* the point-cloud
export). Then validate:
```
python validate_splat_ply.py <exported.ply>
```
This is the reliable demo path — it can't break live.

### Open lane — VGGT -> gsplat (reproducible; the AI-aligned STORY)
Best on the bounded bookshelf. Needs a working CUDA torch; gsplat's rasterizer is
the one compiled dependency (try a prebuilt wheel first — see `03` header).
```
python 02_poses_vggt.py  --scene-dir data/shelf --vggt-repo ../vggt-low-vram
python 03_train_gsplat.py --data-dir data/shelf --gsplat-examples ../gsplat/examples --result-dir out/shelf
```
Keep it OFF the demo critical path: if the CUDA build fights you, train in Postshot
and present this lane as the open/reproducible version with a recorded run.

### Classical fallback — COLMAP / GLOMAP (proves the fundamentals; best for the office)
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

## Export + host (web)

Raw 3DGS `.ply` is huge (250-290 MB) — **never ship it to the web.** Compress: open the
`.ply` in **SuperSplat** (superspl.at/editor), crop/orient, and export **`.spz`**
(object/bookshelf) or **`.sog`** (whole-office enclosure). Drop it at
`docs/viewer/assets/scene.spz` (the viewer also loads `.ply`/`.splat`/`.sog`) and enable
GitHub Pages on `/docs`. Target **< 10-15 MB** and **< ~1.5 M Gaussians** for smooth
mobile. URL: `https://<user>.github.io/spatial-capture/viewer/`.

## Notes

- The VGGT default checkpoint (`facebook/VGGT-1B`) is a research / non-commercial
  checkpoint; the open lane's poses use it for a portfolio demo — don't imply a
  commercial license you don't hold.
- Always run `validate_splat_ply.py` before hosting: some exports silently drop
  opacity/scale/rotation/SH and render as flat dots instead of a splat.
