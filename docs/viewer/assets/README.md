# Put your trained splat here

Drop the compressed splat in this folder. The viewer loads **`./assets/scene.ply`**
by default; to load any other file (recommended: a compressed `.spz`/`.sog`), pass
it as a query param:

    viewer/?src=./assets/scene.spz

Supported: `.ply` / `.splat` / `.spz` / `.sog` / `.ksplat`.

Keep it small for fast mobile load — target **< 10-15 MB** and **< ~1.5 M Gaussians**.
Raw 3DGS `.ply` is ~300 MB (the first real export: 302 MB); run it through
`pipeline/clean_splat.py` then `pipeline/compress_splat.py` (emits `.sog`) first —
scripted, no GUI. See `../../../pipeline/README.md` -> "Compress + host (web)".
