# SPDX-License-Identifier: MIT
"""Pure dataset-splitting helpers (no torch), so the split logic is unit-testable
in the lightweight CI without the ML stack. Re-exported by common.py."""
from __future__ import annotations

import random
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def stratified_split(root, ratios=(0.7, 0.15, 0.15), seed=0):
    """Split each class folder independently so all splits keep every class.

    Layout: root/<class_name>/<image files>. Returns (classes, train, val, test)
    as lists of (path, class_index). Guarantees a non-empty val split for classes
    with >=2 images (int(n*0.15) is 0 for n<=6, which would otherwise leave val
    empty and crash macro-F1 downstream).
    """
    root = Path(root)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not classes:
        raise SystemExit(f"No class subfolders found under {root}")
    rng = random.Random(seed)
    train, val, test = [], [], []
    for ci, c in enumerate(classes):
        files = [p for p in (root / c).iterdir() if p.suffix.lower() in IMG_EXTS]
        rng.shuffle(files)
        n = len(files)
        if n == 0:
            continue
        if n == 1:
            n_tr, n_va, n_te = 1, 0, 0
        elif n == 2:
            n_tr, n_va, n_te = 1, 1, 0
        else:
            n_va = max(1, round(n * ratios[1]))
            n_te = max(1, round(n * ratios[2]))
            n_tr = n - n_va - n_te
            if n_tr < 1:                       # tiny class: guarantee a train sample
                n_tr, n_va, n_te = n - 2, 1, 1
        train += [(p, ci) for p in files[:n_tr]]
        val += [(p, ci) for p in files[n_tr:n_tr + n_va]]
        test += [(p, ci) for p in files[n_tr + n_va:]]
    return classes, train, val, test
