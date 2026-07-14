#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Evaluate the trained classifier on the held-out test split.

Prints accuracy + a per-class report and saves a confusion-matrix PNG. Uses the
exact test split written by train.py, so the number is honest (never trained on).

Example:
  python eval.py --run runs/books
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import common


def plot_confusion(cm, classes, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.6 + len(classes), 1.6 + len(classes)))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    thr = cm.max() / 2 if cm.max() else 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="training output dir")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    device = common.pick_device(args.device)
    model, classes = common.load_for_inference(args.run / "model.pt", device)
    tf = common.build_transforms(model, training=False)

    items = [(Path(p), y) for p, y in json.loads((args.run / "test_split.json").read_text())]
    if not items:
        raise SystemExit("Empty test split — nothing to evaluate.")
    dl = DataLoader(common.ListDataset(items, tf), batch_size=16, shuffle=False, num_workers=0)

    ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            ps += model(x.to(device)).argmax(1).cpu().tolist()
            ys += y.tolist()

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    acc = accuracy_score(ys, ps)
    print(f"Test accuracy: {acc:.3f} on {len(ys)} held-out images\n")
    print(classification_report(ys, ps, target_names=classes, zero_division=0))
    cm = confusion_matrix(ys, ps, labels=list(range(len(classes))))
    plot_confusion(cm, classes, args.run / "confusion_matrix.png")
    print(f"Confusion matrix -> {args.run / 'confusion_matrix.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
