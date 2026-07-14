#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Train the book classifier via transfer learning (timm backbone).

Dataset layout (ImageFolder-style):  DATA/<class_name>/<images>

Lifecycle: stratified per-class train/val/test split -> warm up the classifier
head with the backbone frozen -> fine-tune the whole network at a low LR ->
early-stop on validation macro-F1. Saves the best checkpoint, the class list,
the exact held-out test split (for eval.py), and a training-curve PNG.

Two small-data correctness details, both easy to get wrong:
  * BatchNorm during warmup: freezing requires_grad does NOT stop BatchNorm from
    updating its running mean/var in train() mode -- which quietly corrupts a
    "frozen" backbone's pretrained stats on a tiny dataset. So during warmup we
    put the backbone in eval() and only the head in train(). (convnext_tiny uses
    LayerNorm and is immune -- one reason it's the default here.)
  * Frame leakage: if your images are consecutive ffmpeg frames of the same book,
    a naive split leaks near-duplicates across train/val/test and inflates the
    number. Dedupe near-duplicate frames first, or group by source clip. For a
    production pipeline use sklearn StratifiedGroupKFold grouped by clip (see
    ARCHITECTURE.md).

Example:
  python train.py --data data/books --model convnext_tiny.in12k_ft_in1k \
      --epochs 30 --out runs/books
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import common


def evaluate(model, loader, device):
    """Return (accuracy, macro_f1) on a loader."""
    from sklearn.metrics import f1_score
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps += model(x.to(device)).argmax(1).cpu().tolist()
            ys += y.tolist()
    acc = sum(int(a == b) for a, b in zip(ys, ps)) / max(len(ys), 1)
    f1 = f1_score(ys, ps, average="macro", zero_division=0)
    return acc, f1


def run_epoch(model, loader, device, opt, lossf, warmup: bool) -> float:
    # During warmup the backbone is frozen AND kept in eval() so BatchNorm running
    # stats don't drift; only the head trains. After warmup the whole net trains.
    if warmup:
        model.eval()
        model.get_classifier().train()
    else:
        model.train()
    running, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = lossf(model(x), y)
        loss.backward()
        opt.step()
        running += loss.item() * y.numel()
        n += y.numel()
    return running / max(n, 1)


def save_curve(history, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ep, [h["train_loss"] for h in history], "o-", color="tab:red", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(ep, [h["val_f1"] for h in history], "s-", color="tab:blue", label="val macro-F1")
    ax2.set_ylabel("val macro-F1", color="tab:blue")
    ax2.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path, help="root with <class>/ subfolders")
    ap.add_argument("--out", type=Path, default=Path("runs/books"))
    ap.add_argument("--model", default="convnext_tiny.in12k_ft_in1k",
                    help="timm backbone. Default is LayerNorm-based (BN-safe) with "
                         "strong 12k pretraining; efficientnet_b0 is a fast baseline.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=4, help="epochs with backbone frozen")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-finetune", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1,
                    help="mild regularizer that helps on small datasets")
    ap.add_argument("--patience", type=int, default=8, help="early-stop patience (epochs)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    common.seed_everything(args.seed)
    device = common.pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    classes, tr, va, te = common.stratified_split(args.data, seed=args.seed)
    if len(classes) < 2:
        raise SystemExit("Need at least 2 class folders under --data.")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Split: train={len(tr)} val={len(va)} test={len(te)}   device={device}")
    if not va:
        raise SystemExit("Validation split is empty (each class has only 1 image). "
                         "Add a few more images per class.")
    print(f"Backbone: {args.model}")

    model = common.build_model(args.model, len(classes), pretrained=True).to(device)
    tf_train = common.build_transforms(model, training=True)
    tf_eval = common.build_transforms(model, training=False)
    dl_tr = DataLoader(common.ListDataset(tr, tf_train),
                       batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(common.ListDataset(va, tf_eval),
                       batch_size=args.batch_size, shuffle=False, num_workers=0)

    lossf = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    history, best_f1, best_epoch, bad = [], -1.0, -1, 0
    ckpt = args.out / "model.pt"

    for epoch in range(1, args.epochs + 1):
        warmup = epoch <= args.warmup
        if warmup:
            common.freeze_backbone(model, True)
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr_head, weight_decay=args.weight_decay)
            phase = "head"
        else:
            common.freeze_backbone(model, False)
            opt = torch.optim.AdamW(model.parameters(),
                                    lr=args.lr_finetune, weight_decay=args.weight_decay)
            phase = "finetune"

        trloss = run_epoch(model, dl_tr, device, opt, lossf, warmup)
        vacc, vf1 = evaluate(model, dl_va, device)
        history.append({"epoch": epoch, "phase": phase, "train_loss": trloss,
                        "val_acc": vacc, "val_f1": vf1})
        print(f"epoch {epoch:02d} [{phase:8s}] train_loss={trloss:.4f} "
              f"val_acc={vacc:.3f} val_f1={vf1:.3f}")

        if vf1 > best_f1:
            best_f1, best_epoch, bad = vf1, epoch, 0
            common.save_checkpoint(ckpt, model, classes, args.model)
        elif not warmup:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop at epoch {epoch} (no val-F1 gain in {args.patience}).")
                break

    (args.out / "classes.json").write_text(json.dumps(classes, indent=2))
    (args.out / "test_split.json").write_text(
        json.dumps([[str(p), y] for p, y in te], indent=2))
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    save_curve(history, args.out / "training_curve.png")
    print(f"\nBest val macro-F1={best_f1:.3f} @epoch {best_epoch}. Saved -> {ckpt}")
    print(f"Test split ({len(te)} imgs) -> {args.out/'test_split.json'} (now run eval.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
