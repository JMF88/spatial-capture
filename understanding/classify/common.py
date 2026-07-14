"""Shared helpers for the book classifier (transfer learning with timm).

Kept in one place so train / eval / infer agree on how the model is built, how
images are transformed, and what a checkpoint contains.
"""
from __future__ import annotations

import random
from pathlib import Path

import timm
import torch

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but torch.cuda.is_available() is "
              "False (CPU-only torch?). Falling back to CPU.")
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(name: str, num_classes: int, pretrained: bool = True):
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def build_transforms(model, training: bool, train_scale=(0.65, 1.0)):
    """Model-correct transforms (resize/crop/normalize) straight from timm's
    data config, so we never hard-code the wrong mean/std for a backbone.

    For training we widen RandomResizedCrop's scale from timm's default
    (0.08, 1.0) -- which crops down to 8% of the area and shreds book
    spines/covers -- to a gentler (0.65, 1.0)."""
    from timm.data import create_transform
    try:
        from timm.data import resolve_model_data_config
        cfg = resolve_model_data_config(model)
    except Exception:
        from timm.data import resolve_data_config
        cfg = resolve_data_config({}, model=model)
    cfg = {**cfg, "is_training": training}
    if training and train_scale is not None:
        cfg["scale"] = train_scale
    return create_transform(**cfg)


def freeze_backbone(model, freeze: bool = True) -> None:
    """Freeze everything except the classifier head (transfer-learning warmup)."""
    for p in model.parameters():
        p.requires_grad = not freeze
    for p in model.get_classifier().parameters():
        p.requires_grad = True


class ListDataset(torch.utils.data.Dataset):
    """Dataset over an explicit list of (path, label) pairs."""

    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def stratified_split(root, ratios=(0.7, 0.15, 0.15), seed=0):
    """Split each class folder independently so all splits keep every class.

    Layout expected:  root/<class_name>/<image files>
    Returns (classes, train_items, val_items, test_items).
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
        # Guarantee non-empty splits for small classes: int(n*0.15) is 0 for n<=6,
        # which would give an empty val set and crash macro-F1 downstream.
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


def save_checkpoint(path, model, classes, model_name):
    torch.save(
        {"state_dict": model.state_dict(), "classes": classes, "model_name": model_name},
        path,
    )


def load_for_inference(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["model_name"], len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck["classes"]
