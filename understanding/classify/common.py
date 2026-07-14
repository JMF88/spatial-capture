"""Shared helpers for the book classifier (transfer learning with timm).

Kept in one place so train / eval / infer agree on how the model is built, how
images are transformed, and what a checkpoint contains. Pure dataset-splitting
lives in splits.py (torch-free, so it's unit-tested in the light CI).
"""
from __future__ import annotations

import random

import timm
import torch
from splits import IMG_EXTS, stratified_split  # noqa: F401  (re-exported for train/eval/infer)


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
