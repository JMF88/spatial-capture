"""Unit tests for the frame-extraction sharpness measure (numpy + Pillow only)."""
import numpy as np
from PIL import Image


def test_laplacian_variance_sharp_gt_flat(load_module, tmp_path):
    m = load_module("pipeline/01_extract_frames.py", "extract_frames")

    flat = tmp_path / "flat.png"
    Image.new("L", (64, 64), 128).save(flat)

    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, size=(64, 64)).astype("uint8")
    sharp = tmp_path / "sharp.png"
    Image.fromarray(noise, "L").save(sharp)

    # a high-frequency (noisy) image is "sharper" by variance-of-Laplacian
    assert m.laplacian_variance(sharp) > m.laplacian_variance(flat)


def test_laplacian_variance_flat_is_zero(load_module, tmp_path):
    m = load_module("pipeline/01_extract_frames.py", "extract_frames")
    flat = tmp_path / "flat.png"
    Image.new("L", (48, 48), 200).save(flat)
    assert m.laplacian_variance(flat) == 0.0
