"""Capture-QA gate: does it actually catch the failure it claims to catch?

Each test builds a synthetic capture with exactly one defect and asserts the gate
names that defect and not the others. A gate that passes everything is worse than
no gate, so the negative cases (clean capture -> GO) matter as much as the positives.
"""
import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture
def rc(load_module):
    return load_module("pipeline/rate_capture.py", "rate_capture")


def _texture(seed=0, w=320, h=240):
    """A busy, high-frequency field -- something phase correlation can lock onto."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def _write_pan(tmp_path, n=12, step=6, *, gain=None, wb=None, blur_idx=(), base=None):
    """Write `n` frames panning across one big texture by `step` px each.

    Panning a single source image (rather than making independent frames) is what
    gives consecutive frames real shared content, so overlap is a true measurement
    instead of noise.
    """
    # base must be wide enough for the whole pan, or the last crops come out empty
    big = _texture(1, w=step * (n - 1) + 320, h=240) if base is None else base
    d = tmp_path / "frames"
    d.mkdir(exist_ok=True)
    for i in range(n):
        crop = big[:, i * step: i * step + 320].astype(np.float32).copy()
        if gain is not None:
            crop *= gain[i]
        if wb is not None:
            crop[:, :, 0] *= wb[i]
        if i in blur_idx:
            # crude box blur -> collapses high-frequency detail, like motion blur
            k = 9
            pad = np.pad(crop, ((k // 2, k // 2), (k // 2, k // 2), (0, 0)), mode="edge")
            acc = np.zeros_like(crop)
            for dy in range(k):
                for dx in range(k):
                    acc += pad[dy:dy + crop.shape[0], dx:dx + crop.shape[1]]
            crop = acc / (k * k)
        Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)).save(d / f"f_{i:03d}.jpg", quality=95)
    return d


def test_clean_capture_passes(rc, tmp_path):
    d = _write_pan(tmp_path, n=12, step=6)
    m = rc.analyse(sorted(d.iterdir()))
    v, bad, _ = rc.verdict(m)
    assert not bad, f"clean capture flagged: {bad}"
    assert v in ("GO", "MARGINAL")
    assert m["overlap"]["median"] > 0.9   # 6px of 320 -> ~98% shared


def test_moving_too_fast_is_caught(rc, tmp_path):
    # 150px of a 320px frame -> ~53% shared, just under the floor and still inside
    # the range where the shift measure is exact (see test_aliasing_limit_is_known).
    d = _write_pan(tmp_path, n=10, step=150)
    m = rc.analyse(sorted(d.iterdir()))
    v, bad, _ = rc.verdict(m)
    assert v == "RESHOOT"
    assert any("overlap" in b for b in bad), bad
    assert m["overlap"]["median"] < rc.MIN_OVERLAP


def test_aliasing_limit_is_known(rc):
    """Pin the limitation so nobody trusts the overlap number blindly.

    Phase correlation is ambiguous modulo the frame size. Past half a frame the
    shift wraps and overlap is reported as BETTER than reality -- exactly the wrong
    direction for a safety gate. This test exists to keep that documented and
    visible rather than lurking.
    """
    rng = np.random.default_rng(1)
    big = (rng.integers(0, 255, size=(240, 2200, 3), dtype=np.uint8).astype(np.float32)
           @ np.array([0.299, 0.587, 0.114], dtype=np.float32))

    # inside the range: exact
    for step in (60, 120, 150):
        _, dx, _ = rc.phase_shift(big[:, 0:320], big[:, step:step + 320])
        assert abs(abs(dx) - step) < 2, f"step {step} should measure exactly, got {dx}"

    # past half a frame: wraps, and OVER-reports overlap
    for step, true_ovl in ((200, 0.38), (260, 0.19)):
        dy, dx, peak = rc.phase_shift(big[:, 0:320], big[:, step:step + 320])
        reported = rc.overlap_fraction(dy, dx, (240, 320))
        assert reported > true_ovl + 0.15, "aliasing should over-report; if this fails the limit changed"
        assert peak < 0.35, "peak strength is the tell that survives aliasing"


def test_unlocked_exposure_is_caught(rc, tmp_path):
    # brightness ramps 60% across the take -- an AE that kept re-metering
    gain = np.linspace(0.7, 1.3, 12)
    d = _write_pan(tmp_path, n=12, step=6, gain=gain)
    m = rc.analyse(sorted(d.iterdir()))
    v, bad, _ = rc.verdict(m)
    assert v == "RESHOOT"
    assert any("exposure" in b for b in bad), bad


def test_unlocked_white_balance_is_caught(rc, tmp_path):
    # red channel drifts while overall level holds -- AWB hunting, not a light change
    wb = np.linspace(0.8, 1.25, 12)
    d = _write_pan(tmp_path, n=12, step=6, wb=wb)
    m = rc.analyse(sorted(d.iterdir()))
    v, bad, _ = rc.verdict(m)
    assert v == "RESHOOT"
    assert any("white balance" in b for b in bad), bad


def test_motion_blur_is_caught(rc, tmp_path):
    # half the frames smeared
    d = _write_pan(tmp_path, n=12, step=6, blur_idx=tuple(range(0, 12, 2)))
    m = rc.analyse(sorted(d.iterdir()))
    v, bad, _ = rc.verdict(m)
    assert v == "RESHOOT"
    assert any("sharpness" in b or "blur" in b for b in bad), bad


def test_overlap_fraction_math(rc):
    assert rc.overlap_fraction(0, 0, (100, 100)) == pytest.approx(1.0)
    assert rc.overlap_fraction(0, 50, (100, 100)) == pytest.approx(0.5)
    assert rc.overlap_fraction(0, 200, (100, 100)) == pytest.approx(0.0)


def test_phase_shift_recovers_known_translation(rc):
    base = _texture(7, w=256, h=256).astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    shifted = np.roll(base, 10, axis=1)
    dy, dx, peak = rc.phase_shift(base, shifted)
    assert abs(dy) < 1e-6
    assert abs(abs(dx) - 10) < 1.5, f"expected |dx|~10, got {dx}"
    assert peak > 0.05
