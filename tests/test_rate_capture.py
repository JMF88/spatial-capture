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
    """A meter fighting itself: chases content, overshoots, corrects, overshoots back.

    Oscillation is what a real auto-exposure did on this shelf -- 18.0% and 16.1% wobble
    across two takes. Note the SHAPE, not just the size; see the ramp test below.
    """
    gain = 1.0 + 0.22 * np.sin(np.arange(14) * 1.4)
    d = _write_pan(tmp_path, n=14, step=6, gain=gain)
    m = rc.analyse(sorted(d.iterdir()))
    v, bad, _ = rc.verdict(m)
    assert v == "RESHOOT"
    assert any("wobble" in b for b in bad), bad


def test_smooth_brightness_ramp_is_not_a_defect(rc, tmp_path):
    """The regression that matters, and it is a real one.

    Nine correctly-locked takes were flagged RESHOOT for "59% exposure drift" when the
    brightness had ramped 69->127 monotonically -- a straight line explaining 94% of it.
    The camera was perfect; that end of the shelf was brighter. Frame-mean spread cannot
    tell a re-metering camera from an unevenly lit room, and with the exposure locked
    (which is the advice) the ramp is GUARANTEED. Telling someone to reshoot a correct
    capture is worse than not gating at all, so a smooth ramp must never block.
    """
    gain = np.linspace(0.72, 1.28, 14)          # ~55% spread, perfectly smooth
    d = _write_pan(tmp_path, n=14, step=6, gain=gain)
    m = rc.analyse(sorted(d.iterdir()))
    v, bad, warn = rc.verdict(m)
    assert m["exposure"]["drift"] > 0.25, "the naive metric should still see a big spread"
    assert m["exposure"]["wobble"] < rc.WOBBLE_WARN, \
        f"a smooth ramp must not read as wobble: {m['exposure']['wobble']}"
    assert not any("wobble" in b for b in bad), f"an unevenly lit scene is not a defect: {bad}"
    assert any("smooth" in w.lower() for w in warn), \
        f"it should say the spread is the scene, not the camera: {warn}"


def test_known_limit_smooth_drift_reads_clean(rc, tmp_path):
    """Pin the blind spot instead of pretending it isn't there.

    An auto-exposure that drifts *gracefully*, with no chatter, is mathematically
    indistinguishable from a room that gets brighter -- at the frame-mean level they are
    the same signal. This metric catches meters that chatter, which is what meters do in
    practice. Stated so nobody reads a clean wobble as proof the camera was locked.
    """
    gain = np.linspace(1.0, 1.5, 14)            # a large but perfectly graceful AE drift
    d = _write_pan(tmp_path, n=14, step=6, gain=gain)
    m = rc.analyse(sorted(d.iterdir()))
    assert m["exposure"]["wobble"] < rc.WOBBLE_BAD, \
        "if this starts failing the metric got better than documented -- update the docstring"


def test_white_balance_is_reported_but_never_blocks(rc, tmp_path):
    """Colour is advisory, and this pins WHY rather than leaving it as a mystery.

    A hunting AWB and a subject that is simply a different colour over there produce the
    same frame statistics, and unlike brightness there is no shape that separates them:
    a subject's colour does not change smoothly as you move (white robot -> wood -> beige
    wall), so detrending isolates nothing. On one real shelf the distributions overlapped
    outright -- auto-WB takes measured 3.5% and 4.5% wobble, while two correctly LOCKED
    takes measured 3.7% and 4.2%. Higher than the auto ones. No threshold exists, so any
    blocker here is a coin flip dressed as rigour. Report it; let a human weigh it.
    """
    wb = np.linspace(0.8, 1.25, 12)          # red channel swings hard
    d = _write_pan(tmp_path, n=12, step=6, wb=wb)
    m = rc.analyse(sorted(d.iterdir()))
    _, bad, warn = rc.verdict(m)
    assert m["white_balance"]["drift"] > 0.05, "the measurement should still see it"
    assert not any("colour" in b or "white balance" in b for b in bad), \
        f"colour must never block -- it cannot tell camera from subject: {bad}"
    assert any("colour" in w or "ratios" in w for w in warn), f"but it must be reported: {warn}"


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
