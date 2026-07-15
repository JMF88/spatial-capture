#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Capture QA - grade a capture before spending GPU-hours reconstructing it.

Why this exists: splat training is the expensive step (30-90 min on a laptop 4070),
and it fails for reasons that were already visible in the frames -- exposure that
drifted, white balance that hunted, motion blur, or a camera that moved too fast
between frames to leave the solver any overlap. Every one of those is cheap to
measure and impossible to fix later. So measure first, then decide, rather than
discovering it after the trainer and a reshoot are both gone.

Reads the frames stage 1 produced and reports, per capture:

  sharpness    variance-of-Laplacian, absolute + relative. Catches motion blur.
  exposure     mean luma drift across frames. Catches an unlocked AE (or a
               phone quietly re-metering as you pan past a window).
  white bal.   R/G and B/G drift. Catches an unlocked AWB -- the one stock iOS
               will not let you lock, so this is the one to actually watch.
  flicker      frame-to-frame luma oscillation. Catches mains-frequency banding
               from LED/fluorescent light beating against a fast shutter.
  overlap      phase-correlation shift between consecutive frames -> what
               fraction of the view they share. Too little and SfM has nothing
               to match on; too much just means you walked slowly, which is free.
  clipping     blown / crushed pixels. Windows and bare bulbs become floaters.

Verdict is GO / MARGINAL / RESHOOT, with the reasons named. It is advisory: it
cannot see coverage holes or whether you closed the loop, and it says so.

Usage:
    python pipeline/rate_capture.py --frames data/shelf/frames
    python pipeline/rate_capture.py --frames data/shelf/frames --json out/shelf/capture_qa.json

Dependencies: numpy + pillow. No OpenCV, no GPU, no ML - same as stage 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Thresholds. Deliberately loose -- this gate should catch captures that are
# actually broken, not bikeshed good ones. A gate that cries wolf gets ignored.
MIN_OVERLAP = 0.55          # below: consecutive frames share too little view
LOW_OVERLAP = 0.65          # below: thin, will likely still solve
HIGH_OVERLAP = 0.97         # above: redundant (harmless, just slow)
EXPOSURE_DRIFT_BAD = 0.25   # fraction of mean luma
EXPOSURE_DRIFT_WARN = 0.12
WB_DRIFT_BAD = 0.10         # fraction, R/G or B/G range
WB_DRIFT_WARN = 0.05
SOFT_FRAC_BAD = 0.30        # fraction of frames well below median sharpness
SOFT_FRAC_WARN = 0.15
CLIP_WARN = 0.02            # fraction of pixels blown
FLICKER_WARN = 0.03         # frame-to-frame luma oscillation


def load_gray(path: Path, long_edge: int = 320) -> tuple[np.ndarray, np.ndarray]:
    """Return (gray_small, rgb_mean) for one frame.

    Downscaled: every measure here is about how frames relate to each other, not
    fine detail, and 320px keeps a 300-frame capture under a few seconds.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = long_edge / max(w, h)
        if scale < 1:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
    rgb_mean = arr.reshape(-1, 3).mean(axis=0)
    gray = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return gray, rgb_mean


def laplacian_var(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian. Higher = sharper."""
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1]
        + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    return float(lap.var())


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Translation (dy, dx) from a to b by phase correlation, plus peak strength.

    Phase correlation rather than feature matching: no OpenCV, no descriptors, and
    for a slow orbit or walk the motion between consecutive frames is dominated by
    translation, which is exactly what this recovers. Rotation and zoom degrade it
    -- the peak strength drops, which is why it's reported rather than hidden.

    KNOWN LIMIT -- the shift is ambiguous modulo the frame size, so anything past
    half a frame wraps and comes back *under*-reported. Measured against a synthetic
    pan (see tests/test_rate_capture.py::test_aliasing_limit_is_known):

        true shift   true overlap   reported overlap
             150px            53%               53%   <- exact
             159px            50%               50%   <- exact, the boundary
             170px            47%               53%   <- wrapped
             200px            38%               62%   <- wrapped
             260px            19%               81%   <- wrapped, badly

    So overlap is trustworthy from 50-100% and saturates below that: a genuinely
    reckless capture can report a comfortable number. The peak strength keeps
    falling monotonically across that boundary (0.95 -> 0.15 on the same pan) and
    is the honest tell, but its absolute scale depends on how textured the subject
    is -- random noise peaks far higher than real photographs -- so it is reported
    rather than thresholded. Calibrate `match_confidence` against a real capture
    before trusting a hard floor.
    """
    a = a - a.mean()
    b = b - b.mean()
    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    r = fa * np.conj(fb)
    mag = np.abs(r)
    r = np.divide(r, mag, out=np.zeros_like(r), where=mag > 1e-6)
    corr = np.fft.irfft2(r, s=a.shape)
    peak = float(corr.max())
    dy, dx = np.unravel_index(int(np.argmax(corr)), corr.shape)
    h, w = a.shape
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return float(dy), float(dx), peak


def overlap_fraction(dy: float, dx: float, shape: tuple[int, int]) -> float:
    """Fraction of frame area still shared after a (dy, dx) shift."""
    h, w = shape
    ox = max(0.0, 1.0 - abs(dx) / w)
    oy = max(0.0, 1.0 - abs(dy) / h)
    return ox * oy


def analyse(frames: list[Path]) -> dict:
    grays, means = [], []
    for p in frames:
        g, m = load_gray(p)
        grays.append(g)
        means.append(m)

    sharp = np.array([laplacian_var(g) for g in grays])
    luma = np.array([g.mean() for g in grays])
    rgb = np.array(means)

    # Exposure: spread of frame brightness relative to the average frame.
    exp_drift = float((luma.max() - luma.min()) / max(luma.mean(), 1e-6))

    # Flicker: how much brightness bounces frame to frame, as opposed to drifting
    # slowly. Mains-frequency banding shows up here and not in exp_drift.
    d_luma = np.abs(np.diff(luma)) / max(luma.mean(), 1e-6)
    flicker = float(np.median(d_luma)) if len(d_luma) else 0.0

    # White balance: channel ratios are exposure-invariant, so drift here is the
    # camera changing its mind about colour, not the light changing level.
    rg = rgb[:, 0] / np.maximum(rgb[:, 1], 1e-6)
    bg = rgb[:, 2] / np.maximum(rgb[:, 1], 1e-6)
    wb_drift = float(max(
        (rg.max() - rg.min()) / max(rg.mean(), 1e-6),
        (bg.max() - bg.min()) / max(bg.mean(), 1e-6),
    ))

    med_sharp = float(np.median(sharp))
    soft_frac = float((sharp < 0.5 * med_sharp).mean())

    clipped, crushed = [], []
    for g in grays:
        clipped.append(float((g >= 250).mean()))
        crushed.append(float((g <= 5).mean()))

    overlaps, peaks = [], []
    for i in range(len(grays) - 1):
        dy, dx, pk = phase_shift(grays[i], grays[i + 1])
        overlaps.append(overlap_fraction(dy, dx, grays[i].shape))
        peaks.append(pk)
    overlaps = np.array(overlaps) if overlaps else np.array([1.0])
    peaks = np.array(peaks) if peaks else np.array([0.0])
    # Relative, because absolute peak scale depends on subject texture, not on capture
    # quality: half the median means this pair shares markedly less than the take's norm.
    med_peak = float(np.median(peaks))
    weak_frac = float((peaks < 0.5 * med_peak).mean()) if med_peak > 0 else 0.0

    return {
        "frames": len(frames),
        "sharpness": {
            "median": med_sharp,
            "p10": float(np.percentile(sharp, 10)),
            "soft_fraction": soft_frac,
        },
        "exposure": {"drift": exp_drift, "flicker": flicker},
        "white_balance": {"drift": wb_drift},
        "overlap": {
            "median": float(np.median(overlaps)),
            "p10": float(np.percentile(overlaps, 10)),
            "below_min_fraction": float((overlaps < MIN_OVERLAP).mean()),
            "match_confidence": med_peak,
            "weak_match_fraction": weak_frac,
            "note": "shift-based overlap is exact from 50-100% and saturates below 50% (aliasing) - see phase_shift docstring",
        },
        "clipping": {
            "blown_median": float(np.median(clipped)),
            "crushed_median": float(np.median(crushed)),
        },
    }


def verdict(m: dict) -> tuple[str, list[str], list[str]]:
    """Return (verdict, blockers, warnings)."""
    bad: list[str] = []
    warn: list[str] = []

    ov = m["overlap"]
    if ov["median"] < MIN_OVERLAP:
        bad.append(
            f"overlap median {ov['median']:.0%} < {MIN_OVERLAP:.0%} - you moved too fast "
            f"between frames; SfM has little to match on. Walk slower or raise --fps."
        )
    elif ov["median"] < LOW_OVERLAP:
        warn.append(f"overlap median {ov['median']:.0%} is thin (want >{LOW_OVERLAP:.0%}).")
    elif ov["median"] > HIGH_OVERLAP:
        warn.append(
            f"overlap median {ov['median']:.0%} - near-duplicate frames. Harmless, but you "
            f"could move faster or drop --fps and save training time."
        )
    if ov["below_min_fraction"] > 0.15:
        warn.append(f"{ov['below_min_fraction']:.0%} of frame pairs are below {MIN_OVERLAP:.0%} overlap (jerky spots).")
    # A frame pair whose peak collapses relative to the rest of the take is either a
    # jerk (moved much further than usual) or an aliased shift reading -- both look
    # fine in the overlap number. Relative, not absolute: peak scale is subject-dependent.
    if ov["match_confidence"] > 0 and ov["weak_match_fraction"] > 0.15:
        warn.append(
            f"{ov['weak_match_fraction']:.0%} of frame pairs match far worse than the rest of this take. "
            f"Those are jerks or fast sweeps -- and note the shift measure saturates at {1 - 0.5:.0%} "
            f"overlap, so a fast move can still report a healthy number. Check those spots by eye."
        )
    if ov["match_confidence"] < 0.02:
        warn.append("very low phase-correlation confidence overall - lots of rotation/zoom, or a featureless subject. Treat overlap numbers as soft.")

    e = m["exposure"]
    if e["drift"] > EXPOSURE_DRIFT_BAD:
        bad.append(f"exposure drifted {e['drift']:.0%} across the take - AE was not locked. Lock AE/AF and reshoot.")
    elif e["drift"] > EXPOSURE_DRIFT_WARN:
        warn.append(f"exposure drifted {e['drift']:.0%} - lock looks soft, or the light changed.")
    if e["flicker"] > FLICKER_WARN:
        warn.append(
            f"frame-to-frame brightness bounces {e['flicker']:.1%} - looks like mains flicker beating "
            f"against the shutter. Try 1/120s (or 1/60s) under LED/fluorescent light."
        )

    w = m["white_balance"]["drift"]
    if w > WB_DRIFT_BAD:
        bad.append(f"white balance drifted {w:.0%} - AWB was hunting. Stock iOS AE/AF Lock does NOT lock WB; use a manual camera app.")
    elif w > WB_DRIFT_WARN:
        warn.append(f"white balance drifted {w:.0%} - mixed light sources, or AWB not locked.")

    s = m["sharpness"]
    if s["soft_fraction"] > SOFT_FRAC_BAD:
        bad.append(f"{s['soft_fraction']:.0%} of frames are well below median sharpness - motion blur. More light, faster shutter, slower movement.")
    elif s["soft_fraction"] > SOFT_FRAC_WARN:
        warn.append(f"{s['soft_fraction']:.0%} of frames are soft - tighten --keep to drop them.")

    c = m["clipping"]
    if c["blown_median"] > CLIP_WARN:
        warn.append(f"{c['blown_median']:.1%} of pixels are blown out - windows or bare bulbs. Expect floaters there.")

    if bad:
        return "RESHOOT", bad, warn
    if warn:
        return "MARGINAL", bad, warn
    return "GO", bad, warn


def main() -> int:
    ap = argparse.ArgumentParser(description="Grade a capture before reconstructing it.")
    ap.add_argument("--frames", type=Path, required=True, help="directory of extracted frames")
    ap.add_argument("--json", type=Path, help="also write the report as JSON")
    ap.add_argument("--max-frames", type=int, default=400, help="cap frames analysed (evenly sampled)")
    args = ap.parse_args()

    if not args.frames.is_dir():
        print(f"error: {args.frames} is not a directory", file=sys.stderr)
        return 2

    frames = sorted(p for p in args.frames.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(frames) < 2:
        print(f"error: need >=2 frames, found {len(frames)}", file=sys.stderr)
        return 2
    if len(frames) > args.max_frames:
        idx = np.linspace(0, len(frames) - 1, args.max_frames).astype(int)
        frames = [frames[i] for i in idx]

    m = analyse(frames)
    v, bad, warn = verdict(m)
    m["verdict"] = v
    m["blockers"] = bad
    m["warnings"] = warn

    ov, e = m["overlap"], m["exposure"]
    print(f"\ncapture QA - {args.frames}  ({m['frames']} frames analysed)")
    print(f"  overlap        median {ov['median']:.0%}   p10 {ov['p10']:.0%}")
    print(f"  sharpness      median {m['sharpness']['median']:.0f}   soft {m['sharpness']['soft_fraction']:.0%}")
    print(f"  exposure       drift {e['drift']:.1%}   flicker {e['flicker']:.1%}")
    print(f"  white balance  drift {m['white_balance']['drift']:.1%}")
    print(f"  clipping       blown {m['clipping']['blown_median']:.1%}")
    print(f"\n  VERDICT: {v}")
    for b in bad:
        print(f"    [blocker] {b}")
    for wr in warn:
        print(f"    [warn]    {wr}")
    if v == "GO":
        print("    nothing blocking. Note this can't see coverage holes or loop closure -")
        print("    it grades the frames, not whether you photographed the whole subject.")
    print()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(m, indent=2))
        print(f"wrote {args.json}")

    return 0 if v != "RESHOOT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
