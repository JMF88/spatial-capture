"""Splat culling: the disk-vs-needle distinction, pinned.

The bug this guards against is subtle and cost a 33% cull of a good scene: 3DGS Gaussians
are SUPPOSED to flatten into disks, because that is how a cloud of blobs represents a
surface. Judging "too anisotropic" by longest/shortest therefore punishes the healthy case
-- a disk is 1:1:0.01, a ratio of 100:1, and perfect. The pathology is a needle, 1:0.01:0.01,
which is also 100:1 by that measure and completely different in shape.

longest/MIDDLE separates them: ~1 for a disk, huge for a needle. Measured on a real splat,
longest/shortest > 50:1 culled 33.7% (mostly legitimate disks) while longest/middle > 50:1
culled 3.0% (the actual splinters).
"""
import numpy as np
import pytest


@pytest.fixture
def cs(load_module):
    return load_module("pipeline/clean_splat.py", "clean_splat")


def _ply(tmp_path, xyz, scales, opacity=None, name="in.ply"):
    """Write a minimal but real 3DGS PLY."""
    n = len(xyz)
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    dt = np.dtype([(p, "<f4") for p in props])
    v = np.zeros(n, dtype=dt)
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    v["scale_0"], v["scale_1"], v["scale_2"] = np.log(scales).T
    v["opacity"] = 4.0 if opacity is None else opacity   # sigmoid(4) ~ 0.98
    v["rot_0"] = 1.0
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {n}\n" + "".join(f"property float {p}\n" for p in props) + "end_header\n")
    p = tmp_path / name
    with open(p, "wb") as f:
        f.write(hdr.encode())
        v.tofile(f)
    return p


def test_disks_survive_needles_die(cs, tmp_path):
    """The whole point. Equal numbers of each; only the needles should go."""
    rng = np.random.default_rng(0)
    xyz = rng.normal(0, 1, (200, 3))
    scales = np.empty((200, 3), dtype=np.float64)
    scales[:100] = [0.05, 0.05, 0.0005]      # disks: long/short=100:1 but long/mid=1:1
    scales[100:] = [0.05, 0.0005, 0.0005]    # needles: long/mid=100:1
    src = _ply(tmp_path, xyz, scales)
    out = tmp_path / "out.ply"
    import sys
    argv = sys.argv
    sys.argv = ["clean_splat", str(src), "--out", str(out), "--max-aniso", "30", "--keep", "100"]
    try:
        assert cs.main() == 0
    finally:
        sys.argv = argv
    props, v, _ = cs.read_ply(out)
    lin = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1))
    srt = np.sort(lin, axis=1)
    assert len(v) == 100, f"expected the 100 disks to survive, got {len(v)}"
    assert np.all(srt[:, 2] / srt[:, 1] < 30), "a needle survived"


def test_floaters_do_not_own_the_bbox(cs, tmp_path):
    """0.5% of points 50x away must not define the scene."""
    rng = np.random.default_rng(1)
    core = rng.normal(0, 1, (2000, 3))
    far = rng.normal(0, 1, (10, 3)) + 50.0
    xyz = np.vstack([core, far])
    scales = np.full((len(xyz), 3), 0.02)
    src = _ply(tmp_path, xyz, scales)
    out = tmp_path / "out.ply"
    import sys
    argv = sys.argv
    sys.argv = ["clean_splat", str(src), "--out", str(out)]
    try:
        assert cs.main() == 0
    finally:
        sys.argv = argv
    _, v, _ = cs.read_ply(out)
    kept = np.stack([v["x"], v["y"], v["z"]], 1)
    extent = kept.max(0) - kept.min(0)
    assert np.all(extent < 15), f"floaters still own the bbox: {extent}"


def test_veil_needs_both_halves(cs, tmp_path):
    """big AND faint dies; big-and-opaque and small-and-faint both live.

    This is the test that stops a future edit from "simplifying" the veil rule down to a
    plain opacity threshold. Median opacity in a real splat is 0.088 -- faint is the NORM,
    and culling on faintness alone deletes the surface.
    """
    import sys
    xyz = np.zeros((300, 3), dtype=np.float64)
    xyz[:, 0] = np.linspace(-1, 1, 300)
    scales = np.empty((300, 3))
    logit = np.empty(300)
    scales[:100], logit[:100] = 0.2, -4.0        # big + faint (op~0.018) -> VEIL, must die
    scales[100:200], logit[100:200] = 0.2, 4.0   # big + opaque -> a wall, must live
    scales[200:], logit[200:] = 0.005, -4.0      # small + faint -> detail, must live

    src = _ply(tmp_path, xyz, scales, opacity=logit)
    out = tmp_path / "out.ply"
    argv = sys.argv
    sys.argv = ["clean_splat", str(src), "--out", str(out),
                "--veil-size", "0.05", "--veil-opacity", "0.15",
                "--min-opacity", "0.0", "--keep", "100", "--max-scale-pct", "100"]
    try:
        assert cs.main() == 0
    finally:
        sys.argv = argv

    _, v, _ = cs.read_ply(out)
    size = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).max(1)
    op = 1 / (1 + np.exp(-v["opacity"]))
    assert len(v) == 200, f"expected the 100 veil blobs culled and 200 kept, got {len(v)}"
    assert not np.any((size > 0.05) & (op < 0.15)), "a veil Gaussian survived"
    assert np.sum(op > 0.5) == 100, "the opaque walls were culled -- veil rule lost its size half"
    assert np.sum(size < 0.01) == 100, "the faint detail was culled -- veil rule lost its opacity half"


def test_veil_disabled_by_zero(cs, tmp_path):
    """--veil-opacity 0 must turn the filter off, not cull everything."""
    import sys
    xyz = np.zeros((50, 3))
    xyz[:, 0] = np.linspace(-1, 1, 50)
    src = _ply(tmp_path, xyz, np.full((50, 3), 0.2), opacity=np.full(50, -4.0))
    out = tmp_path / "out.ply"
    argv = sys.argv
    sys.argv = ["clean_splat", str(src), "--out", str(out), "--veil-opacity", "0",
                "--min-opacity", "0.0", "--keep", "100", "--max-scale-pct", "100"]
    try:
        assert cs.main() == 0
    finally:
        sys.argv = argv
    _, v, _ = cs.read_ply(out)
    assert len(v) == 50


def test_refuses_a_point_cloud(cs, tmp_path):
    """A non-splat PLY must be rejected, not silently mangled."""
    props = ["x", "y", "z"]
    dt = np.dtype([(p, "<f4") for p in props])
    v = np.zeros(10, dtype=dt)
    p = tmp_path / "pc.ply"
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex 10\n"
           + "".join(f"property float {x}\n" for x in props) + "end_header\n")
    with open(p, "wb") as f:
        f.write(hdr.encode())
        v.tofile(f)
    import sys
    argv = sys.argv
    sys.argv = ["clean_splat", str(p), "--out", str(tmp_path / "o.ply")]
    try:
        assert cs.main() == 2, "should refuse a point cloud"
    finally:
        sys.argv = argv
