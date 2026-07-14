"""Unit tests for pipeline/validate_splat_ply.py (pure header parsing, no deps)."""


def _write_ply(path, props):
    lines = ["ply", "format binary_little_endian 1.0", "element vertex 1"]
    lines += [f"property float {p}" for p in props]
    lines += ["end_header"]
    path.write_bytes(("\n".join(lines) + "\n").encode("ascii"))


def test_splat_header_accepted(load_module, tmp_path):
    m = load_module("pipeline/validate_splat_ply.py", "vsp")
    p = tmp_path / "splat.ply"
    _write_ply(p, ["x", "y", "z", "scale_0", "rot_0", "opacity", "f_dc_0", "f_rest_0"])
    props = m.read_header_props(p)
    assert all(r in props for r in m.REQUIRED)


def test_point_cloud_header_rejected(load_module, tmp_path):
    m = load_module("pipeline/validate_splat_ply.py", "vsp")
    p = tmp_path / "points.ply"
    _write_ply(p, ["x", "y", "z", "red", "green", "blue"])
    props = m.read_header_props(p)
    assert not all(r in props for r in m.REQUIRED)
