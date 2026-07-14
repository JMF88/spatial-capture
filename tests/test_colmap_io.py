"""Unit tests for the COLMAP binary IO + projection math (numpy only)."""
import numpy as np


def test_roundtrip_and_projection(load_module, tmp_path):
    m = load_module("understanding/fusion/colmap_io.py", "colmap_io")
    cam = m.Camera(1, "PINHOLE", 640, 480, np.array([500., 500., 320., 240.]))
    img = m.Image(1, np.array([1., 0, 0, 0]), np.array([0., 0, 0]), 1, "f0.jpg")
    pt = m.Point3D(7, np.array([0., 0, 5.]), np.array([10, 20, 30]), 0.5)

    m.write_cameras(tmp_path / "cameras.bin", {1: cam})
    m.write_images(tmp_path / "images.bin", {1: img})
    m.write_points3D(tmp_path / "points3D.bin", {7: pt})
    cams, imgs, pts = m.read_model(tmp_path)

    assert cams[1].model == "PINHOLE" and cams[1].width == 640
    assert imgs[1].name == "f0.jpg" and np.allclose(imgs[1].qvec, [1, 0, 0, 0])
    assert np.allclose(pts[7].xyz, [0, 0, 5]) and tuple(pts[7].rgb) == (10, 20, 30)

    # identity pose: X=(0,0,5) projects to the principal point (320,240) at depth 5
    uv, z = m.project(cams[1], imgs[1], np.array([0., 0, 5.]))
    assert np.allclose(uv, [320, 240], atol=1e-6) and abs(z - 5) < 1e-9
    # X=(1,0,5): u = fx*(1/5) + cx = 100 + 320 = 420
    uv2, _ = m.project(cams[1], imgs[1], np.array([1., 0, 5.]))
    assert np.allclose(uv2, [420, 240], atol=1e-6)


def test_projection_is_batched(load_module):
    m = load_module("understanding/fusion/colmap_io.py", "colmap_io")
    cam = m.Camera(1, "SIMPLE_PINHOLE", 100, 100, np.array([50., 50., 50.]))
    img = m.Image(1, np.array([1., 0, 0, 0]), np.array([0., 0, 0]), 1, "a")
    X = np.array([[0., 0, 2.], [0., 0, 4.]])
    uv, z = m.project(cam, img, X)
    assert uv.shape == (2, 2) and np.allclose(z, [2, 4])
    assert np.allclose(uv[0], [50, 50]) and np.allclose(uv[1], [50, 50])
