"""End-to-end fusion test on a synthetic scene: known 3D objects -> multi-view
detections -> assert fusion recovers the objects near their true positions."""
import sys

import numpy as np


def test_fuse_recovers_known_objects(load_module):
    bs = load_module("understanding/fusion/build_scene.py", "build_scene")
    cio = sys.modules["colmap_io"]  # build_scene imports it on load

    truth = {"book": np.array([1.0, 0.0, 6.0]), "plant": np.array([-1.0, 0.5, 6.0])}

    # a tight sparse-point cloud around each object
    rng = np.random.default_rng(0)
    points, pid = {}, 1
    for _name, c in truth.items():
        for _ in range(20):
            points[pid] = cio.Point3D(pid, c + rng.normal(0, 0.03, 3),
                                      np.array([128, 128, 128]), 0.1)
            pid += 1

    cam = cio.Camera(1, "PINHOLE", 800, 600, np.array([600., 600., 400., 300.]))
    cams = {1: cam}

    # 5 cameras translating along x (identity rotation, so t = -center)
    images, detections = {}, []
    for k in range(5):
        center = np.array([0.2 * k - 0.4, 0.0, 0.0])
        img = cio.Image(k + 1, np.array([1., 0, 0, 0]), -center, 1, f"f{k}.jpg")
        images[k + 1] = img
        for name, c in truth.items():
            uv, _z = cio.project(cam, img, c)
            u, v = float(uv[0]), float(uv[1])
            detections.append({
                "image": f"f{k}.jpg", "class": name,
                "box_xyxy": [u - 40, v - 40, u + 40, v + 40], "confidence": 0.9,
            })

    objs = bs.fuse(cams, images, points, detections,
                   dist_thresh=0.5, min_points=3, min_frames=2)

    assert len(objs) == 2, f"expected 2 objects, got {len(objs)}"
    got = {o["category"]: np.array(o["position"]) for o in objs}
    for name, c in truth.items():
        assert name in got
        assert np.linalg.norm(got[name] - c) < 0.2, f"{name} off: {got[name]} vs {c}"
    # schema sanity: viewer-required keys present
    for o in objs:
        assert {"id", "label", "category", "keywords", "position", "aabb",
                "confidence", "source"} <= set(o)


def test_fuse_ignores_unmatched_and_sparse(load_module):
    bs = load_module("understanding/fusion/build_scene.py", "build_scene")
    cio = sys.modules["colmap_io"]
    cam = cio.Camera(1, "PINHOLE", 800, 600, np.array([600., 600., 400., 300.]))
    img = cio.Image(1, np.array([1., 0, 0, 0]), np.array([0., 0, 0]), 1, "f0.jpg")
    points = {1: cio.Point3D(1, np.array([0., 0, 6.]), np.array([1, 2, 3]), 0.1)}
    # detection referencing an image we don't have -> skipped;
    # detection with too few supporting points -> skipped
    dets = [
        {"image": "missing.jpg", "class": "book", "box_xyxy": [0, 0, 800, 600]},
        {"image": "f0.jpg", "class": "book", "box_xyxy": [0, 0, 800, 600], "confidence": 0.5},
    ]
    objs = bs.fuse(cameras={1: cam}, images={1: img}, points3D=points, detections=dets,
                   min_points=5, min_frames=1)
    assert objs == []
