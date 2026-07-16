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


def test_attach_titles_joins_and_fuse_carries_title(load_module):
    """titles.json 'source' keys are '<stem>#<idx>' where idx counts only book
    boxes; attach_titles must reproduce that count (non-book boxes don't shift
    it), respect min_score, and fuse must surface title/text/keywords."""
    bs = load_module("understanding/fusion/build_scene.py", "build_scene")
    cio = sys.modules["colmap_io"]

    c = np.array([1.0, 0.0, 6.0])  # one book, seen from 2 frames
    rng = np.random.default_rng(1)
    points = {i + 1: cio.Point3D(i + 1, c + rng.normal(0, 0.03, 3),
                                 np.array([128, 128, 128]), 0.1) for i in range(20)}
    cam = cio.Camera(1, "PINHOLE", 800, 600, np.array([600., 600., 400., 300.]))
    images, detections = {}, []
    for k in range(2):
        center = np.array([0.2 * k, 0.0, 0.0])
        img = cio.Image(k + 1, np.array([1., 0, 0, 0]), -center, 1, f"f{k}.jpg")
        images[k + 1] = img
        uv, _z = cio.project(cam, img, c)
        u, v = float(uv[0]), float(uv[1])
        # a non-book box FIRST: must not consume a '#idx' slot in the join
        # (full-frame so it lifts into its own object and we can assert on it)
        detections.append({"image": f"f{k}.jpg", "class": "shelf",
                           "box_xyxy": [0, 0, 800, 600], "confidence": 0.9})
        detections.append({"image": f"f{k}.jpg", "class": "book",
                           "box_xyxy": [u - 40, v - 40, u + 40, v + 40],
                           "confidence": 0.9})

    titles = [
        # book #0 of f0: good match -> attaches
        {"source": "f0#0", "ocr": {"text": "RHYTHM< WAR"}, "query": "rhythm war",
         "match": {"title": "Rhythm of War", "score": 0.87}},
        # book #0 of f1: below min_score -> must NOT attach
        {"source": "f1#0", "ocr": {"text": "junk"}, "query": "junk",
         "match": {"title": "Wrong Book", "score": 0.30}},
        # unmatched record: ignored
        {"source": "f1#1", "ocr": {"text": ""}, "query": "", "match": None},
    ]
    n = bs.attach_titles(detections, titles, min_score=0.45)
    assert n == 1
    assert detections[1].get("title_match", {}).get("title") == "Rhythm of War"
    assert "title_match" not in detections[3]  # 0.30 < min_score

    objs = bs.fuse({1: cam}, images, points, detections,
                   dist_thresh=0.5, min_points=3, min_frames=2)
    book = next(o for o in objs if o["category"] == "book")
    assert book["title"] == "Rhythm of War"
    assert book["text"] == "RHYTHM< WAR"
    assert {"rhythm", "war"} <= set(book["keywords"])
    # objects without a title stay schema-identical (fields absent, not null)
    shelf = next(o for o in objs if o["category"] == "shelf")
    assert "title" not in shelf and "text" not in shelf


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
