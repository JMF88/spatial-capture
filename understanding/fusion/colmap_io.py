#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Minimal reader/writer for COLMAP's binary model (cameras/images/points3D.bin).

Dependency-free (struct + numpy). Enough to recover camera intrinsics + poses and
the sparse 3D points that the fusion stage projects detections against, plus the
world->pixel projection math. Layout follows COLMAP's stable binary format.

COLMAP convention: a world point X maps to camera coords as  Xc = R(qvec) @ X + t.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# model_id -> (name, num_params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}
MODEL_IDS = {name: mid for mid, (name, _) in CAMERA_MODELS.items()}
_INVALID_POINT3D = (1 << 64) - 1


def qvec2rotmat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray  # float64

    def K(self) -> np.ndarray:
        """3x3 intrinsics (falls back to focal=params[0], principal at center)."""
        p = self.params
        if self.model == "SIMPLE_PINHOLE":
            fx = fy = p[0]
            cx, cy = p[1], p[2]
        elif self.model == "PINHOLE":
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        else:
            fx = fy = p[0]
            cx, cy = self.width / 2.0, self.height / 2.0
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # (4,) w,x,y,z
    tvec: np.ndarray  # (3,)
    camera_id: int
    name: str
    xys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    point3D_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))

    def R(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)

    def world_to_camera(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X @ self.R().T + self.tvec


@dataclass
class Point3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float


def project(camera: Camera, image: Image, X):
    """Project world points X (..,3) to pixels (..,2). Returns (uv, depth_z)."""
    Xc = image.world_to_camera(X)
    uvw = Xc @ camera.K().T
    uv = uvw[..., :2] / uvw[..., 2:3]
    return uv, Xc[..., 2]


# ---------------- binary IO ----------------
def _r(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


def read_cameras(path) -> dict:
    cams = {}
    with open(path, "rb") as f:
        (n,) = _r(f, "<Q")
        for _ in range(n):
            cid, model_id, w, h = _r(f, "<IiQQ")
            name, num = CAMERA_MODELS[model_id]
            params = np.array(_r(f, "<" + "d" * num), dtype=np.float64)
            cams[cid] = Camera(cid, name, w, h, params)
    return cams


def read_images(path) -> dict:
    images = {}
    with open(path, "rb") as f:
        (n,) = _r(f, "<Q")
        for _ in range(n):
            image_id = _r(f, "<I")[0]
            qvec = np.array(_r(f, "<4d"), dtype=np.float64)
            tvec = np.array(_r(f, "<3d"), dtype=np.float64)
            camera_id = _r(f, "<I")[0]
            name = b""
            while (ch := f.read(1)) != b"\x00":
                name += ch
            (n2d,) = _r(f, "<Q")
            xys = np.zeros((n2d, 2), dtype=np.float64)
            pids = np.zeros((n2d,), dtype=np.int64)
            for i in range(n2d):
                x, y, pid = _r(f, "<ddQ")
                xys[i] = (x, y)
                pids[i] = -1 if pid == _INVALID_POINT3D else pid
            images[image_id] = Image(image_id, qvec, tvec, camera_id,
                                     name.decode("utf-8"), xys, pids)
    return images


def read_points3D(path) -> dict:
    points = {}
    with open(path, "rb") as f:
        (n,) = _r(f, "<Q")
        for _ in range(n):
            pid = _r(f, "<Q")[0]
            xyz = np.array(_r(f, "<3d"), dtype=np.float64)
            rgb = np.array(_r(f, "<3B"), dtype=np.uint8)
            (error,) = _r(f, "<d")
            (track_len,) = _r(f, "<Q")
            if track_len:
                f.read(struct.calcsize("<II") * track_len)  # skip track
            points[pid] = Point3D(pid, xyz, rgb, float(error))
    return points


def read_model(sparse_dir):
    d = Path(sparse_dir)
    return (read_cameras(d / "cameras.bin"),
            read_images(d / "images.bin"),
            read_points3D(d / "points3D.bin"))


def write_cameras(path, cams: dict) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for c in cams.values():
            f.write(struct.pack("<IiQQ", c.id, MODEL_IDS[c.model], c.width, c.height))
            f.write(struct.pack("<" + "d" * len(c.params), *[float(v) for v in c.params]))


def write_images(path, images: dict) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for im in images.values():
            f.write(struct.pack("<I", im.id))
            f.write(struct.pack("<4d", *[float(v) for v in im.qvec]))
            f.write(struct.pack("<3d", *[float(v) for v in im.tvec]))
            f.write(struct.pack("<I", im.camera_id))
            f.write(im.name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", len(im.xys)))
            for (x, y), pid in zip(im.xys, im.point3D_ids):
                out_pid = _INVALID_POINT3D if int(pid) < 0 else int(pid)
                f.write(struct.pack("<ddQ", float(x), float(y), out_pid))


def write_points3D(path, points: dict) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for p in points.values():
            f.write(struct.pack("<Q", p.id))
            f.write(struct.pack("<3d", *[float(v) for v in p.xyz]))
            f.write(struct.pack("<3B", *[int(v) for v in p.rgb]))
            f.write(struct.pack("<d", float(p.error)))
            f.write(struct.pack("<Q", 0))  # empty track
