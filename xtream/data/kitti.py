"""KITTI odometry, sequences 00-10 (monocular: the release has no IMU).

- Images: `sequences/<seq>/image_2/*.png` (left colour camera, rectified), sorted.
- Stamps: `times.txt` [s from the sequence start], parsed exactly.
- Calibration: `calib.txt`, `P2` (rectified, no distortion); K = P2[:3, :3].
- Ground truth: `poses/<seq>.txt`, one cam0 pose per frame, matched by frame index; moved to cam2 exactly
  (rectified cameras: identity rotation, P2's translation).
"""

import os
from decimal import Decimal

import numpy as np
from PIL import Image

from xtream.data.base import Sequence
from xtream.paths import data_dir


def _calib(path):
    rows = {}
    for ln in open(path):
        k, _, v = ln.partition(":")
        if v.strip():
            rows[k.strip()] = np.array([float(x) for x in v.split()]).reshape(3, 4)
    return rows


def load(name):
    root = data_dir("kitti")
    seq_dir = os.path.join(root, "sequences", name)
    img_dir = os.path.join(seq_dir, "image_2")
    files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
    paths = [os.path.join(img_dir, f) for f in files]
    assert [int(f[:-4]) for f in files] == list(range(len(files))), (name, "image_2 files not 000000..N-1")
    stamps = np.array([int(Decimal(ln.strip()) * 1_000_000_000) for ln in open(os.path.join(seq_dir, "times.txt"))
                       if ln.strip()], np.int64)
    assert len(stamps) == len(files), (name, len(stamps), len(files))
    P2 = _calib(os.path.join(seq_dir, "calib.txt"))["P2"]
    K = P2[:, :3].copy()
    assert K[0, 1] == 0 and K[1, 0] == 0 and K[2, 2] == 1, K
    T_cam0_cam2 = np.eye(4)
    T_cam0_cam2[:3, 3] = -np.linalg.solve(K, P2[:, 3])          # X_cam2 = X_cam0 + K^-1 P2[:, 3]
    with Image.open(paths[0]) as im:
        size = im.size
    gt_file = os.path.join(root, "poses", f"{name}.txt")

    def gt():
        """One cam0 pose per frame, on the frames' stamps."""
        m = np.loadtxt(gt_file).reshape(-1, 3, 4)
        assert len(m) == len(files), (name, len(m), len(files))
        T = np.tile(np.eye(4), (len(m), 1, 1))
        T[:, :3] = m
        return stamps.copy(), T

    return Sequence(dataset="kitti", name=name, image_paths=paths, stamps_ns=stamps, K=K, size=size,
                    imu_ns=np.zeros(0, np.int64), acc=np.zeros((0, 3)), gyro=np.zeros((0, 3)), T_cam_imu=np.eye(4),
                    timeshift_ns=0, noise={}, gravity=9.81, gt_frame="cam0", T_gt_cam=T_cam0_cam2, gt=gt,
                    gt_match="index").check()
