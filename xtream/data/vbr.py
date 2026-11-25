"""VBR (the 7 training sequences with ground truth).

- Images: the LoGeR authors' release (`Junyi42/vbr_processed`: `cam_l` raw, distorted, RGB, 1388x700), every
  file of `<frames>/<seq>_processed_aligned/rgb` in natural order. K is the release's `intrinsics.txt`.
- Stamps: the release has none; each file's camera stamp comes from `vbr_frame_maps/<seq>.csv` (file index ->
  camera stamp [ns], matched to the official bags' image messages). Two consecutive files can share a stamp
  (two images in one camera period): both are kept, the second listed in `dup`.
- IMU: `<raw>/<site>/<seq>/imu.csv`, the official bags' `/imu/data` (stamp [ns], gyro xyz, acc xyz), on the
  IMU's clock.
- Calibration: the official `vbr_calib.yaml` (`cam_l`, `imu`: `T_b`, `timeshift_cam_imu`, noise densities and
  random walks, `cam_l` radtan distortion).
- Ground truth: the release's `processed_gt/<seq>_gt.txt`, one camera pose per file, matched by frame index.
"""

import os
import re

import numpy as np
import yaml

from xtream.data.base import Sequence
from xtream.paths import data_dir, dataset_cfg

FRAME_MAPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vbr_frame_maps")


def load(name):
    from evo.core.trajectory import xyz_quat_wxyz_to_se3_poses
    seq_dir = os.path.join(data_dir("vbr"), dataset_cfg("vbr")["sequences"][name])
    frames = data_dir("vbr", "frames")
    rel = os.path.join(frames, f"{name}_processed_aligned")
    files = sorted((f for f in os.listdir(os.path.join(rel, "rgb")) if f.endswith(".png")),
                   key=lambda f: int(re.findall(r"\d+", f)[0]))
    fm = np.loadtxt(os.path.join(FRAME_MAPS, f"{name}.csv"), delimiter=",", dtype=np.int64, comments="#", ndmin=2)
    ts = dict(zip(fm[:, 0].tolist(), fm[:, 1].tolist()))
    idx = [int(re.findall(r"\d+", f)[0]) for f in files]
    assert all(k in ts for k in idx), (name, [k for k in idx if k not in ts][:5])
    paths = [os.path.join(rel, "rgb", f) for f in files]
    stamps = np.array([ts[k] for k in idx], np.int64)
    dup = tuple((np.flatnonzero(np.diff(stamps) == 0) + 1).tolist())
    gt_file = os.path.join(frames, "processed_gt", f"{name}_gt.txt")
    K = np.loadtxt(os.path.join(rel, "intrinsics.txt"))
    calib = yaml.safe_load(open(os.path.join(seq_dir, "vbr_calib.yaml")))
    cam, im = calib["cam_l"], calib["imu"]
    assert np.allclose([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], cam["intrinsics"], rtol=1e-6)
    assert cam["distortion_model"] == "radtan", cam["distortion_model"]
    distortion = tuple(float(x) for x in cam["distortion_coeffs"])
    T_b_cam, T_b_imu = np.asarray(cam["T_b"], float), np.asarray(im["T_b"], float)
    imu = np.loadtxt(os.path.join(seq_dir, "imu.csv"), delimiter=",", comments="#")

    def gt():
        """One camera pose per file (stamped by frame index in the release), on the frames' stamps."""
        m = np.loadtxt(gt_file, comments="#")
        assert len(m) == len(files) and np.array_equal(m[:, 0], np.arange(len(files))), (len(m), len(files))
        poses = xyz_quat_wxyz_to_se3_poses(m[:, 1:4], m[:, [7, 4, 5, 6]])
        return stamps.copy(), np.asarray(poses, float)

    return Sequence(dataset="vbr", name=name, image_paths=paths, stamps_ns=stamps, K=K,
                    size=tuple(cam["resolution"]), imu_ns=imu[:, 0].astype(np.int64), acc=imu[:, 4:7],
                    gyro=imu[:, 1:4], T_cam_imu=np.linalg.inv(T_b_cam) @ T_b_imu,
                    timeshift_ns=int(round(cam["timeshift_cam_imu"] * 1e9)),
                    noise=dict(acc_density=im["accelerometer_noise_density"],
                               gyro_density=im["gyroscope_noise_density"],
                               acc_walk=im["accelerometer_random_walk"], gyro_walk=im["gyroscope_random_walk"]),
                    gravity=9.81, gt_frame="cam", T_gt_cam=np.eye(4), gt=gt,
                    distortion=distortion, dup=dup, gt_match="index").check()
