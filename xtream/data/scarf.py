"""ScaRF.

- Images and IMU: the release's ROS 2 bag converted to a folder (`scripts/prepare_data.py scarf`: every camera
  frame of `/insta/cam0/image_raw/compressed`, IMU `/insta/imu/data_raw`) and the fisheye cam0 rectified onto
  the release's `config/rectified_intrinsic.yaml`, in
  `<prepared>/<seq>/<seq>_bag_sequence/{images/camera_rgb_undistorted, imu/imu_right/data.csv}`.
- Calibration: the release's `config/kalibr_imucam_chain.yaml` (cam0 `T_cam_imu`, `timeshift_cam_imu`) and
  `config/kalibr_imu_chain.yaml` (noise).
- Ground truth: `rNN_gt/poses_gt.txt` (TUM, cam0), read with evo.
"""

import os
import re

import numpy as np
import yaml

from xtream.data.base import Sequence
from xtream.paths import data_dir


def stamp_ns(path):
    """Image stamp [ns] from its file name `image_<s>_<ns>.jpg` (the last two integers of the stem)."""
    numbers = re.findall(r"\d+", os.path.splitext(os.path.basename(path))[0])
    s, ns = numbers[-2], numbers[-1].ljust(9, "0")[:9]
    return int(s) * 1_000_000_000 + int(ns)


def _yaml(path):
    txt = open(path).read()
    return yaml.safe_load(txt.split("\n", 1)[1] if txt.startswith("%YAML") else txt)


def load(name):
    from evo.tools.file_interface import read_tum_trajectory_file
    seq = os.path.join(data_dir("scarf", "prepared"), name, f"{name}_bag_sequence")
    img_dir = os.path.join(seq, "images", "camera_rgb_undistorted")
    paths = sorted((os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")), key=stamp_ns)
    stamps = np.array([stamp_ns(p) for p in paths], np.int64)
    cfg = os.path.join(data_dir("scarf"), "config")
    rect = _yaml(os.path.join(cfg, "rectified_intrinsic.yaml"))
    fx, fy, cx, cy = rect["pinhole_intrinsics"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    cam = _yaml(os.path.join(cfg, "kalibr_imucam_chain.yaml"))["cam0"]
    nz = _yaml(os.path.join(cfg, "kalibr_imu_chain.yaml"))["imu0"]
    imu = np.loadtxt(os.path.join(seq, "imu", "imu_right", "data.csv"), delimiter=",", comments="#")
    gt_file = os.path.join(data_dir("scarf"), name, f"{name}_gt", "poses_gt.txt")

    def gt():
        traj = read_tum_trajectory_file(gt_file)
        return np.round(traj.timestamps * 1e9).astype(np.int64), np.asarray(traj.poses_se3, float)

    return Sequence(dataset="scarf", name=name, image_paths=paths, stamps_ns=stamps, K=K,
                    size=tuple(rect["pinhole_resolution"]), imu_ns=imu[:, 0].astype(np.int64),
                    acc=imu[:, 4:7], gyro=imu[:, 1:4], T_cam_imu=np.asarray(cam["T_cam_imu"], float),
                    timeshift_ns=int(round(cam["timeshift_cam_imu"] * 1e9)),
                    noise=dict(acc_density=nz["accelerometer_noise_density"],
                               gyro_density=nz["gyroscope_noise_density"],
                               acc_walk=nz["accelerometer_random_walk"], gyro_walk=nz["gyroscope_random_walk"]),
                    gravity=9.81, gt_frame="cam0", T_gt_cam=np.eye(4), gt=gt).check()
