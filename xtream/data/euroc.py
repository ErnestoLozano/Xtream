"""EuRoC MAV.

- Images: `mav0/cam0` undistorted as MASt3R-Fusion's `EurocDataset` reads them (`scripts/prepare_data.py euroc`),
  in `<prepared>/<seq>/cam0/{data, data.csv, intrinsics.txt}`, every row of `mav0/cam0/data.csv`.
- IMU, calibration and noise: the release's `mav0/imu0/data.csv`, `mav0/cam0/sensor.yaml` (`T_BS`, body <- cam0)
  and `mav0/imu0/sensor.yaml`.
- Ground truth: `mav0/state_groundtruth_estimate0/data.csv` (body = IMU), read with evo.
"""

import os

import numpy as np
import yaml

from xtream.data.base import Sequence
from xtream.paths import data_dir, dataset_cfg


def load(name):
    from evo.tools.file_interface import read_euroc_csv_trajectory
    mav0 = os.path.join(data_dir("euroc"), dataset_cfg("euroc")["sequences"][name], "mav0")
    cam = os.path.join(data_dir("euroc", "prepared"), name, "cam0")
    rows = np.loadtxt(os.path.join(cam, "data.csv"), delimiter=",", dtype=str, comments="#", ndmin=2)
    paths = [os.path.join(cam, "data", f.strip()) for f in rows[:, 1]]
    stamps = rows[:, 0].astype(np.int64)
    fx, fy, cx, cy, W, H = np.loadtxt(os.path.join(cam, "intrinsics.txt"), comments="#")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    s_cam = yaml.safe_load(open(os.path.join(mav0, "cam0", "sensor.yaml")))
    s_imu = yaml.safe_load(open(os.path.join(mav0, "imu0", "sensor.yaml")))
    T_B_cam = np.asarray(s_cam["T_BS"]["data"], float).reshape(4, 4)
    T_B_imu = np.asarray(s_imu["T_BS"]["data"], float).reshape(4, 4)
    imu = np.loadtxt(os.path.join(mav0, "imu0", "data.csv"), delimiter=",", comments="#")
    gt_file = os.path.join(mav0, "state_groundtruth_estimate0", "data.csv")

    def gt():
        traj = read_euroc_csv_trajectory(gt_file)
        return np.round(traj.timestamps * 1e9).astype(np.int64), np.asarray(traj.poses_se3, float)

    return Sequence(dataset="euroc", name=name, image_paths=paths, stamps_ns=stamps, K=K, size=(int(W), int(H)),
                    imu_ns=imu[:, 0].astype(np.int64), acc=imu[:, 4:7], gyro=imu[:, 1:4],
                    T_cam_imu=np.linalg.inv(T_B_cam) @ T_B_imu, timeshift_ns=0,
                    noise=dict(acc_density=s_imu["accelerometer_noise_density"],
                               gyro_density=s_imu["gyroscope_noise_density"],
                               acc_walk=s_imu["accelerometer_random_walk"], gyro_walk=s_imu["gyroscope_random_walk"]),
                    gravity=9.81, gt_frame="body", T_gt_cam=np.linalg.inv(T_B_imu) @ T_B_cam, gt=gt).check()
