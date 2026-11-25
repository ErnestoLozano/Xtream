"""Oxford Spires.

- Images: the front fisheye camera rectified to a pinhole (`scripts/prepare_data.py oxford_spires`: the fisheye K
  reused as the pinhole K, principal point moved to the image centre), in
  `<prepared>/<sequence>/raw/{cam0, rgb.txt, cam0_intrinsics.txt}`, every frame `rgb.txt` lists, in order.
- IMU: the sequence's `raw/imu.csv` (secs, nsecs, acc, ang_vel), as released.
- Calibration: the dataset's sensor configuration (`oxford_spires_sensor.yaml`); IMU noise from the release's
  `calibration/imu.yaml`.
- Ground truth: `processed/trajectory/gt-tum.txt`, in the base frame, as the dataset's localisation benchmark
  scores it.
"""

import os

import numpy as np
import yaml

from xtream.data.base import Sequence
from xtream.paths import data_dir, dataset_cfg

SENSOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oxford_spires_sensor.yaml")


def _T(t_xyz_q_xyzw):
    """4x4 from t_xyz + q_xyzw (as the dataset's tools build it)."""
    from evo.core.trajectory import xyz_quat_wxyz_to_se3_poses
    t, q = list(t_xyz_q_xyzw[:3]), list(t_xyz_q_xyzw[3:])
    return np.asarray(xyz_quat_wxyz_to_se3_poses([t], [[q[-1]] + q[:-1]])[0], float)


def read_tum_exact(path):
    """TUM file with `<secs>.<nsecs>` stamps: (stamps [ns] int64, poses (N, 4, 4)), stamps parsed exactly."""
    from evo.core.trajectory import xyz_quat_wxyz_to_se3_poses
    st, xyz, q = [], [], []
    for ln in open(path):
        if ln.startswith("#") or not ln.strip():
            continue
        v = ln.split()
        assert len(v) == 8, ln
        sec, nsec = v[0].split(".")
        st.append(int(sec) * 1_000_000_000 + int(nsec.ljust(9, "0")))
        xyz.append([float(x) for x in v[1:4]])
        qx, qy, qz, qw = (float(x) for x in v[4:8])
        q.append([qw, qx, qy, qz])
    return np.array(st, np.int64), np.asarray(xyz_quat_wxyz_to_se3_poses(np.array(xyz), np.array(q)), float)


def load(name):
    long = dataset_cfg("oxford_spires")["sequences"][name]
    seq_dir = os.path.join(data_dir("oxford_spires"), "sequences", long)
    prep = os.path.join(data_dir("oxford_spires", "prepared"), long, "raw")
    rows = [ln.split() for ln in open(os.path.join(prep, "rgb.txt")) if ln.strip() and not ln.startswith("#")]
    paths = [os.path.join(prep, r[1]) for r in rows]
    stamps = np.array([int(s.split(".")[0]) * 1_000_000_000 + int(s.split(".")[1]) for s, _ in rows], np.int64)
    fx, fy, cx, cy, W, H = np.loadtxt(os.path.join(prep, "cam0_intrinsics.txt"), comments="#")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    imu = np.loadtxt(os.path.join(seq_dir, "raw", "imu.csv"), delimiter=",", skiprows=1)
    imu_ns = imu[:, 0].astype(np.int64) * 1_000_000_000 + imu[:, 1].astype(np.int64)
    sensor = yaml.safe_load(open(SENSOR))
    cam = sensor["cam_front"]
    T_cam_imu = _T(cam["T_cam_imu_t_xyz_q_xyzw"])
    T_base_cam = _T(sensor["T_base_lidar_t_xyz_q_xyzw"]) @ np.linalg.inv(_T(cam["T_cam_lidar_t_xyz_q_xyzw"]))
    nz = yaml.safe_load(open(os.path.join(data_dir("oxford_spires"), "calibration", "imu.yaml")))
    noise = dict(acc_density=nz["accelerometer_noise_density"], gyro_density=nz["gyroscope_noise_density"],
                 acc_walk=nz["accelerometer_random_walk"], gyro_walk=nz["gyroscope_random_walk"])

    def gt():
        return read_tum_exact(os.path.join(seq_dir, "processed", "trajectory", "gt-tum.txt"))

    return Sequence(dataset="oxford_spires", name=name, image_paths=paths, stamps_ns=stamps, K=K,
                    size=(int(W), int(H)), imu_ns=imu_ns, acc=imu[:, 2:5], gyro=imu[:, 5:8],
                    T_cam_imu=T_cam_imu, timeshift_ns=0, noise=noise, gravity=9.81,
                    gt_frame="base", T_gt_cam=T_base_cam, gt=gt).check()
