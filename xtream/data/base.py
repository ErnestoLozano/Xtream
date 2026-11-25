# `target_resolution` and `load_images` are adapted from CLoSeR (https://github.com/MoyangLi00/CLoSeR,
# demo_viser.py, `_infer_target_resolution` and `load_images_from_paths`); the repository states no license.

"""One sequence as its official sources give it, and the view the front end and back end read.

`Sequence` holds what the dataset's own files provide: the image files with their stamps, the raw IMU in its own
frame, the calibration and the ground truth in its frame. Nothing is selected by ground truth.

`FrontEndView` presents a `Sequence` to the system: stamps in seconds from the first image, the IMU on the camera
clock and in our camera axes (x forward, y right, z down), the IMU segment of each frame, and the images as the
init's GeoCalib pass reads them.
"""

import dataclasses
import hashlib
import math
import os
from typing import Callable

import numpy as np
import torch
from PIL import Image

# OpenCV camera axes -> ours (x forward, y right, z down)
M_CAM = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@dataclasses.dataclass
class Sequence:
    dataset: str
    name: str
    image_paths: list            # images, in stream order
    stamps_ns: np.ndarray        # image stamps [ns], int64, the dataset's clock
    K: np.ndarray                # pinhole of the images on disk (3x3)
    size: tuple                  # (W, H) of the images on disk
    imu_ns: np.ndarray           # IMU stamps [ns], int64, the IMU's clock
    acc: np.ndarray              # (N, 3) [m/s^2], IMU frame
    gyro: np.ndarray             # (N, 3) [rad/s], IMU frame
    T_cam_imu: np.ndarray        # 4x4, IMU -> camera (OpenCV axes)
    timeshift_ns: int            # t_imu = t_cam + timeshift (Kalibr convention)
    noise: dict                  # acc_density, gyro_density, acc_walk, gyro_walk
    gravity: float               # [m/s^2]
    gt_frame: str                # the ground truth's frame (body, base, cam0, ...)
    T_gt_cam: np.ndarray         # 4x4, camera -> ground-truth frame
    gt: Callable                 # () -> (stamps_ns int64, T_W_gtframe (N, 4, 4))
    distortion: tuple = (0.0, 0.0, 0.0, 0.0)   # radtan k1 k2 p1 p2 of the images on disk
    dup: tuple = ()              # frames sharing the previous frame's stamp (two images in one camera period)
    gt_match: str = "time"       # "time": ground truth by stamp; "index": one pose per frame, in order

    def frame_hash(self):
        h = hashlib.sha1()
        for p, t in zip(self.image_paths, self.stamps_ns):
            h.update(f"{os.path.basename(p)} {int(t)}\n".encode())
        return h.hexdigest()

    def check(self):
        n = len(self.image_paths)
        assert n > 0 and len(self.stamps_ns) == n, (n, len(self.stamps_ns))
        d = np.diff(self.stamps_ns)
        assert np.all(d >= 0) and set((np.flatnonzero(d == 0) + 1).tolist()) == set(self.dup), \
            "image stamps must increase, except at `dup`"
        assert np.all(np.diff(self.imu_ns) > 0), "IMU stamps must increase"
        assert self.acc.shape == self.gyro.shape == (len(self.imu_ns), 3)
        return self


def target_resolution(first_image_path, pixel_limit=255000):
    """(W, H): multiples of 14 with W * H <= `pixel_limit`, keeping the aspect (CLoSeR's
    `demo_viser._infer_target_resolution`)."""
    img = Image.open(first_image_path).convert("RGB")
    W_orig, H_orig = img.size
    scale = math.sqrt(pixel_limit / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    k, m = round(W_orig * scale / 14), round(H_orig * scale / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > W_orig / H_orig:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


def load_images(paths, W, H):
    """Images resized to (W, H) with Lanczos, as [N, 3, H, W] in [0, 1] (CLoSeR's
    `demo_viser.load_images_from_paths`)."""
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    return torch.stack([to_tensor(Image.open(p).convert("RGB").resize((W, H), Image.Resampling.LANCZOS))
                        for p in paths], dim=0)


class FrontEndView(torch.utils.data.Dataset):
    """A `Sequence` as the front end and the back end read it."""

    def __init__(self, seq: Sequence, pixel_limit=255000):
        self.seq = seq
        self.W, self.H = target_resolution(seq.image_paths[0], pixel_limit)
        sx, sy = self.W / seq.size[0], self.H / seq.size[1]
        K = np.array(seq.K, float)
        K[0] *= sx
        K[1] *= sy
        self.K = torch.from_numpy(K).float()
        # one clock: the camera's; IMU stamps moved onto it by the calibrated shift
        self.t0_ns = int(seq.stamps_ns[0])
        self.cam_ts_ns = np.asarray(seq.stamps_ns, np.int64)
        self.cam_times = (self.cam_ts_ns - self.t0_ns).astype(np.float64) * 1e-9
        self.imu_times = (seq.imu_ns - seq.timeshift_ns - self.t0_ns).astype(np.float64) * 1e-9
        # a sequence without IMU (KITTI odometry) is vision-only; every IMU access asserts `has_imu`
        self.has_imu = len(self.imu_times) > 0
        # the first frame the IMU reaches: inertial states start there
        self.imu_first = int(np.searchsorted(self.cam_times, self.imu_times[0])) if self.has_imu else 0
        assert not self.has_imu or self.cam_times[-1] <= self.imu_times[-1], "IMU ends before the last frame"
        R_imu_cam = np.linalg.inv(seq.T_cam_imu)[:3, :3] @ M_CAM.T
        self._acc, self._gyro = seq.acc @ R_imu_cam, seq.gyro @ R_imu_cam
        self._T_cam_imu = np.asarray(seq.T_cam_imu, float)
        nz = seq.noise if self.has_imu else dict.fromkeys(("acc_density", "gyro_density", "acc_walk", "gyro_walk"),
                                                          np.nan)
        self.IMU_ACC_NOISE_DENSITY = float(nz["acc_density"])
        self.IMU_GYRO_NOISE_DENSITY = float(nz["gyro_density"])
        self.IMU_ACC_RANDOM_WALK = float(nz["acc_walk"])
        self.IMU_GYRO_RANDOM_WALK = float(nz["gyro_walk"])
        self.GRAVITY_WORLD = np.array([0.0, 0.0, -float(seq.gravity)])

    def __len__(self):
        return len(self.cam_ts_ns)

    def __getitem__(self, i):
        return {"image": load_images([self.seq.image_paths[i]], self.W, self.H)[0], "K": self.K}

    def get_imu_segment(self, idx):
        """The IMU between the previous frame and this one; at `imu_first`, from the IMU's first sample. Inside a
        gap of the IMU stream, its nearest sample over the whole interval."""
        assert self.has_imu, f"{self.seq.dataset}/{self.seq.name} has no IMU"
        assert idx >= self.imu_first, (idx, self.imu_first)
        imu_times = self.imu_times
        t_end = float(self.cam_times[idx])
        if idx == self.imu_first:
            t_start = min(float(imu_times[0]), t_end)
            mask = imu_times <= t_end
        else:
            t_start = float(self.cam_times[idx - 1])
            mask = (imu_times > t_start) & (imu_times <= t_end)
        t = imu_times[mask]
        if len(t) == 0:
            j = int(np.clip(np.searchsorted(imu_times, (t_start + t_end) * 0.5), 0, len(imu_times) - 1))
            acc, gyro, dts = self._acc[[j]], self._gyro[[j]], np.array([t_end - t_start])
        else:
            dts = np.diff(t, prepend=t_start)
            dts[-1] += t_end - t[-1]
            acc, gyro = self._acc[mask], self._gyro[mask]
        return {"imu_acc": torch.from_numpy(acc).float(), "imu_gyro": torch.from_numpy(gyro).float(),
                "imu_dts": torch.from_numpy(dts).float()}
