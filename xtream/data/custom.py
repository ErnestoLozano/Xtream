"""Your own images, monocular: one folder per sequence, `<raw>/<seq>/`. Also a short example of a loader.

- Images: `images/*.{png,jpg,jpeg}`, in file-name order (e.g. `ffmpeg -i video.mp4 images/%06d.png`).
- Stamps: `times.txt`, one per image in that order [s] (first number of each line; `#` comments), or, without
  it, the frame rate `fps` of the `custom` entry in `configs/datasets.yaml`.
- Calibration: none. Monocular mode is uncalibrated: DA3 predicts the intrinsics; `K` below is a placeholder.
- Ground truth (optional, for `evaluate.py` only): `gt.txt`, TUM (`t x y z qx qy qz qw`, camera to world), on the
  clock of the stamps, matched by time.
"""

import os

import numpy as np
from PIL import Image

from xtream.data.base import Sequence
from xtream.paths import data_dir, dataset_cfg

IMAGE_EXT = (".png", ".jpg", ".jpeg")


def sequence_names():
    """Every sub-folder of `<raw>` holding an `images/` folder."""
    root = data_dir("custom")
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, "images"))) \
        if os.path.isdir(root) else []


def load(name):
    seq_dir = os.path.join(data_dir("custom"), name)
    img_dir = os.path.join(seq_dir, "images")
    assert os.path.isdir(img_dir), f"no {img_dir}: put the sequence's images there (see xtream/data/custom.py)"
    paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if f.lower().endswith(IMAGE_EXT)]
    assert paths, f"no images in {img_dir}"
    times = os.path.join(seq_dir, "times.txt")
    if os.path.exists(times):
        t = [float(ln.split()[0].rstrip(",")) for ln in open(times) if ln.strip() and not ln.startswith("#")]
        assert len(t) == len(paths), f"{times}: {len(t)} stamps for {len(paths)} images"
        stamps = np.round(np.asarray(t) * 1e9).astype(np.int64)
    else:
        stamps = np.round(np.arange(len(paths)) * 1e9 / float(dataset_cfg("custom")["fps"])).astype(np.int64)
    with Image.open(paths[0]) as im:
        size = im.size
    W, H = size
    K = np.array([[max(W, H), 0, W / 2], [0, max(W, H), H / 2], [0, 0, 1.0]])   # placeholder, not used
    gt_file = os.path.join(seq_dir, "gt.txt")

    def gt():
        """The camera poses of `gt.txt` (TUM)."""
        assert os.path.exists(gt_file), f"no {gt_file}: evaluate.py needs the ground truth (TUM, camera to world)"
        from evo.tools.file_interface import read_tum_trajectory_file
        traj = read_tum_trajectory_file(gt_file)
        return np.round(traj.timestamps * 1e9).astype(np.int64), np.asarray(traj.poses_se3, float)

    return Sequence(dataset="custom", name=name, image_paths=paths, stamps_ns=stamps, K=K, size=size,
                    imu_ns=np.zeros(0, np.int64), acc=np.zeros((0, 3)), gyro=np.zeros((0, 3)), T_cam_imu=np.eye(4),
                    timeshift_ns=0, noise={}, gravity=9.81, gt_frame="cam", T_gt_cam=np.eye(4), gt=gt,
                    dup=tuple(int(i) + 1 for i in np.flatnonzero(np.diff(stamps) == 0))).check()
