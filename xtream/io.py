"""Run outputs: `results/<dataset>/<mode>/<sequence>/` with `est.txt` (TUM, camera poses on the camera clock),
`run.json` (configuration, timings), `backend_inputs.pkl` (the back end's inputs, for replays) and `map_inputs/`
(what the map export reads)."""

import json
import os

import numpy as np

from xtream.paths import RESULTS_ROOT


def run_dir(dataset, mode, seq, root=None):
    d = os.path.join(root or RESULTS_ROOT, dataset, mode, seq)
    os.makedirs(d, exist_ok=True)
    return d


def stamp_name(t):
    """`<sec>_<nsec>` of a stamp [ns], the name of a map frame's cloud."""
    return f"{int(t) // 10 ** 9}_{int(t) % 10 ** 9:09d}"


def save_tum(path, stamps_ns, T):
    """Camera-clock stamps [ns] and 4x4 camera-to-world poses -> TUM (t x y z qx qy qz qw)."""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(np.asarray(T)[:, :3, :3]).as_quat()
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for t, P, qq in zip(stamps_ns, T, q):
            f.write(f"{int(t) // 1_000_000_000}.{int(t) % 1_000_000_000:09d} "
                    + " ".join(f"{v:.9f}" for v in (*P[:3, 3], *qq)) + "\n")


def read_tum(path):
    """TUM -> (stamps [ns] int64, poses (N, 4, 4)), stamps parsed exactly."""
    from scipy.spatial.transform import Rotation
    rows = [ln.split() for ln in open(path) if ln.strip() and not ln.startswith("#")]
    st = np.array([int(r[0].split(".")[0]) * 1_000_000_000 + int((r[0].split(".") + ["0"])[1].ljust(9, "0")[:9])
                   for r in rows], np.int64)
    v = np.array([[float(x) for x in r[1:8]] for r in rows])
    T = np.tile(np.eye(4), (len(rows), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(v[:, 3:7]).as_matrix()
    T[:, :3, 3] = v[:, :3]
    return st, T


def write_run(d, seq, mode, wall_s, config, extra=None):
    n = len(seq.image_paths)
    rec = {"method": "xtream", "mode": mode, "dataset": seq.dataset, "sequence": seq.name,
           "frames_hash": seq.frame_hash(), "n_frames": n, "t0_ns": int(seq.stamps_ns[0]), "wall_s": float(wall_s),
           "fps": n / float(wall_s) if wall_s else None, "est_frame": "cam", "gt_frame": seq.gt_frame,
           "config": config, **(extra or {})}
    json.dump(rec, open(os.path.join(d, "run.json"), "w"), indent=1)
    return rec
