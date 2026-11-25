# `match_by_index` is adapted from CLoSeR (https://github.com/MoyangLi00/CLoSeR, scripts/_traj_plot_common.py);
# the repository states no license.

"""Trajectory scores at the camera.

- Frames: the dataset's ground truth is moved to the camera with its calibration (`T_W_cam = T_W_gt @ T_gt_cam`).
- Association: evo's `sync.associate_trajectories`, `max_diff` 0.01 s (the Oxford Spires localisation benchmark's
  setting); on datasets with one ground-truth pose per frame (`gt_match` "index": KITTI, VBR), by frame index.
- ATE: translation RMSE after evo's Umeyama alignment, SE(3) and Sim(3).
- RPE: translation RMSE over 1 m segments (evo, `delta_unit` metres).
- Scale: estimate / ground-truth path length over the associated poses.
"""

import copy
import json
import os

import numpy as np

from xtream.data import load
from xtream.io import read_tum

MAX_DIFF_S = 0.01
RPE_DELTA_M = 1.0


def _traj(stamps_ns, T):
    from evo.core.trajectory import PoseTrajectory3D
    return PoseTrajectory3D(poses_se3=list(T), timestamps=np.asarray(stamps_ns, np.float64) * 1e-9)


def match_by_index(pred_ts, gt_ts):
    """Pairs (estimate row, ground-truth row) whose integer frame indices agree (CLoSeR's
    `scripts/_traj_plot_common.match_by_index`)."""
    gt_pos = {int(v): i for i, v in enumerate(np.rint(gt_ts))}
    pairs = [(i, gt_pos[int(v)]) for i, v in enumerate(np.rint(pred_ts)) if int(v) in gt_pos]
    return np.array(pairs, dtype=int).reshape(-1, 2)


def _by_index(seq, t_e, T_e, T_g):
    """Estimate rows paired with the per-frame ground truth by frame index. A row's frame is the one with its
    stamp; rows repeating a stamp (`dup` frames) take the next frame."""
    f = np.searchsorted(seq.stamps_ns, t_e, side="left")
    for i in range(1, len(f)):
        if t_e[i] == t_e[i - 1]:
            f[i] = f[i - 1] + 1
    assert np.array_equal(seq.stamps_ns[f], t_e), "estimate stamps not on the frames"
    pairs = match_by_index(f.astype(float), np.arange(len(T_g), dtype=float))
    g = pairs[:, 1]
    # a dup frame's row placed 1 us after its partner, so evo's stamps increase
    t = seq.stamps_ns[g] + 1000 * (g - np.searchsorted(seq.stamps_ns, seq.stamps_ns[g], side="left"))
    return _traj(t, T_g[g]), _traj(t, T_e[pairs[:, 0]])


def associate(seq, t_e, T_e):
    """(ground truth, estimate) evo trajectories of the associated camera poses."""
    from evo.core import sync
    t_g, T_g = seq.gt()
    if seq.gt_match == "index":
        return _by_index(seq, t_e, T_e, T_g @ seq.T_gt_cam)
    return sync.associate_trajectories(_traj(t_g, T_g @ seq.T_gt_cam), _traj(t_e, T_e), max_diff=MAX_DIFF_S)


def score(run, seq=None):
    """Scores of one run directory (`est.txt`, `run.json`)."""
    from evo.core import metrics
    rec = json.load(open(os.path.join(run, "run.json")))
    seq = seq or load(rec["dataset"], rec["sequence"])
    t_e, T_e = read_tum(os.path.join(run, "est.txt"))
    ref, est = associate(seq, t_e, T_e)
    out = {"n_assoc": int(est.num_poses), "n_est": int(len(t_e)), "n_frames": rec["n_frames"]}
    for name, sim in (("ate_se3", False), ("ate_sim3", True)):
        e = copy.deepcopy(est)
        e.align(ref, correct_scale=sim)
        ape = metrics.APE(metrics.PoseRelation.translation_part)
        ape.process_data((ref, e))
        out[name] = float(ape.get_statistic(metrics.StatisticsType.rmse))
    rpe = metrics.RPE(metrics.PoseRelation.translation_part, delta=RPE_DELTA_M,
                      delta_unit=metrics.Unit.meters, all_pairs=False)
    try:
        rpe.process_data((ref, est))
        out["rpe"] = float(rpe.get_statistic(metrics.StatisticsType.rmse))
    except Exception:            # fewer poses than one segment
        out["rpe"] = None
    path = lambda p: float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    out["path_gt"] = path(ref.positions_xyz)
    out["scale"] = path(est.positions_xyz) / out["path_gt"]
    out["fps"] = rec.get("fps")
    return out
