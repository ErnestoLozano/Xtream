"""Top view of a run's trajectory against the ground truth, after the alignment its ATE uses (SE(3) mono-inertial,
Sim(3) monocular), on the ground truth's two main axes."""

import copy
import json
import os

import numpy as np

from xtream.data import load
from xtream.eval.trajectory import associate
from xtream.io import read_tum


def plot_topview(run, out):
    """Write the top-view plot of run directory `run` to `out` (PNG). Returns the ATE [m] in the plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from evo.core import metrics
    rec = json.load(open(os.path.join(run, "run.json")))
    seq = load(rec["dataset"], rec["sequence"])
    sim = rec["mode"] == "mono"
    ref, est = associate(seq, *read_tum(os.path.join(run, "est.txt")))
    est = copy.deepcopy(est)
    est.align(ref, correct_scale=sim)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((ref, est))
    ate = float(ape.get_statistic(metrics.StatisticsType.rmse))
    g, e = ref.positions_xyz, est.positions_xyz
    # the plane of the ground truth's two largest extents (its principal axes)
    _, _, Vt = np.linalg.svd(g - g.mean(0), full_matrices=False)
    G, E = (g - g.mean(0)) @ Vt[:2].T, (e - g.mean(0)) @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(G[:, 0], G[:, 1], color="0.35", lw=1.5, ls="--", label="ground truth")
    ax.plot(E[:, 0], E[:, 1], color="tab:blue", lw=1.5, label=f"Xtream ({rec['mode']})")
    ax.plot(*G[0], "o", color="0.35", ms=5)
    # equal axes, the shorter one at least a quarter of the longer (straight trajectories stay readable)
    P = np.concatenate([G, E])
    lo, hi = P.min(0), P.max(0)
    half = np.maximum(hi - lo, 0.25 * (hi - lo).max()) * 0.55
    ax.set_xlim((lo + hi)[0] / 2 - half[0], (lo + hi)[0] / 2 + half[0])
    ax.set_ylim((lo + hi)[1] / 2 - half[1], (lo + hi)[1] / 2 + half[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("[m]")
    ax.set_ylabel("[m]")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.set_title(f"{rec['dataset']}/{rec['sequence']}: ATE {'Sim(3)' if sim else 'SE(3)'} {ate:.3f} m")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return ate
