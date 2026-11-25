"""Trajectory scores of finished runs: ATE after SE(3) and Sim(3) alignment, RPE over 1 m, scale, FPS.

    python scripts/evaluate.py --mode mono-inertial --dataset euroc --seq all
    python scripts/evaluate.py --mode mono --dataset kitti --seq 00,05

Prints one row per sequence and the mean; adds them to results/<dataset>/<mode>/scores.json.
--plot also draws each run's top view against the ground truth (after the ATE's alignment) -> <run>/trajectory.png.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xtream.config import MODES  # noqa: E402
from xtream.data import DATASETS, sequences  # noqa: E402
from xtream.eval.trajectory import score  # noqa: E402
from xtream.io import run_dir  # noqa: E402

COLS = [("ate_se3", "ATE SE(3) [m]"), ("ate_sim3", "ATE Sim(3) [m]"), ("rpe", "RPE [m]"), ("scale", "Scale"),
        ("fps", "FPS")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=list(MODES))
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--seq", default="all", help="sequence name(s), comma-separated, or 'all' (the finished ones)")
    ap.add_argument("--results", default=None)
    ap.add_argument("--plot", action="store_true", help="also write <run>/trajectory.png (top view vs ground truth)")
    a = ap.parse_args()
    names = sequences(a.dataset) if a.seq == "all" else a.seq.split(",")
    rows = {}
    for sq in names:
        run = run_dir(a.dataset, a.mode, sq, a.results)
        if not os.path.exists(os.path.join(run, "run.json")):
            assert a.seq == "all", f"no run in {run}"
            continue
        rows[sq] = score(run)
        if a.plot:
            from xtream.eval.plot import plot_topview
            plot_topview(run, os.path.join(run, "trajectory.png"))
    assert rows, "no finished runs"
    print(f"| {a.dataset} ({a.mode}) | " + " | ".join(h for _, h in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for sq, r in rows.items():
        print(f"| {sq} | " + " | ".join("--" if r[k] is None else f"{r[k]:.3f}" if k != "fps" else f"{r[k]:.1f}"
                                        for k, _ in COLS) + " |")
    mean = {k: sum(r[k] for r in rows.values() if r[k] is not None) / max(1, sum(r[k] is not None for r in rows.values()))
            for k, _ in COLS}
    print("| **Mean** | " + " | ".join(f"{mean[k]:.3f}" if k != "fps" else f"{mean[k]:.1f}" for k, _ in COLS) + " |")
    out = os.path.join(os.path.dirname(run_dir(a.dataset, a.mode, next(iter(rows)), a.results)), "scores.json")
    old = json.load(open(out)) if os.path.exists(out) else {}
    json.dump(dict(old, **rows), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
