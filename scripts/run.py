"""Run Xtream on one sequence, a list of sequences, or a whole dataset.

    python scripts/run.py --mode mono-inertial --dataset euroc --seq V1_01_easy
    python scripts/run.py --mode mono --dataset kitti --seq 00,05
    python scripts/run.py --mode mono --dataset oxford_spires --seq all --map base --ply

Outputs go to results/<dataset>/<mode>/<sequence>/ (est.txt, run.json, backend_inputs.pkl, map_inputs/).
--map also exports the map after each run (see scripts/export_map.py, which re-exports finished runs).
"""

import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from export_map import add_map_args, map_config  # noqa: E402
from xtream.config import MODES, load_config  # noqa: E402
from xtream.data import DATASETS, load, sequences  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=list(MODES), help="monocular or mono-inertial")
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--seq", required=True, help="sequence name(s), comma-separated, or 'all'")
    ap.add_argument("--config", default=None, help="a YAML file instead of the mode's default configuration")
    ap.add_argument("--results", default=None, help="results root (default: results/ or $XTREAM_RESULTS)")
    ap.add_argument("--skip_done", action="store_true", help="skip sequences whose run.json exists")
    add_map_args(ap, required=False)
    a = ap.parse_args()
    assert not (a.mode == "mono-inertial" and a.dataset == "kitti"), "KITTI odometry has no IMU: use --mode mono"
    assert not (a.mode == "mono-inertial" and a.dataset == "custom"), \
        ("custom reads images only (uncalibrated monocular: use --mode mono). Mono-inertial needs the IMU and its "
         "calibration (camera-IMU transform and time offset, noise): write a loader for them, see the README's "
         "'Your own data'")
    assert a.map or not (a.ply or a.range_m or a.raw), "--ply, --range_m and --raw need --map"
    cfg = load_config(a.config or a.mode)
    assert cfg["mode"] == a.mode, (cfg["mode"], a.mode)
    mcfg = map_config(cfg, a)
    import torch

    from xtream.io import run_dir
    from xtream.pipeline import run
    from xtream.reconstruction.export import export_map
    for sq in sequences(a.dataset) if a.seq == "all" else a.seq.split(","):
        d = run_dir(a.dataset, a.mode, sq, a.results)
        if a.skip_done and os.path.exists(os.path.join(d, "run.json")):
            print(f"{a.dataset}/{sq}: done, skipped")
        else:
            d = run(load(a.dataset, sq), cfg, a.results)
        if a.map:
            gc.collect()                       # the run's models leave the GPU before the map's passes
            torch.cuda.empty_cache()
            export_map(d, mcfg, load(a.dataset, sq), a.map, a.ply, a.voxel)


if __name__ == "__main__":
    main()
