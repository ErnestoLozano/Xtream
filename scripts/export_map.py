"""Export the map of finished runs (run.py --map does the same right after each run).

    python scripts/export_map.py --mode mono-inertial --dataset oxford_spires --seq obs02 --map base --ply
    python scripts/export_map.py --mode mono --dataset oxford_spires --seq all --map cond

  base: depth from the front end's own window passes (no extra network pass) -> <run>/map_base/
  cond: extra DA3 passes conditioned on the back end's live poses, one scale per pass -> <run>/map_cond/
  --ply: also one world-frame, coloured, voxel-downsampled cloud -> <run>/map_<base|cond>.ply
  points within 5 m of their camera; --range_m sets another distance, --raw keeps them all
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from xtream.config import MODES, load_config  # noqa: E402
from xtream.data import DATASETS, load, sequences  # noqa: E402
from xtream.io import run_dir  # noqa: E402


def add_map_args(ap, required):
    """The map options shared by export_map.py and run.py."""
    ap.add_argument("--map", required=required, default=None, choices=["base", "cond"],
                    help="base: the front end's window depth; cond: extra DA3 passes on the live poses")
    ap.add_argument("--ply", action="store_true", help="also write the fused world-frame cloud <run>/map_<map>.ply")
    ap.add_argument("--voxel", type=float, default=0.02, help="voxel of the fused cloud [m] (0: every point)")
    ap.add_argument("--range_m", type=float, default=None,
                    help="keep only points up to this distance from their camera [m] (default: the config's, 5)")
    ap.add_argument("--raw", action="store_true", help="keep every confident point, whatever its distance")


def map_config(cfg, a):
    """The configuration with the export range of --range_m / --raw."""
    assert not (a.raw and a.range_m), "--raw and --range_m exclude each other"
    if a.raw or a.range_m:
        cfg = dict(cfg, map=dict(cfg["map"], range_m=None if a.raw else a.range_m))
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=list(MODES))
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--seq", required=True, help="sequence name(s), comma-separated, or 'all'")
    add_map_args(ap, required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--results", default=None)
    a = ap.parse_args()
    cfg = map_config(load_config(a.config or a.mode), a)
    from xtream.reconstruction.export import export_map
    for sq in sequences(a.dataset) if a.seq == "all" else a.seq.split(","):
        run = run_dir(a.dataset, a.mode, sq, a.results)
        assert os.path.exists(os.path.join(run, "est.txt")), f"no run in {run}: scripts/run.py first"
        export_map(run, cfg, load(a.dataset, sq), a.map, a.ply, a.voxel)


if __name__ == "__main__":
    main()
