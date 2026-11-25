"""Export a finished run's map: one camera-frame cloud per map frame, optionally fused into one world-frame PLY."""

import dataclasses
import os


def export_map(run, cfg, seq, kind, ply=False, voxel=0.02):
    """Write the `kind` ("base" or "cond") map of run directory `run` to `<run>/map_<kind>/`, and with `ply` the
    fused cloud `<run>/map_<kind>.ply` (voxel `voxel` [m]). `seq` is the sequence as loaded (its images are swapped
    for the undistorted ones the run read when the mode is calibrated and the camera distorted)."""
    if any(seq.distortion) and cfg["calibrated"]:
        from xtream.data.undistort import undistorted
        from xtream.paths import data_dir
        seq = dataclasses.replace(seq, image_paths=undistorted(
            seq, os.path.join(data_dir(seq.dataset, "undistorted"), seq.name)))
    tag = f"{seq.dataset}/{seq.name} {cfg['mode']} {kind} map"
    if kind == "base":
        from xtream.reconstruction.base_map import export_base
        print(f"{tag}: {export_base(run, cfg, seq)} clouds -> {run}/map_base", flush=True)
    else:
        from xtream.reconstruction.conditioned_map import export_conditioned
        n, P = export_conditioned(run, cfg, seq)
        print(f"{tag}: {P} passes, {n} clouds -> {run}/map_cond", flush=True)
    if ply:
        from xtream.reconstruction.fuse import fuse_ply
        out = os.path.join(run, f"map_{kind}.ply")
        print(f"{tag}: {fuse_ply(run, os.path.join(run, f'map_{kind}'), out, voxel)} points -> {out}", flush=True)
