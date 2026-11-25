"""One world-frame point cloud of a run's map: every map frame's camera-frame cloud placed by the run's `est.txt`
pose, merged and voxel-downsampled, written as a coloured PLY (MeshLab, CloudCompare, Open3D)."""

import glob
import os

import numpy as np

from xtream.io import read_tum


def fuse_ply(run, clouds, out, voxel=0.02):
    """Merge `clouds` (`<sec>_<nsec>.pcd`, camera frame) at the run's poses into `out` (PLY), voxel `voxel` [m]
    (0: none). Returns the number of points written."""
    import open3d as o3d
    t_e, T_e = read_tum(os.path.join(run, "est.txt"))
    ix = {int(t): i for i, t in enumerate(t_e)}
    pts, cols = [], []
    for p in sorted(glob.glob(os.path.join(clouds, "*.pcd"))):
        s, ns = os.path.basename(p)[:-4].split("_")
        pc = o3d.io.read_point_cloud(p).transform(T_e[ix[int(s) * 10 ** 9 + int(ns)]])
        pts.append(np.asarray(pc.points))
        cols.append(np.asarray(pc.colors) if pc.has_colors() else np.full((len(pc.points), 3), 0.5))
    assert pts, f"no clouds in {clouds}"
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts)))
    pc.colors = o3d.utility.Vector3dVector(np.concatenate(cols))
    if voxel > 0:
        pc = pc.voxel_down_sample(voxel)
    assert o3d.io.write_point_cloud(out, pc)
    return len(pc.points)
