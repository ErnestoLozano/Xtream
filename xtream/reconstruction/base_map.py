"""The base map: each map frame's depth from the first window pass holding it, in metres by that window's final
log metre, back-projected with DA3's intrinsics of the frame; points above the window's confidence percentile,
in front of the camera and within `range_m` of it (all when null). One camera-frame cloud per map frame
(`<sec>_<nsec>.pcd`, with the image's colours), placed in the world by the run's `est.txt`."""

import glob
import json
import os

import numpy as np

from xtream.io import read_tum, stamp_name


def backproject(d, K, pix):
    """Camera-frame points (OpenCV axes) of a depth map sampled every `pix`-th pixel of the full image (d already
    sampled), with K of the full image."""
    v, u = np.mgrid[0:d.shape[0] * pix:pix, 0:d.shape[1] * pix:pix]
    return np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1).reshape(-1, 3)


def crop_aspect(run):
    """The aspect the run centre-cropped its images to (wide frames, e.g. KITTI), or None."""
    cfg = json.load(open(os.path.join(run, "run.json")))["config"]
    return cfg["canon_aspect"] if cfg.get("cropped") else None


def frame_colours(path, hw, pix, aspect=None):
    """RGB in [0, 1] of the image at `path` as the network read it (centre-cropped to `aspect` when wider, resized
    to `hw` = (H, W)), every `pix`-th pixel, flattened as `backproject`'s points."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if aspect and w / h > aspect + 1e-6:
        c = int(round(h * aspect))
        im = im.crop(((w - c) // 2, 0, (w - c) // 2 + c, h))
    rgb = np.asarray(im.resize((int(hw[1]), int(hw[0])), Image.BILINEAR), np.float64) / 255.0
    return rgb[::pix, ::pix].reshape(-1, 3)


def write_cloud(path, X, rgb):
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X))
    pc.colors = o3d.utility.Vector3dVector(rgb)
    assert o3d.io.write_point_cloud(path, pc)


def export_base(run, cfg, seq, out=None):
    """Write the base map of run directory `run` (sequence `seq`, the images the run read) to `<run>/map_base/`
    (or `out`). Returns the number of clouds."""
    m = cfg["map"]
    out = out or os.path.join(run, "map_base")
    os.makedirs(out, exist_ok=True)
    st, _ = read_tum(os.path.join(run, "est.txt"))
    l = np.load(os.path.join(run, "map_inputs", "log_metres.npy"))
    aspect = crop_aspect(run)
    n = 0
    for p in sorted(glob.glob(os.path.join(run, "map_inputs", "w*.npz"))):
        k = int(os.path.basename(p)[1:-4])
        z = np.load(p)
        for f, dep, keep, K in zip(z["frames"], z["depth"], z["keep"], z["K"]):
            d = np.asarray(dep, np.float64) * np.exp(l[k])
            X = backproject(d, K, m["pix"])
            ok = keep.reshape(-1) & (np.linalg.norm(X, axis=1) <= (m["range_m"] or np.inf)) & (X[:, 2] > 0)
            if ok.any():
                hw = (int(round(2 * K[1, 2])), int(round(2 * K[0, 2])))     # DA3's principal point: the centre
                assert -(-hw[0] // m["pix"]) == d.shape[0] and -(-hw[1] // m["pix"]) == d.shape[1], (hw, d.shape)
                rgb = frame_colours(seq.image_paths[f], hw, m["pix"], aspect)
                write_cloud(os.path.join(out, stamp_name(st[f]) + ".pcd"), X[ok], rgb[ok])
                n += 1
    return n
