"""The conditioned map: DA3 passes over the map frames, conditioned on the back end's poses, with one scale per pass.

Passes of `pass_frames` map frames every `pass_stride` (the last one ending at the last map frame). Pass p reads
the live poses the back end held when its last frame had arrived (the snapshot of the first window ending after
it), in centimetres (DA3 clamps the median pose distance at 0.1 input units), and the intrinsics of the images
(mono-inertial: the calibration; monocular: DA3's own prediction from the window passes). DA3 aligns its prediction
to the input poses (Umeyama) and returns depth in the poses' unit.

Per-pass log scale x_p (`solve_pass_scales`): a unary 0 with sigma_u^2 = s0^2 + delta^2 / r_p^2 (r_p the RMS camera
spread of the pass), and links between passes sharing frames (the median log depth ratio on their shared pixels
within `link_range_m`, pooled sigma_l); (s0, delta) by a Student-t (nu 4) ML fit on the links; Huber IRLS on every term.
Live-pass rescale: k_p, the Umeyama scale from the pass's conditioning camera centres to the final ones, enters
before the solve (depth_p x k_p: links shift by log k_p - log k_q). Each map frame comes from the pass where it is
most central, its depth x k_p e^{x_p}; export as the base map (`base_map`).
"""

import os

import numpy as np
from scipy.optimize import minimize

from xtream.io import read_tum, stamp_name
from xtream.reconstruction.base_map import backproject, crop_aspect, frame_colours, write_cloud

UNIT, NU, HK = 100.0, 4.0, 1.345


def passes(n, L, stride):
    """(start, end) indices into the map frames of each pass."""
    starts = list(range(0, max(n - L, 0) + 1, stride))
    if starts[-1] + L < n:
        starts.append(n - L)
    return [(s, min(s + L, n)) for s in starts]


def umeyama_scale(x, y):
    """Scale s of y ~ s R x + t (Umeyama), x, y (N, 3)."""
    X_, Y_ = x - x.mean(0), y - y.mean(0)
    U, Sv, Vt = np.linalg.svd(Y_.T @ X_)
    d = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return (Sv * np.diag(d)).sum() / (X_ ** 2).sum()


def solve_pass_scales(link, pairs, sl, r, P):
    """Per-pass log scales x (P,) from the links (pairs (q - p)), the pooled link sigma and the camera spreads."""
    pa, pb = pairs[:, 0], pairs[:, 1]
    g2 = 1 / r[pa] ** 2 + 1 / r[pb] ** 2

    def nll(th):
        s2 = 2 * np.exp(2 * th[0]) + np.exp(2 * th[1]) * g2 + sl ** 2
        return np.sum(0.5 * np.log(s2) + (NU + 1) / 2 * np.log1p(link ** 2 / (NU * s2)))
    s0, dl = np.exp(minimize(nll, np.log([0.01, 0.01]), method="Nelder-Mead",
                             options=dict(xatol=1e-6, fatol=1e-9)).x)
    su = np.sqrt(s0 ** 2 + dl ** 2 / r ** 2)
    nL = len(link)
    Dl = np.zeros((nL, P))
    Dl[np.arange(nL), pa] = -1
    Dl[np.arange(nL), pb] = 1
    A = np.vstack([np.eye(P), Dl])
    y = np.concatenate([np.zeros(P), link])
    s = np.concatenate([su, np.full(nL, sl)])
    h = np.ones(len(y))
    for _ in range(100):
        w = h / s ** 2
        x = np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * y))
        hn = np.minimum(1, HK / np.maximum(np.abs(y - A @ x) / s, 1e-12))
        if np.max(np.abs(hn - h)) < 1e-6:
            break
        h = hn
    return x


def _frame_intrinsics(run, cfg, seq, frames):
    """K (image pixels) per map frame: the calibration (mono-inertial) or DA3's own prediction from the first
    window pass holding the frame, its processed-size K brought back to the image (monocular)."""
    import json
    import pickle
    rec = json.load(open(os.path.join(run, "run.json")))["config"]
    if cfg["calibrated"]:
        K = np.asarray(seq.K, np.float32)
        if any(seq.distortion):
            from xtream.data.undistort import undistorter
            K = np.asarray(undistorter(seq)[0], np.float32)
        return {f: K for f in frames}
    inp = pickle.load(open(os.path.join(run, "backend_inputs.pkl"), "rb"))
    Wimg, Himg = seq.size
    if rec.get("cropped"):
        Wimg = int(round(Himg * rec["canon_aspect"]))
    Kf = {}
    for j, (s, e) in enumerate(inp["windows"]):
        Kj = np.asarray(inp["K_win"][j], np.float64)
        for f in range(s, e):
            if f not in Kf:          # inverse of DA3's upper_bound_resize (x, y scaled apart); pp = processed centre
                K = Kj[f - s].copy()
                K[0] *= Wimg / (2 * K[0, 2])
                K[1] *= Himg / (2 * K[1, 2])
                Kf[f] = K.astype(np.float32)
    return Kf


def export_conditioned(run, cfg, seq, out=None):
    """Write the conditioned map of run directory `run` (sequence `seq`, the images the run read) to
    `<run>/map_cond/` (or `out`). Returns (number of clouds, number of passes)."""
    import json

    import torch

    from xtream.frontend.da3 import crop_to_aspect, load_da3
    from xtream.pipeline import map_grid

    m = cfg["map"]
    out = out or os.path.join(run, "map_cond")
    os.makedirs(out, exist_ok=True)
    rec = json.load(open(os.path.join(run, "run.json")))["config"]
    aspect = crop_aspect(run)
    st, T_fin = read_tum(os.path.join(run, "est.txt"))
    assert np.array_equal(st, np.asarray(seq.stamps_ns, np.int64))
    fr = map_grid(seq.stamps_ns, m["hz"])
    ps = passes(len(fr), m["pass_frames"], m["pass_stride"])
    pass_frames = [fr[a:b] for a, b in ps]
    Z = np.load(os.path.join(run, "map_inputs", "live_poses.npz"))
    lo_k, win_e, P_k = Z["lo"], Z["win"][:, 1], Z["P"]
    Kf = _frame_intrinsics(run, cfg, seq, sorted({f for F in pass_frames for f in F}))

    model = load_da3(cfg["model"])            # puts DA3 on sys.path
    import depth_anything_3.api as da3api
    from depth_anything_3.utils.pose_align import _poses_from_ext
    if rec.get("cropped"):
        crop_to_aspect([model], rec["canon_aspect"])
    rec_b = []
    _orig = da3api.align_poses_umeyama

    def _record(ext_ref, ext_est, **kw):     # DA3's alignment: keep the input camera centres (cm)
        res = _orig(ext_ref, ext_est, **kw)
        rec_b.append(_poses_from_ext(np.asarray(ext_ref, np.float64), np.asarray(ext_est, np.float64))[1][:, :3, 3])
        return res
    da3api.align_poses_umeyama = _record

    own = {}                                   # map frame index -> (distance to its pass centre, pass)
    for p, (a, b) in enumerate(ps):
        ctr = np.mean(np.arange(a, b))
        for i in range(a, b):
            if i not in own or abs(i - ctr) < own[i][0]:
                own[i] = (abs(i - ctr), p)
    px, R, cp = m["pix"], m["link_range_m"], m["conf_pct"]
    R_out = m["range_m"] or np.inf
    keep_full = {}                             # (pass, frame) -> (depth, conf, pct): shared frames for the links
    links, pairs, lvar, owned, b_live, b_fin = [], [], [], {}, [], []
    for p, F in enumerate(pass_frames):
        k = int(np.argmax(win_e > max(F)))
        assert win_e[k] > max(F) and min(F) >= lo_k[k], (k, min(F), max(F))
        w2c = np.stack([np.linalg.inv(P_k[k, f - lo_k[k]]) for f in F]).astype(np.float32)
        w2c[:, :3, 3] *= UNIT
        Ks = np.stack([Kf[f] for f in F]).astype(np.float32)
        with torch.no_grad():
            pr = model.inference([seq.image_paths[f] for f in F], extrinsics=w2c, intrinsics=Ks)
        assert len(rec_b) == p + 1
        b_live.append(rec_b[p])
        w2f = np.stack([np.linalg.inv(T_fin[f]) for f in F]).astype(np.float32)
        w2f[:, :3, 3] *= UNIT
        w2f = np.asarray(w2f, np.float64)
        b_fin.append(_poses_from_ext(w2f, w2f)[1][:, :3, 3])
        depth, conf = np.asarray(pr.depth / UNIT, np.float32), np.asarray(pr.conf, np.float32)
        pct = np.percentile(conf, np.arange(101))
        cur = {}
        for i, f in enumerate(F):
            cur[f] = (depth[i].astype(np.float64), conf[i], pct)
            if own[fr.index(f)][1] == p:
                owned[f] = (p, depth[i], conf[i][::px, ::px] >= pct[cp], np.asarray(pr.intrinsics[i], np.float64),
                            frame_colours(seq.image_paths[f], depth.shape[1:], px, aspect))
        # links to the earlier passes sharing frames
        for q in range(p - 1, -1, -1):
            sh = [f for f in pass_frames[q] if f in cur]
            if not sh:
                if q < p - 1:
                    break
                continue
            v = []
            for f in sh:
                d0, c0, q0 = keep_full[(q, f)]
                d1, c1, q1 = cur[f]
                ok = (c0 >= q0[cp]) & (d0 > 0) & (d0 <= R) & (c1 >= q1[cp]) & (d1 > 0) & (d1 <= R)
                if ok.sum():
                    v.append(np.log(d0[ok]) - np.log(d1[ok]))
            if v:
                pairs.append((q, p))
                links.append(np.median(np.concatenate(v)))
                cj = np.array([np.median(x) for x in v])
                lvar.append(cj.var(ddof=1) / len(cj) if len(cj) > 1 else np.nan)
        for f in cur:
            keep_full[(p, f)] = cur[f]
        for key in [key for key in keep_full if key[0] < p - 2]:
            del keep_full[key]
    da3api.align_poses_umeyama = _orig
    order = np.lexsort((np.array(pairs)[:, 1], np.array(pairs)[:, 0]))
    pairs, links, lvar = np.array(pairs)[order], np.array(links)[order], np.array(lvar)[order]
    sl = np.sqrt(np.nanmean(lvar))
    r = np.array([np.sqrt(((b / UNIT - (b / UNIT).mean(0)) ** 2).sum(1).mean()) for b in b_live])
    lk = np.log([umeyama_scale(bl, bf) for bl, bf in zip(b_live, b_fin)])
    x = solve_pass_scales(links + lk[pairs[:, 0]] - lk[pairs[:, 1]], pairs, sl, r, len(ps))
    n = 0
    for f, (p, dep, keep, K, rgb) in sorted(owned.items()):
        d = np.asarray(dep, np.float64)[::px, ::px] * np.exp(lk[p] + x[p])
        X = backproject(d, K, px)
        ok = keep.reshape(-1) & (np.linalg.norm(X, axis=1) <= R_out) & (X[:, 2] > 0)
        if ok.any():
            write_cloud(os.path.join(out, stamp_name(st[f]) + ".pcd"), X[ok], rgb[ok])
            n += 1
    np.savez(os.path.join(out, "..", "map_cond_scales.npz"), x=x, k=np.exp(lk), pairs=pairs, links=links, r=r)
    return n, len(ps)
