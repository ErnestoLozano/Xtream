"""Xtream on one sequence: the streaming front end, the back end in its own thread, and the run's outputs.

Per window of `window` frames (consecutive windows share `overlap` frames):
  1. SALAD descriptors of its frames and their loop candidates (`frontend.loops`);
  2. one DA3 pass: its poses and depth; the depth ratio on the frame it shares with the previous window links the
     two windows' units (metres);
  3. monocular: the metric prior on its keyframes (`frontend.metric_prior`);
  4. covisibility edges to the windows 1-3 back with a SALAD keyframe match above `salad_sim`: one DA3 pass
     [end of j | keyframes i | keyframes j] each;
  5. loops (CLoSeR's streaming detection): one DA3 pass [end of j | frames of i | frames of j | end of j] each.
Every output goes to the back end (`backend.Backend`) as it is produced; the front end never reads it back.
Map inputs: each map frame (`map.hz` grid) keeps the depth of the first window holding it, for the base map, and
the back end keeps its live poses, for the conditioned map passes (`reconstruction`).
"""

import os
import pickle
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from xtream.backend.backend import Backend
from xtream.backend.sim3 import to_opencv
from xtream.data.base import FrontEndView
from xtream.frontend.da3 import Prefetch, camera_to_world, crop_to_aspect, load_da3
from xtream.frontend.loops import LoopCandidates, load_detector
from xtream.io import run_dir, save_tum, write_run
from xtream.paths import data_dir


def map_grid(stamps_ns, hz):
    """Map frames: every k-th frame, k from the median frame period, at about `hz`."""
    st = np.asarray(stamps_ns, np.int64)
    return list(range(0, len(st), max(1, int(round(1e9 / (hz * np.median(np.diff(st))))))))


def save_map_inputs(path, frames, s, pred, conf_pct, pix):
    """A window's map frames: their depth (every `pix`-th pixel, the window's unit), the mask of pixels above the
    window's `conf_pct` confidence percentile, and DA3's intrinsics of each."""
    conf = np.asarray(pred.conf)
    q = np.percentile(conf, conf_pct)
    np.savez(path, frames=np.array(frames),
             depth=np.stack([np.asarray(pred.depth[f - s], np.float32)[::pix, ::pix] for f in frames]),
             keep=np.stack([conf[f - s][::pix, ::pix] >= q for f in frames]),
             K=np.stack([np.asarray(pred.intrinsics[f - s], np.float64) for f in frames]))


def run(seq, cfg, out_root=None):
    """Run Xtream on `seq` (a `Sequence`) with `cfg` (configs/mono*.yaml). Returns the run directory."""
    import dataclasses

    import torch
    from PIL import Image

    mono = cfg["mode"] == "mono"
    WS, O, K = cfg["window"], cfg["overlap"], cfg["keyframes"]
    seq0 = seq
    if any(seq.distortion) and cfg["calibrated"]:
        from xtream.data.undistort import undistorted
        seq = dataclasses.replace(seq, image_paths=undistorted(
            seq, os.path.join(data_dir(seq.dataset, "undistorted"), seq.name)))
    d = run_dir(seq.dataset, cfg["mode"], seq.name, out_root)
    paths = list(seq.image_paths)
    with Image.open(paths[0]) as im:
        W_img, H_img = im.size
    bk = cfg["backend"]
    B = Backend(FrontEndView(seq), WS, O, lm_steps=bk["lm_steps"], imu_scale=bk["imu_scale"],
                reset_every=bk["reset_every"], keyframes=K, use_imu=not mono, loop_gate_dof=bk["loop_gate_dof"],
                lc_check_t=bk["lc_check_t"], lc_check_iters=bk["lc_check_iters"],
                snapshot_back=cfg["map"]["snapshot_back"])
    W = B.windows
    det = load_detector(paths, WS, cfg["loops"], cfg["loops"]["salad_sim"])
    da3 = load_da3(cfg["model"])
    metric = None
    if mono:
        from xtream.frontend.metric_prior import MetricPrior
        metric = MetricPrior(load_da3(cfg["metric_prior"]["model"]), cfg["metric_prior"])
    crop = bool(cfg["canon_aspect"] and W_img / H_img > cfg["crop_above"] + 1e-6)
    if crop:
        crop_to_aspect([da3] + ([metric.model] if metric else []), cfg["canon_aspect"])
    prefetch = Prefetch(da3, paths)
    cands = LoopCandidates(W, WS, O, cfg["loops"]["chord_arc"], cfg["loops"]["nms_windows"])
    sim = cfg["edges"]["salad_sim"]

    def keyframes(k):
        """The frames the graph keeps of window k's new frames (`ExplicitGraph._keep`)."""
        s, e = W[k]
        new = list(range(s if k == 0 else s + O, e))
        return sorted({new[i] for i in np.round(np.linspace(0, len(new) - 1, max(min(K, len(new)), 2))).astype(int)})

    def desc(frames):
        return det.descriptors[[det._frame_to_row[f] for f in frames]].numpy()

    # map inputs: the first window holding each map frame keeps its depth (every pix-th pixel) and confidence mask
    mcfg = cfg["map"]
    grid, owned = map_grid(seq.stamps_ns, mcfg["hz"]), set()
    mdir = os.path.join(d, "map_inputs")
    os.makedirs(mdir, exist_ok=True)
    writer, writes = ThreadPoolExecutor(1), []

    ops, err, l0 = queue.Queue(), [], {}

    def backend():
        try:
            while (op := ops.get()) is not None:
                B.op(op, l_init=l0["v"]) if mono else B.op(op)
        except Exception:
            import traceback
            err.append(traceback.format_exc())
            raise

    th = threading.Thread(target=backend, daemon=True)
    rec = {k: [] for k in ("t_win", "t_detect", "t_window", "t_edges", "t_loops", "t_metric")}
    K_win, prev_z, n_edges = {}, None, 0
    torch.cuda.reset_peak_memory_stats()
    th.start()
    t_start = last = time.perf_counter()
    for j, (s, e) in enumerate(W):
        fr = list(range(s, e))
        t0 = time.perf_counter()
        det.extract_window_descriptors(fr, [paths[f] for f in fr])
        pend = det.detect_window_loops(fr)
        t1 = time.perf_counter()
        if j + 1 < len(W):
            prefetch(range(*W[j + 1]))
        P, pred = camera_to_world(da3, paths, fr)
        z = np.asarray(pred.depth, np.float64)
        K_win[j] = np.asarray(pred.intrinsics, np.float32)      # DA3's own camera per frame, processed size
        ratio = None
        if prev_z is not None:
            assert prev_z.shape == z[0].shape, (prev_z.shape, z[0].shape)
            ok = np.isfinite(prev_z) & np.isfinite(z[0]) & (z[0] > 0)
            ratio = float(np.median(prev_z[ok] / z[0][ok]))
        prev_z = z[len(z) - O] if len(z) > O else None
        if metric is not None:
            tm = time.perf_counter()
            lm, prior = metric.window(j, paths, keyframes(j), s, z, pred, P, ratio)
            if j == 0:
                l0["v"] = lm
            if prior is not None:
                B.G.metre_prior[j] = prior
            torch.cuda.synchronize()
            rec["t_metric"].append(time.perf_counter() - tm)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        ops.put(("win", j, P, ratio))
        new_maps = [f for f in grid if s <= f < e and f not in owned]
        if new_maps:
            writes.append(writer.submit(save_map_inputs, os.path.join(mdir, f"w{j:05d}.npz"), new_maps, s, pred,
                                        mcfg["conf_pct"], mcfg["pix"]))
            owned.update(new_maps)
        # covisibility edges
        sj = keyframes(j)
        for g in cfg["edges"]["gaps"]:
            i = j - g
            if i < 0:
                continue
            si = keyframes(i)
            S = desc(sj) @ desc(si).T
            # frames shared through the windows' overlap are not covisibility
            m = [(sj[q_], si[p_]) for q_, p_ in zip(*np.where(S > sim)) if sj[q_] - si[p_] > O]
            if not m:
                continue
            Pe, _ = camera_to_world(da3, paths, list(range(e - O, e)) + si + sj)
            n_edges += 1
            ops.put(("loop", (j, i), {"cam_poses": Pe.astype(np.float32), "selected_i_frames": si,
                                      "selected_j_frames": sj, "frame_matches": [[x, y] for x, y in m],
                                      "overlap_size": O}))
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        # loops
        for i in cands.add_window(j, P, pend):
            si, sjl, fm = cands.loop_frames(j, i)
            jend = list(range(e - O, e))
            tl = time.perf_counter()
            Pb, _ = camera_to_world(da3, paths, jend + si + sjl + jend)
            torch.cuda.synchronize()
            rec["t_loops"].append(time.perf_counter() - tl)
            ops.put(("loop", (j, i), {"cam_poses": Pb.astype(np.float32), "selected_i_frames": si,
                                      "selected_j_frames": sjl, "frame_matches": fm, "overlap_size": O}))
        now = time.perf_counter()
        rec["t_detect"].append(t1 - t0)
        rec["t_window"].append(t2 - t1)
        rec["t_edges"].append(t3 - t2)
        rec["t_win"].append(now - last)
        last = now
    t_front = time.perf_counter() - t_start
    ops.put(None)
    th.join()
    assert not err, err[0]
    t_back = time.perf_counter() - t_start
    B.close()
    for w in writes:
        w.result()
    wall = time.perf_counter() - t_start
    poses = B.poses()
    with open(os.path.join(d, "backend_inputs.pkl"), "wb") as f:
        pickle.dump(dict(B.rec, metre_prior=dict(B.G.metre_prior), K_win=K_win), f)
    save_tum(os.path.join(d, "est.txt"), seq.stamps_ns, np.stack([to_opencv(P) for P in poses]))
    # map inputs: the final per-window log metres, the live poses after each window
    np.save(os.path.join(mdir, "log_metres.npy"), B.G.log_metres())
    back = cfg["map"]["snapshot_back"]
    Ps = np.zeros((len(W), back, 4, 4))
    for k in range(len(W)):
        lo, Pk = B.snaps[k]
        Ps[k, :len(Pk)] = Pk
    np.savez_compressed(os.path.join(mdir, "live_poses.npz"), lo=np.array([B.snaps[k][0] for k in range(len(W))]),
                        win=np.array(W), P=Ps)
    med = lambda x: float(np.median(x)) if x else 0.0
    write_run(d, seq0, cfg["mode"], wall,
              config=dict(cfg, cropped=crop, images="undistorted" if seq is not seq0 else "as loaded"),
              extra={"peak_gpu_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                     "realtime_factor": wall / ((seq.stamps_ns[-1] - seq.stamps_ns[0]) / 1e9),
                     "front_s": t_front, "backend_drain_s": t_back - t_front, "finish_s": wall - t_back,
                     "win_s_median": med(rec["t_win"]), "salad_s_median": med(rec["t_detect"]),
                     "da3_window_s_median": med(rec["t_window"]), "da3_edges_s_median": med(rec["t_edges"]),
                     "da3_loop_s_median": med(rec["t_loops"]), "metric_s_median": med(rec["t_metric"]),
                     "n_metric_priors": len(B.G.metre_prior), "n_windows": len(W), "n_edges": n_edges,
                     "n_loops": len(rec["t_loops"]), "n_loops_rejected": len(B.G.lc_rejected),
                     "init_guard_fired": B.guard_fired})
    print(f"xtream {cfg['mode']} {seq.dataset}/{seq.name}: {len(poses) / wall:.1f} fps, {n_edges} edges, "
          f"{len(rec['t_loops'])} loops -> {d}", flush=True)
    return d
