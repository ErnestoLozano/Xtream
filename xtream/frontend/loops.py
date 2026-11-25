# The window-pair candidates, chord/arc filter, window NMS and adjacent fill follow CLoSeR's streaming loop closure
# (https://github.com/MoyangLi00/CLoSeR, loger/models/pi3.py); the repository states no license.

"""Streaming loop detection (CLoSeR's): SALAD retrieval per window, candidate pairs kept when the trajectory
between the two windows is not straight (chord/arc of the window centres below `chord_arc`), one loop pass per
retained window pair after a window-level NMS."""

import os
import sys

import numpy as np

from xtream.paths import CLOSER_ROOT, SALAD_CKPT


def load_detector(paths, window, cfg, sim):
    """CLoSeR's `LoopDetector` with the SALAD checkpoint: top-k retrieval above `sim`, at least `min_gap_windows`
    windows back."""
    assert os.path.exists(os.path.join(CLOSER_ROOT, "loger", "utils", "loop", "detector.py")), \
        f"CLoSeR's loop detector is missing in {CLOSER_ROOT}: run scripts/download_checkpoints.py"
    if CLOSER_ROOT not in sys.path:
        sys.path.insert(0, CLOSER_ROOT)
    from loger.utils.loop.detector import LoopDetector
    det = LoopDetector(image_list=paths, gt_traj_path=None, result_dir=None, ckpt_path=SALAD_CKPT,
                       nms_threshold=25, min_frame_gap=cfg["min_gap_windows"] * window, similarity_threshold=sim,
                       top_k=cfg["top_k"], use_nms=False, save_pair_vis=False)
    det.load_model()
    return det


def fill(core, lo, hi, slots):
    """CLoSeR's adjacent fill of a loop side: the matched frames grown by their neighbours in [lo, hi) up to
    `slots` frames."""
    sel = set(core)
    while len(sel) < slots:
        prev = set(sel)
        for f in sorted(prev):
            if f - 1 >= lo:
                sel.add(f - 1)
            if f + 1 < hi:
                sel.add(f + 1)
        if sel == prev:
            break
    return sorted(sel)[:slots]


class LoopCandidates:
    """Window pairs with SALAD matches, filtered by the chord/arc test on the window centres (each window's first
    frame, windows chained through their shared overlap frame)."""

    def __init__(self, windows, window, overlap, chord_arc, nms_windows):
        self.W, self.WS, self.O = windows, window, overlap
        self.chord_arc, self.nms = chord_arc, nms_windows
        self.wT, self.first, self.prevP = [], [], None
        self.pairs, self.match = {}, {}

    def win_of(self, f):
        """CLoSeR's `_frame_to_window_idx`."""
        return min(max(int(f) - self.O, 0) // (self.WS - self.O), len(self.W) - 1)

    def add_window(self, j, P, pend):
        """Window j's poses P (its pass, OpenCV axes) and its retrieval matches `pend` [(fi, fj, sim)]: chains the
        window centres and files the candidate pairs. Returns the earlier windows to close a loop with now."""
        O = self.O
        if j == 0:
            self.wT.append((np.eye(3), np.zeros(3)))
        else:
            pp, pc = self.prevP[len(self.prevP) - O], P[0]
            Rr = pp[:3, :3] @ pc[:3, :3].T
            tr = pp[:3, 3] - Rr @ pc[:3, 3]
            Rg, tg = self.wT[-1]
            self.wT.append((Rg @ Rr, Rg @ tr + tg))
        self.prevP = P
        self.first.append(self.wT[-1][0] @ P[0][:3, 3] + self.wT[-1][1])
        geo = {}
        for fi, fj, _ in pend:
            wi, wj = self.win_of(fi), self.win_of(fj)
            if abs(wi - wj) <= 1:
                continue
            big, small = max(wi, wj), min(wi, wj)
            if (small, big) not in geo:
                c = np.stack(self.first[small: min(big, len(self.first) - 1) + 1])
                arc = np.linalg.norm(np.diff(c, axis=0), axis=1).sum()
                geo[(small, big)] = len(c) < 2 or arc <= 0 or np.linalg.norm(c[-1] - c[0]) / arc < self.chord_arc
            if not geo[(small, big)]:
                continue
            b = self.pairs.setdefault(big, [])
            if small not in b:
                b.append(small)
            self.match.setdefault((big, small), []).append([int(fi), int(fj)])
        kept = []
        if j in self.pairs:
            cand = self.pairs[j]
            n = {i: len(self.match.get((j, i), [])) for i in cand}
            for i in sorted(cand, key=lambda i: (-n[i], i)):
                if all(abs(i - k) > self.nms for k in kept):
                    kept.append(i)
        return sorted(kept)

    def loop_frames(self, j, i):
        """The frames of loop (j, i): (side i, side j, matches), each side its matched frames filled to half of
        the pass's free slots."""
        WS, O = self.WS, self.O
        s, e = self.W[j]
        fm = self.match.get((j, i), [])
        i0, i1 = self.W[i]
        ci = sorted({y for x, y in fm if i0 <= y < i1}) or list(range(max(i1 - O, i0), i1))
        cj = sorted({x for x, y in fm if s <= x < e}) or list(range(max(e - O, s), e))
        si = fill(ci, i0, i1, (WS - 2 * O) // 2)
        sj = fill(cj, s, e, WS - 2 * O - (WS - 2 * O) // 2)
        return si, sj, fm
