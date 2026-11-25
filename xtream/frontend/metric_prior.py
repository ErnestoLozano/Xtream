"""The monocular metric prior: each window's log metre from a metric depth model on its keyframes.

m_k = median over keyframes of (metric depth / window depth), the metric model's output in metres as
output * f / 300 with f the window pass's own predicted focal. Its sigma is `sigma0` until `n0` windows, then
`mad_c` times the MAD of its inconsistency with the depth links, e_k = (l_k - l_{k-1}) - log r_k over consecutive
prior windows. Windows that barely move (camera extent / median depth below `static_gate`) get no prior.
"""

import numpy as np


class MetricPrior:
    def __init__(self, model, cfg):
        self.model, self.cfg = model, cfg
        self.lm_of, self.e_seen = {}, []

    def window(self, j, paths, keyframes, s, z, pred, P, ratio):
        """Window j (frames from s): its keyframes, depth z (N, H, W), prediction (DA3-Giant's intrinsics), poses P
        and join ratio. Returns (log metre of the window, (log metre, sigma) or None for a static window)."""
        import torch
        cfg = self.cfg
        with torch.no_grad():
            zn = np.asarray(self.model.inference([paths[f] for f in keyframes]).depth, np.float64)
        Hp, Wp = zn.shape[-2:]
        Hg, Wg = z.shape[-2:]
        Kp = np.asarray(pred.intrinsics, np.float64)[[f - s for f in keyframes]]
        zn = zn * ((Kp[:, 0, 0] * Wp / Wg + Kp[:, 1, 1] * Hp / Hg) / 2 / 300.0)[:, None, None]
        zw = np.median(z.reshape(len(z), -1), 1)
        lm = float(np.log(np.median(np.median(zn.reshape(len(keyframes), -1), 1) / zw[[f - s for f in keyframes]])))
        c = P[:, :3, 3]
        if np.linalg.norm(c - c.mean(0), axis=1).max() / np.median(zw) < cfg["static_gate"]:
            return lm, None
        sig = cfg["sigma0"]
        if j - 1 in self.lm_of and ratio and ratio > 0:
            self.e_seen.append(lm - self.lm_of[j - 1] - np.log(ratio))
        self.lm_of[j] = lm
        if len(self.e_seen) >= cfg["n0"]:
            x = np.asarray(self.e_seen)
            sig = cfg["mad_c"] * float(1.4826 * np.median(np.abs(x - np.median(x))))
        return lm, (lm, sig)
