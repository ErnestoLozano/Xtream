"""The back end: the explicit graph fed with the front end's outputs, in arrival order.

Ops: ("win", k, raw window poses (OpenCV axes, the window's unit), depth ratio at the join) and ("loop", (j, i),
loop pass). `rec` keeps every input, with the window-0 log metre, so the same graph can be rebuilt from a run's
`backend_inputs.pkl`. After each window and its loops, the poses of the last `snapshot_back` frames are kept: the
live poses a map pass conditioned online would read (`snaps`).
"""

import numpy as np
import torch

from xtream.backend.explicit_graph import ExplicitGraph
from xtream.backend.sim3 import PM, se3_to_pose7, to_opencv, to_ours

SIGMA = dict(rot=0.0025, trans=0.02, link_block=0.05, link_reset=0.08, tie=0.05, loop_rot=0.01, loop_trans=0.1)


def windows(n, W, O):
    """Windows of W frames overlapping by O over n frames (CLoSeR's `Pi3.forward`): [(start, end)]."""
    out = []
    for s in range(0, n, max(W - O, 1)):
        e = min(s + W, n)
        if e - s >= O or (e == n and s < n):
            out.append((s, e))
        if e == n:
            break
    return out


def window0_metre(ds, cams, idx, span, device="cuda"):
    """The first window's metric scale from VI3's GT-free init (`xtream.init.IMUInit`): gyro bias in closed form
    over `span`, then GeoCalib gravity with the LS-g guard and the all-pairs velocity scale over `idx`. `cams`:
    {frame: pose (our axes, the window's unit)}. Returns (S, guard_fired)."""
    from xtream.init import IMUInit
    from xtream.init.bias import estimate_gyro_bias_window
    imu = {j: ds.get_imu_segment(j) for j in sorted(set(idx) | set(span))}
    bg = estimate_gyro_bias_window([imu[j] for j in span], device,
                                   predicted_poses=[torch.as_tensor(se3_to_pose7(cams[j]), dtype=torch.float32)
                                                    for j in span])
    samples = [dict(ds[idx[0]], **imu[idx[0]])] + [imu[j] for j in idx[1:]]
    T0inv = np.linalg.inv(cams[idx[0]])
    poses = [torch.as_tensor(se3_to_pose7(T0inv @ cams[j]), dtype=torch.float32, device=device) for j in idx]
    gravity = torch.tensor(ds.GRAVITY_WORLD, dtype=torch.float32, device=device)
    _, _, S, fired = IMUInit(ds, device=device).estimate(samples, poses, bg, gravity)
    return S, fired


class Backend:
    """The explicit graph (`ExplicitGraph`) on a sequence's windows."""

    def __init__(self, ds, window, overlap, lm_steps=(1, 100), imu_scale=(10, 1, 10, 10), reset_every=5,
                 keyframes=8, use_imu=True, loop_gate_dof=None, lc_check_t=0.0, lc_check_iters=3,
                 snapshot_back=256):
        """`ds`: a `FrontEndView`. `imu_scale`: factors on the calibration's (acc density, gyro density, acc walk,
        gyro walk). Other arguments as `ExplicitGraph`."""
        assert ds.has_imu or not use_imu, "mono-inertial graph on a sequence without IMU"
        self.ds, self.S, self.O, self.R_every = ds, window - overlap, overlap, reset_every
        self.windows = windows(len(ds), window, overlap)
        lever = PM @ np.asarray(ds._T_cam_imu, float)[:3, 3]
        ka, kg, kwa, kwg = imu_scale
        imu = (ka * ds.IMU_ACC_NOISE_DENSITY, kg * ds.IMU_GYRO_NOISE_DENSITY,
               kwa * ds.IMU_ACC_RANDOM_WALK, kwg * ds.IMU_GYRO_RANDOM_WALK)
        self.G = ExplicitGraph(imu, lever, float(np.linalg.norm(ds.GRAVITY_WORLD)), reset_every, SIGMA,
                               use_imu=use_imu, lm_steps=tuple(lm_steps), imu_first=ds.imu_first,
                               keyframes=keyframes, lc_check_t=lc_check_t, lc_check_iters=lc_check_iters,
                               loop_gate_dof=loop_gate_dof)
        self.seg, self.last = {}, -1
        self.snapshot_back, self.snaps = snapshot_back, {}
        self.guard_fired = None
        self.rec = {"windows": self.windows, "trace": [], "win": {}, "ratio": {}, "loops": {}, "l_init": None,
                    "R_every": reset_every}

    def _t(self, f):
        return float(self.ds.cam_times[f])

    def _imu(self, f):
        if f not in self.seg:
            s = self.ds.get_imu_segment(f)
            self.seg[f] = (s["imu_acc"].numpy().astype(float), s["imu_gyro"].numpy().astype(float),
                           s["imu_dts"].numpy().astype(float))
        return self.seg[f]

    def _snapshot(self, k):
        e = self.windows[k][1]
        lo = max(0, e - self.snapshot_back)
        ps = self.G.poses(list(range(lo, e)))
        self.snaps[k] = (lo, np.stack([to_opencv(ps[f].matrix()) for f in range(lo, e)]))

    def op(self, op, l_init="imu"):
        """One front-end output. `l_init`: the window-0 log metre, "imu" (VI3's init, `window0_metre`) or a value
        (the monocular metric prior's)."""
        rec, windows = self.rec, self.windows
        if op[0] == "win":
            _, k, raw, ratio = op
            assert k == self.last + 1, (k, self.last)
            if k > 0 and self.snapshot_back:
                self._snapshot(k - 1)
            s, e = windows[k]
            cams = to_ours(raw[: e - s])
            rec["win"][k], rec["ratio"][k] = raw[: e - s], ratio
            rec["trace"].append(("win", k))
            li = None
            if k == 0:
                if l_init == "imu":
                    L0 = np.linalg.inv(cams[0]) @ cams
                    idx = list(range(self.ds.imu_first, windows[1][0] if len(windows) > 1 else e))
                    span = list(range(max(self.ds.imu_first, idx[-1] + 1 - self.R_every * self.S), idx[-1] + 1))
                    S_v0, self.guard_fired = window0_metre(self.ds, {i: L0[i] for i in range(e - s)}, idx, span)
                    li = float(np.log(S_v0)) if np.isfinite(S_v0) and S_v0 > 0 else None
                else:
                    li = l_init
                rec["l_init"] = li
            self.last = k
            return self.G.add_window(k, list(range(s, e)), list(cams), self._t, self._imu, ratio=ratio, l_init=li)
        _, key, lp = op
        rec["loops"][key] = lp
        rec["trace"].append(("loop", key))
        assert key[0] == self.last, (key, self.last)
        self.G.add_loop(key[0], key[1], dict(lp, cam_poses=to_ours(lp["cam_poses"])), windows[self.last][1])
        return None

    def close(self):
        """After the last op: the last window's snapshot, then the final solve."""
        if self.snapshot_back and self.last >= 0:
            self._snapshot(self.last)
        self.G.finish()

    def poses(self):
        """Every frame's final pose (our axes), as 4x4."""
        final = self.G.poses(list(range(len(self.ds))))
        return [final[f].matrix() for f in range(len(self.ds))]
