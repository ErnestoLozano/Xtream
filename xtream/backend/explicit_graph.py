"""The explicit graph of Xtream's back end, monocular and mono-inertial alike.

States: per frame a pose X and (with the IMU) a velocity V; an IMU bias B per block of `reset_every` windows; a
log metre L per window (the window's unit in metres, log) and per loop. Factors:
- vision: consecutive frames of a window, their relative pose brought into the window's unit by its metre;
- metre links: consecutive windows' metres tied by the depth ratio on their shared frame;
- metric prior (monocular): a window's metre against a metric depth model's (Huber);
- IMU: GTSAM's preintegrated factors between consecutive frames, the bias's random walk between blocks;
- loops: a loop pass's relative pose between two windows' frames in its own unit (Huber), its metre tied to both
  windows' by the Umeyama scale on each side's frames.
The solve is a warm-started Levenberg-Marquardt over the whole graph, a few steps per window and to convergence
per loop. Settled blocks are compressed to keyframes. Loops are weighted before they enter by their innovation
against the graph (mono-inertial) or checked after the solve (monocular, CLoSeR's consistency check).
All poses are camera-to-world in our camera axes (x forward, y right, z down).
"""

import time

import gtsam
import numpy as np
from gtsam.symbol_shorthand import B, L, V, X
from scipy.spatial.transform import Rotation as Rot

from xtream.backend.sim3 import align

LOOP0 = 1_000_000   # metre keys of loops: L(LOOP0 + n)


def _vision(Zinv, Xa, Xb, l, jac):
    """Log(Z^-1 (R_ab, e^-l t_ab)), (R_ab, t_ab) = X_a^-1 X_b, and (with `jac`) its Jacobians in X_a, X_b, l."""
    rel = Xa.between(Xb)
    s = float(np.exp(-l))
    R, t = rel.rotation(), rel.translation()
    E = Zinv.compose(gtsam.Pose3(R, s * t))
    e = gtsam.Pose3.Logmap(E)
    if not jac:
        return e, None
    Ji = gtsam.Pose3.LogmapDerivative(E)
    Hb = Ji @ np.diag([1.0, 1.0, 1.0, s, s, s])
    Hl = np.zeros((6, 1))
    Hl[3:, 0] = -s * (R.matrix().T @ t)
    return e, (-Hb @ rel.inverse().AdjointMap(), Hb, Ji @ Hl)


def vision_factor(ka, kb, kl, Z, noise):
    """The state brought into the measurement's unit (`_vision`), analytic Jacobians."""
    Zinv = Z.inverse()

    def f(this, v, H):
        a, b, lk = this.keys()
        e, J = _vision(Zinv, v.atPose3(a), v.atPose3(b), v.atVector(lk)[0], H is not None)
        if H is not None:
            H[0], H[1], H[2] = J
        return e
    return gtsam.CustomFactor(noise, [ka, kb, kl], f)


def anchored_vision_factor(kc, kb, kl, T_ca, Z, noise):
    """`vision_factor` between frame a and X_b, frame a removed by compression and held rigidly at
    X_a = X_c T_ca, T_ca its metric pose from its anchor c when it left. Keys X_c, X_b, L_loop; analytic Jacobians."""
    Zinv = Z.inverse()
    Ad = T_ca.inverse().AdjointMap()

    def f(this, v, H):
        c, b, kk = this.keys()
        e, J = _vision(Zinv, v.atPose3(c).compose(T_ca), v.atPose3(b), v.atVector(kk)[0], H is not None)
        if H is not None:
            H[0], H[1], H[2] = J[0] @ Ad, J[1], J[2]     # X_c's right perturbation carried through T_ca
        return e
    return gtsam.CustomFactor(noise, [kc, kb, kl], f)


class ExplicitGraph:
    """Pose and velocity per frame, IMU bias per block, log metre per window and per loop (module docstring)."""

    def __init__(self, imu, lever, g_mag, reset_every, sigma, use_imu=True, lm_steps=(1, 100), imu_first=0,
                 keyframes=8, lc_check_t=0.0, lc_check_iters=3, loop_gate_dof=None):
        """
        Args:
            imu: (acc density, gyro density, acc walk, gyro walk), the dataset's calibration scaled.
            lever: IMU origin in the camera frame (our axes).
            g_mag: gravity magnitude [m/s^2].
            reset_every: windows per IMU-bias block (and per compression step).
            sigma: rot [rad], trans [m], link_block, link_reset, tie, loop_rot, loop_trans [m].
            use_imu: IMU factors and states (mono-inertial); without, the metres come from the links and priors.
            lm_steps: LM steps (per window, per loop).
            imu_first: the first frame the IMU reaches (inertial states start there).
            keyframes: frames kept per window when a settled block is compressed (evenly spaced, first and last
                included); a removed frame takes its share (by time) of its segment's later correction.
            lc_check_t: CLoSeR's loop check (`_lc_check`): loops whose residual translation in their own unit
                exceeds it after the solve are removed (0: off), for at most `lc_check_iters` rounds.
            loop_gate_dof: each loop weighted before it enters by the Student-t weight of its innovation on its
                shape (`_loop_weight`), with these degrees of freedom (None: off).
        """
        self._imu_p = (float(g_mag), imu, np.asarray(lever, float))
        self.p, self.walk = self._params(imu[0] ** 2, imu[1] ** 2), np.r_[[imu[2]] * 3, [imu[3]] * 3]
        self.R, self.sig = int(reset_every), dict(sigma)
        self.imu_first = int(imu_first)
        self.lc_check_t, self.lc_check_iters = float(lc_check_t), int(lc_check_iters)
        assert loop_gate_dof is None or lc_check_t == 0, "loop gate or loop check, not both"
        self.gate_dof = None if loop_gate_dof is None else float(loop_gate_dof)
        self.loop_fac, self.lc_rejected = {}, {}
        self.loop_d2, self.loop_w, self.marg_damped = [], [], []
        self.use_imu = bool(use_imu)
        self.graph, self.lm_steps, self.lam = gtsam.NonlinearFactorGraph(), tuple(lm_steps), 1e-5
        self._kd = []
        self.est = gtsam.Values()
        self.win_of, self.raw = {}, {}
        self.blocks, self.n_loops, self.t_update = set(), 0, []
        self.ties = []
        self.kf = int(keyframes)
        self.new_of, self.win_fac, self.vis_sig, self.anchor = {}, {}, {}, {}
        self.metre_prior = {}   # window -> (log metre, sigma): the monocular metric prior
        self.seg = {}           # removed frame -> (d, T0_cd, tau): its segment's end and pose at compression
        self.n_compressed, self.n_kept = 0, 0

    def _params(self, acc_var, gyr_var):
        g_mag, _, lever = self._imu_p
        p = gtsam.PreintegrationParams.MakeSharedU(g_mag)
        p.setAccelerometerCovariance(np.eye(3) * acc_var)
        p.setGyroscopeCovariance(np.eye(3) * gyr_var)
        p.setIntegrationCovariance(np.eye(3) * 1e-8)
        p.setBodyPSensor(gtsam.Pose3(gtsam.Rot3(), lever))
        return p

    def _l(self, key):
        return float(self.est.atVector(key)[0])

    def _update(self, g, vals, converge=False):
        """The factors join the graph, then a few LM steps (to convergence if `converge`)."""
        self.graph.push_back(g)
        self.est.insert(vals)
        self._lm(self.lm_steps[1] if converge else self.lm_steps[0])

    def _lm(self, steps, rtol=1e-6):
        """Levenberg-Marquardt warm-started from the current estimate: each step relinearises everything and is
        accepted only if the error drops."""
        g, est = self.graph, self.est
        order = gtsam.Ordering.ColamdNonlinearFactorGraph(g)
        err = g.error(est)
        for _ in range(steps):
            lin = g.linearize(est)
            while True:
                damp = gtsam.GaussianFactorGraph()
                damp.push_back(lin)
                for k, d in self._key_dims():
                    damp.add(gtsam.JacobianFactor(k, np.eye(d), np.zeros(d),
                                                  gtsam.noiseModel.Isotropic.Sigma(d, 1.0 / np.sqrt(self.lam))))
                try:
                    new = est.retract(damp.optimize(order))
                    e_new = g.error(new)
                except RuntimeError:          # indeterminate at this damping: as GTSAM's LM, damp more
                    e_new = np.inf
                if e_new < err:
                    self.lam = max(self.lam / 10, 1e-10)
                    break
                self.lam *= 10
                if self.lam > 1e10:
                    self.lam = 1e-5
                    self.est = est
                    return
            done = (err - e_new) < rtol * err
            est, err = new, e_new
            if done:
                break
        self.est = est

    def _key_dims(self):
        if len(self._kd) != self.est.size():
            dim = {"x": 6, "v": 3, "b": 6, "l": 1}
            self._kd = [(k, dim[chr(gtsam.Symbol(k).chr())]) for k in self.est.keys()]
        return self._kd

    def finish(self):
        """The final solve, to convergence."""
        self._lm(self.lm_steps[1])

    def add_window(self, k, frames, poses, t_of, imu_of, ratio=None, l_init=None):
        """Window k: its frames (contiguous) and their poses (our axes, its unit). `ratio`: the depth ratio at the
        join with window k-1 (previous window's depth over this one's on the shared frame); `l_init`: log metre of
        the first window. `t_of(f)`: frame f's time [s]; `imu_of(f)`: its IMU segment (acc, gyro, dts). Returns the
        new frames' poses after the update."""
        g, vals = gtsam.NonlinearFactorGraph(), gtsam.Values()
        P = {f: np.asarray(q, float) for f, q in zip(frames, poses)}
        self.raw[k] = P
        new = [f for f in frames if f not in self.win_of]
        assert new and all(b == a + 1 for a, b in zip(frames, frames[1:]))
        blk = k // self.R
        # log metre of this window: window 0 from its init, then chained by the depth ratios
        if k == 0:
            l0 = 0.0 if l_init is None else float(l_init)
            vals.insert(L(0), np.array([l0]))
            g.add(gtsam.PriorFactorVector(L(0), np.array([l0]), gtsam.noiseModel.Isotropic.Sigma(1, 1.0)))
            lk = l0
        else:
            z = float(np.log(ratio)) if ratio is not None else 0.0
            lk = self._l(L(k - 1)) + z
            vals.insert(L(k), np.array([lk]))
            g.add(gtsam.BetweenFactorVector(L(k - 1), L(k), np.array([z]),
                                            gtsam.noiseModel.Isotropic.Sigma(1, self.sig["link_reset"])))
        if k in self.metre_prior:
            lm, sm = self.metre_prior[k]
            nm = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345),
                                                gtsam.noiseModel.Isotropic.Sigma(1, sm))
            g.add(gtsam.PriorFactorVector(L(k), np.array([lm]), nm))
        self.vis_sig[k] = np.r_[[self.sig["rot"]] * 3, [self.sig["trans"] * np.exp(-lk)] * 3]
        vis = gtsam.noiseModel.Diagonal.Sigmas(self.vis_sig[k])
        # bias of this block
        if self.use_imu and blk not in self.blocks:
            b_prev = (self.est.atConstantBias(B(blk - 1)) if blk > 0 else gtsam.imuBias.ConstantBias())
            vals.insert(B(blk), b_prev)
            if blk == 0:
                g.add(gtsam.PriorFactorConstantBias(B(0), b_prev, gtsam.noiseModel.Diagonal.Sigmas(
                    np.r_[[0.1] * 3, [0.1] * 3])))
            else:
                T_blk = self.R * (t_of(frames[-1]) - t_of(frames[0]))
                g.add(gtsam.BetweenFactorConstantBias(B(blk - 1), B(blk), gtsam.imuBias.ConstantBias(),
                                                      gtsam.noiseModel.Diagonal.Sigmas(np.sqrt(T_blk) * self.walk)))
            self.blocks.add(blk)
        # initial poses of the new frames, chained from the window's own poses
        pos = {}
        if k == 0:
            f0 = new[0]
            X0 = gtsam.Pose3(gtsam.Rot3(self._gravity_frame(P, new, imu_of)), np.zeros(3))
            vals.insert(X(f0), X0)
            pos[f0] = X0
            rp = 1e-6 if not self.use_imu else 0.3
            g.add(gtsam.PriorFactorPose3(X(f0), X0, gtsam.noiseModel.Diagonal.Sigmas(np.r_[[rp] * 3, [1e-3] * 3])))
            todo = new[1:]
        else:
            todo = new
        for f in todo:
            h = f - 1
            Z = gtsam.Pose3(np.linalg.inv(P[h]) @ P[f])
            Xh = pos[h] if h in pos else self.est.atPose3(X(h))
            Xf = Xh.compose(gtsam.Pose3(Z.rotation(), np.exp(lk) * Z.translation()))
            vals.insert(X(f), Xf)
            pos[f] = Xf
            g.add(vision_factor(X(h), X(f), L(k), Z, vis))
        if self.use_imu:
            bias = self.est.atConstantBias(B(blk)) if self.est.exists(B(blk)) else vals.atConstantBias(B(blk))
            vel = {}
            for f in new:
                if f < self.imu_first:
                    continue
                h = f - 1
                if f == self.imu_first:
                    # the first inertial state: no IMU before it
                    vel[f] = (np.zeros(3) if f == 0 else (pos[f].translation() - (
                        pos[h] if h in pos else self.est.atPose3(X(h))).translation()) / (t_of(f) - t_of(h)))
                    vals.insert(V(f), vel[f])
                    continue
                if t_of(f) == t_of(h):
                    # two images at one instant (no time between them): no IMU, one velocity
                    vel[f] = vel[h] if h in vel else self.est.atVector(V(h))
                    vals.insert(V(f), vel[f])
                    g.add(gtsam.BetweenFactorVector(V(h), V(f), np.zeros(3), gtsam.noiseModel.Isotropic.Sigma(3, 1e-6)))
                    continue
                acc, gyr, dts = imu_of(f)
                pim = gtsam.PreintegratedImuMeasurements(self.p, bias)
                for a_, w_, d_ in zip(acc, gyr, dts):
                    if d_ > 0:
                        pim.integrateMeasurement(a_, w_, d_)
                Xh = pos[h] if h in pos else self.est.atPose3(X(h))
                vel[f] = (pos[f].translation() - Xh.translation()) / (t_of(f) - t_of(h))
                vals.insert(V(f), vel[f])
                g.add(gtsam.ImuFactor(X(h), V(h), X(f), V(f), B(blk), pim))
        for f in new:
            self.win_of[f] = k
        self.new_of[k], self._t_of, self._imu_of = new, t_of, imu_of
        t0 = time.perf_counter()
        n0 = self.graph.size()
        self._update(g, vals)
        self.win_fac[k] = (n0, self.graph.size())
        if k % self.R == 0 and k // self.R >= 2:
            # block k // R - 2 has been solved through two resets since: settled, compressed
            self._compress(k // self.R - 2)
        self.t_update.append(time.perf_counter() - t0)
        return {f: self.est.atPose3(X(f)) for f in new}

    def _gravity_frame(self, P, new, imu_of):
        """Rotation of a gravity-aligned world: the mean specific force over the window (camera axes, rotated into
        its first frame by the window) is up. Without the IMU, the first camera's frame."""
        f0 = new[0]
        up = np.zeros(3)
        if self.use_imu:
            for f in new[1:]:
                if f > self.imu_first:
                    up += P[f0][:3, :3].T @ P[f][:3, :3] @ imu_of(f)[0].mean(0)
        up = up / np.linalg.norm(up) if np.linalg.norm(up) > 0 else np.array([0, 0, 1.0])
        ax = np.cross(up, [0, 0, 1.0])
        ang = np.arccos(np.clip(up[2], -1, 1))
        return Rot.from_rotvec(ax / max(np.linalg.norm(ax), 1e-12) * ang).as_matrix()

    def _keep(self, new):
        """Frames of a window that stay at compression: `keyframes` evenly spaced, its first and last new frame
        included, so every compressed segment lies in one window (one unit)."""
        n = len(new)
        return {new[i] for i in np.round(np.linspace(0, n - 1, max(min(self.kf, n), 2))).astype(int)} | \
            {new[0], new[-1]}

    def _compress(self, b):
        """Block b's frames that are not kept leave the graph: between consecutive kept frames c < d of one
        window, its vision factors become one (the raw relative pose c -> d, the chain's covariance composed to
        first order) and its IMU factors one (the raw IMU over (t_c, t_d] preintegrated again, bias of the block).
        A removed frame keeps its raw pose from its anchor, the last kept frame before it in its window."""
        ends = {x for (_, a, b_, _, _) in self.loop_fac.values() for x in (a, b_)}
        g = gtsam.NonlinearFactorGraph()
        for k in range(b * self.R, (b + 1) * self.R):
            if k not in self.new_of:
                continue
            new = self.new_of[k]
            kept = sorted(self._keep(new) | (ends & set(new)) | ({self.imu_first} & set(new)))
            self.n_kept += len(kept)
            inner = set(new[1:])          # frames whose incoming factors lie inside window k
            gone = set(new) - set(kept)
            # remove window k's frame-to-frame vision, IMU and equal-stamp velocity factors inside the window
            for i in range(*self.win_fac[k]):
                f_ = self.graph.at(i)
                if f_ is None:
                    continue
                xs = [gtsam.Symbol(x) for x in f_.keys()]
                tgt = [s.index() for s in xs if chr(s.chr()) in "xv"]
                if tgt and max(tgt) in inner and all(chr(s.chr()) in "xvlb" for s in xs) and \
                        type(f_).__name__ in ("CustomFactor", "ImuFactor", "BetweenFactorVector") and \
                        not (type(f_).__name__ == "BetweenFactorVector" and chr(xs[0].chr()) == "l"):
                    self.graph.remove(i)
            P = self.raw[k]
            Sv = np.diag(self.vis_sig[k] ** 2)
            blk = k // self.R
            bias = self.est.atConstantBias(B(blk)) if self.use_imu else None
            for c, d in zip(kept[:-1], kept[1:]):
                Z = gtsam.Pose3(np.linalg.inv(P[c]) @ P[d])
                # first-order composition of the chain's measurement noise, carried to its end d
                Sig = np.zeros((6, 6))
                for h in range(c, d):
                    Ad = gtsam.Pose3(np.linalg.inv(P[h + 1]) @ P[d]).inverse().AdjointMap()
                    Sig += Ad @ Sv @ Ad.T
                g.add(vision_factor(X(c), X(d), L(k), Z, gtsam.noiseModel.Gaussian.Covariance(Sig)))
                if not self.use_imu or c < self.imu_first:
                    continue
                meas = [(a_, w_, d_) for f in range(c + 1, d + 1) for a_, w_, d_ in zip(*self._imu_of(f)) if d_ > 0]
                pim = gtsam.PreintegratedImuMeasurements(self.p, bias)
                for a_, w_, d_ in meas:
                    pim.integrateMeasurement(a_, w_, d_)
                if pim.deltaTij() > 0:
                    g.add(gtsam.ImuFactor(X(c), V(c), X(d), V(d), B(blk), pim))
                else:
                    g.add(gtsam.BetweenFactorVector(V(c), V(d), np.zeros(3), gtsam.noiseModel.Isotropic.Sigma(3, 1e-6)))
            for f in gone:
                c = max(x for x in kept if x < f)
                self.anchor[f] = (c, self.est.atPose3(X(c)).between(self.est.atPose3(X(f))))
                d = min(x for x in kept if x > f)
                t = self._t_of
                self.seg[f] = (d, self.est.atPose3(X(c)).between(self.est.atPose3(X(d))),
                               (t(f) - t(c)) / (t(d) - t(c)) if t(d) > t(c) else 0.0)
                self.est.erase(X(f))
                if self.est.exists(V(f)):
                    self.est.erase(V(f))
            self.n_compressed += len(gone)
        self.graph.push_back(g)
        self._kd = []

    def _pose_of(self, f, est=None):
        """Frame f's pose: its state, or (removed by compression) its anchor's with the pose it had from it plus its
        share of the segment's correction since."""
        est = self.est if est is None else est
        if est.exists(X(f)):
            return est.atPose3(X(f))
        c, T = self.anchor[f]
        Xf = est.atPose3(X(c)).compose(T)
        d, T0, tau = self.seg[f]
        # d's correction since compression, in d's frame then, carried to f and scaled by tau
        delta = gtsam.Pose3.Logmap(T0.inverse().compose(est.atPose3(X(c)).between(est.atPose3(X(d)))))
        return Xf.compose(gtsam.Pose3.Expmap(tau * T.between(T0).AdjointMap() @ delta))

    def _attach(self, f):
        """(key, T) holding frame f at X_key T: its own state, else (compression) its anchor."""
        if self.est.exists(X(f)):
            return X(f), gtsam.Pose3()
        return X(self.anchor[f][0]), self.anchor[f][1]

    def add_loop(self, wj, wi, lp, frame_end_j):
        """A loop pass between window wj (current) and wi (earlier), or a covisibility edge: one relative pose (its
        middle frames) in the pass's own unit, its metre tied to both windows' by the Umeyama scale on each side's
        frames. `lp`: cam_poses (the pass's poses, our axes), overlap_size, selected_i_frames, selected_j_frames;
        the pass reads [end of j | frames of i | frames of j]. Returns whether it entered the graph."""
        P = lp["cam_poses"]
        ov = lp["overlap_size"]
        si, sj = list(lp["selected_i_frames"]), list(lp["selected_j_frames"])
        jend = list(range(frame_end_j - ov, frame_end_j))
        bi = {f: ov + n for n, f in enumerate(si)}
        bj = {f: ov + len(si) + n for n, f in enumerate(sj)}
        bj.update({f: n for n, f in enumerate(jend)})
        a, b = si[len(si) // 2], sj[len(sj) // 2]
        if not ((self.est.exists(X(a)) or a in self.anchor) and self.est.exists(X(b))):
            return False
        g, vals = gtsam.NonlinearFactorGraph(), gtsam.Values()
        key = L(LOOP0 + self.n_loops)
        self.n_loops += 1

        def side(w, idx):
            """Umeyama scale of the pass onto the window's poses of the same frames, and the sigma of its log: the
            fit's standard error (a side that barely moves carries no unit) on top of the tie sigma."""
            fr = [f for f in idx if f in self.raw[w]]
            Zs = align([self.raw[w][f] for f in fr], [P[idx[f]] for f in fr])
            s, R = float(Zs.scale()), Zs.rotation().matrix()
            y = np.array([self.raw[w][f][:3, 3] for f in fr])
            x = np.array([P[idx[f]][:3, 3] for f in fr])
            x, y = x - x.mean(0), y - y.mean(0)
            r = y - s * x @ R.T
            var_r = float((r ** 2).sum()) / max(3 * len(fr) - 4, 1)
            sx = max(float((x ** 2).sum()), 1e-300)
            return s, float(np.sqrt(self.sig["tie"] ** 2 + var_r / sx / s ** 2))
        (s_i, q_i), (s_j, q_j) = side(wi, bi), side(wj, bj)
        lb = self._l(L(wi)) + np.log(s_i) if q_i <= q_j else self._l(L(wj)) + np.log(s_j)
        vals.insert(key, np.array([lb]))
        # never undetermined: a weak prior on the loop's metre
        g.add(gtsam.PriorFactorVector(key, np.array([lb]), gtsam.noiseModel.Isotropic.Sigma(1, 1.0)))
        Z = gtsam.Pose3(np.linalg.inv(P[bi[a]]) @ P[bj[b]])
        w = 1.0
        if self.gate_dof is not None:
            kl, ls, q = (L(wi), np.log(s_i), q_i) if q_i <= q_j else (L(wj), np.log(s_j), q_j)
            w *= self._loop_weight(a, b, Z, kl, ls, q)
        q_i, q_j = q_i / np.sqrt(w), q_j / np.sqrt(w)
        # indices once g joins the graph: the vision factor (after the prior), then the ties
        n0 = self.graph.size()
        self.loop_fac[(wj, wi)] = ([n0 + 1, n0 + 2, n0 + 3], a, b, Z, key)
        sig = np.r_[[self.sig["loop_rot"]] * 3, [self.sig["loop_trans"] * np.exp(-lb)] * 3] / np.sqrt(w)
        noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345),
                                               gtsam.noiseModel.Diagonal.Sigmas(sig))
        if self.est.exists(X(a)):
            g.add(vision_factor(X(a), X(b), key, Z, noise))
        else:
            # a left the graph at compression: through its anchor and its window's metre
            c, T_ca = self.anchor[a]
            g.add(anchored_vision_factor(X(c), X(b), key, T_ca, Z, noise))
        g.add(gtsam.BetweenFactorVector(L(wi), key, np.array([np.log(s_i)]), gtsam.noiseModel.Isotropic.Sigma(1, q_i)))
        g.add(gtsam.BetweenFactorVector(L(wj), key, np.array([np.log(s_j)]), gtsam.noiseModel.Isotropic.Sigma(1, q_j)))
        self.ties.append((wi, wj, s_i, q_i, s_j, q_j))
        t0 = time.perf_counter()
        self._update(g, vals, converge=True)
        if self.lc_check_t > 0:
            self._lc_check()
        self.t_update.append(time.perf_counter() - t0)
        return True

    def _joint_marginal(self, keys):
        """The graph's joint marginal covariance of `keys`. When the undamped Cholesky meets an ill-conditioned
        system (tight IMU noise), the system damped as LM damps it (lambda I on the whitened Hessian, from GTSAM's
        LM defaults: initial 1e-5, x10 per try, at most 1e5)."""
        try:
            return gtsam.Marginals(self.graph, self.est).jointMarginalCovariance(keys)
        except RuntimeError:
            pass
        lin, z = self.graph.linearize(self.est), self.est.zeroVectors()
        p = gtsam.LevenbergMarquardtParams()
        lam = p.getlambdaInitial()
        while lam <= p.getlambdaUpperBound():
            g = gtsam.GaussianFactorGraph()
            g.push_back(lin)
            for k in self.est.keys():
                d = z.at(k).shape[0]
                g.add(gtsam.JacobianFactor(k, np.sqrt(lam) * np.eye(d), np.zeros(d), gtsam.noiseModel.Unit.Create(d)))
            try:
                jm = gtsam.Marginals(g, self.est).jointMarginalCovariance(keys)
                self.marg_damped.append(lam)
                return jm
            except RuntimeError:
                lam *= p.getlambdaFactor()
        raise RuntimeError("joint marginal: indeterminant even at LM's largest damping")

    def _loop_weight(self, a, b, Z, kl, ls, q):
        """A loop's weight before it enters, from its pose innovation nu = Log(Zhat^-1 X_a^-1 X_b) (Zhat: Z in metres
        by the metre l + log s of its better-tied side): d2 = nu' S^-1 nu, S the graph's joint marginal of X_a (or
        its anchor), X_b and l carried through plus the loop's own noise (loop sigmas, q), minimised over the loop's
        metre (its shape only). Innovations are heavy-tailed with a per-sequence scale lambda, taken from the
        graph's own window vision factors (their mean whitened squared residual, times 1.23); the weight is the
        Student-t one, (dof + p) / (dof + d2 / lambda), at most 1."""
        (ka, T), (kb, Tb) = self._attach(a), self._attach(b)
        Xc, Xb, l0 = self.est.atPose3(ka), self.est.atPose3(kb), self._l(kl)

        def inn(d):
            # d: X_c (6), X_b (6), l (1) | loop pose noise (6, metric), log s (1)
            Xa = Xc.retract(d[:6]).compose(T)
            Zh = gtsam.Pose3(Z.rotation(), np.exp(l0 + d[12] + ls + d[19]) * Z.translation())
            return gtsam.Pose3.Logmap(Zh.compose(gtsam.Pose3.Expmap(d[13:19])).between(
                Xa.between(Xb.retract(d[6:12]).compose(Tb))))
        h = 1e-6
        J = np.stack([(inn(h * e) - inn(-h * e)) / (2 * h) for e in np.eye(20)], 1)
        jm = self._joint_marginal([ka, kb, kl])
        keys, off = [ka, kb, kl], [0, 6, 12, 13]
        Sx = np.zeros((13, 13))
        for u, ku in enumerate(keys):
            for v, kv in enumerate(keys):
                Sx[off[u]:off[u + 1], off[v]:off[v + 1]] = jm.at(ku, kv)
        Sm = np.diag(np.r_[[self.sig["loop_rot"] ** 2] * 3, [self.sig["loop_trans"] ** 2] * 3, q ** 2])
        S = J[:, :13] @ Sx @ J[:, :13].T + J[:, 13:] @ Sm @ J[:, 13:].T
        nu = inn(np.zeros(20))
        nu, S, Jl = nu[0:6], S[0:6, 0:6], J[0:6, 19]
        d2 = float(nu @ np.linalg.solve(S, nu))
        p = len(nu)
        # minimised over the loop's metre (its direction J[:, 19]): d2 of the scale-free components
        Si_nu, Si_j = np.linalg.solve(S, nu), np.linalg.solve(S, Jl)
        d2, p = d2 - float(Jl @ Si_nu) ** 2 / float(Jl @ Si_j), p - 1
        self.loop_d2.append(d2)
        r2, n = 0.0, 0
        for i in range(self.graph.size()):
            f_ = self.graph.at(i)
            if f_ is not None and type(f_).__name__ == "CustomFactor" and gtsam.Symbol(f_.keys()[-1]).index() < LOOP0:
                r2, n = r2 + 2 * f_.error(self.est), n + 6
        lam = 1.23 * r2 / n
        w = min(1.0, (self.gate_dof + p) / (self.gate_dof + d2 / lam))
        self.loop_w.append(w)
        return w

    def _lc_check(self):
        """CLoSeR's loop consistency check (`Pi3._apply_frame_pgo_incremental`, `pgo_lc_check_t`,
        `pgo_lc_check_iters`): loops whose residual translation, in the loop's own unit, exceeds `lc_check_t` after
        the solve are removed and the graph solved again, for at most `lc_check_iters` rounds. Removed loops stay
        removed, with their ties."""
        for _ in range(self.lc_check_iters):
            bad = []
            for k, (idx, a, b, Z, key) in self.loop_fac.items():
                if k in self.lc_rejected:
                    continue
                Xab = self._pose_of(a).between(self._pose_of(b))
                s = np.exp(-self._l(key))
                E = Z.between(gtsam.Pose3(Xab.rotation(), s * Xab.translation()))
                et = float(np.linalg.norm(E.translation()))
                if et > self.lc_check_t:
                    bad.append((k, et, idx))
            if not bad:
                return
            for k, et, idx in bad:
                for i in idx:
                    self.graph.remove(i)
                self.lc_rejected[k] = et
            self._lm(self.lm_steps[1])

    def poses(self, frames):
        return {f: self._pose_of(f) for f in frames}

    def log_metres(self):
        """Every window's current log metre (its unit in metres, log), in window order."""
        n = len(self.new_of)
        return np.array([self._l(L(k)) for k in range(n)])
