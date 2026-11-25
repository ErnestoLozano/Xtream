# From VI3 (https://github.com/ErnestoLozano/vi3, commit a2e5f62), vi3/init/imu_init.py: the init methods Xtream
# uses, with their code and docstrings, behind a new public `estimate()`. VI3's GT init, IMU-target integration and
# diagnostics are omitted, so the docstrings' references to them (`build`, `_build_imu_targets`, the `gt` preset)
# point to VI3.

"""The GT-free visual-inertial initialization of a window, and its IMU targets.

`IMUInit` is the init block of vi3 — the part every grounding method
(``vi3.grounding_methods``) starts from. Given the window's frames, their IMU
streams and the frozen foundation model's feed-forward poses, it estimates

  * frame-0 gravity — GeoCalib on the first image, cross-checked by the LS-g
    consensus guard (``g_ls_reject_deg``) against the gravity the IMU and the
    vision trajectory jointly imply (``_gravity_consistency_guard``);
  * the gyro bias — in closed form from the gyro stream and the model's relative
    rotations (``init_bg_from_imu``; held at zero where it is not observable);
  * the metric scale S of the vision trajectory and the initial velocity v0 —
    from IMU velocity DIFFERENCES (v0-independent), ``v0 = S · v_bar_0`` with
    ``|v0|`` from all pairs (``_v0_from_vel_scale``);

and integrates the IMU from frame 0 with them into the target trajectory
``{"p", "q", "v0", "g_rel"}`` (frame-0-relative, metric). The ``gt`` preset swaps
in GT gravity, GT v0 and GT biases where the dataset has them (EuRoC) — see
``vi3.init.presets``.

Biases are FIXED — set once by the init (plain tensors, never trained), so the
targets are static. Frame-0 gravity has three sources (``init_g_source``):
``geocalib`` (``estimate_gravity_from_image``), ``imu`` (closed-form accelerometer
attitude, ``estimate_gravity_in_frame0``) and ``gt_r0``.
"""

import math

import pypose as pp
import torch

from xtream.init.geocalib import build_geocalib
from xtream.init.gravity import estimate_gravity_from_image, estimate_gravity_in_frame0
from xtream.init.preintegration import IMUPreIntegrator


class IMUInit:
    """GT-free gravity, scale and v0 of one window (see the module docstring)."""

    # Fallback for datasets that don't declare noise densities.
    _FALLBACK_ACC_NOISE_DENSITY  = 0.2
    _FALLBACK_GYRO_NOISE_DENSITY = 0.02

    def __init__(self, dataset, device="cuda", g_ls_reject_deg=20.0, v0_mag_mode="robust_allpairs"):
        """`dataset`: provides ``GRAVITY_WORLD`` and the IMU noise densities.
        `g_ls_reject_deg`: floor of the guard's consensus width (see
        `_gravity_consistency_guard`). `v0_mag_mode`: "robust_allpairs" (|v0|
        from every pair, median) or "first_pair"."""
        self.dataset = dataset
        self.device = device
        self._geocalib_model = None
        self._g_ls_reject_deg = float(g_ls_reject_deg)
        self._v0_mag_mode = str(v0_mag_mode).lower()
        self._v0_mag_msg = None
        self._guard_fired = None
        self._poses = None
        self._preint_cache = {}
        self.ba = torch.zeros(3, device=self.device)
        self.bg = torch.zeros(3, device=self.device)

    @property
    def acc_noise_density(self) -> float:
        return float(getattr(self.dataset, "IMU_ACC_NOISE_DENSITY",
                              self._FALLBACK_ACC_NOISE_DENSITY))

    @property
    def gyro_noise_density(self) -> float:
        return float(getattr(self.dataset, "IMU_GYRO_NOISE_DENSITY",
                              self._FALLBACK_GYRO_NOISE_DENSITY))

    def estimate(self, samples, poses, bg, gravity):
        """Gravity, v0 and scale of one window.

        Args:
            samples: per frame, a dict with ``imu_acc`` [L, 3], ``imu_gyro`` [L, 3],
                ``imu_dts`` [L] (the IMU since the previous frame); the first also
                ``image`` [3, H, W] and ``K`` [3, 3] for GeoCalib.
            poses: per frame, the vision pose relative to frame 0, pose7 [7].
            bg: [3] gyro bias (held fixed); the accelerometer bias is 0.
            gravity: [3] world gravity (its norm sets ‖g‖).

        Returns:
            (g_rel [3], v0 [3], S, guard_fired): frame-0 gravity, initial velocity,
            the metric scale of the vision poses, and whether the guard replaced
            GeoCalib's gravity.
        """
        self._poses = poses
        self._preint_cache.clear()
        with torch.no_grad():
            self.bg.copy_(bg.to(self.bg))
            g = self._resolve_gravity(samples, gravity, self.device)
            v0, s_hat = self._v0_from_vel_scale(samples, g, self.device)
            v0, s_hat, g = self._gravity_consistency_guard(samples, g, v0, s_hat, self.device)
        return g, v0, float(s_hat), bool(self._guard_fired)

    @staticmethod
    def _preint_key(acc, gyro, dts, ba, bg, preintegrator):
        """Content signature for `_preint_cached` (shape + two reductions per
        input). `id()` is unusable: call sites pass fresh `.to(device)` copies."""
        def sig(t):
            return (tuple(t.shape), float(t.sum()), float(t.abs().sum()))
        return (sig(acc), sig(gyro), sig(dts), sig(ba), sig(bg),
                preintegrator.acc_density, preintegrator.gyro_density)

    def _preint_cached(self, preintegrator, acc, gyro, dts, ba, bg):
        """Memoized `IMUPreIntegrator.forward` for the init solves. The
        preintegrated terms depend only on (acc, gyro, dts, ba, bg), never on
        gravity or S, yet the gravity guard's fixed point re-integrates the same
        segments many times per window. One `IMUInit` (and memo) per window.
        """
        key = self._preint_key(acc, gyro, dts, ba, bg, preintegrator)
        hit = self._preint_cache.get(key)
        if hit is None:
            hit = preintegrator(acc, gyro, dts, ba, bg, need_cov=False)
            self._preint_cache[key] = hit
        return hit

    def _v0_from_vel_scale(self, samples, g_rel, device):
        """Vision-IMU v0 init (`init_v0_pose_source="vel_scale"`): all-pairs
        velocity-difference joint least-squares.

        Vision velocities up to scale (frame-0-local):
            v_bar_i = (p_vis_{i+1} - p_vis_i) / dt_i ,   Dv_bar_ij = v_bar_j - v_bar_i.
        Metric IMU velocity CHANGE per pair (v0 cancels in the difference):
            Dv_imu_ij = v_j - v_i = g*(t_j - t_i) + sum_{i<k<=j} R_{k-1} delta_v_k.
        A single scale S aligns them, solved jointly over ALL pairs i<j. Ordinary
        least squares (S = <Dvb,Dvi>/<Dvb,Dvb>) is attenuated toward 0 because the
        predictor Dv_bar is noisy (regression dilution), so we use TOTAL least
        squares (errors-in-variables), the geometric-mean scalar
            S = sign(<Dvb,Dvi>) * sqrt( sum||Dv_imu_ij||^2 / sum||Dv_bar_ij||^2 ),
        which is unbiased. The sign is kept from the OLS correlation so a real
        vision/IMU direction disagreement still flips v0. All pairs (not only
        consecutive ones) matter: long-baseline pairs carry Dv above the noise
        floor, while a consecutive Dv is a second difference of position. Then
        v0 = S * v_bar_0.

        v0's direction is the vision finite difference v_bar_0 (the foundation
        model's translation direction). The IMU chain mirrors the target loop in
        `_build_imu_targets`. Returns ``(v0, S)``.
        """
        preintegrator = IMUPreIntegrator(
            acc_density=self.acc_noise_density,
            gyro_density=self.gyro_noise_density,
        )
        N = len(samples)
        vis_t = [self._poses[i][:3].to(device) for i in range(N)]

        dt0 = max(float(samples[1]["imu_dts"].sum()), 1e-9)
        v_bar_0 = (vis_t[1] - vis_t[0]) / dt0

        # Per-pair(consecutive) vision velocity v_bar[k] (pair k = frames k,k+1)
        # and metric IMU velocity change dv_imu[k] = g*dt + R_k * delta_v -- same
        # terms and order as the target loop's velocity update.
        v_bar, dv_imu = [], []
        R_i = pp.SO3(torch.tensor([0., 0., 0., 1.], device=device))  # frame-0 -> i
        for i in range(1, N):
            acc_i  = samples[i]["imu_acc"].to(device)
            gyro_i = samples[i]["imu_gyro"].to(device)
            dts_i  = samples[i]["imu_dts"].to(device)
            pre = self._preint_cached(preintegrator, acc_i, gyro_i, dts_i,
                                      self.ba, self.bg)
            dt = dts_i.sum()
            dtf = max(float(dt), 1e-9)
            v_bar.append((vis_t[i] - vis_t[i - 1]) / dtf)
            dv_imu.append(g_rel * dt + R_i * pre["delta_v"])
            R_i = R_i * pre["delta_r"].squeeze(0)

        # All-pairs scale S via total least squares: both Dv_bar and Dv_imu are
        # noisy; for a scalar through the origin TLS is the geometric mean
        # sqrt(<Dvi,Dvi>/<Dvb,Dvb>). The sign comes from the OLS correlation.
        # dv_imu over pair (i,j) is the cumulative sum of dv_imu[i+1..j].
        corr = num_bb = num_ii = 0.0
        for i in range(len(v_bar)):
            acc = torch.zeros(3, device=device)
            for j in range(i + 1, len(v_bar)):
                acc = acc + dv_imu[j]                 # Dv_imu_ij
                dvb = v_bar[j] - v_bar[i]             # Dv_bar_ij
                corr   += float((dvb * acc).sum())    # <Dvb, Dvi>  (sign)
                num_bb += float((dvb * dvb).sum())    # <Dvb, Dvb>
                num_ii += float((acc * acc).sum())    # <Dvi, Dvi>
        if num_bb > 1e-12 and num_ii > 0.0:
            # Fit cosine: how well the vision and IMU velocity-difference directions
            # agree (GT-free; the estimator's own confidence), in [-1, +1].
            fit_cos = corr / float((num_bb * num_ii) ** 0.5)
            # Reject only genuine anti-alignment (fit_cos < -0.5): the vision
            # trajectory then disagrees with the inertial motion and its scale is
            # meaningless, so fall back to S=1 (the foundation model's own scale).
            # Otherwise, including fit_cos near 0, the TLS magnitude is used.
            if fit_cos < -0.5:
                s_hat = 1.0
                print(f"             [vel-scale] vision/IMU anti-aligned "
                      f"(fit_cos={fit_cos:+.3f} < -0.5) — rejecting estimate, "
                      f"S=1.0 (defer to foundation model scale).")
            else:
                s_hat = float((num_ii / num_bb) ** 0.5)   # TLS magnitude (sign +)
        else:
            s_hat = 1.0
            print("             [vel-scale] ill-conditioned (den~0); S=1.0")

        if self._v0_mag_mode == "robust_allpairs" and len(v_bar) >= 2:
            # DIRECTION: the gravity-free first-pair v_bar_0. MAGNITUDE: rather
            # than the single noisy first pair, recover ‖v0‖ from every pair. The metric vision
            # velocity at pair k is S·v_bar[k]; the IMU says velocity gained
            # since frame 0 is Σ_{0<m≤k} dv_imu[m], so v0 ≈ S·v_bar[k] − that
            # cumulative Δv. Project each estimate onto v0's direction and take
            # the median.
            v0_dir = v_bar_0 / (v_bar_0.norm() + 1e-9)
            acc_dv = torch.zeros(3, device=device)
            mags = []
            for k in range(len(v_bar)):
                acc_dv = acc_dv + dv_imu[k]              # Σ dv_imu[0..k]
                v0_k = s_hat * v_bar[k] - acc_dv         # back out to t=0
                mags.append(float((v0_k * v0_dir).sum()))  # component along dir
            mags.sort()
            m = len(mags)
            med = mags[m // 2] if m % 2 else 0.5 * (mags[m // 2 - 1] + mags[m // 2])
            # Sign self-consistency. The back-out subtracts Σ dv_imu = Σ(g·dt + R·Δv);
            # a wrong gravity can flip the back-projected v0_k and drive the median
            # negative, i.e. v0 pointing against the vision direction v0_dir. Keep
            # the all-pairs median only when it agrees in sign with the vision
            # motion (med > 0); otherwise fall back to the gravity-free first-pair
            # estimate S·v_bar_0.
            if med > 0.0:
                v0 = med * v0_dir
                mode_used = f"median (n={m} pairs)"
            else:
                v0 = s_hat * v_bar_0                      # back-out sign-inconsistent
                mode_used = "first_pair (back-out points against vision)"
            # The gravity guard re-runs this solve; only the last message is logged.
            self._v0_mag_msg = (
                f"             [vel-scale] v0 mag robust_allpairs: "
                f"first_pair |v0|={float((s_hat * v_bar_0).norm()):.2f}, "
                f"median={med:.2f} -> using {mode_used}, "
                f"|v0|={float(v0.norm()):.2f} m/s")
        else:
            v0 = s_hat * v_bar_0                          # first_pair fallback
        return v0, s_hat

    def _g_from_ls(self, samples, s_hat, device):
        """All-pairs LS for frame-0 gravity, given a metric scale ``s_hat`` and
        ba=0. The counterpart of ``_v0_from_vel_scale``: with S given, the single
        3-vector g is well constrained by the long-baseline pairs.

        Model (velocity-difference form, same terms as vel-scale):
            S · Δv_bar_ij  =  Δv_imu_ij  =  g·Δt_ij + Σ_{i<k≤j} R_{k-1}·Δv_k
        Rearranged, LINEAR in g:
            g·Δt_ij  =  S·Δv_bar_ij − Σ R·Δv_k    ≡  b_ij
        Stacking all pairs i<j: minimise Σ‖Δt_ij·g − b_ij‖² → closed form
            g = (Σ Δt_ij·b_ij) / (Σ Δt_ij²).
        The magnitude is then re-locked to ‖GRAVITY_WORLD‖ (the motion model
        recovers the direction; a residual accel bias only scales ‖·‖). It
        subtracts the vision motion (S·Δv_bar) rather than assuming a = −g, so it
        is as reliable as S. ba is held at 0. Returns a [3] gravity-down tensor, or
        zeros if unsolvable.
        """
        preintegrator = IMUPreIntegrator(
            acc_density=self.acc_noise_density,
            gyro_density=self.gyro_noise_density,
        )
        N = len(samples)
        if N < 3:
            return torch.zeros(3, device=device)
        vis_t = [self._poses[i][:3].to(device) for i in range(N)]

        # Per-frame: metric vision velocity S·v_bar[k], the rotation-transported IMU
        # velocity change R·Δv (NO gravity term — that's what we solve for), and dt.
        s_vbar, rdv, dts = [], [], []
        R_i = pp.SO3(torch.tensor([0., 0., 0., 1.], device=device))  # frame-0 -> i
        for i in range(1, N):
            acc_i  = samples[i]["imu_acc"].to(device)
            gyro_i = samples[i]["imu_gyro"].to(device)
            dts_i  = samples[i]["imu_dts"].to(device)
            pre = self._preint_cached(preintegrator, acc_i, gyro_i, dts_i,
                                      self.ba, self.bg)
            dt = dts_i.sum()
            dtf = max(float(dt), 1e-9)
            s_vbar.append(s_hat * (vis_t[i] - vis_t[i - 1]) / dtf)
            rdv.append((R_i * pre["delta_v"]).squeeze(0))
            dts.append(dt)
            R_i = R_i * pre["delta_r"].squeeze(0)

        # All-pairs: b_ij = S·Δv_bar_ij − Σ R·Δv ; solve g = Σ Δt·b / Σ Δt².
        num = torch.zeros(3, device=device)
        den = 0.0
        M = len(s_vbar)
        for i in range(M):
            rdv_acc = torch.zeros(3, device=device)
            dt_acc = 0.0
            for j in range(i + 1, M):
                rdv_acc = rdv_acc + rdv[j]
                dt_acc = dt_acc + float(dts[j])
                dvbar_ij = s_vbar[j] - s_vbar[i]          # S·Δv_bar_ij
                b_ij = dvbar_ij - rdv_acc                 # = g·Δt_ij
                num = num + dt_acc * b_ij
                den = den + dt_acc * dt_acc
        if den <= 1e-12:
            return torch.zeros(3, device=device)
        g_dir = num / den
        norm_g = g_dir.norm() + 1e-12
        g_mag = float(torch.tensor(self.dataset.GRAVITY_WORLD).norm())
        return g_dir * (g_mag / norm_g)

    def _g_s_fixed_point(self, samples, g_init, device, n_iter=5):
        """Alternate (S given g) ⇄ (g given S) to a fixed point. Starts from
        ``g_init``, estimates S via the existing vel-scale solve, re-estimates g via
        ``_g_from_ls``, repeats. Returns ``(g, s_hat, traj)`` where traj is the list
        of per-iter (S, g) for diagnostics. S and g are coupled through different
        projections (S from magnitudes, g from the vector residual), so the
        iteration can move away from its seed rather than just confirming it."""
        g = g_init
        s_hat = 1.0
        traj = []
        for _ in range(n_iter):
            _v0, s_new = self._v0_from_vel_scale(samples, g, device)
            g_new = self._g_from_ls(samples, s_new, device)
            if g_new.norm() < 1e-6:
                break
            converged = False
            if g.norm() > 1e-6:
                cos = float(torch.clamp(
                    (g / g.norm() * (g_new / g_new.norm())).sum(), -1.0, 1.0))
                converged = cos > 0.99999
            g, s_hat = g_new, s_new
            traj.append((s_hat, g.clone()))
            if converged:
                break
        return g, s_hat, traj

    def _resolve_gravity(self, samples, gravity, device):
        """Frame-0-local gravity from GeoCalib on frame 0's image: a [3] tensor,
        gravity-DOWN, frame-0 camera axes, ‖g‖ = ‖gravity‖."""
        g_magnitude = float(gravity.norm())
        if self._geocalib_model is None:
            self._geocalib_model = build_geocalib(device=device)
        return estimate_gravity_from_image(
            samples[0]["image"], samples[0].get("K"),
            self._geocalib_model, g_magnitude=g_magnitude, device=device,
        )

    def _v0_consistency_score(self, samples, g_rel, device):
        """Gravity-quality signal: how the v0 all-pairs back-out under gravity
        ``g_rel`` agrees in sign with the gravity-free vision velocity. Returns the
        median / first-pair ratio: > 0 when the back-out points the way the vision
        moves, ≤ 0 when it points backwards (a wrong gravity flips Σ dv_imu).
        Independent of LS-g, which minimises the gravity residual instead. Same
        math as `_v0_from_vel_scale`, without the guard."""
        pre = IMUPreIntegrator(acc_density=self.acc_noise_density,
                               gyro_density=self.gyro_noise_density)
        N = len(samples)
        vis_t = [self._poses[i][:3].to(device) for i in range(N)]
        dt0 = max(float(samples[1]["imu_dts"].sum()), 1e-9)
        v_bar_0 = (vis_t[1] - vis_t[0]) / dt0
        v_bar, dv_imu = [], []
        R_i = pp.SO3(torch.tensor([0., 0., 0., 1.], device=device))
        for i in range(1, N):
            acc_i  = samples[i]["imu_acc"].to(device)
            gyro_i = samples[i]["imu_gyro"].to(device)
            dts_i  = samples[i]["imu_dts"].to(device)
            p = self._preint_cached(pre, acc_i, gyro_i, dts_i, self.ba, self.bg)
            dt = dts_i.sum()
            v_bar.append((vis_t[i] - vis_t[i - 1]) / max(float(dt), 1e-9))
            dv_imu.append(g_rel * dt + R_i * p["delta_v"])
            R_i = R_i * p["delta_r"].squeeze(0)
        # S under this gravity (magnitude-only TLS, matches the main solver).
        num_bb = num_ii = 0.0
        for i in range(len(v_bar)):
            acc = torch.zeros(3, device=device)
            for j in range(i + 1, len(v_bar)):
                acc = acc + dv_imu[j]
                dvb = v_bar[j] - v_bar[i]
                num_bb += float((dvb * dvb).sum())
                num_ii += float((acc * acc).sum())
        s_hat = float((num_ii / num_bb) ** 0.5) if num_bb > 1e-12 else 1.0
        v0_dir = v_bar_0 / (v_bar_0.norm() + 1e-9)
        acc_dv = torch.zeros(3, device=device)
        mags = []
        for k in range(len(v_bar)):
            acc_dv = acc_dv + dv_imu[k]
            mags.append(float(((s_hat * v_bar[k] - acc_dv) * v0_dir).sum()))
        mags.sort()
        mm = len(mags)
        med = mags[mm // 2] if mm % 2 else 0.5 * (mags[mm // 2 - 1] + mags[mm // 2])
        fp = float((s_hat * v_bar_0).norm())
        return med / fp if fp > 1e-9 else 0.0

    def _gravity_consistency_guard(self, samples, g_geocalib, v0_gc, s_gc, device):
        """Reject a grossly wrong GeoCalib gravity, by angle (independent of the
        scale S, which also absorbs the foundation model's own scale).

        GeoCalib stays the primary source; as an independent, inertial check, the
        g⇄S fixed point (`_g_s_fixed_point`) is run from two
        seeds, the closed-form IMU gravity (gA) and GeoCalib itself (gB). On an
        informative window both converge to the same direction. GeoCalib is
        rejected only when it lies farther from both converged gravities than they
        lie from each other, with a floor of `_g_ls_reject_deg` (default 20°); then
        the IMU-seeded gravity gA and its scale are adopted, unless the v0 sign
        check (`_v0_consistency_score`) favours GeoCalib. Returns (v0, s_hat, g_rel).
        """
        g_mag = float(g_geocalib.norm())
        g_imu = estimate_gravity_in_frame0(
            samples, self.bg, self.ba, g_magnitude=g_mag, device=device)
        if float(g_imu.norm()) < 1e-6:          # IMU gravity unavailable → keep GC
            return v0_gc, s_gc, g_geocalib

        def _ang(a, b):                          # angle between two [3] gravities
            na, nb = a.norm(), b.norm()
            if na < 1e-9 or nb < 1e-9:
                return 180.0
            c = float(torch.clamp((a / na * (b / nb)).sum(), -1.0, 1.0))
            return math.degrees(math.acos(c))

        # Two independent seeds → converged LS-g gravity each.
        gA, sA, _ = self._g_s_fixed_point(samples, g_imu, device)       # IMU-seeded
        gB, sB, _ = self._g_s_fixed_point(samples, g_geocalib, device)  # GC-seeded
        dA, dB = _ang(g_geocalib, gA), _ang(g_geocalib, gB)
        # Consensus outlier rule. The two fixed points' disagreement
        # d_seed = ∠(gA, gB) is the consensus width: how tightly the inertial
        # evidence pins gravity in this window. GeoCalib is the outlier when it sits
        # farther from the consensus than the consensus is wide (floored at
        # `_g_ls_reject_deg`). The raw accelerometer gravity is only a seed, not the
        # reference; the reference is the converged LS-g gravity.
        d_seed = _ang(gA, gB)
        consensus_w = max(d_seed, self._g_ls_reject_deg)
        if min(dA, dB) > consensus_w:
            # Sign safety: never swap TO a gravity whose v0 back-out points backwards.
            c_gc = self._v0_consistency_score(samples, g_geocalib, device)
            c_ls = self._v0_consistency_score(samples, gA, device)
            if c_gc > 0.0 >= c_ls:
                print(f"             [gravity-guard] reject VETOED: GeoCalib "
                      f"{min(dA,dB):.0f}° from LS-g (consensus width {consensus_w:.0f}°) "
                      f"but its v0 back-out is consistent (c_gc={c_gc:+.2f}) while LS-g "
                      f"is not (c_ls={c_ls:+.2f}) — keeping GeoCalib.")
                self._guard_fired = 0.0          # vetoed -> GeoCalib kept
                return v0_gc, s_gc, g_geocalib
            v0_ls, s_ls = self._v0_from_vel_scale(samples, gA, device)  # IMU-seeded
            print(f"             [gravity-guard] GeoCalib rejected as outlier: "
                  f"{min(dA,dB):.1f}° from LS-g > consensus width {consensus_w:.1f}° "
                  f"(seeds agree to {d_seed:.1f}°) — adopting IMU-seeded LS-g "
                  f"(S {s_gc:.2f}→{s_ls:.2f}).")
            self._guard_fired = 1.0              # GeoCalib rejected -> LS-g adopted
            return v0_ls, s_ls, gA
        self._guard_fired = 0.0                  # GeoCalib kept (inside consensus)
        return v0_gc, s_gc, g_geocalib
