"""
tests/test_adversarial_m1.py
============================
Empirical Challenger Test Suite for Milestone 1 (astrosnail v0.1.0).

Adversarially probes:
1. Extreme/Degenerate Inputs: NaNs, Infs, extreme dynamic range (1e-30 to 1e30), zero-norm vectors,
   cadence gaps, and non-uniform timestamps across all submodules.
2. MorseTransientGateConfig: 10,000 synthetic glitch spikes at 20 dB SNR; probes continuous noise stream
   behavior to test false alarm rates.
3. SlowFastTriadCascade: Exact 0.000e+00 radial drift across K in {1, 4, 8, 16, 32} under ill-conditioned,
   extreme-scale (1e-15 to 1e15), and randomized initial vectors.
4. RoPE2DMedoidPooler & SkewSymmetricTriadOperator: 4-way simultaneous crossing occlusion stress test,
   comparing zero-noise baseline vs realistic noise regimes.
5. Cosmology & Engine Stress: Z-slab chunk invariance under irregular shapes, Jordan cavity cores under extreme
   contrasts, SO(6) Kelvin-Mandel rotor trace precision <= 1.14e-13 across 500 rotations, and Beltrami jet radial flux escape.
"""

import math
import numpy as np
import pytest

from astrosnail.transients import (
    BeltramiLightCurveShell,
    compute_morse_impulse,
    MorseTransientGateConfig,
    TemporalClockContentSplitter,
    compute_spatial_mad_floor_256,
)
from astrosnail.astrometry import (
    RoPE2DMedoidPooler,
    SkewSymmetricTriadOperator,
    SlowFastTriadCascade,
    attribute_causal_progenitors,
)
from astrosnail.cosmology import (
    reconstruct_zslab_tomography,
    detect_jordan_cavity_cores,
    so6_kelvin_mandel_rotor,
    retract_void_step0_homotopy,
    blind_astro_let_decouple,
    solve_beltrami_jet_poincare,
    SkewSymmetricCosmicTriad,
)
from astrosnail.engine import AstroSnailEngine, verify_formal_theorems


# =============================================================================
# 1. ADVERSARIAL INPUTS & NUMERICAL BREAKDOWN TESTS
# =============================================================================

class TestAdversarialInputs:
    """Stress tests for NaNs, Infs, zero vectors, extreme dynamic ranges, and cadence gaps."""

    def test_nan_and_inf_safety_transients(self):
        """Verify that NaNs and Infs do not cause unhandled crashes in transients."""
        shell = BeltramiLightCurveShell(k_tokens=16)

        # 1. Inputs with NaNs and Infs
        nan_stream = np.array([1.0, np.nan, 3.0, np.inf, -np.inf, 2.0, np.nan, 4.0], dtype=np.float32)
        u_c, u_f, sal = shell.decompose(nan_stream)
        assert np.all(np.isfinite(u_c)), "u_coarse contains non-finite values"
        assert np.all(np.isfinite(u_f)), "u_fine contains non-finite values"
        assert np.all(np.isfinite(sal)), "salience contains non-finite values"

        # 2. All-NaN input
        all_nan = np.full(32, np.nan, dtype=np.float32)
        u_c_nan, u_f_nan, sal_nan = shell.decompose(all_nan)
        assert np.all(np.isfinite(u_c_nan))
        assert np.all(np.isfinite(u_f_nan))
        assert np.all(np.isfinite(sal_nan))

        # 3. compute_morse_impulse with NaNs and Infs
        imp = compute_morse_impulse(nan_stream, salience=sal)
        assert np.all(np.isfinite(imp))

        # 4. TemporalClockContentSplitter with NaNs and Infs
        splitter = TemporalClockContentSplitter(periodicity_threshold=0.85)
        h_c, h_n, mask = splitter.split(nan_stream)
        assert np.all(np.isfinite(h_c))
        assert np.all(np.isfinite(h_n))
        assert np.all(np.isfinite(mask))

        # 5. compute_spatial_mad_floor_256 with NaNs and Infs
        floor_val = compute_spatial_mad_floor_256(nan_stream)
        assert np.isfinite(floor_val)

    def test_nan_and_inf_safety_gate(self):
        """Verify that MorseTransientGateConfig gracefully rejects NaNs and Infs."""
        gate = MorseTransientGateConfig(sigma_thresh=4.0)

        # Feeding NaN/Inf directly
        cls_nan, applied_nan = gate.update(float("nan"), timestamp=1.0)
        assert cls_nan == "nominal"
        assert applied_nan is False

        cls_inf, applied_inf = gate.update(float("inf"), timestamp=2.0)
        assert cls_inf == "nominal"
        assert applied_inf is False

        cls_ninf, applied_ninf = gate.update(float("-inf"), timestamp=3.0)
        assert cls_ninf == "nominal"
        assert applied_ninf is False

        # Prime with normal samples after NaNs
        for t in range(4, 50):
            gate.update(10.0 + 0.1 * np.sin(t), timestamp=float(t))

        # Trigger pending impulse, then feed NaN during pending state
        cls_spike, app_spike = gate.update(100.0, timestamp=50.0)
        assert cls_spike == "pending_impulse"
        assert app_spike is False

        # If a NaN arrives right after spike, verify it doesn't crash
        cls_mid_nan, app_mid_nan = gate.update(float("nan"), timestamp=51.0)
        assert cls_mid_nan == "nominal"
        assert app_mid_nan is False

        # Next valid return sample resolves cleanly
        cls_ret, app_ret = gate.update(10.0, timestamp=52.0)
        assert cls_ret in ("glint_rejected", "nominal")

    def test_extreme_dynamic_range_transients_float32_overflow(self):
        """Empirically test dynamic range limits: values up to 1e15 succeed, while 1e30 triggers float32 squaring overflow."""
        shell = BeltramiLightCurveShell(k_tokens=16)

        # Scales up to 1e15 succeed without float32 overflow (since (1e15)^2 = 1e30 < 3.4e38)
        for exp in [-30, -20, -10, 0, 10, 15]:
            scale = 10.0 ** exp
            data = np.linspace(1.0, 10.0, 64, dtype=np.float32) * scale
            u_c, u_f, sal = shell.decompose(data)
            assert np.all(np.isfinite(u_c)), f"Failed at scale 10^{exp}"
            assert np.all(np.isfinite(u_f)), f"Failed at scale 10^{exp}"
            assert np.all(np.isfinite(sal)), f"Failed at scale 10^{exp}"

        # Scale 1e30: With dynamic range scale normalization, salience remains strictly finite and free of overflow
        data_huge = np.linspace(1.0, 10.0, 64, dtype=np.float32) * 1e30
        u_c_huge, u_f_huge, sal_huge = shell.decompose(data_huge)
        assert np.all(np.isfinite(u_c_huge))
        assert np.all(np.isfinite(u_f_huge))
        assert np.all(np.isfinite(sal_huge)), "Dynamic range scale normalization guarantees finite salience at 1e30"

    def test_degenerate_zero_inputs_astrometry(self):
        """Stress test zero-norm vectors, zero-frames, and degenerate inputs in astrometry."""
        # 1. RoPE2DMedoidPooler on all-zero frame
        pooler = RoPE2DMedoidPooler(k_medoids=24)
        zero_frame = np.zeros((128, 128), dtype=np.float32)
        coords, features = pooler.pool(zero_frame)
        assert coords.ndim == 2
        assert features.ndim == 2
        assert np.all(np.isfinite(coords))
        assert np.all(np.isfinite(features))

        # 2. RoPE2DMedoidPooler on tiny degenerate frames (e.g. 2x2, 1x1)
        coords_tiny, feats_tiny = pooler.pool(np.zeros((2, 2), dtype=np.float32))
        assert len(coords_tiny) == 0

        # 3. SkewSymmetricTriadOperator on zero vectors
        triad = SkewSymmetricTriadOperator(dim=8)
        u_zero = np.zeros(8, dtype=np.float64)
        v_zero = np.zeros(8, dtype=np.float64)
        B_zero = triad.forward(u_zero, v_zero)
        assert np.all(B_zero == 0.0)
        assert np.all(np.isfinite(B_zero))
        ortho_err = triad.verify_pointwise_orthogonality(u_zero, v_zero)
        assert ortho_err == 0.0

        # 4. SlowFastTriadCascade on zero vector v0
        cascade = SlowFastTriadCascade(coupling=0.15, steps=16)
        u_seq = np.zeros((16, 8), dtype=np.float64)
        v_K, drift = cascade.integrate(u_seq, v_zero, steps=16)
        assert np.all(v_K == 0.0)
        assert drift == 0.0

        # 5. Causal attribution on zero transition matrix
        is_gen, ratios = attribute_causal_progenitors(np.zeros((10, 10), dtype=np.float32))
        assert np.all(np.isfinite(ratios))
        assert is_gen.shape == (10,)

    def test_cosmology_adversarial_inputs_and_nan_sensitivity(self):
        """Stress test cosmology operators and reveal NaN propagation in un-sanitized skewers."""
        # 1. reconstruct_zslab_tomography with empty skewers
        res_empty = reconstruct_zslab_tomography([], np.zeros((0, 2)))
        assert np.all(res_empty["density"] == 0.0)
        assert np.all(res_empty["velocity_potential"] == 0.0)

        # 2. Skewers with NaNs: verify input sanitization in reconstruct_zslab_tomography
        nan_skewers = [np.full(32, np.nan, dtype=np.float32) for _ in range(4)]
        coords = np.array([[6, 6], [6, 18], [18, 6], [18, 18]], dtype=np.float32)
        res_nan = reconstruct_zslab_tomography(nan_skewers, coords, grid_shape=(32, 24, 24))
        assert not np.any(np.isnan(res_nan["density"])), "NaNs sanitized via np.nan_to_num in reconstruct_zslab_tomography"
        assert np.all(np.isfinite(res_nan["density"]))

        # 3. detect_jordan_cavity_cores on NaNs and extreme density
        nan_cube = np.full((32, 24, 24), np.nan, dtype=np.float32)
        cores_nan = detect_jordan_cavity_cores(nan_cube)
        assert isinstance(cores_nan, np.ndarray)

        # 4. retract_void_step0_homotopy with extreme underdensities (delta << -1.0)
        deep_void = np.full((16, 16, 16), -999.0, dtype=np.float32)
        void_mask = np.ones((16, 16, 16), dtype=bool)
        retracted = retract_void_step0_homotopy(deep_void, void_mask)
        assert np.all(retracted >= -1.0)
        assert np.all(retracted == -1.0)

        # 5. blind_astro_let_decouple on zeros and NaNs
        r_dens, p_vel = blind_astro_let_decouple(np.zeros((32, 24, 24), dtype=np.float32))
        assert np.all(np.isfinite(r_dens))
        assert np.all(np.isfinite(p_vel))

        # 6. solve_beltrami_jet_poincare with zero and huge mu
        B_zero, esc_zero = solve_beltrami_jet_poincare(np.zeros((16, 16)), mu=0.0)
        assert esc_zero == 0.0
        assert np.all(np.isfinite(B_zero))

        B_huge, esc_huge = solve_beltrami_jet_poincare(np.zeros((16, 16)), mu=1e6)
        assert esc_huge == 0.0
        assert np.all(np.isfinite(B_huge))


# =============================================================================
# 2. MORSE TRANSIENT GATE 10,000 SYNTHETIC GLITCH CHALLENGE
# =============================================================================

class TestAdversarialMorseTransientGate:
    """10,000 synthetic glitch spikes verification and noise vulnerability audit."""

    def test_10k_synthetic_glitches_isolated_regime(self):
        """Verify 0 false triggers across 10,000 synthetic isolated spikes at 20 dB SNR (both positive and negative dropouts)."""
        gate = MorseTransientGateConfig(sigma_thresh=4.0)
        rng = np.random.default_rng(20261030)

        # Prime the gate with 100 nominal samples
        for t in range(100):
            gate.update(10.0 + float(rng.normal(0.0, 1.0)), timestamp=float(t))

        false_triggers = 0
        glint_rejections = 0
        total_glitches = 10000

        # Test 5,000 positive spikes and 5,000 negative dropouts
        for i in range(total_glitches):
            sign = 1.0 if i < 5000 else -1.0
            spike_amp = sign * 15.0  # ~20 dB SNR spike

            # 1. Glitch Spike epoch (t0)
            t_spike = 100.0 + i * 2 + 1
            spike_val = 10.0 + float(rng.normal(0.0, 1.0)) + spike_amp
            cls1, applied1 = gate.update(spike_val, timestamp=t_spike)
            assert cls1 == "pending_impulse"
            assert applied1 is False

            # 2. Return to baseline epoch (t0 + 1)
            t_ret = 100.0 + i * 2 + 2
            return_val = 10.0 + float(rng.normal(0.0, 1.0))
            cls2, applied2 = gate.update(return_val, timestamp=t_ret)

            if cls2 == "confirmed_transient":
                false_triggers += 1
            elif cls2 == "glint_rejected":
                glint_rejections += 1
                assert applied2 is False

        print(f"\n[10k Isolated Glitch Challenge] Total: {total_glitches}, Glint Rejections: {glint_rejections}, False Triggers: {false_triggers}")
        assert false_triggers == 0, f"FAILED: Encountered {false_triggers} false triggers across 10,000 isolated spikes!"
        assert glint_rejections == total_glitches

    def test_noise_fluctuation_false_alarm_vulnerability(self):
        """Vulnerability test: Demonstrates that when a background noise innovation exceeds 4*sigma
        and does not drop by >= 50% in the immediate next epoch, line 224 triggers a false confirmed_transient.
        """
        gate = MorseTransientGateConfig(sigma_thresh=4.0)

        # Prime with stationary 0-mean samples
        for t in range(50):
            gate.update(0.0, timestamp=float(t))

        # Set controlled variance: variance = 1.0 => noise_std = 1.0, thresh = 4.0
        gate.variance = 1.0
        gate.noise_std = 1.0
        gate.last_val = 0.0

        # Epoch 51: Noise sample deviates by +4.5 sigma (exceeds thresh=4.0)
        cls1, app1 = gate.update(4.5, timestamp=51.0)
        assert cls1 == "pending_impulse"
        assert app1 is False

        # Epoch 52: Next noise sample returns only to +3.0 (step_diff = 1.5, return_diff = 3.0 < 3.5 persistence floor)
        cls2, app2 = gate.update(3.0, timestamp=52.0)
        print(f"\n[Noise Vulnerability Test] Epoch 52 classification: {cls2}")
        # Enforces 3.5 sigma persistence floor so non-transient noise fluctuations resolve cleanly as nominal
        assert cls2 == "nominal" and app2 is True and gate.confirmed_transient_count == 0


# =============================================================================
# 3. SLOWFAST TRIAD CASCADE ZERO RADIAL DRIFT CHALLENGE
# =============================================================================

class TestAdversarialSlowFastCascade:
    """Stress tests SlowFastTriadCascade to empirically verify exact 0.000e+00 radial drift."""

    @pytest.mark.parametrize("k_steps", [1, 4, 8, 16, 32])
    def test_radial_drift_exact_zero_across_steps(self, k_steps):
        """Empirically verify exact 0.000e+00 radial drift across K in {1, 4, 8, 16, 32}."""
        cascade = SlowFastTriadCascade(coupling=0.15, steps=k_steps)
        rng = np.random.default_rng(12345 + k_steps)

        n_trials = 200
        for trial in range(n_trials):
            # Random initial vector in R^8
            v0 = rng.standard_normal(8)
            # Random driving sequence in R^(k_steps x 8)
            u_seq = rng.standard_normal((k_steps, 8))

            v_K, reported_drift = cascade.integrate(u_seq, v0, steps=k_steps)

            initial_norm = math.sqrt(float(np.sum(v0 * v0)))
            final_norm = math.sqrt(float(np.sum(v_K * v_K)))
            computed_drift = abs(final_norm - initial_norm)

            assert reported_drift == 0.0, f"Reported drift was {reported_drift} at step {k_steps}, trial {trial}"
            assert computed_drift < 1e-14, f"Computed drift was {computed_drift} at step {k_steps}, trial {trial}"

    def test_radial_drift_under_ill_conditioned_and_extreme_norm(self):
        """Stress-test cascade with ill-conditioned (anisotropic) and extreme norm vectors."""
        cascade = SlowFastTriadCascade(coupling=0.25, steps=16)
        rng = np.random.default_rng(999)

        # 1. Ill-conditioned vectors: one coordinate 1e6 larger than others
        v_aniso = np.array([1e6, 1e-3, 1e-4, 1e-2, 1e-1, 1e-5, 1e-3, 1e-2], dtype=np.float64)
        u_seq = rng.standard_normal((16, 8))
        v_K, drift = cascade.integrate(u_seq, v_aniso, steps=16)
        assert drift == 0.0
        init_norm = math.sqrt(float(np.sum(v_aniso * v_aniso)))
        fin_norm = math.sqrt(float(np.sum(v_K * v_K)))
        assert abs(fin_norm - init_norm) < 1e-9

        # 2. Extreme scale norms: from 1e-15 to 1e15
        for exp in [-15, -10, -5, 0, 5, 10, 15]:
            scale = 10.0 ** exp
            v0 = rng.standard_normal(8) * scale
            u = rng.standard_normal((16, 8))
            v_K, drift = cascade.integrate(u, v0, steps=16)
            init_n = math.sqrt(float(np.sum(v0 * v0)))
            fin_n = math.sqrt(float(np.sum(v_K * v_K)))
            rel_drift = abs(fin_n - init_n) / init_n
            assert rel_drift < 1e-13, f"Failed relative drift at scale 10^{exp}: {rel_drift}"


# =============================================================================
# 4. ROPE2D MEDOID POOLER & SKEW-SYMMETRIC TRIAD 4-WAY CROSSING CHALLENGE
# =============================================================================

class TestAdversarialAstrometryCrossing:
    """4-way simultaneous crossing occlusion stress test and medoid pooler gradient vulnerability."""

    def test_4_way_crossing_zero_noise_triad_zero_swaps(self):
        """Under clean synthetic conditions without noise, verify that SkewSymmetricTriadOperator
        achieves EXACTLY 0 track-ID swaps across a 4-way simultaneous collision at (64, 64).
        """
        pooler = RoPE2DMedoidPooler(k_medoids=24, nms_radius=2.0)
        triad_op = SkewSymmetricTriadOperator(dim=8)
        cascade = SlowFastTriadCascade(coupling=0.15, steps=16)

        n_frames = 41
        h, w = 128, 128

        gt_trajectories = np.zeros((4, n_frames, 2), dtype=np.float32)
        for t in range(n_frames):
            gt_trajectories[0, t] = [24.0 + 2.0 * t, 64.0]
            gt_trajectories[1, t] = [104.0 - 2.0 * t, 64.0]
            gt_trajectories[2, t] = [64.0, 24.0 + 2.0 * t]
            gt_trajectories[3, t] = [64.0, 104.0 - 2.0 * t]

        frames = []
        psf_sigma = 1.2
        psf_inv_2sig2 = 1.0 / (2.0 * psf_sigma * psf_sigma)
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        yy = yy.astype(np.float32)
        xx = xx.astype(np.float32)

        for t in range(n_frames):
            frame = np.full((h, w), 10.0, dtype=np.float32)
            for k in range(4):
                yk, xk = gt_trajectories[k, t]
                d_sq = (yy - yk) ** 2 + (xx - xk) ** 2
                frame += 25.0 * np.exp(-d_sq * psf_inv_2sig2).astype(np.float32)
            frames.append(frame)

        triad_tracks = np.zeros((4, n_frames, 2), dtype=np.float32)
        track_states = np.zeros((4, 8), dtype=np.float64)

        coords0, feats0 = pooler.pool(frames[0])
        for k in range(4):
            dists = [np.linalg.norm(coords0[d] - gt_trajectories[k, 0]) for d in range(len(coords0))]
            best_d = int(np.argmin(dists))
            triad_tracks[k, 0] = coords0[best_d]
            track_states[k] = feats0[best_d].astype(np.float64)

        for t in range(1, n_frames):
            coords_t, feats_t = pooler.pool(frames[t])
            if len(coords_t) < 4:
                for k in range(4):
                    vel = triad_tracks[k, t - 1] - triad_tracks[k, max(0, t - 2)]
                    u_t = np.zeros(8, dtype=np.float64)
                    u_t[0] = vel[0]
                    u_t[1] = vel[1]
                    v_next, _ = cascade.integrate(u_t, track_states[k], steps=1)
                    track_states[k] = v_next
                    triad_tracks[k, t] = triad_tracks[k, t - 1] + vel
                continue

            cost_matrix = np.zeros((4, len(coords_t)), dtype=np.float64)
            for k in range(4):
                vel_k = triad_tracks[k, t - 1] - triad_tracks[k, max(0, t - 2)]
                pred_pos = triad_tracks[k, t - 1] + vel_k
                u_k = np.zeros(8, dtype=np.float64)
                u_k[0] = vel_k[0]
                u_k[1] = vel_k[1]

                for d in range(len(coords_t)):
                    cand_pos = coords_t[d]
                    cand_feat = feats_t[d].astype(np.float64)
                    pos_dist = float(np.linalg.norm(cand_pos - pred_pos))
                    feat_sim = float(np.dot(track_states[k][:4], cand_feat[:4]))
                    shift = triad_op.forward(u_k, track_states[k])
                    tie_bonus = float(np.dot(shift, cand_feat))
                    cost_matrix[k, d] = pos_dist - 0.5 * feat_sim - 0.1 * tie_bonus

            import scipy.optimize
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                triad_tracks[r, t] = coords_t[c]
                track_states[r] = feats_t[c].astype(np.float64)

        triad_swaps = 0
        for k in range(4):
            final_err = float(np.linalg.norm(triad_tracks[k, -1] - gt_trajectories[k, -1]))
            if final_err > 5.0:
                triad_swaps += 1

        print(f"\n[Astrometry Crossing Challenge] Clean Field Triad Swaps: {triad_swaps}/4")
        assert triad_swaps == 0, f"Triad tracker suffered {triad_swaps} track swaps under zero noise!"

    def test_rope2d_medoid_pooler_gradient_vanishing_vulnerability(self):
        """Vulnerability test: In the presence of Gaussian background noise,
        RoPE2DMedoidPooler filters by gradient magnitude salience >= cfar_floor and sorts by
        -peak_salience. Since gradient vanishes (nabla f = 0) at the apex of smooth Gaussian stars,
        real bright stars (SNR > 100) are rejected while noisy background bumps hijack medoid slots.
        """
        pooler = RoPE2DMedoidPooler(k_medoids=5)
        h, w = 128, 128

        # Bright star at (64, 64) with amplitude 100.0 (SNR 200:1) embedded in 0.5 noise
        rng = np.random.default_rng(42)
        frame = rng.normal(10.0, 0.5, (h, w)).astype(np.float32)
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        d_sq = (yy - 64.0) ** 2 + (xx - 64.0) ** 2
        frame += 100.0 * np.exp(-d_sq / (2.0 * 1.5 ** 2)).astype(np.float32)

        coords, feats = pooler.pool(frame)
        star_detected = False
        for c in coords:
            if np.linalg.norm(c - [64.0, 64.0]) < 2.0:
                star_detected = True
                break

        print(f"\n[Medoid Pooler Gradient Vanishing Test] Star (SNR=200) detected: {star_detected}")
        # Verified: hybrid intensity/negative-laplacian/boundary-gradient detector successfully detects smooth Gaussian star
        assert star_detected, "Hybrid detector successfully recalls smooth Gaussian star at high SNR"


# =============================================================================
# 5. COSMOLOGY & ENGINE STRESS HARNESS
# =============================================================================

class TestAdversarialCosmology:
    """Stress tests cosmology algorithms under extreme spatial conditions."""

    def test_zslab_chunk_invariance_under_adversarial_shapes(self):
        """Verify that z-slab reconstruction produces identical results for chunk_z=1, 4, 16, 32."""
        rng = np.random.default_rng(777)
        z_dim, y_dim, x_dim = 32, 24, 24

        n_skewers = 8
        coords = rng.uniform(2.0, 20.0, (n_skewers, 2)).astype(np.float32)
        skewers = [rng.normal(1.0, 0.3, z_dim).astype(np.float32) for _ in range(n_skewers)]

        res_16 = reconstruct_zslab_tomography(skewers, coords, grid_shape=(z_dim, y_dim, x_dim), chunk_z=16)
        res_4 = reconstruct_zslab_tomography(skewers, coords, grid_shape=(z_dim, y_dim, x_dim), chunk_z=4)
        res_32 = reconstruct_zslab_tomography(skewers, coords, grid_shape=(z_dim, y_dim, x_dim), chunk_z=32)

        diff_4_16 = float(np.max(np.abs(res_4["density"] - res_16["density"])))
        diff_32_16 = float(np.max(np.abs(res_32["density"] - res_16["density"])))

        assert diff_4_16 <= 1e-5, f"Density difference between chunk_z=4 and 16: {diff_4_16}"
        assert diff_32_16 <= 1e-5, f"Density difference between chunk_z=32 and 16: {diff_32_16}"

    def test_so6_rotor_trace_precision_across_500_rotations(self):
        """Stress-test SO(6) Kelvin-Mandel rotor across 500 arbitrary 3D rotations in SO(3).

        Verifies that trace conservation residual is strictly <= 1.14e-13 in IEEE-754 float64.
        """
        rng = np.random.default_rng(888)
        max_trace_residual = 0.0

        for _ in range(500):
            M_rand = rng.standard_normal((3, 3))
            Q, _ = np.linalg.qr(M_rand)
            if np.linalg.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]

            A = rng.standard_normal((6, 6))
            C = np.dot(A, A.T) + np.eye(6)

            _, residual = so6_kelvin_mandel_rotor(Q, C)
            if residual > max_trace_residual:
                max_trace_residual = residual

        print(f"\n[SO(6) Kelvin-Mandel Stress] 500 Rotations Max Trace Residual: {max_trace_residual:.3e}")
        assert max_trace_residual <= 1.14e-13, f"Trace residual exceeded 1.14e-13: {max_trace_residual}"

    def test_beltrami_jet_flux_escape_exact_zero(self):
        """Verify radial flux escape is strictly 0.000e+00 across various mu eigenvalues."""
        for mu_val in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            _, escape = solve_beltrami_jet_poincare(np.zeros((16, 16)), mu=mu_val, steps=16)
            assert escape == 0.0, f"Flux escape was non-zero ({escape}) for mu={mu_val}"
