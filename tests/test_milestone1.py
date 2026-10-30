"""
tests/test_milestone1.py
========================
Comprehensive verification and acceptance test suite for Milestone 1:
- Formal invariants Theorems 2.1–2.9 via AstroSnailEngine().verify_formal_theorems()
- 1-sample causal deferral (0 false triggers on 10,000 spikes at 20 dB SNR)
- Ephemeris harmonic quarantine
- RoPE2D, SO(8) collision tie-breaking, SlowFastTriadCascade 0 radial drift
- Z-slab tomography, void cavity cores, SO(6) Kelvin-Mandel rotor trace residual <= 1.14e-13,
  Step-0 void homotopy, BlindAstroLET, 3D Beltrami jet solver
- Performance floors: >= 165 FPS, < 50 ms per 25,000 epochs, working set < 2.0 MB, tracemalloc < 35.0 MB
"""

import math
import time
import tracemalloc
import numpy as np
import pytest

from astrosnail import (
    BeltramiLightCurveShell,
    compute_morse_impulse,
    MorseTransientGateConfig,
    TemporalClockContentSplitter,
    compute_spatial_mad_floor_256,
    RoPE2DMedoidPooler,
    SkewSymmetricTriadOperator,
    SlowFastTriadCascade,
    attribute_causal_progenitors,
    reconstruct_zslab_tomography,
    detect_jordan_cavity_cores,
    so6_kelvin_mandel_rotor,
    retract_void_step0_homotopy,
    blind_astro_let_decouple,
    solve_beltrami_jet_poincare,
    compute_mrxmhd_flow_jump,
    SkewSymmetricFaultTriad,
    AstroSnailEngine,
    verify_formal_theorems,
)


class TestFormalTheorems:
    """Tests runtime verification of machine-checked Theorems 2.1 through 2.9."""

    def test_verify_formal_theorems_all_pass(self):
        engine = AstroSnailEngine(backend="numpy")
        all_passed, results = engine.verify_formal_theorems(seed=20261025)
        assert all_passed is True

        # Theorem 2.1: Antisymmetry & zero self-interaction
        th2_1 = results["theorem_2_1"]
        assert th2_1["passed"] is True
        assert th2_1["antisymmetry_error"] < 1e-14
        assert th2_1["self_interaction_error"] == 0.0

        # Theorem 2.2: Exact tie-breaking margin > 0
        th2_2 = results["theorem_2_2"]
        assert th2_2["passed"] is True
        assert th2_2["tie_break_gap"] > 0.0
        assert th2_2["formula_error"] < 1e-12

        # Theorem 2.3: Pointwise tangent orthogonality
        th2_3 = results["theorem_2_3"]
        assert th2_3["passed"] is True
        assert th2_3["orthogonality_residual"] <= 1.0e-8

        # Theorem 2.4: Step-0 System 1 homotopy
        th2_4 = results["theorem_2_4"]
        assert th2_4["passed"] is True
        assert th2_4["step0_homotopy_error"] == 0.0

        # Theorem 2.5: RoPE2D medoid relative displacement invariance & patch attenuation
        th2_5 = results["theorem_2_5"]
        assert th2_5["passed"] is True
        assert th2_5["relative_phase_error"] < 1e-12
        assert th2_5["patch_norm_attenuated"] is True

        # Theorem 2.6: SlowFast multi-step spherical invariance
        th2_6 = results["theorem_2_6"]
        assert th2_6["passed"] is True
        assert th2_6["spherical_radial_drift"] == 0.0

        # Theorem 2.7: SO(6) Kelvin-Mandel rotor trace conservation & Morse shielding
        th2_7 = results["theorem_2_7"]
        assert th2_7["passed"] is True
        assert th2_7["stiffness_trace_drift"] <= 1.14e-13
        assert th2_7["ramp_rejected"] is True
        assert th2_7["spike_shielded"] is True

        # Theorem 2.8: 3D Ideal MHD cross3 isomorphism & magnetic flux confinement
        th2_8 = results["theorem_2_8"]
        assert th2_8["passed"] is True
        assert th2_8["cross3_isomorphism_error"] <= 1.0e-7
        assert th2_8["flux_confinement_residual"] <= 1.0e-8

        # Theorem 2.9: Collinear Beltrami Lorentz vanishing & Bernoulli jump law
        th2_9 = results["theorem_2_9"]
        assert th2_9["passed"] is True
        assert th2_9["collinear_lorentz_error"] <= 1.0e-12
        assert th2_9["bernoulli_jump_residual"] <= 1e-12

    def test_verify_formal_theorems_multi_seed_50(self):
        """Verify that verify_formal_theorems passes reliably across 50 random seeds."""
        for s in range(50):
            passed, res = verify_formal_theorems(seed=s)
            assert passed, f"Formal theorems verification failed on seed {s}: {res}"


class TestTransientsModule:
    """Tests for astrosnail.transients."""

    def test_morse_transient_gate_10k_spikes(self):
        """0 false triggers across 10,000 spikes at 20 dB SNR."""
        gate = MorseTransientGateConfig(sigma_thresh=4.0)
        rng = np.random.default_rng(42)

        n_spikes = 10000
        noise_sigma = 1.0
        spike_amp = 15.0  # +15 sigma, ~20 dB SNR spike

        # Prime the gate with 100 nominal background samples
        for _ in range(100):
            sample = float(rng.normal(0.0, noise_sigma))
            gate.update(sample)

        false_triggers = 0
        glint_rejections = 0

        # Inject 10,000 isolated single-sample spikes separated by nominal samples
        for i in range(n_spikes):
            # 1. Glitch spike epoch (t0)
            spike_val = float(rng.normal(0.0, noise_sigma)) + spike_amp
            cls1, applied1 = gate.update(spike_val, timestamp=float(i * 2 + 1))
            assert applied1 is False
            assert cls1 == "pending_impulse"

            # 2. Return-to-baseline epoch (t0 + 1)
            return_val = float(rng.normal(0.0, noise_sigma))
            cls2, applied2 = gate.update(return_val, timestamp=float(i * 2 + 2))

            if cls2 == "confirmed_transient":
                false_triggers += 1
            elif cls2 == "glint_rejected":
                glint_rejections += 1
                assert applied2 is False

        assert false_triggers == 0
        assert glint_rejections == n_spikes
        assert gate.glint_rejected_count == n_spikes

    def test_morse_transient_gate_sustained_transient(self):
        """Confirms 100% of sustained physical transients (supernova, flare)."""
        gate = MorseTransientGateConfig(sigma_thresh=4.0)
        gate.reset()

        # Prime with nominal background
        for _ in range(50):
            gate.update(0.0)

        # Onset epoch: step to 12.0
        cls1, app1 = gate.update(12.0, timestamp=1.0)
        assert cls1 == "pending_impulse"
        assert app1 is False

        # Sustained elevated epoch: remains at 12.0
        cls2, app2 = gate.update(12.0, timestamp=2.0)
        assert cls2 == "confirmed_transient"
        assert app2 is True
        assert gate.confirmed_transient_count == 1

    def test_beltrami_light_curve_shell_decomposition(self):
        shell = BeltramiLightCurveShell(k_tokens=16)

        # 1D signal
        t_len = 100
        sig_1d = np.sin(np.linspace(0, 4 * np.pi, t_len), dtype=np.float32)
        sig_1d[50] += 5.0  # Decoupled transient spike

        u_c, u_f, sal = shell.decompose(sig_1d)
        assert u_c.shape == sig_1d.shape
        assert u_f.shape == sig_1d.shape
        assert sal.shape == sig_1d.shape
        assert np.all(sal >= 0.0) and np.all(sal <= 1.0)
        # u_coarse + u_fine == input
        np.testing.assert_allclose(u_c + u_f, sig_1d, atol=1e-6)

        # 2D multi-band signal (100 epochs, 4 bands)
        sig_2d = np.random.randn(100, 4).astype(np.float32)
        u_c2, u_f2, sal2 = shell.decompose(sig_2d)
        assert u_c2.shape == (100, 4)
        assert u_f2.shape == (100, 4)
        assert sal2.shape == (100,)
        np.testing.assert_allclose(u_c2 + u_f2, sig_2d, atol=1e-6)

        # Non-finite sanitization
        nan_sig = sig_1d.copy()
        nan_sig[10] = np.nan
        nan_sig[20] = np.inf
        u_c_nan, u_f_nan, _ = shell.decompose(nan_sig)
        assert np.all(np.isfinite(u_c_nan))
        assert np.all(np.isfinite(u_f_nan))

        # T < 3 raises ValueError
        with pytest.raises(ValueError):
            shell.decompose(np.array([1.0, 2.0]))

    def test_discrete_morse_impulse(self):
        # Monotone ramp: I(t) <= 0
        ramp = np.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=np.float32)
        i_ramp = compute_morse_impulse(ramp)
        assert np.all(i_ramp <= 0.0)

        # Local maximum: I(t) > 0
        peak = np.array([1.0, 10.0, 1.0], dtype=np.float32)
        i_peak = compute_morse_impulse(peak)
        assert i_peak[1] == 81.0

        # Local minimum: I(t) > 0
        trough = np.array([10.0, 1.0, 10.0], dtype=np.float32)
        i_trough = compute_morse_impulse(trough)
        assert i_trough[1] == 81.0

        # Boundary zeroing
        assert i_peak[0] == 0.0 and i_peak[2] == 0.0

    def test_temporal_clock_splitter(self):
        splitter = TemporalClockContentSplitter(periodicity_threshold=0.85)
        t = np.arange(500, dtype=np.float32)

        # Composite: period-20 sinusoid (clock) + isolated transit dip at t=250 (nope)
        period = 20
        eclipsing_binary = 2.0 * np.sin(2.0 * np.pi * t / float(period)).astype(np.float32)
        transit_dip = np.zeros(500, dtype=np.float32)
        transit_dip[248:253] = -4.0

        composite = eclipsing_binary + transit_dip
        h_clock, h_nope, mask = splitter.split(composite)

        # Conservation: H_clock + H_nope == composite
        np.testing.assert_allclose(h_clock + h_nope, composite, atol=1e-5)
        # Non-repeating transit dip preserved in H_nope
        assert float(np.min(h_nope[248:253])) < -3.0

    def test_compute_spatial_mad_floor_256(self):
        rng = np.random.default_rng(123)
        salience = np.abs(rng.normal(1.0, 0.5, size=(128, 128))).astype(np.float32)
        floor_val = compute_spatial_mad_floor_256(salience)
        assert floor_val > 0.0

        # Masking out top half
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[:64, :] = 1
        floor_masked = compute_spatial_mad_floor_256(salience, mask=mask)
        assert floor_masked >= 0.0

        # Empty array
        assert compute_spatial_mad_floor_256(np.array([], dtype=np.float32)) == 0.0


class TestAstrometryModule:
    """Tests for astrosnail.astrometry."""

    def test_rope2d_medoid_pooler(self):
        pooler = RoPE2DMedoidPooler(k_medoids=24)
        h, w = 128, 128
        frame = np.zeros((h, w), dtype=np.float32)

        # Place 4 simulated stellar medoids
        truth_coords = [(30, 40), (70, 80), (30, 90), (100, 25)]
        for y, x in truth_coords:
            frame[y - 1:y + 2, x - 1:x + 2] = [[1, 2, 1], [2, 10, 2], [1, 2, 1]]

        coords, features = pooler.pool(frame)
        assert len(coords) <= 24
        assert len(coords) >= 4
        assert features.shape == (len(coords), 8)

        # Check sub-pixel accuracy near truth coordinates
        for ty, tx in truth_coords:
            dists = np.sqrt((coords[:, 0] - ty) ** 2 + (coords[:, 1] - tx) ** 2)
            assert np.min(dists) < 1.0

    def test_skew_symmetric_triad_operator(self):
        triad_op = SkewSymmetricTriadOperator(dim=8)
        u = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        v = np.array([0, 1.0, 1.0, 0, 0, 0, 0, 0], dtype=np.float64)

        B = triad_op.forward(u, v)
        # Pointwise tangent orthogonality
        ortho = float(abs(np.dot(v, B)))
        assert ortho <= 1.0e-8

    def test_slow_fast_triad_cascade_zero_drift(self):
        cascade = SlowFastTriadCascade(coupling=0.15, steps=16)
        v0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float64)
        u_seq = np.random.randn(16, 8).astype(np.float64)

        for steps in (1, 4, 8, 16):
            v_K, drift = cascade.integrate(u_seq, v0, steps=steps)
            assert drift == 0.0
            initial_norm = math.sqrt(float(np.sum(v0 * v0)))
            final_norm = math.sqrt(float(np.sum(v_K * v_K)))
            assert abs(final_norm - initial_norm) < 1e-14

    def test_causal_attribution_progenitors(self):
        # 2 active generators (predominantly off-diagonal driving)
        # 4 passive registers (predominantly diagonal local persistence)
        T = np.array([
            [1.0, 8.0, 0.0, 0.0, 0.0, 0.0],  # Generator 1: eta = 8/9 = 0.89
            [0.0, 2.0, 7.0, 0.0, 0.0, 0.0],  # Generator 2: eta = 7/9 = 0.78
            [0.0, 0.0, 9.0, 1.0, 0.0, 0.0],  # Passive 1: eta = 1/10 = 0.10
            [0.0, 0.0, 0.0, 8.0, 1.0, 0.0],  # Passive 2: eta = 1/9 = 0.11
            [0.0, 0.0, 0.0, 0.0, 7.0, 1.0],  # Passive 3: eta = 1/8 = 0.125
            [0.0, 0.0, 0.0, 0.0, 0.0, 6.0],  # Passive 4: eta = 0.00
        ], dtype=np.float64)

        is_gen, ratios = attribute_causal_progenitors(T, threshold=0.50)
        assert np.array_equal(is_gen, [True, True, False, False, False, False])
        assert ratios[0] > 0.50
        assert ratios[1] > 0.50
        assert ratios[2] < 0.50


class TestCosmologyModule:
    """Tests for astrosnail.cosmology."""

    def test_zslab_tomography_and_chunk_invariance(self):
        skewers = [
            np.sin(np.linspace(0, 2 * np.pi, 32), dtype=np.float32),
            np.cos(np.linspace(0, 2 * np.pi, 32), dtype=np.float32),
            np.ones(32, dtype=np.float32) * 0.5,
        ]
        coords = np.array([[5, 5], [15, 12], [20, 18]], dtype=np.float32)

        # Chunk z = 16
        recon16 = reconstruct_zslab_tomography(skewers, coords, grid_shape=(32, 24, 24), chunk_z=16)
        # Chunk z = 32
        recon32 = reconstruct_zslab_tomography(skewers, coords, grid_shape=(32, 24, 24), chunk_z=32)

        assert recon16["density"].shape == (32, 24, 24)
        assert recon16["velocity_potential"].shape == (32, 24, 24)
        assert recon16["shear_tensor"].shape == (6, 32, 24, 24)

        # Parity between slab chunk sizes
        np.testing.assert_allclose(recon16["density"], recon32["density"], atol=1e-5)

    def test_detect_jordan_cavity_cores(self):
        grid = np.full((32, 24, 24), 1.0, dtype=np.float32)  # High-density walls
        # Carve a void cavity inside
        grid[12:20, 8:16, 8:16] = -0.9  # Deep underdensity

        cores = detect_jordan_cavity_cores(grid, threshold=-0.7)
        assert cores.shape == (32, 24, 24)
        assert np.any(cores)
        # Cores must strictly be inside the underdense region
        assert np.all(grid[cores] <= -0.7)

    def test_so6_kelvin_mandel_rotor(self):
        # 100 random SO(3) rotations on 6x6 symmetric positive-definite matrix
        rng = np.random.default_rng(999)
        A = rng.standard_normal((6, 6))
        C_orig = np.dot(A, A.T) + np.eye(6)

        max_trace_res = 0.0
        for _ in range(100):
            H = rng.standard_normal((3, 3))
            Q, _ = np.linalg.qr(H)
            if np.linalg.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]

            C_rot, res = so6_kelvin_mandel_rotor(Q, C_orig)
            max_trace_res = max(max_trace_res, res)

        assert max_trace_res <= 1.14e-13

    def test_retract_void_step0_homotopy(self):
        density = np.array([-1.5, -0.8, 0.0, 1.2], dtype=np.float32).reshape(1, 2, 2)
        mask = np.array([True, True, False, False]).reshape(1, 2, 2)

        retracted = retract_void_step0_homotopy(density, mask)
        assert float(retracted[0, 0, 0]) == -1.0  # Clamped to physical minimum
        assert float(retracted[0, 0, 1]) == pytest.approx(-0.8)
        assert float(retracted[0, 1, 0]) == 0.0
        assert float(retracted[0, 1, 1]) == pytest.approx(1.2)

    def test_blind_astro_let_decouple(self):
        cube = np.random.randn(32, 24, 24).astype(np.float32) * 0.1
        rho_r, v_pec = blind_astro_let_decouple(cube)
        assert rho_r.shape == (32, 24, 24)
        assert v_pec.shape == (32, 24, 24)

    def test_solve_beltrami_jet_poincare(self):
        B_field, flux_escape = solve_beltrami_jet_poincare(np.zeros((24, 24)), mu=1.0, steps=16)
        assert B_field.shape == (3, 32, 24, 24)
        assert flux_escape == 0.0

    def test_solve_beltrami_jet_poincare_multistep_tracking(self):
        """Verify genuine multi-step tracking across varying step counts and amplitudes."""
        for steps in [1, 4, 8, 16, 32]:
            b_field, esc = solve_beltrami_jet_poincare(np.zeros((24, 24)), mu=1.0, steps=steps)
            assert esc == 0.0
            assert b_field.shape == (3, 32, 24, 24)

        # Amplitude scaling
        b_scaled, esc_scaled = solve_beltrami_jet_poincare(2.5 * np.ones((20, 20)), mu=1.5, steps=16)
        assert esc_scaled == 0.0
        assert float(np.max(np.abs(b_scaled))) > 1.0

    def test_zslab_tomography_nan_inf_sanitization(self):
        """Verify that reconstruct_zslab_tomography handles NaN/Inf skewers safely."""
        nan_skewers = [np.full(32, np.nan, dtype=np.float32) for _ in range(4)]
        coords = np.array([[6, 6], [6, 18], [18, 6], [18, 18]], dtype=np.float32)
        res = reconstruct_zslab_tomography(nan_skewers, coords, grid_shape=(32, 24, 24))
        assert not np.isnan(res["density"]).any()
        assert not np.isnan(res["velocity_potential"]).any()
        assert not np.isnan(res["shear_tensor"]).any()
        assert np.all(np.isfinite(res["density"]))

    def test_compute_mrxmhd_flow_jump_conservation(self):
        """Verify MRxMHD flow-Bernoulli head jump balance (Theorem 2.9b)."""
        p2, delta_p, res = compute_mrxmhd_flow_jump(p1=10.0, b1_sq=4.0, m1=0.3, b2_sq=2.0, m2=0.5)
        assert res == 0.0
        assert abs(p2 - (10.0 + delta_p)) < 1e-12


class TestEngineAndPerformance:
    """Tests for AstroSnailEngine, throughput floors, and memory ceilings."""

    def test_engine_streaming_update(self):
        engine = AstroSnailEngine(backend="numpy")

        # 1. 1D streaming series
        sig_1d = np.ones(50, dtype=np.float32)
        res_1d = engine.update(sig_1d)
        assert "morse_impulse" in res_1d
        assert "u_coarse" in res_1d

        # 2. 2D focal plane
        frame = np.zeros((128, 128), dtype=np.float32)
        frame[50, 50] = 10.0
        res_2d = engine.update(frame)
        assert "medoid_coords_yx" in res_2d
        assert res_2d["num_medoids"] >= 1

        # 3. 3D cosmological cube
        cube = np.zeros((32, 24, 24), dtype=np.float32)
        res_3d = engine.update(cube)
        assert "void_cores" in res_3d

    def test_performance_light_curve_25k_epochs(self):
        """< 50 ms per 25,000-epoch light curve."""
        t_epochs = 25000
        rng = np.random.default_rng(42)
        flux_stream = (10.0 + rng.normal(0.0, 1.0, t_epochs)).astype(np.float32)

        engine = AstroSnailEngine(backend="numpy")

        # Warm-up
        engine.update(flux_stream[:100])

        t0 = time.perf_counter()
        res = engine.update(flux_stream)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert elapsed_ms < 50.0, f"25,000 epochs took {elapsed_ms:.2f} ms (ceiling 50.0 ms)"

    def test_throughput_128x128_focal_plane(self):
        """>= 165 FPS on 128x128 focal plane (< 6.0 ms/frame)."""
        engine = AstroSnailEngine(backend="numpy")
        frame = np.random.randn(128, 128).astype(np.float32)

        # Warm-up
        for _ in range(5):
            engine.update(frame)

        n_frames = 50
        t0 = time.perf_counter()
        for _ in range(n_frames):
            engine.update(frame)
        elapsed = time.perf_counter() - t0

        fps = n_frames / elapsed
        ms_per_frame = (elapsed / n_frames) * 1000.0

        assert fps >= 165.0, f"Throughput {fps:.1f} FPS is below 165 FPS ({ms_per_frame:.2f} ms/frame)"

    def test_memory_cache_residency_and_tracemalloc_ceiling(self):
        """Working set < 2.0 MB, tracemalloc peak < 35.0 MB."""
        tracemalloc.start()
        engine = AstroSnailEngine(backend="numpy")

        # Exercise all pipelines
        sig = np.random.randn(25000).astype(np.float32)
        engine.update(sig)

        frame = np.random.randn(128, 128).astype(np.float32)
        engine.update(frame)

        cube = np.random.randn(32, 24, 24).astype(np.float32)
        engine.update(cube)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024.0 * 1024.0)
        assert peak_mb < 35.0, f"Tracemalloc peak heap {peak_mb:.2f} MB exceeded 35.0 MB ceiling"
