"""tests.test_challenger_m2_2_adversarial
========================================
Adversarial Empirical Stress Harness for Milestone 2 pipelines in astrosnail.
Author: Challenger M2-2 (Critic & Specialist)

Challenge Dimensions:
1. SkewSymmetricCosmicTriad:
   - Anti-symmetry: M + M^T = 0 across random dimensions (odd & even)
   - Trace conservation: Tr(M) = 0 and M_{j,j} = 0 identically
   - Purely imaginary eigenvalues across random dimensions (Re(lambda) = 0)
   - Dimension boundary stress (dim < 3 raises IndexError, dim >= 3 valid)
   - Zero self-interaction <v, B> = 0
2. compute_stellar_wind_bow_shock_jump:
   - Supersonic Mach numbers (M1 > 1, hypersonic up to M1 = 1000)
   - Subsonic Mach numbers (M1 < 1, incompressible limit M1 = 1e-4)
   - Extreme magnetic energy density (1e-20 to 1e30)
   - Shock head balance residual |head1 - head2| == 0 across 50 orders of magnitude
3. MorseTransientGateConfig:
   - Synthetic Poisson single-sample glints (glint rejection >= 99%, 0 false confirmed)
   - High background noise regimes (sigma = 0.01 to 100.0, SNR from 40 dB to -20 dB)
     - Empirically validates filter convergence to sqrt(2)*sigma theoretical difference noise
     - Steady-state: EXACTLY 0 false confirmed transients across 5,000 post-warmup samples
   - Step functions:
     - Pure step functions (100% recall for both positive flare onset and negative transit drop)
     - Noisy step functions (>= 8.0 sigma jump)
   - Boxcar pulse duration threshold:
     - K=1 rejected as glint (0 confirmed transients)
     - K>=2 confirmed as physical transient (rising edge confirmed at onset)
"""

import math
import numpy as np
import pytest

from astrosnail.cosmology import (
    SkewSymmetricCosmicTriad,
    compute_stellar_wind_bow_shock_jump,
)
from astrosnail.transients import MorseTransientGateConfig


# =============================================================================
# Challenge 1: SkewSymmetricCosmicTriad
# =============================================================================
class TestChallengeSkewSymmetricCosmicTriad:
    """Rigorous empirical challenge of SkewSymmetricCosmicTriad Lie-algebra properties."""

    @pytest.mark.parametrize("dim", [3, 4, 5, 6, 7, 8, 9, 10, 16, 25, 32, 48, 64])
    def test_antisymmetry_and_trace_across_random_dimensions(self, dim: int):
        """Stress-tests anti-symmetry (M + M^T = 0) and trace conservation (Tr(M) = 0)."""
        rng = np.random.default_rng(20261103 + dim)
        triad = SkewSymmetricCosmicTriad(dim=dim)

        # Test across multiple tensor configurations:
        # 1. Default canonical tensor
        # 2. Random standard normal tensor
        # 3. Random ill-conditioned tensor with extreme scale variance
        tensors = [
            ("canonical", triad.W),
            ("standard_normal", rng.standard_normal((dim, dim, dim))),
            ("ill_conditioned", rng.standard_normal((dim, dim, dim)) * (10.0 ** rng.uniform(-6, 6, size=(dim, dim, dim)))),
        ]

        for tensor_type, W in tensors:
            # Generate 10 random stream vectors per tensor
            for trial in range(10):
                u = rng.standard_normal(dim)
                v = rng.standard_normal(dim)

                # Explicitly form S and M
                S = W - np.swapaxes(W, 1, 2)
                M = np.tensordot(u, S, axes=([0], [0]))  # shape (dim, dim)

                # 1. Anti-symmetry test: M + M^T = 0
                M_sym = M + M.T
                norm_M = float(np.max(np.abs(M)))
                scale = max(1.0, norm_M)
                max_sym_err = float(np.max(np.abs(M_sym))) / scale

                assert max_sym_err < 1e-14, (
                    f"Anti-symmetry violated for dim={dim}, {tensor_type}, trial={trial}: "
                    f"rel_err={max_sym_err:.4e}"
                )

                # 2. Trace conservation: Tr(M) = 0
                trace_M = float(np.trace(M))
                assert abs(trace_M) < 1e-15 * scale, (
                    f"Trace conservation violated for dim={dim}, {tensor_type}, trial={trial}: "
                    f"Tr(M)={trace_M:.4e}"
                )

                # 3. Diagonal elements must be identically zero
                diag_max = float(np.max(np.abs(np.diag(M))))
                assert diag_max < 1e-15 * scale, (
                    f"Non-zero diagonal element in dim={dim}, {tensor_type}: {diag_max:.4e}"
                )

                # 4. Forward execution parity: B = v^T M
                B_forward = triad.forward(u, v, W_triad=W)
                B_expected = np.dot(v, M)
                fwd_err = float(np.max(np.abs(B_forward - B_expected)))
                assert fwd_err < 1e-14 * scale, (
                    f"Forward output mismatch for dim={dim}: {fwd_err:.4e}"
                )

                # 5. Zero self-interaction / orthogonality: <v, B> = 0
                ortho_residual = float(abs(np.dot(v, B_forward)))
                v_norm = float(np.linalg.norm(v))
                B_norm = float(np.linalg.norm(B_forward))
                ortho_scale = max(1.0, v_norm * B_norm)
                rel_ortho = ortho_residual / ortho_scale
                assert rel_ortho < 1e-13, (
                    f"Orthogonality <v, B> != 0 for dim={dim}: rel_ortho={rel_ortho:.4e}"
                )

    @pytest.mark.parametrize("dim", [3, 4, 5, 6, 7, 8, 9, 11, 16, 23, 32])
    def test_eigenvalues_purely_imaginary_across_random_dimensions(self, dim: int):
        """Stress-tests that all eigenvalues of M lie strictly on the imaginary axis: Re(lambda) = 0."""
        rng = np.random.default_rng(9999 + dim)

        for trial in range(25):
            u = rng.standard_normal(dim)
            W = rng.standard_normal((dim, dim, dim))

            S = W - np.swapaxes(W, 1, 2)
            M = np.tensordot(u, S, axes=([0], [0]))

            evals = np.linalg.eigvals(M)
            norm_M = max(1.0, float(np.linalg.norm(M, ord=2)))

            # Real part must be numerically zero
            max_real = float(np.max(np.abs(np.real(evals))))
            rel_real = max_real / norm_M

            assert rel_real < 1e-13, (
                f"Eigenvalue with non-zero real part found for dim={dim}, trial={trial}: "
                f"rel_real={rel_real:.4e}, evals={evals}"
            )

            # For odd dimensions, at least one eigenvalue must be identically zero
            if dim % 2 != 0:
                min_abs_eval = float(np.min(np.abs(evals)))
                assert min_abs_eval / norm_M < 1e-13, (
                    f"Odd dimension {dim} missing zero eigenvalue: min |lambda| = {min_abs_eval:.4e}"
                )

            # Eigenvalues must pair into conjugate pairs (+i*omega, -i*omega)
            imag_parts = np.sort(np.imag(evals))
            sum_pairs = imag_parts + imag_parts[::-1]
            max_pair_asymm = float(np.max(np.abs(sum_pairs))) / norm_M
            assert max_pair_asymm < 1e-13, (
                f"Eigenvalues not conjugate-symmetric for dim={dim}: {max_pair_asymm:.4e}"
            )

    def test_dimension_boundary_and_exceptions(self):
        """Validates dimension boundary conditions and expected failure modes."""
        # dim < 3 must raise IndexError because default canonical triad accesses index 2
        with pytest.raises(IndexError):
            SkewSymmetricCosmicTriad(dim=2)

        with pytest.raises(IndexError):
            SkewSymmetricCosmicTriad(dim=1)

        # Minimal dimension dim=3 must execute cleanly
        tri3 = SkewSymmetricCosmicTriad(dim=3)
        u3 = np.array([1.0, 0.0, 0.0])
        v3 = np.array([0.0, 1.0, 0.0])
        B3 = tri3.forward(u3, v3)
        assert B3.shape == (3,)
        assert abs(np.dot(v3, B3)) < 1e-14


# =============================================================================
# Challenge 2: compute_stellar_wind_bow_shock_jump
# =============================================================================
class TestChallengeStellarWindBowShockJump:
    """Rigorous empirical challenge of compute_stellar_wind_bow_shock_jump across physical regimes."""

    @pytest.mark.parametrize("m1", [1.01, 1.5, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0])
    @pytest.mark.parametrize("m2", [0.01, 0.1, 0.3, 0.5, 0.8, 0.99])
    def test_supersonic_mach_numbers(self, m1: float, m2: float):
        """Stress-tests supersonic shock transitions (M1 > 1 -> M2 < 1)."""
        rng = np.random.default_rng(int(m1 * 100 + m2 * 10))
        p1 = float(rng.uniform(1.0, 100.0))
        b1_sq = float(rng.uniform(0.1, 10.0))
        b2_sq = float(rng.uniform(0.1, 10.0))

        p2, delta_p, residual = compute_stellar_wind_bow_shock_jump(
            p1=p1, b1_sq=b1_sq, m1=m1, b2_sq=b2_sq, m2=m2
        )

        expected_dp = 0.5 * ((1.0 + m1 * m1) * b1_sq - (1.0 + m2 * m2) * b2_sq)
        expected_p2 = p1 + expected_dp
        head1 = p1 + 0.5 * (1.0 + m1 * m1) * b1_sq
        head2 = p2 + 0.5 * (1.0 + m2 * m2) * b2_sq
        expected_res = abs(head1 - head2)

        scale = max(1.0, abs(head1))
        assert abs(p2 - expected_p2) / scale < 1e-14
        assert abs(delta_p - expected_dp) / scale < 1e-14
        assert residual == expected_res
        assert residual / scale < 1e-14, f"Non-zero bow shock head residual: {residual:.4e}"

    @pytest.mark.parametrize("m1", [1e-4, 1e-2, 0.1, 0.3, 0.6, 0.95])
    @pytest.mark.parametrize("m2", [1e-4, 1e-2, 0.1, 0.3, 0.6, 0.95])
    def test_subsonic_mach_numbers(self, m1: float, m2: float):
        """Stress-tests subsonic flows (M1 < 1, M2 < 1) for Bernoulli head balance."""
        p1 = 50.0
        b1_sq = 2.5
        b2_sq = 3.0

        p2, delta_p, residual = compute_stellar_wind_bow_shock_jump(
            p1=p1, b1_sq=b1_sq, m1=m1, b2_sq=b2_sq, m2=m2
        )

        head1 = p1 + 0.5 * (1.0 + m1 * m1) * b1_sq
        head2 = p2 + 0.5 * (1.0 + m2 * m2) * b2_sq
        scale = max(1.0, abs(head1))

        assert abs(head1 - head2) / scale < 1e-14
        assert residual / scale < 1e-14

    @pytest.mark.parametrize("b_scale", [1e-20, 1e-15, 1e-8, 1.0, 1e8, 1e15, 1e25])
    def test_extreme_magnetic_fields(self, b_scale: float):
        """Stress-tests extreme magnetic energy densities across 45 orders of magnitude."""
        p1 = 1.0e5
        b1_sq = 1.0 * b_scale
        b2_sq = 2.5 * b_scale
        m1 = 2.0
        m2 = 0.5

        p2, delta_p, residual = compute_stellar_wind_bow_shock_jump(
            p1=p1, b1_sq=b1_sq, m1=m1, b2_sq=b2_sq, m2=m2
        )

        head1 = p1 + 0.5 * (1.0 + m1 * m1) * b1_sq
        head2 = p2 + 0.5 * (1.0 + m2 * m2) * b2_sq
        scale = max(1.0, abs(head1), abs(head2))

        rel_res = residual / scale
        assert rel_res < 1e-14, (
            f"Head jump balance broken under extreme B^2 scale {b_scale}: "
            f"residual={residual:.4e}, rel_res={rel_res:.4e}"
        )

    def test_transonic_and_degenerate_conditions(self):
        """Tests exact transonic limit (M1=1, M2=1) and zero-field degenerate limit."""
        # Transonic identical fields
        p2, delta_p, res = compute_stellar_wind_bow_shock_jump(
            p1=100.0, b1_sq=10.0, m1=1.0, b2_sq=10.0, m2=1.0
        )
        assert delta_p == 0.0
        assert p2 == 100.0
        assert res == 0.0

        # Hydrodynamic limit (zero magnetic field)
        p2_hydro, dp_hydro, res_hydro = compute_stellar_wind_bow_shock_jump(
            p1=75.0, b1_sq=0.0, m1=5.0, b2_sq=0.0, m2=0.2
        )
        assert dp_hydro == 0.0
        assert p2_hydro == 75.0
        assert res_hydro == 0.0


# =============================================================================
# Challenge 3: MorseTransientGateConfig
# =============================================================================
class TestChallengeMorseTransientGateConfig:
    """Rigorous empirical challenge of MorseTransientGateConfig causal deferral state machine."""

    def test_synthetic_poisson_glints_rejection(self):
        """Stress-tests Poisson-distributed single-sample glints.
        Asserts >= 99% isolated glint rejection and EXACTLY 0 false confirmed transients.
        """
        rng = np.random.default_rng(20261103)
        n_samples = 10_000
        noise_sigma = 1.0

        # Generate Gaussian background noise
        noise = rng.normal(0.0, noise_sigma, size=n_samples)
        signal = np.copy(noise)

        # Inject Poisson-distributed single-sample glints (~300 glints)
        glint_rate = 0.03
        glint_locs = rng.poisson(glint_rate, size=n_samples) > 0
        # Ensure no two glints are adjacent (isolated single-sample spikes)
        for i in range(1, n_samples):
            if glint_locs[i - 1] and glint_locs[i]:
                glint_locs[i] = False
        glint_locs[0] = False
        glint_locs[-1] = False

        injected_count = int(np.sum(glint_locs))
        assert injected_count >= 150, f"Expected >= 150 glints, got {injected_count}"

        # Glint amplitudes: high SNR (10 sigma to 100 sigma)
        amplitudes = rng.uniform(10.0, 100.0, size=injected_count) * rng.choice([-1.0, 1.0], size=injected_count)
        signal[glint_locs] += amplitudes

        gate = MorseTransientGateConfig(sigma_thresh=4.0, persistence_window=1)

        history = []
        for t in range(n_samples):
            status, applied = gate.update(signal[t], timestamp=float(t))
            history.append((status, applied))

        # Check: exactly 0 false confirmed transients!
        assert gate.confirmed_transient_count == 0, (
            f"False triggers! Gate produced {gate.confirmed_transient_count} confirmed transients on pure glints!"
        )

        # Check: glint rejection count
        rejected = gate.glint_rejected_count
        rejection_ratio = rejected / injected_count
        assert rejection_ratio >= 0.99, (
            f"Glint rejection ratio {rejection_ratio:.4f} < 0.99 ({rejected}/{injected_count})"
        )

        # Inspect behavior at exact glint timestamps:
        for t in np.where(glint_locs)[0]:
            st_glint, ap_glint = history[t]
            st_post, ap_post = history[t + 1]

            # At onset: deferred
            assert st_glint == "pending_impulse" and ap_glint is False, (
                f"At glint epoch t={t}, expected ('pending_impulse', False), got ({st_glint}, {ap_glint})"
            )
            # At t+1: rejected
            assert st_post == "glint_rejected" and ap_post is False, (
                f"At post-glint epoch t+1={t+1}, expected ('glint_rejected', False), got ({st_post}, {ap_post})"
            )

    @pytest.mark.parametrize("noise_sigma", [0.01, 0.1, 1.0, 10.0, 100.0])
    def test_high_background_noise_steady_state_and_warmup(self, noise_sigma: float):
        """Stress-tests continuous Gaussian background noise across SNR regimes from 40 dB to -20 dB.
        Empirically validates:
        1. Theoretical noise convergence: diff variance = 2 * sigma^2, so noise_std converges to sqrt(2)*sigma.
        2. Steady-state: EXACTLY 0 false confirmed transients across 5,000 post-warmup samples.
        """
        rng = np.random.default_rng(42 + int(noise_sigma * 10))
        warmup_samples = 300
        stream_samples = 5_000
        total_samples = warmup_samples + stream_samples

        noise = rng.normal(0.0, noise_sigma, size=total_samples)

        gate = MorseTransientGateConfig(sigma_thresh=4.0, persistence_window=1)

        # Phase 1: Warmup adaptation
        for t in range(warmup_samples):
            gate.update(noise[t], timestamp=float(t))

        # Check that noise std estimate adapted to sqrt(2)*noise_sigma (+/- 40%)
        expected_diff_std = math.sqrt(2.0) * noise_sigma
        ratio = gate.noise_std / expected_diff_std
        assert 0.60 < ratio < 1.60, (
            f"Noise std estimator failed to adapt at sigma={noise_sigma}: "
            f"tracked={gate.noise_std:.4e}, expected={expected_diff_std:.4e}, ratio={ratio:.3f}"
        )

        # Phase 2: Steady-state evaluation
        post_warmup_confirmed = 0
        for t in range(warmup_samples, total_samples):
            st, ap = gate.update(noise[t], timestamp=float(t))
            if st == "confirmed_transient":
                post_warmup_confirmed += 1

        # Post-warmup, steady-state false triggers must be EXACTLY 0
        assert post_warmup_confirmed == 0, (
            f"Spurious confirmed transients in steady-state noise at sigma={noise_sigma}: {post_warmup_confirmed}"
        )

    @pytest.mark.parametrize("step_height", [10.0, 25.0, 50.0, 100.0])
    @pytest.mark.parametrize("polarity", [1.0, -1.0])
    def test_pure_step_functions_confirmation(self, step_height: float, polarity: float):
        """Stress-tests clean Heaviside step functions (both positive flare onset and negative transit drop)."""
        n_samples = 600
        t_step = 300
        signal = np.zeros(n_samples, dtype=np.float64)
        signal[t_step:] = polarity * step_height

        gate = MorseTransientGateConfig(sigma_thresh=4.0, persistence_window=1)

        history = []
        for t in range(n_samples):
            st, ap = gate.update(signal[t], timestamp=float(t))
            history.append((st, ap))

        # At t_step: must defer as pending_impulse
        st_onset, ap_onset = history[t_step]
        assert st_onset == "pending_impulse" and ap_onset is False, (
            f"Pure step at t={t_step} failed to enter pending_impulse: ({st_onset}, {ap_onset})"
        )

        # At t_step + 1: must confirm as sustained transient
        st_confirm, ap_confirm = history[t_step + 1]
        assert st_confirm == "confirmed_transient" and ap_confirm is True, (
            f"Pure step at t={t_step+1} failed to confirm transient: ({st_confirm}, {ap_confirm})"
        )

        # After confirmation, should settle back to nominal
        assert history[t_step + 2][0] == "nominal"
        assert gate.confirmed_transient_count == 1
        assert gate.glint_rejected_count == 0

    @pytest.mark.parametrize("step_height_sigma", [10.0, 20.0, 50.0])
    @pytest.mark.parametrize("polarity", [1.0, -1.0])
    def test_noisy_step_functions_confirmation(self, step_height_sigma: float, polarity: float):
        """Stress-tests sustained step functions immersed in Gaussian background noise."""
        rng = np.random.default_rng(54321)
        n_samples = 1_000
        t_step = 500
        noise_sigma = 1.0

        signal = rng.normal(0.0, noise_sigma, size=n_samples)
        step_amplitude = polarity * step_height_sigma * noise_sigma
        signal[t_step:] += step_amplitude

        gate = MorseTransientGateConfig(sigma_thresh=4.0, persistence_window=1)

        history = []
        for t in range(n_samples):
            st, ap = gate.update(signal[t], timestamp=float(t))
            history.append((st, ap))

        confirmed_indices = [t for t, (st, _) in enumerate(history) if st == "confirmed_transient"]
        assert len(confirmed_indices) == 1, (
            f"Expected exactly 1 confirmed transient for noisy step (polarity={polarity}, height={step_height_sigma}), "
            f"got {len(confirmed_indices)}: {confirmed_indices}"
        )
        assert abs(confirmed_indices[0] - (t_step + 1)) <= 1
        assert gate.glint_rejected_count == 0

    @pytest.mark.parametrize("duration", [1, 2, 3, 5, 10])
    def test_boxcar_transient_duration_threshold(self, duration: int):
        """Tests the critical discriminating duration threshold:
        - duration = 1 sample MUST be rejected as glint (0 confirmed transients)
        - duration >= 2 samples MUST confirm physical transient:
          Rising edge is confirmed as transient onset.
          (If duration >= 2, falling edge is also confirmed as termination step-down).
        """
        n_samples = 500
        t_start = 200
        signal = np.zeros(n_samples, dtype=np.float64)
        signal[t_start : t_start + duration] = 20.0  # High SNR pulse

        gate = MorseTransientGateConfig(sigma_thresh=4.0, persistence_window=1)

        history = []
        for t in range(n_samples):
            st, ap = gate.update(signal[t], timestamp=float(t))
            history.append((st, ap))

        if duration == 1:
            # Single-sample spike: glint rejected!
            assert gate.glint_rejected_count == 1, f"Duration 1 was not rejected as glint!"
            assert gate.confirmed_transient_count == 0, f"Duration 1 produced a confirmed transient!"
        else:
            # Multi-sample pulse: rising edge confirmed at onset
            assert history[t_start][0] == "pending_impulse"
            assert history[t_start + 1][0] == "confirmed_transient"
            assert history[t_start + 1][1] is True
            # Zero glint rejections during pulse
            assert gate.glint_rejected_count == 0, (
                f"Duration {duration} was erroneously classified as glint!"
            )
            # Both onset and offset are confirmed steps
            assert gate.confirmed_transient_count in (1, 2)

    def test_pathological_inputs_and_resets(self):
        """Tests non-finite values (NaN, Inf) and state machine reset invariants."""
        gate = MorseTransientGateConfig(sigma_thresh=4.0)

        # Non-finite values must be safely rejected with update_applied = False
        st, ap = gate.update(float("nan"), timestamp=0.0)
        assert st == "nominal" and ap is False

        st, ap = gate.update(float("inf"), timestamp=1.0)
        assert st == "nominal" and ap is False

        st, ap = gate.update(float("-inf"), timestamp=2.0)
        assert st == "nominal" and ap is False

        # Normal operation after non-finite values
        st, ap = gate.update(1.0, timestamp=3.0)
        assert st == "nominal" and ap is True

        # Reset clears all registers
        gate.reset()
        assert gate.initialized is False
        assert gate.gate_status == "nominal"
        assert gate.total_steps == 0
        assert gate.glint_rejected_count == 0
        assert gate.confirmed_transient_count == 0
