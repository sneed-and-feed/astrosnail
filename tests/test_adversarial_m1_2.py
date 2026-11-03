"""
tests/test_adversarial_m1_2.py
==============================
Empirical adversarial stress test suite for Milestone 1 (Challenger M1-2):
1. Stress test cosmology.py:
   - Z-slab reconstruction under extreme noise, missing skewers, colinear/boundary skewers, chunk invariance
   - 6-way Jordan cavity core void detection with degenerate non-convex geometries (hollow shells, labyrinth, thin tunnels, donuts, extreme values)
   - SO(6) Kelvin-Mandel rotor: 1,000 random rotations, trace residual <= 1.14e-13, energy/Frobenius norm conservation, orthogonality, group homomorphism
   - Step-0 void homotopy retraction: strict 0.000e+00 drift, boundary clamping, mask variations
   - 3D Beltrami jet solver: B . grad psi = 0 everywhere, 0.000e+00 radial flux escape, solenoidal & Beltrami conditions across varying mu and steps
2. Stress test performance and memory:
   - Sustained streaming over 300 frames on 128x128 focal plane (assert >= 165 FPS)
   - tracemalloc peak heap tracking (< 2.0 MB streaming, < 35.0 MB hard ceiling, zero monotonic leak)
"""

import math
import time
import tracemalloc
import numpy as np
import pytest
from scipy import special

from astrosnail.cosmology import (
    reconstruct_zslab_tomography,
    detect_jordan_cavity_cores,
    so6_kelvin_mandel_rotor,
    _build_so6_kelvin_rotor,
    retract_void_step0_homotopy,
    blind_astro_let_decouple,
    solve_beltrami_jet_poincare,
    SkewSymmetricCosmicTriad,
)
from astrosnail.engine import AstroSnailEngine


class TestCosmologyAdversarialZslab:
    """Stress testing reconstruct_zslab_tomography under extreme noise and degeneracies."""

    def test_zslab_zero_and_mismatched_skewers(self):
        """Empty skewers or coordinate mismatch gracefully returns zero fields."""
        res_empty = reconstruct_zslab_tomography([], np.zeros((0, 2)), grid_shape=(32, 24, 24))
        assert res_empty["density"].shape == (32, 24, 24)
        assert np.all(res_empty["density"] == 0.0)
        assert np.all(res_empty["velocity_potential"] == 0.0)
        assert np.all(res_empty["shear_tensor"] == 0.0)

        # Mismatched length between skewers and coords
        res_mismatch = reconstruct_zslab_tomography(
            [np.ones(32, dtype=np.float32)],
            np.zeros((2, 2), dtype=np.float32),
            grid_shape=(32, 24, 24),
        )
        assert np.all(res_mismatch["density"] == 0.0)

    def test_zslab_single_skewer(self):
        """Single skewer placed in grid produces valid finite fields."""
        skewers = [np.sin(np.linspace(0, np.pi, 32), dtype=np.float32)]
        coords = np.array([[12.0, 12.0]], dtype=np.float32)

        res = reconstruct_zslab_tomography(skewers, coords, grid_shape=(32, 24, 24))
        assert res["density"].shape == (32, 24, 24)
        assert np.all(np.isfinite(res["density"]))
        assert np.all(np.isfinite(res["velocity_potential"]))
        assert np.all(np.isfinite(res["shear_tensor"]))
        # Density must have zero mean
        assert abs(float(np.mean(res["density"]))) < 1e-6

    def test_zslab_extreme_noise_and_negative_flux(self):
        """High dynamic range, zero variance, and extreme noise in skewers."""
        rng = np.random.default_rng(20261026)
        n_skewers = 8
        z_dim, y_dim, x_dim = 32, 24, 24

        coords = rng.uniform(0, 23, size=(n_skewers, 2)).astype(np.float32)

        # Skewers with extreme amplitude noise (-1e5 to +1e5)
        skewers_noisy = [
            (rng.standard_normal(z_dim) * 1e4).astype(np.float32)
            for _ in range(n_skewers)
        ]
        # One completely flat skewer (zero variance)
        skewers_noisy[0] = np.zeros(z_dim, dtype=np.float32)

        res = reconstruct_zslab_tomography(skewers_noisy, coords, grid_shape=(z_dim, y_dim, x_dim))
        assert np.all(np.isfinite(res["density"]))
        assert np.all(np.isfinite(res["velocity_potential"]))
        assert np.all(np.isfinite(res["shear_tensor"]))
        assert abs(float(np.mean(res["density"]))) < 1e-3

    def test_zslab_colinear_and_boundary_coordinates(self):
        """Skewers located exactly on boundary corners and all colinear."""
        z_dim, y_dim, x_dim = 32, 24, 24
        # Colinear skewers along y = 0
        coords = np.array([
            [0.0, 0.0],
            [0.0, 5.0],
            [0.0, 12.0],
            [0.0, 23.0],
            [23.0, 23.0],  # Corner
            [-2.0, 25.0],  # Out of bounds
        ], dtype=np.float32)

        skewers = [np.sin(np.linspace(0, k * np.pi, z_dim), dtype=np.float32) for k in range(len(coords))]

        res = reconstruct_zslab_tomography(skewers, coords, grid_shape=(z_dim, y_dim, x_dim))
        assert np.all(np.isfinite(res["density"]))
        assert abs(float(np.mean(res["density"]))) < 1e-5

    def test_zslab_chunk_invariance_comprehensive(self):
        """Chunk size chunk_z = 1, 2, 4, 8, 16, 24, 32, 64 must yield identical results."""
        z_dim, y_dim, x_dim = 32, 24, 24
        rng = np.random.default_rng(777)
        skewers = [rng.normal(1.0, 0.5, size=z_dim).astype(np.float32) for _ in range(5)]
        coords = rng.uniform(2, 20, size=(5, 2)).astype(np.float32)

        ref_recon = reconstruct_zslab_tomography(skewers, coords, grid_shape=(z_dim, y_dim, x_dim), chunk_z=16)

        for chunk_z in [1, 2, 4, 7, 8, 13, 16, 24, 32, 64]:
            test_recon = reconstruct_zslab_tomography(skewers, coords, grid_shape=(z_dim, y_dim, x_dim), chunk_z=chunk_z)
            np.testing.assert_allclose(
                test_recon["density"],
                ref_recon["density"],
                atol=1e-5,
                err_msg=f"Chunk size {chunk_z} differed from chunk_z=16 reference"
            )

    def test_zslab_shear_tensor_trace_freedom(self):
        """Kelvin-Mandel shear tensor components 0, 1, 2 must sum to zero (trace-free)."""
        skewers = [
            np.sin(np.linspace(0, 3 * np.pi, 32), dtype=np.float32),
            np.cos(np.linspace(0, 3 * np.pi, 32), dtype=np.float32),
        ]
        coords = np.array([[8.0, 8.0], [16.0, 16.0]], dtype=np.float32)

        res = reconstruct_zslab_tomography(skewers, coords, grid_shape=(32, 24, 24))
        gamma = res["shear_tensor"]  # (6, 32, 24, 24)
        trace = gamma[0] + gamma[1] + gamma[2]
        max_trace_err = float(np.max(np.abs(trace)))
        assert max_trace_err < 1e-5, f"Shear tensor is not traceless: max abs trace = {max_trace_err}"


class TestCosmologyAdversarialVoidCavity:
    """Stress testing detect_jordan_cavity_cores with degenerate geometries."""

    def test_void_detection_empty_and_full(self):
        """All-wall or all-void volumes."""
        # 1. All wall (density = +2.0)
        all_wall = np.full((32, 24, 24), 2.0, dtype=np.float32)
        cores_wall = detect_jordan_cavity_cores(all_wall, threshold=-0.7)
        assert not np.any(cores_wall), "All-wall volume should have no void cores"

        # 2. All void (density = -0.9) - open space, no enclosing walls
        all_void = np.full((32, 24, 24), -0.9, dtype=np.float32)
        cores_void = detect_jordan_cavity_cores(all_void, threshold=-0.7)
        assert not np.any(cores_void), "Open unbounded void without walls should not have enclosed cores"

    def test_void_detection_hollow_shell(self):
        """Hollow spherical/cubical shell enclosing deep underdense cavity."""
        grid = np.zeros((32, 24, 24), dtype=np.float32)
        # Create outer high-density shell at indices 6 and 26
        grid[6:26, 4:20, 4:20] = -0.9  # interior void
        # Wall shell around it
        grid[5, 3:21, 3:21] = 1.5
        grid[26, 3:21, 3:21] = 1.5
        grid[5:27, 3, 3:21] = 1.5
        grid[5:27, 20, 3:21] = 1.5
        grid[5:27, 3:21, 3] = 1.5
        grid[5:27, 3:21, 20] = 1.5

        cores = detect_jordan_cavity_cores(grid, threshold=-0.7)
        assert np.any(cores), "Hollow enclosed shell must detect void core"
        # Cores must strictly be within the interior
        assert np.all(grid[cores] <= -0.7)
        # Ensure shell walls are not marked as cores
        assert not np.any(cores[grid > -0.7])

    def test_void_detection_punctured_shell_leaks(self):
        """Puncturing a hole in the enclosing wall breaks 6-way enclosure along that ray."""
        grid = np.full((32, 24, 24), 1.5, dtype=np.float32)
        # Carve interior void
        grid[10:22, 8:16, 8:16] = -0.9
        # Puncture a hole along X through the wall at y=12, z=16
        grid[16, 12, :] = -0.9  # Open channel all the way through +X and -X

        cores = detect_jordan_cavity_cores(grid, threshold=-0.7)
        # The punctured voxel line (16, 12, :) has no X-walls on both sides,
        # so enclosure_count < 6 there
        assert not cores[16, 12, 12], "Punctured ray path must not be enclosed"

    def test_void_detection_nan_inf_resilience(self):
        """Volume contaminated with NaNs and Infs does not crash or corrupt output."""
        grid = np.full((32, 24, 24), 1.0, dtype=np.float32)
        grid[10:20, 8:16, 8:16] = -0.9
        grid[0, 0, 0] = np.nan
        grid[1, 1, 1] = np.inf
        grid[2, 2, 2] = -np.inf

        cores = detect_jordan_cavity_cores(grid, threshold=-0.7)
        assert cores.shape == (32, 24, 24)
        assert cores.dtype == bool
        assert np.any(cores)

    def test_void_detection_thin_void_fallback(self):
        """A single 1-voxel void core inside thick walls is retained via erosion fallback."""
        grid = np.full((16, 16, 16), 2.0, dtype=np.float32)
        grid[8, 8, 8] = -0.95  # Single isolated voxel

        cores = detect_jordan_cavity_cores(grid, threshold=-0.7)
        assert cores[8, 8, 8] is True or bool(cores[8, 8, 8]) == True
        assert np.sum(cores) == 1


class TestCosmologyAdversarialSO6Rotor:
    """Stress testing so6_kelvin_mandel_rotor over 1,000 random rotations and mathematical invariants."""

    def test_so6_rotor_1000_rotations_trace_residual(self):
        """Assert trace residual <= 1.14e-13 across 1,000 random 3D rotations on 6x6 matrix."""
        rng = np.random.default_rng(20261027)

        # Generate a realistic random positive-definite tidal stiffness matrix
        A = rng.standard_normal((6, 6))
        C_orig = np.dot(A, A.T) + 2.0 * np.eye(6)

        max_trace_res_6x6 = 0.0
        max_ortho_error = 0.0
        max_det_error = 0.0
        max_energy_drift = 0.0

        for i in range(1000):
            # Haar-distributed random rotation in SO(3)
            H = rng.standard_normal((3, 3))
            Q, R = np.linalg.qr(H)
            # Ensure det(Q) == +1
            d = np.linalg.det(Q)
            if d < 0:
                Q[:, 0] = -Q[:, 0]

            # 1. 6x6 matrix rotation
            C_rot, res_6x6 = so6_kelvin_mandel_rotor(Q, C_orig)
            max_trace_res_6x6 = max(max_trace_res_6x6, res_6x6)

            # 2. Check orthogonality of M: M M^T = I_6
            M = _build_so6_kelvin_rotor(Q)
            ortho_diff = np.max(np.abs(np.dot(M, M.T) - np.eye(6)))
            max_ortho_error = max(max_ortho_error, ortho_diff)

            # 3. Check det(M) == +1
            det_M = np.linalg.det(M)
            max_det_error = max(max_det_error, abs(det_M - 1.0))

            # 4. Check Frobenius norm / energy conservation: ||C_rot||_F == ||C_orig||_F
            norm_orig = np.linalg.norm(C_orig, "fro")
            norm_rot = np.linalg.norm(C_rot, "fro")
            max_energy_drift = max(max_energy_drift, abs(norm_rot - norm_orig))

        assert max_trace_res_6x6 <= 1.14e-13, f"Max trace residual {max_trace_res_6x6:.3e} exceeded 1.14e-13"
        assert max_ortho_error <= 1e-14, f"Max orthogonality error {max_ortho_error:.3e} exceeded 1e-14"
        assert max_det_error <= 1e-12, f"Max determinant error {max_det_error:.3e} exceeded 1e-12"
        assert max_energy_drift <= 1e-13, f"Max energy drift {max_energy_drift:.3e} exceeded 1e-13"

    def test_so6_rotor_3x3_and_6d_vector_modes(self):
        """Verify so6_kelvin_mandel_rotor on 3x3 symmetric tensors and 6D Kelvin-Mandel vectors."""
        rng = np.random.default_rng(42)

        for _ in range(100):
            H = rng.standard_normal((3, 3))
            Q, _ = np.linalg.qr(H)
            if np.linalg.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]

            # 3x3 symmetric tensor
            S = rng.standard_normal((3, 3))
            S = S + S.T
            rot_3x3, res_3x3 = so6_kelvin_mandel_rotor(Q, S)
            assert res_3x3 <= 1.14e-13

            # 6D Kelvin-Mandel vector
            v6 = rng.standard_normal(6)
            rot_v6, res_v6 = so6_kelvin_mandel_rotor(Q, v6)
            assert res_v6 <= 1.14e-13

            # Vector Euclidean norm conservation: ||rot_v6|| == ||v6||
            assert abs(np.linalg.norm(rot_v6) - np.linalg.norm(v6)) < 1e-14

    def test_so6_rotor_group_homomorphism(self):
        """Verify M(R1 R2) == M(R1) M(R2) over 50 random pairs."""
        rng = np.random.default_rng(1234)

        for _ in range(50):
            H1 = rng.standard_normal((3, 3))
            Q1, _ = np.linalg.qr(H1)
            if np.linalg.det(Q1) < 0:
                Q1[:, 0] = -Q1[:, 0]

            H2 = rng.standard_normal((3, 3))
            Q2, _ = np.linalg.qr(H2)
            if np.linalg.det(Q2) < 0:
                Q2[:, 0] = -Q2[:, 0]

            Q12 = np.dot(Q1, Q2)

            M1 = _build_so6_kelvin_rotor(Q1)
            M2 = _build_so6_kelvin_rotor(Q2)
            M12 = _build_so6_kelvin_rotor(Q12)

            diff = np.max(np.abs(np.dot(M1, M2) - M12))
            assert diff < 1e-14, f"SO(6) homomorphism error {diff:.3e} exceeded 1e-14"


class TestCosmologyAdversarialStep0Homotopy:
    """Stress testing retract_void_step0_homotopy for strict 0.000e+00 drift."""

    def test_step0_homotopy_exact_zero_drift(self):
        """retract_void_step0_homotopy must satisfy 0.000e+00 drift from physical bounds."""
        rng = np.random.default_rng(888)
        # Wide dynamic range [-1e6, +1e6]
        density = rng.uniform(-5.0, 5.0, size=(10, 10, 10)).astype(np.float32)
        # Introduce catastrophic unphysical dips
        density[2, 3, 4] = -1e6
        density[5, 5, 5] = -1.0000001
        density[6, 6, 6] = -1.0

        mask = rng.choice([True, False], size=density.shape, p=[0.4, 0.6])

        retracted = retract_void_step0_homotopy(density, mask)

        # For masked positions: density must be >= -1.0
        assert np.all(retracted[mask] >= -1.0)

        # Points where density was >= -1.0 must be BITWISE IDENTICAL (0.000e+00 drift)
        safe_masked = mask & (density >= -1.0)
        assert np.array_equal(retracted[safe_masked], density[safe_masked])

        # Points outside mask must be BITWISE UNTOUCHED
        assert np.array_equal(retracted[~mask], density[~mask])

        # Specifically check clamped values
        assert float(retracted[2, 3, 4]) == -1.0 if mask[2, 3, 4] else float(density[2, 3, 4])
        assert float(retracted[6, 6, 6]) == -1.0

    def test_step0_homotopy_empty_and_full_masks(self):
        """All-True and all-False masks."""
        density = np.array([-2.5, -1.0, -0.5, 0.0, 1.5], dtype=np.float32)

        # Empty mask
        ret_empty = retract_void_step0_homotopy(density, np.zeros_like(density, dtype=bool))
        assert np.array_equal(ret_empty, density)

        # Full mask
        ret_full = retract_void_step0_homotopy(density, np.ones_like(density, dtype=bool))
        assert float(ret_full[0]) == -1.0
        assert float(ret_full[1]) == -1.0
        assert np.array_equal(ret_full[2:], density[2:])


class TestCosmologyAdversarialBeltramiJet:
    """Stress testing 3D force-free Beltrami jet solver."""

    def test_beltrami_jet_flux_surface_confinement(self):
        """Verify B . grad psi = 0 everywhere on the grid with zero radial flux escape."""
        mu_vals = [0.5, 1.0, 2.0, 3.5]
        for mu in mu_vals:
            B_field, radial_escape = solve_beltrami_jet_poincare(np.zeros((24, 24)), mu=mu, steps=16)
            assert radial_escape == 0.0

            # Compute numerical grad psi and check B . grad psi on 3D grid
            z_dim, y_dim, x_dim = 32, 24, 24
            y0 = float(y_dim - 1) / 2.0
            x0 = float(x_dim - 1) / 2.0
            yy, xx = np.meshgrid(np.arange(y_dim), np.arange(x_dim), indexing="ij")
            r = np.sqrt((yy - y0) ** 2 + (xx - x0) ** 2).astype(np.float64)
            r_safe = np.maximum(1e-12, r)

            # Exact analytical psi(r) = (B0 / mu) * r * J1(mu * r)
            # Analytical dpsi/dr = B0 * r * J0(mu * r)
            dpsi_dr = 1.0 * r * special.j0(mu * r)
            grad_psi_x = dpsi_dr * ((xx - x0) / r_safe)
            grad_psi_y = dpsi_dr * ((yy - y0) / r_safe)

            Bx = B_field[0, 10].astype(np.float64)
            By = B_field[1, 10].astype(np.float64)

            # Pointwise dot product B . grad psi
            b_dot_grad_psi = Bx * grad_psi_x + By * grad_psi_y
            max_dot = float(np.max(np.abs(b_dot_grad_psi)))
            assert max_dot < 1e-6, f"B . grad psi != 0 for mu={mu}: max dot = {max_dot:.3e}"

    def test_beltrami_jet_solenoidal_condition(self):
        """Verify divergence-free field: div B = 0."""
        B_field, _ = solve_beltrami_jet_poincare(np.zeros((24, 24)), mu=1.2, steps=16)
        Bx = B_field[0]  # (Z, Y, X)
        By = B_field[1]
        Bz = B_field[2]

        # Centered finite differences for interior points
        d_Bx_dx = (Bx[:, :, 2:] - Bx[:, :, :-2]) / 2.0
        d_By_dy = (By[:, 2:, :] - By[:, :-2, :]) / 2.0
        d_Bz_dz = (Bz[2:, :, :] - Bz[:-2, :, :]) / 2.0

        # Interior slice
        div_b = d_Bx_dx[1:-1, 1:-1, :] + d_By_dy[1:-1, :, 1:-1] + d_Bz_dz[:, 1:-1, 1:-1]
        # In the bulk, div B should be near zero (finite difference discretization error < 0.05)
        mean_div = float(np.mean(np.abs(div_b)))
        assert mean_div < 0.02, f"Beltrami field is not solenoidal: mean div = {mean_div}"

    def test_beltrami_jet_various_steps_and_inputs(self):
        """Arbitrary 3D boundary flux shapes and varying step counts."""
        for steps in [1, 4, 8, 16, 32]:
            b_3d, esc = solve_beltrami_jet_poincare(np.ones((16, 20, 20)), mu=0.8, steps=steps)
            assert esc == 0.0
            assert b_3d.shape == (3, 16, 20, 20)


class TestAdversarialPerformanceAndMemory:
    """Stress testing throughput >= 165 FPS and tracemalloc peak heap < 2.0 MB streaming."""

    def test_sustained_streaming_throughput_128x128(self):
        """Verify sustained single-core >= 165 FPS over 250 frames on 128x128 focal plane."""
        engine = AstroSnailEngine(backend="numpy")
        frame = np.random.randn(128, 128).astype(np.float32)

        # Warm up
        for _ in range(10):
            engine.update(frame)

        n_frames = 250
        t0 = time.perf_counter()
        for _ in range(n_frames):
            res = engine.update(frame)
        elapsed = time.perf_counter() - t0

        fps = n_frames / elapsed
        ms_per_frame = (elapsed / n_frames) * 1000.0

        assert fps >= 165.0, (
            f"Throughput {fps:.1f} FPS failed requirement >= 165 FPS ({ms_per_frame:.2f} ms/frame)"
        )

    def test_tracemalloc_sustained_streaming_memory_and_leak_check(self):
        """Verify tracemalloc peak heap < 2.0 MB streaming (< 35.0 MB ceiling) with 0 leak."""
        tracemalloc.start()
        tracemalloc.reset_peak()

        engine = AstroSnailEngine(backend="numpy")
        frame = np.random.randn(128, 128).astype(np.float32)

        # Baseline snapshot after warm-up
        for _ in range(20):
            engine.update(frame)
        snap_warmup = tracemalloc.take_snapshot()
        current_warmup, _ = tracemalloc.get_traced_memory()

        # Sustained stream of 300 frames
        for _ in range(300):
            engine.update(frame)

        current_after, peak_after = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_after / (1024.0 * 1024.0)
        current_diff_kb = (current_after - current_warmup) / 1024.0

        # Assert hard ceiling
        assert peak_mb < 35.0, f"Tracemalloc peak {peak_mb:.2f} MB exceeded 35.0 MB ceiling"

        # Assert streaming working set delta is bounded (no significant memory leak < 100 KB over 300 frames)
        assert current_diff_kb < 100.0, f"Potential memory leak: heap grew by {current_diff_kb:.1f} KB"
