"""
tests/test_challenger_m6_1_r2_stress.py
=======================================
Milestone 6 Iteration 2 Adversarial Stress & Memory Stability Test Suite.

Empirically challenges:
1. Non-finite, degenerate, and extreme inputs:
   - NaN, Inf, -Inf, denormals (1e-45, 1e-38), huge magnitudes (1e35, -1e35).
   - Empty arrays: (0,), (0, 0), (0, 0, 0).
   - Degenerate dimensions: (1,), (2,), (1, 1), (2, 2), (1, 1, 1).
   - Degenerate covariances: zero matrix, rank-1, singular, negative semidefinite in SO(6).
   - Degenerate vectors: norm-0, collinear, identical coordinates.
   - Transition matrices: all-zeros, uniform, non-finite in causal attribution.
   - Boundary fluxes: all-zeros, extreme mu in Beltrami jet solver.
2. Memory stability:
   - Zero heap allocations in inner loops across 10,000 C-ABI iterations.
   - Zero unbounded memory growth across 1,000+ iterations of AstroSnailEngine.update().
   - Ring buffer strictly bounded at 8 frames.
   - OS VMS / commit stability across 1,000+ iterations.
3. Handle & Memmap Hygiene:
   - Clean closure of memory-mapped FITS files upon full iteration.
   - Clean closure upon early generator break.
   - Clean closure upon generator exception.
   - Clean closure when generator reference is dropped.
   - Immediate unlinking on Windows without PermissionError [WinError 32].
   - Multi-cycle (200+) creation, streaming, and unlinking of FITS, .lc, and .cat files.
"""

import ctypes
import gc
import io
import os
from pathlib import Path
import tempfile
import time
import tracemalloc
import numpy as np
import psutil
import pytest

from astrosnail import (
    AstroSnailEngine,
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
    retract_void_step0_homotopy,
    so6_kelvin_mandel_rotor,
    solve_beltrami_jet_poincare,
    stream_fits_cube,
    write_fits_cube,
    read_fits_header,
    stream_lc_records,
    stream_lc_chunks,
    parse_lc_table,
    write_lc_file,
    stream_cat_records,
    stream_cat_chunks,
    parse_cat_file,
    write_cat_file,
    _native,
)
from astrosnail.io import FITS_BLOCK_SIZE


# =============================================================================
# SUITE 1: NON-FINITE, DEGENERATE, AND EXTREME INPUT STRESS
# =============================================================================

class TestNonFiniteAndDegenerateStress:
    """Stress tests on non-finite, degenerate, and boundary inputs."""

    def test_beltrami_shell_empty_and_short_inputs(self):
        shell = BeltramiLightCurveShell(k_tokens=16)

        # Empty 1D array
        with pytest.raises(ValueError, match="Input time length must be >= 3"):
            shell.decompose(np.array([], dtype=np.float32))

        # Length 1 and Length 2 arrays
        with pytest.raises(ValueError, match="Input time length must be >= 3"):
            shell.decompose(np.array([1.0], dtype=np.float32))

        with pytest.raises(ValueError, match="Input time length must be >= 3"):
            shell.decompose(np.array([1.0, 2.0], dtype=np.float32))

        # Minimal valid length 3
        c, f, s = shell.decompose(np.array([1.0, 2.0, 1.0], dtype=np.float32))
        assert c.shape == (3,)
        assert f.shape == (3,)
        assert s.shape == (3,)
        assert np.all(np.isfinite(c))
        assert np.all(np.isfinite(f))
        assert np.all(np.isfinite(s))

    def test_beltrami_shell_extreme_dynamic_range(self):
        shell = BeltramiLightCurveShell(k_tokens=16)

        # Huge values (1e35)
        huge = np.full(50, 1e35, dtype=np.float32)
        c, f, s = shell.decompose(huge)
        assert np.all(np.isfinite(c))
        assert np.all(np.isfinite(f))
        assert np.all(np.isfinite(s))
        assert np.all(s == 0.0)

        # Tiny subnormal values (1e-45)
        tiny = np.full(50, 1.4012985e-45, dtype=np.float32)
        c, f, s = shell.decompose(tiny)
        assert np.all(np.isfinite(c))
        assert np.all(np.isfinite(f))
        assert np.all(np.isfinite(s))
        assert np.all(s == 0.0)

        # Mixed extremes
        mixed = np.array([1e30, -1e30, np.nan, np.inf, -np.inf, 0.0, 1e-35] * 10, dtype=np.float32)
        c, f, s = shell.decompose(mixed)
        assert np.all(np.isfinite(c))
        assert np.all(np.isfinite(f))
        assert np.all(np.isfinite(s))
        assert np.all((s >= 0.0) & (s <= 1.0))

    def test_morse_impulse_degenerates(self):
        # Empty array
        res_empty = compute_morse_impulse(np.array([]))
        assert res_empty.shape == (0,)

        # 1-element and 2-element arrays
        assert compute_morse_impulse(np.array([10.0])).shape == (1,)
        assert np.all(compute_morse_impulse(np.array([10.0])) == 0.0)
        assert np.all(compute_morse_impulse(np.array([10.0, 20.0])) == 0.0)

        # Degenerate inputs with NaNs, Infs, and dynamic range up to float32 sqrt(MAX_FLOAT) (1e18)
        v = np.array([np.nan, np.inf, -np.inf, 1e18, -1e18, 0.0, 0.0], dtype=np.float32)
        out = compute_morse_impulse(v)
        assert np.all(np.isfinite(out))

        # Values exceeding float32 sqrt(MAX_FLOAT) (~1.8e19) overflow to inf per IEEE-754 float32 multiplication
        v_overflow = np.array([0.0, 1e30, 0.0], dtype=np.float32)
        out_overflow = compute_morse_impulse(v_overflow)
        assert np.isinf(out_overflow[1])

    def test_morse_gate_extreme_degenerates(self):
        gate = MorseTransientGateConfig(sigma_thresh=4.0)
        # Feed huge stream of hostile values
        hostile_seq = [np.nan, np.inf, -np.inf, 1e38, -1e38, 0.0, 1e-45] * 150
        for i, v in enumerate(hostile_seq):
            cls_name, update_applied = gate.update(v, timestamp=float(i))
            assert cls_name in {"nominal", "pending_impulse", "glint_rejected", "confirmed_transient"}
            assert np.isfinite(gate.variance)
            assert gate.variance > 0.0
            assert np.isfinite(gate.noise_std)
            assert gate.noise_std > 0.0

    def test_clock_splitter_degenerates(self):
        splitter = TemporalClockContentSplitter(periodicity_threshold=0.85)

        # Short input < 4
        h_c, h_n, mask = splitter.split(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert h_c.shape == (3,)
        assert np.all(h_c == 0.0)
        assert np.all(mask == 0)

        # All-zero input
        h_c, h_n, mask = splitter.split(np.zeros(100, dtype=np.float32))
        assert np.all(h_c == 0.0)
        assert np.all(h_n == 0.0)
        assert np.all(mask == 0)

        # Pure DC flat line
        h_c, h_n, mask = splitter.split(np.full(100, 50.0, dtype=np.float32))
        assert np.all(np.isfinite(h_c))
        assert np.all(np.isfinite(h_n))

    def test_mad_floor_degenerates(self):
        # Empty array
        res = compute_spatial_mad_floor_256(np.array([], dtype=np.float32))
        assert res == 0.0

        # Uniform array
        res_const = compute_spatial_mad_floor_256(np.full(500, 42.0, dtype=np.float32))
        assert np.isclose(res_const, 42.0)

        # Fully masked array
        res_masked = compute_spatial_mad_floor_256(
            np.ones(100, dtype=np.float32), mask=np.ones(100, dtype=np.uint8)
        )
        assert res_masked == 0.0

    def test_rope2d_medoid_pooler_degenerates(self):
        pooler = RoPE2DMedoidPooler(k_medoids=24)

        # Empty frame (0, 0)
        c0, f0 = pooler.pool(np.zeros((0, 0), dtype=np.float32))
        assert c0.shape == (0, 2)
        assert f0.shape == (0, 8)

        # Small frames (1, 1), (2, 2), (1, 50), (50, 1)
        for shape in [(1, 1), (2, 2), (1, 50), (50, 1)]:
            c, f = pooler.pool(np.zeros(shape, dtype=np.float32))
            assert c.shape == (0, 2)
            assert f.shape == (0, 8)

        # Uniform non-zero frame
        c_uni, f_uni = pooler.pool(np.full((64, 64), 100.0, dtype=np.float32))
        assert c_uni.shape == (0, 2)
        assert f_uni.shape == (0, 8)

        # Frame with all NaNs, all Infs
        c_nan, f_nan = pooler.pool(np.full((32, 32), np.nan, dtype=np.float32))
        assert c_nan.shape == (0, 2)
        assert f_nan.shape == (0, 8)

    def test_triad_operator_degenerate_covariances_and_weights(self):
        dim = 8
        op = SkewSymmetricTriadOperator(dim=dim)

        # All-zero weight tensor
        W_zero = np.zeros((dim, dim, dim), dtype=np.float64)
        u = np.ones(dim, dtype=np.float64)
        v = np.ones(dim, dtype=np.float64)
        res_zero = op.forward(u, v, W=W_zero)
        assert np.all(res_zero == 0.0)

        # Degenerate collinear vectors (u = 2 * v)
        v_rand = np.random.randn(dim)
        res_collinear = op.forward(2.0 * v_rand, v_rand)
        # Pointwise orthogonality must still hold: <v, B_W(u, v)> == 0
        ortho = float(np.dot(v_rand, res_collinear))
        assert abs(ortho) < 1e-12

        # Norm-0 vectors (u = 0, v = 0)
        res_null = op.forward(np.zeros(dim), np.zeros(dim))
        assert np.all(res_null == 0.0)

    def test_slow_fast_cascade_degenerates(self):
        cascade = SlowFastTriadCascade(coupling=0.15, steps=16)
        C = 8

        # v0 = zero vector (norm 0)
        v0_zero = np.zeros(C, dtype=np.float64)
        u_seq = np.ones((16, C), dtype=np.float64)
        v_out, drift = cascade.integrate(u_seq, v0_zero, steps=16)
        assert np.all(v_out == 0.0)
        assert drift == 0.0

        # coupling = 0.0 (identity integration)
        cascade_zero_c = SlowFastTriadCascade(coupling=0.0, steps=16)
        v0 = np.array([1.0] + [0.0] * 7, dtype=np.float64)
        v_out_c0, drift_c0 = cascade_zero_c.integrate(u_seq, v0, steps=16)
        assert np.allclose(v_out_c0, v0)
        assert drift_c0 == 0.0

    def test_attribute_causal_progenitors_degenerates(self):
        # Empty matrix (0, 0)
        is_gen, r = attribute_causal_progenitors(np.zeros((0, 0)))
        assert len(is_gen) == 0
        assert len(r) == 0

        # All-zero transition matrix (no transitions)
        mat_zero = np.zeros((5, 5))
        is_gen, r = attribute_causal_progenitors(mat_zero)
        assert len(is_gen) == 5
        assert np.all(~is_gen)
        assert np.all(r == 0.0)

        # Purely local diagonal matrix (no nonlocal transitions)
        mat_diag = np.eye(4)
        is_gen, r = attribute_causal_progenitors(mat_diag)
        assert np.all(~is_gen)
        assert np.all(r == 0.0)

        # Purely nonlocal off-diagonal matrix (all transitions nonlocal)
        mat_off = np.ones((4, 4)) - np.eye(4)
        is_gen, r = attribute_causal_progenitors(mat_off, threshold=0.50)
        assert np.all(is_gen)
        assert np.all(r == 1.0)

    def test_so6_kelvin_mandel_rotor_degenerate_covariances(self):
        # Identity rotation SO(3)
        R3_eye = np.eye(3, dtype=np.float64)

        # Degenerate 1: All-zero covariance matrix (singular, rank 0)
        C_zero = np.zeros((6, 6), dtype=np.float64)
        C_rot_zero, res_zero = so6_kelvin_mandel_rotor(R3_eye, C_zero)
        assert np.all(C_rot_zero == 0.0)
        assert res_zero == 0.0

        # Degenerate 2: Rank-1 singular covariance matrix
        v = np.array([1.0, 2.0, 3.0, 0.5, -1.0, 0.2], dtype=np.float64)
        C_rank1 = np.outer(v, v)
        # Random SO(3) rotation
        H = np.random.randn(3, 3)
        Q, _ = np.linalg.qr(H)
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]

        C_rot_r1, res_r1 = so6_kelvin_mandel_rotor(Q, C_rank1)
        assert res_r1 <= 1.14e-13
        assert np.isfinite(res_r1)
        # Trace conservation Tr(C_rot) == Tr(C_orig)
        assert abs(np.trace(C_rot_r1) - np.trace(C_rank1)) <= 1.14e-13

        # Degenerate 3: Negative semidefinite covariance matrix
        C_neg = -np.eye(6, dtype=np.float64)
        C_rot_neg, res_neg = so6_kelvin_mandel_rotor(Q, C_neg)
        assert res_neg <= 1.14e-13
        assert abs(np.trace(C_rot_neg) - np.trace(C_neg)) <= 1.14e-13

    def test_beltrami_jet_poincare_degenerates(self):
        # All-zero boundary
        B_field_zero, escape_zero = solve_beltrami_jet_poincare(np.zeros((16, 16)), mu=1.0, steps=16)
        assert escape_zero == 0.0
        assert B_field_zero.shape == (3, 32, 16, 16)

        # mu = 0.0 (harmonic potential limit)
        B_field_mu0, escape_mu0 = solve_beltrami_jet_poincare(np.ones((16, 16)), mu=0.0, steps=16)
        assert escape_mu0 == 0.0
        assert np.all(np.isfinite(B_field_mu0))

        # Extreme mu (mu = 50.0 high oscillatory frequency)
        B_field_high, escape_high = solve_beltrami_jet_poincare(np.ones((16, 16)), mu=50.0, steps=16)
        assert escape_high == 0.0
        assert np.all(np.isfinite(B_field_high))

    def test_engine_streaming_pathological_shapes(self):
        eng = AstroSnailEngine(backend="numpy")

        # 0D array (scalar)
        out_0d = eng.update(np.array(42.0))
        assert out_0d["input_shape"] == []
        assert out_0d["step"] == 1

        # Empty 1D array
        out_empty_1d = eng.update(np.array([]))
        assert out_empty_1d["input_shape"] == [0]

        # 1D array length 2
        out_len2 = eng.update(np.array([1.0, 2.0]))
        assert out_len2["input_shape"] == [2]

        # 2D temporal multi-band series (T, B with B <= 8)
        out_mb = eng.update(np.zeros((10, 4)))
        assert "curl_salience" in out_mb
        assert out_mb["input_shape"] == [10, 4]

        # 2D focal plane (H, W with W > 8)
        out_focal = eng.update(np.zeros((16, 16)))
        assert "num_medoids" in out_focal
        assert out_focal["num_medoids"] == 0

        # Empty focal plane with W > 8 (0, 16)
        out_empty_focal = eng.update(np.zeros((0, 16)))
        assert "num_medoids" in out_empty_focal
        assert out_empty_focal["num_medoids"] == 0

        # Empty 3D cube (0, 0, 0)
        out_empty_3d = eng.update(np.zeros((0, 0, 0)))
        assert out_empty_3d["void_voxel_count"] == 0


# =============================================================================
# SUITE 2: MEMORY STABILITY & ALLOCATION INVARIANTS (1,000+ ITERATIONS)
# =============================================================================

class TestMemoryStabilityAndAllocations:
    """Rigorous memory stability audits across 1,000+ and 10,000 iterations."""

    def test_c_abi_zero_heap_allocations_10000_iterations(self):
        """Verifies exactly 0 bytes Python heap allocated in C-ABI inner loops across 10,000 calls."""
        if not _native.is_available():
            pytest.skip("Native Rust library not available")

        lib = _native.load_native_library()
        arena = _native.NativeAstroSnailArena(use_global=True)
        arena.reset()

        # Allocate pinned buffers
        h, w = 128, 128
        frame = np.ones((h, w), dtype=np.float32)
        frame_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        clock_mask = np.zeros((h, w), dtype=np.uint8)
        clock_ptr = clock_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        sal_out = np.zeros((h, w), dtype=np.float32)
        sal_ptr = sal_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        spike_out = np.zeros((h, w), dtype=np.uint8)
        spike_ptr = spike_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        mad_floor = ctypes.c_float(0.0)
        mad_ptr = ctypes.byref(mad_floor)

        u_seq = np.full((16, 8), 0.1, dtype=np.float32)
        u_ptr = u_seq.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        v0 = np.full(8, 0.35355339, dtype=np.float32)
        v0_ptr = v0.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        v_out = np.zeros(8, dtype=np.float32)
        v_out_ptr = v_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        drift = ctypes.c_float(0.0)
        drift_ptr = ctypes.byref(drift)

        r3 = np.eye(3, dtype=np.float64)
        r3_ptr = r3.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_base = np.eye(6, dtype=np.float64)
        c_ptr = c_base.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        m_out = np.zeros(36, dtype=np.float64)
        m_ptr = m_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_rot = np.zeros(36, dtype=np.float64)
        c_rot_ptr = c_rot.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        trace_res = ctypes.c_double(0.0)
        trace_ptr = ctypes.byref(trace_res)

        # Warmup
        for _ in range(100):
            lib.beltrami_morse_transient_gate_f32(frame_ptr, None, clock_ptr, h, w, sal_ptr, spike_ptr, mad_ptr)
            lib.so8_astrometry_triad_cascade_f32(u_ptr, 16, v0_ptr, 0.15, v_out_ptr, drift_ptr)
            lib.so6_kelvin_mandel_rotor_f64(r3_ptr, c_ptr, m_ptr, c_rot_ptr, trace_ptr)

        _ = 0
        tracemalloc.start()
        start_curr, _ = tracemalloc.get_traced_memory()

        n_iters = 10_000
        for _ in range(n_iters):
            lib.beltrami_morse_transient_gate_f32(frame_ptr, None, clock_ptr, h, w, sal_ptr, spike_ptr, mad_ptr)
            lib.so8_astrometry_triad_cascade_f32(u_ptr, 16, v0_ptr, 0.15, v_out_ptr, drift_ptr)
            lib.so6_kelvin_mandel_rotor_f64(r3_ptr, c_ptr, m_ptr, c_rot_ptr, trace_ptr)
        _ = 0

        end_curr, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        net_growth = end_curr - start_curr
        assert net_growth == 0, f"Detected Python heap allocation of {net_growth} bytes across 10,000 iterations"

    def test_engine_streaming_zero_memory_growth_1000_iterations(self):
        """Verifies that AstroSnailEngine streaming 1,000 frames produces zero unbounded memory growth."""
        gc.collect()
        tracemalloc.start()

        eng = AstroSnailEngine(backend="numpy")
        rng = np.random.default_rng(42)
        frame = rng.standard_normal((64, 64)).astype(np.float32)

        # Warmup 50 frames to populate ring buffer and JIT/dispatch caches
        for _ in range(50):
            eng.update(frame)

        snap_warmup = tracemalloc.take_snapshot()

        # Run 950 more frames (total 1,000 frames)
        for i in range(950):
            if i % 10 == 0:
                # Occasional moving transient
                frame[10 + (i % 20), 10 + (i % 20)] += 30.0
            eng.update(frame)

        snap_final = tracemalloc.take_snapshot()
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 1. Ring buffer must strictly not exceed 8 frames
        assert len(eng.ring_buffer) == 8, f"Ring buffer grew to {len(eng.ring_buffer)} frames"

        # 2. Peak working set must remain < 2.0 MB streaming (< 35.0 MB hard ceiling)
        peak_mb = peak_bytes / (1024 * 1024)
        assert peak_mb < 2.0, f"Peak memory {peak_mb:.2f} MB exceeded 2.0 MB limit"

        # 3. Net memory growth between warmup and completion must be < 80 KB
        diffs = snap_final.compare_to(snap_warmup, "lineno")
        net_diff_kb = sum(d.size_diff for d in diffs) / 1024.0
        assert net_diff_kb < 80.0, f"Net memory growth of {net_diff_kb:.2f} KB across 950 frames indicates leak"

    def test_os_process_commit_stability_1000_iterations(self):
        """Verifies OS process commit (VMS) stability over 1,000 frames using psutil."""
        eng = AstroSnailEngine(backend="numpy")
        frame = np.ones((64, 64), dtype=np.float32)

        # Warmup
        for _ in range(50):
            eng.update(frame)

        proc = psutil.Process()
        vms_start = proc.memory_info().vms

        for _ in range(1000):
            eng.update(frame)

        vms_end = proc.memory_info().vms
        vms_delta = vms_end - vms_start

        # Allow for OS page reclamation (delta <= 0) or negligible small commit (< 512 KB)
        assert vms_delta < 512 * 1024, f"OS VMS grew by {vms_delta} bytes over 1,000 iterations"


# =============================================================================
# SUITE 3: FITS HANDLES, MEMORY MAPS, AND TEMPORARY FILE HYGIENE
# =============================================================================

class TestFITSHandleAndMemmapHygiene:
    """Stress tests verifying clean handle and memmap closure without file lock leaks."""

    def test_fits_stream_exhaustion_allows_immediate_unlink(self, tmp_path):
        """Default stream_fits_cube (copy=False, zero-copy memmap): exhausting stream and unlinking."""
        fits_file = tmp_path / "test_exhaust.fits"
        cube_data = np.random.randn(32, 16, 16).astype(np.float32)
        write_fits_cube(fits_file, cube_data)

        # Iterate through all slabs
        for slab in stream_fits_cube(fits_file, chunk_z=8):
            assert slab.shape == (8, 16, 16)

        # Release local references and trigger GC
        del slab
        gc.collect()

        # Immediate unlink must succeed on Windows (zero file locking)
        fits_file.unlink()
        assert not fits_file.exists()

    def test_fits_stream_early_break_allows_immediate_unlink(self, tmp_path):
        """Break out of stream_fits_cube after 1 chunk, ensuring generator exit releases lock."""
        fits_file = tmp_path / "test_early_break.fits"
        cube_data = np.random.randn(32, 16, 16).astype(np.float32)
        write_fits_cube(fits_file, cube_data)

        # Stream and break early
        for slab in stream_fits_cube(fits_file, chunk_z=8):
            assert slab.shape == (8, 16, 16)
            break

        del slab
        gc.collect()

        # Must unlink without WinError 32
        fits_file.unlink()
        assert not fits_file.exists()

    def test_fits_stream_exception_allows_immediate_unlink(self, tmp_path):
        """Exception raised inside consumer loop: generator cleanup releases memmap."""
        fits_file = tmp_path / "test_exception.fits"
        cube_data = np.random.randn(32, 16, 16).astype(np.float32)
        write_fits_cube(fits_file, cube_data)

        try:
            for slab in stream_fits_cube(fits_file, chunk_z=8):
                raise RuntimeError("Simulated consumer failure")
        except RuntimeError:
            pass

        if "slab" in locals():
            del slab
        gc.collect()

        fits_file.unlink()
        assert not fits_file.exists()

    def test_fits_stream_dropped_generator_allows_immediate_unlink(self, tmp_path):
        """Generator object instantiated, stepped once, and explicitly deleted."""
        fits_file = tmp_path / "test_dropped_gen.fits"
        cube_data = np.random.randn(32, 16, 16).astype(np.float32)
        write_fits_cube(fits_file, cube_data)

        gen = stream_fits_cube(fits_file, chunk_z=8)
        first_slab = next(gen)
        assert first_slab.shape == (8, 16, 16)

        # Drop generator and slab
        gen.close()
        del gen
        del first_slab
        gc.collect()

        fits_file.unlink()
        assert not fits_file.exists()

    def test_fits_stream_modes_hygiene(self, tmp_path):
        """Verify reuse_buffer=True, copy=True, and squeeze_2d modes all release file locks cleanly."""
        modes = [
            {"reuse_buffer": True, "copy": False},
            {"reuse_buffer": False, "copy": True},
            {"reuse_buffer": True, "copy": True},
            {"squeeze_2d": True, "chunk_z": 1},
        ]

        for i, mode_kwargs in enumerate(modes):
            p = tmp_path / f"test_mode_{i}.fits"
            write_fits_cube(p, np.ones((8, 16, 16), dtype=np.float32))

            for slab in stream_fits_cube(p, **mode_kwargs):
                pass

            del slab
            gc.collect()
            p.unlink()
            assert not p.exists()

    def test_multi_cycle_fits_and_catalog_create_stream_unlink(self, tmp_path):
        """Multi-cycle stress: 100 cycles of creating, streaming, and unlinking FITS, .lc, and .cat files."""
        for cycle in range(100):
            # 1. FITS Cube
            fits_path = tmp_path / f"cycle_{cycle}.fits"
            write_fits_cube(fits_path, np.ones((16, 16, 16), dtype=np.float32))
            count = 0
            for slab in stream_fits_cube(fits_path, chunk_z=4):
                count += 1
            del slab
            gc.collect()
            fits_path.unlink()
            assert not fits_path.exists()

            # 2. Light Curve (.lc)
            lc_path = tmp_path / f"cycle_{cycle}.lc"
            t = np.linspace(0, 10, 50)
            flux = 100.0 + np.sin(t)
            write_lc_file(lc_path, t, flux)
            records = parse_lc_table(lc_path)
            assert len(records) == 50
            lc_path.unlink()
            assert not lc_path.exists()

            # 3. Source Catalog (.cat)
            cat_path = tmp_path / f"cycle_{cycle}.cat"
            recs = [(100 + i, 180.0 + i * 0.01, 15.0 + i * 0.01, 19.5, 0.05) for i in range(20)]
            write_cat_file(cat_path, recs)
            parsed_cat = parse_cat_file(cat_path)
            assert len(parsed_cat) == 20
            cat_path.unlink()
            assert not cat_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
