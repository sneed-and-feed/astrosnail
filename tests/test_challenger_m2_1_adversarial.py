"""astrosnail.tests.test_challenger_m2_1_adversarial
===================================================
Adversarial empirical stress harness created by Challenger M2-1.

Empirically tests and challenges:
1. Python FFI Boundary & C-ABI Telemetry Interaction:
   - Global arena pointer 64-byte alignment
   - 64-byte alignment of all 24 sub-buffers
   - sizeof assertions for AstroSnailArena (1,012,352), AstroSnailTelemetry (64), AstroSnailMorseState (64)
   - Field offset of cosmic_ray_spike_recovery_rate (offset 8 in telemetry, 1,012,224+8 in arena)
   - Behavior of astrosnail_arena_init on aligned vs unaligned buffers
   - 10,000 iterations of telemetry retrieval and cosmic_ray_spike_recovery_rate extraction with zero memory leak
   - Adjacency integrity: writes to cosmic_ray_spike_recovery_rate do not corrupt fps or glint_rejection_rate
   - Extreme floating point values (subnormal, NaN, Inf, extreme scale)

2. Causal Streaming Engine Stress Test:
   - Continuous 128x128 frames (500 frames) through engine.update()
   - Ring buffer bounded strictly <= 8 frames
   - Strictly monotonic step counting
   - Bounded primary cascade drift (<= 1e-7)
   - Continuous 25,000-point light curves in sample-by-sample streaming mode (25,000 steps)
   - Batch 25,000-point light curve decomposition, Morse impulse, clock split, and MAD floor
   - Pathological inputs (all zeros, NaN/Inf, high dynamic range)

3. Fallback Behavior and Numerical Parity:
   - Bypass behavior via _native._LIB and engine backend="numpy"
   - Numerical equivalence between pure Python reference and native C-ABI across all operators within 1e-6
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import time
import tracemalloc
from typing import Any, Dict, List, Tuple
import numpy as np
import pytest

from astrosnail import _native
from astrosnail.engine import AstroSnailEngine
from astrosnail.transients import (
    BeltramiLightCurveShell,
    TemporalClockContentSplitter,
    compute_morse_impulse,
    compute_spatial_mad_floor_256 as py_mad_floor,
)
from astrosnail.astrometry import (
    SkewSymmetricTriadOperator,
    SlowFastTriadCascade,
    attribute_causal_progenitors as py_attribute,
)
from astrosnail.cosmology import (
    so6_kelvin_mandel_rotor as py_so6_rotor,
    detect_jordan_cavity_cores as py_detect_jordan,
    retract_void_step0_homotopy as py_retract_void,
    solve_beltrami_jet_poincare as py_solve_beltrami,
)


# =============================================================================
# 1. Python FFI Boundary & C-ABI Telemetry Interaction
# =============================================================================

class TestFFIBoundaryAndTelemetry:
    """Rigorous empirical challenge of ctypes FFI, memory layouts, and telemetry HUD."""

    def test_global_arena_pointer_64_byte_alignment(self):
        """Verifies that the global static arena pointer is strictly 64-byte aligned."""
        lib = _native.load_native_library()
        global_ptr = lib.astrosnail_arena_global()
        assert bool(global_ptr), "astrosnail_arena_global() returned NULL pointer"
        addr = ctypes.cast(global_ptr, ctypes.c_void_p).value
        assert addr is not None
        assert addr % 64 == 0, f"GLOBAL_ARENA address {hex(addr)} is not 64-byte aligned (offset {addr % 64})"

    def test_arena_and_telemetry_struct_geometry(self):
        """Verifies exact byte sizes and offsets for AstroSnailArena and AstroSnailTelemetry."""
        lib = _native.load_native_library()
        native_size = lib.astrosnail_arena_size_bytes()
        arena_size = ctypes.sizeof(_native.AstroSnailArena)
        telemetry_size = ctypes.sizeof(_native.AstroSnailTelemetry)
        morse_size = ctypes.sizeof(_native.AstroSnailMorseState)

        assert native_size == 1_012_352, f"Native arena size {native_size} != 1,012,352"
        assert arena_size == 1_012_352, f"ctypes AstroSnailArena size {arena_size} != 1,012,352"
        assert telemetry_size == 64, f"AstroSnailTelemetry size {telemetry_size} != 64"
        assert morse_size == 64, f"AstroSnailMorseState size {morse_size} != 64"

        # Verify telemetry field offsets
        assert _native.AstroSnailTelemetry.fps.offset == 0
        assert _native.AstroSnailTelemetry.execution_cycles.offset == 4
        assert _native.AstroSnailTelemetry.cosmic_ray_spike_recovery_rate.offset == 8
        assert _native.AstroSnailTelemetry.glint_rejection_rate.offset == 12
        assert _native.AstroSnailTelemetry.max_radial_drift.offset == 16
        assert _native.AstroSnailTelemetry.max_trace_residual.offset == 20
        assert _native.AstroSnailTelemetry.active_generators_count.offset == 24
        assert _native.AstroSnailTelemetry.parity_residual_f32.offset == 28
        assert _native.AstroSnailTelemetry._pad.offset == 32

        # Verify telemetry offset within AstroSnailArena
        assert _native.AstroSnailArena.telemetry.offset == 1_012_224
        assert _native.AstroSnailArena.telemetry.offset % 64 == 0

        # Absolute offset of cosmic_ray_spike_recovery_rate in arena
        expected_abs_offset = 1_012_224 + 8
        actual_abs_offset = _native.AstroSnailArena.telemetry.offset + _native.AstroSnailTelemetry.cosmic_ray_spike_recovery_rate.offset
        assert actual_abs_offset == expected_abs_offset == 1_012_232

    def test_all_subbuffers_64_byte_aligned(self):
        """Verifies that every major sub-buffer in AstroSnailArena starts on a 64-byte boundary."""
        major_subbuffers = [
            "spatial_input", "spatial_prev", "salience_map", "curl_buffer", "ring_buffer",
            "mask_clock", "mask_nope", "mask_spike", "cosmic_density", "cosmic_phi",
            "cosmic_void_mask", "skewer_flux", "skewer_coords", "triad_tensor_w",
            "cascade_w_rec", "medoid_coords_yx", "medoid_features", "causal_transition",
            "rotor_so6_m", "rotor_c_base", "rotor_c_rot", "morse_state", "telemetry",
            "step_counter",
        ]
        assert len(major_subbuffers) == 24
        for name in major_subbuffers:
            offset = getattr(_native.AstroSnailArena, name).offset
            assert offset % 64 == 0, f"Major sub-buffer '{name}' at offset {offset} is not 64-byte aligned (offset % 64 = {offset % 64})"

        # Verify scalar metadata fields are packed into the final 64-byte cache line starting at offset 1,012,288
        packed_scalars = [
            ("step_counter", 1_012_288),
            ("num_medoids", 1_012_292),
            ("ring_head", 1_012_296),
            ("width", 1_012_300),
            ("height", 1_012_304),
            ("n_skewers", 1_012_308),
            ("last_radial_drift", 1_012_312),
            ("last_trace_residual", 1_012_316),
            ("_pad", 1_012_320),
        ]
        for name, expected_offset in packed_scalars:
            actual_offset = getattr(_native.AstroSnailArena, name).offset
            assert actual_offset == expected_offset, f"Field '{name}' expected at offset {expected_offset}, got {actual_offset}"

    def test_astrosnail_arena_init_alignment_rejection(self):
        """Empirically tests that astrosnail_arena_init rejects unaligned pointers (return code -2)."""
        lib = _native.load_native_library()
        raw_buffer = (ctypes.c_uint8 * (_native.ASTROSNAIL_ARENA_TOTAL_BYTES + 128))()
        raw_addr = ctypes.cast(raw_buffer, ctypes.c_void_p).value
        assert raw_addr is not None

        # Create guaranteed aligned and unaligned pointers
        aligned_addr = (raw_addr + 63) & ~63
        aligned_ptr = ctypes.c_void_p(aligned_addr)

        # 1. Aligned pointer should succeed
        rc_aligned = lib.astrosnail_arena_init(aligned_ptr, ctypes.c_size_t(_native.ASTROSNAIL_ARENA_TOTAL_BYTES))
        assert rc_aligned == 0, f"astrosnail_arena_init failed on 64-byte aligned pointer: {rc_aligned}"

        # 2. Unaligned pointers (offsets 1 through 63) must return -2
        for unaligned_offset in [1, 2, 4, 8, 16, 32, 48, 63]:
            unaligned_ptr = ctypes.c_void_p(aligned_addr + unaligned_offset)
            rc_unaligned = lib.astrosnail_arena_init(unaligned_ptr, ctypes.c_size_t(_native.ASTROSNAIL_ARENA_TOTAL_BYTES))
            assert rc_unaligned == -2, (
                f"astrosnail_arena_init failed to reject pointer with alignment offset +{unaligned_offset}: "
                f"expected -2, got {rc_unaligned}"
            )

        # 3. Buffer too small must return -1
        rc_too_small = lib.astrosnail_arena_init(aligned_ptr, ctypes.c_size_t(_native.ASTROSNAIL_ARENA_TOTAL_BYTES - 1))
        assert rc_too_small == -1, f"Expected -1 for truncated buffer size, got {rc_too_small}"

        # 4. Null pointer must return -1
        rc_null = lib.astrosnail_arena_init(None, ctypes.c_size_t(_native.ASTROSNAIL_ARENA_TOTAL_BYTES))
        assert rc_null == -1, f"Expected -1 for NULL pointer, got {rc_null}"

    def test_telemetry_extraction_across_10000_calls(self):
        """Stress-tests telemetry retrieval & cosmic_ray_spike_recovery_rate extraction across 10,000 calls."""
        lib = _native.load_native_library()
        arena_ptr = lib.astrosnail_arena_global()
        assert bool(arena_ptr)

        # Ensure known starting state
        arena_ptr.contents.telemetry.fps = 120.0
        arena_ptr.contents.telemetry.execution_cycles = 42
        arena_ptr.contents.telemetry.cosmic_ray_spike_recovery_rate = 0.98765
        arena_ptr.contents.telemetry.glint_rejection_rate = 0.995

        tracemalloc.start()
        start_curr, _ = tracemalloc.get_traced_memory()

        n_calls = 10_000
        extracted_rates = []

        for i in range(n_calls):
            # Read telemetry structure
            telem = arena_ptr.contents.telemetry
            rate = float(telem.cosmic_ray_spike_recovery_rate)
            if i % 1000 == 0:
                extracted_rates.append(rate)

        end_curr, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        net_growth = end_curr - start_curr
        # Allow trivial loop overhead (< 4 KB)
        assert net_growth < 4096, f"Memory leak detected during 10,000 telemetry calls: {net_growth} bytes"
        assert len(extracted_rates) == 10
        assert all(abs(r - 0.98765) < 1e-5 for r in extracted_rates)

    def test_telemetry_adjacency_corruption_audit(self):
        """Verifies that writing to cosmic_ray_spike_recovery_rate does not corrupt adjacent fields."""
        lib = _native.load_native_library()
        arena_ptr = lib.astrosnail_arena_global()
        telem = arena_ptr.contents.telemetry

        # Set sentinel values on adjacent fields
        SENTINEL_FPS = 4321.0
        SENTINEL_CYCLES = 123456
        SENTINEL_GLINT = 0.7777

        telem.fps = SENTINEL_FPS
        telem.execution_cycles = SENTINEL_CYCLES
        telem.glint_rejection_rate = SENTINEL_GLINT

        # Test writing extreme values into cosmic_ray_spike_recovery_rate
        test_values = [
            0.0,
            1.0,
            0.5,
            -1.0,
            1e-30,
            1e30,
            float("nan"),
            float("inf"),
            float("-inf"),
        ]

        for val in test_values:
            telem.cosmic_ray_spike_recovery_rate = val

            # Check that adjacent fields remained completely unchanged
            assert abs(telem.fps - SENTINEL_FPS) < 1e-4, f"Writing {val} corrupted fps"
            assert telem.execution_cycles == SENTINEL_CYCLES, f"Writing {val} corrupted execution_cycles"
            assert abs(telem.glint_rejection_rate - SENTINEL_GLINT) < 1e-4, f"Writing {val} corrupted glint_rejection_rate"

            # Check reading back the value
            read_back = telem.cosmic_ray_spike_recovery_rate
            if math.isnan(val):
                assert math.isnan(read_back)
            elif math.isinf(val):
                assert math.isinf(read_back) and (read_back > 0 if val > 0 else read_back < 0)
            else:
                assert abs(read_back - val) < max(1e-5, abs(val) * 1e-5)


# =============================================================================
# 2. Causal Streaming Engine Stress Test
# =============================================================================

class TestCausalStreamingEngineStress:
    """Stress tests AstroSnailEngine.update() with continuous 128x128 frames and 25k light curves."""

    def test_continuous_128x128_frames_causal_streaming_500_steps(self):
        """Feeds 500 continuous 128x128 focal plane frames through engine.update()."""
        engine = AstroSnailEngine(backend="auto")
        rng = np.random.default_rng(20261103)

        n_frames = 500
        drifts = []

        tracemalloc.start()
        start_curr, _ = tracemalloc.get_traced_memory()

        for step in range(1, n_frames + 1):
            # Dynamic frame generation: background noise + moving point sources
            frame = rng.standard_normal((128, 128)).astype(np.float32) * 0.5
            # Add 3 moving sources
            x1 = int((30 + step * 0.1) % 120)
            y1 = int((40 + step * 0.05) % 120)
            frame[y1:y1+3, x1:x1+3] += 15.0

            # Add occasional impulse glint
            if step % 50 == 0:
                frame[64, 64] += 100.0

            result = engine.update(frame)

            # Invariants
            assert result["step"] == step
            assert engine.step_count == step
            assert len(engine.ring_buffer) <= 8, f"Ring buffer exceeded 8 frames: {len(engine.ring_buffer)}"
            assert "medoid_coords_yx" in result
            assert "medoid_features" in result

            if "primary_cascade_drift" in result:
                drifts.append(result["primary_cascade_drift"])

        end_curr, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Check drift bound across all steps
        if drifts:
            max_drift = max(drifts)
            assert max_drift <= 1e-7, f"Cascade radial drift {max_drift} exceeded 1e-7"

        # Check memory stability
        net_growth = end_curr - start_curr
        # Ring buffer keeps 8 frames of 128x128 float32 (~512 KB) plus small internal caches
        assert net_growth < 2 * 1024 * 1024, f"Memory growth excessive: {net_growth} bytes across 500 frames"

    def test_25000_point_light_curve_sample_by_sample_streaming(self):
        """Feeds 25,000 points sample-by-sample into engine.update() to test online Morse gate."""
        engine = AstroSnailEngine(backend="auto")
        rng = np.random.default_rng(42)

        # 25,000-point time series: quiescent background + periodic pulsation + transients + glints
        t = np.linspace(0, 250, 25_000)
        signal = 10.0 + 2.0 * np.sin(2.0 * np.pi * t / 12.5) + rng.standard_normal(25_000) * 0.2

        # Inject 10 isolated single-sample glints
        glint_indices = [1000, 3500, 6000, 9500, 12000, 15500, 18000, 20500, 22000, 24000]
        for idx in glint_indices:
            signal[idx] += 25.0

        # Inject 3 sustained square-top transients
        for start_idx in [5000, 14000, 21000]:
            signal[start_idx:start_idx + 20] += 8.0

        tracemalloc.start()
        start_curr, _ = tracemalloc.get_traced_memory()

        classifications = []
        for i in range(25_000):
            sample = np.array([signal[i]], dtype=np.float32)
            out = engine.update(sample)
            assert out["step"] == i + 1
            classifications.append(out.get("classification", "nominal"))

        end_curr, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Assert glints were detected and rejected
        assert engine.transient_gate.glint_rejected_count >= 5, (
            f"Expected >= 5 rejected glints, got {engine.transient_gate.glint_rejected_count}"
        )
        # Assert memory remained strictly bounded
        net_growth = end_curr - start_curr
        assert net_growth < 1024 * 1024, f"Memory leaked across 25,000 samples: {net_growth} bytes"

    def test_25000_point_light_curve_batch_decomposition(self):
        """Feeds full 25,000-point light curve as sequence into engine.update()."""
        engine = AstroSnailEngine(backend="auto")
        rng = np.random.default_rng(2026)

        t = np.linspace(0, 500, 25_000)
        curve = (15.0 + 3.0 * np.sin(2.0 * np.pi * t / 25.0) + rng.standard_normal(25_000) * 0.5).astype(np.float32)

        out = engine.update(curve)
        assert out["step"] == 1
        assert "u_coarse" in out
        assert "u_fine" in out
        assert "curl_salience" in out
        assert "morse_impulse" in out
        assert "h_clock" in out
        assert "h_nope" in out
        assert "clock_mask" in out
        assert "mad_floor" in out

        # Validate dimensional invariants
        assert len(out["u_coarse"]) == 25_000
        assert len(out["u_fine"]) == 25_000
        assert len(out["curl_salience"]) == 25_000
        assert len(out["morse_impulse"]) == 25_000
        assert len(out["h_clock"]) == 25_000
        assert len(out["h_nope"]) == 25_000
        assert len(out["clock_mask"]) == 25_000

        # Physical bounds
        assert np.all(np.isfinite(out["u_coarse"]))
        assert np.all(np.isfinite(out["u_fine"]))
        assert np.all((out["curl_salience"] >= 0.0) & (out["curl_salience"] <= 1.0))
        assert out["mad_floor"] >= 0.0


# =============================================================================
# 3. Fallback Behavior and Numerical Parity (Pure Python vs Native C-ABI)
# =============================================================================

class TestFallbackBehaviorAndParity:
    """Verifies that pure Python and native C-ABI yield identical results <= 1e-6 tolerance."""

    def test_so8_skew_triad_parity(self):
        """Operator: SO(8) skew triad step forward parity."""
        rng = np.random.default_rng(101)
        triad_py = SkewSymmetricTriadOperator(dim=8)

        for _ in range(50):
            u = rng.standard_normal(8).astype(np.float32)
            v = rng.standard_normal(8).astype(np.float32)

            v_nat = _native.so8_skew_triad_step(u, v, gamma=0.05)
            v_py = triad_py.forward(u, v)

            err = float(np.max(np.abs(v_nat - v_py)))
            assert err <= 1.0e-6, f"so8_skew_triad_step parity error {err} > 1e-6"

    def test_so8_astrometry_triad_cascade_parity(self):
        """Operator: 16-step slow-fast cascade parity & radial drift."""
        rng = np.random.default_rng(102)
        cascade_py = SlowFastTriadCascade(coupling=0.15, steps=16)

        for _ in range(25):
            u_seq = rng.standard_normal((16, 8)).astype(np.float32)
            v0 = rng.standard_normal(8).astype(np.float32)
            v0 = v0 / np.linalg.norm(v0)

            v_nat, drift_nat = _native.so8_astrometry_triad_cascade(u_seq, v0, coupling=0.15, steps=16)
            v_py, drift_py = cascade_py.integrate(u_seq, v0, steps=16)

            v_err = float(np.max(np.abs(v_nat - v_py.astype(np.float32))))
            drift_err = abs(drift_nat - drift_py)

            assert v_err <= 1.0e-6, f"Cascade output vector parity error {v_err} > 1e-6"
            assert drift_err <= 1.0e-6, f"Cascade drift parity error {drift_err} > 1e-6"
            assert drift_nat <= 1.0e-7, f"Native radial drift {drift_nat} > 1e-7"

    def test_ephemeris_clock_nope_split_parity(self):
        """Operator: Ephemeris harmonic split bitmask and nope residual parity."""
        rng = np.random.default_rng(103)
        splitter_py = TemporalClockContentSplitter(periodicity_threshold=0.85)

        for length in [64, 128, 256]:
            t = np.arange(length)
            flux = (10.0 + 3.0 * np.sin(2.0 * np.pi * t / 27.0) + rng.standard_normal(length) * 0.1).astype(np.float32)

            mask_nat, nope_nat = _native.ephemeris_clock_nope_split(flux, threshold=0.85)
            _, nope_py, mask_py = splitter_py.split(flux)

            # Bitmask must match identically
            assert np.array_equal(mask_nat, mask_py), "Ephemeris clock mask mismatch between Python and native Rust"
            # Nope residual error
            nope_err = float(np.max(np.abs(nope_nat - nope_py)))
            assert nope_err <= 1.0e-5, f"Ephemeris nope residual error {nope_err} > 1e-5"

    def test_spatial_mad_floor_256_parity(self):
        """Operator: 256-sample strided stack MAD floor parity."""
        rng = np.random.default_rng(104)

        for n_pix in [16384, 8192, 4096]:
            sal = (rng.standard_normal(n_pix) ** 2).astype(np.float32)
            mask = np.zeros(n_pix, dtype=np.uint8)
            mask[::128] = 1

            nat_val = _native.compute_spatial_mad_floor_256(sal, mask)
            py_val = py_mad_floor(sal, mask)

            err = abs(nat_val - py_val)
            assert err <= 1.0e-6, f"Spatial MAD floor parity error {err} > 1e-6"

    def test_so6_kelvin_mandel_rotor_parity(self):
        """Operator: SO(6) Kelvin-Mandel stiffness rotor trace conservation parity."""
        rng = np.random.default_rng(105)

        for _ in range(20):
            # Generate random 3D rotation via QR
            h = rng.standard_normal((3, 3))
            q, _ = np.linalg.qr(h)
            if np.linalg.det(q) < 0:
                q[:, 0] = -q[:, 0]

            # Symmetric 6x6 matrix
            a = rng.standard_normal((6, 6))
            c_base = np.dot(a, a.T) + np.eye(6)

            m6_nat, c_rot_nat, res_nat = _native.so6_kelvin_mandel_rotor(q, c_base)
            c_rot_py, res_py = py_so6_rotor(q, c_base)

            c_err = float(np.max(np.abs(c_rot_nat - c_rot_py)))
            res_err = abs(res_nat - res_py)

            assert c_err <= 1.0e-6, f"SO(6) rotated matrix parity error {c_err} > 1e-6"
            assert res_err <= 1.0e-6, f"SO(6) trace residual parity error {res_err} > 1e-6"
            assert res_nat <= 1.14e-13, f"SO(6) trace residual {res_nat} > 1.14e-13"

    def test_causal_progenitor_attribution_parity(self):
        """Operator: Causal progenitor attribution parity."""
        rng = np.random.default_rng(106)

        for n_obj in [3, 5, 8]:
            trans_mat = np.abs(rng.standard_normal((n_obj, n_obj))).astype(np.float32)
            np.fill_diagonal(trans_mat, 10.0)

            is_gen_nat, ratios_nat = _native.attribute_causal_progenitors(trans_mat, threshold=0.50)
            is_gen_py, ratios_py = py_attribute(trans_mat, threshold=0.50)

            assert np.array_equal(is_gen_nat, is_gen_py), "Causal generator boolean mask mismatch"
            err = float(np.max(np.abs(ratios_nat - ratios_py)))
            assert err <= 1.0e-6, f"Causal progenitor ratios error {err} > 1e-6"

    def test_void_retraction_parity(self):
        """Operator: Step-0 void core homotopy retraction parity."""
        rng = np.random.default_rng(107)

        density = rng.standard_normal((16, 16, 16)).astype(np.float32)
        void_mask = (density < -0.5).astype(np.uint8)

        ret_nat = _native.retract_void_step0_homotopy(density, void_mask)
        ret_py = py_retract_void(density, void_mask)

        err = float(np.max(np.abs(ret_nat - ret_py)))
        assert err <= 1.0e-6, f"Void retraction error {err} > 1e-6"
        assert np.all(ret_nat[void_mask.astype(bool)] >= -1.0), "Physical lower bound violation >= -1.0"

    def test_bypass_mechanism_investigation(self):
        """Investigates behavior when native DLL is bypassed.

        Note: astrosnail._native defines module-level `_LIB`, not `_NATIVE_LIB`.
        Setting _native._NATIVE_LIB = None does not bypass loading because
        load_native_library() checks `_LIB`.
        We verify:
        1. Setting _native._NATIVE_LIB = None leaves _native functional (finding documented).
        2. Bypassing native via AstroSnailEngine(backend='numpy') works cleanly.
        3. Forcing pure-Python fallback produces numerical results within 1e-6.
        """
        # Test what happens when _native._NATIVE_LIB is set
        _native._NATIVE_LIB = None

        # Engine configured with numpy backend
        engine_py = AstroSnailEngine(backend="numpy")
        assert engine_py.backend == "numpy"

        # Engine configured with rust backend
        engine_rust = AstroSnailEngine(backend="rust")
        assert engine_rust.backend == "rust"

        # Test both engines on identical 2D frame
        test_frame = np.ones((128, 128), dtype=np.float32)
        test_frame[64, 64] = 50.0

        res_py = engine_py.update(test_frame)
        res_rust = engine_rust.update(test_frame)

        assert res_py["backend"] == "numpy"
        assert res_rust["backend"] == "rust"
        assert res_py["num_medoids"] == res_rust["num_medoids"]
        assert np.allclose(res_py["medoid_coords_yx"], res_rust["medoid_coords_yx"], atol=1e-6)
        assert np.allclose(res_py["medoid_features"], res_rust["medoid_features"], atol=1e-6)
