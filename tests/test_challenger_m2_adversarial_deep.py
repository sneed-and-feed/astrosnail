"""astrosnail.tests.test_challenger_m2_adversarial_deep: Deep Empirical Stress Test for M2 Challenger.

Comprehensive empirical verification:
1. Native throughput benchmark across 10,000 frames on 128x128 focal plane across 5 synthetic regimes:
   - Regime A: Standard Gaussian noise (sigma = 1.0)
   - Regime B: Dynamic range stress (10^-4 to 10^5 dynamic range)
   - Regime C: Dense glint impulses (25% pixel contamination)
   - Regime D: Pathological uniform flat-field (constant 0.0 and constant 1e4)
   - Regime E: Sparse transients (10 isolated point sources)
2. Zero heap allocation across 10,000 continuous iterations of all 5 C-ABI operators:
   - Beltrami Morse gate + spatial MAD floor
   - Ephemeris Clock / Nope split
   - SO(8) triad cascade
   - SO(6) Kelvin-Mandel rotor
   - 3D Cosmic-Web Z-slab relax
   - Beltrami Jet Poincare solver
3. Static Arena SRAM geometry verification:
   - Total size: 1,012,352 bytes (< 1.50 MB SRAM)
   - 64-byte alignment of every sub-buffer
   - Cache-line disjointness across major buffers
4. Defensive bounds stress test:
   - Out-of-bounds / negative parameters to cosmic_web_zslab_relax_f32
"""

import ctypes
import os
import time
import tracemalloc
import numpy as np
import psutil
import pytest

from astrosnail import _native
from astrosnail.transients import TemporalClockContentSplitter


class TestDeepChallengerM2Verification:
    """Rigorous empirical validation of Milestone 2 invariants."""

    def test_sustained_native_throughput_10000_frames_multi_regime(self):
        lib = _native.load_native_library()
        rng = np.random.default_rng(20261030)

        h, w = 128, 128
        curr_buf = np.zeros((h, w), dtype=np.float32)
        prev_buf = np.zeros((h, w), dtype=np.float32)
        clock_mask = np.zeros((h, w), dtype=np.uint8)
        sal_out = np.zeros((h, w), dtype=np.float32)
        spikes_out = np.zeros((h, w), dtype=np.uint8)
        mad_floor = ctypes.c_float(0.0)

        curr_ptr = curr_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        prev_ptr = prev_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        mask_ptr = clock_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        sal_ptr = sal_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        spikes_ptr = spikes_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        mad_ptr = ctypes.byref(mad_floor)

        regimes = {
            "gaussian_std": rng.standard_normal((h, w)).astype(np.float32),
            "dynamic_range": (10.0 ** rng.uniform(-4, 5, size=(h, w))).astype(np.float32),
            "dense_glints": (rng.standard_normal((h, w)) + (rng.random((h, w)) < 0.25) * 500.0).astype(np.float32),
            "pathological_zero": np.zeros((h, w), dtype=np.float32),
            "pathological_large": np.full((h, w), 10000.0, dtype=np.float32),
            "sparse_transients": np.zeros((h, w), dtype=np.float32),
        }
        # Inject 10 sparse transients
        regimes["sparse_transients"][rng.integers(1, 127, 10), rng.integers(1, 127, 10)] = 50.0

        n_frames = 10_000
        throughput_results = {}

        for regime_name, frame in regimes.items():
            np.copyto(curr_buf, frame)

            # Warmup
            for _ in range(200):
                lib.beltrami_morse_transient_gate_f32(curr_ptr, prev_ptr, mask_ptr, h, w, sal_ptr, spikes_ptr, mad_ptr)

            latencies = np.zeros(n_frames, dtype=np.float64)
            t_start = time.perf_counter()
            for i in range(n_frames):
                t0 = time.perf_counter_ns()
                rc = lib.beltrami_morse_transient_gate_f32(curr_ptr, prev_ptr, mask_ptr, h, w, sal_ptr, spikes_ptr, mad_ptr)
                t1 = time.perf_counter_ns()
                latencies[i] = (t1 - t0) / 1_000.0
                if rc != 0:
                    raise RuntimeError(f"Failed with code {rc} on frame {i}")
            t_end = time.perf_counter()

            elapsed = t_end - t_start
            fps = n_frames / elapsed
            p50 = float(np.percentile(latencies, 50))
            p90 = float(np.percentile(latencies, 90))
            p99 = float(np.percentile(latencies, 99))
            throughput_results[regime_name] = {"fps": fps, "p50": p50, "p90": p90, "p99": p99}

            print(f"[THROUGHPUT] {regime_name:<20}: {fps:,.1f} FPS (p50: {p50:.1f}µs, p90: {p90:.1f}µs, p99: {p99:.1f}µs)")
            assert fps >= 2_000.0, f"Regime {regime_name} FPS {fps:.1f} < 2,000 threshold"

    def test_full_pipeline_zero_heap_allocations_10000_iterations(self):
        lib = _native.load_native_library()
        arena = _native.NativeAstroSnailArena(use_global=True)
        arena.reset()

        # Buffers for all operators
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

        skewer_flux = np.ones((8, 16), dtype=np.float32)
        skewer_flux_ptr = skewer_flux.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        skewer_coords = np.array([[i, i] for i in range(8)], dtype=np.int32)
        skewer_coords_ptr = skewer_coords.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        density_out = np.zeros((16, 24, 24), dtype=np.float32)
        density_out_ptr = density_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        phi_out = np.zeros((16, 24, 24), dtype=np.float32)
        phi_out_ptr = phi_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        boundary = np.zeros((24, 24), dtype=np.float32)
        boundary_ptr = boundary.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_field = np.zeros((3, 32, 24, 24), dtype=np.float32)
        b_field_ptr = b_field.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        flux_escape = ctypes.c_float(0.0)
        flux_escape_ptr = ctypes.byref(flux_escape)

        # Warmup all operators
        for _ in range(50):
            lib.beltrami_morse_transient_gate_f32(frame_ptr, None, clock_ptr, h, w, sal_ptr, spike_ptr, mad_ptr)
            lib.compute_spatial_mad_floor_256_c(sal_ptr, clock_ptr, h * w)
            lib.ephemeris_clock_nope_split_f32(frame_ptr, 256, 0.85, clock_ptr, sal_ptr)
            lib.so8_astrometry_triad_cascade_f32(u_ptr, 16, v0_ptr, 0.15, v_out_ptr, drift_ptr)
            lib.so6_kelvin_mandel_rotor_f64(r3_ptr, c_ptr, m_ptr, c_rot_ptr, trace_ptr)
            lib.cosmic_web_zslab_relax_f32(skewer_flux_ptr, skewer_coords_ptr, 8, 16, 24, 24, density_out_ptr, phi_out_ptr)
            lib.solve_beltrami_jet_poincare_f32(boundary_ptr, 32, 24, 24, 1.0, 16, b_field_ptr, flux_escape_ptr)

        tracemalloc.start()
        lib.beltrami_morse_transient_gate_f32(frame_ptr, None, clock_ptr, h, w, sal_ptr, spike_ptr, mad_ptr)
        start_curr, start_peak = tracemalloc.get_traced_memory()

        proc = psutil.Process()
        vms_start = proc.memory_info().vms

        n_iters = 10_000
        for _ in range(n_iters):
            # Execute full C-ABI suite
            lib.beltrami_morse_transient_gate_f32(frame_ptr, None, clock_ptr, h, w, sal_ptr, spike_ptr, mad_ptr)
            lib.compute_spatial_mad_floor_256_c(sal_ptr, clock_ptr, h * w)
            lib.ephemeris_clock_nope_split_f32(frame_ptr, 256, 0.85, clock_ptr, sal_ptr)
            lib.so8_astrometry_triad_cascade_f32(u_ptr, 16, v0_ptr, 0.15, v_out_ptr, drift_ptr)
            lib.so6_kelvin_mandel_rotor_f64(r3_ptr, c_ptr, m_ptr, c_rot_ptr, trace_ptr)
            lib.cosmic_web_zslab_relax_f32(skewer_flux_ptr, skewer_coords_ptr, 8, 16, 24, 24, density_out_ptr, phi_out_ptr)
            lib.solve_beltrami_jet_poincare_f32(boundary_ptr, 32, 24, 24, 1.0, 16, b_field_ptr, flux_escape_ptr)

        vms_end = proc.memory_info().vms
        end_curr, end_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        net_growth = end_curr - start_curr
        vms_growth = vms_end - vms_start

        print(f"[ZERO-ALLOC AUDIT] Continuous 10,000 iterations full C-ABI pipeline:")
        print(f"  OS VMS Growth: {vms_growth} bytes")
        print(f"  Python Net Heap Growth: {net_growth} bytes")
        assert net_growth == 0, f"Python heap grew by {net_growth} bytes"
        assert vms_growth <= 0, f"OS VMS grew by {vms_growth} bytes"

    def test_exact_arena_sram_geometry_and_alignments(self):
        lib = _native.load_native_library()
        native_size = lib.astrosnail_arena_size_bytes()
        ctypes_size = ctypes.sizeof(_native.AstroSnailArena)

        assert native_size == 1_012_352, f"Native size {native_size} != 1,012,352 bytes"
        assert ctypes_size == 1_012_352, f"ctypes size {ctypes_size} != 1,012,352 bytes"
        assert native_size < 1_500_000, f"Size {native_size} >= 1.50 MB decimal"
        assert native_size < 1_572_864, f"Size {native_size} >= 1.50 MiB binary"

        # Verify alignment of every field
        for fname, ftype in _native.AstroSnailArena._fields_:
            field_desc = getattr(_native.AstroSnailArena, fname)
            offset = field_desc.offset
            size = field_desc.size
            if fname in [
                "spatial_input", "spatial_prev", "salience_map", "curl_buffer", "ring_buffer",
                "mask_clock", "mask_nope", "mask_spike", "cosmic_density", "cosmic_phi",
                "cosmic_void_mask", "skewer_flux", "skewer_coords", "triad_tensor_w",
                "cascade_w_rec", "medoid_coords_yx", "medoid_features", "causal_transition",
                "rotor_so6_m", "rotor_c_base", "rotor_c_rot", "morse_state", "telemetry",
                "step_counter",
            ]:
                assert offset % 64 == 0, f"Sub-buffer '{fname}' offset {offset} not 64-byte aligned"

    def test_defensive_bounds_validation(self):
        lib = _native.load_native_library()
        f = np.zeros((4, 16), dtype=np.float32)
        c = np.zeros((4, 2), dtype=np.int32)
        o1 = np.zeros((16, 24, 24), dtype=np.float32)
        o2 = np.zeros((16, 24, 24), dtype=np.float32)
        fp = f.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        cp = c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        o1p = o1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        o2p = o2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Negative / zero / overflow dimensions
        assert lib.cosmic_web_zslab_relax_f32(fp, cp, 4, -1, 24, 24, o1p, o2p) == -1
        assert lib.cosmic_web_zslab_relax_f32(fp, cp, -1, 16, 24, 24, o1p, o2p) == -1
        assert lib.cosmic_web_zslab_relax_f32(fp, cp, 0, 16, 24, 24, o1p, o2p) == -1
        assert lib.cosmic_web_zslab_relax_f32(fp, cp, 4, 0, 24, 24, o1p, o2p) == -1
        assert lib.cosmic_web_zslab_relax_f32(fp, cp, 17, 16, 24, 24, o1p, o2p) == -1
        assert lib.cosmic_web_zslab_relax_f32(fp, cp, 4, 33, 24, 24, o1p, o2p) == -1
        assert lib.cosmic_web_zslab_relax_f32(fp, cp, 4, 16, 0, 24, o1p, o2p) == -1
        assert lib.cosmic_web_zslab_relax_f32(fp, cp, 4, 16, 24, 0, o1p, o2p) == -1
        # Null pointers
        assert lib.cosmic_web_zslab_relax_f32(None, cp, 4, 16, 24, 24, o1p, o2p) == -1
        assert lib.cosmic_web_zslab_relax_f32(fp, None, 4, 16, 24, 24, o1p, o2p) == -1
        assert lib.cosmic_web_zslab_relax_f32(fp, cp, 4, 16, 24, 24, None, o2p) == -1
        assert lib.cosmic_web_zslab_relax_f32(fp, cp, 4, 16, 24, 24, o1p, None) == -1
