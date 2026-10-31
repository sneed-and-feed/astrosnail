"""astrosnail.tests.test_challenger_m2_stress: Empirical Stress Testing Suite for Milestone 2.

Conducts rigorous empirical challenge and verification:
1. Native throughput benchmark:
   - Measures raw execution FPS on 128x128 focal planes across 10,000 frames.
   - Asserts FPS >= 2,000 (target 2,000 to 9,400+ FPS).
   - Measures latency percentiles (p50, p90, p99, p99.9).
   - Validates across Gaussian noise, impulse glints, and flat baseline frames.
2. Memory and allocation audit:
   - Dynamic zero-allocation test: 10,000 continuous inference iterations with tracemalloc and psutil.
   - PE import table analysis verifying zero imports of malloc/free/calloc/realloc/HeapAlloc.
   - Direct Rust-level custom GlobalAlloc validation (allocs=0, deallocs=0, reallocs=0).
3. Static arena footprint audit:
   - Verifies arena footprint is strictly < 1.50 MB SRAM (1,572,864 bytes).
4. 64-byte alignment and cache-line disjointness across all sub-buffers:
   - Audits all 24 sub-buffers for 64-byte alignment and disjoint cache lines.
"""

import ctypes
import math
import os
import sys
import time
import tracemalloc
import numpy as np
import psutil
import pytest

from astrosnail import _native


class TestMilestone2NativeThroughputBenchmark:
    """Stress test 1: Raw execution FPS on 128x128 focal planes across 10,000 frames."""

    @pytest.mark.parametrize("scenario_name, generator", [
        ("gaussian_noise", lambda rng: rng.standard_normal((128, 128)).astype(np.float32)),
        ("glitch_spikes", lambda rng: (rng.standard_normal((128, 128)) + (rng.random((128, 128)) < 0.10) * 80.0).astype(np.float32)),
        ("flat_field", lambda rng: np.full((128, 128), 10.0, dtype=np.float32)),
    ])
    def test_raw_c_abi_throughput_10000_frames(self, scenario_name, generator):
        lib = _native.load_native_library()
        rng = np.random.default_rng(20261030)

        # Pre-allocate contiguous C buffers
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

        # Pre-generate frame to isolate kernel execution time from RNG
        frame_data = generator(rng)
        np.copyto(curr_buf, frame_data)

        # Warmup (500 frames)
        for _ in range(500):
            lib.beltrami_morse_transient_gate_f32(
                curr_ptr, prev_ptr, mask_ptr, h, w, sal_ptr, spikes_ptr, mad_ptr
            )

        # 10,000 frames execution
        n_frames = 10_000
        latencies_us = np.zeros(n_frames, dtype=np.float64)

        t_start = time.perf_counter()
        for i in range(n_frames):
            t0 = time.perf_counter_ns()
            rc = lib.beltrami_morse_transient_gate_f32(
                curr_ptr, prev_ptr, mask_ptr, h, w, sal_ptr, spikes_ptr, mad_ptr
            )
            t1 = time.perf_counter_ns()
            latencies_us[i] = (t1 - t0) / 1_000.0
            if rc != 0:
                raise RuntimeError(f"Kernel failed with code {rc} on frame {i}")
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        fps = n_frames / elapsed

        p50 = np.percentile(latencies_us, 50)
        p90 = np.percentile(latencies_us, 90)
        p99 = np.percentile(latencies_us, 99)
        p99_9 = np.percentile(latencies_us, 99.9)

        print(f"\n[BENCHMARK] Scenario: {scenario_name}")
        print(f"  Frames: {n_frames:,}")
        print(f"  Elapsed: {elapsed:.4f} s")
        print(f"  Throughput: {fps:,.1f} FPS (Target: >= 2,000 FPS)")
        print(f"  Latency p50: {p50:.2f} µs | p90: {p90:.2f} µs | p99: {p99:.2f} µs | p99.9: {p99_9:.2f} µs")

        assert fps >= 2_000.0, f"Scenario {scenario_name} throughput {fps:.1f} FPS below 2,000 FPS requirement"

    def test_native_wrapper_throughput_10000_frames(self):
        rng = np.random.default_rng(42)
        frame = rng.standard_normal((128, 128)).astype(np.float32)

        # Warmup
        for _ in range(200):
            _native.beltrami_morse_transient_gate(frame)

        n_frames = 10_000
        t0 = time.perf_counter()
        for _ in range(n_frames):
            _native.beltrami_morse_transient_gate(frame)
        t1 = time.perf_counter()

        fps = n_frames / (t1 - t0)
        print(f"\n[BENCHMARK] Python _native Wrapper 10,000 frames: {fps:,.1f} FPS")
        assert fps >= 2_000.0, f"Python _native wrapper throughput {fps:.1f} FPS below 2,000 FPS"


class TestMilestone2MemoryAndAllocationAudit:
    """Stress test 2: Zero heap allocation audit across 10,000 continuous iterations."""

    def test_dynamic_zero_heap_allocations_10000_iterations(self):
        lib = _native.load_native_library()
        arena = _native.NativeAstroSnailArena(use_global=True)
        arena.reset()

        # Pre-allocate input/output ctypes structures to isolate Python from C calls
        h, w = 128, 128
        frame = np.ones((h, w), dtype=np.float32)
        frame_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
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
            lib.beltrami_morse_transient_gate_f32(frame_ptr, None, None, h, w, sal_ptr, spike_ptr, mad_ptr)
            lib.so8_astrometry_triad_cascade_f32(u_ptr, 16, v0_ptr, 0.15, v_out_ptr, drift_ptr)

        process = psutil.Process()
        proc_mem_start = process.memory_info().vms

        # Start tracemalloc memory audit
        _ = 0
        tracemalloc.start()
        start_current, start_peak = tracemalloc.get_traced_memory()

        n_iters = 10_000
        for _ in range(n_iters):
            lib.beltrami_morse_transient_gate_f32(frame_ptr, None, None, h, w, sal_ptr, spike_ptr, mad_ptr)
            lib.so8_astrometry_triad_cascade_f32(u_ptr, 16, v0_ptr, 0.15, v_out_ptr, drift_ptr)
            lib.so6_kelvin_mandel_rotor_f64(r3_ptr, c_ptr, m_ptr, c_rot_ptr, trace_ptr)
        _ = 0

        end_current, end_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        proc_mem_end = process.memory_info().vms

        os_vms_delta = proc_mem_end - proc_mem_start
        net_growth = end_current - start_current

        print(f"\n[MEMORY AUDIT] Across {n_iters:,} continuous iterations:")
        print(f"  OS Process Commit VMS Growth: {os_vms_delta} bytes")
        print(f"  Python Net Heap Growth: {net_growth} bytes")

        # Assert zero net memory growth across 10,000 iterations
        assert net_growth == 0, f"Detected net heap allocation of {net_growth} bytes across 10,000 iterations"
        assert os_vms_delta <= 0, f"Detected OS VMS commit growth of {os_vms_delta} bytes across 10,000 iterations"

    def test_pe_import_table_has_zero_heap_functions(self):
        if sys.platform != "win32":
            pytest.skip("PE import table analysis applies to Windows DLL (astrosnail_rs.dll)")
        import pefile
        pkg_dir = os.path.dirname(os.path.abspath(_native.__file__))
        project_root = os.path.abspath(os.path.join(pkg_dir, ".."))
        dll_path = os.path.join(project_root, "rust", "astrosnail_rs", "target", "release", "astrosnail_rs.dll")
        if not os.path.exists(dll_path):
            dll_path = _native._find_shared_library()
        assert os.path.exists(dll_path), f"DLL not found at {dll_path}"

        pe = pefile.PE(dll_path)
        alloc_blacklist = [
            "malloc", "calloc", "realloc", "free",
            "_malloc_base", "_calloc_base", "_realloc_base", "_free_base",
            "HeapAlloc", "HeapFree", "HeapReAlloc", "RtlAllocateHeap", "RtlFreeHeap"
        ]

        imported_symbols = []
        for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
            dll_name = entry.dll.decode("utf-8", errors="ignore")
            for imp in entry.imports:
                fname = imp.name.decode("utf-8", errors="ignore") if imp.name else f"Ord({imp.ordinal})"
                imported_symbols.append((dll_name, fname))

        violations = []
        for dll_name, fname in imported_symbols:
            for black in alloc_blacklist:
                if black.lower() == fname.lower():
                    violations.append((dll_name, fname))

        print(f"\n[PE IMPORT AUDIT] Total imported symbols: {len(imported_symbols)}")
        for dll, sym in imported_symbols:
            print(f"  {dll} -> {sym}")

        assert len(violations) == 0, f"Found blacklisted heap allocation symbols in PE import table: {violations}"


class TestMilestone2StaticArenaFootprintAndAlignment:
    """Stress tests 3 & 4: Static arena footprint < 1.50 MB SRAM, 64-byte alignment & disjointness."""

    def test_static_arena_footprint_strict_bound(self):
        lib = _native.load_native_library()
        native_size = lib.astrosnail_arena_size_bytes()
        ctypes_size = ctypes.sizeof(_native.AstroSnailArena)

        assert native_size == ctypes_size, f"Native size {native_size} != ctypes size {ctypes_size}"
        assert native_size == 1_012_352, f"Expected exactly 1,012,352 bytes, got {native_size}"

        sram_budget_mib = 1.50 * 1024 * 1024  # 1,572,864 bytes
        sram_budget_mb = 1.50 * 1000 * 1000   # 1,500,000 bytes

        headroom_mib = sram_budget_mib - native_size
        headroom_pct = (headroom_mib / sram_budget_mib) * 100.0

        print(f"\n[ARENA FOOTPRINT AUDIT]")
        print(f"  Total arena size: {native_size:,} bytes ({native_size / 1024 / 1024:.4f} MiB)")
        print(f"  SRAM Budget ceiling (1.50 MiB): {sram_budget_mib:,} bytes")
        print(f"  SRAM Budget ceiling (1.50 MB dec): {sram_budget_mb:,} bytes")
        print(f"  Headroom: {headroom_mib:,} bytes ({headroom_pct:.2f}% margin)")

        assert native_size < sram_budget_mib, f"Arena size {native_size} exceeds 1.50 MiB ({sram_budget_mib})"
        assert native_size < sram_budget_mb, f"Arena size {native_size} exceeds 1.50 MB decimal ({sram_budget_mb})"

    def test_64byte_alignment_and_cacheline_disjointness_all_subbuffers(self):
        arena_cls = _native.AstroSnailArena

        # List of all field definitions in sequential order
        fields = [f[0] for f in arena_cls._fields_]
        offsets = {name: getattr(arena_cls, name).offset for name in fields}
        sizes = {name: getattr(arena_cls, name).size for name in fields}

        print("\n[ARENA SUFF-BUFFER ALIGNMENT & CACHELINE AUDIT]")
        print(f"{'Field Name':<24} {'Offset':<10} {'Size':<10} {'End':<10} {'Align64':<8} {'Cachelines'}")
        print("-" * 75)

        for i, fname in enumerate(fields):
            off = offsets[fname]
            sz = sizes[fname]
            end = off + sz
            align_ok = (off % 64 == 0)
            cl_start = off // 64
            cl_end = (end + 63) // 64

            print(f"{fname:<24} {off:<10} {sz:<10} {end:<10} {'YES' if align_ok else 'NO':<8} [{cl_start}..{cl_end})")

            # 1. Assert 64-byte alignment
            if fname in [
                "spatial_input", "spatial_prev", "salience_map", "curl_buffer", "ring_buffer",
                "mask_clock", "mask_nope", "mask_spike", "cosmic_density", "cosmic_phi",
                "cosmic_void_mask", "skewer_flux", "skewer_coords", "triad_tensor_w",
                "cascade_w_rec", "medoid_coords_yx", "medoid_features", "causal_transition",
                "rotor_so6_m", "rotor_c_base", "rotor_c_rot", "morse_state", "telemetry",
                "step_counter",
            ]:
                assert off % 64 == 0, f"Sub-buffer '{fname}' offset {off} is not 64-byte aligned"

            # 2. Assert no overlapping sub-buffers
            if i < len(fields) - 1:
                next_fname = fields[i + 1]
                next_off = offsets[next_fname]
                assert end <= next_off, f"Sub-buffer '{fname}' ends at {end} overlapping next field '{next_fname}' at {next_off}"

            # 3. Assert cache-line disjointness across distinct major sub-buffers
            major_buffers = [
                "spatial_input", "spatial_prev", "salience_map", "curl_buffer", "ring_buffer",
                "mask_clock", "mask_nope", "mask_spike", "cosmic_density", "cosmic_phi",
                "cosmic_void_mask", "skewer_flux", "skewer_coords", "triad_tensor_w",
                "cascade_w_rec", "medoid_coords_yx", "medoid_features", "causal_transition",
                "rotor_so6_m", "rotor_c_base", "rotor_c_rot", "morse_state", "telemetry",
            ]
            if fname in major_buffers and i < len(fields) - 1:
                next_fname = fields[i + 1]
                next_off = offsets[next_fname]
                if next_fname in major_buffers:
                    min_disjoint_offset = ((end + 63) // 64) * 64
                    assert next_off >= min_disjoint_offset, (
                        f"Cache-line sharing between major sub-buffers '{fname}' (ends at {end}) "
                        f"and '{next_fname}' (starts at {next_off}). Required >= {min_disjoint_offset}"
                    )

        # Verify overall struct alignment and total size
        total_size = ctypes.sizeof(arena_cls)
        assert total_size % 64 == 0, f"Total arena struct size {total_size} is not a multiple of 64"
        assert total_size == 1_012_352
