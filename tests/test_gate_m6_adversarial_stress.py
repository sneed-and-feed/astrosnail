"""
tests/test_gate_m6_adversarial_stress.py
========================================
Empirical stress test suite for Milestone 6 Gate Challenge.
Thoroughly stress-tests:
1. Non-finite inputs (NaN, Inf, -Inf, subnormals) across all transient detectors and gates.
2. Out-of-bounds FITS streaming, header corruption, and strict 2880-byte block integrity.
3. Multi-cycle execution to verify memory stability, zero leaks, and absence of state leakage.
"""

import ctypes
import gc
import io
import os
from pathlib import Path
import tracemalloc
import numpy as np
import pytest

from astrosnail import (
    AstroSnailEngine,
    BeltramiLightCurveShell,
    compute_morse_impulse,
    MorseTransientGateConfig,
    TemporalClockContentSplitter,
    compute_spatial_mad_floor_256,
    RoPE2DMedoidPooler,
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
    _native,
)
from astrosnail.io import FITS_BLOCK_SIZE


# =============================================================================
# 1. Non-finite Inputs (NaN, Inf) across Transient Detectors & Gates
# =============================================================================


def test_stress_beltrami_shell_non_finite_inputs():
    """Stress BeltramiLightCurveShell with diverse non-finite signals."""
    shell = BeltramiLightCurveShell(k_tokens=16)

    # 1D: all NaNs
    nan_arr = np.full(50, np.nan, dtype=np.float32)
    c, f, s = shell.decompose(nan_arr)
    assert np.all(np.isfinite(c))
    assert np.all(np.isfinite(f))
    assert np.all(np.isfinite(s))
    assert np.all(s == 0.0)

    # 1D: all +Inf and -Inf
    inf_arr = np.array([np.inf if i % 2 == 0 else -np.inf for i in range(50)], dtype=np.float32)
    c_inf, f_inf, s_inf = shell.decompose(inf_arr)
    assert np.all(np.isfinite(c_inf))
    assert np.all(np.isfinite(f_inf))
    assert np.all(np.isfinite(s_inf))

    # 1D: subnormals and denormals
    subnormal_arr = np.full(50, 1.4012985e-45, dtype=np.float32)
    c_sub, f_sub, s_sub = shell.decompose(subnormal_arr)
    assert np.all(np.isfinite(c_sub))
    assert np.all(np.isfinite(s_sub))

    # Multi-band (50, 4) with random NaNs and Infs injected
    rng = np.random.default_rng(42)
    mb = rng.standard_normal((60, 4)).astype(np.float32)
    mb[10:15, 0] = np.nan
    mb[20:25, 1] = np.inf
    mb[30:35, 2] = -np.inf
    mb[40, :] = np.nan

    c_mb, f_mb, s_mb = shell.decompose(mb)
    assert c_mb.shape == (60, 4)
    assert f_mb.shape == (60, 4)
    assert s_mb.shape == (60,)
    assert np.all(np.isfinite(c_mb))
    assert np.all(np.isfinite(f_mb))
    assert np.all(np.isfinite(s_mb))
    assert np.all((s_mb >= 0.0) & (s_mb <= 1.0))


def test_stress_morse_impulse_non_finite_inputs():
    """Stress compute_morse_impulse with NaNs, Infs, and extreme salience values."""
    # 1D with mixed non-finites
    v = np.array([np.nan, 10.0, np.inf, 20.0, -np.inf, np.nan, 5.0, 5.0, 5.0], dtype=np.float32)
    imp = compute_morse_impulse(v)
    assert imp.shape == v.shape
    assert np.all(np.isfinite(imp))

    # With non-finite salience
    sal = np.array([np.nan, np.inf, -np.inf, np.nan, 0.5, 1.0, np.inf, 0.0, 0.0], dtype=np.float32)
    imp_sal = compute_morse_impulse(v, salience=sal)
    assert imp_sal.shape == v.shape
    assert np.all(np.isfinite(imp_sal))

    # Multi-band
    v_mb = np.full((30, 4), np.nan, dtype=np.float32)
    v_mb[15, :] = 100.0  # Spike amidst NaNs
    imp_mb = compute_morse_impulse(v_mb)
    assert imp_mb.shape == (30, 4)
    assert np.all(np.isfinite(imp_mb))


def test_stress_morse_gate_non_finite_hammer():
    """Hammer MorseTransientGateConfig with 2,000 non-finite and extreme value steps."""
    gate = MorseTransientGateConfig(sigma_thresh=4.0)

    # Sequence of adversarial values: NaNs, Infs, huge jumps, subnormals
    adversarial_inputs = [
        np.nan, np.inf, -np.inf, 1e38, -1e38, 1e-40, 0.0,
        100.0, np.nan, 0.0,  # Spike followed by NaN
        100.0, np.inf, 0.0,  # Spike followed by Inf
        np.nan, 100.0, 0.0,  # NaN followed by spike and return (glint pattern)
    ] * 200

    allowed_classes = {"nominal", "pending_impulse", "glint_rejected", "confirmed_transient"}

    for i, val in enumerate(adversarial_inputs):
        cls_name, update_applied = gate.update(val, timestamp=float(i))
        assert cls_name in allowed_classes
        assert isinstance(update_applied, bool)
        assert np.isfinite(gate.variance)
        assert np.isfinite(gate.noise_std)
        assert gate.variance > 0.0
        assert gate.noise_std > 0.0

    # Ensure gate recovers cleanly and functions normally after the assault
    cls_norm, _ = gate.update(0.0)
    assert cls_norm in allowed_classes


def test_stress_clock_splitter_non_finite_inputs():
    """Stress TemporalClockContentSplitter with non-finite arrays."""
    splitter = TemporalClockContentSplitter(periodicity_threshold=0.85)

    # All NaNs
    h_c, h_n, mask = splitter.split(np.full(100, np.nan, dtype=np.float32))
    assert np.all(np.isfinite(h_c))
    assert np.all(np.isfinite(h_n))
    assert np.all(mask == 0)

    # All Infs
    h_c, h_n, mask = splitter.split(np.full(100, np.inf, dtype=np.float32))
    assert np.all(np.isfinite(h_c))
    assert np.all(np.isfinite(h_n))
    assert np.all(mask == 0)

    # Periodic signal with periodic NaN drops
    t = np.arange(200)
    sig = np.sin(2 * np.pi * t / 10).astype(np.float32)
    sig[::5] = np.nan
    h_c, h_n, mask = splitter.split(sig)
    assert np.all(np.isfinite(h_c))
    assert np.all(np.isfinite(h_n))
    assert mask.shape == (200,)


def test_stress_spatial_mad_floor_non_finites():
    """Stress compute_spatial_mad_floor_256 with non-finite values and extreme masks."""
    # All NaNs and Infs
    bad_sal = np.array([np.nan, np.inf, -np.inf] * 100, dtype=np.float32)
    floor_bad = compute_spatial_mad_floor_256(bad_sal)
    assert np.isfinite(floor_bad)
    assert floor_bad >= 0.0

    # Mask masking everything
    sal = np.linspace(0.0, 10.0, 500, dtype=np.float32)
    mask = np.ones(500, dtype=np.uint8)
    floor_masked = compute_spatial_mad_floor_256(sal, mask=mask)
    assert np.isfinite(floor_masked)
    assert floor_masked == 0.0


def test_stress_engine_streaming_non_finite_frames():
    """Stress AstroSnailEngine with non-finite 1D, 2D, and 3D inputs."""
    eng = AstroSnailEngine(backend="numpy")

    # 1D non-finites
    out_1d_nan = eng.update(np.array([np.nan]))
    assert out_1d_nan["classification"] == "nominal"
    assert out_1d_nan["update_applied"] is False

    out_1d_inf = eng.update(np.array([np.inf]))
    assert out_1d_inf["classification"] == "nominal"

    # 1D series of NaNs
    out_1d_seq = eng.update(np.full(30, np.nan, dtype=np.float32))
    assert np.all(np.isfinite(out_1d_seq["curl_salience"]))

    # 2D frame with NaNs and Infs
    frame_bad = np.zeros((32, 32), dtype=np.float32)
    frame_bad[0:5, 0:5] = np.nan
    frame_bad[10:15, 10:15] = np.inf
    frame_bad[20:25, 20:25] = -np.inf
    out_2d = eng.update(frame_bad)
    assert "medoid_coords_yx" in out_2d
    assert np.all(np.isfinite(out_2d["medoid_coords_yx"]))

    # 3D cube with NaNs
    cube_bad = np.full((16, 16, 16), np.nan, dtype=np.float32)
    out_3d = eng.update(cube_bad)
    assert "void_cores" in out_3d
    assert out_3d["void_cores"].shape == (16, 16, 16)
    assert out_3d["void_voxel_count"] == 0
    assert out_3d["void_cores"].dtype == bool


def test_stress_native_rust_non_finite_inputs():
    """Verify native Rust C-ABI resilience against NaNs and Infs (zero crash/segfault)."""
    if not _native.is_available():
        pytest.skip("Native Rust library not available")

    lib = _native.load_native_library()
    h, w = 128, 128
    curr_buf = np.full((h, w), np.nan, dtype=np.float32)
    prev_buf = np.full((h, w), np.inf, dtype=np.float32)
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

    res = lib.beltrami_morse_transient_gate_f32(
        curr_ptr, prev_ptr, mask_ptr, h, w, sal_ptr, spikes_ptr, mad_ptr
    )
    # Zero crash / segfault, returns success status code
    assert res == 0
    assert np.isfinite(mad_floor.value)

    # Test with subnormals and extreme scales (1e-38, 1e38)
    curr_sub = np.full((h, w), 1.0e-38, dtype=np.float32)
    prev_sub = np.full((h, w), 2.0e-38, dtype=np.float32)
    curr_sub_ptr = curr_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    prev_sub_ptr = prev_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    res2 = lib.beltrami_morse_transient_gate_f32(
        curr_sub_ptr, prev_sub_ptr, mask_ptr, h, w, sal_ptr, spikes_ptr, mad_ptr
    )
    assert res2 == 0
    assert np.all(np.isfinite(sal_out))


# =============================================================================
# 2. Out-of-Bounds FITS Streaming & 2880-Byte Block Integrity
# =============================================================================


def test_fits_2880_block_integrity_arbitrary_geometries(tmp_path):
    """Verify strict 2880-byte alignment across non-standard shapes and header sizes."""
    shapes = [
        (1, 1),
        (3, 7),
        (17, 23),
        (1, 1, 1),
        (5, 7, 11),
        (33, 15, 9),
        (128, 128),
    ]
    dtypes = [np.float32, np.float64, np.int16, np.int32, np.int64, np.uint8, np.uint16]

    for i, (shape, dt) in enumerate(zip(shapes, dtypes)):
        data = np.ones(shape, dtype=dt)
        p = tmp_path / f"geom_{i}_{dt.__name__}.fits"

        # Generate variable header card count (e.g. 50 extra cards spanning multiple 2880-byte blocks)
        # Using compliant 8-character FITS keyword format EX_XXXX
        extra = {f"EX_{j:04d}": (f"VAL_{j}", f"Comment for card {j}") for j in range(50)}
        n_bytes = write_fits_cube(p, data, header_extra=extra)

        assert n_bytes % FITS_BLOCK_SIZE == 0
        assert p.stat().st_size == n_bytes
        assert p.stat().st_size % FITS_BLOCK_SIZE == 0

        # Read back header and verify metadata
        hdr = read_fits_header(p)
        assert hdr.shape == shape
        assert hdr.data_offset % FITS_BLOCK_SIZE == 0
        assert hdr["EX_0049"] == "VAL_49"


def test_fits_out_of_bounds_and_chunk_limits(tmp_path):
    """Test streaming with chunk_z exceeding cube depth and chunk_z = 1."""
    p = tmp_path / "shallow_cube.fits"
    write_fits_cube(p, np.ones((5, 16, 16), dtype=np.float32))

    # chunk_z = 20 > Z=5
    slabs = list(stream_fits_cube(p, chunk_z=20))
    assert len(slabs) == 1
    assert slabs[0].shape == (5, 16, 16)

    # chunk_z = 1 on 3D cube
    slabs_1 = list(stream_fits_cube(p, chunk_z=1, squeeze_2d=True))
    assert len(slabs_1) == 5
    assert slabs_1[0].shape == (16, 16)

    # chunk_z <= 0 must fail gracefully
    with pytest.raises(Exception):
        list(stream_fits_cube(p, chunk_z=0))


def test_fits_corrupted_and_truncated_files(tmp_path):
    """Ensure truncated or malformed FITS files raise clear exceptions without hanging."""
    # 1. Truncated header (ends before 2880 bytes)
    p_trunc_hdr = tmp_path / "trunc_header.fits"
    p_trunc_hdr.write_bytes(b"SIMPLE  =                    T / standard FITS"[:40])
    hdr_trunc = read_fits_header(p_trunc_hdr)
    assert hdr_trunc.shape == ()

    # 2. Header without END card (filled with blank spaces to 2880 bytes)
    p_no_end = tmp_path / "no_end.fits"
    p_no_end.write_bytes(b" " * 2880)
    hdr_no_end = read_fits_header(p_no_end)
    assert hdr_no_end.shape == ()

    # 3. Data truncated mid-cube
    p_trunc_data = tmp_path / "trunc_data.fits"
    write_fits_cube(p_trunc_data, np.ones((10, 16, 16), dtype=np.float32))
    # Truncate to just header + 100 bytes of data
    content = p_trunc_data.read_bytes()
    truncated_content = content[: 2880 + 100]
    p_trunc_data.write_bytes(truncated_content)

    bio = io.BytesIO(truncated_content)
    # Stream from truncated buffer: should raise or return partial without hanging
    try:
        slabs = list(stream_fits_cube(bio, chunk_z=2))
        assert len(slabs) < 5
    except (ValueError, EOFError, IndexError):
        pass

    # Ensure Windows file handle was freed and file can be unlinked immediately
    del content
    gc.collect()
    p_trunc_data.unlink()
    assert not p_trunc_data.exists()


# =============================================================================
# 3. Multi-Cycle Execution, Memory Stability & Zero State Leakage
# =============================================================================


def test_stress_multi_cycle_engine_memory_stability():
    """Verify memory stability across 1,000 continuous causal frames."""
    gc.collect()
    tracemalloc.start()

    eng = AstroSnailEngine(backend="numpy")
    rng = np.random.default_rng(1337)

    # Warmup 50 frames
    for i in range(50):
        frame = rng.standard_normal((32, 32)).astype(np.float32)
        eng.update(frame)

    snap_start = tracemalloc.take_snapshot()

    # Run 950 more frames (total 1000)
    for i in range(950):
        frame = rng.standard_normal((32, 32)).astype(np.float32)
        if i % 10 == 0:
            frame[10:15, 10:15] += 50.0  # Synthetic transient
        eng.update(frame)

    snap_end = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 1. Assert ring buffer strictly bounded to 8 frames
    assert len(eng.ring_buffer) == 8

    # 2. Assert peak memory under 2.0 MB working set
    peak_mb = peak / (1024 * 1024)
    assert peak_mb < 2.0, f"Peak memory {peak_mb:.2f} MB exceeded 2.0 MB limit"

    # 3. Assert zero unbounded memory leak between step 50 and step 1000
    top_diffs = snap_end.compare_to(snap_start, "lineno")
    total_net_growth = sum(d.size_diff for d in top_diffs)
    growth_kb = total_net_growth / 1024
    assert growth_kb < 150.0, f"Memory grew by {growth_kb:.2f} KB across 950 iterations"


def test_stress_zero_state_leakage_between_instances():
    """Verify that prior runs do not pollute state across independent engine instances."""
    rng = np.random.default_rng(2026)
    clean_signal = rng.standard_normal((100,)).astype(np.float32)
    clean_signal[40] = 80.0  # Transient spike

    # Run Instance 1 on golden signal
    eng1 = AstroSnailEngine(backend="numpy")
    out1 = eng1.update(clean_signal)

    # Run Instance 2 on dirty, hostile, non-finite signals for 200 cycles
    eng2 = AstroSnailEngine(backend="numpy")
    for _ in range(200):
        hostile = np.array([np.nan, np.inf, 1e20, -1e20, np.nan], dtype=np.float32)
        eng2.update(hostile)

    # Run Instance 3 (clean instance initialized after hostile runs)
    eng3 = AstroSnailEngine(backend="numpy")
    out3 = eng3.update(clean_signal)

    # Verify bit-identical results between eng1 and eng3
    for k in ["u_coarse", "u_fine", "curl_salience", "morse_impulse", "h_clock", "h_nope", "clock_mask", "mad_floor"]:
        if isinstance(out1[k], np.ndarray):
            np.testing.assert_array_equal(out1[k], out3[k], err_msg=f"State leakage detected in output '{k}'")
        else:
            assert out1[k] == out3[k], f"State leakage in scalar '{k}'"


def test_stress_multi_cycle_fits_streaming_leaks(tmp_path):
    """Stream a FITS cube 50 times in a loop, ensuring zero file lock or memory accumulation."""
    p_cube = tmp_path / "stream_stress.fits"
    write_fits_cube(p_cube, np.ones((16, 32, 32), dtype=np.float32))

    for cycle in range(50):
        slabs = list(stream_fits_cube(p_cube, chunk_z=4, reuse_buffer=True))
        assert len(slabs) == 4
        assert slabs[0].shape == (4, 32, 32)

    # Ensure file can be deleted immediately on Windows (proves all memmaps closed)
    gc.collect()
    p_cube.unlink()
    assert not p_cube.exists()
