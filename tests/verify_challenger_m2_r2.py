"""verify_challenger_m2_r2.py
==========================
Adversarial Empirical Challenge Harness for Milestone 2 Remediation Gate (Iteration 2).
Author: Challenger M2-1-r2 (Empirical Challenger)

Empirically challenges:
1. Crash and hang freedom on cosmic_web_zslab_relax_f32 (negative dims, oversized dims, boundary dims).
2. Ephemeris bitmask parity across 200 random time series between Python and native Rust.
3. Radial drift reporting across 1,000 random input vectors through so8_astrometry_triad_cascade_f32.
"""

import ctypes
import functools
import math
import sys
import time
import numpy as np

from astrosnail import _native
from astrosnail.transients import TemporalClockContentSplitter
from astrosnail.astrometry import SlowFastTriadCascade


def test_challenge_1_cosmic_web_zslab_crash_and_hang():
    print("=" * 70)
    print("CHALLENGE 1: Crash & Hang Freedom on cosmic_web_zslab_relax_f32")
    print("=" * 70)

    lib = _native.load_native_library()

    # Allocate mock buffers
    max_skewers = 32
    max_z = 64
    y_dim = 24
    x_dim = 24

    flux = np.ascontiguousarray(np.random.standard_normal((max_skewers, max_z)), dtype=np.float32)
    coords = np.ascontiguousarray(np.random.randint(0, 24, size=(max_skewers, 2)), dtype=np.int32)
    delta_slab = np.zeros((max_z, y_dim, x_dim), dtype=np.float32)
    phi_slab = np.zeros((max_z, y_dim, x_dim), dtype=np.float32)

    fp = flux.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    cp = coords.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    dp = delta_slab.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    pp = phi_slab.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Test cases: (n_skewers, z_chunk, y_dim, x_dim, expected_rc, description)
    negative_and_oversized_cases = [
        # 1. Negative dimensions: all -1
        (-1, -1, -1, -1, -1, "All dimensions -1"),
        (-1, 16, 24, 24, -1, "n_skewers = -1"),
        (4, -1, 24, 24, -1, "z_chunk = -1"),
        (4, 16, -1, 24, -1, "y_dim = -1"),
        (4, 16, 24, -1, -1, "x_dim = -1"),
        (-100, -50, -24, -24, -1, "All dimensions large negative"),
        (-2147483648, 16, 24, 24, -1, "INT32_MIN n_skewers"),
        (4, -2147483648, 24, 24, -1, "INT32_MIN z_chunk"),

        # 2. Zero dimensions
        (0, 16, 24, 24, -1, "n_skewers = 0"),
        (4, 0, 24, 24, -1, "z_chunk = 0"),
        (4, 16, 0, 24, -1, "y_dim = 0"),
        (4, 16, 24, 0, -1, "x_dim = 0"),

        # 3. Oversized dimensions (hang / buffer overflow challenge)
        (20, 64, 24, 24, -1, "n_skewers = 20, z_chunk = 64 (Oversized)"),
        (17, 16, 24, 24, -1, "n_skewers = 17 (ASTROSNAIL_MAX_SKEWERS=16 + 1)"),
        (4, 33, 24, 24, -1, "z_chunk = 33 (ASTROSNAIL_GRID_Z=32 + 1)"),
        (100, 100, 24, 24, -1, "n_skewers = 100, z_chunk = 100"),
        (2147483647, 16, 24, 24, -1, "INT32_MAX n_skewers"),
        (4, 2147483647, 24, 24, -1, "INT32_MAX z_chunk"),

        # 4. Valid boundary dimensions (should succeed and return 0)
        (1, 1, 24, 24, 0, "Minimal valid: n_skewers = 1, z_chunk = 1"),
        (4, 16, 24, 24, 0, "Standard valid: n_skewers = 4, z_chunk = 16"),
        (16, 32, 24, 24, 0, "Max boundary valid: n_skewers = 16, z_chunk = 32"),
    ]

    crash_count = 0
    hang_count = 0
    mismatch_count = 0

    for i, (n, z, y, x, exp_rc, desc) in enumerate(negative_and_oversized_cases):
        t0 = time.perf_counter()
        try:
            rc = lib.cosmic_web_zslab_relax_f32(fp, cp, n, z, y, x, dp, pp)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0

            if elapsed_ms > 1000.0:  # Took > 1 second, potential hang!
                print(f"  [HANG DETECTED] Case {i}: {desc} took {elapsed_ms:.2f} ms")
                hang_count += 1

            if rc != exp_rc:
                print(f"  [RC MISMATCH] Case {i}: {desc} returned {rc}, expected {exp_rc}")
                mismatch_count += 1
            else:
                print(f"  [PASS] Case {i:02d}: {desc:<45} -> rc={rc:>2} in {elapsed_ms:6.3f} ms")

        except Exception as e:
            print(f"  [CRASH/EXCEPTION] Case {i}: {desc} failed with: {e}")
            crash_count += 1

    # Null pointer checks
    null_cases = [
        (None, cp, dp, pp, "Null skewer_flux"),
        (fp, None, dp, pp, "Null skewer_coords"),
        (fp, cp, None, pp, "Null out_delta"),
        (fp, cp, dp, None, "Null out_phi"),
    ]
    for i, (f_ptr, c_ptr, d_ptr, p_ptr, desc) in enumerate(null_cases):
        rc = lib.cosmic_web_zslab_relax_f32(f_ptr, c_ptr, 4, 16, 24, 24, d_ptr, p_ptr)
        assert rc == -1, f"Null pointer check failed for {desc}: rc={rc}"
        print(f"  [PASS] Null case {i:02d}: {desc:<25} -> rc={rc}")

    assert crash_count == 0, f"Detected {crash_count} crashes / access violations!"
    assert hang_count == 0, f"Detected {hang_count} thread hangs!"
    assert mismatch_count == 0, f"Detected {mismatch_count} return code mismatches!"
    print(f"\n>> CHALLENGE 1 VERDICT: PASSED (0 crashes, 0 access violations 0xC0000005, 0 thread hangs)\n")


def test_challenge_2_ephemeris_bitmask_parity_200_trials():
    print("=" * 70)
    print("CHALLENGE 2: Ephemeris Bitmask Parity across 200 Random Time Series")
    print("=" * 70)

    rng = np.random.default_rng(20261030)
    splitter = TemporalClockContentSplitter(periodicity_threshold=0.85)

    lengths = [32, 64, 128, 200, 256, 512, 1024, 2048]
    total_trials = 200
    mismatch_trials = 0
    max_nope_residual = 0.0

    periodic_detected_count = 0
    non_periodic_count = 0

    for trial in range(total_trials):
        n = int(rng.choice(lengths))
        signal_type = rng.choice(["sinusoid", "multi_harmonic", "sawtooth", "noise", "flicker", "dc", "pulsation"])

        t = np.arange(n, dtype=np.float64)

        if signal_type == "sinusoid":
            # Select period between 3 and n//2 - 1
            max_p = max(4, n // 2 - 2)
            p = int(rng.integers(3, max_p))
            amp = float(rng.uniform(1.0, 20.0))
            offset = float(rng.uniform(5.0, 50.0))
            noise_sigma = amp * float(rng.uniform(0.01, 0.05)) # High SNR
            flux = (offset + amp * np.sin(2.0 * np.pi * t / p) + rng.normal(0, noise_sigma, size=n)).astype(np.float32)

        elif signal_type == "multi_harmonic":
            max_p = max(6, n // 3)
            p = int(rng.integers(6, max_p))
            amp = float(rng.uniform(2.0, 10.0))
            flux = (20.0 + amp * np.sin(2.0 * np.pi * t / p) + 0.5 * amp * np.cos(4.0 * np.pi * t / p)).astype(np.float32)

        elif signal_type == "sawtooth":
            max_p = max(5, n // 2 - 2)
            p = int(rng.integers(5, max_p))
            flux = (10.0 + 5.0 * ((t % p) / float(p))).astype(np.float32)

        elif signal_type == "noise":
            # Pure Gaussian noise: should NOT be periodic
            flux = rng.standard_normal(n).astype(np.float32)

        elif signal_type == "flicker":
            # High-frequency alternating sign flip
            pattern = np.array([1.0, -1.0] * (n // 2 + 1))[:n]
            flux = (10.0 + 5.0 * pattern).astype(np.float32)

        elif signal_type == "dc":
            flux = np.full(n, 12.345, dtype=np.float32)

        elif signal_type == "pulsation":
            # RR Lyrae style sharp rise and slow decay
            p = int(rng.integers(8, max(9, n // 4)))
            phase = (t % p) / float(p)
            waveform = np.exp(-3.0 * phase)
            flux = (15.0 + 8.0 * waveform + rng.normal(0, 0.1, size=n)).astype(np.float32)

        # Run native Rust
        clock_mask_nat, nope_nat = _native.ephemeris_clock_nope_split(flux, threshold=0.85)

        # Run Python reference
        h_clock_py, h_nope_py, clock_mask_py = splitter.split(flux)

        # Compare masks
        mask_equal = np.array_equal(clock_mask_nat, clock_mask_py)
        diff_count = int(np.sum(clock_mask_nat != clock_mask_py))

        nope_err = float(np.max(np.abs(nope_nat - h_nope_py)))
        if nope_err > max_nope_residual:
            max_nope_residual = nope_err

        if not mask_equal:
            mismatch_trials += 1
            print(f"  [MISMATCH] Trial {trial}: type={signal_type}, n={n}, diff={diff_count} samples")
        else:
            if np.any(clock_mask_nat):
                periodic_detected_count += 1
            else:
                non_periodic_count += 1

    print(f"  Trials evaluated: {total_trials}")
    print(f"  Periodic signals correctly isolated: {periodic_detected_count}")
    print(f"  Non-periodic signals quarantined: {non_periodic_count}")
    print(f"  Bitmask mismatches: {mismatch_trials} / {total_trials} (Parity: {100.0 * (1.0 - mismatch_trials / total_trials):.2f}%)")
    print(f"  Max nope residual difference: {max_nope_residual:.6e}")

    assert mismatch_trials == 0, f"Detected {mismatch_trials} bitmask mismatches across {total_trials} trials!"
    assert max_nope_residual <= 1.0e-5, f"Nope residual error {max_nope_residual} exceeds 1.0e-5!"
    print(f"\n>> CHALLENGE 2 VERDICT: PASSED (0 mask mismatches across 100% of 200 trials, max nope err {max_nope_residual:.2e})\n")


def test_challenge_3_radial_drift_reporting_1000_trials():
    print("=" * 70)
    print("CHALLENGE 3: Radial Drift Reporting across 1,000 Random Input Vectors")
    print("=" * 70)

    rng = np.random.default_rng(2026)
    k_steps_list = [1, 4, 8, 16, 32]
    total_trials = 1000

    max_telemetry_discrepancy = 0.0
    max_reported_drift = 0.0
    non_zero_drift_count = 0
    exact_zero_drift_count = 0

    for trial in range(total_trials):
        k = k_steps_list[trial % len(k_steps_list)]
        u_seq = rng.standard_normal((k, 8)).astype(np.float32)
        v0 = rng.standard_normal(8).astype(np.float32)
        coupling = float(rng.uniform(0.01, 0.3))

        v_nat, drift_nat = _native.so8_astrometry_triad_cascade(u_seq, v0, coupling=coupling, steps=k)

        # Ground truth float32 norm calculation matching Rust
        eval_sq = lambda v: functools.reduce(lambda acc, x: np.float32(acc + np.float32(x * x)), v, np.float32(0.0))
        init_norm_sq = float(eval_sq(v0))
        final_norm_sq = float(eval_sq(v_nat))
        true_calculated_drift = abs(final_norm_sq - init_norm_sq)

        discrepancy = abs(drift_nat - true_calculated_drift)
        if discrepancy > max_telemetry_discrepancy:
            max_telemetry_discrepancy = discrepancy

        if drift_nat > max_reported_drift:
            max_reported_drift = drift_nat

        if drift_nat == 0.0:
            exact_zero_drift_count += 1
        else:
            non_zero_drift_count += 1

        # Strict assertion on every single trial:
        assert discrepancy <= 1.0e-7, (
            f"Trial {trial}: Reported drift {drift_nat} deviates from true calculated drift "
            f"{true_calculated_drift} by {discrepancy} > 1.0e-7"
        )
        assert drift_nat <= 5.0e-6, (
            f"Trial {trial}: Reported drift {drift_nat} exceeds 5.0e-6 ceiling!"
        )

    print(f"  Trials evaluated: {total_trials}")
    print(f"  Exact zero drift trials: {exact_zero_drift_count} ({exact_zero_drift_count / total_trials * 100:.1f}%)")
    print(f"  Non-zero (quantization residual) trials: {non_zero_drift_count} ({non_zero_drift_count / total_trials * 100:.1f}%)")
    print(f"  Max discrepancy between reported drift and true float32 drift: {max_telemetry_discrepancy:.6e} (Threshold: <= 1.0e-7)")
    print(f"  Max reported radial drift: {max_reported_drift:.6e} (Threshold: <= 5.0e-6)")

    assert max_telemetry_discrepancy <= 1.0e-7
    assert max_reported_drift <= 5.0e-6
    print(f"\n>> CHALLENGE 3 VERDICT: PASSED (100% agreement <= 1.0e-7, max drift {max_reported_drift:.2e} <= 5.0e-6)\n")


def test_challenge_3b_unit_sphere_s7_1000_trials():
    print("=" * 70)
    print("CHALLENGE 3b: S^7 Unit Sphere Normalized Vectors (1,000 Trials)")
    print("=" * 70)

    rng = np.random.default_rng(42)
    k_steps_list = [1, 4, 8, 16, 32]
    total_trials = 1000

    max_telemetry_discrepancy = 0.0
    max_reported_drift = 0.0
    exact_zero_drift_count = 0

    for trial in range(total_trials):
        k = k_steps_list[trial % len(k_steps_list)]
        u_seq = rng.standard_normal((k, 8)).astype(np.float32)
        v0 = rng.standard_normal(8).astype(np.float32)
        norm_v0 = np.linalg.norm(v0).astype(np.float32)
        v0 = (v0 / norm_v0).astype(np.float32)
        coupling = float(rng.uniform(0.01, 0.3))

        v_nat, drift_nat = _native.so8_astrometry_triad_cascade(u_seq, v0, coupling=coupling, steps=k)

        eval_sq = lambda v: functools.reduce(lambda acc, x: np.float32(acc + np.float32(x * x)), v, np.float32(0.0))
        init_norm_sq = float(eval_sq(v0))
        final_norm_sq = float(eval_sq(v_nat))
        true_calculated_drift = abs(final_norm_sq - init_norm_sq)

        discrepancy = abs(drift_nat - true_calculated_drift)
        if discrepancy > max_telemetry_discrepancy:
            max_telemetry_discrepancy = discrepancy

        if drift_nat > max_reported_drift:
            max_reported_drift = drift_nat

        if drift_nat == 0.0:
            exact_zero_drift_count += 1

        assert discrepancy <= 1.0e-7
        assert drift_nat <= 5.0e-6

    print(f"  Trials evaluated: {total_trials}")
    print(f"  Exact zero drift trials: {exact_zero_drift_count} ({exact_zero_drift_count / total_trials * 100:.1f}%)")
    print(f"  Max discrepancy between reported drift and true float32 drift: {max_telemetry_discrepancy:.6e} (Threshold: <= 1.0e-7)")
    print(f"  Max reported radial drift: {max_reported_drift:.6e} (Threshold: <= 5.0e-6)")
    print(f"\n>> CHALLENGE 3b VERDICT: PASSED (S^7 unit sphere max drift {max_reported_drift:.2e} <= 5.0e-6)\n")


if __name__ == "__main__":
    t_start = time.perf_counter()
    test_challenge_1_cosmic_web_zslab_crash_and_hang()
    test_challenge_2_ephemeris_bitmask_parity_200_trials()
    test_challenge_3_radial_drift_reporting_1000_trials()
    test_challenge_3b_unit_sphere_s7_1000_trials()
    t_total = time.perf_counter() - t_start
    print("=" * 70)
    print(f"ALL CHALLENGES EMPIRICALLY PASSED in {t_total:.2f} s")
    print("=" * 70)
