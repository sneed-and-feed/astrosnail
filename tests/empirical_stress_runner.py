"""
tests/empirical_stress_runner.py
================================
Empirical benchmark and stress harness for Challenger M1-2.
Generates comprehensive quantitative receipts.
"""

import gc
import json
import math
import sys
import time
import tracemalloc
import numpy as np
from scipy import special

from astrosnail.cosmology import (
    reconstruct_zslab_tomography,
    detect_jordan_cavity_cores,
    so6_kelvin_mandel_rotor,
    _build_so6_kelvin_rotor,
    retract_void_step0_homotopy,
    blind_astro_let_decouple,
    solve_beltrami_jet_poincare,
)
from astrosnail.engine import AstroSnailEngine


def run_so6_rotor_stress(n_trials: int = 1000):
    rng = np.random.default_rng(20261028)
    A = rng.standard_normal((6, 6))
    C_orig = np.dot(A, A.T) + 2.0 * np.eye(6)

    trace_residuals = []
    ortho_errors = []
    det_errors = []
    energy_drifts = []

    for _ in range(n_trials):
        H = rng.standard_normal((3, 3))
        Q, _ = np.linalg.qr(H)
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]

        C_rot, res = so6_kelvin_mandel_rotor(Q, C_orig)
        trace_residuals.append(res)

        M = _build_so6_kelvin_rotor(Q)
        ortho_errors.append(float(np.max(np.abs(np.dot(M, M.T) - np.eye(6)))))
        det_errors.append(abs(float(np.linalg.det(M)) - 1.0))

        norm_orig = float(np.linalg.norm(C_orig, "fro"))
        norm_rot = float(np.linalg.norm(C_rot, "fro"))
        energy_drifts.append(abs(norm_rot - norm_orig))

    return {
        "trials": n_trials,
        "max_trace_residual": float(np.max(trace_residuals)),
        "mean_trace_residual": float(np.mean(trace_residuals)),
        "p99_trace_residual": float(np.percentile(trace_residuals, 99)),
        "max_ortho_error": float(np.max(ortho_errors)),
        "max_det_error": float(np.max(det_errors)),
        "max_energy_drift": float(np.max(energy_drifts)),
        "passed_bound_1_14e_13": bool(np.max(trace_residuals) <= 1.14e-13),
    }


def run_beltrami_jet_stress():
    mu_values = [0.25, 0.5, 1.0, 2.0, 3.5, 5.0]
    results = {}

    for mu in mu_values:
        B_field, radial_escape = solve_beltrami_jet_poincare(np.zeros((24, 24)), mu=mu, steps=16)
        z_dim, y_dim, x_dim = 32, 24, 24
        y0 = float(y_dim - 1) / 2.0
        x0 = float(x_dim - 1) / 2.0
        yy, xx = np.meshgrid(np.arange(y_dim), np.arange(x_dim), indexing="ij")
        r = np.sqrt((yy - y0) ** 2 + (xx - x0) ** 2).astype(np.float64)
        r_safe = np.maximum(1e-12, r)

        dpsi_dr = 1.0 * r * special.j0(mu * r)
        grad_psi_x = dpsi_dr * ((xx - x0) / r_safe)
        grad_psi_y = dpsi_dr * ((yy - y0) / r_safe)

        Bx = B_field[0, 16].astype(np.float64)
        By = B_field[1, 16].astype(np.float64)

        b_dot_grad_psi = Bx * grad_psi_x + By * grad_psi_y
        max_b_dot_grad = float(np.max(np.abs(b_dot_grad_psi)))

        results[f"mu_{mu}"] = {
            "radial_flux_escape": float(radial_escape),
            "max_abs_b_dot_grad_psi": max_b_dot_grad,
            "flux_escape_zero": bool(radial_escape == 0.0),
            "confinement_satisfied": bool(max_b_dot_grad < 1e-6),
        }

    return results


def run_zslab_chunk_stress():
    z_dim, y_dim, x_dim = 32, 24, 24
    rng = np.random.default_rng(999)
    skewers = [rng.normal(1.0, 0.5, size=z_dim).astype(np.float32) for _ in range(8)]
    coords = rng.uniform(2, 20, size=(8, 2)).astype(np.float32)

    ref_recon = reconstruct_zslab_tomography(skewers, coords, grid_shape=(z_dim, y_dim, x_dim), chunk_z=16)

    chunk_sizes = [1, 2, 4, 8, 16, 24, 32, 48]
    deviations = {}

    for cz in chunk_sizes:
        recon = reconstruct_zslab_tomography(skewers, coords, grid_shape=(z_dim, y_dim, x_dim), chunk_z=cz)
        max_dev = float(np.max(np.abs(recon["density"] - ref_recon["density"])))
        deviations[f"chunk_z_{cz}"] = {
            "max_abs_deviation": max_dev,
            "invariant": bool(max_dev < 1e-5),
        }

    return deviations


def run_step0_homotopy_stress():
    rng = np.random.default_rng(10101)
    density = rng.uniform(-10.0, 10.0, size=(32, 24, 24)).astype(np.float32)
    void_mask = density < -0.7

    retracted = retract_void_step0_homotopy(density, void_mask)

    unphysical_before = int(np.sum(density[void_mask] < -1.0))
    unphysical_after = int(np.sum(retracted[void_mask] < -1.0))
    safe_masked = void_mask & (density >= -1.0)
    drift_safe = float(np.max(np.abs(retracted[safe_masked] - density[safe_masked]))) if np.any(safe_masked) else 0.0
    drift_unmasked = float(np.max(np.abs(retracted[~void_mask] - density[~void_mask])))

    return {
        "unphysical_before": unphysical_before,
        "unphysical_after": unphysical_after,
        "drift_safe_masked": drift_safe,
        "drift_unmasked": drift_unmasked,
        "exact_zero_drift": bool(unphysical_after == 0 and drift_safe == 0.0 and drift_unmasked == 0.0),
    }


def run_throughput_and_memory_stress(n_frames: int = 500):
    engine = AstroSnailEngine(backend="numpy")
    frame = np.random.randn(128, 128).astype(np.float32)

    # 1. Clean throughput measurement (no tracemalloc instrumentation)
    for _ in range(20):
        engine.update(frame)

    frame_times = []
    for _ in range(n_frames):
        t0 = time.perf_counter()
        engine.update(frame)
        frame_times.append(time.perf_counter() - t0)

    total_time = sum(frame_times)
    fps = n_frames / total_time
    ms_per_frame = (total_time / n_frames) * 1000.0
    p50_ms = float(np.percentile(frame_times, 50)) * 1000.0
    p95_ms = float(np.percentile(frame_times, 95)) * 1000.0
    p99_ms = float(np.percentile(frame_times, 99)) * 1000.0

    # 2. Independent tracemalloc measurement
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()

    for _ in range(20):
        engine.update(frame)
    current_warmup, _ = tracemalloc.get_traced_memory()

    for _ in range(n_frames):
        engine.update(frame)

    current_after, peak_after = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_after / (1024.0 * 1024.0)
    leak_kb = (current_after - current_warmup) / 1024.0

    return {
        "frames_evaluated": n_frames,
        "fps": fps,
        "ms_per_frame": ms_per_frame,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "fps_floor_satisfied": bool(fps >= 165.0),
        "peak_heap_mb": peak_mb,
        "peak_heap_ceiling_satisfied": bool(peak_mb < 35.0),
        "leak_kb": leak_kb,
        "leak_bounded": bool(leak_kb < 100.0),
    }


if __name__ == "__main__":
    import os
    report = {}
    print("--- Running SO(6) Kelvin-Mandel Rotor Stress (1,000 trials) ---")
    report["so6_rotor"] = run_so6_rotor_stress(1000)
    print(f"Max trace residual: {report['so6_rotor']['max_trace_residual']:.3e} (threshold: 1.14e-13)")

    print("--- Running Beltrami Jet Stress ---")
    report["beltrami_jet"] = run_beltrami_jet_stress()
    print("Beltrami jet complete.")

    print("--- Running Z-Slab Chunk Invariance Stress ---")
    report["zslab_chunk"] = run_zslab_chunk_stress()
    print("Z-slab chunk invariance complete.")

    print("--- Running Step-0 Homotopy Retraction Stress ---")
    report["step0_homotopy"] = run_step0_homotopy_stress()
    print(f"Step-0 homotopy exact zero drift: {report['step0_homotopy']['exact_zero_drift']}")

    print("--- Running Sustained Throughput & Memory Stress (500 frames) ---")
    report["throughput_memory"] = run_throughput_and_memory_stress(500)
    print(f"FPS: {report['throughput_memory']['fps']:.1f} FPS (threshold: 165 FPS)")
    print(f"Peak Heap: {report['throughput_memory']['peak_heap_mb']:.2f} MB (ceiling: 35.0 MB)")
    print(f"Leak: {report['throughput_memory']['leak_kb']:.2f} KB over 500 frames")

    out_path = os.path.join(os.path.dirname(__file__), "empirical_stress_receipts.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSUCCESS: All stress testing runs complete. Receipts saved to {out_path}.")
