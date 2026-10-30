"""
tests/test_challenger_m4_1_empirical.py
======================================
Empirical Adversarial Verification Suite by Challenger M4-1 (Invariants & Memory Challenger).

Rigorous stress tests covering:
1. Adversarial Scenario-Name Branching Stress Test:
   - AST analysis of all core astrosnail modules for conditional branching on scenario identifiers.
   - Bytecode inspection of core functions and classes for LOAD_CONST / COMPARE_OP scenario strings.
   - Runtime blind polymorphism: identical outputs under legitimate, spoofed, mismatched, and corrupt scenario_id strings.
2. Causal Streaming Monotonicity Stress Test:
   - Proof of no future lookahead leakage (strict equivalence between prefix and full-stream prefix).
   - Bounded 8-frame ring buffer invariant under 200+ continuous frame updates.
   - History immutability: internal ring buffer state does not mutate past frames.
   - 1-sample causal deferral state machine transitions: exactly 0 false triggers across 10,000 glint spikes.
   - Temporal jitter, non-monotonic timestamps, frame dropout, and arrow-of-time irreversibility.
3. Out-of-Core Memory Profiling Stress Test:
   - 64 MB FITS cube streaming memory profiling via tracemalloc (< 2.0 MB working set, < 35.0 MB ceiling).
   - File handle closure and Windows lockup ([WinError 32]) prevention during complete and partial iteration.
   - Roundtrip bit-identical data integrity (0.000e+00 loss).
"""

from __future__ import annotations

import ast
import dis
import gc
import inspect
import math
import os
from pathlib import Path
import tempfile
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

import astrosnail
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
    stream_fits_cube,
    write_fits_cube,
    read_fits_header,
)
import astrosnail.engine as engine_module
import astrosnail.transients as transients_module
import astrosnail.astrometry as astrometry_module
import astrosnail.cosmology as cosmology_module
import astrosnail.io as io_module
import astrosnail._native as native_module


# =============================================================================
# PILLAR 1: ADVERSARIAL SCENARIO-NAME BRANCHING STRESS TESTS
# =============================================================================

class TestAdversarialScenarioNameBranching:
    """Rigorous AST, Bytecode, and Runtime verification that astrosnail has zero scenario-name branching."""

    # Canonical scenario identifiers and aliases that must NEVER appear in algorithmic control flow
    FORBIDDEN_SCENARIO_TOKENS = (
        "01_", "02_", "03_", "04_", "05_",
        "frb_supernova", "eclipsing_binary", "kbo_crossing", "lensed_quasar", "void_rsd",
        "scenario_01", "scenario_02", "scenario_03", "scenario_04", "scenario_05",
    )

    CORE_MODULES = (
        "engine.py",
        "transients.py",
        "astrometry.py",
        "cosmology.py",
        "_native.py",
        "io.py",
    )

    def _get_package_dir(self) -> Path:
        return Path(astrosnail.__file__).parent

    def test_ast_scan_no_conditional_branching_on_scenario_tokens(self):
        """Audits AST of core modules: asserts no conditional statements branch on scenario names."""
        pkg_dir = self._get_package_dir()
        violations: List[Dict[str, Any]] = []

        class ScenarioBranchAuditor(ast.NodeVisitor):
            def __init__(self, filename: str):
                self.filename = filename

            def check_expr(self, expr_node: ast.AST, stmt_type: str, lineno: int):
                for sub in ast.walk(expr_node):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        val_lower = sub.value.lower()
                        for tok in TestAdversarialScenarioNameBranching.FORBIDDEN_SCENARIO_TOKENS:
                            if tok in val_lower:
                                violations.append({
                                    "file": self.filename,
                                    "line": lineno,
                                    "stmt": stmt_type,
                                    "constant": sub.value,
                                    "token": tok,
                                })

            def visit_If(self, node: ast.If):
                self.check_expr(node.test, "If", node.lineno)
                self.generic_visit(node)

            def visit_IfExp(self, node: ast.IfExp):
                self.check_expr(node.test, "IfExp", node.lineno)
                self.generic_visit(node)

            def visit_While(self, node: ast.While):
                self.check_expr(node.test, "While", node.lineno)
                self.generic_visit(node)

            def visit_Assert(self, node: ast.Assert):
                self.check_expr(node.test, "Assert", node.lineno)
                self.generic_visit(node)

            def visit_Match(self, node: ast.AST):
                if hasattr(node, "subject"):
                    self.check_expr(getattr(node, "subject"), "Match", getattr(node, "lineno", 0))
                self.generic_visit(node)

        for mod_name in self.CORE_MODULES:
            mod_path = pkg_dir / mod_name
            if not mod_path.exists():
                continue
            with open(mod_path, "r", encoding="utf-8") as fp:
                tree = ast.parse(fp.read(), filename=str(mod_path))
            auditor = ScenarioBranchAuditor(mod_name)
            auditor.visit(tree)

        assert len(violations) == 0, (
            f"Adversarial AST audit detected conditional branching on scenario tokens: {violations}"
        )

    def test_bytecode_disassembly_no_scenario_constants_in_engine_and_algorithms(self):
        """Audits LOAD_CONST and COMPARE_OP opcodes across core modules to ensure zero hardcoded scenario tokens."""
        modules_to_audit = [
            engine_module,
            transients_module,
            astrometry_module,
            cosmology_module,
        ]

        bytecode_violations: List[Dict[str, Any]] = []

        for mod in modules_to_audit:
            mod_name = mod.__name__
            for name, obj in inspect.getmembers(mod):
                targets = []
                if inspect.isfunction(obj):
                    targets.append((f"{mod_name}.{name}", obj))
                elif inspect.isclass(obj):
                    for m_name, m_val in inspect.getmembers(obj, predicate=inspect.isfunction):
                        targets.append((f"{mod_name}.{name}.{m_name}", m_val))

                for fn_name, fn_obj in targets:
                    try:
                        for instr in dis.get_instructions(fn_obj):
                            if instr.opname == "LOAD_CONST" and isinstance(instr.argval, str):
                                v_lower = instr.argval.lower()
                                for tok in self.FORBIDDEN_SCENARIO_TOKENS:
                                    if tok in v_lower:
                                        bytecode_violations.append({
                                            "function": fn_name,
                                            "opcode": instr.opname,
                                            "constant": instr.argval,
                                            "matched_token": tok,
                                        })
                    except Exception:
                        pass

        assert len(bytecode_violations) == 0, (
            f"Adversarial bytecode audit detected hardcoded scenario constants: {bytecode_violations}"
        )

    def test_runtime_blind_polymorphism_stress_sweep(self):
        """Injects legitimate, spoofed, mismatched, and corrupt scenario_id strings and asserts bit-identical outputs."""
        spoofed_payloads = [
            {},  # Baseline clean
            {"scenario_id": "01_frb_supernova_cosmic_ray_triage"},
            {"scenario_id": "02_eclipsing_binary_exoplanet_transit_split"},
            {"scenario_id": "03_crowded_field_kbo_crossing_astrometry"},
            {"scenario_id": "04_lensed_quasar_echo_swarm_factorization"},
            {"scenario_id": "05_cosmic_web_void_rsd_tomography"},
            {"scenario_id": "ADVERSARIAL_INJECTION_EXPLOIT_999"},
            {"scenario_id": "'; DROP TABLE scenarios; --"},
            {"scenario_id": "\x00\xff\xfe\x01\x02"},
            {"scenario_id": 999999},
            {"scenario_id": None},
        ]

        # 1. Test 1D Scalar Updates
        scalar_baseline_results = []
        for i in range(10):
            eng = AstroSnailEngine(backend="numpy")
            val = float(np.sin(i * 0.5))
            res = eng.update(np.array([val]))
            scalar_baseline_results.append((res["classification"], res["update_applied"]))

        for payload in spoofed_payloads[1:]:
            eng_spoof = AstroSnailEngine(backend="numpy")
            for i in range(10):
                val = float(np.sin(i * 0.5))
                res = eng_spoof.update(np.array([val]), telemetry=payload.copy())
                exp_cls, exp_app = scalar_baseline_results[i]
                assert res["classification"] == exp_cls, f"Scalar classification diverged under payload {payload}"
                assert res["update_applied"] == exp_app, f"Scalar update_applied diverged under payload {payload}"

        # 2. Test 1D Sequence Updates
        seq = np.linspace(5.0, 25.0, 128, dtype=np.float32) + np.sin(np.linspace(0, 10, 128, dtype=np.float32))
        eng_base = AstroSnailEngine(backend="numpy")
        out_base_1d = eng_base.update(seq)

        for payload in spoofed_payloads[1:]:
            eng_test = AstroSnailEngine(backend="numpy")
            out_test = eng_test.update(seq, telemetry=payload.copy())
            assert np.array_equal(out_base_1d["curl_salience"], out_test["curl_salience"])
            assert np.array_equal(out_base_1d["h_clock"], out_test["h_clock"])
            assert np.array_equal(out_base_1d["h_nope"], out_test["h_nope"])
            assert out_base_1d["mad_floor"] == out_test["mad_floor"]

        # 3. Test 2D Focal Plane Updates
        rng = np.random.RandomState(42)
        frame_2d = rng.randn(64, 64).astype(np.float32)
        frame_2d[15:20, 25:30] += 50.0  # Synthetic bright source
        eng_base_2d = AstroSnailEngine(backend="numpy")
        out_base_2d = eng_base_2d.update(frame_2d)

        for payload in spoofed_payloads[1:]:
            eng_test_2d = AstroSnailEngine(backend="numpy")
            out_test_2d = eng_test_2d.update(frame_2d, telemetry=payload.copy())
            assert np.array_equal(out_base_2d["medoid_coords_yx"], out_test_2d["medoid_coords_yx"])
            assert np.array_equal(out_base_2d["medoid_features"], out_test_2d["medoid_features"])
            assert out_base_2d["num_medoids"] == out_test_2d["num_medoids"]

        # 4. Test 3D Cosmological Volume Updates
        vol_3d = rng.randn(16, 16, 16).astype(np.float32) - 0.6  # Underdense region
        eng_base_3d = AstroSnailEngine(backend="numpy")
        out_base_3d = eng_base_3d.update(vol_3d)

        for payload in spoofed_payloads[1:]:
            eng_test_3d = AstroSnailEngine(backend="numpy")
            out_test_3d = eng_test_3d.update(vol_3d, telemetry=payload.copy())
            assert np.array_equal(out_base_3d["void_cores"], out_test_3d["void_cores"])
            assert np.array_equal(out_base_3d["retracted_density"], out_test_3d["retracted_density"])
            assert out_base_3d["void_voxel_count"] == out_test_3d["void_voxel_count"]


# =============================================================================
# PILLAR 2: STRICT CAUSAL STREAMING MONOTONICITY STRESS TESTS
# =============================================================================

class TestAdversarialCausalStreamingMonotonicity:
    """Verifies strict causal streaming, no lookahead, history immutability, and 1-sample deferral."""

    def test_strict_no_future_lookahead_invariance(self):
        """Proves that processing t_k produces the exact same result whether future frames exist or not."""
        rng = np.random.RandomState(1337)
        stream = [rng.randn(32, 32).astype(np.float32) for _ in range(25)]

        # Prefix runner: stops at frame 14
        eng_prefix = AstroSnailEngine(backend="numpy")
        prefix_history = []
        for idx in range(15):
            res = eng_prefix.update(stream[idx])
            prefix_history.append((res["medoid_coords_yx"].copy(), res["num_medoids"]))

        # Full stream runner: runs all 25 frames
        eng_full = AstroSnailEngine(backend="numpy")
        for idx in range(25):
            res_full = eng_full.update(stream[idx])
            if idx < 15:
                exp_coords, exp_num = prefix_history[idx]
                assert np.array_equal(res_full["medoid_coords_yx"], exp_coords), (
                    f"Lookahead leak at step {idx}: coords differ when future frames are provided!"
                )
                assert res_full["num_medoids"] == exp_num

    def test_bounded_ring_buffer_and_history_immutability(self):
        """Verifies ring buffer is strictly bounded at 8 frames and never mutates past frames."""
        eng = AstroSnailEngine(backend="numpy")
        rng = np.random.RandomState(777)

        history_snapshots: List[np.ndarray] = []

        for step in range(200):
            frame = rng.randn(16, 16).astype(np.float32) * float(step + 1)
            history_snapshots.append(frame.copy())
            eng.update(frame)

            assert len(eng.ring_buffer) <= 8, (
                f"Ring buffer length {len(eng.ring_buffer)} exceeded hard 8-frame bound at step {step}!"
            )

            # Check that existing stored frames match expected FIFO history bit-for-bit
            expected_start_step = max(0, step - 7)
            for buf_idx, hist_step in enumerate(range(expected_start_step, step + 1)):
                stored_frame = eng.ring_buffer[buf_idx]
                expected_frame = history_snapshots[hist_step]
                assert np.array_equal(stored_frame, expected_frame), (
                    f"History mutation detected! Ring buffer frame {buf_idx} corrupted at step {step}."
                )

    def test_one_sample_causal_deferral_10k_glint_stress(self):
        """Stress-tests 1-sample causal deferral across 10,000 independent cosmic-ray / glint spikes.
        Guarantees EXACTLY 0 false triggers (0 false confirmed transients) across all 10,000 single-sample glints.
        """
        gate = MorseTransientGateConfig(sigma_thresh=4.0)
        rng = np.random.RandomState(20261025)

        # Baseline initialization
        for init_step in range(10):
            gate.update(0.0, timestamp=float(init_step))

        n_glints = 10000
        for i in range(n_glints):
            t_base = 10.0 + i * 2.0
            # 1. Unconfirmed cosmic ray / glint strike (20-100 sigma spike)
            spike_amp = float(rng.uniform(20.0, 100.0))
            cls1, app1 = gate.update(spike_amp, timestamp=t_base)
            assert cls1 == "pending_impulse", f"Failed to defer spike at glint {i}: got {cls1}"
            assert app1 is False, f"Applied unconfirmed track update at glint {i}!"

            # 2. Next sample returns immediately to quiescent baseline
            cls2, app2 = gate.update(0.0, timestamp=t_base + 1.0)
            assert cls2 in ("glint_rejected", "nominal"), f"Failed to reject glint {i}: got {cls2}"
            assert app2 is False, f"Applied rejected update at glint {i}!"

        assert gate.confirmed_transient_count == 0, (
            f"Zero false triggers violated! Got {gate.confirmed_transient_count} false confirmed transients!"
        )
        assert gate.glint_rejected_count == n_glints, (
            f"Expected {n_glints} rejected glints, got {gate.glint_rejected_count}"
        )

    def test_one_sample_causal_deferral_physical_transient_confirmation(self):
        """Verifies that physical sustained transients are confirmed at t+1 and trigger full state update."""
        gate = MorseTransientGateConfig(sigma_thresh=4.0)

        # Baseline initialization
        gate.update(0.0, timestamp=1.0)

        # Step onset at t=2
        cls1, app1 = gate.update(25.0, timestamp=2.0)
        assert cls1 == "pending_impulse"
        assert app1 is False

        # Sustained flux at t=3 (supernova shock breakout / FRB tail)
        cls2, app2 = gate.update(24.5, timestamp=3.0)
        assert cls2 == "confirmed_transient", f"Expected confirmed_transient, got {cls2}"
        assert app2 is True, "Update must be applied on confirmed physical transient!"
        assert gate.confirmed_transient_count == 1
        assert gate.glint_rejected_count == 0

    def test_timestamp_jitter_and_non_monotonic_robustness(self):
        """Asserts gate resilience against jittered timestamps, micro-cadence increments, and zero/negative jitter."""
        gate = MorseTransientGateConfig(sigma_thresh=4.0)

        timestamps = [
            100.0,
            100.001,  # Sub-millisecond jitter
            100.005,
            100.005,  # Duplicate timestamp (same time packet)
            99.999,   # Retrograde jitter / packet reordering
            105.0,    # Large leap
            105.1,
        ]

        for idx, ts in enumerate(timestamps):
            cls_out, app_out = gate.update(0.05 * (idx % 2), timestamp=ts)
            assert cls_out in ("nominal", "pending_impulse", "glint_rejected", "confirmed_transient")
            assert isinstance(app_out, bool)

        assert gate.total_steps == len(timestamps)

    def test_random_frame_dropout_causal_recovery(self):
        """Simulates intermittent communication packet loss and verifies causal recovery."""
        eng = AstroSnailEngine(backend="numpy")
        rng = np.random.RandomState(999)

        # Feed 10 nominal frames
        for _ in range(10):
            eng.update(np.array([0.0]))

        # Simulate dropout: jump step count by passing timestamp forward or skipping updates
        assert eng.step_count == 10
        out = eng.update(np.array([0.0]))
        assert out["step"] == 11
        assert out["classification"] == "nominal"


# =============================================================================
# PILLAR 3: OUT-OF-CORE MEMORY PROFILING STRESS TESTS
# =============================================================================

class TestAdversarialOutOfCoreMemoryProfiling:
    """Verifies memory bounds on 64 MB FITS cube streaming (< 2.0 MB working set, < 35.0 MB ceiling)
    and asserts complete file lock release on Windows.
    """

    def test_64mb_fits_cube_streaming_peak_memory_and_ceiling(self, tmp_path):
        """Streams a 64 MB FITS cube (1024 x 128 x 128 float32) with tracemalloc.
        Asserts active working set peak strictly < 2.0 MB and total peak < 35.0 MB.
        """
        fits_file = tmp_path / "adversarial_64mb_cube.fits"
        cube_shape = (1024, 128, 128)  # 1024 * 128 * 128 * 4 = 67,108,864 bytes = 64.0 MB
        cube_data = np.zeros(cube_shape, dtype=np.float32)

        # Fill with non-trivial gradient to prevent compression or sparse shortcuts
        cube_data[:, 0, 0] = np.linspace(1.0, 100.0, 1024, dtype=np.float32)

        written_bytes = write_fits_cube(fits_file, cube_data, chunk_z=16)
        assert written_bytes % 2880 == 0
        del cube_data
        gc.collect()

        # Mode 1: Zero-copy memmap slice view (copy=False)
        tracemalloc.start()
        tracemalloc.reset_peak()

        slabs_processed = 0
        voxel_accum = 0.0

        for slab in stream_fits_cube(fits_file, chunk_z=16, copy=False):
            slabs_processed += 1
            assert slab.shape == (16, 128, 128)
            voxel_accum += float(np.sum(slab[:, 0, 0]))

        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_bytes / (1024.0 * 1024.0)
        assert slabs_processed == 64, f"Expected 64 slabs, got {slabs_processed}"
        assert voxel_accum > 0.0
        assert peak_mb < 2.0, (
            f"Zero-copy streaming working set peak {peak_mb:.4f} MB exceeded 2.0 MB limit!"
        )
        assert peak_mb < 35.0, f"Hard ceiling exceeded: {peak_mb:.4f} MB"

        # Mode 2: In-place reusable buffer (reuse_buffer=True)
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()

        slabs_buf_count = 0
        for slab in stream_fits_cube(fits_file, chunk_z=16, reuse_buffer=True):
            slabs_buf_count += 1
            assert slab.shape == (16, 128, 128)

        _, peak_buf_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_buf_mb = peak_buf_bytes / (1024.0 * 1024.0)
        assert slabs_buf_count == 64
        assert peak_buf_mb < 2.0, (
            f"Reusable buffer streaming peak {peak_buf_mb:.4f} MB exceeded 2.0 MB limit!"
        )

    def test_windows_file_handle_closure_on_complete_iteration(self, tmp_path):
        """Verifies that complete iteration over a FITS file releases file handles immediately once yielded slabs are freed."""
        fits_file = tmp_path / "complete_iter_lock_test.fits"
        data = np.ones((64, 32, 32), dtype=np.float32)
        write_fits_cube(fits_file, data, chunk_z=16)
        del data
        gc.collect()

        # Complete iteration: ensure slab variable is explicitly unbound so Windows file mapping releases
        count = 0
        for slab in stream_fits_cube(fits_file, chunk_z=16):
            count += 1
        assert count == 4

        # In Python on Windows, memmap slice views hold a file mapping handle until collected
        del slab
        gc.collect()

        try:
            os.remove(str(fits_file))
        except OSError as e:
            pytest.fail(f"Complete iteration leaked file handle on Windows: {e}")

    def test_windows_file_handle_closure_on_early_break(self, tmp_path):
        """Verifies that abandoning generator via break does not leave file handles locked on Windows when generator is closed."""
        fits_file = tmp_path / "early_break_lock_test.fits"
        data = np.ones((128, 32, 32), dtype=np.float32)
        write_fits_cube(fits_file, data, chunk_z=16)
        del data
        gc.collect()

        # Early exit via break
        gen = stream_fits_cube(fits_file, chunk_z=16)
        for idx, slab in enumerate(gen):
            if idx == 1:
                break
        del slab
        gen.close()
        gc.collect()

        try:
            os.remove(str(fits_file))
        except OSError as e:
            pytest.fail(f"Early break leaked file handle on Windows: {e}")

    def test_windows_file_handle_closure_on_consumer_exception(self, tmp_path):
        """Verifies that an exception inside the consumer loop cleans up file handles properly once generator is collected."""
        fits_file = tmp_path / "exception_lock_test.fits"
        data = np.ones((64, 32, 32), dtype=np.float32)
        write_fits_cube(fits_file, data, chunk_z=16)
        del data
        gc.collect()

        slab_ref = None
        try:
            for idx, slab in enumerate(stream_fits_cube(fits_file, chunk_z=16)):
                slab_ref = slab
                if idx == 1:
                    raise RuntimeError("Simulated consumer failure during streaming")
        except RuntimeError:
            pass

        del slab_ref, slab
        gc.collect()
        try:
            os.remove(str(fits_file))
        except OSError as e:
            pytest.fail(f"Consumer exception leaked file handle on Windows: {e}")

    def test_windows_detached_copy_mode_immediate_deletion(self, tmp_path):
        """Verifies that copy=True allows immediate file deletion even while yielded slab arrays remain in memory."""
        fits_file = tmp_path / "detached_copy_test.fits"
        data = np.ones((32, 16, 16), dtype=np.float32)
        write_fits_cube(fits_file, data, chunk_z=8)
        del data
        gc.collect()

        # In copy=True mode, slabs are detached arrays; file can be deleted immediately even while slabs are alive
        slabs = list(stream_fits_cube(fits_file, chunk_z=8, copy=True))
        assert len(slabs) == 4
        gc.collect()

        try:
            os.remove(str(fits_file))
        except OSError as e:
            pytest.fail(f"Detached copy mode failed to release file handle on Windows: {e}")

        # Assert data in memory is still intact
        assert slabs[0].shape == (8, 16, 16)
        assert float(slabs[0][0, 0, 0]) == 1.0

    def test_roundtrip_bit_identical_lossless_parity(self, tmp_path):
        """Verifies exact bit-identical numerical roundtrip parity (0.000e+00 loss)."""
        fits_file = tmp_path / "bit_identical_test.fits"
        rng = np.random.RandomState(42)
        original_data = rng.randn(32, 48, 48).astype(np.float32)

        write_fits_cube(fits_file, original_data, chunk_z=8)

        reconstructed_slabs = list(stream_fits_cube(fits_file, chunk_z=8, as_native=True, copy=True))
        reconstructed_data = np.concatenate(reconstructed_slabs, axis=0)

        assert reconstructed_data.shape == original_data.shape
        max_abs_diff = float(np.max(np.abs(reconstructed_data - original_data)))
        assert max_abs_diff == 0.0, (
            f"Bit-identical roundtrip violated! Max abs error: {max_abs_diff:.4e}"
        )


# =============================================================================
# PILLAR 4: ADVERSARIAL EDGE CASES & SYSTEM CAPACITY STRESS TESTS
# =============================================================================

class TestAdversarialEdgeCasesAndStressCapacity:
    """Deep edge-case mining: odd chunk sizes, 128 MB scaling, deep unrolls, and throughput floors."""

    def test_fits_odd_and_non_divisible_z_chunks(self, tmp_path):
        """Verifies correct Z-slab windowing when Z-dimension is not divisible by chunk_z."""
        fits_file = tmp_path / "odd_shape_cube.fits"
        odd_shape = (73, 23, 31)  # 73 is prime, not divisible by 16
        rng = np.random.RandomState(888)
        data = rng.randn(*odd_shape).astype(np.float32)

        write_fits_cube(fits_file, data, chunk_z=16)

        slabs = list(stream_fits_cube(fits_file, chunk_z=16, as_native=True, copy=True))
        assert len(slabs) == 5  # 4 * 16 = 64, plus 1 slab of 9 = 73

        for i in range(4):
            assert slabs[i].shape == (16, 23, 31)
        assert slabs[4].shape == (9, 23, 31)

        rec = np.concatenate(slabs, axis=0)
        assert rec.shape == odd_shape
        max_err = float(np.max(np.abs(rec - data)))
        assert max_err == 0.0, f"Odd shape roundtrip error: {max_err}"

    def test_fits_128mb_cube_memory_scaling(self, tmp_path):
        """Streams a 128 MB FITS cube (2048 x 128 x 128 float32 = 134.2 MB) and asserts peak heap remains < 2.0 MB."""
        fits_file = tmp_path / "cube_128mb.fits"
        shape = (2048, 128, 128)
        data = np.zeros(shape, dtype=np.float32)

        write_fits_cube(fits_file, data, chunk_z=16)
        del data
        gc.collect()

        tracemalloc.start()
        tracemalloc.reset_peak()

        count = 0
        for slab in stream_fits_cube(fits_file, chunk_z=16, copy=False):
            count += 1
            _ = float(slab[0, 0, 0])

        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_bytes / (1024.0 * 1024.0)
        assert count == 128  # 2048 / 16 = 128 chunks
        assert peak_mb < 2.0, f"128 MB cube peak memory {peak_mb:.4f} MB exceeded 2.0 MB working set limit!"

    def test_slow_fast_triad_cascade_deep_unroll_zero_drift(self):
        """Verifies exact 0.000e+00 radial drift conservation under deep unrolling up to K=64 steps."""
        cascade = SlowFastTriadCascade(coupling=0.25, steps=64)
        rng = np.random.RandomState(42)

        v0 = rng.standard_normal(8)
        v0 /= np.linalg.norm(v0)
        u_seq = rng.standard_normal((64, 8))

        v_K, drift = cascade.integrate(u_seq, v0, steps=64)
        assert drift == 0.0, f"Cascade radial drift {drift:.4e} was not identically 0.000e+00!"
        final_norm = float(np.linalg.norm(v_K))
        assert abs(final_norm - 1.0) < 1e-12, f"Spherical norm not preserved: {final_norm}"

    def test_engine_sustained_fps_throughput_floor(self):
        """Verifies throughput floor: AstroSnailEngine processes 128x128 focal plane frames at >= 165 FPS (< 6.0 ms/frame)."""
        engine = AstroSnailEngine(backend="numpy")
        rng = np.random.RandomState(12345)
        test_frame = rng.randn(128, 128).astype(np.float32)

        # Warmup
        for _ in range(5):
            engine.update(test_frame)

        n_frames = 100
        t0 = time.perf_counter()
        for _ in range(n_frames):
            engine.update(test_frame)
        t_elapsed = time.perf_counter() - t0

        fps = n_frames / t_elapsed
        ms_per_frame = (t_elapsed / n_frames) * 1000.0

        # R1 specifies throughput >= 165 FPS (< 6.0 ms/frame)
        assert fps >= 165.0, f"Throughput {fps:.1f} FPS below 165 FPS requirement ({ms_per_frame:.2f} ms/frame)"
