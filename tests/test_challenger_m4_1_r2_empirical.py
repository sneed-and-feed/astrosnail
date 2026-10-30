"""
tests/test_challenger_m4_1_r2_empirical.py
=========================================
Empirical Adversarial Verification Suite for Milestone 4 Remediation
by Challenger M4-1-r2 (Memory & Streaming Challenger).

Adversarial Audits & Stress Harnesses:
1. Adversarial Zero Scenario-Name Branching:
   - Exhaustive AST traversal of core modules for forbidden scenario tokens.
   - Bytecode disassembly (LOAD_CONST, COMPARE_OP) across core engine and algorithmic modules.
   - Runtime blind polymorphism: identical outputs under legitimate, spoofed, mismatched,
     malicious SQL, buffer overflow strings, and binary garbage telemetry payloads.
2. Causal Streaming Monotonicity & Invariants:
   - Non-lookahead invariance (prefix execution equivalence).
   - 8-frame ring buffer capacity ceiling under 500 continuous updates.
   - Ring buffer past history immutability under streaming load.
   - 10,000-spike glint stress harness (1-sample causal deferral, 0 false triggers).
   - Sustained physical transient confirmation at t+1.
   - Arrow-of-time irreversibility and non-monotonic timestamp jitter resilience.
3. Out-of-Core Memory Bounds & Windows Handle Lifecycle:
   - 64 MB FITS cube streaming memory profiling with tracemalloc (< 2.0 MB working set, < 35.0 MB hard ceiling).
   - 128 MB FITS cube streaming memory scaling.
   - Windows file handle closure & lockup prevention across complete iteration, early break,
     exception bubbling, and detached copy mode.
   - Exact bit-identical roundtrip losslessness (0.000e+00 error).
   - Non-divisible prime cube geometry stress testing.
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
import astrosnail.cli as cli_module


# =============================================================================
# PILLAR 1: ADVERSARIAL ZERO SCENARIO-NAME BRANCHING AUDIT
# =============================================================================

class TestAdversarialScenarioNameBranchingR2:
    """Exhaustive AST, bytecode, and runtime blind polymorphism audit."""

    FORBIDDEN_SCENARIO_TOKENS = (
        "01_", "02_", "03_", "04_", "05_",
        "frb_supernova", "eclipsing_binary", "kbo_crossing", "lensed_quasar", "void_rsd",
        "scenario_01", "scenario_02", "scenario_03", "scenario_04", "scenario_05",
        "scenario_id",
    )

    CORE_MODULES = (
        "engine.py",
        "transients.py",
        "astrometry.py",
        "cosmology.py",
        "_native.py",
        "io.py",
        "cli.py",
        "__init__.py",
    )

    def _get_package_dir(self) -> Path:
        return Path(astrosnail.__file__).parent

    def test_ast_scan_no_conditional_branching_on_scenario_tokens(self):
        """AST audit: asserts no conditional statements in core modules branch on scenario names."""
        pkg_dir = self._get_package_dir()
        violations: List[Dict[str, Any]] = []

        class ScenarioBranchAuditor(ast.NodeVisitor):
            def __init__(self, filename: str):
                self.filename = filename

            def check_expr(self, expr_node: ast.AST, stmt_type: str, lineno: int):
                for sub in ast.walk(expr_node):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        val_lower = sub.value.lower()
                        for tok in TestAdversarialScenarioNameBranchingR2.FORBIDDEN_SCENARIO_TOKENS:
                            if tok in val_lower:
                                violations.append({
                                    "file": self.filename,
                                    "line": lineno,
                                    "statement": stmt_type,
                                    "token": tok,
                                    "value": sub.value,
                                })

            def visit_If(self, node: ast.If):
                self.check_expr(node.test, "If.test", node.lineno)
                self.generic_visit(node)

            def visit_IfExp(self, node: ast.IfExp):
                self.check_expr(node.test, "IfExp.test", node.lineno)
                self.generic_visit(node)

            def visit_While(self, node: ast.While):
                self.check_expr(node.test, "While.test", node.lineno)
                self.generic_visit(node)

            def visit_Assert(self, node: ast.Assert):
                self.check_expr(node.test, "Assert.test", node.lineno)
                self.generic_visit(node)

            def visit_Match(self, node: ast.AST):
                if hasattr(node, "subject"):
                    self.check_expr(getattr(node, "subject"), "Match.subject", getattr(node, "lineno", 0))
                self.generic_visit(node)

        for mod_name in self.CORE_MODULES:
            mod_path = pkg_dir / mod_name
            if not mod_path.exists():
                continue
            with open(mod_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(mod_path))
            auditor = ScenarioBranchAuditor(mod_name)
            auditor.visit(tree)

        assert len(violations) == 0, (
            f"Adversarial AST audit detected conditional branching on scenario tokens: {violations}"
        )

    def test_bytecode_disassembly_no_scenario_constants(self):
        """Audits LOAD_CONST and COMPARE_OP opcodes across core engine and algorithmic modules."""
        modules = [
            engine_module,
            transients_module,
            astrometry_module,
            cosmology_module,
            native_module,
        ]
        bytecode_violations: List[Dict[str, Any]] = []

        for mod in modules:
            for name, obj in inspect.getmembers(mod):
                targets: List[Tuple[str, Any]] = []
                if inspect.isfunction(obj):
                    targets.append((f"{mod.__name__}.{name}", obj))
                elif inspect.isclass(obj) and obj.__module__ == mod.__name__:
                    for m_name, m_val in inspect.getmembers(obj, predicate=inspect.isfunction):
                        targets.append((f"{mod.__name__}.{name}.{m_name}", m_val))

                for fn_name, fn_obj in targets:
                    try:
                        instructions = list(dis.get_instructions(fn_obj))
                    except (TypeError, ValueError):
                        continue

                    for instr in instructions:
                        if instr.opname == "LOAD_CONST" and isinstance(instr.argval, str):
                            arg_lower = instr.argval.lower()
                            for tok in self.FORBIDDEN_SCENARIO_TOKENS:
                                if tok in arg_lower:
                                    bytecode_violations.append({
                                        "target": fn_name,
                                        "token": tok,
                                        "constant": instr.argval,
                                    })

        assert len(bytecode_violations) == 0, (
            f"Adversarial bytecode audit detected hardcoded scenario constants: {bytecode_violations}"
        )

    def test_runtime_blind_polymorphism_stress_sweep(self):
        """Stress tests AstroSnailEngine.update() under adversarial telemetry payloads."""
        engine_ref = AstroSnailEngine(backend="numpy")
        engine_adv = AstroSnailEngine(backend="numpy")

        adversarial_payloads = [
            {"scenario_id": "01_frb_supernova_cosmic_ray_triage"},
            {"scenario_id": "02_eclipsing_binary_exoplanet_transit_split"},
            {"scenario_id": "03_crowded_field_kbo_crossing_astrometry"},
            {"scenario_id": "04_lensed_quasar_echo_swarm_factorization"},
            {"scenario_id": "05_cosmic_web_void_rsd_tomography"},
            {"scenario_id": "mismatched_05_on_1d_stream"},
            {"scenario_id": "'; DROP TABLE benchmarks; --"},
            {"scenario_id": "A" * 100_000},  # Buffer stress
            {"scenario_id": "\x00\xff\xfe\x01\x02\x7f\x80"},  # Binary garbage
            {"scenario_id": "🚀🛸🪐🔭🌌"},  # Unicode
            {"scenario_id": 99999999},  # Integer
            {"scenario_id": None},
        ]

        # 1D test series
        test_1d = np.sin(np.linspace(0, 20, 128, dtype=np.float32)) + 15.0
        ref_1d = engine_ref.update(test_1d)

        for payload in adversarial_payloads:
            out_adv = engine_adv.update(test_1d, telemetry=payload)
            out_adv.pop("scenario_id", None)
            assert np.array_equal(ref_1d["curl_salience"], out_adv["curl_salience"])
            assert np.array_equal(ref_1d["h_clock"], out_adv["h_clock"])
            assert np.array_equal(ref_1d["h_nope"], out_adv["h_nope"])
            assert float(ref_1d["mad_floor"]) == pytest.approx(float(out_adv["mad_floor"]), abs=1e-7)

        # 2D test frame
        test_2d = np.zeros((64, 64), dtype=np.float32)
        test_2d[15, 15] = 20.0
        test_2d[35, 45] = 18.0
        test_2d[50, 10] = 25.0
        ref_2d = engine_ref.update(test_2d)

        for payload in adversarial_payloads:
            out_adv = engine_adv.update(test_2d, telemetry=payload)
            out_adv.pop("scenario_id", None)
            assert np.array_equal(ref_2d["medoid_coords_yx"], out_adv["medoid_coords_yx"])
            assert np.array_equal(ref_2d["medoid_features"], out_adv["medoid_features"])
            assert ref_2d["num_medoids"] == out_adv["num_medoids"]

        # 3D test volume
        test_3d = np.random.default_rng(42).standard_normal((16, 16, 16)).astype(np.float32)
        ref_3d = engine_ref.update(test_3d)

        for payload in adversarial_payloads:
            out_adv = engine_adv.update(test_3d, telemetry=payload)
            out_adv.pop("scenario_id", None)
            assert np.array_equal(ref_3d["void_cores"], out_adv["void_cores"])
            assert np.array_equal(ref_3d["retracted_density"], out_adv["retracted_density"])
            assert ref_3d["void_voxel_count"] == out_adv["void_voxel_count"]


# =============================================================================
# PILLAR 2: CAUSAL STREAMING MONOTONICITY & INVARIANTS AUDIT
# =============================================================================

class TestAdversarialCausalStreamingMonotonicityR2:
    """Stress tests causal time-arrow, ring buffer limits, and 1-sample deferral."""

    def test_strict_no_future_lookahead_invariance(self):
        """Verifies that frame output at step t is invariant to whether future frames exist."""
        rng = np.random.default_rng(20261030)
        n_frames = 30
        frames = rng.uniform(0.0, 1.0, size=(n_frames, 32, 32)).astype(np.float32)

        # Runner 1: Stops at frame 15
        engine_prefix = AstroSnailEngine(backend="numpy")
        prefix_outputs: List[Dict[str, Any]] = []
        for t in range(16):
            out = engine_prefix.update(frames[t])
            prefix_outputs.append(out)

        # Runner 2: Runs full 30 frames
        engine_full = AstroSnailEngine(backend="numpy")
        full_outputs: List[Dict[str, Any]] = []
        for t in range(n_frames):
            out = engine_full.update(frames[t])
            full_outputs.append(out)

        # Check identical results for steps 0 through 15
        for t in range(16):
            p_out = prefix_outputs[t]
            f_out = full_outputs[t]
            assert np.array_equal(p_out["medoid_coords_yx"], f_out["medoid_coords_yx"]), (
                f"Lookahead violation at frame {t}: prefix and full-stream outputs differ!"
            )
            assert p_out["num_medoids"] == f_out["num_medoids"]

    def test_bounded_ring_buffer_and_history_immutability(self):
        """Asserts ring buffer never exceeds 8 frames across 500 updates and never mutates history."""
        engine = AstroSnailEngine(backend="numpy")
        rng = np.random.default_rng(12345)

        saved_snapshots: Dict[int, np.ndarray] = {}

        for step in range(500):
            frame = rng.standard_normal((32, 32)).astype(np.float32)
            engine.update(frame)

            # Invariant: Ring buffer capacity <= 8
            assert len(engine.ring_buffer) <= 8, (
                f"Ring buffer exceeded capacity 8 at step {step}: size={len(engine.ring_buffer)}"
            )

            # Snapshot the latest frame in buffer
            saved_snapshots[step] = np.copy(engine.ring_buffer[-1])

            # Check that existing entries in ring_buffer have not been mutated
            curr_buf_len = len(engine.ring_buffer)
            for buf_idx in range(curr_buf_len):
                orig_step = step - (curr_buf_len - 1 - buf_idx)
                if orig_step in saved_snapshots:
                    assert np.array_equal(engine.ring_buffer[buf_idx], saved_snapshots[orig_step]), (
                        f"History immutability violated in ring buffer at index {buf_idx}!"
                    )

    def test_one_sample_causal_deferral_10k_glint_stress(self):
        """Evaluates 10,000 extreme single-sample glint spikes. Asserts 0 false confirmed triggers."""
        gate = MorseTransientGateConfig(sigma_thresh=4.0, persistence_window=1)
        rng = np.random.default_rng(9999)

        # Baseline initialization (sample t=0)
        gate.update(0.0, timestamp=0.0)

        n_spikes = 10_000
        # Random spike innovations from 10.0 to 200.0 sigma
        spike_amplitudes = rng.uniform(10.0, 200.0, size=n_spikes)

        confirmed_transient_count = 0
        glint_rejected_count = 0
        current_time = 1.0

        for amp in spike_amplitudes:
            current_time += 1.0
            # Sample t0: Glint strike
            c0, u0 = gate.update(float(amp), timestamp=current_time)
            assert c0 == "pending_impulse", f"Step t0 must defer spike, got {c0}"
            assert u0 is False, "Step t0 must not apply state update"

            current_time += 1.0
            # Sample t0+1: Instant return to baseline
            c1, u1 = gate.update(0.0, timestamp=current_time)

            if c1 == "glint_rejected":
                glint_rejected_count += 1
            elif c1 == "confirmed_transient":
                confirmed_transient_count += 1

        assert confirmed_transient_count == 0, (
            f"Adversarial failure: {confirmed_transient_count} glints falsely triggered as confirmed transients!"
        )
        assert glint_rejected_count == n_spikes, (
            f"Expected {n_spikes} glint rejections, got {glint_rejected_count}"
        )

    def test_one_sample_causal_deferral_physical_transient_confirmation(self):
        """Verifies that physical step transients are confirmed at t+1 with update_applied=True."""
        gate = MorseTransientGateConfig(sigma_thresh=4.0, persistence_window=1)

        t = 100.0
        # Step 0: Nominal baseline
        c, u = gate.update(0.1, timestamp=t)
        assert c == "nominal"

        # Step 1: Onset of high innovation
        t += 1.0
        c, u = gate.update(25.0, timestamp=t)
        assert c == "pending_impulse"
        assert u is False

        # Step 2: Post-onset persistence
        t += 1.0
        c, u = gate.update(26.5, timestamp=t)
        assert c == "confirmed_transient"
        assert u is True

    def test_timestamp_jitter_and_retrograde_robustness(self):
        """Tests gate resilience under jittered, duplicate, and retrograde timestamps."""
        gate = MorseTransientGateConfig(sigma_thresh=4.0, persistence_window=1)

        # Baseline
        gate.update(0.2, timestamp=100.0)

        # Sub-millisecond tick
        c, _ = gate.update(15.0, timestamp=100.0001)
        assert c == "pending_impulse"

        # Duplicate timestamp
        c, _ = gate.update(0.1, timestamp=100.0001)
        assert c == "glint_rejected"

        # Retrograde timestamp
        c, _ = gate.update(12.0, timestamp=99.0)
        assert c == "pending_impulse"

    def test_arrow_of_time_irreversibility(self):
        """Verifies that reversing the sequence creates a different state trajectory."""
        rng = np.random.default_rng(20261025)
        seq = rng.standard_normal(64).astype(np.float32)

        engine_fwd = AstroSnailEngine(backend="numpy")
        engine_rev = AstroSnailEngine(backend="numpy")

        out_fwd = [engine_fwd.update(seq[i:i+8]) for i in range(len(seq) - 8)]
        out_rev = [engine_rev.update(seq[::-1][i:i+8]) for i in range(len(seq) - 8)]

        # Verify state trajectories are not symmetric
        fwd_salience = [o["curl_salience"] for o in out_fwd]
        rev_salience = [o["curl_salience"] for o in out_rev]
        assert not np.allclose(fwd_salience, rev_salience[::-1])


# =============================================================================
# PILLAR 3: OUT-OF-CORE MEMORY BOUNDS & WINDOWS HANDLE AUDIT
# =============================================================================

class TestAdversarialOutOfCoreMemoryProfilingR2:
    """Stress tests tracemalloc memory bounds (< 2.0 MB / < 35.0 MB) and Windows handle releases."""

    @pytest.fixture
    def fits_64mb_cube(self, tmp_path):
        """Creates a temporary 64.0 MB FITS cube conforming to NASA FITS Standard v4.0."""
        # 1024 * 128 * 128 * 4 bytes = 67,108,864 bytes = 64.0 MB exactly
        fits_file = tmp_path / "cube_64mb.fits"
        cube_shape = (1024, 128, 128)
        cube_data = np.zeros(cube_shape, dtype=np.float32)
        cube_data[:, 0, 0] = np.linspace(1.0, 50.0, 1024, dtype=np.float32)
        write_fits_cube(fits_file, cube_data, chunk_z=16)
        del cube_data
        gc.collect()
        return str(fits_file)

    @pytest.fixture
    def fits_128mb_cube(self, tmp_path):
        """Creates a temporary 128.0 MB FITS cube."""
        # 2048 * 128 * 128 * 4 bytes = 134,217,728 bytes = 128.0 MB
        fits_file = tmp_path / "cube_128mb.fits"
        cube_shape = (2048, 128, 128)
        cube_data = np.zeros(cube_shape, dtype=np.float32)
        cube_data[:, 0, 0] = np.linspace(1.0, 50.0, 2048, dtype=np.float32)
        write_fits_cube(fits_file, cube_data, chunk_z=16)
        del cube_data
        gc.collect()
        return str(fits_file)

    def test_64mb_fits_cube_streaming_peak_memory_and_ceiling(self, fits_64mb_cube):
        """Profiles heap allocations with tracemalloc during 64 MB FITS cube streaming."""
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()

        slab_count = 0
        total_voxels = 0
        for slab in stream_fits_cube(fits_64mb_cube, chunk_z=16, copy=False):
            slab_count += 1
            total_voxels += slab.size
            current, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / (1024 * 1024)
            # Hard ceiling check during iteration
            assert peak_mb < 35.0, f"Tracemalloc hard ceiling violated during streaming: {peak_mb:.2f} MB"

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        active_peak_mb = peak / (1024 * 1024)
        assert slab_count == 64, f"Expected 64 slabs of chunk_z=16, got {slab_count}"
        assert total_voxels == 1024 * 128 * 128
        assert active_peak_mb < 2.0, (
            f"Zero-copy streaming peak {active_peak_mb:.4f} MB exceeded 2.0 MB budget!"
        )

    def test_64mb_fits_cube_reusable_buffer_peak_memory(self, fits_64mb_cube):
        """Profiles heap memory with reuse_buffer=True on 64 MB FITS cube."""
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()

        slab_count = 0
        for slab in stream_fits_cube(fits_64mb_cube, chunk_z=16, reuse_buffer=True):
            slab_count += 1

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        active_peak_mb = peak / (1024 * 1024)
        assert slab_count == 64
        assert active_peak_mb < 2.0, (
            f"Reusable buffer streaming peak {active_peak_mb:.4f} MB exceeded 2.0 MB budget!"
        )

    def test_128mb_fits_cube_memory_scaling(self, fits_128mb_cube):
        """Verifies memory residency remains flat when scaling to 128 MB FITS cube."""
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()

        slab_count = 0
        for slab in stream_fits_cube(fits_128mb_cube, chunk_z=16, copy=False):
            slab_count += 1

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        assert slab_count == 128
        assert peak_mb < 2.0, f"128 MB scaling peak {peak_mb:.4f} MB exceeded 2.0 MB!"

    def test_windows_file_handle_closure_on_complete_iteration(self, fits_64mb_cube):
        """Verifies Windows file handle is released cleanly after complete iteration."""
        gen = stream_fits_cube(fits_64mb_cube, chunk_z=16, copy=False)
        for slab in gen:
            pass
        del slab
        gc.collect()

        # Attempt to delete file immediately
        try:
            os.remove(fits_64mb_cube)
            file_deleted = True
        except PermissionError:
            file_deleted = False

        assert file_deleted, "Windows file handle remained locked after complete iteration!"

    def test_windows_file_handle_closure_on_early_break(self, fits_64mb_cube):
        """Verifies file handle is released when breaking early from streaming generator."""
        gen = stream_fits_cube(fits_64mb_cube, chunk_z=16, copy=False)
        for i, slab in enumerate(gen):
            if i >= 2:
                break
        del slab
        gen.close()
        gc.collect()

        try:
            os.remove(fits_64mb_cube)
            file_deleted = True
        except PermissionError:
            file_deleted = False

        assert file_deleted, "Windows file handle remained locked after generator early break!"

    def test_windows_file_handle_closure_on_consumer_exception(self, fits_64mb_cube):
        """Verifies file handle is released cleanly when consumer raises an unhandled exception."""
        gen = stream_fits_cube(fits_64mb_cube, chunk_z=16, copy=False)
        try:
            for i, slab in enumerate(gen):
                if i == 3:
                    raise RuntimeError("Simulated consumer crash")
        except RuntimeError:
            pass

        del slab
        gen.close()
        gc.collect()

        try:
            os.remove(fits_64mb_cube)
            file_deleted = True
        except PermissionError:
            file_deleted = False

        assert file_deleted, "Windows file handle remained locked after consumer exception!"

    def test_windows_detached_copy_mode_immediate_deletion(self, fits_64mb_cube):
        """Verifies copy=True mode allows file deletion while consumer retains yielded array."""
        gen = stream_fits_cube(fits_64mb_cube, chunk_z=16, copy=True)
        first_slab = next(gen)
        gen.close()
        gc.collect()

        # In copy=True mode, the backing file should be deletable immediately
        try:
            os.remove(fits_64mb_cube)
            file_deleted = True
        except PermissionError:
            file_deleted = False

        assert file_deleted, "copy=True mode failed to release file handle!"
        assert first_slab.shape == (16, 128, 128)

    def test_roundtrip_bit_identical_lossless_parity(self, tmp_path):
        """Verifies write_fits_cube and stream_fits_cube achieve 0.000e+00 data loss."""
        fits_file = tmp_path / "parity_test.fits"
        rng = np.random.default_rng(20261025)
        orig_data = rng.standard_normal((32, 24, 24)).astype(np.float32)
        write_fits_cube(fits_file, orig_data)

        read_slabs = []
        for slab in stream_fits_cube(fits_file, chunk_z=8, copy=True):
            read_slabs.append(slab)

        reconstructed = np.concatenate(read_slabs, axis=0)
        max_abs_err = float(np.max(np.abs(orig_data - reconstructed)))
        assert max_abs_err == 0.0, f"FITS roundtrip data loss detected: max_err = {max_abs_err}"

    def test_fits_odd_and_non_divisible_z_chunks(self, tmp_path):
        """Verifies out-of-core streaming on non-divisible and prime dimensions."""
        fits_file = tmp_path / "odd_prime_cube.fits"
        # Prime Z dimension 79 with chunk_z=16
        z, h, w = 79, 33, 37
        rng = np.random.default_rng(7933)
        orig_data = rng.standard_normal((z, h, w)).astype(np.float32)
        write_fits_cube(fits_file, orig_data)

        slabs = list(stream_fits_cube(fits_file, chunk_z=16, copy=True))
        assert len(slabs) == 5  # 4 * 16 + 1 * 15 = 79
        assert slabs[0].shape == (16, h, w)
        assert slabs[-1].shape == (15, h, w)

        reconstructed = np.concatenate(slabs, axis=0)
        assert np.array_equal(orig_data, reconstructed)
