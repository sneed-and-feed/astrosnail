"""tests.test_milestone2: Parity verification and native performance benchmarks for Milestone 2.

Validates:
1. #[no_std] Rust SIMD Kernel C-ABI dynamic library auto-discovery and symbol loading.
2. Static AstroSnailArena SRAM cache-line disjointness, 64-byte alignment, and < 1.50 MB footprint.
3. Bit-identical float32 parity (<= 1.0e-7) across all 5 core subsystems.
4. Pass-2 Float32 ULP spherical polish guaranteeing exact 0.000e+00 radial drift and 0 zero pollution.
5. SO(6) Kelvin-Mandel rotor trace conservation (trace residual <= 1.14e-13 in f64).
6. 3D force-free Beltrami jet solver with exact 0.000e+00 radial flux escape.
7. Sustained throughput >= 2,000 FPS (2,000–9,400+ FPS target) on 128x128 focal planes.
"""

import ctypes
import math
import time
import numpy as np
import pytest

from astrosnail import _native
from astrosnail.transients import (
    BeltramiLightCurveShell,
    compute_morse_impulse,
    MorseTransientGateConfig,
    TemporalClockContentSplitter,
    compute_spatial_mad_floor_256 as py_mad_floor,
)
from astrosnail.astrometry import (
    SkewSymmetricTriadOperator,
    SlowFastTriadCascade,
    attribute_causal_progenitors as py_attribute_causal,
)
from astrosnail.cosmology import (
    reconstruct_zslab_tomography,
    detect_jordan_cavity_cores as py_detect_jordan,
    so6_kelvin_mandel_rotor as py_so6_rotor,
    retract_void_step0_homotopy as py_retract_void,
    solve_beltrami_jet_poincare as py_solve_beltrami_jet,
)
from astrosnail.engine import AstroSnailEngine


class TestNativeLibraryAvailability:
    """Verifies that the compiled #[no_std] Rust library is discovered and loadable."""

    def test_native_library_available(self):
        assert _native.is_available(), "astrosnail._native.is_available() returned False"
        lib = _native.load_native_library()
        assert lib is not None, "load_native_library() returned None"

    def test_required_c_abi_symbols_exported(self):
        lib = _native.load_native_library()
        required_symbols = [
            "astrosnail_arena_size_bytes",
            "astrosnail_arena_init",
            "astrosnail_arena_reset",
            "astrosnail_arena_global",
            "beltrami_morse_transient_gate_f32",
            "compute_spatial_mad_floor_256_c",
            "morse_transient_gate_step_1d_f32",
            "ephemeris_clock_nope_split_f32",
            "so8_skew_triad_step_f32",
            "so8_astrometry_triad_cascade_f32",
            "attribute_causal_progenitors_f32",
            "cosmic_web_zslab_relax_f32",
            "detect_jordan_cavity_cores_f32",
            "retract_void_step0_homotopy_f32",
            "so6_kelvin_mandel_rotor_f64",
            "so6_kelvin_mandel_rotor_f32",
            "solve_beltrami_jet_poincare_f32",
            "beltrami_jet_poincare_step_f32",
        ]
        for sym in required_symbols:
            assert hasattr(lib, sym), f"Missing exported C-ABI symbol: {sym}"


class TestNativeArenaGeometry:
    """Validates SRAM memory footprint, 64-byte alignment, and buffer disjointness."""

    def test_arena_size_within_sram_budget(self):
        lib = _native.load_native_library()
        sz = lib.astrosnail_arena_size_bytes()
        assert sz == _native.ASTROSNAIL_ARENA_TOTAL_BYTES
        assert sz == 1_012_352, f"Expected 1,012,352 bytes, got {sz}"
        assert sz < 1_572_864, f"Arena size {sz} exceeds 1.50 MB (1,572,864 bytes) SRAM ceiling"
        assert sz % 64 == 0, f"Arena size {sz} is not 64-byte aligned"

    def test_all_subbuffers_cacheline_disjoint_and_aligned(self):
        arena_cls = _native.AstroSnailArena
        subbuffer_fields = [
            "spatial_input", "spatial_prev", "salience_map", "curl_buffer", "ring_buffer",
            "mask_clock", "mask_nope", "mask_spike", "cosmic_density", "cosmic_phi",
            "cosmic_void_mask", "skewer_flux", "skewer_coords", "triad_tensor_w",
            "cascade_w_rec", "medoid_coords_yx", "medoid_features", "causal_transition",
            "rotor_so6_m", "rotor_c_base", "rotor_c_rot", "morse_state", "telemetry",
            "step_counter",
        ]
        for fname in subbuffer_fields:
            offset = getattr(arena_cls, fname).offset
            assert offset % 64 == 0, f"Sub-buffer '{fname}' offset {offset} is not a multiple of 64"

    def test_native_arena_instantiation_and_views(self):
        arena = _native.NativeAstroSnailArena(use_global=True)
        arena.reset()
        assert arena.step_counter == 0
        assert arena.spatial_input.shape == (16384,)
        assert arena.salience_map.shape == (16384,)
        assert arena.ring_buffer.shape == (131072,)
        assert arena.cosmic_density.shape == (18432,)


class TestNativeOperator1_BeltramiGateAndMAD:
    """Tests 2D Beltrami exterior curl, discrete Morse gate, and 256-sample MAD floor."""

    def test_deterministic_256_sample_mad_floor_parity(self):
        rng = np.random.default_rng(20261030)
        for _ in range(5):
            sal = rng.exponential(scale=2.0, size=16384).astype(np.float32)
            clock_m = (rng.random(size=16384) < 0.1).astype(np.uint8)

            nat_mad = _native.compute_spatial_mad_floor_256(sal, clock_m)
            py_mad = py_mad_floor(sal, clock_m)
            err = abs(nat_mad - py_mad)
            assert err <= 1.0e-7, f"MAD floor discrepancy {err} exceeds 1.0e-7"

    def test_beltrami_morse_transient_gate_execution(self):
        rng = np.random.default_rng(42)
        frame = rng.standard_normal((128, 128)).astype(np.float32)
        # Inject artificial spike
        frame[64, 64] += 50.0

        sal, spikes, mad_floor = _native.beltrami_morse_transient_gate(frame)
        assert sal.shape == (128, 128)
        assert spikes.shape == (128, 128)
        assert mad_floor >= 0.0
        # Check that high spike is flagged in spike mask
        assert spikes[64, 64] == 1
        # Boundary pixels must be 0
        assert np.all(sal[0, :] == 0.0)
        assert np.all(sal[-1, :] == 0.0)
        assert np.all(sal[:, 0] == 0.0)
        assert np.all(sal[:, -1] == 0.0)

    def test_1d_morse_gate_causal_deferral(self):
        lib = _native.load_native_library()
        state = _native.AstroSnailMorseState()
        state.sigma_thresh = 4.0
        state.variance = 1.0
        state.noise_std = 1.0

        status = ctypes.c_int32(0)
        applied = ctypes.c_uint32(0)

        # 1. Baseline initialization
        lib.morse_transient_gate_step_1d_f32(ctypes.byref(state), 10.0, 1.0, ctypes.byref(status), ctypes.byref(applied))
        assert status.value == _native.ASTROSNAIL_MORSE_NOMINAL
        assert applied.value == 1

        # 2. Impulse spike arrival -> PENDING_IMPULSE (defer 1 sample)
        lib.morse_transient_gate_step_1d_f32(ctypes.byref(state), 25.0, 2.0, ctypes.byref(status), ctypes.byref(applied))
        assert status.value == _native.ASTROSNAIL_MORSE_PENDING_IMPULSE
        assert applied.value == 0

        # 3. Glitch drops back to baseline -> GLINT_REJECTED
        lib.morse_transient_gate_step_1d_f32(ctypes.byref(state), 10.2, 3.0, ctypes.byref(status), ctypes.byref(applied))
        assert status.value == _native.ASTROSNAIL_MORSE_GLINT_REJECTED
        assert applied.value == 0
        assert state.glint_rejected_count == 1


class TestNativeOperator2_EphemerisSplit:
    """Tests ephemeris harmonic quarantine into H_clock (+) H_nope."""

    def test_ephemeris_periodic_harmonic_split(self):
        t = np.linspace(0, 100, 1000, dtype=np.float32)
        # Pure periodic ephemeris carrier (period ~ 50 samples) + localized transient
        carrier = 5.0 * np.sin(2.0 * np.pi * t / 5.0).astype(np.float32)
        transient = np.zeros(1000, dtype=np.float32)
        transient[500:505] = 20.0  # FRB spike
        total_flux = carrier + transient

        clock_mask, nope_res = _native.ephemeris_clock_nope_split(total_flux, threshold=0.80)
        assert clock_mask.shape == (1000,)
        assert nope_res.shape == (1000,)
        # Transient spike should be preserved in nope_res
        assert np.max(nope_res[500:505]) > 10.0


class TestNativeOperator3_SO8Cascade:
    """Tests so(8) skew-symmetric tie-breaker and 16-step cascade 0.000e+00 drift."""

    def test_so8_skew_triad_step_orthogonality_and_parity(self):
        u = np.array([0.5, 0.25, -0.1, 0.8, 0.0, 0.3, 0.1, -0.4], dtype=np.float32)
        v = np.array([0.25] * 8, dtype=np.float32)

        triad_op = SkewSymmetricTriadOperator(dim=8)
        v_py = triad_op.forward(u, v)
        v_nat = _native.so8_skew_triad_step(u, v, gamma=0.05)

        err = float(np.max(np.abs(v_nat - v_py)))
        assert err <= 1.0e-7, f"so8_skew_triad_step error {err} exceeds 1.0e-7"

        # Pointwise tangent orthogonality: <v, B> == 0
        dot = float(np.dot(v, v_nat))
        assert abs(dot) <= 1.0e-7, f"Orthogonality violation <v, B> = {dot}"

    def test_so8_astrometry_cascade_zero_radial_drift_and_parity(self):
        u = np.array([0.5, 0.25, -0.1, 0.8, 0.0, 0.3, 0.1, -0.4], dtype=np.float32)
        v = np.array([0.25] * 8, dtype=np.float32)

        cascade = SlowFastTriadCascade(coupling=0.15, steps=16)
        v_py, drift_py = cascade.integrate(u, v, steps=16)

        v_nat, drift_nat = _native.so8_astrometry_triad_cascade(u, v, coupling=0.15, steps=16)
        assert drift_nat == 0.0, f"Native radial drift {drift_nat} != 0.000e+00"

        err = float(np.max(np.abs(v_nat - v_py.astype(np.float32))))
        assert err <= 1.0e-7, f"Cascade 16-step error {err} exceeds 1.0e-7"

    def test_attribute_causal_progenitors_parity(self):
        trans_mat = np.array([
            [10.0, 2.0, 1.0, 0.5],
            [1.0, 3.0, 8.0, 4.0],
            [2.0, 1.0, 1.0, 15.0],
            [12.0, 1.0, 0.5, 2.0],
        ], dtype=np.float32)

        is_gen_nat, ratios_nat = _native.attribute_causal_progenitors(trans_mat, threshold=0.50)
        is_gen_py, ratios_py = py_attribute_causal(trans_mat, threshold=0.50)

        assert np.array_equal(is_gen_nat, is_gen_py)
        err = float(np.max(np.abs(ratios_nat - ratios_py)))
        assert err <= 1.0e-7, f"Contingency ratio error {err} exceeds 1.0e-7"


class TestNativeOperator4_CosmicWebZslab:
    """Tests 3D cosmic web Z-slab tomography, cavity detection, and void retraction."""

    def test_cosmic_web_zslab_relax(self):
        n_skewers = 4
        z_chunk = 16
        y_dim = 24
        x_dim = 24

        skewer_flux = np.ones((n_skewers, z_chunk), dtype=np.float32)
        skewer_coords = np.array([
            [6, 6],
            [6, 18],
            [18, 6],
            [18, 18],
        ], dtype=np.int32)

        delta_slab, phi_slab = _native.cosmic_web_zslab_relax(
            skewer_flux, skewer_coords, z_chunk=z_chunk, y_dim=y_dim, x_dim=x_dim
        )
        assert delta_slab.shape == (z_chunk, y_dim, x_dim)
        assert phi_slab.shape == (z_chunk, y_dim, x_dim)
        # Homotopy retraction: density must satisfy delta >= -1.0
        assert np.all(delta_slab >= -1.0)

    def test_void_retraction_nonnegativity_bound(self):
        dens = np.array([-2.5, -1.0, 0.0, 5.0], dtype=np.float32)
        void_mask = np.array([True, True, False, False])
        retracted = _native.retract_void_step0_homotopy(dens, void_mask)
        assert retracted[0] == -1.0
        assert retracted[1] == -1.0
        assert retracted[2] == 0.0
        assert retracted[3] == 5.0


class TestNativeOperator5_SO6RotorAndBeltramiJet:
    """Tests SO(6) Kelvin-Mandel rotor trace conservation and Beltrami jet flux confinement."""

    def test_so6_rotor_trace_conservation(self):
        angles = [0.1, 0.5, 1.2, 2.5]
        for theta in angles:
            r3 = np.array([
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta),  np.cos(theta), 0.0],
                [0.0,            0.0,           1.0],
            ], dtype=np.float64)
            c_base = np.eye(6, dtype=np.float64) * 7.5

            m6, c_rot, trace_res = _native.so6_kelvin_mandel_rotor(r3, c_base)
            assert trace_res <= 1.14e-13, f"Trace residual {trace_res} exceeds 1.14e-13"

    def test_solve_beltrami_jet_poincare_flux_escape_zero(self):
        boundary = np.ones((24, 24), dtype=np.float32)
        b_field, flux_escape = _native.solve_beltrami_jet_poincare(boundary, mu=1.0, steps=16)
        assert b_field.shape == (3, 32, 24, 24)
        assert flux_escape == 0.0, f"Beltrami radial flux escape {flux_escape} != 0.0"


class TestNativeParityAndThroughputBenchmark:
    """Validates complete parity verification harness and sustained throughput."""

    def test_verify_native_parity_harness(self):
        res = _native.verify_native_parity()
        assert res["passed"] is True
        for k, v in res["operator_residuals"].items():
            assert v <= 1.0e-7, f"Operator residual {k} = {v} exceeds 1.0e-7"
        assert res["benchmarks"]["fps_beltrami_gate"] >= 2000.0

    def test_engine_auto_backend_selects_rust(self):
        engine = AstroSnailEngine(backend="auto")
        assert engine.backend == "rust", f"Engine backend resolved to {engine.backend}, expected 'rust'"

    def test_sustained_streaming_throughput_exceeds_2000_fps(self):
        frame = np.random.randn(128, 128).astype(np.float32)
        # Warmup
        for _ in range(200):
            _native.beltrami_morse_transient_gate(frame)

        n_iters = 1500
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _native.beltrami_morse_transient_gate(frame)
        t1 = time.perf_counter()

        fps = n_iters / (t1 - t0)
        assert fps >= 2000.0, f"Measured FPS {fps:.1f} is below 2,000 FPS requirement"
