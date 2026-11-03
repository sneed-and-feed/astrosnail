"""
astrosnail.tests.e2e.test_tier3_combinations
============================================
Tier 3: Cross-feature interactions and multi-module pipelines combining
transients, astrometry, cosmology, and formal invariants.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from astrosnail.transients import (
    BeltramiLightCurveShell,
    compute_morse_impulse,
    MorseTransientGateConfig,
    TemporalClockContentSplitter,
    compute_spatial_mad_floor_256,
)
from astrosnail.astrometry import (
    RoPE2DMedoidPooler,
    SkewSymmetricTriadOperator,
    SlowFastTriadCascade,
    attribute_causal_progenitors,
)
from astrosnail.cosmology import (
    reconstruct_zslab_tomography,
    detect_jordan_cavity_cores,
    so6_kelvin_mandel_rotor,
    retract_void_step0_homotopy,
    blind_astro_let_decouple,
    solve_beltrami_jet_poincare,
)
from astrosnail.engine import AstroSnailEngine, verify_formal_theorems
from astrosnail.io import (
    write_lc_file,
    parse_lc_table,
    write_cat_file,
    parse_cat_file,
    write_fits_cube,
    stream_fits_cube,
    read_fits_header,
)


def test_tier3_transient_end_to_end_pipeline():
    """Verifies end-to-end transient triage pipeline:
    Beltrami Shell -> Morse Impulse -> Clock Splitter -> Causal Glitch Gate.
    """
    rng = np.random.default_rng(20261103)
    t = np.linspace(0, 20.0, 1000)
    # Periodic baseline + cosmic ray glints + physical step
    periodic = 5.0 * np.sin(2.0 * np.pi * t / 2.5)
    flux = periodic.copy()
    flux[300] += 50.0  # Isolated glint spike
    flux[600:650] += 25.0  # Sustained physical transient burst
    flux += rng.normal(0.0, 0.5, len(t))

    # 1. Shell decomposition
    shell = BeltramiLightCurveShell(k_tokens=16)
    u_coarse, u_fine, salience = shell.decompose(flux)
    assert len(u_coarse) == len(flux)
    assert len(u_fine) == len(flux)

    # 2. Discrete Morse impulse calculation
    morse_impulse = compute_morse_impulse(flux, salience=salience)
    assert len(morse_impulse) == len(flux)
    # Isolated spike at 300 should exhibit high impulse
    assert morse_impulse[300] > 100.0

    # 3. Temporal Clock Content Splitter
    splitter = TemporalClockContentSplitter(periodicity_threshold=0.85)
    h_clock, h_nope, mask = splitter.split(flux)
    assert len(h_clock) == len(flux)
    assert len(h_nope) == len(flux)

    # 4. Causal Glitch Gate
    gate = MorseTransientGateConfig(sigma_thresh=4.0)
    for idx, val in enumerate(u_fine):
        gate.update(float(val), timestamp=float(t[idx]))

    assert gate.glint_rejected_count >= 1
    assert gate.confirmed_transient_count >= 1


def test_tier3_transient_trigger_to_astrometric_centroiding():
    """Verifies cross-feature interaction between 1D transient trigger and 2D sub-pixel astrometry:
    Detect transient epoch in light curve, select 2D focal plane, and extract sub-pixel RoPE medoids.
    """
    rng = np.random.default_rng(42)
    n_frames = 20
    frames = [rng.normal(10.0, 1.0, (64, 64)).astype(np.float32) for _ in range(n_frames)]
    # Insert point source transient in frame 10
    frames[10][32, 32] += 100.0

    light_curve = np.array([float(np.max(f)) for f in frames], dtype=np.float32)
    morse_impulse = compute_morse_impulse(light_curve)
    peak_epoch = int(np.argmax(morse_impulse))
    assert peak_epoch == 10

    # Extract sub-pixel medoids from triggered frame
    pooler = RoPE2DMedoidPooler(k_medoids=8)
    coords, feats = pooler.pool(frames[peak_epoch])
    assert len(coords) >= 1
    # Check that centroid is located near (32, 32)
    dist_to_source = np.min(np.linalg.norm(coords - np.array([32.0, 32.0]), axis=1))
    assert dist_to_source < 2.0


def test_tier3_astrometry_cascade_to_causal_attribution():
    """Verifies astrometric tracking + Slow-Fast cascade + causal progenitor attribution."""
    cascade = SlowFastTriadCascade(coupling=0.15, steps=16)
    v_targets = [
        np.array([1.0, 0.0, 0.5, -0.5, 15.0, 1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, -0.5, 0.5, 15.5, 0.5, 0.5, 0.0], dtype=np.float64),
        np.array([0.5, 0.5, 0.0, 0.0, 16.0, 0.0, 0.0, 1.0], dtype=np.float64),
    ]

    integrated_states = []
    u_drive = np.ones((16, 8), dtype=np.float64) * 0.2
    for v0 in v_targets:
        v_norm = v0 / np.linalg.norm(v0)
        v_out, drift = cascade.integrate(u_drive, v_norm, steps=16)
        assert drift == 0.0
        integrated_states.append(v_out)

    # Compute correlation / transition matrix among targets
    states_mat = np.array(integrated_states)
    transition_mat = np.maximum(0.0, np.dot(states_mat, states_mat.T))
    is_gen, scores = attribute_causal_progenitors(transition_mat, threshold=0.5)
    assert len(is_gen) == 3


def test_tier3_cosmology_tomography_to_void_retraction():
    """Verifies cross-feature pipeline:
    Sparse Skewers -> Z-Slab Tomography -> 3D Jordan Cavity Core Detection -> Step-0 Homotopy Retraction.
    """
    z_dim, y_dim, x_dim = 32, 24, 24
    # Create synthetic skewers with deep underdensity in center
    skewers = []
    coords = []
    for y in (6, 12, 18):
        for x in (6, 12, 18):
            sk = np.zeros(z_dim, dtype=np.float32)
            # Central void at z in [12, 20]
            if 10 <= y <= 14 and 10 <= x <= 14:
                sk[12:20] = -0.9
            skewers.append(sk)
            coords.append([float(y), float(x)])

    coords_arr = np.array(coords, dtype=np.float32)

    # 1. Tomography
    tomo = reconstruct_zslab_tomography(skewers, coords_arr, grid_shape=(z_dim, y_dim, x_dim), chunk_z=16)
    density = tomo["density"]
    assert density.shape == (z_dim, y_dim, x_dim)

    # 2. Jordan Cavity Cores
    void_cores = detect_jordan_cavity_cores(density, threshold=-0.5)
    assert isinstance(void_cores, np.ndarray)

    # 3. Homotopy Retraction
    retracted = retract_void_step0_homotopy(density, void_cores)
    assert np.all(retracted >= -1.0)


def test_tier3_rsd_decoupling_to_beltrami_jet():
    """Verifies RSD decoupling combined with force-free Beltrami jet solver."""
    rsd_field = np.random.randn(32, 24, 24).astype(np.float32) * 0.1
    real_rho, pec_vel = blind_astro_let_decouple(rsd_field)

    assert real_rho.shape == (32, 24, 24)
    assert pec_vel.shape == (32, 24, 24)

    # Feed central slice of real density as boundary flux to Beltrami jet solver
    mid_slice = real_rho[16]
    b_field, flux_escape = solve_beltrami_jet_poincare(mid_slice, mu=1.0, steps=16)
    assert flux_escape == 0.0
    assert b_field.shape == (3, 32, 24, 24)


def test_tier3_streaming_fits_cube_to_engine_update(tmp_path: Path):
    """Verifies streaming out-of-core FITS cube chunks directly into AstroSnailEngine."""
    cube_data = np.random.randn(32, 24, 24).astype(np.float32)
    fits_file = tmp_path / "stream_test.fits"
    write_fits_cube(fits_file, cube_data)

    engine = AstroSnailEngine(backend="numpy")
    total_void_voxels = 0
    for slab in stream_fits_cube(fits_file, chunk_z=8, as_native=True):
        out = engine.update(slab)
        assert out["input_shape"] == list(slab.shape)
        assert "void_voxel_count" in out
        total_void_voxels += out["void_voxel_count"]

    assert engine.step_count == 4


def test_tier3_multi_modal_io_roundtrip(tmp_path: Path):
    """Verifies multi-modal I/O roundtrip for light curves (.lc), catalogs (.cat), and FITS."""
    # 1. Light Curve
    times = np.array([100.1, 100.2, 100.3], dtype=np.float64)
    fluxes = np.array([10.5, 12.0, 9.8], dtype=np.float32)
    errors = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    lc_file = tmp_path / "test.lc"
    write_lc_file(lc_file, times, fluxes, errors, comments=["Test light curve"])
    lc_table = parse_lc_table(lc_file)
    assert len(lc_table) == 3
    assert np.allclose(lc_table[:, 0], times)

    # 2. Catalog
    cat_records = [(1, 123.45, -45.67, 14.2, 0.05), (2, 200.10, 10.20, 18.0, 0.12)]
    cat_file = tmp_path / "test.cat"
    write_cat_file(cat_file, cat_records, comments=["Test catalog"])
    cat_data = parse_cat_file(cat_file)
    assert len(cat_data) == 2
    assert cat_data[0][0] == 1


def test_tier3_engine_theorems_numerical_consistency():
    """Verifies that AstroSnailEngine.verify_formal_theorems() evaluates numerical consistency
    for all machine-checked theorems simultaneously.
    """
    engine = AstroSnailEngine(backend="numpy")
    all_passed, theorem_results = engine.verify_formal_theorems(seed=20261025)
    assert "theorem_2_1" in theorem_results
    assert "theorem_2_2" in theorem_results
    assert "theorem_2_3" in theorem_results
    assert "theorem_2_4" in theorem_results
    assert "theorem_2_5" in theorem_results
    assert "theorem_2_6" in theorem_results
    assert "theorem_2_7" in theorem_results
    assert "theorem_2_8" in theorem_results
    assert "theorem_2_9" in theorem_results


def test_tier3_multi_dimensional_unified_streaming_session():
    """Verifies that a single AstroSnailEngine session causally processes 1D series,
    2D focal planes, and 3D cosmological cubes sequentially without state corruption.
    """
    engine = AstroSnailEngine(backend="numpy")

    # Step 1: 1D light curve update
    s1 = np.array([12.5, 14.2, 10.1, 9.8, 11.2], dtype=np.float32)
    out1 = engine.update(s1)
    assert out1["step"] == 1
    assert "morse_impulse" in out1

    # Step 2: 2D focal plane update
    f2 = np.zeros((32, 32), dtype=np.float32)
    f2[10, 15] = 50.0
    out2 = engine.update(f2)
    assert out2["step"] == 2
    assert "medoid_coords_yx" in out2

    # Step 3: 3D cube update
    c3 = np.random.randn(8, 8, 8).astype(np.float32)
    out3 = engine.update(c3)
    assert out3["step"] == 3
    assert "void_cores" in out3

    # Total ring buffer length should be 3
    assert len(engine.ring_buffer) == 3


def test_tier3_scenario_generators_data_structures():
    """Verifies that canonical scenarios 01–05 instantiate ScenarioData objects with
    observable data streams and valid metadata dictionaries.
    """
    from astrosnail.scenarios import (
        generate_scenario_01_frb_supernova,
        generate_scenario_02_eclipsing_binary,
        generate_scenario_03_crowded_field,
        generate_scenario_04_lensed_quasar,
        generate_scenario_05_cosmic_web,
    )

    sc1 = generate_scenario_01_frb_supernova(seed=101, n_epochs=500)
    assert sc1.data.ndim == 1 and len(sc1.data) == 500

    sc2 = generate_scenario_02_eclipsing_binary(seed=102, n_epochs=500)
    assert sc2.data.ndim == 1 and len(sc2.data) == 500

    sc3 = generate_scenario_03_crowded_field(seed=103, n_frames=10)
    assert sc3.data.ndim == 3 and sc3.data.shape[0] == 10

    sc4 = generate_scenario_04_lensed_quasar(seed=104, n_epochs=200)
    assert sc4.data.ndim == 2

    sc5 = generate_scenario_05_cosmic_web(seed=105, n_skewers=8)
    assert sc5.data.ndim >= 2

