"""
tests/test_adversarial_tier5_coverage.py
========================================
Tier 5 Adversarial Edge Case & White-Box Coverage Test Suite for AstroSnail.
Covers boundary conditions, error paths, and CLI permutations across all modules.
"""

import io
from pathlib import Path
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
    so6_kelvin_mandel_rotor,
    solve_beltrami_jet_poincare,
    compute_mrxmhd_flow_jump,
    SkewSymmetricFaultTriad,
    stream_fits_cube,
    write_fits_cube,
    read_fits_header,
    stream_lc_records,
    parse_lc_table,
    profile_stream_memory,
)
from astrosnail.cli import main as cli_main
from astrosnail.studio import verify_studio_html, serve_studio, main as studio_main


def test_beltrami_light_curve_boundary_validation():
    with pytest.raises(ValueError, match="k_tokens must be >= 2"):
        BeltramiLightCurveShell(k_tokens=1)

    shell = BeltramiLightCurveShell(k_tokens=16)
    with pytest.raises(ValueError, match="must be >= 3"):
        shell.decompose(np.array([1.0, 2.0]))

    # Quiescent & zero signal
    c, f, s = shell.decompose(np.zeros(20, dtype=np.float32))
    assert np.all(s == 0.0)
    assert np.all(c == 0.0)

    # Multi-band zero signal
    c_mb, f_mb, s_mb = shell.decompose(np.zeros((20, 4), dtype=np.float32))
    assert np.all(s_mb == 0.0)

    # NaN / Inf non-finite resilience
    with_nan = np.array([1.0, np.nan, np.inf, 2.0, -np.inf, 3.0], dtype=np.float32)
    c_nan, f_nan, s_nan = shell.decompose(with_nan)
    assert np.all(np.isfinite(c_nan)) and np.all(np.isfinite(s_nan))


def test_morse_gate_and_content_splitter_edge_cases():
    assert np.all(compute_morse_impulse(np.array([1.0, 2.0])) == 0.0)

    gate = MorseTransientGateConfig()
    cls_nan, applied = gate.update(np.nan)
    assert cls_nan == "nominal" and not applied

    gate.update(0.0)
    cls_p, _ = gate.update(100.0)
    assert cls_p == "pending_impulse"
    cls_opp, _ = gate.update(-100.0)
    assert cls_opp == "pending_impulse"

    gate.reset()
    gate.update(0.0)
    gate.update(10.0)
    cls_esc, _ = gate.update(25.0)
    assert cls_esc == "pending_impulse"

    splitter = TemporalClockContentSplitter()
    h_c, h_n, m = splitter.split(np.array([1.0, 2.0]))
    assert np.all(h_c == 0.0) and np.all(m == 0)

    h_c_c, _, _ = splitter.split(np.full(50, 5.0))
    assert np.all(h_c_c == 0.0)

    flicker = np.array([10.0, -10.0] * 30, dtype=np.float32)
    _, _, m_flicker = splitter.split(flicker)
    assert np.any(m_flicker == 1)

    assert compute_spatial_mad_floor_256(np.array([])) == 0.0


def test_astrometry_pooler_and_attribution_edge_cases():
    pooler = RoPE2DMedoidPooler(k_medoids=8)
    c_tiny, f_tiny = pooler.pool(np.zeros((2, 2)))
    assert len(c_tiny) == 0 and len(f_tiny) == 0

    c_blank, f_blank = pooler.pool(np.full((32, 32), 4.5))
    assert len(c_blank) == 0 and len(f_blank) == 0

    cascade = SlowFastTriadCascade()
    v_out, drift = cascade.integrate(np.ones((4, 8)), np.zeros(8))
    assert drift == 0.0 and np.all(v_out == 0.0)

    count_table = np.array([[10.0, 1.0], [2.0, 8.0], [5.0, 5.0]])
    is_gen, ratios = attribute_causal_progenitors(count_table, threshold=0.5)
    assert not is_gen[0] and is_gen[1] and is_gen[2]

    is_gen_1d, _ = attribute_causal_progenitors(np.array([0.2, 0.8]), threshold=0.5)
    assert not is_gen_1d[0] and is_gen_1d[1]

    with pytest.raises(ValueError, match="Invalid transition_matrix shape"):
        attribute_causal_progenitors(np.zeros((2, 2, 2)))


def test_cosmology_tomography_and_rotor_edge_cases():
    res_empty = reconstruct_zslab_tomography([], np.zeros((0, 2)), grid_shape=(32, 24, 24))
    assert np.all(res_empty["density"] == 0.0)

    res_mismatch = reconstruct_zslab_tomography([np.zeros(32)], np.zeros((2, 2)), grid_shape=(32, 24, 24))
    assert np.all(res_mismatch["density"] == 0.0)

    short_skewer = [np.ones(16, dtype=np.float32)]
    res_short = reconstruct_zslab_tomography(short_skewer, np.array([[12, 12]]), grid_shape=(32, 24, 24))
    assert res_short["density"].shape == (32, 24, 24)

    cores = detect_jordan_cavity_cores(np.zeros((32, 24, 24)), threshold=-0.7)
    assert np.sum(cores) == 0

    r3 = np.eye(3)
    _, tr3 = so6_kelvin_mandel_rotor(r3, np.eye(3) * 5.0)
    assert tr3 <= 1.14e-13
    _, tr6 = so6_kelvin_mandel_rotor(r3, np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]))
    assert tr6 <= 1.14e-13

    with pytest.raises(ValueError, match="Invalid shear shape"):
        so6_kelvin_mandel_rotor(r3, np.zeros((4, 4)))

    b_zero, esc_zero = solve_beltrami_jet_poincare(np.zeros((16, 16)), mu=0.0, steps=8)
    assert esc_zero == 0.0

    _, _, h_res = compute_mrxmhd_flow_jump(p1=10.0, b1_sq=2.0, m1=0.2, b2_sq=1.0, m2=0.4)
    assert h_res < 1e-14

    fault = SkewSymmetricFaultTriad(dim=8)
    b_fault = fault.forward(np.ones(8), np.ones(8))
    assert b_fault.shape == (8,)


def test_engine_edge_cases():
    with pytest.raises(ValueError, match="Unknown backend"):
        AstroSnailEngine(backend="non_existent")

    eng = AstroSnailEngine(backend="numpy")
    out_1d = eng.update(np.array([4.2]))
    assert "classification" in out_1d

    for _ in range(12):
        eng.update(np.zeros((16, 16)))
    assert len(eng.ring_buffer) == 8

    out_telem = eng.update(np.zeros((16, 16)), telemetry={"custom_flag": True})
    assert out_telem.get("custom_flag") is True


def test_io_streaming_and_dtypes(tmp_path):
    with pytest.raises(ValueError, match="requires 2D or 3D"):
        write_fits_cube(tmp_path / "1d.fits", np.zeros(10))

    for dt in [np.float64, np.int16, np.int32, np.int64, np.uint8, np.uint16]:
        p = tmp_path / f"test_{dt.__name__}.fits"
        n_bytes = write_fits_cube(p, np.ones((4, 4), dtype=dt), header_extra={"TESTKEY": ("VAL", "Comment")})
        assert n_bytes % 2880 == 0
        hdr = read_fits_header(p)
        assert hdr["TESTKEY"] == "VAL"

    bio = io.BytesIO()
    write_fits_cube(bio, np.ones((16, 8, 8), dtype=np.float32))
    bio.seek(0)
    slabs = list(stream_fits_cube(bio, chunk_z=4))
    assert len(slabs) == 4 and slabs[0].shape == (4, 8, 8)

    p_cube = tmp_path / "cube.fits"
    write_fits_cube(p_cube, np.ones((8, 12, 12), dtype=np.float32))
    squeezed = list(stream_fits_cube(p_cube, chunk_z=1, squeeze_2d=True))
    assert squeezed[0].shape == (12, 12)

    reused = list(stream_fits_cube(p_cube, chunk_z=2, reuse_buffer=True))
    assert len(reused) == 4

    empty_lc = tmp_path / "empty.lc"
    empty_lc.write_text("# empty\n", encoding="utf-8")
    table = parse_lc_table(empty_lc)
    assert table.shape == (0, 3)

    two_col = tmp_path / "two_col.lc"
    two_col.write_text("1.0 10.5\n2.0 11.2\n", encoding="utf-8")
    recs = list(stream_lc_records(two_col))
    assert len(recs) == 2 and recs[0][2] == 0.0

    cnt, peak = profile_stream_memory(iter([np.zeros(100) for _ in range(5)]))
    assert cnt == 5 and peak >= 0.0


def test_cli_permutations(tmp_path):
    assert cli_main(["run"]) == 2
    assert cli_main(["run", "non_existent_file_xyz.fits"]) == 3

    sc_out = tmp_path / "sc_run"
    assert cli_main(["run", "01", "--output-dir", str(sc_out), "--format", "csv", "--fps-limit", "5000.0", "--profile-memory", "--quiet"]) == 0
    assert (sc_out / "telemetry_summary.csv").exists()

    assert cli_main(["run", "01", "--output-dir", str(sc_out), "--format", "fits", "--quiet"]) == 0
    assert (sc_out / "telemetry_summary.fits").exists()

    assert cli_main(["run", "01", "--output-dir", str(sc_out), "--format", "none", "--quiet"]) == 0

    cat_dir = tmp_path / "cat_test"
    assert cli_main(["run", "--generate-catalogs", "--output-dir", str(cat_dir), "--quiet"]) == 0
    assert (cat_dir / "variable_stars.lc").exists()

    assert cli_main(["verify", "--formal", "--quiet"]) == 0
    assert cli_main(["verify", "--native", "--quiet"]) == 0
    assert cli_main(["verify", "--all", "--json", "--seed", "12345"]) == 0

    bench_dir = tmp_path / "bench_test"
    assert cli_main(["benchmark", "--scenario", "01", "-n", "1", "--output", str(bench_dir), "--quiet"]) == 0


def test_studio_edge_cases(tmp_path):
    with pytest.raises(FileNotFoundError, match="Studio HTML not found"):
        serve_studio(tmp_path / "does_not_exist.html")

    tiny = tmp_path / "tiny.html"
    tiny.write_text("<html>small</html>", encoding="utf-8")
    ok, errors = verify_studio_html(tiny)
    assert not ok and any("below 50,000 bytes" in err for err in errors)

    out_html = tmp_path / "out_studio.html"
    studio_main(["--output", str(out_html), "--no-data", "--downsample", "10", "--verify"])
    assert out_html.exists()
