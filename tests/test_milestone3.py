"""
tests/test_milestone3.py
========================
Comprehensive verification test suite for Milestone 3 (Scenarios, Out-of-Core I/O, CLI & Synthetic Catalogs):
- Scenario generators 01–05, analytical ground truths, and zero scenario-name branching.
- Memory-mapped streaming I/O (< 2.0 MB active heap on 64 MB FITS cube verified with tracemalloc).
- FITS format round-trip numerical parity (0.000e+00 max error) and 2880-byte record alignment.
- Streaming light curve (.lc) and source catalog (.cat) parsers and writers.
- CLI entry points: astrosnail-run, astrosnail-benchmark, astrosnail-verify, astrosnail-catalogs.
"""

import json
import os
from pathlib import Path
import tempfile
import tracemalloc
from typing import Any, Dict, List, Tuple
import numpy as np
import pytest

from astrosnail import (
    ScenarioData,
    get_scenario,
    generate_scenario_01_frb_supernova,
    generate_scenario_02_eclipsing_binary,
    generate_scenario_03_crowded_field,
    generate_scenario_04_lensed_quasar,
    generate_scenario_05_cosmic_web,
    stream_fits_cube,
    write_fits_cube,
    read_fits_header,
    stream_lc_records,
    stream_lc_chunks,
    parse_lc_table,
    write_lc_file,
    stream_cat_records,
    stream_cat_chunks,
    parse_cat_file,
    write_cat_file,
    generate_standard_catalogs,
    AstroSnailEngine,
    MorseTransientGateConfig,
    attribute_causal_progenitors,
    detect_jordan_cavity_cores,
)
from astrosnail.cli import (
    main,
    main_benchmark,
    main_verify,
    main_generate_catalogs,
)


# =============================================================================
# 1. SCENARIO GENERATOR & ANALYTICAL GROUND TRUTH TESTS
# =============================================================================

class TestAstrophysicsScenarios:
    """Verifies all 5 canonical astrophysics benchmark scenarios."""

    def test_all_five_scenarios_instantiate_and_deterministic(self):
        """Validates that scenarios 01–05 instantiate and produce bit-identical data for identical seed."""
        for sc_id in ["01", "02", "03", "04", "05"]:
            s_a = get_scenario(sc_id, seed=20261025)
            s_b = get_scenario(sc_id, seed=20261025)
            assert isinstance(s_a, ScenarioData)
            assert s_a.scenario_id.startswith(sc_id)
            assert s_a.data is not None
            assert np.array_equal(s_a.data, s_b.data), f"Scenario {sc_id} not bit-identical across runs!"
            assert isinstance(s_a.to_dict(), dict)

    def test_scenario_alias_resolution(self):
        """Verifies canonical names and colloquial aliases."""
        alias_map = {
            "01": "01_frb_supernova_cosmic_ray_triage",
            "frb": "01_frb_supernova_cosmic_ray_triage",
            "02": "02_eclipsing_binary_exoplanet_transit_split",
            "transit": "02_eclipsing_binary_exoplanet_transit_split",
            "03": "03_crowded_field_kbo_crossing_astrometry",
            "crossing": "03_crowded_field_kbo_crossing_astrometry",
            "04": "04_lensed_quasar_echo_swarm_factorization",
            "quasar": "04_lensed_quasar_echo_swarm_factorization",
            "05": "05_cosmic_web_void_rsd_tomography",
            "void": "05_cosmic_web_void_rsd_tomography",
        }
        for alias, expected in alias_map.items():
            sc = get_scenario(alias)
            assert sc.scenario_id == expected

        with pytest.raises(ValueError, match="Unknown scenario"):
            get_scenario("invalid_scenario_name_999")

    def test_scenario_01_frb_supernova_triage_invariants(self):
        """Validates Scenario 01: FRB, 2 SNe, 80 cosmic ray spikes, 0 false triggers."""
        sc = generate_scenario_01_frb_supernova(seed=20261025, n_epochs=25000)
        gt = sc.ground_truth

        assert len(sc.data) == 25000
        assert gt["frb_epoch"] == 10000
        assert len(gt["supernova_events"]) == 2
        assert len(gt["cosmic_ray_epochs"]) == 80

        # Cosmic rays must be separated by >= 15 epochs
        cr_epochs = gt["cosmic_ray_epochs"]
        diffs = np.diff(cr_epochs)
        assert np.all(diffs >= 15), "Cosmic ray spikes are not sufficiently spaced"

        # Causal deferral simulation on cosmic ray spikes: must yield 0 false alarms
        gate = MorseTransientGateConfig(sigma_thresh=4.0, persistence_window=1)
        cr_set = set(cr_epochs)
        confirmed_cr_spikes = 0

        # Run around a subset of cosmic ray spikes to verify 1-sample deferral and glint rejection
        for ep in cr_epochs[:10]:
            # Step ep - 1: nominal
            gate.update(float(sc.data[ep - 1]), timestamp=float(ep - 1))
            # Step ep: huge spike -> pending_impulse
            cls_spike, update_applied = gate.update(float(sc.data[ep]), timestamp=float(ep))
            assert cls_spike == "pending_impulse"
            assert not update_applied
            # Step ep + 1: return to baseline -> glint_rejected
            cls_next, update_next = gate.update(float(sc.data[ep + 1]), timestamp=float(ep + 1))
            assert cls_next == "glint_rejected"
            assert not update_next

    def test_scenario_02_eclipsing_binary_split_invariants(self):
        """Validates Scenario 02: P1=3.412d EB, RR Lyrae, P2=18.71d transit, satellite burn."""
        sc = generate_scenario_02_eclipsing_binary(seed=20261025, n_epochs=25000)
        gt = sc.ground_truth

        assert len(sc.data) == 25000
        assert gt["p1_eb_period_days"] == 3.412
        assert gt["p2_transit_period_days"] == 18.71
        assert gt["transit_depth_mmag"] == 0.8
        assert len(gt["h_clock_true"]) == 25000
        assert len(gt["h_nope_true"]) == 25000
        assert gt["satellite_burn_epoch"] > 0

    def test_scenario_03_crowded_field_kbo_astrometry_invariants(self):
        """Validates Scenario 03: 240 frames (128x128), 500 stars, 4 KBOs with 90° and 180° crossings."""
        sc = generate_scenario_03_crowded_field(seed=20261025, n_frames=240, height=128, width=128)
        gt = sc.ground_truth

        assert sc.data.shape == (240, 128, 128)
        assert sc.data.dtype == np.float32
        assert gt["crossing_frame"] == 120
        assert gt["crossing_center"] == [64.0, 64.0]
        assert gt["expected_id_swaps"] == 0

        trajs = gt["gt_trajectories"]
        assert trajs.shape == (4, 240, 2)
        # At frame 120, all 4 bodies are exactly at (64.0, 64.0)
        for b_idx in range(4):
            assert trajs[b_idx, 120, 0] == pytest.approx(64.0, abs=1e-5)
            assert trajs[b_idx, 120, 1] == pytest.approx(64.0, abs=1e-5)

    def test_scenario_04_lensed_quasar_causal_attribution(self):
        """Validates Scenario 04: 2 active generators (G_act), 10 passive registers (Omega_reg)."""
        sc = generate_scenario_04_lensed_quasar(seed=20261025, n_epochs=2500)
        gt = sc.ground_truth

        assert sc.data.shape == (2500, 12)
        assert gt["num_progenitors"] == 2
        assert gt["num_echoes"] == 10
        assert len(gt["is_generator_true"]) == 12

        # Verify causal attribution splitter
        t_mat = gt["transition_matrix"]
        is_gen, ratios = attribute_causal_progenitors(t_mat, threshold=0.50)
        assert np.array_equal(is_gen, gt["is_generator_true"])
        assert ratios[0] >= 0.50
        assert ratios[1] >= 0.50
        assert np.all(ratios[2:] < 0.50)

    def test_scenario_05_cosmic_web_void_rsd_invariants(self):
        """Validates Scenario 05: (32, 24, 24) volume, sheets, knot, Jordan void, Beltrami jet."""
        sc = generate_scenario_05_cosmic_web(seed=20261025, grid_shape=(32, 24, 24), n_skewers=8)
        gt = sc.ground_truth

        delta = sc.data
        assert delta.shape == (32, 24, 24)
        assert delta.dtype == np.float32

        # Cluster knot peak overdensity >= +6.0
        assert np.max(delta) >= 6.0

        # Jordan void core minimum delta <= -0.80
        assert gt["void_min_delta"] <= -0.80

        # Jordan cavity core detection
        void_cores = detect_jordan_cavity_cores(delta)
        assert np.sum(void_cores) > 0, "Topological cavity core not detected"

        # Beltrami jet field shape
        b_field = gt["beltrami_b_field"]
        assert b_field.shape == (3, 32, 24, 24)

        # 8 skewers
        assert len(gt["skewers"]) == 8
        assert gt["skewer_coords"].shape == (8, 2)

    def test_zero_scenario_name_branching_polymorphism(self):
        """Crucial invariant: AstroSnailEngine processes data purely via ndim without scenario branching."""
        engine = AstroSnailEngine(backend="numpy")

        # 1D series (Scenario 01 & 02)
        s1 = get_scenario("01")
        out1 = engine.update(s1.data)
        assert "curl_salience" in out1 or "u_coarse" in out1

        s2 = get_scenario("02")
        out2 = engine.update(s2.data)
        assert "h_clock" in out2

        # 2D focal plane (Scenario 03)
        s3 = get_scenario("03")
        frame0 = s3.data[0]
        out3 = engine.update(frame0)
        assert "num_medoids" in out3
        assert "primary_cascade_drift" in out3

        # 3D cosmological cube (Scenario 05)
        s5 = get_scenario("05")
        out5 = engine.update(s5.data)
        assert "void_cores" in out5
        assert "retracted_density" in out5


# =============================================================================
# 2. OUT-OF-CORE STREAMING I/O TESTS
# =============================================================================

class TestOutOfCoreStreamingIO:
    """Verifies memory-mapped FITS streaming, format parity, and ASCII table parsers."""

    def test_stream_fits_cube_memory_ceiling_64mb(self, tmp_path):
        """Proves that streaming a 64 MB FITS cube maintains peak active heap strictly < 2.0 MB."""
        fits_file = tmp_path / "survey_cube_64mb.fits"
        # 1024 * 128 * 128 * 4 bytes = 67,108,864 bytes = 64.0 MB
        cube_shape = (1024, 128, 128)
        cube_data = np.zeros(cube_shape, dtype=np.float32)

        written_bytes = write_fits_cube(fits_file, cube_data, chunk_z=16)
        assert written_bytes % 2880 == 0
        del cube_data  # Free generation array

        tracemalloc.start()
        tracemalloc.reset_peak()

        chunk_count = 0
        for slab in stream_fits_cube(fits_file, chunk_z=16, copy=False):
            chunk_count += 1
            assert slab.shape == (16, 128, 128)
            # Lightweight operation without allocating new large arrays
            _ = float(np.min(slab))

        cur_bytes, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_bytes / (1024.0 * 1024.0)
        assert chunk_count == 64, f"Expected 64 chunks of 16-slabs, got {chunk_count}"
        assert peak_mb < 2.0, f"Memory ceiling breached: {peak_mb:.4f} MB >= 2.0 MB"

    def test_fits_roundtrip_parity_and_2880_byte_alignment(self, tmp_path):
        """Validates exact numerical parity (0.000e+00 error) and 2880-byte record alignment."""
        fits_file = tmp_path / "roundtrip_test.fits"
        orig_data = np.arange(32 * 24 * 24, dtype=np.float32).reshape(32, 24, 24)

        written_bytes = write_fits_cube(
            fits_file,
            orig_data,
            header_extra={"OBJECT": "Target-Alpha", "CREATOR": "AstroSnail-v0.1.0"},
            chunk_z=8,
        )

        assert fits_file.stat().st_size % 2880 == 0, "FITS file size is not a multiple of 2880 bytes"
        assert written_bytes == fits_file.stat().st_size

        # Read back via streaming generator
        slabs = list(stream_fits_cube(fits_file, chunk_z=8, as_native=True))
        rec_data = np.concatenate(slabs, axis=0)

        assert rec_data.shape == orig_data.shape
        max_abs_err = float(np.max(np.abs(rec_data - orig_data)))
        assert max_abs_err == 0.0, f"Roundtrip numerical corruption: max error = {max_abs_err}"

    def test_fits_header_parsing(self, tmp_path):
        """Validates parsing of BITPIX, NAXIS1..3, BSCALE, BZERO, string cards, and comments."""
        fits_file = tmp_path / "header_test.fits"
        data = np.ones((16, 32, 32), dtype=np.float32) * 5.0
        write_fits_cube(fits_file, data, header_extra={"INSTRUME": "DarkSnail-Camera", "EXPTIME": 30.0})

        hdr = read_fits_header(fits_file)
        assert hdr.shape == (16, 32, 32)
        assert hdr.bitpix == -32
        assert hdr.bscale == 1.0
        assert hdr.bzero == 0.0
        assert hdr.get("INSTRUME") == "DarkSnail-Camera"
        assert hdr.get("EXPTIME") == 30.0
        assert hdr.data_offset % 2880 == 0

    def test_streaming_lc_parser_and_writer(self, tmp_path):
        """Validates .lc streaming parser and writer with comment skipping and delimiter flexibility."""
        lc_file = tmp_path / "lightcurve.lc"
        time = np.array([0.0, 1.5, 3.0, 4.5, 6.0])
        flux = np.array([100.2, 105.4, 98.1, 102.3, 101.0])
        err = np.array([0.5, 0.4, 0.6, 0.5, 0.5])

        write_lc_file(lc_file, time, flux, err, comments=["Test survey light curve", "Instrument: LSST"])

        # Stream records
        recs = list(stream_lc_records(lc_file))
        assert len(recs) == 5
        assert recs[0] == pytest.approx((0.0, 100.2, 0.5))
        assert recs[2] == pytest.approx((3.0, 98.1, 0.6))

        # Parse full table
        table = parse_lc_table(lc_file)
        assert table.shape == (5, 3)
        assert np.allclose(table[:, 1], flux)

    def test_streaming_cat_parser_and_writer(self, tmp_path):
        """Validates .cat streaming parser and writer for astrometric source catalogs."""
        cat_file = tmp_path / "sources.cat"
        records = [
            (1001, 180.2541, 15.3421, 18.52, 0.03),
            (1002, 180.2614, 15.3409, 19.14, 0.05),
            ("SRC_A", 180.2589, 15.3487, 20.02, 0.08),
        ]
        write_cat_file(cat_file, records, comments=["Astrometric point sources", "Epoch: J2000"])

        recs = list(stream_cat_records(cat_file))
        assert len(recs) == 3
        assert recs[0][0] == 1001
        assert recs[2][0] == "SRC_A"
        assert recs[1][3] == pytest.approx(19.14)

        cat_list = parse_cat_file(cat_file)
        assert len(cat_list) == 3

    def test_generate_standard_catalogs(self, tmp_path):
        """Validates synthetic catalog generator and loadability."""
        cat_dir = tmp_path / "catalogs"
        catalog_paths = generate_standard_catalogs(cat_dir)

        assert (cat_dir / "kbo_crossing.cat").exists()
        assert (cat_dir / "variable_stars.lc").exists()
        assert (cat_dir / "cosmic_web_skewers.fits").exists()

        # Check that generated files parse cleanly
        cat_recs = list(stream_cat_records(catalog_paths["kbo_crossing_cat"]))
        assert len(cat_recs) >= 4

        lc_recs = list(stream_lc_records(catalog_paths["variable_stars_lc"]))
        assert len(lc_recs) == 2500

        fits_slabs = list(stream_fits_cube(catalog_paths["cosmic_web_skewers_fits"], chunk_z=16))
        assert len(fits_slabs) == 2
        assert fits_slabs[0].shape == (16, 24, 24)


# =============================================================================
# 3. CLI ENTRY POINTS & AUTOMATION TESTS
# =============================================================================

class TestCommandLineInterface:
    """Verifies all AstroSnail CLI entry points, execution loops, and exit codes."""

    def test_cli_run_help_and_syntax(self, capsys):
        """Verifies --help on astrosnail-run."""
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "AstroSnail Causal Streaming Outlier Engine" in captured.out

    def test_cli_benchmark_help(self, capsys):
        """Verifies --help on astrosnail-benchmark."""
        with pytest.raises(SystemExit) as exc:
            main_benchmark(["--help"])
        assert exc.value.code == 0

    def test_cli_verify_help(self, capsys):
        """Verifies --help on astrosnail-verify."""
        with pytest.raises(SystemExit) as exc:
            main_verify(["--help"])
        assert exc.value.code == 0

    def test_cli_catalogs_help(self, capsys):
        """Verifies --help on astrosnail-catalogs."""
        with pytest.raises(SystemExit) as exc:
            main_generate_catalogs(["--help"])
        assert exc.value.code == 0

    def test_cli_missing_input_exits_2(self):
        """Calling astrosnail-run without arguments exits with code 2."""
        code = main([])
        assert code == 2

    def test_cli_run_streaming_scenario(self, tmp_path):
        """Executes astrosnail-run streaming Scenario 01."""
        out_dir = tmp_path / "run_sc01"
        code = main(["01", "--output-dir", str(out_dir), "--format", "json", "--quiet"])
        assert code == 0

        summary_file = out_dir / "run_summary.json"
        assert summary_file.exists()
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["status"] == "completed"
        assert summary["total_steps"] == 25000
        assert summary["sustained_fps"] > 0.0

    def test_cli_run_fits_file_streaming(self, tmp_path):
        """Executes astrosnail-run streaming a FITS cube."""
        fits_file = tmp_path / "test_cube.fits"
        cube = np.zeros((32, 24, 24), dtype=np.float32)
        write_fits_cube(fits_file, cube)

        out_dir = tmp_path / "run_fits"
        code = main([str(fits_file), "--output-dir", str(out_dir), "--format", "json", "--chunk-z", "16", "--quiet"])
        assert code == 0

        summary_file = out_dir / "run_summary.json"
        assert summary_file.exists()
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["total_steps"] == 2  # 32 / 16 = 2 chunks

    def test_cli_verify_formal_and_native(self):
        """Verifies astrosnail-verify runtime audit returns exit code 0."""
        code = main_verify(["--all", "--quiet"])
        assert code == 0

    def test_cli_verify_json_output(self, capsys):
        """Verifies astrosnail-verify with --json flag."""
        code = main_verify(["--all", "--json"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["all_passed"] is True
        assert "formal_theorems" in data
        assert "native_parity" in data

    def test_cli_benchmark_execution(self, tmp_path):
        """Verifies astrosnail-benchmark runs trials and writes evaluation summary."""
        out_dir = tmp_path / "benchmark_results"
        code = main_benchmark(["--scenario", "01", "--trials", "2", "--output", str(out_dir), "--quiet"])
        assert code == 0

        summary_file = (
            out_dir / "astrosnail_benchmark_summary.json"
            if (out_dir / "astrosnail_benchmark_summary.json").exists()
            else out_dir / "astrosnail_evaluation_summary.json"
        )
        assert summary_file.exists()
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["all_passed"] is True
        assert "01" in summary["scenarios"]

    def test_cli_generate_catalogs(self, tmp_path):
        """Verifies astrosnail-catalogs creates synthetic reference files."""
        out_dir = tmp_path / "catalogs"
        code = main_generate_catalogs(["--output-dir", str(out_dir), "--quiet"])
        assert code == 0
        assert (out_dir / "kbo_crossing.cat").exists()
        assert (out_dir / "variable_stars.lc").exists()
        assert (out_dir / "cosmic_web_skewers.fits").exists()


# =============================================================================
# 4. REMEDIATION REGRESSION TESTS (AUDIT ITERATION 2)
# =============================================================================

class TestMilestone3Remediations:
    """Verifies surgical remediations for multi-band broadcasting, catalog CLI ingestion,
    top-level subcommand routing, console script packaging, and Windows memmap cleanup."""

    def test_multiband_lightcurve_morse_impulse_broadcasting(self):
        """Verifies compute_morse_impulse correctly handles multi-band light curves (N, 4) with (N,) salience."""
        from astrosnail.transients import BeltramiLightCurveShell, compute_morse_impulse
        shell = BeltramiLightCurveShell(k_tokens=16)

        # 1. Synthetic multi-band light curve (500 epochs, 4 photometric bands: g, r, i, z)
        rng = np.random.RandomState(20261025)
        flux_multiband = rng.uniform(10.0, 50.0, size=(500, 4)).astype(np.float32)
        # Inject transient spike at epoch 250 in bands 0 and 2
        flux_multiband[250, 0] += 30.0
        flux_multiband[250, 2] += 40.0

        u_coarse, u_fine, curl_sal = shell.decompose(flux_multiband)
        assert curl_sal.shape == (500,), f"Expected salience shape (500,), got {curl_sal.shape}"

        # Must compute Morse impulse across all 4 channels without broadcasting error
        morse = compute_morse_impulse(flux_multiband, salience=curl_sal)
        assert morse.shape == (500, 4), f"Expected Morse shape (500, 4), got {morse.shape}"
        assert morse.dtype == np.float32

        # Check positive impulse on peak samples in perturbed bands (Theorem 2.7)
        assert morse[250, 0] > 0.0, "Band 0 transient impulse must be positive"
        assert morse[250, 2] > 0.0, "Band 2 transient impulse must be positive"

        # 2. Scenario 01 multi_band_observed array (25000, 4)
        sc01 = get_scenario("01", seed=20261025)
        multi_band_data = sc01.ground_truth.get("multi_band_observed")
        assert multi_band_data is not None, "Scenario 01 ground truth must contain multi_band_observed"
        assert multi_band_data.shape == (25000, 4)

        u_c, u_f, sal_full = shell.decompose(multi_band_data)
        assert sal_full.shape == (25000,)
        morse_full = compute_morse_impulse(multi_band_data, salience=sal_full)
        assert morse_full.shape == (25000, 4)

        # 3. AstroSnailEngine update with multi-band chunk
        engine = AstroSnailEngine(backend="numpy")
        update_out = engine.update(multi_band_data[:100])
        assert update_out is not None
        assert "morse_impulse" in update_out
        assert update_out["morse_impulse"].shape == (100, 4)

    def test_cli_ingestion_variable_stars_lc(self, tmp_path):
        """Verifies astrosnail-run can stream and process variable_stars.lc catalog without shape mismatch."""
        catalog_path = Path("data/catalogs/variable_stars.lc")
        if not catalog_path.exists():
            generate_standard_catalogs(output_dir="data/catalogs", seed=20261025)

        out_dir = tmp_path / "run_lc"
        exit_code = main([str(catalog_path), "--output-dir", str(out_dir), "--quiet", "--format", "json"])
        assert exit_code == 0, f"Expected exit code 0 for variable_stars.lc, got {exit_code}"

        summary_file = out_dir / "run_summary.json"
        assert summary_file.exists(), "run_summary.json must exist after processing .lc catalog"
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["status"] == "completed"
        assert summary["total_steps"] > 0

    def test_cli_ingestion_kbo_crossing_cat(self, tmp_path):
        """Verifies astrosnail-run can stream and process kbo_crossing.cat catalog without shape collision."""
        catalog_path = Path("data/catalogs/kbo_crossing.cat")
        if not catalog_path.exists():
            generate_standard_catalogs(output_dir="data/catalogs", seed=20261025)

        out_dir = tmp_path / "run_cat"
        exit_code = main([str(catalog_path), "--output-dir", str(out_dir), "--quiet", "--format", "json"])
        assert exit_code == 0, f"Expected exit code 0 for kbo_crossing.cat, got {exit_code}"

        summary_file = out_dir / "run_summary.json"
        assert summary_file.exists(), "run_summary.json must exist after processing .cat catalog"
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["status"] == "completed"
        assert summary["total_steps"] > 0

    def test_cli_top_level_subcommand_routing(self, tmp_path):
        """Verifies python -m astrosnail.cli <subcommand> routes to verify, benchmark, catalogs, and run."""
        # 1. verify subcommand routing
        code_v = main(["verify", "--all", "--quiet"])
        assert code_v == 0, f"Expected main(['verify', ...]) to return 0, got {code_v}"

        # 2. benchmark subcommand routing
        out_b = tmp_path / "bench_routed"
        code_b = main(["benchmark", "--scenario", "01", "--trials", "1", "--output", str(out_b), "--quiet"])
        assert code_b == 0, f"Expected main(['benchmark', ...]) to return 0, got {code_b}"
        assert (out_b / "astrosnail_benchmark_summary.json").exists() or (out_b / "astrosnail_evaluation_summary.json").exists()

        # 3. catalogs subcommand routing
        out_c = tmp_path / "catalogs_routed"
        code_c = main(["catalogs", "--output-dir", str(out_c), "--quiet"])
        assert code_c == 0, f"Expected main(['catalogs', ...]) to return 0, got {code_c}"
        assert (out_c / "variable_stars.lc").exists()
        assert (out_c / "kbo_crossing.cat").exists()
        assert (out_c / "cosmic_web_skewers.fits").exists()

        # 4. run subcommand routing
        out_r = tmp_path / "run_routed"
        code_r = main(["run", "01", "--output-dir", str(out_r), "--quiet"])
        assert code_r == 0, f"Expected main(['run', ...]) to return 0, got {code_r}"
        assert (out_r / "run_summary.json").exists()

        # 5. default (no subcommand, direct scenario input) routing
        out_d = tmp_path / "default_routed"
        code_d = main(["01", "--output-dir", str(out_d), "--quiet"])
        assert code_d == 0, f"Expected main(['01', ...]) to return 0, got {code_d}"
        assert (out_d / "run_summary.json").exists()

    def test_cli_catalogs_packaging_and_subprocess(self, tmp_path):
        """Verifies pyproject.toml registration and subprocess execution of astrosnail-catalogs."""
        import shutil
        import subprocess
        import sys

        # 1. Verify pyproject.toml scripts registration
        pyproject_path = Path("pyproject.toml")
        assert pyproject_path.exists(), "pyproject.toml must exist in repo root"
        content = pyproject_path.read_text(encoding="utf-8")
        assert 'astrosnail-catalogs = "astrosnail.cli:main_generate_catalogs"' in content, (
            "pyproject.toml must register astrosnail-catalogs in [project.scripts]"
        )

        # 2. Verify module-level execution via python -m astrosnail.cli catalogs
        out_mod = tmp_path / "mod_catalogs"
        res_mod = subprocess.run(
            [sys.executable, "-m", "astrosnail.cli", "catalogs", "--output-dir", str(out_mod), "--quiet"],
            capture_output=True,
            text=True,
            timeout=30.0,
        )
        assert res_mod.returncode == 0, f"python -m astrosnail.cli catalogs failed: {res_mod.stderr}"
        assert (out_mod / "variable_stars.lc").exists()
        assert (out_mod / "kbo_crossing.cat").exists()
        assert (out_mod / "cosmic_web_skewers.fits").exists()

        # 3. Verify console script execution if installed in environment
        script_name = "astrosnail-catalogs.exe" if os.name == "nt" else "astrosnail-catalogs"
        exe_path = shutil.which("astrosnail-catalogs")
        if not exe_path:
            candidate = Path(sys.executable).parent / "Scripts" / script_name
            if candidate.exists():
                exe_path = str(candidate)

        if exe_path:
            res_exe = subprocess.run([exe_path, "--help"], capture_output=True, text=True, timeout=30.0)
            assert res_exe.returncode == 0, f"{exe_path} --help failed: {res_exe.stderr}"
            assert "AstroSnail Standard Synthetic Catalog Generator" in res_exe.stdout

    def test_fits_stream_windows_memmap_file_release(self, tmp_path):
        """Verifies stream_fits_cube cleans up memory-mapped handles immediately so file can be removed on Windows."""
        fits_path = tmp_path / "temp_cube.fits"
        data = np.ones((32, 16, 16), dtype=np.float32)
        write_fits_cube(fits_path, data)

        # Iterate generator fully
        generator = stream_fits_cube(fits_path, chunk_z=16)
        for slab in generator:
            assert slab.shape == (16, 16, 16)
        del slab, generator
        import gc
        gc.collect()

        # File must be removable immediately without [WinError 32]
        fits_path.unlink()
        assert not fits_path.exists()

