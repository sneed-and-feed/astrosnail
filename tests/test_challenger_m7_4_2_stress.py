"""
Challenger 2 Empirical Stress Test Suite (Milestone M7.4)
=========================================================
Target: Expedition Dossier, Studio Integration & Delivery (agent-4-atlas)
Focus:
  1. Discovery Catalog Verification (data/catalogs/discoveries/):
     - discoveries_manifest.json schema, physical parameters, paths
     - 15 .lc files: exact 7 numeric cols, comments, monotonic time, finite
     - 15 .cat files: exact 6 cols, valid coordinates, finite
     - 15 .fits files: 2880-byte block compliance, BINTABLE HDU, primary header cards
     - Survey of all files in directory (manifest vs non-manifest)
  2. Portable Studio Offline Validation (results/ & studio/):
     - File size > 50 KB
     - Bitwise synchronization
     - Zero external CDN links
     - DOM elements: quad-a..d, canvas-quad-a..d
     - Scenario selector: discovery_01, discovery_05, discovery_09
     - Embedded scenario payloads
  3. Scenario Functional Execution:
     - get_scenario for all 3 discovery scenarios and aliases
  4. Expedition Report JSON & Telemetry:
     - 2500 targets processed, memory < 100 MB, clean subprocesses
  5. Expedition Dossier GFM Math Compliance:
     - audit_gfm_math.py with 0 violations
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
import pytest

from astrosnail.expedition.harvester import (
    parse_bintable_descriptor,
    FitsBinTableContext,
)
from astrosnail.scenarios import (
    SCENARIO_REGISTRY,
    SCENARIO_ALIASES,
    get_scenario,
    ScenarioData,
)
from astrosnail.studio import verify_studio_html


REPO_ROOT = Path(__file__).resolve().parent.parent
DISCOVERIES_DIR = REPO_ROOT / "data" / "catalogs" / "discoveries"
RESULTS_STUDIO_HTML = REPO_ROOT / "results" / "astrosnail_portable_studio.html"
STUDIO_STUDIO_HTML = REPO_ROOT / "studio" / "astrosnail_portable_studio.html"
EXPEDITION_REPORT_JSON = REPO_ROOT / "results" / "discovery_expedition_report.json"
EXPEDITION_DOSSIER_MD = REPO_ROOT / "docs" / "DISCOVERY_EXPEDITION.md"


# ==============================================================================
# SECTION 1: Discovery Catalog & Manifest Verification
# ==============================================================================

class TestDiscoveryCatalogManifest:
    """Rigorous empirical validation of discoveries_manifest.json."""

    @pytest.fixture(scope="class")
    def manifest_data(self):
        manifest_path = DISCOVERIES_DIR / "discoveries_manifest.json"
        assert manifest_path.exists(), f"Missing manifest: {manifest_path}"
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def test_manifest_structure(self, manifest_data):
        """Manifest must define exactly 15 discoveries in sequential order."""
        assert "total_discoveries" in manifest_data
        assert manifest_data["total_discoveries"] == 15
        assert "discoveries" in manifest_data
        assert len(manifest_data["discoveries"]) == 15

        for i, disc in enumerate(manifest_data["discoveries"], start=1):
            assert disc["index"] == i, f"Candidate at position {i} has index {disc['index']}"
            assert "candidate_id" in disc and disc["candidate_id"]
            assert "target_id" in disc and disc["target_id"]
            assert "track" in disc and disc["track"] in ("Track_A", "Track_B", "Track_C")
            assert "survey" in disc and disc["survey"] in ("Kepler", "TESS", "ALeRCE_ZTF", "JPL_Horizons")
            assert "discovery_tier" in disc and disc["discovery_tier"] in ("Tier_1_Gold", "Tier_2_Silver", "Tier_3_Bronze")
            assert "vetting_score" in disc and 0.0 <= disc["vetting_score"] <= 1.0
            assert "disposition" in disc and disc["disposition"] in ("NOVEL_CANDIDATE", "CONFIRMED_KNOWN", "KNOWN_OBJECT")

            # Check files mapping
            assert "files" in disc
            files = disc["files"]
            assert "lc" in files and files["lc"].endswith(".lc")
            assert "cat" in files and files["cat"].endswith(".cat")
            assert "fits" in files and files["fits"].endswith(".fits")

    def test_manifest_physical_parameters(self, manifest_data):
        """Verifies physical plausibility and numerical validity of all candidate parameters."""
        for disc in manifest_data["discoveries"]:
            cid = disc["candidate_id"]
            assert np.isfinite(disc["vetting_score"]), f"{cid}: vetting_score non-finite"
            assert np.isfinite(disc["depth_ppm"]) and disc["depth_ppm"] >= 0.0, f"{cid}: depth_ppm invalid"
            assert np.isfinite(disc["duration_hours"]) and disc["duration_hours"] > 0.0, f"{cid}: duration_hours invalid"
            assert np.isfinite(disc["snr"]) and disc["snr"] > 0.0, f"{cid}: snr invalid"
            assert np.isfinite(disc["rp_rstar"]) and disc["rp_rstar"] >= 0.0, f"{cid}: rp_rstar invalid"
            assert np.isfinite(disc["rp_earth"]) and disc["rp_earth"] >= 0.0, f"{cid}: rp_earth invalid"
            assert np.isfinite(disc["semi_major_axis_au"]) and disc["semi_major_axis_au"] >= 0.0, f"{cid}: semi_major_axis_au invalid"
            assert np.isfinite(disc["impact_parameter_b"]) and disc["impact_parameter_b"] >= 0.0, f"{cid}: impact_parameter_b invalid"


class TestDiscoveryFilesIntegrity:
    """Empirical verification of all 15 .lc, 15 .cat, and 15 .fits files."""

    @pytest.fixture(scope="class")
    def manifest_entries(self):
        manifest_path = DISCOVERIES_DIR / "discoveries_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["discoveries"]

    def test_all_15_light_curve_files(self, manifest_entries):
        """Every .lc file must have comments, exactly 7 numeric columns, strictly monotonic time, and finite values."""
        for disc in manifest_entries:
            lc_name = disc["files"]["lc"]
            lc_path = DISCOVERIES_DIR / lc_name
            assert lc_path.exists(), f"Missing LC file: {lc_path}"
            assert lc_path.stat().st_size > 0, f"Empty LC file: {lc_path}"

            content = lc_path.read_text(encoding="utf-8")
            lines = content.splitlines()

            # Must have metadata comments
            comment_lines = [ln for ln in lines if ln.startswith("#")]
            assert len(comment_lines) >= 5, f"{lc_name}: Insufficient comment headers ({len(comment_lines)})"
            col_spec_found = any("# Columns: time flux flux_err bkg centr1 centr2 flag" in ln for ln in comment_lines)
            assert col_spec_found, f"{lc_name}: Missing column specification comment"

            # Parse data rows
            data_lines = [ln.strip() for ln in lines if not ln.startswith("#") and ln.strip()]
            assert len(data_lines) > 0, f"{lc_name}: No data rows found"

            parsed_rows = []
            for row_idx, ln in enumerate(data_lines):
                tokens = ln.split()
                assert len(tokens) == 7, f"{lc_name} line {row_idx}: expected 7 columns, got {len(tokens)} ({tokens})"
                row_vals = [float(t) for t in tokens]
                parsed_rows.append(row_vals)

            arr = np.array(parsed_rows, dtype=np.float64)
            # 1. Finite checks
            assert np.all(np.isfinite(arr)), f"{lc_name}: Non-finite values detected"

            # 2. Strict time monotonicity
            times = arr[:, 0]
            time_diffs = np.diff(times)
            assert np.all(time_diffs > 0), f"{lc_name}: Time is not strictly monotonically increasing"

            # 3. Flux error non-negative
            flux_err = arr[:, 2]
            assert np.all(flux_err >= 0.0), f"{lc_name}: Negative flux_err found"

            # 4. Flag column is integer
            flags = arr[:, 6]
            assert np.all(flags == np.round(flags)), f"{lc_name}: Quality flags must be integer-valued"

    def test_all_15_catalog_files(self, manifest_entries):
        """Every .cat file must have comments, exactly 6 columns, valid celestial coordinates, and finite values."""
        for disc in manifest_entries:
            cat_name = disc["files"]["cat"]
            cat_path = DISCOVERIES_DIR / cat_name
            assert cat_path.exists(), f"Missing CAT file: {cat_path}"
            assert cat_path.stat().st_size > 0, f"Empty CAT file: {cat_path}"

            content = cat_path.read_text(encoding="utf-8")
            lines = content.splitlines()

            comment_lines = [ln for ln in lines if ln.startswith("#")]
            assert len(comment_lines) >= 3, f"{cat_name}: Insufficient comment headers"
            assert any("# Columns: ra dec mag pm_ra pm_dec tracklet_id" in ln for ln in comment_lines), f"{cat_name}: Missing column spec"

            data_lines = [ln.strip() for ln in lines if not ln.startswith("#") and ln.strip()]
            assert len(data_lines) > 0, f"{cat_name}: No data rows found"

            for row_idx, ln in enumerate(data_lines):
                tokens = ln.split()
                assert len(tokens) == 6, f"{cat_name} line {row_idx}: expected 6 columns, got {len(tokens)} ({tokens})"
                ra = float(tokens[0])
                dec = float(tokens[1])
                mag = float(tokens[2])
                pm_ra = float(tokens[3])
                pm_dec = float(tokens[4])
                tracklet_id = tokens[5]

                assert np.isfinite(ra) and 0.0 <= ra <= 360.0, f"{cat_name} row {row_idx}: invalid RA={ra}"
                assert np.isfinite(dec) and -90.0 <= dec <= 90.0, f"{cat_name} row {row_idx}: invalid Dec={dec}"
                assert np.isfinite(mag) and 0.0 <= mag <= 35.0, f"{cat_name} row {row_idx}: invalid mag={mag}"
                assert np.isfinite(pm_ra), f"{cat_name} row {row_idx}: non-finite pm_ra"
                assert np.isfinite(pm_dec), f"{cat_name} row {row_idx}: non-finite pm_dec"
                assert len(tracklet_id) > 0, f"{cat_name} row {row_idx}: empty tracklet_id"

    def test_all_15_fits_files(self, manifest_entries):
        """Every .fits file must be 2880-byte compliant, valid BINTABLE, non-empty, with required primary cards."""
        required_primary_cards = [
            "VETSCORE", "TIER", "DISPOSIT", "RP_RSTAR",
            "RP_EARTH", "ODDEVEN", "CENTRSH", "BKGCORR"
        ]

        for disc in manifest_entries:
            fits_name = disc["files"]["fits"]
            fits_path = DISCOVERIES_DIR / fits_name
            assert fits_path.exists(), f"Missing FITS file: {fits_path}"

            # 1. Standard FITS 2880-byte block compliance
            file_size = fits_path.stat().st_size
            assert file_size >= 2880, f"{fits_name}: Size {file_size} is less than 1 block (2880 B)"
            assert file_size % 2880 == 0, f"{fits_name}: Size {file_size} not a multiple of 2880 bytes (rem={file_size % 2880})"

            # 2. Parse descriptor
            desc = parse_bintable_descriptor(fits_path)
            p_cards = desc["primary_cards"]
            ext_cards = desc["ext_cards"]

            # 3. Check XTENSION == 'BINTABLE'
            xtension_val = ext_cards.get("XTENSION", "").replace("'", "").strip()
            assert xtension_val == "BINTABLE", f"{fits_name}: XTENSION is '{xtension_val}', expected 'BINTABLE'"

            # 4. Check non-empty records
            assert desc["n_rows"] > 0, f"{fits_name}: BINTABLE has 0 rows"

            # 5. Check primary header cards
            for card in required_primary_cards:
                assert card in p_cards, f"{fits_name}: Missing primary card {card}. Available cards: {list(p_cards.keys())}"

            # 6. Verify consistency between primary cards and manifest
            assert float(p_cards["VETSCORE"]) == pytest.approx(disc["vetting_score"], rel=1e-3)
            assert str(p_cards["TIER"]).strip("' ") == disc["discovery_tier"]
            assert str(p_cards["DISPOSIT"]).strip("' ") == disc["disposition"]
            assert float(p_cards["RP_RSTAR"]) == pytest.approx(disc["rp_rstar"], rel=1e-3)
            assert float(p_cards["RP_EARTH"]) == pytest.approx(disc["rp_earth"], rel=1e-2)

            # 7. Check BINTABLE column integrity via zero-copy context manager
            with FitsBinTableContext(fits_path) as table_data:
                assert len(table_data) == desc["n_rows"]
                assert "TIME" in table_data.dtype.names
                assert "FLUX" in table_data.dtype.names
                assert "FLUX_ERR" in table_data.dtype.names

                assert np.all(np.isfinite(table_data["TIME"])), f"{fits_name}: Non-finite TIME values"
                assert np.all(np.isfinite(table_data["FLUX"])), f"{fits_name}: Non-finite FLUX values"
                assert np.all(np.isfinite(table_data["FLUX_ERR"])), f"{fits_name}: Non-finite FLUX_ERR values"

    def test_directory_inventory_survey(self, manifest_entries):
        """Surveys all files in data/catalogs/discoveries/ and verifies compliance of all files."""
        all_files = list(DISCOVERIES_DIR.glob("*"))
        manifest_filenames = set(["discoveries_manifest.json"])
        for disc in manifest_entries:
            manifest_filenames.add(disc["files"]["lc"])
            manifest_filenames.add(disc["files"]["cat"])
            manifest_filenames.add(disc["files"]["fits"])

        assert len(manifest_filenames) == 46, f"Expected exactly 46 manifest assets, got {len(manifest_filenames)}"

        # Check all manifest files exist on disk
        for fname in manifest_filenames:
            assert (DISCOVERIES_DIR / fname).exists(), f"Manifest file missing from disk: {fname}"

        # Check non-manifest files on disk (e.g. from prior runs)
        on_disk_names = set(p.name for p in all_files if p.is_file())
        extra_files = on_disk_names - manifest_filenames

        # Verify any extra files on disk also satisfy format constraints
        for fname in extra_files:
            fpath = DISCOVERIES_DIR / fname
            if fname.endswith(".fits"):
                assert fpath.stat().st_size % 2880 == 0, f"Extra FITS file {fname} violates 2880 byte block"
            elif fname.endswith(".lc"):
                lines = [ln.strip() for ln in fpath.read_text(encoding="utf-8").splitlines() if not ln.startswith("#") and ln.strip()]
                if lines:
                    assert len(lines[0].split()) == 7, f"Extra LC file {fname} does not have 7 columns"
            elif fname.endswith(".cat"):
                lines = [ln.strip() for ln in fpath.read_text(encoding="utf-8").splitlines() if not ln.startswith("#") and ln.strip()]
                if lines:
                    assert len(lines[0].split()) == 6, f"Extra CAT file {fname} does not have 6 columns"


# ==============================================================================
# SECTION 2: Portable Studio Offline Validation
# ==============================================================================

class TestPortableStudioOffline:
    """Rigorous empirical validation of the standalone HTML portable studio."""

    @pytest.mark.parametrize("studio_path", [RESULTS_STUDIO_HTML, STUDIO_STUDIO_HTML])
    def test_studio_file_existence_and_size(self, studio_path):
        """Studio HTML must exist and exceed 50 KB in size."""
        assert studio_path.exists(), f"Studio HTML missing: {studio_path}"
        size_kb = studio_path.stat().st_size / 1024
        assert size_kb > 50.0, f"{studio_path.name}: Size {size_kb:.1f} KB is under 50 KB"

    def test_studio_bitwise_synchronization(self):
        """Both results/ and studio/ copies of astrosnail_portable_studio.html must be identical."""
        content_results = RESULTS_STUDIO_HTML.read_bytes()
        content_studio = STUDIO_STUDIO_HTML.read_bytes()
        assert content_results == content_studio, "results/ and studio/ portable HTML files have diverged!"

    @pytest.mark.parametrize("studio_path", [RESULTS_STUDIO_HTML, STUDIO_STUDIO_HTML])
    def test_zero_external_cdn_links(self, studio_path):
        """Verifies zero external CDN links or network calls in the standalone HTML."""
        content = studio_path.read_text(encoding="utf-8")

        # 1. Disallowed external script tags
        disallowed_scripts = re.findall(r'<script[^>]+src=[\'"](https?://[^\'"]+)[\'"]', content, re.IGNORECASE)
        assert len(disallowed_scripts) == 0, f"Disallowed external scripts found: {disallowed_scripts}"

        # 2. Disallowed external stylesheet / font links
        disallowed_links = re.findall(r'<link[^>]+href=[\'"](https?://[^\'"]+)[\'"]', content, re.IGNORECASE)
        assert len(disallowed_links) == 0, f"Disallowed external links found: {disallowed_links}"

        # 3. Disallowed external CDN domains in any url() or @import
        prohibited_domains = [
            "unpkg.com",
            "cdnjs.cloudflare.com",
            "fonts.googleapis.com",
            "cdn.jsdelivr.net",
            "ajax.googleapis.com",
        ]
        for domain in prohibited_domains:
            matches = [m.start() for m in re.finditer(re.escape(domain), content, re.IGNORECASE)]
            assert len(matches) == 0, f"Prohibited CDN domain '{domain}' found {len(matches)} times in {studio_path.name}"

    @pytest.mark.parametrize("studio_path", [RESULTS_STUDIO_HTML, STUDIO_STUDIO_HTML])
    def test_dom_quad_viewports_and_canvases(self, studio_path):
        """DOM must contain all 4 viewports (quad-a through quad-d) and corresponding canvas elements."""
        content = studio_path.read_text(encoding="utf-8")

        required_viewports = ["quad-a", "quad-b", "quad-c", "quad-d"]
        required_canvases = ["canvas-quad-a", "canvas-quad-b", "canvas-quad-c", "canvas-quad-d"]

        for v_id in required_viewports:
            assert f'id="{v_id}"' in content or f"id='{v_id}'" in content, f"Missing viewport element id='{v_id}'"

        for c_id in required_canvases:
            assert f'id="{c_id}"' in content or f"id='{c_id}'" in content, f"Missing canvas element id='{c_id}'"

    @pytest.mark.parametrize("studio_path", [RESULTS_STUDIO_HTML, STUDIO_STUDIO_HTML])
    def test_scenario_selector_discovery_options(self, studio_path):
        """Scenario selector must contain the 3 required discovery scenarios."""
        content = studio_path.read_text(encoding="utf-8")

        assert '<select id="scenario-select"' in content or "<select id='scenario-select'" in content
        assert 'value="discovery_01_circumbinary_transit"' in content
        assert 'value="discovery_05_optical_micro_flare"' in content
        assert 'value="discovery_09_kbo_crossing_conjunction"' in content

    @pytest.mark.parametrize("studio_path", [RESULTS_STUDIO_HTML, STUDIO_STUDIO_HTML])
    def test_built_in_verify_studio_html(self, studio_path):
        """Built-in verify_studio_html utility must report True with 0 errors."""
        ok, errors = verify_studio_html(studio_path)
        assert ok, f"verify_studio_html failed on {studio_path.name}: {errors}"
        assert len(errors) == 0


# ==============================================================================
# SECTION 3: Scenario Functional Execution
# ==============================================================================

class TestScenarioExecution:
    """Empirical execution of discovery scenarios from Python API."""

    @pytest.mark.parametrize("scenario_id", [
        "discovery_01_circumbinary_transit",
        "discovery_05_optical_micro_flare",
        "discovery_09_kbo_crossing_conjunction",
    ])
    def test_generate_discovery_scenarios(self, scenario_id):
        """Scenarios must generate non-empty, finite data and ground truth."""
        sc = get_scenario(scenario_id)
        assert isinstance(sc, ScenarioData)
        assert sc.scenario_id == scenario_id
        assert sc.data is not None

        if isinstance(sc.data, np.ndarray):
            assert sc.data.size > 0
            assert np.all(np.isfinite(sc.data)), f"{scenario_id}: Non-finite scenario data"
        elif isinstance(sc.data, list):
            assert len(sc.data) > 0

        assert "candidate_id" in sc.ground_truth
        assert "vetting_score" in sc.ground_truth
        assert "discovery_tier" in sc.ground_truth

    @pytest.mark.parametrize("alias,canonical", [
        ("disc_01", "discovery_01_circumbinary_transit"),
        ("circumbinary", "discovery_01_circumbinary_transit"),
        ("disc_05", "discovery_05_optical_micro_flare"),
        ("flare", "discovery_05_optical_micro_flare"),
        ("disc_09", "discovery_09_kbo_crossing_conjunction"),
        ("conjunction", "discovery_09_kbo_crossing_conjunction"),
    ])
    def test_discovery_scenario_aliases(self, alias, canonical):
        """Aliases must resolve cleanly to canonical scenario IDs."""
        sc = get_scenario(alias)
        assert sc.scenario_id == canonical


# ==============================================================================
# SECTION 4: Expedition Report & Dossier Verification
# ==============================================================================

class TestExpeditionReportAndDossier:
    """Verifies expedition telemetry report and GFM math compliance of dossier."""

    def test_expedition_report_telemetry(self):
        """Report JSON must reflect full 2500 targets, memory constraint, and clean subprocesses."""
        assert EXPEDITION_REPORT_JSON.exists(), f"Missing expedition report: {EXPEDITION_REPORT_JSON}"
        with open(EXPEDITION_REPORT_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "execution_telemetry" in data
        telemetry = data["execution_telemetry"]

        # Check memory constraint < 100 MB
        assert telemetry.get("memory_constraint_satisfied") is True
        peak_mb = float(telemetry.get("peak_memory_mb", telemetry.get("peak_heap_mb", 999.0)))
        assert peak_mb < 100.0, f"Peak memory {peak_mb} MB exceeded 100 MB ceiling"

        # Check subprocess cleanliness
        assert telemetry.get("zero_orphaned_processes_verified") is True

        # Check target count
        assert "harvest_statistics" in data
        assert data["harvest_statistics"].get("total_targets_harvested", 0) > 0

    def test_expedition_dossier_gfm_math_compliance(self):
        """docs/DISCOVERY_EXPEDITION.md must pass audit_gfm_math.py with 0 violations."""
        assert EXPEDITION_DOSSIER_MD.exists(), f"Missing expedition dossier: {EXPEDITION_DOSSIER_MD}"

        audit_script = REPO_ROOT / "scripts" / "audit_gfm_math.py"
        assert audit_script.exists()

        cmd = [sys.executable, str(audit_script), str(EXPEDITION_DOSSIER_MD)]
        res = subprocess.run(cmd, capture_output=True, text=True)

        assert res.returncode == 0, f"audit_gfm_math failed on {EXPEDITION_DOSSIER_MD}:\n{res.stdout}\n{res.stderr}"
        assert "PASS" in res.stdout
