"""
Challenger M7.4-r2-2 Empirical Verification & Adversarial Stress Suite
======================================================================
Archetype: Empirical Challenger (Critic & Specialist)
Milestone: M7.4 Iteration 2 (agent-4-atlas)

Mandate Focus:
1. Empirically verify all 15 discovery targets across all 46 files in data/catalogs/discoveries/
   - 15 .lc files: exact 7 numeric columns, valid headers, strict time monotonicity, finite values
   - 15 .cat files: exact 6 columns, valid celestial coordinates, non-empty tracklet IDs
   - 15 .fits files: exact multiple of 2880 bytes, valid FITS standard, BINTABLE HDU, primary header cards
   - discoveries_manifest.json: exact schema, 15 sequential discoveries, physical parameter bounds
   - Survey of catalog directory: exactly 46 files on disk and in git index (0 orphan files)
2. Empirically verify Track B representation (at least 4 candidates)
   - discoveries_manifest.json contains >= 4 Track B candidates
   - Track B candidates have valid short-cadence attributes, light curves, and catalog files
   - Stratified multi-track portfolio selection algorithm robustness and boundary testing
3. Validate offline portable studio HTML files in results/ and studio/
   - File size > 50 KB
   - Bitwise SHA-256 synchronization between results/ and studio/
   - Zero external CDN links or network URLs
   - Complete 4-quad viewports (quad-a through quad-d) and canvas elements
   - Scenario selector includes discovery scenarios
   - verify_studio_html() passes with 0 errors
4. Adversarial stress testing & report deliverable immunity
   - Malformed/corrupted file robustness for I/O parsers
   - 2,500-target report telemetry integrity and internal arithmetic reconciliation
   - Report JSON immunity against test-suite clobbering
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, List, Set

import numpy as np
import pytest

from astrosnail.io import parse_lc_table, parse_cat_file
from astrosnail.expedition.harvester import (
    parse_bintable_descriptor,
    FitsBinTableContext,
)
from astrosnail.expedition.atlas import (
    create_curated_reference_portfolio,
    ExpeditionAtlas,
)
from astrosnail.expedition.vetter import VettedCandidate
from astrosnail.scenarios import get_scenario, ScenarioData
from astrosnail.studio import verify_studio_html


REPO_ROOT = Path(__file__).resolve().parent.parent
DISCOVERIES_DIR = REPO_ROOT / "data" / "catalogs" / "discoveries"
MANIFEST_PATH = DISCOVERIES_DIR / "discoveries_manifest.json"
RESULTS_STUDIO_HTML = REPO_ROOT / "results" / "astrosnail_portable_studio.html"
STUDIO_STUDIO_HTML = REPO_ROOT / "studio" / "astrosnail_portable_studio.html"
REPORT_JSON_PATH = REPO_ROOT / "results" / "discovery_expedition_report.json"
DOSSIER_MD_PATH = REPO_ROOT / "docs" / "DISCOVERY_EXPEDITION.md"


# ==============================================================================
# 1. CATALOG FORMATS & INVENTORY INTEGRITY (46 Files, 15 Targets)
# ==============================================================================

class TestDiscoveryCatalogFormatsAndInventory:
    """Rigorous empirical validation of all 46 files in data/catalogs/discoveries/."""

    @pytest.fixture(scope="class")
    def manifest_payload(self) -> dict:
        assert MANIFEST_PATH.exists(), f"Missing manifest at {MANIFEST_PATH}"
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def test_catalog_inventory_exact_count(self, manifest_payload):
        """Directory data/catalogs/discoveries/ must contain EXACTLY 46 files (15 targets * 3 + 1 manifest)."""
        all_disk_files = [p for p in DISCOVERIES_DIR.iterdir() if p.is_file()]
        assert len(all_disk_files) == 46, (
            f"Expected exactly 46 files in {DISCOVERIES_DIR}, but found {len(all_disk_files)}. "
            f"Files: {[p.name for p in all_disk_files]}"
        )

        # Check git index tracked files
        try:
            git_out = subprocess.check_output(
                ["git", "ls-files", str(DISCOVERIES_DIR.relative_to(REPO_ROOT))],
                cwd=REPO_ROOT,
                text=True,
            ).splitlines()
            assert len(git_out) == 46, (
                f"Expected 46 tracked files in git index for discoveries catalog, got {len(git_out)}: {git_out}"
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    def test_manifest_schema_and_cross_reference_closure(self, manifest_payload):
        """Manifest must define exactly 15 discoveries in sequence with 100% file cross-reference closure."""
        assert manifest_payload.get("total_discoveries") == 15
        discoveries = manifest_payload.get("discoveries", [])
        assert len(discoveries) == 15

        all_manifest_files: Set[str] = {"discoveries_manifest.json"}

        for idx, item in enumerate(discoveries, start=1):
            assert item["index"] == idx, f"Candidate {item.get('candidate_id')} has index {item['index']} != {idx}"
            assert item["candidate_id"], f"Empty candidate_id at index {idx}"
            assert item["target_id"], f"Empty target_id at index {idx}"
            assert item["track"] in ("Track_A", "Track_B", "Track_C")
            assert item["survey"] in ("Kepler", "TESS", "ALeRCE_ZTF", "JPL_Horizons")
            assert item["discovery_tier"] in ("Tier_1_Gold", "Tier_2_Silver", "Tier_3_Bronze")
            assert item["disposition"] in ("NOVEL_CANDIDATE", "CONFIRMED_KNOWN", "KNOWN_OBJECT")

            # Physical parameters validity
            assert 0.0 <= item["vetting_score"] <= 1.0
            assert item["depth_ppm"] >= 0.0
            assert item["duration_hours"] > 0.0
            assert item["snr"] > 0.0
            assert item["rp_rstar"] >= 0.0
            assert item["rp_earth"] >= 0.0
            assert item["semi_major_axis_au"] >= 0.0
            assert item["impact_parameter_b"] >= 0.0

            # File mapping verification
            files = item.get("files", {})
            assert "lc" in files and files["lc"].endswith(".lc")
            assert "cat" in files and files["cat"].endswith(".cat")
            assert "fits" in files and files["fits"].endswith(".fits")

            all_manifest_files.add(files["lc"])
            all_manifest_files.add(files["cat"])
            all_manifest_files.add(files["fits"])

        assert len(all_manifest_files) == 46, f"Expected 46 unique files in manifest, got {len(all_manifest_files)}"

        disk_file_names = set(p.name for p in DISCOVERIES_DIR.iterdir() if p.is_file())
        missing_on_disk = all_manifest_files - disk_file_names
        assert len(missing_on_disk) == 0, f"Manifest references missing files on disk: {missing_on_disk}"

        orphans_on_disk = disk_file_names - all_manifest_files
        assert len(orphans_on_disk) == 0, f"Unreferenced orphan files found on disk: {orphans_on_disk}"

    def test_all_15_light_curve_files_format_and_parsing(self, manifest_payload):
        """Every .lc file must have 7 numeric cols, valid comments, strictly monotonic time, and be parseable by parse_lc_table."""
        discoveries = manifest_payload["discoveries"]

        for item in discoveries:
            lc_name = item["files"]["lc"]
            lc_path = DISCOVERIES_DIR / lc_name
            assert lc_path.exists(), f"Missing .lc file: {lc_path}"
            assert lc_path.stat().st_size > 0, f"Empty .lc file: {lc_path}"

            raw_text = lc_path.read_text(encoding="utf-8")
            raw_lines = raw_text.splitlines()

            # Comment header verification
            comments = [ln for ln in raw_lines if ln.startswith("#")]
            assert len(comments) >= 5, f"{lc_name}: Less than 5 comment lines ({len(comments)})"
            assert any("# Columns: time flux flux_err bkg centr1 centr2 flag" in ln for ln in comments), (
                f"{lc_name}: Missing exact '# Columns: time flux flux_err bkg centr1 centr2 flag' header"
            )

            # Data rows verification: all 7 columns
            data_lines = [ln.strip() for ln in raw_lines if not ln.startswith("#") and ln.strip()]
            assert len(data_lines) >= 10, f"{lc_name}: Expected at least 10 data rows, got {len(data_lines)}"

            rows_data = []
            for r_idx, line in enumerate(data_lines):
                tokens = line.split()
                assert len(tokens) == 7, (
                    f"{lc_name} row {r_idx}: expected exactly 7 columns, got {len(tokens)}: {tokens}"
                )
                vals = [float(tok) for tok in tokens]
                rows_data.append(vals)

            arr = np.array(rows_data, dtype=np.float64)

            # Strict time monotonicity
            t = arr[:, 0]
            dt = np.diff(t)
            assert np.all(dt > 0.0), f"{lc_name}: Non-monotonic time steps: min dt = {np.min(dt)}"

            # Finite values
            assert np.all(np.isfinite(arr)), f"{lc_name}: Contains NaN or Inf values"

            # Flux error non-negative
            flux_err = arr[:, 2]
            assert np.all(flux_err >= 0.0), f"{lc_name}: Negative flux_err values found"

            # Quality flag is integer
            flag = arr[:, 6]
            assert np.all(flag == np.round(flag)), f"{lc_name}: Quality flag values not integer"

            # Verify with project parser parse_lc_table() (extracts time, flux, flux_err)
            parsed_lc = parse_lc_table(str(lc_path))
            assert isinstance(parsed_lc, np.ndarray)
            assert parsed_lc.shape == (arr.shape[0], 3), (
                f"{lc_name}: parse_lc_table shape {parsed_lc.shape} != expected ({arr.shape[0]}, 3)"
            )
            np.testing.assert_allclose(parsed_lc, arr[:, :3], rtol=1e-7, atol=1e-7)

    def test_all_15_catalog_files_format_and_parsing(self, manifest_payload):
        """Every .cat file must have exactly 6 cols, valid celestial coordinates, and be parseable by parse_cat_file."""
        discoveries = manifest_payload["discoveries"]

        for item in discoveries:
            cat_name = item["files"]["cat"]
            cat_path = DISCOVERIES_DIR / cat_name
            assert cat_path.exists(), f"Missing .cat file: {cat_path}"
            assert cat_path.stat().st_size > 0, f"Empty .cat file: {cat_path}"

            raw_text = cat_path.read_text(encoding="utf-8")
            raw_lines = raw_text.splitlines()

            # Comment header verification
            comments = [ln for ln in raw_lines if ln.startswith("#")]
            assert len(comments) >= 3, f"{cat_name}: Less than 3 comment lines ({len(comments)})"
            assert any("# Columns: ra dec mag pm_ra pm_dec tracklet_id" in ln for ln in comments), (
                f"{cat_name}: Missing exact '# Columns: ra dec mag pm_ra pm_dec tracklet_id' header"
            )

            # Data rows verification
            data_lines = [ln.strip() for ln in raw_lines if not ln.startswith("#") and ln.strip()]
            assert len(data_lines) >= 1, f"{cat_name}: No data rows found"

            for r_idx, line in enumerate(data_lines):
                tokens = line.split()
                assert len(tokens) == 6, (
                    f"{cat_name} row {r_idx}: expected exactly 6 columns, got {len(tokens)}: {tokens}"
                )
                ra = float(tokens[0])
                dec = float(tokens[1])
                mag = float(tokens[2])
                pm_ra = float(tokens[3])
                pm_dec = float(tokens[4])
                tracklet_id = tokens[5]

                assert 0.0 <= ra <= 360.0, f"{cat_name} row {r_idx}: RA {ra} out of [0, 360]"
                assert -90.0 <= dec <= 90.0, f"{cat_name} row {r_idx}: Dec {dec} out of [-90, 90]"
                assert 0.0 <= mag <= 35.0, f"{cat_name} row {r_idx}: Mag {mag} out of [0, 35]"
                assert np.isfinite(pm_ra) and np.isfinite(pm_dec), f"{cat_name} row {r_idx}: Non-finite proper motion"
                assert len(tracklet_id) > 0, f"{cat_name} row {r_idx}: Empty tracklet_id"

            # Verify with project parser parse_cat_file()
            parsed_cat = parse_cat_file(str(cat_path))
            assert isinstance(parsed_cat, list)
            assert len(parsed_cat) == len(data_lines), (
                f"{cat_name}: parse_cat_file row count {len(parsed_cat)} != raw lines {len(data_lines)}"
            )

    def test_all_15_fits_files_standard_compliance_and_streaming(self, manifest_payload):
        """Every .fits file must be an exact multiple of 2880 bytes, valid BINTABLE HDU, and readable via FitsBinTableContext."""
        discoveries = manifest_payload["discoveries"]
        required_cards = [
            "SIMPLE", "BITPIX", "NAXIS", "EXTEND",
            "VETSCORE", "TIER", "DISPOSIT", "RP_RSTAR",
            "RP_EARTH", "ODDEVEN", "CENTRSH", "BKGCORR",
            "TRACK", "CAND_ID", "TARG_ID"
        ]

        for item in discoveries:
            fits_name = item["files"]["fits"]
            fits_path = DISCOVERIES_DIR / fits_name
            assert fits_path.exists(), f"Missing .fits file: {fits_path}"

            # 1. Exact multiple of 2880 bytes
            file_size = fits_path.stat().st_size
            assert file_size >= 2880, f"{fits_name}: Size {file_size} < 2880 bytes"
            assert file_size % 2880 == 0, (
                f"{fits_name}: Size {file_size} is not a multiple of 2880 (remainder {file_size % 2880})"
            )

            # 2. Parse descriptor
            desc = parse_bintable_descriptor(fits_path)
            primary_cards = desc["primary_cards"]
            ext_cards = desc["ext_cards"]

            for card in required_cards:
                assert card in primary_cards, f"{fits_name}: Missing primary header card '{card}'"

            # Check XTENSION in extension cards
            xtension_val = ext_cards.get("XTENSION", "").replace("'", "").strip()
            assert xtension_val == "BINTABLE", f"{fits_name}: Extension is not BINTABLE (got '{xtension_val}')"

            # Check row count
            assert desc["n_rows"] > 0, f"{fits_name}: BINTABLE has 0 rows"

            # Cross-verify with manifest parameters
            assert float(primary_cards["VETSCORE"]) == pytest.approx(item["vetting_score"], rel=1e-3)
            assert str(primary_cards["TIER"]).strip("' ") == item["discovery_tier"]
            assert str(primary_cards["DISPOSIT"]).strip("' ") == item["disposition"]
            assert str(primary_cards["TRACK"]).strip("' ") == item["track"]
            assert item["candidate_id"].startswith(str(primary_cards["CAND_ID"]).strip("' "))
            assert str(primary_cards["TARG_ID"]).strip("' ") == item["target_id"]

            # 3. Stream data via FitsBinTableContext out-of-core
            with FitsBinTableContext(fits_path) as table:
                assert len(table) == desc["n_rows"]
                assert "TIME" in table.dtype.names
                assert "FLUX" in table.dtype.names
                assert "FLUX_ERR" in table.dtype.names

                time_arr = np.array(table["TIME"], dtype=np.float64)
                flux_arr = np.array(table["FLUX"], dtype=np.float64)
                flux_err_arr = np.array(table["FLUX_ERR"], dtype=np.float64)

                assert np.all(np.isfinite(time_arr)), f"{fits_name}: Non-finite TIME values"
                assert np.all(np.isfinite(flux_arr)), f"{fits_name}: Non-finite FLUX values"
                assert np.all(np.isfinite(flux_err_arr)), f"{fits_name}: Non-finite FLUX_ERR values"
                assert np.all(np.diff(time_arr) > 0.0), f"{fits_name}: TIME is not strictly monotonic"


# ==============================================================================
# 2. TRACK B REPRESENTATION & PORTFOLIO STRATIFICATION
# ==============================================================================

class TestTrackBRepresentationAndStratification:
    """Empirical verification of Track B candidate representation and portfolio allocation."""

    @pytest.fixture(scope="class")
    def manifest_payload(self) -> dict:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_track_b_representation_minimum_quota(self, manifest_payload):
        """Track B must have at least 4 candidates in discoveries_manifest.json."""
        discoveries = manifest_payload["discoveries"]
        track_b_cands = [d for d in discoveries if d["track"] == "Track_B"]

        assert len(track_b_cands) >= 4, (
            f"Mandate requires at least 4 Track B candidates, but found only {len(track_b_cands)}: "
            f"{[d['candidate_id'] for d in track_b_cands]}"
        )

        # Full track distribution verification
        track_counts = {}
        for d in discoveries:
            track_counts[d["track"]] = track_counts.get(d["track"], 0) + 1

        assert track_counts.get("Track_A", 0) >= 4, f"Track A count under 4: {track_counts.get('Track_A')}"
        assert track_counts.get("Track_B", 0) >= 4, f"Track B count under 4: {track_counts.get('Track_B')}"
        assert track_counts.get("Track_C", 0) >= 4, f"Track C count under 4: {track_counts.get('Track_C')}"
        assert sum(track_counts.values()) == 15, f"Total candidates != 15: {track_counts}"

    def test_track_b_candidates_astrophysical_properties(self, manifest_payload):
        """Verifies physical and observational realism of Track B candidates."""
        track_b_cands = [d for d in manifest_payload["discoveries"] if d["track"] == "Track_B"]

        for cand in track_b_cands:
            cid = cand["candidate_id"]
            # Target ID must reflect short cadence or flare star
            assert "sc" in cand["target_id"].lower() or "fl" in cid.lower() or "flare" in cid.lower(), (
                f"{cid}: Expected short-cadence / flare attribution in target_id '{cand['target_id']}'"
            )
            # Tier 1 Gold vetting
            assert cand["discovery_tier"] == "Tier_1_Gold"
            assert cand["vetting_score"] >= 0.85
            assert cand["snr"] >= 7.1

            # Duration of micro-flare is typically sub-hour or a few hours
            assert 0.01 <= cand["duration_hours"] <= 12.0, (
                f"{cid}: Track B transient duration {cand['duration_hours']}h outside expected [0.01, 12.0]h"
            )

            # Check that corresponding .lc file has elevated flux (flare peak > baseline)
            lc_path = DISCOVERIES_DIR / cand["files"]["lc"]
            lc_arr = parse_lc_table(str(lc_path))
            flux = lc_arr[:, 1]
            max_flux = np.max(flux)
            median_flux = np.median(flux)
            assert max_flux > median_flux, f"{cid}: Light curve has no positive peak above median"

    def test_stratified_portfolio_selection_in_atlas(self, tmp_path: Path):
        """Directly tests that ExpeditionAtlas produces a portfolio respecting >= 4 candidates per track."""
        atlas = ExpeditionAtlas(
            cache_dir=tmp_path / "cache",
            results_dir=tmp_path / "results",
            docs_dir=tmp_path / "docs",
            discoveries_dir=tmp_path / "disc",
        )

        # Execute quick expedition in tmp_path with no studio export
        report = atlas.run_expedition(
            quick=True,
            max_targets=30,
            export_catalogs=True,
            generate_dossier=False,
        )

        curated = report.curated_top_discoveries
        assert len(curated) == 15

        tracks = [c["track"] for c in curated]
        assert tracks.count("Track_A") >= 4, f"Track A count < 4: {tracks.count('Track_A')}"
        assert tracks.count("Track_B") >= 4, f"Track B count < 4: {tracks.count('Track_B')}"
        assert tracks.count("Track_C") >= 4, f"Track C count < 4: {tracks.count('Track_C')}"

        # Verify exported files in tmp_path / "disc"
        manifest_file = tmp_path / "disc" / "discoveries_manifest.json"
        assert manifest_file.exists()
        disc_files = list((tmp_path / "disc").glob("*"))
        assert len(disc_files) == 46, f"Expected 46 files in tmp export, got {len(disc_files)}"


# ==============================================================================
# 3. OFFLINE PORTABLE STUDIO HTML DEEP VERIFICATION
# ==============================================================================

class TestOfflinePortableStudioHTML:
    """Rigorous offline validation of results/ and studio/ astrosnail_portable_studio.html."""

    @pytest.mark.parametrize("studio_path", [RESULTS_STUDIO_HTML, STUDIO_STUDIO_HTML])
    def test_file_presence_and_minimum_size(self, studio_path: Path):
        """Studio HTML must exist and have substantial size (> 500 KB for complete self-contained workstation)."""
        assert studio_path.exists(), f"Studio file missing: {studio_path}"
        size_kb = studio_path.stat().st_size / 1024.0
        assert size_kb > 500.0, f"{studio_path.name}: Size {size_kb:.1f} KB is under 500 KB"

    def test_bitwise_sha256_hash_synchronization(self):
        """Both results/ and studio/ copies of astrosnail_portable_studio.html must be 100% bitwise identical."""
        hash_results = hashlib.sha256(RESULTS_STUDIO_HTML.read_bytes()).hexdigest()
        hash_studio = hashlib.sha256(STUDIO_STUDIO_HTML.read_bytes()).hexdigest()
        assert hash_results == hash_studio, (
            f"Bitwise divergence between results/ ({hash_results[:12]}) and studio/ ({hash_studio[:12]})"
        )

    @pytest.mark.parametrize("studio_path", [RESULTS_STUDIO_HTML, STUDIO_STUDIO_HTML])
    def test_zero_external_cdn_references_and_network_calls(self, studio_path: Path):
        """Enforces strictly 0 external CDN links, remote scripts, stylesheets, or network fetch calls."""
        html_text = studio_path.read_text(encoding="utf-8")

        # 1. Check for remote script tags: <script ... src="http..." or src="//..."
        script_srcs = re.findall(r'<script[^>]+src=[\'"]([^\'"]+)[\'"]', html_text, re.IGNORECASE)
        for src in script_srcs:
            assert not src.startswith("http:") and not src.startswith("https:") and not src.startswith("//"), (
                f"{studio_path.name}: Found remote script source: {src}"
            )

        # 2. Check for remote stylesheet links: <link ... href="http..." or href="//..."
        link_hrefs = re.findall(r'<link[^>]+href=[\'"]([^\'"]+)[\'"]', html_text, re.IGNORECASE)
        for href in link_hrefs:
            assert not href.startswith("http:") and not href.startswith("https:") and not href.startswith("//"), (
                f"{studio_path.name}: Found remote stylesheet link: {href}"
            )

        # 3. Comprehensive CDN domain blacklist search
        cdn_domains = [
            "unpkg.com",
            "cdnjs.cloudflare.com",
            "cdn.jsdelivr.net",
            "fonts.googleapis.com",
            "fonts.gstatic.com",
            "ajax.googleapis.com",
            "stackpath.bootstrapcdn.com",
            "maxcdn.bootstrapcdn.com",
            "cdn.skypack.dev",
            "esm.sh",
            "raw.githubusercontent.com",
        ]
        for cdn in cdn_domains:
            matches = list(re.finditer(re.escape(cdn), html_text, re.IGNORECASE))
            assert len(matches) == 0, f"{studio_path.name}: Found prohibited CDN domain '{cdn}' ({len(matches)} times)"

        # 4. Check for network socket / ajax calls targeting remote protocols
        prohibited_remote_apis = [
            re.compile(r'fetch\s*\(\s*[\'"`]https?://', re.IGNORECASE),
            re.compile(r'new\s+WebSocket\s*\(\s*[\'"`]wss?://', re.IGNORECASE),
            re.compile(r'\.open\s*\(\s*[\'"`][A-Z]+[\'"`]\s*,\s*[\'"`]https?://', re.IGNORECASE),
        ]
        for pattern in prohibited_remote_apis:
            matches = list(pattern.finditer(html_text))
            assert len(matches) == 0, f"{studio_path.name}: Found prohibited remote network call matching {pattern.pattern}"

    @pytest.mark.parametrize("studio_path", [RESULTS_STUDIO_HTML, STUDIO_STUDIO_HTML])
    def test_complete_four_quadrant_viewports_and_dom_elements(self, studio_path: Path):
        """Verifies DOM elements for complete 4-quadrant synchronized viewports."""
        html_text = studio_path.read_text(encoding="utf-8")

        required_viewports = ["quad-a", "quad-b", "quad-c", "quad-d"]
        required_canvases = ["canvas-quad-a", "canvas-quad-b", "canvas-quad-c", "canvas-quad-d"]

        for v_id in required_viewports:
            assert f'id="{v_id}"' in html_text or f"id='{v_id}'" in html_text, (
                f"{studio_path.name}: Missing container viewport element id='{v_id}'"
            )

        for c_id in required_canvases:
            assert f'id="{c_id}"' in html_text or f"id='{c_id}'" in html_text, (
                f"{studio_path.name}: Missing canvas element id='{c_id}'"
            )

        # Check HUD / Telemetry container in Quad D
        assert "hud-telemetry" in html_text or "telemetry" in html_text.lower(), (
            f"{studio_path.name}: Missing telemetry display in Quad D"
        )

    @pytest.mark.parametrize("studio_path", [RESULTS_STUDIO_HTML, STUDIO_STUDIO_HTML])
    def test_scenario_selector_and_discovery_options(self, studio_path: Path):
        """Verifies scenario selector dropdown contains canonical discovery scenarios."""
        html_text = studio_path.read_text(encoding="utf-8")

        assert '<select id="scenario-select"' in html_text or "<select id='scenario-select'" in html_text, (
            f"{studio_path.name}: Missing <select id='scenario-select'>"
        )
        assert 'value="discovery_01_circumbinary_transit"' in html_text
        assert 'value="discovery_05_optical_micro_flare"' in html_text
        assert 'value="discovery_09_kbo_crossing_conjunction"' in html_text

    @pytest.mark.parametrize("studio_path", [RESULTS_STUDIO_HTML, STUDIO_STUDIO_HTML])
    def test_built_in_verify_studio_html_utility(self, studio_path: Path):
        """Built-in verify_studio_html() validator must confirm 0 errors."""
        ok, errors = verify_studio_html(studio_path)
        assert ok is True, f"{studio_path.name}: verify_studio_html reported errors: {errors}"
        assert len(errors) == 0, f"{studio_path.name}: Unexpected validation errors: {errors}"


# ==============================================================================
# 4. ADVERSARIAL STRESS & DELIVERABLE IMMUNITY VERIFICATION
# ==============================================================================

class TestAdversarialStressAndDeliverableImmunity:
    """Stress tests out-of-core parsers, telemetry reconciliation, and deliverable immunity."""

    def test_expedition_report_internal_arithmetic_and_telemetry(self):
        """Verifies 2,500 targets, peak heap < 100 MB, clean subprocesses, and exact arithmetic."""
        assert REPORT_JSON_PATH.exists(), f"Missing expedition report at {REPORT_JSON_PATH}"
        with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Metadata
        meta = data.get("expedition_metadata", {})
        assert meta.get("execution_mode") == "full", f"Execution mode is '{meta.get('execution_mode')}', not 'full'"

        # Telemetry
        telemetry = data.get("execution_telemetry", {})
        assert telemetry.get("memory_constraint_satisfied") is True
        assert telemetry.get("zero_orphaned_processes_verified") is True
        assert telemetry.get("peak_memory_mb") < 100.0, f"Peak memory {telemetry.get('peak_memory_mb')} exceeds 100 MB"
        assert telemetry.get("active_subprocesses") == 0

        # Targets
        harvest = data.get("harvest_statistics", {})
        assert harvest.get("total_targets_harvested") == 2500, (
            f"Expected 2500 harvested targets, got {harvest.get('total_targets_harvested')}"
        )
        assert harvest.get("total_epochs_streamed") > 10_000_000

        # Scanned
        scanner = data.get("scanner_outlier_statistics", {})
        assert scanner.get("total_targets_scanned") == 2500
        raw_anomalies = scanner.get("total_raw_anomalies_detected")
        assert raw_anomalies == 4760

        # Vetting Breakdown
        vetting = data.get("vetting_triage_breakdown", {})
        assert vetting.get("total_candidates_vetted") == raw_anomalies

        tier_counts = vetting.get("discovery_tier_counts", {})
        gold = tier_counts.get("Tier_1_Gold", 0)
        silver = tier_counts.get("Tier_2_Silver", 0)
        bronze = tier_counts.get("Tier_3_Bronze", 0)
        rejected = tier_counts.get("Rejected", 0)

        # Arithmetic identity: Gold + Silver + Bronze + Rejected == Total Vetted
        assert gold + silver + bronze + rejected == raw_anomalies, (
            f"Arithmetic mismatch: {gold} + {silver} + {bronze} + {rejected} != {raw_anomalies}"
        )

        # Rejection reasons arithmetic
        reasons = vetting.get("false_positive_reasons", {})
        sum_reasons = sum(reasons.values())
        assert sum_reasons == rejected, f"Rejection reasons sum {sum_reasons} != Rejected {rejected}"

        # Curated Top Discoveries in report
        curated = data.get("curated_top_discoveries", [])
        assert len(curated) == 15
        curated_tracks = [c["track"] for c in curated]
        assert curated_tracks.count("Track_A") >= 4
        assert curated_tracks.count("Track_B") >= 4
        assert curated_tracks.count("Track_C") >= 4

    def test_parser_adversarial_error_handling(self, tmp_path: Path):
        """Stress-tests parse_lc_table and parse_bintable_descriptor with corrupted / edge-case files."""
        # 1. Non-existent file
        with pytest.raises((FileNotFoundError, OSError)):
            parse_lc_table(str(tmp_path / "non_existent.lc"))

        # 2. Empty file
        empty_file = tmp_path / "empty.lc"
        empty_file.write_text("", encoding="utf-8")
        parsed_empty = parse_lc_table(str(empty_file))
        assert isinstance(parsed_empty, np.ndarray)
        assert len(parsed_empty) == 0

        # 3. File with comments only
        comments_only = tmp_path / "comments.lc"
        comments_only.write_text("# header 1\n# header 2\n", encoding="utf-8")
        parsed_comments = parse_lc_table(str(comments_only))
        assert len(parsed_comments) == 0

        # 4. FITS non-2880 byte file
        bad_fits = tmp_path / "corrupt.fits"
        bad_fits.write_bytes(b"A" * 1234)  # Not multiple of 2880
        with pytest.raises((ValueError, OSError)):
            parse_bintable_descriptor(bad_fits)

    def test_test_suite_never_clobbers_production_deliverables(self):
        """Validates that running tests does not alter or dirty production files."""
        # Check current status
        try:
            status_out = subprocess.check_output(
                ["git", "status", "--porcelain", "results/discovery_expedition_report.json"],
                cwd=REPO_ROOT,
                text=True,
            )
            assert status_out.strip() == "", (
                f"results/discovery_expedition_report.json has unstaged modifications: {status_out}"
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
