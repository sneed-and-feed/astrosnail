"""
Unit & Integration Tests for Milestone M7.4: ExpeditionAtlas Engine & Discovery Delivery
========================================================================================
Tests:
  1. Dataclasses & Containers: ExpeditionReport serialization, sanitization, and save_json.
  2. Curated Reference Portfolio: Candidate structure, derived physical parameters, 3-track coverage.
  3. Portfolio Exporter: Verifies exact 7-column .lc, 6-column .cat, binary .fits table, and manifest.
  4. ExpeditionAtlas Coordinator:
     - End-to-end streaming execution in quick mode.
     - Out-of-core memory tracking (< 100 MB active heap ceiling).
     - Subprocess cleanliness (zero orphaned processes).
     - JSON report schema validation.
  5. Functional API & CLI Harness:
     - run_discovery_expedition programmatic function.
     - scripts/run_discovery_expedition.py CLI argument parsing and exit code.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tracemalloc
import numpy as np
import pytest

from astrosnail.expedition import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SEED,
    ExpeditionAtlas,
    ExpeditionReport,
    VettedCandidate,
    create_curated_reference_portfolio,
    run_discovery_expedition,
)


# ==============================================================================
# 1. ExpeditionReport Dataclass & Serialization Tests
# ==============================================================================

def test_expedition_report_serialization(tmp_path: Path):
    """Verifies ExpeditionReport to_dict and save_json sanitization."""
    report = ExpeditionReport(
        expedition_metadata={
            "name": "Test Expedition",
            "seed": np.int64(20261030),
            "float_val": np.float32(3.14159),
            "is_valid": np.bool_(True),
        },
        execution_telemetry={
            "peak_memory_mb": 12.5,
            "zero_orphaned_processes_verified": True,
        },
        harvest_statistics={"total": 60},
        scanner_outlier_statistics={"anomalies": 10},
        vetting_triage_breakdown={"gold": 5, "silver": 5},
        curated_top_discoveries=[{"id": "VET_01", "score": np.float64(0.985)}],
        candidate_catalog=[{"id": "VET_01"}],
    )

    d = report.to_dict()
    assert isinstance(d, dict)
    assert isinstance(d["expedition_metadata"]["seed"], int)
    assert isinstance(d["expedition_metadata"]["float_val"], float)
    assert isinstance(d["expedition_metadata"]["is_valid"], bool)

    json_file = tmp_path / "test_report.json"
    out = report.save_json(json_file)
    assert out.exists()

    with open(out, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["expedition_metadata"]["name"] == "Test Expedition"
    assert loaded["curated_top_discoveries"][0]["score"] == 0.985


# ==============================================================================
# 2. Curated Reference Portfolio Structure Tests
# ==============================================================================

def test_curated_reference_portfolio_tracks_and_parameters():
    """Verifies that the canonical reference portfolio spans Tracks A, B, and C with physical attribution."""
    portfolio = create_curated_reference_portfolio()
    assert len(portfolio) >= 11

    tracks = set(c.track for c in portfolio)
    assert "Track_A" in tracks
    assert "Track_B" in tracks
    assert "Track_C" in tracks

    tiers = set(c.discovery_tier for c in portfolio)
    assert "Tier_1_Gold" in tiers
    assert "Tier_2_Silver" in tiers

    # Check Track A candidate physical parameters
    track_a_cands = [c for c in portfolio if c.track == "Track_A"]
    assert len(track_a_cands) >= 4
    for c in track_a_cands:
        assert c.rp_rstar > 0.0
        assert c.rp_earth > 0.0
        assert c.semi_major_axis_au > 0.0
        assert c.vetting_score >= 0.70

    # Check Track B flare / dipper properties
    track_b_cands = [c for c in portfolio if c.track == "Track_B"]
    assert len(track_b_cands) >= 4
    for c in track_b_cands:
        assert c.amplitude > 0.0 or c.depth_ppm > 0.0
        assert c.vetting_score >= 0.70

    # Check Track C astrometric moving body properties
    track_c_cands = [c for c in portfolio if c.track == "Track_C"]
    assert len(track_c_cands) >= 3
    for c in track_c_cands:
        assert c.delta_centroid_mpix <= 3.0
        assert c.triad_drift == 0.0


# ==============================================================================
# 3. Portfolio Exporter Schema & File Formats Tests
# ==============================================================================

def test_export_discovery_portfolio_files_and_manifest(tmp_path: Path):
    """Verifies export of .lc (7 cols), .cat (6 cols), .fits, and discoveries_manifest.json."""
    atlas = ExpeditionAtlas(discoveries_dir=tmp_path)
    entries = atlas.export_discovery_portfolio(top_n=5, output_dir=tmp_path)

    assert len(entries) == 5
    manifest_file = tmp_path / "discoveries_manifest.json"
    assert manifest_file.exists()

    with open(manifest_file, "r", encoding="utf-8") as f:
        m_data = json.load(f)
    assert m_data["total_discoveries"] == 5

    for entry in entries:
        lc_path = tmp_path / entry["files"]["lc"]
        cat_path = tmp_path / entry["files"]["cat"]
        fits_path = tmp_path / entry["files"]["fits"]

        assert lc_path.exists(), f"Missing LC: {lc_path}"
        assert cat_path.exists(), f"Missing CAT: {cat_path}"
        assert fits_path.exists(), f"Missing FITS: {fits_path}"

        # 1. Verify 7-column LC format
        lc_content = lc_path.read_text(encoding="utf-8")
        assert "# Columns: time flux flux_err bkg centr1 centr2 flag" in lc_content
        data_lines = [ln.strip() for ln in lc_content.splitlines() if not ln.startswith("#") and ln.strip()]
        assert len(data_lines) > 0
        cols_first = data_lines[0].split()
        assert len(cols_first) == 7, f"Expected 7 columns in LC, got: {cols_first}"

        # 2. Verify 6-column CAT format
        cat_content = cat_path.read_text(encoding="utf-8")
        assert "# Columns: ra dec mag pm_ra pm_dec tracklet_id" in cat_content
        cat_lines = [ln.strip() for ln in cat_content.splitlines() if not ln.startswith("#") and ln.strip()]
        assert len(cat_lines) > 0
        cat_first = cat_lines[0].split()
        assert len(cat_first) == 6, f"Expected 6 columns in CAT, got: {cat_first}"

        # 3. Verify FITS format (> 2880 bytes, header block padding)
        assert fits_path.stat().st_size >= 2880
        assert fits_path.stat().st_size % 2880 == 0


# ==============================================================================
# 4. ExpeditionAtlas End-to-End Quick Run & Memory Tracking
# ==============================================================================

def test_expedition_atlas_quick_run(tmp_path: Path):
    """Executes ExpeditionAtlas in quick mode, verifying memory bounds and report schema."""
    cache_dir = tmp_path / "cache"
    results_dir = tmp_path / "results"
    docs_dir = tmp_path / "docs"
    discoveries_dir = tmp_path / "discoveries"

    atlas = ExpeditionAtlas(
        cache_dir=cache_dir,
        results_dir=results_dir,
        docs_dir=docs_dir,
        discoveries_dir=discoveries_dir,
        offline_mode=True,
        seed=20261030,
    )

    tracemalloc.start()
    report = atlas.run_expedition(
        quick=True,
        max_targets=30,
        export_catalogs=True,
        generate_dossier=True,
    )
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_mem / (1024 * 1024)
    assert peak_mb < 100.0, f"Peak memory exceeded 100 MB: {peak_mb:.2f} MB"

    # Verify report sections
    rep_dict = report.to_dict()
    assert "expedition_metadata" in rep_dict
    assert "execution_telemetry" in rep_dict
    assert "harvest_statistics" in rep_dict
    assert "scanner_outlier_statistics" in rep_dict
    assert "vetting_triage_breakdown" in rep_dict
    assert "curated_top_discoveries" in rep_dict
    assert "candidate_catalog" in rep_dict

    assert rep_dict["execution_telemetry"]["memory_constraint_satisfied"] is True
    assert rep_dict["execution_telemetry"]["zero_orphaned_processes_verified"] is True
    assert len(rep_dict["curated_top_discoveries"]) > 0

    # Verify generated files
    report_json = results_dir / "discovery_expedition_report.json"
    assert report_json.exists()


# ==============================================================================
# 5. Programmatic API and CLI Runner Verification
# ==============================================================================

def test_run_discovery_expedition_api(tmp_path: Path):
    """Verifies functional entry point run_discovery_expedition."""
    report = run_discovery_expedition(
        target_count=20,
        quick=True,
        max_targets=20,
        offline_mode=True,
        cache_dir=tmp_path / "cache",
        results_dir=tmp_path / "results",
        docs_dir=tmp_path / "docs",
        discoveries_dir=tmp_path / "discoveries",
        seed=20261030,
    )
    assert isinstance(report, ExpeditionReport)
    assert report.execution_telemetry["memory_constraint_satisfied"] is True


def test_cli_runner_quick_mode(tmp_path: Path):
    """Verifies scripts/run_discovery_expedition.py executes cleanly via CLI with exit code 0."""
    script_path = Path("scripts/run_discovery_expedition.py")
    assert script_path.exists()

    cmd = [
        sys.executable,
        str(script_path),
        "--quick",
        "--offline",
        "--output-json",
        str(tmp_path / "test_cli_report.json"),
        "--discoveries-dir",
        str(tmp_path / "discoveries"),
        "--dossier-md",
        str(tmp_path / "dossier.md"),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--no-export-studio",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    assert res.returncode == 0, f"CLI runner failed with return code {res.returncode}:\n{res.stderr}"
    assert "EXPEDITION STATUS: SUCCESSFUL DELIVERY" in res.stdout
    assert "Zero orphaned processes" in res.stdout
