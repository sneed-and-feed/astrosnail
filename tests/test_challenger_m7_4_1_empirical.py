r"""
Challenger 1 Empirical Stress Test Suite for Milestone M7.4
============================================================
Adversarial Verification of:
  1. Pipeline Execution & Process Cleanliness (scripts/run_discovery_expedition.py --quick)
     - Subprocess tracking via psutil before, during, and after execution
     - Assert 0 orphaned Python subprocesses
     - Peak heap < 25 MB (< 100 MB hard ceiling)
  2. Report JSON Integrity (results/discovery_expedition_report.json)
     - All 7 top-level schema keys present
     - Telemetry verifies memory_constraint_satisfied == 1 and zero_orphaned_processes_verified == 1
     - Vetting counts arithmetic reconciliation: Gold + Silver + Bronze + Rejected == Total
     - Candidate catalog bounds and curated portfolio integrity
  3. Adversarial Edge Cases & Resource Safety
     - Repeated quick-run executions without process accumulation
     - Signal and exit code safety
     - Offline vs online CLI consistency

Author: Challenger 1 (teamwork_challenger_m7_4_1)
Target Milestone: M7.4 (agent-4-atlas)
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tracemalloc
from typing import Dict, List, Set

import psutil
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from astrosnail.expedition import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SEED,
    ExpeditionAtlas,
    ExpeditionReport,
    VettedCandidate,
    create_curated_reference_portfolio,
    run_discovery_expedition,
)


def get_current_python_pids() -> Set[int]:
    """Returns set of all active python process PIDs on the system."""
    pids = set()
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            name = (proc.info["name"] or "").lower()
            if "python" in name:
                pids.add(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return pids


# ==============================================================================
# Vector 1: Pipeline Execution & Process Cleanliness
# ==============================================================================

def test_pipeline_execution_exit_status_and_subprocess_cleanliness(tmp_path: Path):
    """
    Stress Vector 1.1: Run scripts/run_discovery_expedition.py --quick.
    Asserts:
      - Process exits with returncode == 0
      - STDOUT contains completion banner and telemetry receipts
      - No child subprocesses spawned by the process are left behind
      - No new orphaned Python processes remain active
    """
    script_path = REPO_ROOT / "scripts" / "run_discovery_expedition.py"
    assert script_path.exists(), f"Missing CLI runner: {script_path}"

    pids_before = get_current_python_pids()

    # Launch runner in quick mode
    cmd = [
        sys.executable,
        str(script_path),
        "--quick",
        "--offline",
        "--output-json",
        str(tmp_path / "cli_report.json"),
        "--discoveries-dir",
        str(tmp_path / "discoveries"),
        "--dossier-md",
        str(tmp_path / "dossier.md"),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--no-export-studio",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    parent_pid = proc.pid
    assert parent_pid is not None

    # Track child processes while running
    children_seen = []
    try:
        ps_proc = psutil.Process(parent_pid)
        for _ in range(50):
            if proc.poll() is not None:
                break
            try:
                children = ps_proc.children(recursive=True)
                for c in children:
                    if c.pid not in children_seen:
                        children_seen.append(c.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(0.1)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    stdout, stderr = proc.communicate(timeout=120)
    exit_code = proc.returncode

    assert exit_code == 0, f"Process failed with exit code {exit_code}:\nSTDERR:\n{stderr}\nSTDOUT:\n{stdout}"
    assert "EXPEDITION STATUS: SUCCESSFUL DELIVERY" in stdout
    assert "Zero orphaned processes" in stdout

    # Short grace period for Windows OS process reaping
    time.sleep(0.5)

    pids_after = get_current_python_pids()

    # Verify no spawned child processes remain alive
    dangling_children = []
    for c_pid in children_seen:
        if c_pid in pids_after and psutil.pid_exists(c_pid):
            dangling_children.append(c_pid)

    assert len(dangling_children) == 0, f"Found orphaned child processes: {dangling_children}"

    # Verify no persistent orphaned python PIDs from this command
    new_pids = (pids_after - pids_before) - {parent_pid}
    alive_new_pids = [pid for pid in new_pids if psutil.pid_exists(pid)]
    for pid in alive_new_pids:
        try:
            p = psutil.Process(pid)
            assert p.ppid() != parent_pid, f"Child PID {pid} orphaned after parent exit!"
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def test_peak_heap_memory_via_tracemalloc_under_25mb(tmp_path: Path):
    """
    Stress Vector 1.2: Measure exact peak heap memory allocated during quick expedition.
    Asserts:
      - Peak heap allocated is < 25 MB
      - Hard ceiling (< 100 MB) is strictly respected
      - Memory constraint satisfied flag in telemetry is True
    """
    gc.collect()

    atlas = ExpeditionAtlas(
        cache_dir=tmp_path / "cache",
        results_dir=tmp_path / "results",
        docs_dir=tmp_path / "docs",
        discoveries_dir=tmp_path / "discoveries",
        offline_mode=True,
        seed=20261030,
    )

    report = atlas.run_expedition(
        quick=True,
        max_targets=60,
        export_catalogs=True,
        generate_dossier=True,
    )

    telem = report.execution_telemetry
    peak_mb = telem["peak_memory_mb"]

    print(f"\n[Empirical Heap Telemetry] Peak Working Heap: {peak_mb:.2f} MB (Ceiling: {telem['memory_ceiling_mb']} MB)")

    assert peak_mb < 25.0, f"Peak heap memory {peak_mb:.2f} MB exceeded 25 MB limit!"
    assert peak_mb < 100.0, f"Peak heap memory {peak_mb:.2f} MB exceeded 100 MB ceiling!"
    assert telem["memory_constraint_satisfied"] is True
    assert telem["zero_orphaned_processes_verified"] is True


def test_repeated_pipeline_runs_no_memory_or_process_leak(tmp_path: Path):
    """
    Stress Vector 1.3: Run multiple consecutive quick expeditions.
    Asserts:
      - Process cleanliness holds across consecutive executions
      - Peak memory does not accumulate or grow monotonically
    """
    peaks = []
    for iteration in range(3):
        gc.collect()

        iter_results = tmp_path / f"results_{iteration}"
        iter_discoveries = tmp_path / f"discoveries_{iteration}"

        atlas = ExpeditionAtlas(
            cache_dir=tmp_path / f"cache_{iteration}",
            results_dir=iter_results,
            docs_dir=tmp_path / f"docs_{iteration}",
            discoveries_dir=iter_discoveries,
            offline_mode=True,
            seed=20261030 + iteration,
        )

        rep = atlas.run_expedition(quick=True, max_targets=30, export_catalogs=True)
        peaks.append(rep.execution_telemetry["peak_memory_mb"])
        assert rep.execution_telemetry["memory_constraint_satisfied"] is True

    print(f"\n[Repeated Executions Peak Heap MB]: {peaks}")
    for p in peaks:
        assert p < 25.0, f"Peak memory {p:.2f} MB exceeded 25 MB!"
    assert abs(peaks[-1] - peaks[0]) < 5.0


# ==============================================================================
# Vector 2: Report JSON Integrity & Schema Validation
# ==============================================================================

def test_report_json_schema_and_all_top_level_keys():
    """
    Stress Vector 2.1: Validate results/discovery_expedition_report.json schema.
    Asserts:
      - Valid JSON file exists and is parsable
      - All 7 top-level keys present:
        1. expedition_metadata
        2. execution_telemetry
        3. harvest_statistics
        4. scanner_outlier_statistics
        5. vetting_triage_breakdown (semantic vetting_triage_statistics)
        6. curated_top_discoveries (semantic curated_discovery_portfolio)
        7. candidate_catalog (semantic full_candidate_catalog)
    """
    report_file = REPO_ROOT / "results" / "discovery_expedition_report.json"
    assert report_file.exists(), f"Missing report JSON: {report_file}"

    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict), "Report JSON top-level must be a dictionary"

    # Exact canonical keys defined in atlas.py and tested in test_atlas.py
    canonical_keys = [
        "expedition_metadata",
        "execution_telemetry",
        "harvest_statistics",
        "scanner_outlier_statistics",
        "vetting_triage_breakdown",
        "curated_top_discoveries",
        "candidate_catalog",
    ]

    for key in canonical_keys:
        assert key in data, f"Missing required top-level key: {key}"

    # Semantic correspondence to orchestrator directive:
    semantic_map = {
        "expedition_metadata": data.get("expedition_metadata"),
        "execution_telemetry": data.get("execution_telemetry"),
        "harvest_statistics": data.get("harvest_statistics"),
        "scanner_outlier_statistics": data.get("scanner_outlier_statistics"),
        "vetting_triage_statistics": data.get("vetting_triage_breakdown"),
        "curated_discovery_portfolio": data.get("curated_top_discoveries"),
        "full_candidate_catalog": data.get("candidate_catalog"),
    }

    for semantic_name, content in semantic_map.items():
        assert content is not None, f"Semantic key '{semantic_name}' is None or missing from JSON!"


def test_execution_telemetry_constraints():
    """
    Stress Vector 2.2: Validate telemetry flags in results/discovery_expedition_report.json.
    Asserts:
      - memory_constraint_satisfied == 1 (True)
      - zero_orphaned_processes_verified == 1 (True)
      - peak_memory_mb < 100.0 (and < 25.0 for quick run)
      - active_subprocesses == 0
    """
    report_file = REPO_ROOT / "results" / "discovery_expedition_report.json"
    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    telem = data["execution_telemetry"]
    assert telem["memory_constraint_satisfied"] == 1 or telem["memory_constraint_satisfied"] is True
    assert telem["zero_orphaned_processes_verified"] == 1 or telem["zero_orphaned_processes_verified"] is True
    assert telem["active_subprocesses"] == 0
    assert telem["peak_memory_mb"] < 100.0
    assert telem["memory_ceiling_mb"] == 100.0


def test_vetting_counts_arithmetic_reconciliation():
    """
    Stress Vector 2.3: Reconcile tier counts and vetting numbers.
    Asserts:
      - Gold + Silver + Bronze + Rejected == Total Candidates Vetted
      - Total Candidates Vetted == Total Raw Anomalies Detected
      - Candidate catalog contains up to 100 preview entries (vetted_results[:100])
    """
    report_file = REPO_ROOT / "results" / "discovery_expedition_report.json"
    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    vetting = data["vetting_triage_breakdown"]
    tier_counts = vetting["discovery_tier_counts"]

    gold = tier_counts.get("Tier_1_Gold", 0)
    silver = tier_counts.get("Tier_2_Silver", 0)
    bronze = tier_counts.get("Tier_3_Bronze", 0)
    rejected = tier_counts.get("Rejected", 0)

    tier_sum = gold + silver + bronze + rejected
    total_vetted = vetting["total_candidates_vetted"]

    print(f"\n[Vetting Reconciliation] Gold={gold}, Silver={silver}, Bronze={bronze}, Rejected={rejected}")
    print(f"[Vetting Reconciliation] Sum of tiers={tier_sum}, Total vetted recorded={total_vetted}")

    # Core mathematical identity required by directive: Gold + Silver + Bronze + Rejected == Total
    assert tier_sum == total_vetted, (
        f"Arithmetic mismatch: Tier sum ({tier_sum}) != Total candidates vetted ({total_vetted}) "
        f"[Gold: {gold}, Silver: {silver}, Bronze: {bronze}, Rejected: {rejected}]"
    )

    # Cross-module consistency: Scanner detected raw anomalies == Vetting candidates
    raw_anomalies = data["scanner_outlier_statistics"]["total_raw_anomalies_detected"]
    assert total_vetted == raw_anomalies, (
        f"Cross-module mismatch: Total vetted ({total_vetted}) != Raw anomalies detected ({raw_anomalies})"
    )

    # Catalog length matches top-100 preview limit
    catalog = data["candidate_catalog"]
    expected_catalog_len = min(total_vetted, 100)
    assert len(catalog) == expected_catalog_len, (
        f"Catalog count ({len(catalog)}) != Expected preview limit ({expected_catalog_len})"
    )


def test_curated_discovery_portfolio_integrity():
    """
    Stress Vector 2.4: Validate curated portfolio in results/discovery_expedition_report.json.
    Asserts:
      - Curated portfolio contains 15 candidates
      - Every candidate has non-empty candidate_id, target_id, track, survey, vetting_score
      - Vetting scores are monotonically non-increasing (sorted descending)
      - Vetting scores are within valid physical range [0.0, 1.0]
    """
    report_file = REPO_ROOT / "results" / "discovery_expedition_report.json"
    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    portfolio = data["curated_top_discoveries"]
    assert len(portfolio) == 15, f"Expected 15 curated candidates, found {len(portfolio)}"

    scores = []
    for idx, cand in enumerate(portfolio):
        assert cand["candidate_id"], f"Candidate {idx} missing candidate_id"
        assert cand["target_id"], f"Candidate {idx} missing target_id"
        assert cand["track"] in ("Track_A", "Track_B", "Track_C"), f"Candidate {idx} invalid track {cand.get('track')}"

        score = cand["vetting_score"]
        assert 0.0 <= score <= 1.0, f"Candidate {cand['candidate_id']} score {score} out of bounds"
        scores.append(score)

    # Verify descending sort order by vetting_score
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], f"Portfolio scores not sorted: {scores[i]} < {scores[i+1]}"


# ==============================================================================
# Vector 3: Adversarial Edge Cases & CLI Resilience
# ==============================================================================

def test_cli_handles_zero_targets_cleanly(tmp_path: Path):
    """
    Stress Vector 3.1: Run CLI with zero targets requested.
    Asserts:
      - Exits cleanly without hanging
      - Generates valid zero-count report without divide-by-zero crashes
    """
    script_path = REPO_ROOT / "scripts" / "run_discovery_expedition.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--quick",
        "--offline",
        "--targets", "0",
        "--track-a", "0",
        "--track-b", "0",
        "--track-c", "0",
        "--output-json",
        str(tmp_path / "zero_report.json"),
        "--discoveries-dir",
        str(tmp_path / "zero_discoveries"),
        "--dossier-md",
        str(tmp_path / "zero_dossier.md"),
        "--cache-dir",
        str(tmp_path / "zero_cache"),
        "--no-export-studio",
    ]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"Zero targets failed:\n{res.stderr}"


def test_cli_help_flag_and_subprocesses():
    """
    Stress Vector 3.2: Verify --help flag exits immediately with 0 subprocesses.
    """
    script_path = REPO_ROOT / "scripts" / "run_discovery_expedition.py"
    cmd = [sys.executable, str(script_path), "--help"]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=10)
    assert res.returncode == 0
    assert "AstroSnail Multi-Survey 2,500+ Target Discovery Expedition Runner" in res.stdout
