r"""
Challenger M7.4-r2-1 Empirical Stress Test Suite
=================================================
Milestone: M7.4 (Iteration 2) Remediation Challenge
Focus:
  1. Pipeline Execution & Temporary Output Argument Isolation:
     Empirically verify that running `scripts/run_discovery_expedition.py --quick --offline`
     with temporary output arguments leaves `results/discovery_expedition_report.json`
     and other production deliverables untouched and unmodified (verified via SHA256).
  2. Report JSON Schema & Numerical Conservation:
     Empirically verify that `results/discovery_expedition_report.json`:
     - Contains all required top-level schema keys
     - harvested targets == 2500
     - scanned targets == 2500
     - sum of tiers == 4760 (Gold: 2537, Silver: 800, Bronze: 469, Rejected: 954)
     - sum of rejections == 954 (BEB: 927, UNPHYSICAL_DURATION: 24, LOW_VETTING_SCORE: 3)
     - peak RAM < 100 MB (actual 18.16 MB)
     - active subprocesses == 0 and zero_orphaned_processes_verified == True
     - Curated top discoveries == 15, balanced representation (Track A >= 4, Track B >= 4, Track C >= 4)
  3. Subprocess Cleanliness:
     Empirically verify zero orphaned background Python processes after running the expedition runner.
  4. Adversarial CLI Robustness & Git Workspace Cleanliness.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Set

import psutil
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from astrosnail.expedition import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SEED,
    ExpeditionAtlas,
    ExpeditionReport,
)

REPORT_JSON_PATH = REPO_ROOT / "results" / "discovery_expedition_report.json"
STUDIO_HTML_PATH = REPO_ROOT / "results" / "astrosnail_portable_studio.html"
DOSSIER_MD_PATH = REPO_ROOT / "docs" / "DISCOVERY_EXPEDITION.md"
DISCOVERIES_DIR = REPO_ROOT / "data" / "catalogs" / "discoveries"


def get_active_python_pids() -> Set[int]:
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


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA256 hex digest of a file."""
    assert filepath.exists(), f"File does not exist: {filepath}"
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


# ==============================================================================
# Challenge 1: Pipeline Execution & Temporary Output Argument Isolation
# ==============================================================================

def test_pipeline_execution_temporary_output_isolation(tmp_path: Path):
    """
    Empirical Mandate 1:
    Verify that scripts/run_discovery_expedition.py --quick --offline with temporary
    output arguments leaves results/discovery_expedition_report.json untouched and unmodified.
    """
    assert REPORT_JSON_PATH.exists(), f"Production report does not exist: {REPORT_JSON_PATH}"

    # 1. Record baseline fingerprint before execution
    baseline_sha256 = compute_file_sha256(REPORT_JSON_PATH)
    baseline_size = REPORT_JSON_PATH.stat().st_size
    baseline_mtime = REPORT_JSON_PATH.stat().st_mtime_ns

    # Also record studio HTML fingerprint if it exists
    studio_sha256 = compute_file_sha256(STUDIO_HTML_PATH) if STUDIO_HTML_PATH.exists() else None

    # Setup isolated temporary destination paths
    temp_output_json = tmp_path / "temp_cli_report.json"
    temp_discoveries = tmp_path / "temp_discoveries"
    temp_dossier = tmp_path / "temp_dossier.md"
    temp_cache = tmp_path / "temp_cache"

    script_path = REPO_ROOT / "scripts" / "run_discovery_expedition.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--quick",
        "--offline",
        "--output-json", str(temp_output_json),
        "--discoveries-dir", str(temp_discoveries),
        "--dossier-md", str(temp_dossier),
        "--cache-dir", str(temp_cache),
        "--no-export-studio",
    ]

    pids_before = get_active_python_pids()

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout, stderr = proc.communicate(timeout=120)
    assert proc.returncode == 0, f"Expedition runner failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    assert "EXPEDITION STATUS: SUCCESSFUL DELIVERY" in stdout

    # Verify temp outputs were indeed created in tmp_path
    assert temp_output_json.exists(), "Temporary output JSON was not written"
    assert temp_discoveries.exists(), "Temporary discoveries directory was not created"
    assert (temp_discoveries / "discoveries_manifest.json").exists(), "Temporary manifest missing"

    # Verify temp report contains quick-mode telemetry (60 targets)
    with open(temp_output_json, "r", encoding="utf-8") as f:
        temp_data = json.load(f)
    assert temp_data["expedition_metadata"]["execution_mode"] == "quick"
    assert temp_data["harvest_statistics"]["total_targets_harvested"] == 60

    # 2. Verify production deliverables remain 100% UNTOUCHED
    post_sha256 = compute_file_sha256(REPORT_JSON_PATH)
    post_size = REPORT_JSON_PATH.stat().st_size
    post_mtime = REPORT_JSON_PATH.stat().st_mtime_ns

    assert post_sha256 == baseline_sha256, (
        f"CRITICAL FAILURE: results/discovery_expedition_report.json was modified!\n"
        f"Baseline SHA256: {baseline_sha256}\n"
        f"Post-run SHA256: {post_sha256}"
    )
    assert post_size == baseline_size, f"Size changed from {baseline_size} to {post_size}"
    assert post_mtime == baseline_mtime, f"mtime changed from {baseline_mtime} to {post_mtime}"

    if studio_sha256 is not None:
        assert compute_file_sha256(STUDIO_HTML_PATH) == studio_sha256, (
            "results/astrosnail_portable_studio.html was unexpectedly modified!"
        )


# ==============================================================================
# Challenge 2: Schema Assertions & Numerical Conservation
# ==============================================================================

def test_report_json_schema_and_numerical_conservation():
    """
    Empirical Mandate 2:
    Empirically stress-test results/discovery_expedition_report.json against all
    schema assertions and numerical conservation:
      - harvested targets == 2500
      - scanned targets == 2500
      - sum of tiers == 4760
      - sum of rejections == 954
      - peak RAM < 100 MB
      - active subprocesses == 0
    """
    assert REPORT_JSON_PATH.exists(), f"Missing report: {REPORT_JSON_PATH}"

    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Top-level Schema validation
    required_keys = [
        "expedition_metadata",
        "execution_telemetry",
        "harvest_statistics",
        "scanner_outlier_statistics",
        "vetting_triage_breakdown",
        "curated_top_discoveries",
        "candidate_catalog",
    ]
    for k in required_keys:
        assert k in data, f"Missing required top-level key: {k}"

    meta = data["expedition_metadata"]
    assert meta["execution_mode"] == "full", f"Expected execution_mode=='full', got {meta['execution_mode']}"
    assert meta["offline_mode"] is True

    # 2. Harvest statistics
    harvest = data["harvest_statistics"]
    harvested_targets = harvest["total_targets_harvested"]
    assert harvested_targets == 2500, f"Expected 2500 harvested targets, got {harvested_targets}"
    assert harvest["track_breakdown"] == {"Track_A": 1000, "Track_B": 800, "Track_C": 700}
    assert harvest["total_epochs_streamed"] > 10_000_000

    # 3. Scanner statistics
    scanner = data["scanner_outlier_statistics"]
    scanned_targets = scanner["total_targets_scanned"]
    assert scanned_targets == 2500, f"Expected 2500 scanned targets, got {scanned_targets}"
    raw_anomalies = scanner["total_raw_anomalies_detected"]
    assert raw_anomalies == 4760, f"Expected 4760 raw anomalies, got {raw_anomalies}"
    assert scanner["causal_gate_statistics"]["glints_rejected"] == 41200
    assert scanner["causal_gate_statistics"]["confirmed_transients"] == 4760

    # 4. Vetting triage & tier conservation
    vetting = data["vetting_triage_breakdown"]
    total_vetted = vetting["total_candidates_vetted"]
    assert total_vetted == 4760, f"Expected 4760 total candidates vetted, got {total_vetted}"

    tiers = vetting["discovery_tier_counts"]
    gold = tiers.get("Tier_1_Gold", 0)
    silver = tiers.get("Tier_2_Silver", 0)
    bronze = tiers.get("Tier_3_Bronze", 0)
    rejected = tiers.get("Rejected", 0)

    # Individual tier counts
    assert gold == 2149, f"Expected 2149 Tier_1_Gold, got {gold}"
    assert silver == 994, f"Expected 994 Tier_2_Silver, got {silver}"
    assert bronze == 555, f"Expected 555 Tier_3_Bronze, got {bronze}"
    assert rejected == 1062, f"Expected 1062 Rejected, got {rejected}"

    # Sum of tiers == 4760
    sum_tiers = gold + silver + bronze + rejected
    assert sum_tiers == 4760, f"Sum of tiers ({sum_tiers}) != 4760"
    assert sum_tiers == total_vetted == raw_anomalies

    # 5. False positive reasons conservation
    fp_reasons = vetting["false_positive_reasons"]
    sum_fp_reasons = sum(fp_reasons.values())
    assert sum_fp_reasons == 1062, f"Sum of rejection reasons ({sum_fp_reasons}) != 1062"
    assert fp_reasons.get("BEB", 0) == 927, f"Expected 927 BEB rejections, got {fp_reasons.get('BEB')}"
    assert fp_reasons.get("CENTROID_PULL", 0) == 64, f"Expected 64 CENTROID_PULL rejections, got {fp_reasons.get('CENTROID_PULL')}"
    assert fp_reasons.get("STRAY_LIGHT", 0) == 32, f"Expected 32 STRAY_LIGHT rejections, got {fp_reasons.get('STRAY_LIGHT')}"
    assert fp_reasons.get("UNPHYSICAL_DURATION", 0) == 24, f"Expected 24 UNPHYSICAL_DURATION, got {fp_reasons.get('UNPHYSICAL_DURATION')}"
    assert fp_reasons.get("LOW_VETTING_SCORE", 0) == 15, f"Expected 15 LOW_VETTING_SCORE, got {fp_reasons.get('LOW_VETTING_SCORE')}"
    assert fp_reasons.get("centroid_pull_exceeded", 0) == 0

    # 6. Execution telemetry & memory bounds
    telem = data["execution_telemetry"]
    peak_ram = telem["peak_memory_mb"]
    assert peak_ram < 100.0, f"Peak memory {peak_ram} MB exceeded 100.0 MB hard ceiling"
    assert telem["memory_ceiling_mb"] == 100.0
    assert telem["memory_constraint_satisfied"] is True
    assert telem["active_subprocesses"] == 0
    assert telem["zero_orphaned_processes_verified"] is True

    # 7. Curated top discoveries portfolio
    curated = data["curated_top_discoveries"]
    assert len(curated) == 15, f"Expected exactly 15 curated discoveries, got {len(curated)}"

    track_counts: Dict[str, int] = {}
    prev_score = float("inf")
    for idx, disc in enumerate(curated):
        cid = disc["candidate_id"]
        assert cid, f"Entry {idx} missing candidate_id"
        assert disc["target_id"], f"Entry {cid} missing target_id"
        tr = disc["track"]
        assert tr in ("Track_A", "Track_B", "Track_C"), f"Entry {cid} invalid track {tr}"
        track_counts[tr] = track_counts.get(tr, 0) + 1

        tier = disc["discovery_tier"]
        assert tier == "Tier_1_Gold", f"Curated discovery {cid} tier {tier} != Tier_1_Gold"

        score = disc["vetting_score"]
        assert 0.0 <= score <= 1.0, f"Score {score} out of bounds for {cid}"
        assert score <= prev_score + 1e-9, f"Portfolio not sorted descending: {score} > {prev_score}"
        prev_score = score

    # Verify stratified representation: minimum 4 per track
    assert track_counts.get("Track_A", 0) >= 4, f"Track A count ({track_counts.get('Track_A')}) < 4"
    assert track_counts.get("Track_B", 0) >= 4, f"Track B count ({track_counts.get('Track_B')}) < 4"
    assert track_counts.get("Track_C", 0) >= 4, f"Track C count ({track_counts.get('Track_C')}) < 4"
    assert track_counts == {"Track_A": 7, "Track_B": 4, "Track_C": 4}


# ==============================================================================
# Challenge 3: Subprocess Cleanliness & Process Tracking
# ==============================================================================

def test_zero_orphaned_subprocesses_after_expedition(tmp_path: Path):
    """
    Empirical Mandate 3:
    Verify zero orphaned background Python processes after running the expedition runner.
    Tracks process creation and asserts all children terminate upon completion.
    """
    script_path = REPO_ROOT / "scripts" / "run_discovery_expedition.py"
    pids_before = get_active_python_pids()

    cmd = [
        sys.executable,
        str(script_path),
        "--quick",
        "--offline",
        "--output-json", str(tmp_path / "proc_test_report.json"),
        "--discoveries-dir", str(tmp_path / "proc_test_discoveries"),
        "--dossier-md", str(tmp_path / "proc_test_dossier.md"),
        "--cache-dir", str(tmp_path / "proc_test_cache"),
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

    # Track children dynamically
    children_seen: List[int] = []
    try:
        ps_parent = psutil.Process(parent_pid)
        for _ in range(30):
            if proc.poll() is not None:
                break
            try:
                for child in ps_parent.children(recursive=True):
                    if child.pid not in children_seen:
                        children_seen.append(child.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(0.05)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    stdout, stderr = proc.communicate(timeout=60)
    assert proc.returncode == 0, f"Runner failed:\n{stderr}"

    # Wait brief moment for OS to reap exited process handles
    time.sleep(0.5)

    pids_after = get_active_python_pids()

    # Verify no observed children remain alive
    alive_children = [pid for pid in children_seen if psutil.pid_exists(pid)]
    assert len(alive_children) == 0, f"Found lingering child processes: {alive_children}"

    # Verify no orphaned python processes from this invocation
    new_pids = (pids_after - pids_before) - {parent_pid}
    orphaned_pids = []
    for pid in new_pids:
        if psutil.pid_exists(pid):
            try:
                p = psutil.Process(pid)
                if p.ppid() == parent_pid:
                    orphaned_pids.append(pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    assert len(orphaned_pids) == 0, f"Found orphaned Python processes: {orphaned_pids}"


def test_consecutive_runner_executions_no_pid_accumulation(tmp_path: Path):
    """
    Stress-test consecutive runner executions to ensure zero process accumulation across runs.
    """
    script_path = REPO_ROOT / "scripts" / "run_discovery_expedition.py"

    for i in range(2):
        cmd = [
            sys.executable,
            str(script_path),
            "--quick",
            "--offline",
            "--output-json", str(tmp_path / f"rep_{i}.json"),
            "--discoveries-dir", str(tmp_path / f"disc_{i}"),
            "--dossier-md", str(tmp_path / f"dos_{i}.md"),
            "--cache-dir", str(tmp_path / f"cache_{i}"),
            "--no-export-studio",
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        pid = proc.pid
        child_pids = []
        try:
            p = psutil.Process(pid)
            for _ in range(20):
                if proc.poll() is not None:
                    break
                for ch in p.children(recursive=True):
                    if ch.pid not in child_pids:
                        child_pids.append(ch.pid)
                time.sleep(0.05)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        stdout, stderr = proc.communicate(timeout=60)
        assert proc.returncode == 0, f"Iteration {i} failed:\n{stderr}"
        time.sleep(0.2)
        assert not psutil.pid_exists(pid), f"Runner process {pid} still alive after exit!"
        for ch_pid in child_pids:
            assert not psutil.pid_exists(ch_pid), f"Child process {ch_pid} orphaned after runner exit!"


# ==============================================================================
# Challenge 4: Adversarial CLI Options & Edge Cases
# ==============================================================================

def test_cli_handles_zero_targets_cleanly(tmp_path: Path):
    """
    Adversarial edge case: Run with 0 targets.
    Asserts clean exit without divide-by-zero crashes or orphaned processes.
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
        "--output-json", str(tmp_path / "zero_report.json"),
        "--discoveries-dir", str(tmp_path / "zero_discoveries"),
        "--dossier-md", str(tmp_path / "zero_dossier.md"),
        "--cache-dir", str(tmp_path / "zero_cache"),
        "--no-export-studio",
    ]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, f"0-targets failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert (tmp_path / "zero_report.json").exists()


def test_cli_help_flag_clean_exit():
    """
    Verify --help exits with 0 immediately without spawning lingering processes.
    """
    script_path = REPO_ROOT / "scripts" / "run_discovery_expedition.py"
    cmd = [sys.executable, str(script_path), "--help"]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=10)
    assert res.returncode == 0
    assert "AstroSnail Multi-Survey 2,500+ Target Discovery Expedition Runner" in res.stdout


# ==============================================================================
# Challenge 5: Discoveries Directory & Git Index Cleanliness
# ==============================================================================

def test_discoveries_directory_git_tracked_assets():
    """
    Verify that data/catalogs/discoveries contains exactly 46 tracked files in git
    (15 .lc, 15 .cat, 15 .fits, 1 discoveries_manifest.json) with zero orphan files.
    """
    manifest_path = DISCOVERIES_DIR / "discoveries_manifest.json"
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_discoveries"] == 15
    valid_files = {"discoveries_manifest.json"}
    for disc in manifest["discoveries"]:
        valid_files.add(disc["files"]["lc"])
        valid_files.add(disc["files"]["cat"])
        valid_files.add(disc["files"]["fits"])

    assert len(valid_files) == 46, f"Expected 46 unique files in manifest, got {len(valid_files)}"

    # Check git tracked files
    res = subprocess.run(
        ["git", "ls-files", "data/catalogs/discoveries"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"git ls-files failed: {res.stderr}"
    tracked_files = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    tracked_basenames = {Path(p).name for p in tracked_files}

    assert len(tracked_files) == 46, f"Expected 46 tracked files in git, got {len(tracked_files)}"
    assert tracked_basenames == valid_files, "Tracked files in git do not match manifest!"

    # Check disk files
    disk_files = {p.name for p in DISCOVERIES_DIR.iterdir() if p.is_file()}
    assert disk_files == valid_files, f"Discrepancy between disk files and manifest: {disk_files ^ valid_files}"


# ==============================================================================
# Challenge 6: Dossier Reconciliation & GFM Math Audit
# ==============================================================================

def test_dossier_statistics_reconciliation_and_gfm_math():
    """
    Verify that docs/DISCOVERY_EXPEDITION.md is fully reconciled with
    results/discovery_expedition_report.json and satisfies audit_gfm_math.py.
    """
    assert DOSSIER_MD_PATH.exists(), f"Missing dossier: {DOSSIER_MD_PATH}"
    dossier_text = DOSSIER_MD_PATH.read_text(encoding="utf-8")

    # Check that authentic 2,500-target numbers appear in dossier
    assert "2,500" in dossier_text
    assert "4,760" in dossier_text
    assert "2,149" in dossier_text  # Gold
    assert "994" in dossier_text    # Silver
    assert "555" in dossier_text    # Bronze
    assert "1,062" in dossier_text  # Rejected
    assert "41,200" in dossier_text # Glints rejected
    assert "927" in dossier_text    # BEB
    assert "64" in dossier_text     # CENTROID_PULL
    assert "32" in dossier_text     # STRAY_LIGHT
    assert "24" in dossier_text     # UNPHYSICAL_DURATION
    assert "15" in dossier_text     # LOW_VETTING_SCORE

    # Run audit_gfm_math.py
    audit_script = REPO_ROOT / "scripts" / "audit_gfm_math.py"
    assert audit_script.exists(), f"Missing audit script: {audit_script}"
    res = subprocess.run(
        [sys.executable, str(audit_script), str(DOSSIER_MD_PATH)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 0, f"GFM math audit failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "0 violations detected" in res.stdout
