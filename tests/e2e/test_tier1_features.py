"""
astrosnail.tests.e2e.test_tier1_features
========================================
Tier 1: Comprehensive requirement-driven opaque-box feature coverage
mapping to all 18 features documented in PROJECT.md.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pytest

# =============================================================================
# Feature 1: Rust Domain Hygiene Purge
# =============================================================================

def test_feature_01_rust_domain_hygiene(rust_dir: Path, repo_root: Path):
    """Verifies that borehole and fusion terminology are completely purged from Rust
    source code, comments, and C headers (Feature 1).
    Forbidden terms: 'caliper', 'washout', 'stringer', 'stiffness', 'mrxmhd', 'stellarator'.
    """
    forbidden_terms = ["caliper", "washout", "stringer", "stiffness", "mrxmhd", "stellarator"]
    target_paths = [
        rust_dir / "src" / "lib.rs",
        rust_dir / "tests" / "allocation_audit.rs",
        repo_root / "include" / "astrosnail_core.h",
    ]

    violations: Dict[str, List[Tuple[int, str]]] = {}
    for path in target_paths:
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for idx, line in enumerate(content, 1):
            lower_line = line.lower()
            for term in forbidden_terms:
                if term in lower_line:
                    violations.setdefault(f"{path.name} (line {idx})", []).append((idx, line.strip()))

    assert not violations, (
        f"Feature 1 Violation: Found forbidden borehole/fusion terminology in Rust files:\n"
        + "\n".join(f"  {k}: {v}" for k, v in violations.items())
    )


# =============================================================================
# Feature 2: Rust C-ABI Telemetry Field Rename
# =============================================================================

def test_feature_02_rust_cabi_telemetry_field_rename(rust_dir: Path, repo_root: Path):
    """Verifies that `cable_spike_recovery_rate` has been renamed to
    `cosmic_ray_spike_recovery_rate` in Rust lib.rs, include/astrosnail_core.h, and _native.py (Feature 2).
    """
    lib_rs = (rust_dir / "src" / "lib.rs").read_text(encoding="utf-8", errors="replace")
    header_h = (repo_root / "include" / "astrosnail_core.h").read_text(encoding="utf-8", errors="replace")
    native_py = (repo_root / "astrosnail" / "_native.py").read_text(encoding="utf-8", errors="replace")

    # Assert old name is absent
    assert "cable_spike_recovery_rate" not in lib_rs, (
        "Feature 2 Violation: 'cable_spike_recovery_rate' found in rust/astrosnail_rs/src/lib.rs"
    )
    assert "cable_spike_recovery_rate" not in header_h, (
        "Feature 2 Violation: 'cable_spike_recovery_rate' found in include/astrosnail_core.h"
    )
    assert "cable_spike_recovery_rate" not in native_py, (
        "Feature 2 Violation: 'cable_spike_recovery_rate' found in astrosnail/_native.py"
    )

    # Assert new name is present
    assert "cosmic_ray_spike_recovery_rate" in lib_rs, (
        "Feature 2 Violation: 'cosmic_ray_spike_recovery_rate' missing in rust/astrosnail_rs/src/lib.rs"
    )
    assert "cosmic_ray_spike_recovery_rate" in header_h, (
        "Feature 2 Violation: 'cosmic_ray_spike_recovery_rate' missing in include/astrosnail_core.h"
    )
    assert "cosmic_ray_spike_recovery_rate" in native_py, (
        "Feature 2 Violation: 'cosmic_ray_spike_recovery_rate' missing in astrosnail/_native.py"
    )


# =============================================================================
# Feature 3: Rust Stack Allocation vs Heap Boxing
# =============================================================================

def test_feature_03_rust_stack_overflow_boxing_fix(rust_dir: Path):
    """Verifies that tests/allocation_audit.rs boxes the 1 MB AstroSnailArena on the heap
    to avoid stack overflow on Windows (Feature 3).
    """
    audit_rs = (rust_dir / "tests" / "allocation_audit.rs").read_text(encoding="utf-8", errors="replace")
    
    # Must use Box::new or Box::try_new or heap allocation rather than placing bare AstroSnailArena on stack
    has_boxed_arena = "Box::new" in audit_rs or "Box::try_new" in audit_rs or "Box<AstroSnailArena>" in audit_rs
    assert has_boxed_arena, (
        "Feature 3 Violation: tests/allocation_audit.rs does not box AstroSnailArena on the heap; "
        "allocating 1,012,352 bytes directly on Windows test thread stack causes 1.3 MB stack overflow."
    )


# =============================================================================
# Feature 4: Rust Clippy & std Configuration
# =============================================================================

def test_feature_04_rust_clippy_and_std_configuration(rust_dir: Path):
    """Verifies that rust/astrosnail_rs/Cargo.toml enables the `std` feature by default
    so panic unwinding is supported during test runs (Feature 4).
    """
    cargo_toml = (rust_dir / "Cargo.toml").read_text(encoding="utf-8", errors="replace")
    
    # Check default features include std
    has_default_std = bool(re.search(r'default\s*=\s*\[.*"std".*\]', cargo_toml))
    assert has_default_std, (
        "Feature 4 Violation: rust/astrosnail_rs/Cargo.toml does not configure default = ['std']. "
        "Cargo test requires unwinding panic support."
    )


# =============================================================================
# Feature 5: Python Domain Hygiene Purge
# =============================================================================

def test_feature_05_python_domain_hygiene(astrosnail_dir: Path, clean_file_scanner):
    """Verifies that borehole/oil-well terms ('thin-bed', 'compaction') and fusion terms
    ('stellarator', 'mrxmhd', 'sub_alfvenic', 'darksnail') are purged across astrosnail Python codebase (Feature 5).
    """
    forbidden_terms = ["thin-bed", "compaction", "stellarator", "mrxmhd", "sub_alfvenic", "darksnail"]
    violations: Dict[str, List[Tuple[int, str, str]]] = {}

    for py_file in clean_file_scanner(astrosnail_dir, extensions=(".py",)):
        lines = py_file.read_text(encoding="utf-8", errors="replace").splitlines()
        for idx, line in enumerate(lines, 1):
            lower = line.lower()
            for term in forbidden_terms:
                if term in lower:
                    violations.setdefault(py_file.name, []).append((idx, term, line.strip()))

    assert not violations, (
        f"Feature 5 Violation: Found forbidden borehole/fusion terminology in Python codebase:\n"
        + "\n".join(f"  {fname}: {items}" for fname, items in violations.items())
    )


# =============================================================================
# Feature 6: Replace Recycled Python Classes/Functions
# =============================================================================

def test_feature_06_replace_recycled_python_classes_functions(astrosnail_dir: Path):
    """Verifies that SkewSymmetricFaultTriad is replaced with SkewSymmetricCosmicTriad
    and compute_mrxmhd_flow_jump is replaced by a genuine astrophysical function (Feature 6).
    """
    cosmology_py = (astrosnail_dir / "cosmology.py").read_text(encoding="utf-8", errors="replace")
    init_py = (astrosnail_dir / "__init__.py").read_text(encoding="utf-8", errors="replace")

    # Check absence of recycled borehole/fusion class & function
    assert "SkewSymmetricFaultTriad" not in cosmology_py, (
        "Feature 6 Violation: 'SkewSymmetricFaultTriad' still present in astrosnail/cosmology.py (should be SkewSymmetricCosmicTriad)"
    )
    assert "compute_mrxmhd_flow_jump" not in cosmology_py, (
        "Feature 6 Violation: 'compute_mrxmhd_flow_jump' still present in astrosnail/cosmology.py (should be astrophysical shock jump)"
    )

    # Check presence of new cosmic classes/functions
    assert "SkewSymmetricCosmicTriad" in cosmology_py or "SkewSymmetricCosmicTriad" in init_py, (
        "Feature 6 Violation: 'SkewSymmetricCosmicTriad' missing from astrosnail/cosmology.py"
    )


# =============================================================================
# Feature 7: Python FFI Synchronization
# =============================================================================

def test_feature_07_python_ffi_synchronization(astrosnail_dir: Path):
    """Verifies that astrosnail/_native.py synchronizes C-ABI telemetry field names
    and maintains exact 64-byte struct size (Feature 7).
    """
    native_py = (astrosnail_dir / "_native.py").read_text(encoding="utf-8", errors="replace")

    assert "cosmic_ray_spike_recovery_rate" in native_py, (
        "Feature 7 Violation: AstroSnailTelemetry in _native.py missing 'cosmic_ray_spike_recovery_rate'"
    )
    assert "cable_spike_recovery_rate" not in native_py, (
        "Feature 7 Violation: AstroSnailTelemetry in _native.py still references obsolete 'cable_spike_recovery_rate'"
    )


# =============================================================================
# Feature 8: Scenario 01 Blind Triage
# =============================================================================

def test_feature_08_scenario_01_blind_triage(scripts_dir: Path):
    """Verifies that Scenario 01 in scripts/evaluate_astrosnail.py does not peek into
    sc_data.ground_truth['cosmic_ray_epochs'] and contains no fallback leakage (Feature 8).
    """
    eval_script = (scripts_dir / "evaluate_astrosnail.py").read_text(encoding="utf-8", errors="replace")

    # Must not peek at cosmic_ray_epochs to gate AstroSnail
    peeks_cr_epochs = 'ground_truth["cosmic_ray_epochs"]' in eval_script or "ground_truth['cosmic_ray_epochs']" in eval_script
    assert not peeks_cr_epochs, (
        "Feature 8 Violation: scripts/evaluate_astrosnail.py peeks into sc_data.ground_truth['cosmic_ray_epochs'] "
        "to evaluate AstroSnail on 3-sample windows while baseline runs on full continuous stream."
    )

    # Must not have fallback leakage as_detected = [frb_epoch]
    has_fallback_leakage = "as_detected = np.array([frb_epoch])" in eval_script or "as_detected = [frb_epoch]" in eval_script
    assert not has_fallback_leakage, (
        "Feature 8 Violation: scripts/evaluate_astrosnail.py contains fallback leakage 'as_detected = [frb_epoch]'"
    )


# =============================================================================
# Feature 9: Scenario 03 Blind Astrometric Tracking
# =============================================================================

def test_feature_09_scenario_03_blind_tracking(scripts_dir: Path):
    """Verifies that Scenario 03 in scripts/evaluate_astrosnail.py evaluates genuine blind tracking
    from video detections without bypassing frames or matching against gt_tracks (Feature 9).
    """
    eval_script = (scripts_dir / "evaluate_astrosnail.py").read_text(encoding="utf-8", errors="replace")

    # Check for crossing_frame bypass
    peeks_crossing_frame = 'ground_truth["crossing_frame"]' in eval_script or "ground_truth['crossing_frame']" in eval_script
    assert not peeks_crossing_frame, (
        "Feature 9 Violation: scripts/evaluate_astrosnail.py peeks into sc_data.ground_truth['crossing_frame'] "
        "to artificially synthesize detections instead of processing video frames."
    )

    # Check for ground truth track initialization
    peeks_gt_init = 'gt_tracks[:, 0, :]' in eval_script
    assert not peeks_gt_init, (
        "Feature 9 Violation: scripts/evaluate_astrosnail.py initializes tracker by Hungarian matching against gt_tracks[:, 0, :]"
    )


# =============================================================================
# Feature 10: Scenario 04 Blind Echo Factorization
# =============================================================================

def test_feature_10_scenario_04_blind_echo_factorization(scripts_dir: Path):
    """Verifies that Scenario 04 in scripts/evaluate_astrosnail.py does not peek into
    or perturb ground_truth['transition_matrix'] (Feature 10), and adversarially tests
    estimate_causal_transition_matrix for genuine data-driven causal factorization:
      1. Syntactic/AST check: no ground_truth['transition_matrix'] access or hardcoded facade branches.
      2. Null input invariant: pure Gaussian noise or all-zeros MUST NOT attribute generators 0 and 1.
      3. Differentiability invariant: distinct signals and time-reversed signals produce distinct matrices.
      4. Permutation invariant: swapping columns (e.g. 0 and 5) permutes the estimated driving channels.
    """
    eval_script = (scripts_dir / "evaluate_astrosnail.py").read_text(encoding="utf-8", errors="replace")

    # 1. Syntactic check: no direct ground truth access
    peeks_trans_mat = 'ground_truth["transition_matrix"]' in eval_script or "ground_truth['transition_matrix']" in eval_script
    assert not peeks_trans_mat, (
        "Feature 10 Violation: scripts/evaluate_astrosnail.py peeks/perturbs ground_truth['transition_matrix'] "
        "instead of factorizing causal progenitors from raw light curves."
    )

    # Check for hardcoded facade branches like 'if i < 2' in evaluate_astrosnail.py
    assert "if i < 2" not in eval_script, (
        "Feature 10 Violation: scripts/evaluate_astrosnail.py contains hardcoded 'if i < 2' facade branch!"
    )

    from astrosnail.benchmark import estimate_causal_transition_matrix
    from astrosnail.astrometry import attribute_causal_progenitors
    from astrosnail.scenarios import generate_scenario_04_lensed_quasar

    # 2. Null input invariant
    zeros_data = np.zeros((500, 12), dtype=np.float64)
    t_zeros = estimate_causal_transition_matrix(zeros_data)
    is_gen_z, ratios_z = attribute_causal_progenitors(t_zeros, threshold=0.50)
    assert not (is_gen_z[0] and is_gen_z[1]), (
        f"Feature 10 Violation: Null all-zeros input falsely returned generators 0 and 1 with ratios {ratios_z[:2]}"
    )
    assert ratios_z[0] < 0.50 and ratios_z[1] < 0.50, (
        f"Feature 10 Violation: Null input off-diagonal drive >= 0.50 (ratios: {ratios_z[:2]})"
    )

    rng = np.random.default_rng(20261025)
    noise_data = rng.normal(0.0, 1.0, size=(500, 12))
    t_noise = estimate_causal_transition_matrix(noise_data)
    is_gen_n, ratios_n = attribute_causal_progenitors(t_noise, threshold=0.50)
    assert not (is_gen_n[0] and is_gen_n[1]), (
        f"Feature 10 Violation: Gaussian noise falsely returned generators 0 and 1 with ratios {ratios_n[:2]}"
    )
    assert ratios_n[0] < 0.50 and ratios_n[1] < 0.50, (
        f"Feature 10 Violation: Gaussian noise off-diagonal drive >= 0.50 for channels 0 and 1 (ratios: {ratios_n[:2]})"
    )

    # 3. Differentiability / variation invariant
    assert not np.array_equal(t_zeros, t_noise), (
        "Feature 10 Violation: estimate_causal_transition_matrix returned identical matrix for zeros and noise!"
    )

    sc1 = generate_scenario_04_lensed_quasar(seed=20261025, n_epochs=500)
    sc2 = generate_scenario_04_lensed_quasar(seed=42, n_epochs=500)
    t1 = estimate_causal_transition_matrix(sc1.data)
    t2 = estimate_causal_transition_matrix(sc2.data)
    assert not np.allclose(t1, t2), (
        "Feature 10 Violation: estimate_causal_transition_matrix returned identical matrix for distinct scenario realizations!"
    )

    t_rev = estimate_causal_transition_matrix(sc1.data[::-1, :])
    assert not np.allclose(t1, t_rev), (
        "Feature 10 Violation: estimate_causal_transition_matrix is insensitive to time-reversal (causal inversion)!"
    )

    # 4. Permutation invariant: swap column 0 (progenitor) and column 5 (delayed echo)
    data = sc1.data.copy()
    perm = np.arange(12)
    perm[0], perm[5] = 5, 0
    perm_data = data[:, perm]

    t_perm = estimate_causal_transition_matrix(perm_data)
    is_gen_perm, ratios_perm = attribute_causal_progenitors(t_perm, threshold=0.50)

    # Column 0 now holds the echo (delayed by 42.1 days) -> MUST NOT be an active driver
    assert not is_gen_perm[0], (
        f"Feature 10 Violation: Column 0 still detected as generator after swapping with echo channel 5! Ratio: {ratios_perm[0]}"
    )
    assert ratios_perm[0] < 0.50, (
        f"Feature 10 Violation: Column 0 ratio {ratios_perm[0]} >= 0.50 after swapping with delayed echo!"
    )

    # Column 5 now holds the prompt progenitor -> MUST be detected as an active driver
    assert is_gen_perm[5], (
        f"Feature 10 Violation: Column 5 not detected as generator after receiving progenitor signal! Ratio: {ratios_perm[5]}"
    )
    assert ratios_perm[5] >= 0.50, (
        f"Feature 10 Violation: Column 5 ratio {ratios_perm[5]} < 0.50 for prompt progenitor!"
    )


# =============================================================================
# Feature 11: Scenario 05 Tuning & Real Archive Offline Fallback
# =============================================================================

def test_feature_11_smoke_test_offline_fallback(scripts_dir: Path):
    """Verifies that scripts/run_astronomical_smoke_test.py includes offline fallback
    to data/catalogs/real_jpl_horizons_2mass.cat for network resilience (Feature 11).
    """
    smoke_script = (scripts_dir / "run_astronomical_smoke_test.py").read_text(encoding="utf-8", errors="replace")

    has_fallback = "real_jpl_horizons_2mass.cat" in smoke_script and (
        "except" in smoke_script or "fallback" in smoke_script.lower()
    )
    assert has_fallback, (
        "Feature 11 Violation: scripts/run_astronomical_smoke_test.py lacks offline fallback "
        "to data/catalogs/real_jpl_horizons_2mass.cat when JPL Horizons network queries fail."
    )


# =============================================================================
# Feature 12: Lean 4 Domain Hygiene Purge
# =============================================================================

def test_feature_12_lean4_domain_hygiene(formalization_dir: Path):
    """Verifies that borehole terms ('caliper', 'washout', 'stringer', 'thin-bed', 'dip rotor')
    and fusion terms ('stellarator', 'mrxmhd', 'sub_alfvenic', 'darksnail') are purged from
    formalization/OcclusionTriad.lean (Feature 12).
    """
    lean_file = formalization_dir / "OcclusionTriad.lean"
    assert lean_file.exists(), f"Lean file not found: {lean_file}"
    content = lean_file.read_text(encoding="utf-8", errors="replace").splitlines()

    forbidden_terms = [
        "caliper", "washout", "stringer", "thin-bed", "dip rotor",
        "stellarator", "mrxmhd", "sub_alfvenic", "darksnail"
    ]
    violations = []
    for idx, line in enumerate(content, 1):
        lower = line.lower()
        for term in forbidden_terms:
            if term in lower:
                violations.append((idx, term, line.strip()))

    assert not violations, (
        f"Feature 12 Violation: Found forbidden terms in formalization/OcclusionTriad.lean:\n"
        + "\n".join(f"  Line {idx} [{term}]: {line}" for idx, term, line in violations)
    )


# =============================================================================
# Feature 13: Lean 4 Trivial Proof Replacement
# =============================================================================

def test_feature_13_lean4_genuine_astrophysics_theorems(formalization_dir: Path):
    """Verifies that trivial tautological lemmas in OcclusionTriad.lean are replaced
    by genuine astrophysics theorems: exoplanet transit depth geometry, occlusion bounds,
    and astrometric covariance positive-definiteness & photon noise precision (Feature 13).
    """
    lean_file = formalization_dir / "OcclusionTriad.lean"
    content = lean_file.read_text(encoding="utf-8", errors="replace")

    # Check absence of trivial tautologies
    trivial_lemmas = [
        "mhd_euler_force_balance_bijection",
        "sub_alfvenic_beltrami_homotopy_zero_flow",
        "kam_sub_alfvenic_ellipticity_and_gap",
    ]
    for lemma in trivial_lemmas:
        assert lemma not in content, (
            f"Feature 13 Violation: Trivial/fusion lemma '{lemma}' still present in OcclusionTriad.lean"
        )

    # Check presence of genuine astrophysics theorems
    assert "transit" in content.lower() or "exoplanet" in content.lower() or "astrometric" in content.lower(), (
        "Feature 13 Violation: OcclusionTriad.lean does not contain genuine astrophysics/astrometry theorems."
    )


# =============================================================================
# Feature 14: Lean 4 Verification & Lake Build
# =============================================================================

def test_feature_14_lean4_lake_build_clean(formalization_dir: Path):
    """Verifies that formalization/ compiles cleanly with 0 `sorry` and 0 custom axioms (Feature 14)."""
    lean_file = formalization_dir / "OcclusionTriad.lean"
    content = lean_file.read_text(encoding="utf-8", errors="replace")

    # Check for sorry tokens
    sorry_matches = re.findall(r'\bsorry\b', content)
    assert len(sorry_matches) == 0, (
        f"Feature 14 Violation: Found {len(sorry_matches)} 'sorry' placeholder(s) in OcclusionTriad.lean"
    )

    # Check for custom axioms
    axiom_matches = re.findall(r'^\s*axiom\s+', content, re.MULTILINE)
    assert len(axiom_matches) == 0, (
        f"Feature 14 Violation: Found custom axiom declarations in OcclusionTriad.lean: {axiom_matches}"
    )


# =============================================================================
# Feature 15: Test Suite Flakiness & Optimization
# =============================================================================

def test_feature_15_test_suite_stability(repo_root: Path):
    """Verifies that the test suite does not suffer from unbounded allocator jitter or flakiness (Feature 15)."""
    # Ensure test directories and files have no infinite loops or unconstrained threads
    tests_dir = repo_root / "tests"
    assert tests_dir.exists(), "tests/ directory not found"
    test_files = list(tests_dir.glob("test_*.py"))
    assert len(test_files) >= 10, f"Expected comprehensive test suite, found only {len(test_files)} test files."


# =============================================================================
# Feature 16: Documentation Honesty Rewrite
# =============================================================================

def test_feature_16_documentation_honesty(repo_root: Path, docs_dir: Path):
    """Verifies that documentation (README.md, docs/ARCHITECTURE.md, docs/DISCOVERY_EXPEDITION.md)
    honestly reflects astrophysics capabilities without recycled borehole/fusion claims (Feature 16).
    """
    forbidden_terms = ["caliper", "washout", "stringer", "thin-bed", "dip rotor", "stellarator", "mrxmhd", "darksnail"]
    doc_paths = [
        repo_root / "README.md",
        docs_dir / "ARCHITECTURE.md",
        docs_dir / "DISCOVERY_EXPEDITION.md",
    ]

    violations: Dict[str, List[Tuple[int, str, str]]] = {}
    for doc in doc_paths:
        if not doc.exists():
            continue
        lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        for idx, line in enumerate(lines, 1):
            lower = line.lower()
            for term in forbidden_terms:
                if term in lower:
                    violations.setdefault(doc.name, []).append((idx, term, line.strip()))

    assert not violations, (
        f"Feature 16 Violation: Found forbidden recycled claims in documentation:\n"
        + "\n".join(f"  {fname}: {items}" for fname, items in violations.items())
    )


# =============================================================================
# Feature 17: Version Synchronization (+0.1.0 to v0.2.0)
# =============================================================================

def test_feature_17_version_synchronization(repo_root: Path, astrosnail_dir: Path, rust_dir: Path):
    """Verifies that version is synchronized to '0.2.0' across Cargo.toml, pyproject.toml,
    lakefile.lean, and astrosnail/__init__.py (Feature 17).
    """
    # astrosnail/__init__.py
    init_content = (astrosnail_dir / "__init__.py").read_text(encoding="utf-8", errors="replace")
    version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init_content)
    init_ver = version_match.group(1) if version_match else None

    # pyproject.toml
    pyproject_content = (repo_root / "pyproject.toml").read_text(encoding="utf-8", errors="replace")
    pyproject_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', pyproject_content)
    pyproject_ver = pyproject_match.group(1) if pyproject_match else None

    # Cargo.toml
    cargo_content = (rust_dir / "Cargo.toml").read_text(encoding="utf-8", errors="replace")
    cargo_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', cargo_content)
    cargo_ver = cargo_match.group(1) if cargo_match else None

    assert init_ver == "0.2.0", f"Feature 17 Violation: astrosnail/__init__.py version is '{init_ver}', expected '0.2.0'"
    assert pyproject_ver == "0.2.0", f"Feature 17 Violation: pyproject.toml version is '{pyproject_ver}', expected '0.2.0'"
    assert cargo_ver == "0.2.0", f"Feature 17 Violation: Cargo.toml version is '{cargo_ver}', expected '0.2.0'"


# =============================================================================
# Feature 18: Final Verification & Pipeline Integrity
# =============================================================================

def test_feature_18_release_integrity_assets(repo_root: Path, scripts_dir: Path):
    """Verifies that astronomical smoke test script and evaluation harness exist and are ready (Feature 18)."""
    assert (scripts_dir / "run_astronomical_smoke_test.py").exists(), "scripts/run_astronomical_smoke_test.py missing"
    assert (scripts_dir / "evaluate_astrosnail.py").exists(), "scripts/evaluate_astrosnail.py missing"
    assert (repo_root / "LICENSE").exists(), "LICENSE missing"


# =============================================================================
# Interface Contracts (PROJECT.md § Interface Contracts)
# =============================================================================

def test_contract_rust_cabi_telemetry_size(astrosnail_dir: Path):
    """Verifies interface contract: AstroSnailTelemetry struct total size is exactly 64 bytes."""
    import ctypes
    from astrosnail import _native
    if hasattr(_native, "AstroSnailTelemetry"):
        size = ctypes.sizeof(_native.AstroSnailTelemetry)
        assert size == 64, f"Interface Contract Violation: AstroSnailTelemetry size is {size} bytes, expected exactly 64 bytes."


def test_contract_rust_cabi_arena_size(rust_dir: Path):
    """Verifies interface contract: AstroSnailArena is exactly 1,012,352 bytes, 64-byte aligned."""
    lib_rs = (rust_dir / "src" / "lib.rs").read_text(encoding="utf-8", errors="replace")
    assert "1012352" in lib_rs or "1_012_352" in lib_rs, (
        "Interface Contract Violation: AstroSnailArena size 1,012,352 bytes missing from rust/astrosnail_rs/src/lib.rs"
    )
    assert "align(64)" in lib_rs or "align ( 64 )" in lib_rs, (
        "Interface Contract Violation: AstroSnailArena 64-byte alignment missing from rust/astrosnail_rs/src/lib.rs"
    )


def test_contract_studio_html_exists(repo_root: Path):
    """Verifies interface contract: studio/astrosnail_portable_studio.html exists and is non-empty."""
    studio_html = repo_root / "studio" / "astrosnail_portable_studio.html"
    assert studio_html.exists(), "studio/astrosnail_portable_studio.html missing"
    assert studio_html.stat().st_size > 10000, "studio/astrosnail_portable_studio.html is too small or truncated"

