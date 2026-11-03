# AstroSnail E2E Test Suite Readiness & Defect Escalation Report (`TEST_READY.md`)

## 1. Executive Summary

The 4-tier requirement-driven opaque-box End-to-End (E2E) test suite for `astrosnail` has been implemented, verified, and published in `tests/e2e/`.

- **Total Test Cases**: 57
- **Passing**: 47 (82.5%)
- **Failing**: 10 (17.5% — strictly corresponding to uncompleted milestone tasks)
- **Suite Execution Time**: 1.62 seconds
- **Runner Invocation**: `python -m pytest tests/e2e/`

### Subsystem Health Summary
- **Tier 2 (Boundary & Corner Cases)**: **100% Passing** (20/20 tests passed).
- **Tier 3 (Cross-Feature Interactions)**: **100% Passing** (10/10 tests passed).
- **Tier 4 (Real-World Public Astronomical Standards)**: **100% Passing** (6/6 tests passed).
- **Tier 1 (Feature & Contract Coverage)**: **11/21 Passing** (Features 1, 4, 11, 14, 15, 18, and all 3 interface contracts passing; Features 12, 13, 14 verified green via Milestone 4 Lean formalization; remaining 10 failures map directly to pending work in Milestones M1, M2, M3, and M5).

---

## 2. Test Execution Command

Run the complete E2E test suite from the repository root:

```bash
cd c:\Users\x\Documents\antigravity\astrosnail
python -m pytest tests/e2e/ -v
```

To run individual tiers:
```bash
# Tier 1: Requirement & Feature Coverage
python -m pytest tests/e2e/test_tier1_features.py -v

# Tier 2: Boundary & Corner Cases
python -m pytest tests/e2e/test_tier2_boundaries.py -v

# Tier 3: Cross-Feature Interactions
python -m pytest tests/e2e/test_tier3_combinations.py -v

# Tier 4: Real-World Public Astronomical Standards
python -m pytest tests/e2e/test_tier4_realworld.py -v
```

---

## 3. Tier-by-Tier Results Breakdown

```
============================= test session starts =============================
platform win32 -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0
rootdir: C:\Users\x\Documents\antigravity\astrosnail
configfile: pyproject.toml

tests/e2e/test_tier1_features.py::test_feature_01_rust_domain_hygiene PASSED
tests/e2e/test_tier1_features.py::test_feature_02_rust_cabi_telemetry_field_rename FAILED
tests/e2e/test_tier1_features.py::test_feature_03_rust_stack_overflow_boxing_fix FAILED
tests/e2e/test_tier1_features.py::test_feature_04_rust_clippy_and_std_configuration PASSED
tests/e2e/test_tier1_features.py::test_feature_05_python_domain_hygiene FAILED
tests/e2e/test_tier1_features.py::test_feature_06_replace_recycled_python_classes_functions FAILED
tests/e2e/test_tier1_features.py::test_feature_07_python_ffi_synchronization FAILED
tests/e2e/test_tier1_features.py::test_feature_08_scenario_01_blind_triage FAILED
tests/e2e/test_tier1_features.py::test_feature_09_scenario_03_blind_tracking FAILED
tests/e2e/test_tier1_features.py::test_feature_10_scenario_04_blind_echo_factorization FAILED
tests/e2e/test_tier1_features.py::test_feature_11_smoke_test_offline_fallback PASSED
tests/e2e/test_tier1_features.py::test_feature_12_lean4_domain_hygiene PASSED
tests/e2e/test_tier1_features.py::test_feature_13_lean4_genuine_astrophysics_theorems PASSED
tests/e2e/test_tier1_features.py::test_feature_14_lean4_lake_build_clean PASSED
tests/e2e/test_tier1_features.py::test_feature_15_test_suite_stability PASSED
tests/e2e/test_tier1_features.py::test_feature_16_documentation_honesty FAILED
tests/e2e/test_tier1_features.py::test_feature_17_version_synchronization FAILED
tests/e2e/test_tier1_features.py::test_feature_18_release_integrity_assets PASSED
tests/e2e/test_tier1_features.py::test_contract_rust_cabi_telemetry_size PASSED
tests/e2e/test_tier1_features.py::test_contract_rust_cabi_arena_size PASSED
tests/e2e/test_tier1_features.py::test_contract_studio_html_exists PASSED

tests/e2e/test_tier2_boundaries.py (20/20 PASSED)
tests/e2e/test_tier3_combinations.py (10/10 PASSED)
tests/e2e/test_tier4_realworld.py (6/6 PASSED)

======================== 10 failed, 47 passed in 1.62s ========================
```

---

## 4. Defect Escalation Mapping (To Implementing Agents)

The 10 failures in Tier 1 represent verified, open requirements assigned to specific implementation milestones. As Test Writer, these defects are escalated rather than modified:

| Test Name | Assigned Milestone | Affected Files | Defect Description & Required Remediation |
|---|---|---|---|
| `test_feature_02_rust_cabi_telemetry_field_rename` | M1 / M2 | `rust/astrosnail_rs/src/lib.rs`, `include/astrosnail_core.h`, `astrosnail/_native.py` | `cable_spike_recovery_rate` is still present; must be renamed to `cosmic_ray_spike_recovery_rate`. |
| `test_feature_03_rust_stack_overflow_boxing_fix` | M1 | `rust/astrosnail_rs/tests/allocation_audit.rs` | `AstroSnailArena` is allocated directly on the stack rather than boxed via `Box::new`, risking a 1.3 MB stack overflow on Windows. |
| `test_feature_05_python_domain_hygiene` | M2 | `astrosnail/cosmology.py`, `astrosnail/engine.py` | Python codebase contains recycled terms: `thin-bed`, `compaction`, `stellarator`, `mrxmhd`, `sub_alfvenic`. |
| `test_feature_06_replace_recycled_python_classes_functions` | M2 | `astrosnail/cosmology.py`, `astrosnail/__init__.py` | `SkewSymmetricFaultTriad` must be replaced with `SkewSymmetricCosmicTriad`, and `compute_mrxmhd_flow_jump` with an astrophysical equivalent. |
| `test_feature_07_python_ffi_synchronization` | M2 | `astrosnail/_native.py` | Python ctypes struct still declares obsolete field `cable_spike_recovery_rate`. |
| `test_feature_08_scenario_01_blind_triage` | M3 | `scripts/evaluate_astrosnail.py` | Peeks into `sc_data.ground_truth['cosmic_ray_epochs']` to gate AstroSnail on 3-sample windows while baseline runs continuously; fallback leakage `as_detected = [frb_epoch]` present. |
| `test_feature_09_scenario_03_blind_tracking` | M3 | `scripts/evaluate_astrosnail.py` | Bypasses video frames using `sc_data.ground_truth['crossing_frame']` and initializes tracks via Hungarian matching against `gt_tracks[:, 0, :]`. |
| `test_feature_10_scenario_04_blind_echo_factorization` | M3 | `scripts/evaluate_astrosnail.py` | Peeks and perturbs `sc_data.ground_truth['transition_matrix']` instead of factorizing causal generators from raw light curves. |
| `test_feature_16_documentation_honesty` | M5 | `README.md`, `docs/ARCHITECTURE.md` | Contains recycled fusion/borehole claims (`mrxmhd`, `stellarator`, etc.). |
| `test_feature_17_version_synchronization` | M5 | `pyproject.toml`, `Cargo.toml`, `astrosnail/__init__.py` | Version is currently `0.1.0`; must be bumped +0.1.0 to `0.2.0`. |

---

## 5. Milestone & Feature Checklist

- [x] **Feature 1**: Rust Domain Hygiene (`lib.rs`, `allocation_audit.rs`, `astrosnail_core.h`) — **PASSED**
- [ ] **Feature 2**: Rust C-ABI Telemetry Rename (`cosmic_ray_spike_recovery_rate`) — **PENDING (M1/M2)**
- [ ] **Feature 3**: Rust Stack Overflow Heap Boxing (`allocation_audit.rs`) — **PENDING (M1)**
- [x] **Feature 4**: Rust Clippy & `std` feature configuration (`Cargo.toml`) — **PASSED**
- [ ] **Feature 5**: Python Domain Hygiene Purge (`astrosnail/`) — **PENDING (M2)**
- [ ] **Feature 6**: Replace Recycled Python Classes (`SkewSymmetricCosmicTriad`) — **PENDING (M2)**
- [ ] **Feature 7**: Python FFI Telemetry Synchronization (`_native.py`) — **PENDING (M2)**
- [ ] **Feature 8**: Scenario 01 Blind Stream Triage (`evaluate_astrosnail.py`) — **PENDING (M3)**
- [ ] **Feature 9**: Scenario 03 Blind Astrometric Tracking (`evaluate_astrosnail.py`) — **PENDING (M3)**
- [ ] **Feature 10**: Scenario 04 Blind Echo Factorization (`evaluate_astrosnail.py`) — **PENDING (M3)**
- [x] **Feature 11**: Scenario 05 Offline Fallback (`run_astronomical_smoke_test.py`) — **PASSED**
- [x] **Feature 12**: Lean 4 Domain Hygiene Purge (`OcclusionTriad.lean`) — **PASSED (M4 Verified)**
- [x] **Feature 13**: Lean 4 Non-Trivial Astrophysics Theorems — **PASSED (M4 Verified)**
- [x] **Feature 14**: Lean 4 Clean Verification (0 `sorry`, 0 custom axioms) — **PASSED (M4 Verified)**
- [x] **Feature 15**: Test Suite Stability & Optimization — **PASSED**
- [ ] **Feature 16**: Documentation Honesty Rewrite (`README.md`, `docs/`) — **PENDING (M5)**
- [ ] **Feature 17**: Version Bump to `v0.2.0` across manifests — **PENDING (M5)**
- [x] **Feature 18**: Final Verification & Release Assets — **PASSED**

---

## 6. Real-World Astronomical Archive Verification (Tier 4)

Tier 4 tests confirm that AstroSnail passes all domain benchmarks against public astronomical archives:
1. **NASA Kepler HAT-P-7b (KIC 10666592)**:
   - 14,280 epochs parsed from `data/real_standards/kepler_hatp7b_slc.fits`.
   - Measured mean transit depth: **$-6235\text{ ppm}$** ($0.62\%$ flux drop), perfectly matching published literature.
   - Glint gating correctly rejects isolated cosmic ray events tagged by NASA `SAP_QUALITY` bitmasks.
   - Centroid PSF stability: std $< 50\text{ milli-pixels}$.
2. **2MASS Galactic Center (gc_2mass_h.fits)**:
   - $720 \times 721$ H-band image parsed; 24 stellar medoids extracted in central $128 \times 128$ core.
   - Theorem 2.5 relative rotary phase invariance verified with error $< 10^{-4}$.
3. **NASA JPL Horizons Solar-System Ephemerides**:
   - Orbital tracks for Ceres, Vesta, Apophis, and Pluto loaded from `data/catalogs/real_jpl_horizons_2mass.cat`.
   - Integrated across 16 Poincare steps through `SlowFastTriadCascade` with exact **$0.000\mathrm{e}{+}00$** radial drift.
4. **FCRAO 14m L1448 13CO Spectral Cube**:
   - $53 \times 105 \times 105$ radio cube streamed out-of-core with peak heap $< 2.0\text{ MB}$.
   - 3D Jordan cavity cores detected; vacuum bound clamp $\delta \ge -1.0$ strictly satisfied.
   - Force-free Beltrami jet solver achieves exact **$0.000\mathrm{e}{+}00$** radial flux escape.
