# AstroSnail E2E Test Infrastructure Specification (`TEST_INFRA.md`)

## 1. Test Philosophy & Core Principles

The `astrosnail` end-to-end (E2E) testing infrastructure enforces a strict **requirement-driven, opaque-box (black-box)** verification methodology. All tests operate strictly against published interfaces, mathematical contracts, and public astronomical standards without reliance on private internals or mock shortcuts.

### Core Testing Pillars:
1. **Opaque-Box Verification**: The system is treated as a black box. Detections, trajectory tracking, tomography, and mathematical invariants are evaluated through public module APIs, C-ABI FFI definitions, and CLI/script harnesses.
2. **Authoritative Expected Output Derivation**: Every test case derives expected outputs from explicit, independent authoritative sources:
   - *Mathematical Oracles*: Pure functional derivations of $\mathfrak{so}(C)$ antisymmetry, discrete Morse extrema filtering, and SO(6) Kelvin-Mandel rotor trace conservation.
   - *Public Astronomical Standards*: Real NASA Kepler short-cadence FITS data (HAT-P-7b), 2MASS Galactic Center H-band images, NASA JPL Horizons solar-system ephemerides, and FCRAO 14m L1448 13CO radio spectral line cubes.
   - *Formal Proof Verifiers*: The Lean 4 compiler (`lake build`) verifying machine-checked proofs with 0 `sorry` and 0 custom axioms.
3. **Domain Hygiene Enforcement**: Zero-tolerance scanning across repository files for recycled oil-well logging terms (`caliper`, `washout`, `stringer`, `thin-bed`, `dip rotor`) and fusion reactor terms (`stellarator`, `mrxmhd`, `sub_alfvenic`, `darksnail`).
4. **Zero-Leakage Blind Benchmarking**: Rejection of ground-truth peeking (`sc_data.ground_truth['cosmic_ray_epochs']`, `sc_data.ground_truth['crossing_frame']`, Hungarian matching against `gt_tracks[:, 0, :]`, and perturbation of `ground_truth['transition_matrix']`). Both AstroSnail and comparative baselines must execute on raw, unconditioned streams.
5. **Progressive Testability & Defect Transparency**: Tests evaluate features mapped directly to project milestones. Pending implementation requirements fail cleanly and provide unambiguous defect diagnostic messages for implementation teams.

---

## 2. Test Suite Architecture

The test suite is structured into a 4-tier hierarchy located in `tests/e2e/`:

```
tests/e2e/
├── __init__.py                  # Test package marker
├── conftest.py                  # Shared fixtures, directory pruners, mathematical reference oracles
├── test_tier1_features.py       # Tier 1: Requirement & feature coverage (Features 1–18 + Interface Contracts)
├── test_tier2_boundaries.py     # Tier 2: Boundary, corner cases, extreme SNRs, and dynamic ranges
├── test_tier3_combinations.py   # Tier 3: Cross-feature interactions and multi-module pipelines
└── test_tier4_realworld.py      # Tier 4: Real-world public astronomical standards workloads
```

### Filesystem Scanning Optimization
To prevent infinite traversals over massive package dependency trees (such as Mathlib in `.lake/packages/mathlib` with >50,000 files), all repository scanners utilize `conftest.py:walk_clean_files()`, which dynamically prunes `.git`, `.lake`, `target`, `__pycache__`, `.pytest_cache`, and `build`. This keeps repository audits sub-second (< 0.10s) across thousands of project files.

---

## 3. Test Inventory & Mapping

The suite contains 57 automated tests covering all 18 requirements in `PROJECT.md`:

| Tier | Test Function | Target Feature / Requirement | Authoritative Source | Verification Focus |
|------|---------------|------------------------------|----------------------|--------------------|
| 1 | `test_feature_01_rust_domain_hygiene` | Feature 1 (Rust Purge) | PROJECT.md R1 | Checks `lib.rs`, `allocation_audit.rs`, `astrosnail_core.h` |
| 1 | `test_feature_02_rust_cabi_telemetry_field_rename` | Feature 2 (Telemetry Rename) | PROJECT.md R1/R4 | Verifies `cosmic_ray_spike_recovery_rate` rename |
| 1 | `test_feature_03_rust_stack_overflow_boxing_fix` | Feature 3 (Rust Boxing) | PROJECT.md § Architecture | Verifies heap boxing of 1 MB `AstroSnailArena` |
| 1 | `test_feature_04_rust_clippy_and_std_configuration` | Feature 4 (Rust Cargo.toml) | Cargo.toml | Checks `default = ["std"]` unwinding panic support |
| 1 | `test_feature_05_python_domain_hygiene` | Feature 5 (Python Purge) | PROJECT.md R1 | Scans `astrosnail/` for oil-well / fusion leftovers |
| 1 | `test_feature_06_replace_recycled_python_classes_functions` | Feature 6 (Python Classes) | PROJECT.md R1 | Verifies `SkewSymmetricCosmicTriad` & shock jump |
| 1 | `test_feature_07_python_ffi_synchronization` | Feature 7 (Python FFI) | PROJECT.md § Contracts | Verifies C-ABI telemetry struct size and fields |
| 1 | `test_feature_08_scenario_01_blind_triage` | Feature 8 (Scenario 01 Blind) | PROJECT.md R2 | Eliminates `cr_epochs` peeking & fallback leakage |
| 1 | `test_feature_09_scenario_03_blind_tracking` | Feature 9 (Scenario 03 Blind) | PROJECT.md R2 | Eliminates `crossing_frame` & Hungarian gt matching |
| 1 | `test_feature_10_scenario_04_blind_echo_factorization` | Feature 10 (Scenario 04 Blind) | PROJECT.md R2 | Eliminates `transition_matrix` ground-truth peeking |
| 1 | `test_feature_11_smoke_test_offline_fallback` | Feature 11 (Offline Fallback) | PROJECT.md R2 | Verifies fallback to `real_jpl_horizons_2mass.cat` |
| 1 | `test_feature_12_lean4_domain_hygiene` | Feature 12 (Lean 4 Purge) | PROJECT.md R1/R3 | Scans `formalization/OcclusionTriad.lean` for borehole terms |
| 1 | `test_feature_13_lean4_genuine_astrophysics_theorems` | Feature 13 (Lean 4 Proofs) | PROJECT.md R3 | Verifies transit geometry & astrometric covariance theorems |
| 1 | `test_feature_14_lean4_lake_build_clean` | Feature 14 (Lean 4 Build) | Lean 4 lake compiler | Verifies 0 `sorry` and 0 custom axioms |
| 1 | `test_feature_15_test_suite_stability` | Feature 15 (Test Stability) | Test suite audit | Confirms non-flaky test suite presence |
| 1 | `test_feature_16_documentation_honesty` | Feature 16 (Docs Rewrite) | PROJECT.md R4 | Audits `README.md`, `ARCHITECTURE.md`, `EXPEDITION.md` |
| 1 | `test_feature_17_version_synchronization` | Feature 17 (Version Bump) | Project Manifests | Verifies `v0.2.0` across pyproject, Cargo, and `__init__` |
| 1 | `test_feature_18_release_integrity_assets` | Feature 18 (Release Ready) | Repository Root | Verifies smoke test, evaluate script, and LICENSE |
| 1 | `test_contract_rust_cabi_telemetry_size` | Interface Contract | `_native.py` / C-ABI | Exact 64-byte `AstroSnailTelemetry` struct size |
| 1 | `test_contract_rust_cabi_arena_size` | Interface Contract | `lib.rs` / C-ABI | Exactly 1,012,352 bytes, 64-byte alignment |
| 1 | `test_contract_studio_html_exists` | Interface Contract | `studio/` | Self-contained observatory studio HTML existence |
| 2 | `test_tier2_empty_1d_signal_handling` | Tier 2 (Boundary) | Boundary Contract | 0-length 1D signal handling & exception contracts |
| 2 | `test_tier2_single_sample_stream_update` | Tier 2 (Boundary) | Causal Contract | 1-sample scalar streaming update in `AstroSnailEngine` |
| 2 | `test_tier2_short_2_epoch_sequence` | Tier 2 (Boundary) | Boundary Contract | Handling 2-epoch sequence shorter than 3-sample Morse |
| 2 | `test_tier2_empty_or_flat_2d_focal_plane` | Tier 2 (Boundary) | Astrometry Contract | Flat zero 2D frame in `RoPE2DMedoidPooler` |
| 2 | `test_tier2_single_pixel_2d_frame` | Tier 2 (Boundary) | Astrometry Contract | 1x1 focal plane coordinate bounds |
| 2 | `test_tier2_empty_and_single_skewer_tomography` | Tier 2 (Boundary) | Tomography Contract | Empty and 1-skewer cosmological reconstructions |
| 2 | `test_tier2_coincident_skewer_coordinates` | Tier 2 (Boundary) | Tomography Contract | Coincident skewer coordinates zero-divide safety |
| 2 | `test_tier2_extreme_submerged_snr_minus_30db` | Tier 2 (Extreme SNR) | Signal Processing | -30 dB deep noise floor glint stability |
| 2 | `test_tier2_extreme_clean_snr_plus_100db` | Tier 2 (Extreme SNR) | Signal Processing | +100 dB near-noiseless physical step detection |
| 2 | `test_tier2_extreme_dynamic_ranges` | Tier 2 (Dynamic Range) | Numerical Stability | $10^{-15}$ to $10^{12}$ flux dynamic range stability |
| 2 | `test_tier2_nan_and_inf_input_resilience` | Tier 2 (Sanitization) | Robustness Contract | NaN and Inf input poisoning sanitization |
| 2 | `test_tier2_ring_buffer_boundary_overflow` | Tier 2 (Memory Bounds) | Architecture Contract | 8-frame ring buffer bounded size under 100 updates |
| 2 | `test_tier2_triad_cascade_coupling_extremes` | Tier 2 (Boundary) | Lie Algebra Contract | Coupling=0.0 and Coupling=1.0 norm conservation |
| 2 | `test_tier2_kelvin_mandel_degenerate_rotations` | Tier 2 (Boundary) | SO(6) Contract | Identity and 180° rotation trace conservation |
| 2 | `test_tier2_poincare_beltrami_jet_mu_extremes` | Tier 2 (Boundary) | MHD Contract | $\mu=0.0$ and $\mu=-1.0$ force-free jet flux confinement |
| 2 | `test_tier2_causal_attribution_boundary_matrices` | Tier 2 (Boundary) | Network Contract | All-zero, all-one, and identity contingency matrices |
| 2 | `test_tier2_jordan_cavity_cores_uniform_field` | Tier 2 (Boundary) | Topology Contract | Uniform flat and deep void topological ray casting |
| 2 | `test_tier2_mad_floor_boundary_sizes` | Tier 2 (Boundary) | Noise Floor Contract | Input lengths < 256 and fully masked arrays |
| 2 | `test_tier2_retract_void_boundary_masks` | Tier 2 (Boundary) | Homotopy Contract | Empty void mask and all-True vacuum bound clamps |
| 2 | `test_tier2_so6_kelvin_mandel_matrix_shapes` | Tier 2 (Boundary) | Tensor Contract | (3, 3) tensor and (6, 6) stiffness matrix shapes |
| 3 | `test_tier3_transient_end_to_end_pipeline` | Tier 3 (Pipeline) | Transients Subsystem | Shell -> Morse -> Clock Splitter -> Glitch Gate |
| 3 | `test_tier3_transient_trigger_to_astrometric_centroiding` | Tier 3 (Cross-Feature) | Transient + Astrometry | Light curve spike triggers 2D sub-pixel RoPE medoids |
| 3 | `test_tier3_astrometry_cascade_to_causal_attribution` | Tier 3 (Cross-Feature) | Astrometry + Network | Medoid states -> Slow-Fast Cascade -> Causal Attribution |
| 3 | `test_tier3_cosmology_tomography_to_void_retraction` | Tier 3 (Cross-Feature) | Cosmology Subsystem | Skewers -> Tomography -> Jordan Cores -> Retraction |
| 3 | `test_tier3_rsd_decoupling_to_beltrami_jet` | Tier 3 (Cross-Feature) | Cosmology Subsystem | RSD Decoupling -> Velocity Shear -> Beltrami Jet |
| 3 | `test_tier3_streaming_fits_cube_to_engine_update` | Tier 3 (Streaming) | Engine + FITS I/O | Out-of-core FITS streaming into `AstroSnailEngine` |
| 3 | `test_tier3_multi_modal_io_roundtrip` | Tier 3 (I/O) | I/O Subsystem | Light curve (.lc) and catalog (.cat) roundtrip parity |
| 3 | `test_tier3_engine_theorems_numerical_consistency` | Tier 3 (Formal Invariants) | Lean 4 Theorems | Live validation of Theorems 2.1–2.9 in engine |
| 3 | `test_tier3_multi_dimensional_unified_streaming_session` | Tier 3 (Unified Engine) | Engine Subsystem | 1D series + 2D planes + 3D cubes in single causal run |
| 3 | `test_tier3_scenario_generators_data_structures` | Tier 3 (Benchmark Scenarios) | Benchmark Scenarios | Scenarios 01–05 data streaming structures |
| 4 | `test_tier4_kepler_hatp7b_real_standards` | Tier 4 (Real Archive) | NASA Kepler Archive | 14,280 epochs, HAT-P-7b transit depth (-6200 ppm) |
| 4 | `test_tier4_2mass_galactic_center_real_standards` | Tier 4 (Real Archive) | 2MASS Galactic Center | 720x721 H-band FITS, stellar medoids & RoPE phase |
| 4 | `test_tier4_jpl_horizons_solar_system_tracking` | Tier 4 (Real Archive) | NASA JPL Horizons | Ceres/Vesta/Apophis/Pluto celestial tracking |
| 4 | `test_tier4_l1448_13co_spectral_cube_tomography` | Tier 4 (Real Archive) | FCRAO 14m Radio Archive | 53x105x105 13CO cube, cavity cores & Beltrami jet |
| 4 | `test_tier4_kepler_centroid_astrometric_jitter` | Tier 4 (Real Archive) | NASA Kepler Archive | MOM_CENTR1/2 PSF centroid jitter (< 50 milli-pixels) |
| 4 | `test_tier4_fcrao_moment0_integrated_intensity` | Tier 4 (Real Archive) | FCRAO 14m Radio Archive | Moment-0 velocity channel integrated line emission |

---

## 4. Coverage Thresholds & Quality Gates

The E2E test harness establishes the following hard quality thresholds:

1. **Functional Invariant Thresholds**:
   - Tiers 2, 3, and 4 must maintain a **100% pass rate** at all times (currently 36/36 passing).
   - Radial drift across multi-step Slow-Fast cascade must be strictly **0.000e+00**.
   - Kelvin-Mandel SO(6) rotor trace conservation residual must satisfy $\le 1.14 \times 10^{-13}$.
   - Beltrami jet radial flux escape must be strictly **0.000e+00**.
   - Step-0 void homotopy vacuum bound clamp must satisfy $\delta \ge -1.0$.
2. **Real Astronomical Archive Fidelity**:
   - NASA Kepler HAT-P-7b measured mean transit depth must fall strictly within $[-8000, -4500]\text{ ppm}$.
   - Kepler centroid PSF jitter must not exceed $50\text{ milli-pixels}$.
   - 2MASS Galactic Center RoPE rotary phase invariance error must be $< 10^{-4}$.
3. **Execution Runtime & Memory Budgets**:
   - Full 57-test E2E execution runtime must complete in $< 3.0\text{ seconds}$ (current: 2.01s).
   - Peak active heap for 53-slice 3D FITS streaming must remain $< 2.0\text{ MB}$.
4. **Milestone Transition Gate**:
   - Every completed milestone must transition its corresponding Tier 1 tests from FAILED to PASSED.
   - Milestone 4 (Lean 4 Formalization) has successfully passed all verification gates (Features 12, 13, 14 passing with 0 `sorry` and 0 custom axioms).
   - Final release (v0.2.0) requires 57/57 tests passing (100% green).
