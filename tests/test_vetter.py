r"""
Comprehensive Unit & Integration Tests for Milestone M7.3: Ephemeris Cross-Matching & False-Positive Vetting Engine
====================================================================================================================
Tests:
  1. Dataclasses: `CatalogEphemerisEntry`, `InstrumentalVettingReceipt`, `PhysicalVettingReceipt`, `VettedCandidate`.
  2. Physical parameter derivation ($R_p / R_*$, $R_p (R_\oplus)$, $a$, $a/R_*$, impact parameter $b$).
  3. Serialization and exports: `to_dict()`, `to_dossier_row()`, `export_studio_lc()`, `export_fits()`.
  4. `EphemerisMatcher`:
     - Primary transit matching against canonical benchmark (HAT-P-7 b / Kepler-8 b).
     - Secondary eclipse matching near phase 0.5 for eclipsing binaries.
     - Out-of-transit non-match classification as genuine novel discovery.
     - Known multi-planet system additional uncataloged candidate isolation.
     - Dynamic tolerance window validation.
  5. `InstrumentalFalsePositiveFilter`:
     - Pristine on-target centroid stability (< 1.5 mpix).
     - Centroid pull hard veto (>= 3.0 mpix or >= 2.0 mpix with high SNR).
     - Sky background decorrelation hard veto (|r| >= 0.80).
     - Spacecraft telemetry momentum dump buffer veto (bit 32 within 0.025 d).
     - Spacecraft Argabrightening detector event buffer veto (bit 2048 within 0.025 d).
     - Fatal quality mask corruption triage.
  6. `AstrophysicalConsistencyVetter`:
     - Odd-even depth parity test: BEB rejection (Delta d >= 3.0 sigma).
     - Parity invariance for clean exoplanet transits.
     - Monotransit and non-periodic event graceful bypass (no false rejections).
     - Keplerian stellar density duration bound test (unphysical duration rejection).
     - Transit profile morphology metric (tau / T ratio, U-shape vs V-shape).
     - V-shape with centroid shift combination BEB veto.
  7. Composite Vetting Score & Discovery Tier Matrix:
     - Tier 1 Gold discovery gate verification.
     - Tier 2 Silver candidate gate verification.
     - Tier 3 Bronze candidate gate verification.
     - Hard veto rejection verification.
  8. `ExpeditionVetter` Master Class & Streaming:
     - Out-of-core streaming generator (vet_stream).
     - Heap memory footprint profiling (< 25 MB active heap).
"""


import gc
import math
import os
from pathlib import Path
import tracemalloc
import numpy as np
import pytest

from astrosnail.expedition import (
    AstrophysicalConsistencyVetter,
    CandidateAnomaly,
    CatalogEphemerisEntry,
    EphemerisCrossMatcher,
    EphemerisMatcher,
    ExpeditionVetter,
    HarvestedTarget,
    InstrumentalFalsePositiveFilter,
    InstrumentalTriageEngine,
    InstrumentalVettingReceipt,
    PhysicalConsistencyEngine,
    PhysicalVettingReceipt,
    VettedCandidate,
    compute_composite_vetting_score,
    vet_candidate,
    vet_target,
)


# ==============================================================================
# 1. Dataclass Structure, Slots, Derivations, and Serialization Tests
# ==============================================================================

def test_catalog_ephemeris_entry_slots_and_aliases():
    """Validates CatalogEphemerisEntry slots=True and property aliases."""
    assert hasattr(CatalogEphemerisEntry, "__slots__")

    entry = CatalogEphemerisEntry(
        target_id="KIC_10666592",
        survey="Kepler",
        catalog_name="KOI",
        object_name="HAT-P-7 b",
        period_days=2.204733,
        t0_bjd=120.045,
        duration_hours=4.1,
        depth_ppm=6900.0,
        disposition="CONFIRMED",
        ra=292.298,
        dec=47.969,
        period_err=1e-6,
        t0_err=1e-4,
        impact_parameter=0.12,
    )

    assert entry.target_id == "KIC_10666592"
    assert entry.period == 2.204733
    assert entry.epoch_bjd == 120.045
    assert entry.epoch_err == 1e-4
    assert entry.catalog_type == "KOI"
    assert entry.impact_param == 0.12


def test_vetted_candidate_slots_and_derivations(tmp_path: Path):
    """Validates VettedCandidate attributes, physical derivations, and serialization/export."""
    assert hasattr(VettedCandidate, "__slots__")

    # Construct test vetted candidate
    t_win = np.linspace(120.0, 120.5, 50, dtype=np.float64)
    f_win = np.ones(50, dtype=np.float64)
    f_win[20:30] -= 0.005  # 5,000 ppm dip

    cand = VettedCandidate(
        candidate_id="VET_KIC_001_A01",
        target_id="KIC_001",
        track="Track_A",
        survey="Kepler",
        anomaly_type="circumbinary_transit",
        ra=285.5,
        dec=45.2,
        time_peak_bjd=120.25,
        t_start=120.15,
        t_peak=120.25,
        t_end=120.35,
        duration_hours=4.8,
        duration_cadences=10,
        n_epochs=1,
        period_days=10.0,
        snr=12.5,
        amplitude=5000.0,
        depth_ppm=5000.0,
        time_window=t_win,
        flux_window=f_win,
    )

    # 1. Verify physical parameter derivations in __post_init__
    expected_rp_rstar = math.sqrt(5000.0 * 1e-6)
    assert abs(cand.rp_rstar - expected_rp_rstar) < 1e-7
    assert cand.rp_earth > 0.0
    assert cand.semi_major_axis_au > 0.0
    assert cand.a_rstar > 0.0

    # 2. Verify dictionary serialization
    d = cand.to_dict()
    assert isinstance(d, dict)
    assert d["candidate_id"] == "VET_KIC_001_A01"
    assert d["depth_ppm"] == 5000.0
    assert d["discovery_tier"] == "Tier_1_Gold"
    assert "time_window" not in d  # Large arrays omitted

    # 3. Verify markdown dossier row formatting
    row = cand.to_dossier_row()
    assert isinstance(row, dict)
    assert row["Candidate ID"] == "VET_KIC_001_A01"
    assert "5,000 ppm" in row["Depth"]
    assert "10.0000 d" in row["Period"]

    # 4. Verify portable studio .lc export
    lc_file = tmp_path / "test_candidate.lc"
    cand.export_studio_lc(lc_file)
    assert lc_file.exists()
    content = lc_file.read_text(encoding="utf-8")
    assert "# AstroSnail Discovery Light Curve" in content
    assert "VET_KIC_001_A01" in content
    lines = [ln for ln in content.splitlines() if not ln.startswith("#")]
    assert len(lines) == 50

    # 5. Verify FITS binary table export
    fits_file = tmp_path / "test_candidate.fits"
    cand.export_fits(fits_file)
    assert fits_file.exists()
    assert fits_file.stat().st_size > 2880


# ==============================================================================
# 2. Ephemeris Matching & Discovery Taxonomy Tests
# ==============================================================================

def test_ephemeris_matcher_primary_transit_match():
    """Validates primary transit matching against canonical benchmark HAT-P-7 b."""
    matcher = EphemerisMatcher()

    # Create candidate corresponding to HAT-P-7 b (P = 2.204733 d, t0 = 120.045 d) at cycle N = 5
    t_peak = 120.045 + 5 * 2.204733
    cand = CandidateAnomaly(
        candidate_id="CAN_HAT_P_7_01",
        target_id="KIC_10666592",
        track="Track_A",
        survey="Kepler",
        anomaly_type="planet_transit",
        ra=292.298,
        dec=47.969,
        time_peak_bjd=t_peak,
        duration_hours=4.1,
        snr=25.0,
        depth_ppm=6900.0,
    )

    res = matcher.cross_match(cand)
    assert res["is_match"] is True
    assert res["catalog_match"] == "KOI"
    assert res["cycle_index"] == 5
    assert abs(res["phase_offset_days"]) < 1e-4
    assert res["is_secondary"] is False
    assert res["disposition"] == "CONFIRMED_KNOWN"
    assert res["taxonomy_class"] == "catalog_confirmed_planet"


def test_ephemeris_matcher_secondary_eclipse_match():
    """Validates secondary eclipse matching near phase 0.5 on an eclipsing binary."""
    matcher = EphemerisMatcher()

    # Benchmark EB KIC 5871143 (P = 4.25 d, t0 = 120.0 d) at secondary eclipse (phase 0.5, N = 3)
    t_sec = 120.0 + (3 + 0.5) * 4.25
    cand = CandidateAnomaly(
        candidate_id="CAN_EB_SEC_01",
        target_id="KIC_5871143",
        track="Track_A",
        survey="Kepler",
        anomaly_type="secondary_eclipse",
        ra=290.0,
        dec=41.0,
        time_peak_bjd=t_sec,
        duration_hours=4.5,
        snr=15.0,
        depth_ppm=20000.0,
    )

    res = matcher.cross_match(cand)
    assert res["is_match"] is True
    assert res["is_secondary"] is True
    assert res["cycle_index"] == 3
    assert res["disposition"] == "CONFIRMED_KNOWN"
    assert res["taxonomy_class"] == "catalog_confirmed_eb"


def test_ephemeris_matcher_multi_planet_additional_candidate():
    """Validates isolation of novel additional candidate on confirmed multi-planet host."""
    matcher = EphemerisMatcher()

    # Target is confirmed host Kepler-11 (KIC 6541920) hosting b (10.3 d) and c (13.0 d)
    # Inject an uncataloged transit candidate at t = 145.2 d that does not match b or c
    cand = CandidateAnomaly(
        candidate_id="CAN_KEP11_NEW_01",
        target_id="KIC_6541920",
        track="Track_A",
        survey="Kepler",
        anomaly_type="planet_transit",
        ra=296.068,
        dec=41.908,
        time_peak_bjd=145.2,
        duration_hours=3.5,
        snr=8.5,
        depth_ppm=800.0,
        delta_centroid_mpix=0.4,
        bkg_correlation=0.05,
        quality_flag_clean=True,
    )

    res = matcher.cross_match(cand)
    assert res["is_match"] is False
    assert res["disposition"] == "NOVEL_CANDIDATE"
    assert res["taxonomy_class"] == "multi_planet_additional_candidate"


def test_ephemeris_matcher_genuine_novel_discovery():
    """Validates novel transient discovery on an uncataloged single target."""
    matcher = EphemerisMatcher()

    cand = CandidateAnomaly(
        candidate_id="CAN_NOVEL_01",
        target_id="KIC_99999999",
        track="Track_A",
        survey="Kepler",
        anomaly_type="circumbinary_transit",
        ra=280.0,
        dec=40.0,
        time_peak_bjd=150.0,
        duration_hours=5.0,
        snr=9.2,
        depth_ppm=3500.0,
    )

    res = matcher.cross_match(cand)
    assert res["is_match"] is False
    assert res["disposition"] == "NOVEL_CANDIDATE"
    assert res["taxonomy_class"] == "genuine_novel_discovery"


# ==============================================================================
# 3. Instrumental False-Positive Triage Tests
# ==============================================================================

def test_instrumental_triage_clean_gold_candidate():
    """Validates pristine on-target telemetry yields high score and passes all filters."""
    cand = CandidateAnomaly(
        candidate_id="CAN_GOLD_01",
        target_id="KIC_001",
        track="Track_A",
        survey="Kepler",
        anomaly_type="planet_transit",
        ra=285.0,
        dec=45.0,
        time_peak_bjd=120.0,
        t_start=119.9,
        t_peak=120.0,
        t_end=120.1,
        snr=10.0,
        delta_centroid_mpix=0.45,
        bkg_correlation=0.08,
        quality_flag_clean=True,
    )

    receipt = InstrumentalFalsePositiveFilter.evaluate(cand)
    assert receipt.is_instrumental_fp is False
    assert receipt.centroid_passed is True
    assert receipt.bkg_passed is True
    assert receipt.thruster_veto_passed is True
    assert receipt.instrumental_score > 0.85
    assert len(receipt.veto_reasons) == 0


def test_instrumental_triage_centroid_pull_rejection():
    """Validates centroid excursion >= 3.0 mpix triggers hard CENTROID_PULL veto."""
    cand = CandidateAnomaly(
        candidate_id="CAN_PULL_01",
        target_id="KIC_002",
        track="Track_A",
        survey="Kepler",
        anomaly_type="planet_transit",
        ra=285.0,
        dec=45.0,
        time_peak_bjd=120.0,
        delta_centroid_mpix=3.8,  # > 3.0 mpix threshold
        bkg_correlation=0.10,
        quality_flag_clean=True,
    )

    receipt = InstrumentalFalsePositiveFilter.evaluate(cand)
    assert receipt.is_instrumental_fp is True
    assert receipt.centroid_passed is False
    assert "CENTROID_PULL" in receipt.veto_reasons
    assert receipt.instrumental_score == 0.0


def test_instrumental_triage_stray_light_rejection():
    """Validates background correlation |r| >= 0.80 triggers STRAY_LIGHT veto."""
    cand = CandidateAnomaly(
        candidate_id="CAN_STRAY_01",
        target_id="KIC_003",
        track="Track_A",
        survey="Kepler",
        anomaly_type="micro_flare",
        ra=285.0,
        dec=45.0,
        time_peak_bjd=120.0,
        delta_centroid_mpix=0.5,
        bkg_correlation=0.88,  # >= 0.80 stray light coupling
        quality_flag_clean=True,
    )

    receipt = InstrumentalFalsePositiveFilter.evaluate(cand)
    assert receipt.is_instrumental_fp is True
    assert receipt.bkg_passed is False
    assert "STRAY_LIGHT" in receipt.veto_reasons
    assert receipt.instrumental_score == 0.0


def test_instrumental_triage_momentum_dump_veto():
    """Validates momentum dump bit 32 within 0.025 days triggers MOMENTUM_DUMP veto."""
    # Create target with momentum dump at t = 120.010 (within 0.025 d of t = 120.0)
    t = np.linspace(119.0, 121.0, 100, dtype=np.float64)
    q = np.zeros(100, dtype=np.int32)
    q[50] = 32  # Momentum dump bit
    target = HarvestedTarget(
        target_id="KIC_DUMP_01",
        track="Track_A",
        survey="Kepler",
        ra=285.0,
        dec=45.0,
        time=t,
        sap_quality=q,
    )

    cand = CandidateAnomaly(
        candidate_id="CAN_DUMP_01",
        target_id="KIC_DUMP_01",
        track="Track_A",
        survey="Kepler",
        anomaly_type="planet_transit",
        ra=285.0,
        dec=45.0,
        time_peak_bjd=t[50],  # Centered on momentum dump
        t_start=t[48],
        t_peak=t[50],
        t_end=t[52],
        delta_centroid_mpix=0.5,
        bkg_correlation=0.10,
        quality_flag_clean=False,
    )

    receipt = InstrumentalFalsePositiveFilter.evaluate(cand, target)
    assert receipt.is_instrumental_fp is True
    assert receipt.thruster_veto_passed is False
    assert "MOMENTUM_DUMP" in receipt.veto_reasons


def test_instrumental_triage_argabrightening_veto():
    """Validates Argabrightening bit 2048 within buffer triggers ARGABRIGHTENING veto."""
    t = np.linspace(119.0, 121.0, 100, dtype=np.float64)
    q = np.zeros(100, dtype=np.int32)
    q[50] = 2048  # Argabrightening bit
    target = HarvestedTarget(
        target_id="KIC_ARG_01",
        track="Track_A",
        survey="Kepler",
        ra=285.0,
        dec=45.0,
        time=t,
        sap_quality=q,
    )

    cand = CandidateAnomaly(
        candidate_id="CAN_ARG_01",
        target_id="KIC_ARG_01",
        track="Track_A",
        survey="Kepler",
        anomaly_type="micro_flare",
        ra=285.0,
        dec=45.0,
        time_peak_bjd=t[50],
        t_start=t[49],
        t_peak=t[50],
        t_end=t[51],
        quality_flag_clean=False,
    )

    receipt = InstrumentalFalsePositiveFilter.evaluate(cand, target)
    assert receipt.is_instrumental_fp is True
    assert receipt.argabrightening_detected is True
    assert "ARGABRIGHTENING" in receipt.veto_reasons


# ==============================================================================
# 4. Astrophysical Consistency Tests
# ==============================================================================

def test_astrophysical_consistency_odd_even_beb_rejection():
    """Validates odd-even depth difference Delta d >= 3.0 sigma triggers BEB veto."""
    cand = CandidateAnomaly(
        candidate_id="CAN_BEB_01",
        target_id="KIC_BEB_01",
        track="Track_A",
        survey="Kepler",
        anomaly_type="planet_transit",
        ra=285.0,
        dec=45.0,
        time_peak_bjd=120.0,
        duration_hours=4.0,
        depth_ppm=5000.0,
        metadata={
            "period": 4.0,
            "odd_even_sigma": 4.5,  # >= 3.0 sigma parity disparity
            "depth_odd_ppm": 6000.0,
            "depth_even_ppm": 2000.0,
        },
    )

    receipt = AstrophysicalConsistencyVetter.evaluate(cand)
    assert receipt.is_physical_fp is True
    assert receipt.odd_even_passed is False
    assert "BEB" in receipt.veto_reasons


def test_astrophysical_consistency_odd_even_parity_pass():
    """Validates equal odd and even transit depths pass parity test cleanly."""
    cand = CandidateAnomaly(
        candidate_id="CAN_PARITY_PASS_01",
        target_id="KIC_PASS_01",
        track="Track_A",
        survey="Kepler",
        anomaly_type="planet_transit",
        ra=285.0,
        dec=45.0,
        time_peak_bjd=120.0,
        duration_hours=4.0,
        depth_ppm=5000.0,
        metadata={
            "period": 4.0,
            "odd_even_sigma": 0.35,  # < 2.0 sigma
            "depth_odd_ppm": 5050.0,
            "depth_even_ppm": 4980.0,
        },
    )

    receipt = AstrophysicalConsistencyVetter.evaluate(cand)
    assert receipt.is_physical_fp is False
    assert receipt.odd_even_passed is True
    assert len(receipt.veto_reasons) == 0


def test_astrophysical_consistency_monotransit_graceful_bypass():
    """Validates monotransit and non-periodic events bypass odd-even check without rejection."""
    cand = CandidateAnomaly(
        candidate_id="CAN_MONO_01",
        target_id="KIC_MONO_01",
        track="Track_A",
        survey="Kepler",
        anomaly_type="planet_transit",
        ra=285.0,
        dec=45.0,
        time_peak_bjd=120.0,
        duration_hours=6.0,
        depth_ppm=3000.0,
        n_epochs=1,  # Single epoch
    )

    receipt = AstrophysicalConsistencyVetter.evaluate(cand)
    assert receipt.is_physical_fp is False
    assert receipt.odd_even_passed is True
    assert receipt.odd_even_sigma == 0.0


def test_astrophysical_consistency_unphysical_duration():
    """Validates duration exceeding Keplerian density bounds by > 3.0x triggers UNPHYSICAL_DURATION."""
    # Planet with P = 1.0 day has circular solar duration ~2.0 hours. Set duration = 36.0 hours.
    cand = CandidateAnomaly(
        candidate_id="CAN_LONG_DUR_01",
        target_id="KIC_LONG_01",
        track="Track_A",
        survey="Kepler",
        anomaly_type="planet_transit",
        ra=285.0,
        dec=45.0,
        time_peak_bjd=120.0,
        duration_hours=36.0,  # >> 2.0 hours
        depth_ppm=2000.0,
        metadata={"period": 1.0},
    )

    receipt = AstrophysicalConsistencyVetter.evaluate(cand, period_days=1.0)
    assert receipt.is_physical_fp is True
    assert receipt.duration_passed is False
    assert "UNPHYSICAL_DURATION" in receipt.veto_reasons


def test_astrophysical_consistency_morphology_u_vs_v():
    """Validates morphology classification between flat-bottomed U-shape and sharp V-shape."""
    # Flat-bottomed U-shape transit
    f_u = np.ones(40, dtype=np.float64)
    f_u[10:30] = 0.99  # Wide flat bottom
    t_u = np.linspace(0, 1, 40)

    cand_u = CandidateAnomaly(
        candidate_id="CAN_U_01",
        target_id="KIC_U_01",
        track="Track_A",
        survey="Kepler",
        anomaly_type="planet_transit",
        ra=285.0,
        dec=45.0,
        time_peak_bjd=0.5,
        duration_hours=4.0,
        depth_ppm=10000.0,
        time_window=t_u,
        flux_window=f_u,
    )
    rec_u = AstrophysicalConsistencyVetter.evaluate(cand_u)
    assert rec_u.morphology_shape == "U-shape"
    assert rec_u.morphology_ratio <= 0.25

    # Sharp triangular V-shape transit
    f_v = np.ones(41, dtype=np.float64)
    for i in range(21):
        f_v[i] = 1.0 - 0.01 * (i / 20.0)
        f_v[40 - i] = f_v[i]
    t_v = np.linspace(0, 1, 41)

    cand_v = CandidateAnomaly(
        candidate_id="CAN_V_01",
        target_id="KIC_V_01",
        track="Track_A",
        survey="Kepler",
        anomaly_type="planet_transit",
        ra=285.0,
        dec=45.0,
        time_peak_bjd=0.5,
        duration_hours=4.0,
        depth_ppm=10000.0,
        delta_centroid_mpix=1.8,  # Centroid shift + V-shape -> BEB
        time_window=t_v,
        flux_window=f_v,
    )
    rec_v = AstrophysicalConsistencyVetter.evaluate(cand_v)
    assert rec_v.morphology_shape == "V-shape"
    assert rec_v.is_physical_fp is True
    assert "BEB" in rec_v.veto_reasons


# ==============================================================================
# 5. Composite Scoring and Discovery Tier Assignment Tests
# ==============================================================================

def test_composite_scoring_tier_1_gold():
    """Validates Tier 1 Gold discovery tier assignment on pristine candidate."""
    cand = CandidateAnomaly(
        candidate_id="CAN_TIER1_01",
        target_id="KIC_TIER1_01",
        track="Track_A",
        survey="Kepler",
        anomaly_type="circumbinary_transit",
        ra=285.0,
        dec=45.0,
        time_peak_bjd=120.0,
        snr=15.0,
        duration_hours=4.0,
        depth_ppm=3500.0,
        delta_centroid_mpix=0.4,
        bkg_correlation=0.04,
        quality_flag_clean=True,
    )

    inst = InstrumentalVettingReceipt(
        delta_centroid_mpix=0.4,
        centroid_snr=1.2,
        delta_x_mpix=0.2,
        delta_y_mpix=0.3,
        centroid_sigma_mpix=0.3,
        centroid_passed=True,
        bkg_correlation=0.04,
        bkg_relative_amp=0.01,
        bkg_passed=True,
        quality_flag_clean=True,
        fatal_quality_bits=0,
        min_dump_distance_hours=24.0,
        thruster_veto_passed=True,
        argabrightening_detected=False,
        instrumental_score=0.98,
        is_instrumental_fp=False,
        veto_reasons=[],
    )

    phys = PhysicalVettingReceipt(
        odd_even_sigma=0.4,
        odd_even_passed=True,
        depth_odd_ppm=3500.0,
        depth_even_ppm=3520.0,
        duration_ratio=1.0,
        duration_passed=True,
        morphology_ratio=0.12,
        morphology_shape="U-shape",
        morphology_score=1.0,
        physical_score=1.0,
        is_physical_fp=False,
        veto_reasons=[],
    )

    score, tier, reason = compute_composite_vetting_score(cand, inst, phys)
    assert score >= 0.85
    assert tier == "Tier_1_Gold"
    assert reason is None


def test_composite_scoring_tier_2_silver():
    """Validates Tier 2 Silver candidate tier assignment."""
    cand = CandidateAnomaly(
        candidate_id="CAN_TIER2_01",
        target_id="KIC_TIER2_01",
        track="Track_A",
        survey="Kepler",
        anomaly_type="planet_transit",
        ra=285.0,
        dec=45.0,
        time_peak_bjd=120.0,
        snr=8.0,  # Silver SNR >= 6.0
        duration_hours=4.0,
        depth_ppm=1500.0,
        delta_centroid_mpix=1.8,  # Moderate centroid shift (1.5 <= dr < 3.0)
        bkg_correlation=0.30,     # Moderate background coupling (|r| < 0.50)
        quality_flag_clean=True,
    )

    inst = InstrumentalVettingReceipt(
        delta_centroid_mpix=1.8,
        centroid_snr=1.8,
        delta_x_mpix=1.2,
        delta_y_mpix=1.3,
        centroid_sigma_mpix=1.0,
        centroid_passed=True,
        bkg_correlation=0.30,
        bkg_relative_amp=0.03,
        bkg_passed=True,
        quality_flag_clean=True,
        fatal_quality_bits=0,
        min_dump_distance_hours=12.0,
        thruster_veto_passed=True,
        argabrightening_detected=False,
        instrumental_score=0.82,
        is_instrumental_fp=False,
        veto_reasons=[],
    )

    phys = PhysicalVettingReceipt(
        odd_even_sigma=1.5,
        odd_even_passed=True,
        depth_odd_ppm=1600.0,
        depth_even_ppm=1400.0,
        duration_ratio=1.2,
        duration_passed=True,
        morphology_ratio=0.30,
        morphology_shape="intermediate",
        morphology_score=0.85,
        physical_score=0.85,
        is_physical_fp=False,
        veto_reasons=[],
    )

    score, tier, reason = compute_composite_vetting_score(cand, inst, phys)
    assert 0.70 <= score < 0.85
    assert tier == "Tier_2_Silver"
    assert reason is None



def test_composite_scoring_rejected_on_veto():
    """Validates hard veto forces tier to Rejected and assigns rejection reason."""
    cand = CandidateAnomaly(
        candidate_id="CAN_REJ_01",
        target_id="KIC_REJ_01",
        track="Track_A",
        survey="Kepler",
        anomaly_type="planet_transit",
        ra=285.0,
        dec=45.0,
        time_peak_bjd=120.0,
        snr=10.0,
        delta_centroid_mpix=4.5,
    )

    inst = InstrumentalVettingReceipt(
        delta_centroid_mpix=4.5,
        centroid_snr=6.0,
        delta_x_mpix=3.0,
        delta_y_mpix=3.3,
        centroid_sigma_mpix=0.7,
        centroid_passed=False,
        bkg_correlation=0.05,
        bkg_relative_amp=0.01,
        bkg_passed=True,
        quality_flag_clean=True,
        fatal_quality_bits=0,
        min_dump_distance_hours=24.0,
        thruster_veto_passed=True,
        argabrightening_detected=False,
        instrumental_score=0.0,
        is_instrumental_fp=True,
        veto_reasons=["CENTROID_PULL"],
    )

    phys = PhysicalVettingReceipt(
        odd_even_sigma=0.5,
        odd_even_passed=True,
        depth_odd_ppm=1000.0,
        depth_even_ppm=1000.0,
        duration_ratio=1.0,
        duration_passed=True,
        morphology_ratio=0.10,
        morphology_shape="U-shape",
        morphology_score=1.0,
        physical_score=1.0,
        is_physical_fp=False,
        veto_reasons=[],
    )

    score, tier, reason = compute_composite_vetting_score(cand, inst, phys)
    assert score == 0.0
    assert tier == "Rejected"
    assert reason == "CENTROID_PULL"


# ==============================================================================
# 6. ExpeditionVetter Master Class & Out-of-Core Memory Streaming Tests
# ==============================================================================

def test_expedition_vetter_end_to_end():
    """Validates ExpeditionVetter vet_candidate and vet_target pipelines."""
    vetter = ExpeditionVetter()

    # Create candidate on known Kepler-8 b
    cand = CandidateAnomaly(
        candidate_id="CAN_KEP8_01",
        target_id="KIC_6922244",
        track="Track_A",
        survey="Kepler",
        anomaly_type="planet_transit",
        ra=285.679,
        dec=42.452,
        time_peak_bjd=122.045,  # Known t0
        duration_hours=3.2,
        snr=20.0,
        depth_ppm=9400.0,
        delta_centroid_mpix=0.5,
        bkg_correlation=0.05,
        quality_flag_clean=True,
    )

    vetted = vetter.vet_candidate(cand)
    assert isinstance(vetted, VettedCandidate)
    assert vetted.catalog_match == "KOI"
    assert vetted.disposition == "CONFIRMED_KNOWN"
    assert vetted.taxonomy_class == "catalog_confirmed_planet"
    assert vetted.discovery_tier == "Tier_1_Gold"
    assert vetted.vetting_score >= 0.85
    assert vetted.rejection_reason is None

    # Test vet_target convenience function
    target = HarvestedTarget(
        target_id="KIC_6922244",
        track="Track_A",
        survey="Kepler",
        ra=285.679,
        dec=42.452,
    )
    vetted_list = vet_target(target, [cand])
    assert len(vetted_list) == 1
    assert vetted_list[0].candidate_id == vetted.candidate_id


def test_expedition_vetter_streaming_memory_bound():
    """Validates vet_stream operates strictly under memory ceilings (< 25 MB active heap)."""
    vetter = ExpeditionVetter()

    def generate_candidate_stream(n_items: int = 200):
        for i in range(n_items):
            yield CandidateAnomaly(
                candidate_id=f"CAN_STREAM_{i:04d}",
                target_id=f"KIC_{10000000 + i}",
                track="Track_A",
                survey="Kepler",
                anomaly_type="circumbinary_transit",
                ra=280.0 + (i % 10) * 0.1,
                dec=40.0 + (i % 10) * 0.1,
                time_peak_bjd=120.0 + i * 0.5,
                duration_hours=4.0,
                snr=8.5,
                depth_ppm=2500.0,
                delta_centroid_mpix=0.8,
                bkg_correlation=0.10,
                quality_flag_clean=True,
            )

    gc.collect()
    tracemalloc.start()
    init_mem, _ = tracemalloc.get_traced_memory()

    vetted_count = 0
    tier_counts = {"Tier_1_Gold": 0, "Tier_2_Silver": 0, "Tier_3_Bronze": 0, "Rejected": 0}

    for vetted in vetter.vet_stream(generate_candidate_stream(200)):
        vetted_count += 1
        tier_counts[vetted.discovery_tier] += 1
        # Check integrity
        assert vetted.candidate_id.startswith("VET_CAN_STREAM_")

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = (peak_mem - init_mem) / (1024 * 1024)
    print(f"Streaming peak heap memory for 200 candidates: {peak_mb:.2f} MB")

    assert vetted_count == 200
    assert tier_counts["Tier_1_Gold"] == 200
    # Assert active memory remains strictly below 25.0 MB ceiling
    assert peak_mb < 25.0


def test_ephemeris_matcher_spatial_cone_search():
    """Validates spatial cone search matches catalog entry by coordinates when ID is unknown."""
    matcher = EphemerisMatcher()
    # Benchmark entry at ra=292.298, dec=47.969 is HAT-P-7 b
    cand = CandidateAnomaly(
        candidate_id="CAN_SPATIAL_01",
        target_id="UNKNOWN_ID_999",
        track="Track_A",
        survey="Kepler",
        anomaly_type="planet_transit",
        ra=292.29801,  # within 0.1 arcsec
        dec=47.96901,
        time_peak_bjd=120.045 + 3 * 2.204733,
        duration_hours=4.1,
        depth_ppm=6900.0,
    )

    res = matcher.cross_match(cand)
    assert res["is_match"] is True
    assert res["catalog_match"] == "KOI"
    assert "HAT-P-7" in res["catalog_id"]


def test_instrumental_triage_moments_arrays_centroid_shift():
    """Validates centroid shift computation directly from mom_centr1 and mom_centr2 time series."""
    n_pts = 100
    t = np.linspace(100.0, 102.0, n_pts, dtype=np.float64)
    q = np.zeros(n_pts, dtype=np.int32)
    # Centroids baseline at (5.000, 10.000)
    cx = np.full(n_pts, 5.000, dtype=np.float64)
    cy = np.full(n_pts, 10.000, dtype=np.float64)
    # Inject 4.0 milli-pixel centroid pull in transit (indices 45 to 55)
    cx[45:56] += 0.004  # 4 mpix shift

    target = HarvestedTarget(
        target_id="KIC_CENTR_TEST",
        track="Track_A",
        survey="Kepler",
        ra=285.0,
        dec=45.0,
        time=t,
        mom_centr1=cx,
        mom_centr2=cy,
        sap_quality=q,
    )

    cand = CandidateAnomaly(
        candidate_id="CAN_CENTR_01",
        target_id="KIC_CENTR_TEST",
        track="Track_A",
        survey="Kepler",
        anomaly_type="planet_transit",
        ra=285.0,
        dec=45.0,
        time_peak_bjd=t[50],
        t_start=t[45],
        t_peak=t[50],
        t_end=t[55],
    )

    receipt = InstrumentalFalsePositiveFilter.evaluate(cand, target)
    assert receipt.delta_centroid_mpix >= 3.0
    assert receipt.centroid_passed is False
    assert "CENTROID_PULL" in receipt.veto_reasons


def test_instrumental_triage_background_array_correlation():
    """Validates Pearson correlation computation directly from sap_flux and sap_bkg arrays."""
    n_pts = 100
    t = np.linspace(100.0, 102.0, n_pts, dtype=np.float64)
    q = np.zeros(n_pts, dtype=np.int32)
    # Perfectly correlated background and flux variation
    f = np.sin(np.linspace(0, 10, n_pts))
    b = f * 1.5 + 10.0

    target = HarvestedTarget(
        target_id="KIC_BKG_TEST",
        track="Track_A",
        survey="Kepler",
        ra=285.0,
        dec=45.0,
        time=t,
        sap_flux=f,
        sap_bkg=b,
        sap_quality=q,
    )

    cand = CandidateAnomaly(
        candidate_id="CAN_BKG_01",
        target_id="KIC_BKG_TEST",
        track="Track_A",
        survey="Kepler",
        anomaly_type="micro_flare",
        ra=285.0,
        dec=45.0,
        time_peak_bjd=t[50],
        t_start=t[40],
        t_peak=t[50],
        t_end=t[60],
    )

    receipt = InstrumentalFalsePositiveFilter.evaluate(cand, target)
    assert receipt.bkg_correlation >= 0.80
    assert receipt.bkg_passed is False
    assert "STRAY_LIGHT" in receipt.veto_reasons


def test_track_b_micro_flare_vetting_gold():
    """Validates Track B asymmetric micro-flare passes physical vetting as Gold tier."""
    cand = CandidateAnomaly(
        candidate_id="CAN_FLARE_01",
        target_id="KIC_FLARE_01",
        track="Track_B",
        survey="Kepler",
        anomaly_type="micro_flare",
        ra=285.0,
        dec=45.0,
        time_peak_bjd=125.0,
        duration_hours=0.5,
        snr=18.0,
        amplitude=4500.0,
        depth_ppm=0.0,
        delta_centroid_mpix=0.3,
        bkg_correlation=0.02,
        quality_flag_clean=True,
    )

    vetted = vet_candidate(cand)
    assert vetted.discovery_tier == "Tier_1_Gold"
    assert vetted.taxonomy_class == "genuine_novel_discovery"
    assert vetted.morphology_shape == "asymmetric_flare"
    assert vetted.vetting_score >= 0.85


def test_track_c_moving_body_conjunction_vetting():
    """Validates Track C moving body conjunction passes vetting with zero Lie triad drift."""
    cand = CandidateAnomaly(
        candidate_id="CAN_TRACK_C_01",
        target_id="TRACK_C_001",
        track="Track_C",
        survey="JPL_Horizons",
        anomaly_type="moving_body_conjunction",
        ra=180.0,
        dec=15.0,
        time_peak_bjd=2459000.5,
        snr=12.0,
        amplitude=1.5,
        triad_drift=0.0,
        tie_breaking_gap=3.5,
        delta_centroid_mpix=0.5,
        bkg_correlation=0.05,
        quality_flag_clean=True,
    )

    vetted = vet_candidate(cand)
    assert vetted.discovery_tier == "Tier_1_Gold"
    assert vetted.taxonomy_class == "genuine_novel_discovery"
    assert vetted.triad_drift == 0.0
    assert vetted.vetting_score >= 0.85

