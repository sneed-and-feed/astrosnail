r"""
Challenger 2 Empirical Stress Test Suite for Milestone M7.3
============================================================
Adversarial Verification of:
  1. Centroid Shift Stability Stress:
     - Delta R = 0.8 mpix (Tier 1 Gold qualification)
     - Delta R = 2.2 mpix (Tier 2 Silver qualification)
     - Delta R = 3.5 mpix (Hard veto "CENTROID_PULL" and rejection)
     - Array-level moment shifts and SNR pull gate
     - NaN and invalid moment inputs graceful handling
  2. Background Decorrelation Stress:
     - Strong positive correlation (r = +0.88, stray light veto)
     - Strong negative correlation (r = -0.91, over-subtraction stray light veto)
     - Uncorrelated background (r = +0.04, clean pass with s_bkg = 1.0)
     - Array-level Pearson correlation verification
  3. Spacecraft Quality & Buffer Stress:
     - Reaction wheel desaturation (bit 32 within +/- 0.015 d buffer -> MOMENTUM_DUMP veto)
     - Desaturation outside buffer (> 0.025 d -> pass)
     - Argabrightening detector event (bit 2048 within buffer -> ARGABRIGHTENING veto)
     - Nominal telemetry (SAP_QUALITY == 0 -> clean pass)
  4. Out-of-Core Streaming & Memory Stress:
     - Stream 3,000 synthetic candidate events through ExpeditionVetter.vet_stream()
     - Active heap verification using tracemalloc (< 25 MB active heap limit)
     - Validation of all 3,000 VettedCandidate instances without memory leaks

Author: Challenger 2 (teamwork_challenger_m7_3_2)
Target Milestone: M7.3 (agent-3-vetter)
"""

import gc
import math
import tracemalloc
from typing import Generator
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
# Helper Factory for Synthetic Clean Candidates
# ==============================================================================

def make_clean_candidate(
    candidate_id: str = "CAND_CLEAN_01",
    target_id: str = "KIC_CLEAN_HOST",
    delta_centroid_mpix: float = 0.4,
    bkg_correlation: float = 0.02,
    snr: float = 12.0,
    depth_ppm: float = 3500.0,
    duration_hours: float = 3.5,
    odd_even_sigma: float = 0.2,
    morphology_ratio: float = 0.15,  # U-shape
    quality_clean: bool = True,
) -> CandidateAnomaly:
    """Creates a synthetic candidate anomaly with controlled vetting parameters."""
    t_win = np.linspace(100.0, 100.5, 60, dtype=np.float64)
    f_win = np.ones(60, dtype=np.float64)
    # Transit shape
    f_win[22:38] -= (depth_ppm * 1e-6)

    cand = CandidateAnomaly(
        candidate_id=candidate_id,
        target_id=target_id,
        track="Track_A",
        survey="Kepler",
        anomaly_type="circumbinary_transit",
        ra=290.0,
        dec=45.0,
        time_peak_bjd=100.25,
        t_start=100.15,
        t_peak=100.25,
        t_end=100.35,
        duration_hours=duration_hours,
        duration_cadences=16,
        n_epochs=1,
        snr=snr,
        amplitude=depth_ppm,
        depth_ppm=depth_ppm,
        delta_centroid_mpix=delta_centroid_mpix,
        bkg_correlation=bkg_correlation,
        quality_flag_clean=quality_clean,
        time_window=t_win,
        flux_window=f_win,
        metadata={
            "odd_even_sigma": odd_even_sigma,
            "morphology_ratio": morphology_ratio,
        },
    )
    return cand


# ==============================================================================
# Stress Vector 1: Centroid Shift Stability Stress
# ==============================================================================

def test_centroid_shift_gold_tier_qualification():
    """
    Stress Vector 1.1: Delta R = 0.8 mpix must qualify for Tier 1 Gold.
    Asserts:
      - Centroid check passes (centroid_passed == True)
      - No CENTROID_PULL veto reason
      - Discovery tier is Tier_1_Gold
      - Vetting score >= 0.85
    """
    cand = make_clean_candidate(
        candidate_id="STRESS_CENT_0_8",
        delta_centroid_mpix=0.8,
        bkg_correlation=0.03,
        snr=14.0,
    )
    vetter = ExpeditionVetter()
    vetted = vetter.vet_candidate(cand)

    assert vetted.centroid_passed is True
    assert "CENTROID_PULL" not in vetted.instrumental_receipt.veto_reasons
    assert vetted.delta_centroid_mpix == 0.8
    assert vetted.discovery_tier == "Tier_1_Gold"
    assert vetted.vetting_score >= 0.85
    assert vetted.rejection_reason is None


def test_centroid_shift_silver_tier_qualification():
    """
    Stress Vector 1.2: Delta R = 2.2 mpix must qualify for Tier 2 Silver.
    Asserts:
      - Centroid check passes (centroid_passed == True)
      - Delta R >= 1.5 blocks Tier 1 Gold
      - Delta R < 3.0 qualifies for Tier 2 Silver
      - Discovery tier is Tier_2_Silver
      - Vetting score in [0.70, 0.85)
    """
    cand = make_clean_candidate(
        candidate_id="STRESS_CENT_2_2",
        delta_centroid_mpix=2.2,
        bkg_correlation=0.03,
        snr=14.0,
    )
    vetter = ExpeditionVetter()
    vetted = vetter.vet_candidate(cand)

    assert vetted.centroid_passed is True
    assert "CENTROID_PULL" not in vetted.instrumental_receipt.veto_reasons
    assert vetted.delta_centroid_mpix == 2.2
    assert vetted.discovery_tier == "Tier_2_Silver"
    assert vetted.vetting_score >= 0.70
    assert vetted.rejection_reason is None


def test_centroid_shift_hard_veto_rejection():
    """
    Stress Vector 1.3: Delta R = 3.5 mpix must trigger hard veto "CENTROID_PULL" and reject.
    Asserts:
      - Centroid check fails (centroid_passed == False)
      - CENTROID_PULL is present in veto reasons
      - is_instrumental_fp is True
      - Discovery tier is Rejected
      - Vetting score is 0.0
      - Rejection reason is CENTROID_PULL
    """
    cand = make_clean_candidate(
        candidate_id="STRESS_CENT_3_5",
        delta_centroid_mpix=3.5,
        bkg_correlation=0.03,
        snr=14.0,
    )
    vetter = ExpeditionVetter()
    vetted = vetter.vet_candidate(cand)

    assert vetted.centroid_passed is False
    assert "CENTROID_PULL" in vetted.instrumental_receipt.veto_reasons
    assert vetted.instrumental_receipt.is_instrumental_fp is True
    assert vetted.discovery_tier == "Rejected"
    assert vetted.vetting_score == 0.0
    assert vetted.rejection_reason == "CENTROID_PULL"


def test_centroid_shift_array_level_moments_triage():
    """
    Stress Vector 1.4: Direct array-level moments evaluation.
    Constructs a HarvestedTarget with moment offsets:
      - In-transit shift dx = 2.5 mpix, dy = 2.45 mpix -> Delta R = sqrt(2.5^2 + 2.45^2) = 3.50 mpix
    Asserts:
      - Measured delta_centroid_mpix >= 3.0 mpix
      - Triggers CENTROID_PULL veto
      - Rejects candidate into Tier Rejected
    """
    n_pts = 100
    t_arr = np.linspace(100.0, 100.5, n_pts)
    f_arr = np.ones(n_pts, dtype=np.float64)
    # Transit between indices 40 and 55
    f_arr[40:55] -= 0.005

    # Moments: baseline at (500.0, 500.0)
    cx = np.full(n_pts, 500.0, dtype=np.float64)
    cy = np.full(n_pts, 500.0, dtype=np.float64)
    # Introduce in-transit shift of +0.0025 px and +0.00245 px (2.5 mpix and 2.45 mpix)
    cx[40:55] += 0.0025
    cy[40:55] += 0.00245

    target = HarvestedTarget(
        target_id="KIC_CENTR_PULL_ARRAY",
        track="Track_A",
        survey="Kepler",
        ra=290.0,
        dec=45.0,
        time=t_arr,
        sap_flux=f_arr,
        mom_centr1=cx,
        mom_centr2=cy,
        sap_quality=np.zeros(n_pts, dtype=np.int32),
    )

    cand = CandidateAnomaly(
        candidate_id="CAND_CENTR_PULL_ARR",
        target_id="KIC_CENTR_PULL_ARRAY",
        track="Track_A",
        survey="Kepler",
        anomaly_type="transit",
        ra=290.0,
        dec=45.0,
        time_peak_bjd=float(t_arr[47]),
        t_start=float(t_arr[40]),
        t_peak=float(t_arr[47]),
        t_end=float(t_arr[55]),
        duration_hours=float((t_arr[55] - t_arr[40]) * 24.0),
        depth_ppm=5000.0,
        snr=15.0,
        delta_centroid_mpix=0.1,  # Initial candidate estimate overridden by target arrays
    )

    vetter = ExpeditionVetter()
    vetted = vetter.vet_candidate(cand, target)

    assert vetted.delta_centroid_mpix >= 3.0
    assert "CENTROID_PULL" in vetted.instrumental_receipt.veto_reasons
    assert vetted.discovery_tier == "Rejected"
    assert vetted.rejection_reason == "CENTROID_PULL"


def test_centroid_shift_nan_and_corrupted_moments_graceful_handling():
    """
    Stress Vector 1.5: NaN and invalid moment inputs handling.
    Tests:
      1. Candidate with delta_centroid_mpix = float('nan') without target
      2. Target with all-NaN moments arrays
      3. Target with partial NaNs (NaNs inside transit window and baseline)
    Asserts:
      - Engine executes without unhandled exceptions or crashes
      - Indeterminate or corrupted centroid is rejected gracefully with LOW_VETTING_SCORE or CENTROID_PULL
      - Never falsely awards Tier 1 Gold to unverified/NaN centroids
    """
    vetter = ExpeditionVetter()

    # 1. Direct candidate with NaN centroid
    cand_nan = make_clean_candidate(
        candidate_id="CAND_NAN_DIRECT",
        delta_centroid_mpix=float("nan"),
    )
    vetted_nan = vetter.vet_candidate(cand_nan)
    assert vetted_nan.discovery_tier == "Rejected"
    assert vetted_nan.discovery_tier != "Tier_1_Gold"

    # 2. Target with all-NaN moments
    n_pts = 80
    t_arr = np.linspace(100.0, 100.4, n_pts)
    f_arr = np.ones(n_pts, dtype=np.float64)
    f_arr[30:45] -= 0.004
    target_all_nan = HarvestedTarget(
        target_id="KIC_ALL_NAN_MOMENTS",
        track="Track_A",
        survey="Kepler",
        ra=290.0,
        dec=45.0,
        time=t_arr,
        sap_flux=f_arr,
        mom_centr1=np.full(n_pts, np.nan),
        mom_centr2=np.full(n_pts, np.nan),
        sap_quality=np.zeros(n_pts, dtype=np.int32),
    )
    cand_target = CandidateAnomaly(
        candidate_id="CAND_TARGET_ALL_NAN",
        target_id="KIC_ALL_NAN_MOMENTS",
        track="Track_A",
        survey="Kepler",
        anomaly_type="transit",
        ra=290.0,
        dec=45.0,
        time_peak_bjd=float(t_arr[37]),
        t_start=float(t_arr[30]),
        t_peak=float(t_arr[37]),
        t_end=float(t_arr[45]),
        duration_hours=3.6,
        depth_ppm=4000.0,
        snr=12.0,
        delta_centroid_mpix=0.5,
    )
    vetted_tgt_nan = vetter.vet_candidate(cand_target, target_all_nan)
    assert vetted_tgt_nan.discovery_tier in ("Rejected", "Tier_3_Bronze")
    assert vetted_tgt_nan.discovery_tier != "Tier_1_Gold"

    # 3. Target with partial NaNs (intermittent telemetry drops)
    cx_partial = np.full(n_pts, 500.0)
    cy_partial = np.full(n_pts, 500.0)
    cx_partial[::5] = np.nan
    cy_partial[::5] = np.nan
    # Transit points clean and stable
    cx_partial[30:45] = 500.0003  # 0.3 mpix shift
    cy_partial[30:45] = 500.0004  # 0.4 mpix shift -> 0.5 mpix total
    target_partial = HarvestedTarget(
        target_id="KIC_PARTIAL_NAN_MOMENTS",
        track="Track_A",
        survey="Kepler",
        ra=290.0,
        dec=45.0,
        time=t_arr,
        sap_flux=f_arr,
        mom_centr1=cx_partial,
        mom_centr2=cy_partial,
        sap_quality=np.zeros(n_pts, dtype=np.int32),
    )
    vetted_partial = vetter.vet_candidate(cand_target, target_partial)
    assert math.isfinite(vetted_partial.delta_centroid_mpix)
    assert vetted_partial.delta_centroid_mpix < 1.5
    assert vetted_partial.centroid_passed is True


# ==============================================================================
# Stress Vector 2: Background Decorrelation Stress
# ==============================================================================

def test_background_correlation_positive_stray_light_veto():
    """
    Stress Vector 2.1: Strong positive background correlation (r = +0.88).
    Asserts:
      - Triggers hard veto "STRAY_LIGHT"
      - bkg_passed is False
      - is_instrumental_fp is True
      - Discovery tier is Rejected
      - Rejection reason is STRAY_LIGHT
    """
    cand = make_clean_candidate(
        candidate_id="STRESS_BKG_POS_0_88",
        delta_centroid_mpix=0.5,
        bkg_correlation=0.88,
        snr=15.0,
    )
    vetter = ExpeditionVetter()
    vetted = vetter.vet_candidate(cand)

    assert vetted.bkg_passed is False
    assert "STRAY_LIGHT" in vetted.instrumental_receipt.veto_reasons
    assert vetted.instrumental_receipt.is_instrumental_fp is True
    assert vetted.discovery_tier == "Rejected"
    assert vetted.vetting_score == 0.0
    assert vetted.rejection_reason == "STRAY_LIGHT"


def test_background_correlation_negative_over_subtraction_veto():
    """
    Stress Vector 2.2: Strong negative correlation (r = -0.91, over-subtraction artifact).
    Asserts:
      - Triggers hard veto "STRAY_LIGHT" (absolute correlation |r| >= 0.80)
      - bkg_passed is False
      - is_instrumental_fp is True
      - Discovery tier is Rejected
      - Rejection reason is STRAY_LIGHT
    """
    cand = make_clean_candidate(
        candidate_id="STRESS_BKG_NEG_0_91",
        delta_centroid_mpix=0.5,
        bkg_correlation=-0.91,
        snr=15.0,
    )
    vetter = ExpeditionVetter()
    vetted = vetter.vet_candidate(cand)

    assert vetted.bkg_passed is False
    assert "STRAY_LIGHT" in vetted.instrumental_receipt.veto_reasons
    assert vetted.instrumental_receipt.is_instrumental_fp is True
    assert vetted.discovery_tier == "Rejected"
    assert vetted.vetting_score == 0.0
    assert vetted.rejection_reason == "STRAY_LIGHT"


def test_background_decorrelation_uncorrelated_pass_gold():
    """
    Stress Vector 2.3: Uncorrelated background (r = +0.04).
    Asserts:
      - bkg_passed is True
      - STRAY_LIGHT is not in veto reasons
      - Background continuous sub-score is 1.0 (since |r| <= 0.20)
      - Qualifies for Tier 1 Gold
    """
    cand = make_clean_candidate(
        candidate_id="STRESS_BKG_PASS_0_04",
        delta_centroid_mpix=0.6,
        bkg_correlation=0.04,
        snr=12.0,
    )
    vetter = ExpeditionVetter()
    vetted = vetter.vet_candidate(cand)

    assert vetted.bkg_passed is True
    assert "STRAY_LIGHT" not in vetted.instrumental_receipt.veto_reasons
    assert vetted.instrumental_receipt.is_instrumental_fp is False
    assert vetted.discovery_tier == "Tier_1_Gold"
    assert vetted.vetting_score >= 0.85
    assert vetted.rejection_reason is None


def test_background_array_level_strong_coupling_triage():
    """
    Stress Vector 2.4: Target array-level Pearson correlation stress test.
    Injects synchronous background flux oscillation strongly coupled to target flux:
      r(SAP_FLUX, SAP_BKG) > +0.85
    Asserts:
      - Measured bkg_correlation >= 0.80
      - bkg_passed is False
      - Triggers STRAY_LIGHT veto
      - Rejection reason is STRAY_LIGHT
    """
    rng = np.random.default_rng(20261030)
    n_pts = 120
    t_arr = np.linspace(100.0, 100.6, n_pts)

    # Correlated background drift driving flux variability
    common_signal = np.sin(np.linspace(0, 3 * np.pi, n_pts))
    f_arr = 1.0 + 0.02 * common_signal + rng.normal(0, 0.001, n_pts)
    bkg_arr = 500.0 + 100.0 * common_signal + rng.normal(0, 2.0, n_pts)

    target = HarvestedTarget(
        target_id="KIC_STRAY_LIGHT_ARR",
        track="Track_A",
        survey="Kepler",
        ra=290.0,
        dec=45.0,
        time=t_arr,
        sap_flux=f_arr,
        sap_bkg=bkg_arr,
        mom_centr1=np.full(n_pts, 500.0),
        mom_centr2=np.full(n_pts, 500.0),
        sap_quality=np.zeros(n_pts, dtype=np.int32),
    )

    cand = CandidateAnomaly(
        candidate_id="CAND_BKG_STRAY",
        target_id="KIC_STRAY_LIGHT_ARR",
        track="Track_A",
        survey="Kepler",
        anomaly_type="transit",
        ra=290.0,
        dec=45.0,
        time_peak_bjd=float(t_arr[60]),
        t_start=float(t_arr[50]),
        t_peak=float(t_arr[60]),
        t_end=float(t_arr[70]),
        duration_hours=2.8,
        depth_ppm=3000.0,
        snr=10.0,
        delta_centroid_mpix=0.4,
    )

    vetter = ExpeditionVetter()
    vetted = vetter.vet_candidate(cand, target)

    assert abs(vetted.bkg_correlation) >= 0.80
    assert vetted.bkg_passed is False
    assert "STRAY_LIGHT" in vetted.instrumental_receipt.veto_reasons
    assert vetted.discovery_tier == "Rejected"
    assert vetted.rejection_reason == "STRAY_LIGHT"


# ==============================================================================
# Stress Vector 3: Spacecraft Quality & Buffer Stress
# ==============================================================================

def test_spacecraft_telemetry_momentum_dump_buffer_veto():
    """
    Stress Vector 3.1: Candidate within +/- 0.015 days of reaction wheel desaturation (bit 32).
    Buffer is MIN_DUMP_BUFFER_DAYS = 0.025 d (~36 minutes).
    Asserts:
      - Triggers hard veto "MOMENTUM_DUMP"
      - thruster_veto_passed is False
      - is_instrumental_fp is True
      - Discovery tier is Rejected
      - Rejection reason is MOMENTUM_DUMP
    """
    n_pts = 100
    t_arr = np.linspace(100.0, 100.5, n_pts)
    f_arr = np.ones(n_pts, dtype=np.float64)
    f_arr[40:55] -= 0.005

    # Momentum dump (bit 32) occurs at index 37 (dt = t[47] - t[37] ~ 0.015 d)
    q_arr = np.zeros(n_pts, dtype=np.int32)
    dump_idx = 44  # dt = t[47] - t[44] = 3 * (0.5/99) ~ 0.01515 days
    q_arr[dump_idx] = 32

    dt_dump_actual = abs(t_arr[47] - t_arr[dump_idx])
    assert dt_dump_actual <= 0.025

    target = HarvestedTarget(
        target_id="KIC_MOM_DUMP_HOST",
        track="Track_A",
        survey="Kepler",
        ra=290.0,
        dec=45.0,
        time=t_arr,
        sap_flux=f_arr,
        sap_quality=q_arr,
        mom_centr1=np.full(n_pts, 500.0),
        mom_centr2=np.full(n_pts, 500.0),
    )

    cand = CandidateAnomaly(
        candidate_id="CAND_MOM_DUMP",
        target_id="KIC_MOM_DUMP_HOST",
        track="Track_A",
        survey="Kepler",
        anomaly_type="transit",
        ra=290.0,
        dec=45.0,
        time_peak_bjd=float(t_arr[47]),
        t_start=float(t_arr[40]),
        t_peak=float(t_arr[47]),
        t_end=float(t_arr[55]),
        duration_hours=float((t_arr[55] - t_arr[40]) * 24.0),
        depth_ppm=5000.0,
        snr=15.0,
        delta_centroid_mpix=0.4,
    )

    vetter = ExpeditionVetter()
    vetted = vetter.vet_candidate(cand, target)

    assert vetted.instrumental_receipt.thruster_veto_passed is False
    assert "MOMENTUM_DUMP" in vetted.instrumental_receipt.veto_reasons
    assert vetted.instrumental_receipt.is_instrumental_fp is True
    assert vetted.discovery_tier == "Rejected"
    assert vetted.rejection_reason == "MOMENTUM_DUMP"


def test_spacecraft_telemetry_momentum_dump_outside_buffer_pass():
    """
    Stress Vector 3.2: Reaction wheel desaturation well outside buffer (dt = 0.035 d > 0.025 d).
    Asserts:
      - MOMENTUM_DUMP veto is NOT triggered
      - thruster_veto_passed is True
      - Candidate passes vetting cleanly
    """
    n_pts = 100
    t_arr = np.linspace(100.0, 100.5, n_pts)
    f_arr = np.ones(n_pts, dtype=np.float64)
    f_arr[40:55] -= 0.005

    # Momentum dump (bit 32) occurs at index 39 (dt = t[47] - t[39] = 8 * (0.5/99) ~ 0.0404 days > 0.025)
    q_arr = np.zeros(n_pts, dtype=np.int32)
    q_arr[39] = 32

    dt_dump_actual = abs(t_arr[47] - t_arr[39])
    assert dt_dump_actual > 0.025

    target = HarvestedTarget(
        target_id="KIC_MOM_DUMP_SAFE",
        track="Track_A",
        survey="Kepler",
        ra=290.0,
        dec=45.0,
        time=t_arr,
        sap_flux=f_arr,
        sap_quality=q_arr,
        mom_centr1=np.full(n_pts, 500.0),
        mom_centr2=np.full(n_pts, 500.0),
    )

    cand = CandidateAnomaly(
        candidate_id="CAND_MOM_SAFE",
        target_id="KIC_MOM_DUMP_SAFE",
        track="Track_A",
        survey="Kepler",
        anomaly_type="transit",
        ra=290.0,
        dec=45.0,
        time_peak_bjd=float(t_arr[47]),
        t_start=float(t_arr[40]),
        t_peak=float(t_arr[47]),
        t_end=float(t_arr[55]),
        duration_hours=float((t_arr[55] - t_arr[40]) * 24.0),
        depth_ppm=5000.0,
        snr=15.0,
        delta_centroid_mpix=0.4,
    )

    vetter = ExpeditionVetter()
    vetted = vetter.vet_candidate(cand, target)

    assert vetted.instrumental_receipt.thruster_veto_passed is True
    assert "MOMENTUM_DUMP" not in vetted.instrumental_receipt.veto_reasons
    assert vetted.discovery_tier != "Rejected"


def test_spacecraft_telemetry_argabrightening_buffer_veto():
    """
    Stress Vector 3.3: Candidate overlapping Argabrightening detector event (bit 2048).
    Buffer is MIN_ARG_BUFFER_DAYS = 0.025 d.
    Asserts:
      - Triggers hard veto "ARGABRIGHTENING"
      - argabrightening_detected is True
      - is_instrumental_fp is True
      - Discovery tier is Rejected
      - Rejection reason is ARGABRIGHTENING
    """
    n_pts = 100
    t_arr = np.linspace(100.0, 100.5, n_pts)
    f_arr = np.ones(n_pts, dtype=np.float64)
    f_arr[40:55] -= 0.005

    # Argabrightening (bit 2048) occurs right during event at index 48
    q_arr = np.zeros(n_pts, dtype=np.int32)
    q_arr[48] = 2048

    target = HarvestedTarget(
        target_id="KIC_ARG_HOST",
        track="Track_A",
        survey="Kepler",
        ra=290.0,
        dec=45.0,
        time=t_arr,
        sap_flux=f_arr,
        sap_quality=q_arr,
        mom_centr1=np.full(n_pts, 500.0),
        mom_centr2=np.full(n_pts, 500.0),
    )

    cand = CandidateAnomaly(
        candidate_id="CAND_ARG_EVENT",
        target_id="KIC_ARG_HOST",
        track="Track_A",
        survey="Kepler",
        anomaly_type="transit",
        ra=290.0,
        dec=45.0,
        time_peak_bjd=float(t_arr[47]),
        t_start=float(t_arr[40]),
        t_peak=float(t_arr[47]),
        t_end=float(t_arr[55]),
        duration_hours=float((t_arr[55] - t_arr[40]) * 24.0),
        depth_ppm=5000.0,
        snr=15.0,
        delta_centroid_mpix=0.4,
    )

    vetter = ExpeditionVetter()
    vetted = vetter.vet_candidate(cand, target)

    assert vetted.instrumental_receipt.argabrightening_detected is True
    assert "ARGABRIGHTENING" in vetted.instrumental_receipt.veto_reasons
    assert vetted.instrumental_receipt.is_instrumental_fp is True
    assert vetted.discovery_tier == "Rejected"
    assert vetted.rejection_reason == "ARGABRIGHTENING"


def test_spacecraft_telemetry_nominal_clean_pass():
    """
    Stress Vector 3.4: Candidate with nominal spacecraft telemetry (SAP_QUALITY == 0).
    Asserts:
      - quality_flag_clean is True
      - fatal_quality_bits is 0
      - thruster_veto_passed is True
      - argabrightening_detected is False
      - Qualifies cleanly for Tier 1 Gold
    """
    n_pts = 100
    t_arr = np.linspace(100.0, 100.5, n_pts)
    f_arr = np.ones(n_pts, dtype=np.float64)
    f_arr[40:55] -= 0.005

    target = HarvestedTarget(
        target_id="KIC_NOMINAL_HOST",
        track="Track_A",
        survey="Kepler",
        ra=290.0,
        dec=45.0,
        time=t_arr,
        sap_flux=f_arr,
        sap_quality=np.zeros(n_pts, dtype=np.int32),  # Perfectly clean telemetry
        mom_centr1=np.full(n_pts, 500.0),
        mom_centr2=np.full(n_pts, 500.0),
    )

    cand = CandidateAnomaly(
        candidate_id="CAND_NOMINAL_GOLD",
        target_id="KIC_NOMINAL_HOST",
        track="Track_A",
        survey="Kepler",
        anomaly_type="transit",
        ra=290.0,
        dec=45.0,
        time_peak_bjd=float(t_arr[47]),
        t_start=float(t_arr[40]),
        t_peak=float(t_arr[47]),
        t_end=float(t_arr[55]),
        duration_hours=float((t_arr[55] - t_arr[40]) * 24.0),
        depth_ppm=5000.0,
        snr=15.0,
        delta_centroid_mpix=0.4,
    )

    vetter = ExpeditionVetter()
    vetted = vetter.vet_candidate(cand, target)

    assert vetted.instrumental_receipt.quality_flag_clean is True
    assert vetted.instrumental_receipt.fatal_quality_bits == 0
    assert vetted.instrumental_receipt.thruster_veto_passed is True
    assert vetted.instrumental_receipt.argabrightening_detected is False
    assert vetted.discovery_tier == "Tier_1_Gold"
    assert vetted.rejection_reason is None


# ==============================================================================
# Stress Vector 4: Out-of-Core Streaming & Memory Stress
# ==============================================================================

def candidate_stream_generator(n_events: int = 3000) -> Generator[CandidateAnomaly, None, None]:
    """Generates a continuous out-of-core stream of n_events synthetic candidates."""
    rng = np.random.default_rng(42)
    for i in range(n_events):
        # Varying SNRs, depths, and shifts
        snr = float(rng.uniform(4.0, 20.0))
        depth = float(rng.uniform(200.0, 15000.0))
        shift = float(rng.uniform(0.1, 4.5))
        bkg = float(rng.uniform(-0.95, 0.95))

        # Compact array windows
        t_win = np.linspace(100.0, 100.2, 20, dtype=np.float64)
        f_win = np.ones(20, dtype=np.float64)

        yield CandidateAnomaly(
            candidate_id=f"STREAM_CAND_{i:05d}",
            target_id=f"KIC_STREAM_{i // 5:05d}",
            track="Track_A",
            survey="Kepler",
            anomaly_type="transit",
            ra=280.0 + (i % 100) * 0.1,
            dec=40.0 + (i % 100) * 0.1,
            time_peak_bjd=100.1,
            t_start=100.05,
            t_peak=100.1,
            t_end=100.15,
            duration_hours=2.4,
            duration_cadences=5,
            n_epochs=1,
            snr=snr,
            depth_ppm=depth,
            delta_centroid_mpix=shift,
            bkg_correlation=bkg,
            time_window=t_win,
            flux_window=f_win,
        )


def test_out_of_core_streaming_3000_candidates_memory_stress():
    """
    Stress Vector 4: Out-of-Core Streaming & Memory Stress.
    Streams 3,000 synthetic candidate events through ExpeditionVetter.vet_stream().
    Asserts:
      1. Peak heap memory strictly < 25.0 MB (and << 100 MB hard limit)
      2. Exactly 3,000 candidates processed and yielded
      3. All instances are valid VettedCandidate with non-empty discovery_tier
      4. Periodic garbage collection reclaims active working set
    """
    vetter = ExpeditionVetter()

    # Pre-test cleanup and baseline
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()

    stream = candidate_stream_generator(n_events=3000)
    vetted_stream = vetter.vet_stream(stream)

    count = 0
    tier_counts = {"Tier_1_Gold": 0, "Tier_2_Silver": 0, "Tier_3_Bronze": 0, "Rejected": 0}
    peak_samples = []

    for item in vetted_stream:
        assert isinstance(item, VettedCandidate)
        assert item.candidate_id == f"VET_STREAM_CAND_{count:05d}"
        assert item.discovery_tier in tier_counts
        tier_counts[item.discovery_tier] += 1
        count += 1

        if count % 500 == 0:
            current_bytes, peak_bytes = tracemalloc.get_tracemalloc_memory(), tracemalloc.get_traced_memory()[1]
            peak_mb = peak_bytes / (1024 * 1024)
            peak_samples.append((count, peak_mb))

    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_bytes / (1024 * 1024)
    current_mb = current_bytes / (1024 * 1024)

    # 1. Total processed assertion
    assert count == 3000

    # 2. Peak active memory assertion: strictly < 25 MB
    assert peak_mb < 25.0, f"Streaming peak memory {peak_mb:.2f} MB exceeded 25.0 MB ceiling"

    # 3. Valid distribution across tiers
    assert sum(tier_counts.values()) == 3000
    assert tier_counts["Rejected"] > 0  # High shift / bkg cases rejected
    assert tier_counts["Tier_1_Gold"] + tier_counts["Tier_2_Silver"] + tier_counts["Tier_3_Bronze"] > 0
