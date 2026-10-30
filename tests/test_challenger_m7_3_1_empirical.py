r"""
Challenger 1 Empirical Stress Test Suite for Milestone M7.3
============================================================
Adversarial Verification of:
  1. Ephemeris Wrap-Around & Boundary Tolerance (N=1..500, epsilon boundaries, secondary eclipses, multi-planet hosts)
  2. Astrophysical Consistency & BEB Filtering (Odd-even parity, Tier 1 Gold transits, Monotransits, Track B flares, Unphysical duration)

Author: Challenger 1 (teamwork_challenger_m7_3_1)
Target Milestone: M7.3 (agent-3-vetter)
"""

import math
from pathlib import Path
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
    InstrumentalVettingReceipt,
    PhysicalConsistencyEngine,
    PhysicalVettingReceipt,
    VettedCandidate,
    compute_composite_vetting_score,
    vet_candidate,
)


# ==============================================================================
# Vector 1: Ephemeris Wrap-Around & Boundary Tolerance
# ==============================================================================

def test_ephemeris_cycle_matching_across_hundreds_of_periods():
    """
    Stress Vector 1.1: Ephemeris Wrap-Around across hundreds of periods.
    Tests cycle indices N in [1, 10, 50, 250, 500] and negative cycles [-1, -50].
    Asserts:
      - cycle_index is exact integer N
      - is_match is True
      - taxonomy_class is 'catalog_confirmed_planet'
      - phase_offset_days is within floating point precision (< 1e-10)
    """
    matcher = EphemerisMatcher()

    # Use canonical benchmark HAT-P-7 b: P = 2.204733 d, t0 = 120.045
    p_hat7 = 2.204733
    t0_hat7 = 120.045

    test_cycles = [1, 10, 50, 250, 500, -1, -50]

    for N in test_cycles:
        t_cand = t0_hat7 + N * p_hat7
        cand = CandidateAnomaly(
            candidate_id=f"STRESS_HAT7_N_{N}",
            target_id="KIC_10666592",
            track="Track_A",
            survey="Kepler",
            anomaly_type="tertiary_eclipse",
            ra=292.298,
            dec=47.969,
            time_peak_bjd=t_cand,
            t_start=t_cand - 0.1,
            t_peak=t_cand,
            t_end=t_cand + 0.1,
            duration_hours=4.1,
            duration_cadences=8,
            snr=15.0,
            amplitude=6900.0,
            depth_ppm=6900.0,
        )

        match_res = matcher.cross_match(cand)

        assert match_res["is_match"] is True, f"Failed match at N={N}"
        assert match_res["cycle_index"] == N, f"Cycle index mismatch at N={N}: got {match_res['cycle_index']}"
        assert match_res["taxonomy_class"] == "catalog_confirmed_planet"
        assert abs(match_res["phase_offset_days"]) < 1e-10, f"Non-zero phase offset at N={N}: {match_res['phase_offset_days']}"
        assert match_res["disposition"] == "CONFIRMED_KNOWN"
        assert match_res["is_secondary"] is False


def test_ephemeris_epsilon_boundary_tolerance():
    """
    Stress Vector 1.2: Epsilon boundary matching tolerance.
    For dynamic tolerance window delta_t_window:
      - Perturbation dt = dt_window - 1e-5 must match (is_match = True)
      - Perturbation dt = dt_window + 1e-5 must NOT match (is_match = False)
    Tests both positive and negative perturbations, at N=1 and N=100.
    """
    matcher = EphemerisMatcher()
    p_hat7 = 2.204733
    t0_hat7 = 120.045
    dur_days = 4.1 / 24.0

    for N in [1, 100]:
        sigma_eph = math.sqrt(1e-4**2 + (N * 1e-6)**2)
        base_window = max(0.50 * dur_days, 0.02 * p_hat7, 0.0408) + 3.0 * sigma_eph
        dt_window = min(base_window, 0.20 * p_hat7, 1.0)

        eps = 1e-5

        # Case A: Inside positive boundary (+dt_window - eps) -> MUST MATCH
        t_inside_pos = t0_hat7 + N * p_hat7 + (dt_window - eps)
        cand_in_pos = CandidateAnomaly(
            candidate_id=f"STRESS_EPS_IN_POS_N{N}",
            target_id="KIC_10666592",
            track="Track_A",
            survey="Kepler",
            anomaly_type="tertiary_eclipse",
            ra=292.298,
            dec=47.969,
            time_peak_bjd=t_inside_pos,
            t_start=t_inside_pos - 0.05,
            t_peak=t_inside_pos,
            t_end=t_inside_pos + 0.05,
            duration_hours=4.1,
            duration_cadences=8,
            snr=10.0,
            amplitude=6900.0,
            depth_ppm=6900.0,
        )
        res_in_pos = matcher.cross_match(cand_in_pos)
        assert res_in_pos["is_match"] is True, f"Boundary failure: dt_window - eps should match at N={N}"
        assert res_in_pos["cycle_index"] == N

        # Case B: Outside positive boundary (+dt_window + eps) -> MUST NOT MATCH
        t_outside_pos = t0_hat7 + N * p_hat7 + (dt_window + eps)
        cand_out_pos = CandidateAnomaly(
            candidate_id=f"STRESS_EPS_OUT_POS_N{N}",
            target_id="KIC_10666592",
            track="Track_A",
            survey="Kepler",
            anomaly_type="tertiary_eclipse",
            ra=292.298,
            dec=47.969,
            time_peak_bjd=t_outside_pos,
            t_start=t_outside_pos - 0.05,
            t_peak=t_outside_pos,
            t_end=t_outside_pos + 0.05,
            duration_hours=4.1,
            duration_cadences=8,
            snr=10.0,
            amplitude=6900.0,
            depth_ppm=6900.0,
        )
        res_out_pos = matcher.cross_match(cand_out_pos)
        assert res_out_pos["is_match"] is False, f"Boundary failure: dt_window + eps should NOT match at N={N}"

        # Case C: Inside negative boundary (-dt_window + eps) -> MUST MATCH
        t_inside_neg = t0_hat7 + N * p_hat7 - (dt_window - eps)
        cand_in_neg = CandidateAnomaly(
            candidate_id=f"STRESS_EPS_IN_NEG_N{N}",
            target_id="KIC_10666592",
            track="Track_A",
            survey="Kepler",
            anomaly_type="tertiary_eclipse",
            ra=292.298,
            dec=47.969,
            time_peak_bjd=t_inside_neg,
            t_start=t_inside_neg - 0.05,
            t_peak=t_inside_neg,
            t_end=t_inside_neg + 0.05,
            duration_hours=4.1,
            duration_cadences=8,
            snr=10.0,
            amplitude=6900.0,
            depth_ppm=6900.0,
        )
        res_in_neg = matcher.cross_match(cand_in_neg)
        assert res_in_neg["is_match"] is True, f"Boundary failure: -(dt_window - eps) should match at N={N}"
        assert res_in_neg["cycle_index"] == N

        # Case D: Outside negative boundary (-dt_window - eps) -> MUST NOT MATCH
        t_outside_neg = t0_hat7 + N * p_hat7 - (dt_window + eps)
        cand_out_neg = CandidateAnomaly(
            candidate_id=f"STRESS_EPS_OUT_NEG_N{N}",
            target_id="KIC_10666592",
            track="Track_A",
            survey="Kepler",
            anomaly_type="tertiary_eclipse",
            ra=292.298,
            dec=47.969,
            time_peak_bjd=t_outside_neg,
            t_start=t_outside_neg - 0.05,
            t_peak=t_outside_neg,
            t_end=t_outside_neg + 0.05,
            duration_hours=4.1,
            duration_cadences=8,
            snr=10.0,
            amplitude=6900.0,
            depth_ppm=6900.0,
        )
        res_out_neg = matcher.cross_match(cand_out_neg)
        assert res_out_neg["is_match"] is False, f"Boundary failure: -(dt_window + eps) should NOT match at N={N}"


def test_ephemeris_secondary_eclipse_matching_and_eccentric_phases():
    """
    Stress Vector 1.3: Secondary eclipse matching at phi = 0.5 and eccentric phases.
    Asserts:
      - Circular secondary at phi = 0.5 matches with is_secondary=True and taxonomy_class='catalog_confirmed_eb'.
      - Small eccentric secondary within tolerance (e.g. phi = 0.51) matches as secondary.
      - Large eccentric secondary outside tolerance (e.g. phi = 0.65) does not match secondary.
    """
    matcher = EphemerisMatcher()

    # KIC 5871143 EB benchmark: P = 4.25 d, t0 = 120.0 d, duration = 5.0 h
    p_eb = 4.25
    t0_eb = 120.0

    # 1. Circular Secondary at phi = 0.5 (N = 5)
    t_sec_circ = t0_eb + (5 + 0.50) * p_eb
    cand_sec_circ = CandidateAnomaly(
        candidate_id="STRESS_EB_SEC_CIRC",
        target_id="KIC_5871143",
        track="Track_A",
        survey="Kepler",
        anomaly_type="tertiary_eclipse",
        ra=290.0,
        dec=41.0,
        time_peak_bjd=t_sec_circ,
        t_start=t_sec_circ - 0.1,
        t_peak=t_sec_circ,
        t_end=t_sec_circ + 0.1,
        duration_hours=5.0,
        duration_cadences=10,
        snr=25.0,
        amplitude=25000.0,
        depth_ppm=25000.0,
    )
    res_circ = matcher.cross_match(cand_sec_circ)
    assert res_circ["is_match"] is True
    assert res_circ["is_secondary"] is True
    assert res_circ["taxonomy_class"] == "catalog_confirmed_eb"
    assert res_circ["cycle_index"] == 5

    # 2. Mildly eccentric secondary at phi = 0.51 (offset = 0.01 * 4.25 = 0.0425 d < dt_window)
    t_sec_ecc_in = t0_eb + (5 + 0.51) * p_eb
    cand_ecc_in = CandidateAnomaly(
        candidate_id="STRESS_EB_SEC_ECC_IN",
        target_id="KIC_5871143",
        track="Track_A",
        survey="Kepler",
        anomaly_type="tertiary_eclipse",
        ra=290.0,
        dec=41.0,
        time_peak_bjd=t_sec_ecc_in,
        t_start=t_sec_ecc_in - 0.1,
        t_peak=t_sec_ecc_in,
        t_end=t_sec_ecc_in + 0.1,
        duration_hours=5.0,
        duration_cadences=10,
        snr=25.0,
        amplitude=25000.0,
        depth_ppm=25000.0,
    )
    res_ecc_in = matcher.cross_match(cand_ecc_in)
    assert res_ecc_in["is_match"] is True
    assert res_ecc_in["is_secondary"] is True

    # 3. Highly eccentric secondary at phi = 0.65 (offset = 0.15 * 4.25 = 0.6375 d > dt_window)
    t_sec_ecc_out = t0_eb + (5 + 0.65) * p_eb
    cand_ecc_out = CandidateAnomaly(
        candidate_id="STRESS_EB_SEC_ECC_OUT",
        target_id="KIC_5871143",
        track="Track_A",
        survey="Kepler",
        anomaly_type="tertiary_eclipse",
        ra=290.0,
        dec=41.0,
        time_peak_bjd=t_sec_ecc_out,
        t_start=t_sec_ecc_out - 0.1,
        t_peak=t_sec_ecc_out,
        t_end=t_sec_ecc_out + 0.1,
        duration_hours=5.0,
        duration_cadences=10,
        snr=25.0,
        amplitude=25000.0,
        depth_ppm=25000.0,
    )
    res_ecc_out = matcher.cross_match(cand_ecc_out)
    assert res_ecc_out["is_match"] is False
    assert res_ecc_out["disposition"] == "NOVEL_CANDIDATE"


def test_ephemeris_multi_planet_additional_candidate():
    """
    Stress Vector 1.4: Candidate on known host with distinct period.
    Asserts:
      - Distinct period on known multi-planet host classifies as 'multi_planet_additional_candidate'.
      - Disposition is 'NOVEL_CANDIDATE'.
    """
    matcher = EphemerisMatcher()

    # Kepler-11 (KIC 6541920) has known planets b (P=10.30d) and c (P=13.02d).
    # Synthesize candidate with distinct period P = 23.5 d at t = 145.0
    cand = CandidateAnomaly(
        candidate_id="STRESS_KEPLER11_EXTRA_PLANET",
        target_id="KIC_6541920",
        track="Track_A",
        survey="Kepler",
        anomaly_type="circumbinary_transit",
        ra=296.068,
        dec=41.908,
        time_peak_bjd=145.0,
        t_start=144.8,
        t_peak=145.0,
        t_end=145.2,
        duration_hours=5.2,
        duration_cadences=11,
        snr=8.5,
        amplitude=800.0,
        depth_ppm=800.0,
        metadata={"period": 23.5},
    )

    res = matcher.cross_match(cand)

    assert res["is_match"] is False
    assert res["disposition"] == "NOVEL_CANDIDATE"
    assert res["taxonomy_class"] == "multi_planet_additional_candidate"


# ==============================================================================
# Vector 2: Astrophysical Consistency & BEB Filtering
# ==============================================================================

def test_astrophysical_consistency_beb_alternating_depths():
    """
    Stress Vector 2.1: Synthesize blended background eclipsing binary with alternating depths:
      d_odd = 10000 ppm, d_even = 8000 ppm, sigma = 200 ppm.
    Asserts:
      - Delta d = 2000 ppm
      - sigma_eff = sqrt(200^2 + 200^2) = 282.84 ppm
      - Delta d_sigma = 2000 / 282.84 = 7.071 sigma >= 3.0 sigma
      - triggers 'BEB' veto rejection
      - discovery_tier is 'Rejected'
    """
    d_odd = 10000.0
    d_even = 8000.0
    sigma = 200.0
    delta_d = abs(d_odd - d_even)
    sigma_eff = math.sqrt(sigma**2 + sigma**2)
    delta_d_sigma = delta_d / sigma_eff

    # Verify analytical value
    assert abs(delta_d_sigma - 7.0710678) < 1e-4
    assert delta_d_sigma >= 3.0

    cand = CandidateAnomaly(
        candidate_id="STRESS_BEB_ALTERNATING_DEPTH",
        target_id="KIC_9999999",
        track="Track_A",
        survey="Kepler",
        anomaly_type="tertiary_eclipse",
        ra=285.0,
        dec=45.0,
        t_start=120.0,
        t_peak=120.1,
        t_end=120.2,
        duration_hours=4.8,
        duration_cadences=10,
        depth_ppm=9000.0,
        metadata={
            "depth_odd_ppm": d_odd,
            "depth_even_ppm": d_even,
            "odd_even_sigma": delta_d_sigma,
            "catalog_period": 3.5,
        },
    )

    receipt = PhysicalConsistencyEngine.evaluate(cand)

    assert receipt.odd_even_passed is False
    assert "BEB" in receipt.veto_reasons
    assert receipt.is_physical_fp is True
    assert abs(receipt.odd_even_sigma - delta_d_sigma) < 1e-5

    # Check composite score rejection
    inst_receipt = InstrumentalVettingReceipt(
        delta_centroid_mpix=0.5,
        centroid_snr=1.0,
        delta_x_mpix=0.3,
        delta_y_mpix=0.4,
        centroid_sigma_mpix=0.1,
        centroid_passed=True,
        bkg_correlation=0.05,
        bkg_relative_amp=0.01,
        bkg_passed=True,
        quality_flag_clean=True,
        fatal_quality_bits=0,
        min_dump_distance_hours=10.0,
        thruster_veto_passed=True,
        argabrightening_detected=False,
        instrumental_score=1.0,
        is_instrumental_fp=False,
        veto_reasons=[],
    )

    score, tier, reason = compute_composite_vetting_score(cand, inst_receipt, receipt)
    assert score == 0.0
    assert tier == "Rejected"
    assert reason == "BEB"


def test_astrophysical_consistency_genuine_exoplanet_tier1_gold():
    """
    Stress Vector 2.2: Synthesize genuine exoplanet transits:
      d_odd = 1000 ppm, d_even = 1005 ppm, sigma = 100 ppm.
    Asserts:
      - Delta d_sigma = 5 / 141.42 = 0.035 sigma < 2.0 sigma
      - qualifies for Tier 1 Gold Discovery
      - vetting_score >= 0.85
    """
    d_odd = 1000.0
    d_even = 1005.0
    sigma = 100.0
    delta_d_sigma = abs(d_odd - d_even) / math.sqrt(2 * sigma**2)
    assert delta_d_sigma < 2.0

    t_win = np.linspace(120.0, 120.5, 60)
    f_win = np.ones(60)
    # U-shaped transit: flat bottom from idx 22 to 38
    f_win[18:22] = np.linspace(1.0, 0.999, 4)
    f_win[22:38] = 0.999  # 1000 ppm flat bottom
    f_win[38:42] = np.linspace(0.999, 1.0, 4)

    cand = CandidateAnomaly(
        candidate_id="STRESS_GENUINE_EXOPLANET_GOLD",
        target_id="KIC_8888888",
        track="Track_A",
        survey="Kepler",
        anomaly_type="circumbinary_transit",
        ra=285.0,
        dec=45.0,
        time_peak_bjd=120.25,
        t_start=120.15,
        t_peak=120.25,
        t_end=120.35,
        duration_hours=4.8,
        duration_cadences=10,
        snr=12.0,  # >= 7.1
        amplitude=1000.0,
        depth_ppm=1000.0,
        delta_centroid_mpix=0.4,  # < 1.5 mpix
        bkg_correlation=0.05,     # < 0.20
        quality_flag_clean=True,
        time_window=t_win,
        flux_window=f_win,
        metadata={
            "depth_odd_ppm": d_odd,
            "depth_even_ppm": d_even,
            "odd_even_sigma": delta_d_sigma,
            "catalog_period": 5.0,
        },
    )

    phys_receipt = PhysicalConsistencyEngine.evaluate(cand, period_days=5.0)
    assert phys_receipt.odd_even_passed is True
    assert phys_receipt.odd_even_sigma < 2.0
    assert phys_receipt.morphology_shape == "U-shape"
    assert phys_receipt.is_physical_fp is False

    inst_receipt = InstrumentalFalsePositiveFilter.evaluate(cand)
    assert inst_receipt.is_instrumental_fp is False
    assert inst_receipt.delta_centroid_mpix < 1.5
    assert abs(inst_receipt.bkg_correlation) < 0.20

    score, tier, reason = compute_composite_vetting_score(cand, inst_receipt, phys_receipt)

    assert tier == "Tier_1_Gold", f"Expected Tier_1_Gold, got {tier}"
    assert score >= 0.85, f"Expected vetting_score >= 0.85, got {score}"
    assert reason is None


def test_astrophysical_consistency_monotransit_and_track_b_microflare_bypass():
    """
    Stress Vector 2.3: Single-epoch monotransit (n_epochs=1) and Track B micro-flares.
    Asserts:
      - Odd-even parity test gracefully bypasses without false BEB rejection (evaluable_oe = False).
      - Monotransit with clean instrumental profile passes physical consistency.
      - Track B micro-flare with asymmetric impulse passes physical consistency without BEB rejection.
    """
    # 1. Monotransit (n_epochs = 1, period is None)
    t_win = np.linspace(200.0, 200.5, 50)
    f_win = np.ones(50)
    f_win[20:30] -= 0.002  # 2000 ppm dip

    monotransit = CandidateAnomaly(
        candidate_id="STRESS_MONOTRANSIT_SINGLE_EPOCH",
        target_id="TIC_12345678",
        track="Track_A",
        survey="TESS",
        anomaly_type="circumbinary_transit",
        ra=95.0,
        dec=-60.0,
        time_peak_bjd=200.25,
        t_start=200.15,
        t_peak=200.25,
        t_end=200.35,
        duration_hours=4.8,
        duration_cadences=10,
        n_epochs=1,
        snr=10.0,
        amplitude=2000.0,
        depth_ppm=2000.0,
        delta_centroid_mpix=0.3,
        bkg_correlation=0.02,
        quality_flag_clean=True,
        time_window=t_win,
        flux_window=f_win,
        metadata={},  # No odd-even data
    )

    mono_receipt = PhysicalConsistencyEngine.evaluate(monotransit)
    assert mono_receipt.odd_even_passed is True
    assert mono_receipt.odd_even_sigma == 0.0
    assert "BEB" not in mono_receipt.veto_reasons
    assert mono_receipt.is_physical_fp is False

    # 2. Track B Micro-flare (non-periodic impulsive event)
    t_flare = np.linspace(150.0, 150.05, 50)
    f_flare = np.ones(50)
    # Rapid rise, exponential decay
    f_flare[15:18] += np.linspace(0.0, 0.015, 3)
    f_flare[18:35] += 0.015 * np.exp(-np.linspace(0, 3, 17))

    microflare = CandidateAnomaly(
        candidate_id="STRESS_TRACK_B_MICROFLARE",
        target_id="KIC_FLARE_001",
        track="Track_B",
        survey="Kepler",
        anomaly_type="micro_flare",
        ra=290.0,
        dec=40.0,
        time_peak_bjd=150.018,
        t_start=150.015,
        t_peak=150.018,
        t_end=150.035,
        duration_hours=0.48,
        duration_cadences=5,
        snr=14.0,
        amplitude=15000.0,
        depth_ppm=0.0,
        delta_centroid_mpix=0.2,
        bkg_correlation=0.01,
        quality_flag_clean=True,
        time_window=t_flare,
        flux_window=f_flare,
        metadata={"category": "micro_flare"},
    )

    flare_receipt = PhysicalConsistencyEngine.evaluate(microflare)
    assert flare_receipt.odd_even_passed is True
    assert flare_receipt.morphology_shape == "asymmetric_flare"
    assert "BEB" not in flare_receipt.veto_reasons
    assert flare_receipt.is_physical_fp is False


def test_astrophysical_consistency_unphysical_duration():
    """
    Stress Vector 2.4: Unphysical duration (R_dur = 4.5 on solar dwarf).
    Asserts:
      - R_dur = 4.5 > 3.0 triggers 'UNPHYSICAL_DURATION' veto.
      - duration_passed is False.
      - discovery_tier is 'Rejected'.
    """
    period_days = 2.0
    depth_ppm = 2500.0
    delta = depth_ppm * 1e-6
    # Solar dwarf circular duration bound:
    t_circ = 1.791 * (1.0 + math.sqrt(delta)) * (period_days ** (1.0 / 3.0))  # ~2.37 hours
    # Synthesize unphysical observed duration with R_dur = 4.5:
    dur_unphysical = 4.5 * t_circ  # ~10.66 hours

    cand = CandidateAnomaly(
        candidate_id="STRESS_UNPHYSICAL_DURATION",
        target_id="KIC_7777777",
        track="Track_A",
        survey="Kepler",
        anomaly_type="circumbinary_transit",
        ra=285.0,
        dec=45.0,
        time_peak_bjd=120.5,
        t_start=120.0,
        t_peak=120.5,
        t_end=121.0,
        duration_hours=dur_unphysical,
        duration_cadences=22,
        snr=8.0,
        amplitude=depth_ppm,
        depth_ppm=depth_ppm,
        metadata={"catalog_period": period_days},
    )

    phys_receipt = PhysicalConsistencyEngine.evaluate(cand, period_days=period_days)

    assert phys_receipt.duration_passed is False
    assert abs(phys_receipt.duration_ratio - 4.5) < 1e-4
    assert "UNPHYSICAL_DURATION" in phys_receipt.veto_reasons
    assert phys_receipt.is_physical_fp is True

    # Composite score check
    inst_receipt = InstrumentalVettingReceipt(
        delta_centroid_mpix=0.5,
        centroid_snr=1.0,
        delta_x_mpix=0.3,
        delta_y_mpix=0.4,
        centroid_sigma_mpix=0.1,
        centroid_passed=True,
        bkg_correlation=0.05,
        bkg_relative_amp=0.01,
        bkg_passed=True,
        quality_flag_clean=True,
        fatal_quality_bits=0,
        min_dump_distance_hours=10.0,
        thruster_veto_passed=True,
        argabrightening_detected=False,
        instrumental_score=1.0,
        is_instrumental_fp=False,
        veto_reasons=[],
    )

    score, tier, reason = compute_composite_vetting_score(cand, inst_receipt, phys_receipt)
    assert score == 0.0
    assert tier == "Rejected"
    assert reason == "UNPHYSICAL_DURATION"
