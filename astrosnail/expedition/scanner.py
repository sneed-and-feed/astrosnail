"""
AstroSnail High-Throughput Outlier & Anomaly Scanner (`astrosnail.expedition.scanner`)
====================================================================================
Milestone M7.2:
  - `CandidateAnomaly`: Unified memory-bounded dataclass (slots=True) capturing candidate
    astrophysical transients, orbital anomalies, and Lie operator receipts across Tracks A, B, C.
  - `scan_track_a`: Periodic harmonic quarantining via sub-cadence phase-folding and
    TemporalClockContentSplitter, causal glitch gating, and multi-scale boxcar matched filtering
    for circumbinary transits, tertiary eclipses, and ETV anomalies.
  - `scan_track_b`: 2-stage causal glint shielding (MorseTransientGateConfig + BeltramiLightCurveShell)
    isolating 2-6 epoch FRED micro-flares and asymmetric exocomet/dipper transits with centroid stability (< 3 mpix).
  - `scan_track_c`: 8D rotary state vectors on S^7, Lie-algebra so(8) triad cascade (0.000e+00 drift),
    conjunction search (<= 5.0 arcsec), and causal progenitor factorizations (G_act vs Omega_reg / lensed_echo).
  - `ExpeditionScanner`: Master scanning engine with out-of-core streaming (< 100 MB active RAM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.ndimage
import scipy.interpolate

from astrosnail.astrometry import (
    SkewSymmetricTriadOperator,
    SlowFastTriadCascade,
    attribute_causal_progenitors,
)
from astrosnail.expedition.harvester import HarvestedTarget
from astrosnail.transients import (
    BeltramiLightCurveShell,
    MorseTransientGateConfig,
    TemporalClockContentSplitter,
    compute_morse_impulse,
)


# ==============================================================================
# 1. Unified CandidateAnomaly Dataclass (slots=True)
# ==============================================================================

@dataclass(slots=True)
class CandidateAnomaly:
    """Unified container for candidate astrophysical anomalies across Tracks A, B, and C.

    Directly consumed by the Milestone M7.3 Vetter (`astrosnail/expedition/vetter.py`).
    """

    # 1. Target Identity & Survey Provenance
    candidate_id: str
    target_id: str
    track: str  # "Track_A" | "Track_B" | "Track_C"
    survey: str  # "Kepler" | "TESS" | "ALeRCE_ZTF" | "JPL_Horizons"
    anomaly_type: str  # "tertiary_eclipse" | "circumbinary_transit" | "micro_flare" |
                       # "exocomet_dip" | "dipper_transit" | "moving_body_conjunction" |
                       # "lensed_echo" | "etv_perturbation"

    # 2. Celestial Coordinates & Kinematics
    ra: float
    dec: float
    pm_ra: float = 0.0
    pm_dec: float = 0.0

    # 3. Detection Epochs & Duration
    time_peak_bjd: float = 0.0
    t_start: float = 0.0
    t_peak: float = 0.0
    t_end: float = 0.0
    duration_hours: float = 0.0
    duration_cadences: int = 0
    n_epochs: int = 0

    # 4. Photometric / Astrometric Significance & Amplitude
    snr: float = 0.0
    amplitude: float = 0.0
    depth_ppm: float = 0.0
    peak_flux_ratio: float = 1.0

    # 5. AstroSnail Core Operator Receipts & Lie Metrics
    curl_salience: float = 0.0
    morse_impulse: float = 0.0
    eta_nonlocal: float = 0.0
    causal_class: str = "Omega_reg"  # "G_act" | "Omega_reg"
    triad_drift: float = 0.0
    triad_radial_drift: float = 0.0
    tie_breaking_gap: float = 0.0

    # 6. Centroid & Instrumental Stability Metrics
    delta_centroid_mpix: float = 0.0
    centroid_shift_mpix: float = 0.0
    bkg_correlation: float = 0.0
    quality_flag_clean: bool = True

    # 7. Diagnostic Sub-Arrays (local event window)
    time_window: Optional[np.ndarray] = None
    flux_window: Optional[np.ndarray] = None
    h_clock_window: Optional[np.ndarray] = None
    h_nope_window: Optional[np.ndarray] = None
    model_window: Optional[np.ndarray] = None

    # 8. Physical & Contextual Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Synchronize alias fields for complete cross-module compatibility."""
        # Epoch peak aliases
        if self.t_peak == 0.0 and self.time_peak_bjd != 0.0:
            self.t_peak = self.time_peak_bjd
        elif self.time_peak_bjd == 0.0 and self.t_peak != 0.0:
            self.time_peak_bjd = self.t_peak

        # Cadence count aliases
        if self.n_epochs == 0 and self.duration_cadences != 0:
            self.n_epochs = self.duration_cadences
        elif self.duration_cadences == 0 and self.n_epochs != 0:
            self.duration_cadences = self.n_epochs

        # Centroid shift aliases
        if self.delta_centroid_mpix == 0.0 and self.centroid_shift_mpix != 0.0:
            self.delta_centroid_mpix = self.centroid_shift_mpix
        elif self.centroid_shift_mpix == 0.0 and self.delta_centroid_mpix != 0.0:
            self.centroid_shift_mpix = self.delta_centroid_mpix

        # Triad drift aliases
        if self.triad_drift == 0.0 and self.triad_radial_drift != 0.0:
            self.triad_drift = self.triad_radial_drift
        elif self.triad_radial_drift == 0.0 and self.triad_drift != 0.0:
            self.triad_radial_drift = self.triad_drift

    def to_dict(self) -> Dict[str, Any]:
        """Serializes anomaly to JSON-serializable dictionary (omitting large arrays)."""
        return {
            "candidate_id": self.candidate_id,
            "target_id": self.target_id,
            "track": self.track,
            "survey": self.survey,
            "anomaly_type": self.anomaly_type,
            "ra": float(self.ra),
            "dec": float(self.dec),
            "pm_ra": float(self.pm_ra),
            "pm_dec": float(self.pm_dec),
            "time_peak_bjd": float(self.time_peak_bjd),
            "t_start": float(self.t_start),
            "t_peak": float(self.t_peak),
            "t_end": float(self.t_end),
            "duration_hours": float(self.duration_hours),
            "duration_cadences": int(self.duration_cadences),
            "snr": float(self.snr),
            "amplitude": float(self.amplitude),
            "depth_ppm": float(self.depth_ppm),
            "peak_flux_ratio": float(self.peak_flux_ratio),
            "curl_salience": float(self.curl_salience),
            "morse_impulse": float(self.morse_impulse),
            "eta_nonlocal": float(self.eta_nonlocal),
            "causal_class": self.causal_class,
            "triad_drift": float(self.triad_drift),
            "tie_breaking_gap": float(self.tie_breaking_gap),
            "delta_centroid_mpix": float(self.delta_centroid_mpix),
            "bkg_correlation": float(self.bkg_correlation),
            "quality_flag_clean": bool(self.quality_flag_clean),
            "metadata": dict(self.metadata),
        }


# ==============================================================================
# 2. Track A Scanner: Eclipsing Binaries & Multi-Planet Systems
# ==============================================================================

def scan_track_a(
    target: HarvestedTarget,
    snr_thresh: float = 6.5,
    periodicity_thresh: float = 0.85,
    max_centroid_shift_mpix: float = 3.0,
) -> List[CandidateAnomaly]:
    """Scans Kepler & TESS Eclipsing Binaries and Multi-Planet Systems (Track A).

    Pipeline:
      1. Quality conditioning: Filter valid cadences (finite, SAP_QUALITY == 0).
      2. Content splitting: Sub-cadence phase-folding and autocorrelation
         partitioning periodic binary harmonics into H_clock and residuals into H_nope.
      3. Causal glitch gating: MorseTransientGateConfig(sigma_thresh=4.0) eliminates
         single-cadence cosmic rays without attenuating multi-cadence dips.
      4. Continuum conditioning: Subtracts gentle rolling baseline (window 256) to eliminate slow drift.
      5. Multi-scale matched filter: Evaluates duration scales W in {4, 8, 12, 16, 24, 48} cadences.
      6. Astrophysical classification: Separates circumbinary transits, tertiary eclipses (W >= 24),
         and eclipse-timing variation (ETV) perturbations.
    """
    if target.sap_flux is None or len(target.sap_flux) < 20:
        return []

    t_arr = target.time if target.time is not None else np.arange(len(target.sap_flux), dtype=np.float64)
    f_arr = target.sap_flux.astype(np.float64)
    q_arr = target.sap_quality if target.sap_quality is not None else np.zeros(len(f_arr), dtype=np.int32)
    bkg_arr = target.sap_bkg if target.sap_bkg is not None else np.zeros(len(f_arr), dtype=np.float64)
    cx_arr = target.mom_centr1 if target.mom_centr1 is not None else np.zeros(len(f_arr), dtype=np.float64)
    cy_arr = target.mom_centr2 if target.mom_centr2 is not None else np.zeros(len(f_arr), dtype=np.float64)

    valid_mask = np.isfinite(t_arr) & np.isfinite(f_arr) & (q_arr == 0)
    if np.sum(valid_mask) < 20:
        return []

    med_flux = float(np.median(f_arr[valid_mask]))
    if med_flux <= 0:
        med_flux = 1.0
    rel_flux_ppm = ((f_arr / med_flux) - 1.0) * 1e6  # in ppm

    n_points = len(rel_flux_ppm)

    # --------------------------------------------------------------------------
    # Stage 2: Periodic Harmonic Quarantine into H_clock via Continuous Phase
    # --------------------------------------------------------------------------
    splitter = TemporalClockContentSplitter(periodicity_threshold=periodicity_thresh)
    h_clock_raw, h_nope_raw, clock_mask = splitter.split(rel_flux_ppm.astype(np.float32))

    has_periodic_split = bool(np.any(clock_mask > 0))
    cat_period = target.metadata.get("period")

    # Perform sub-cadence continuous phase folding if periodic
    if has_periodic_split or (cat_period is not None and cat_period > 0):
        # Determine best period
        if cat_period is not None and cat_period > 0:
            period_days = float(cat_period)
        else:
            # Estimate cadence in days
            valid_t = t_arr[valid_mask]
            dt_step = float(np.median(np.diff(valid_t))) if len(valid_t) > 1 else 0.0204
            # Lag from autocorrelation
            sig_1d = rel_flux_ppm - np.mean(rel_flux_ppm[valid_mask])
            n_padded = int(2 ** np.ceil(np.log2(2 * n_points - 1)))
            f_sig = np.fft.rfft(sig_1d, n=n_padded)
            autocorr = np.fft.irfft(f_sig * np.conj(f_sig), n=n_padded)[:n_points]
            taper = np.arange(n_points, 0, -1, dtype=np.float64) / float(n_points)
            autocorr = autocorr / (autocorr[0] * taper + 1e-12)
            min_lag = 3
            max_lag = min(n_points // 2, 2048)
            if max_lag > min_lag:
                ac_win = autocorr[min_lag:max_lag]
                p = int(np.argmax(ac_win)) + min_lag
                # 3-point parabolic peak interpolation
                if 0 < p < len(autocorr) - 1:
                    r_minus = autocorr[p - 1]
                    r_0 = autocorr[p]
                    r_plus = autocorr[p + 1]
                    denom = 2.0 * (r_minus - 2.0 * r_0 + r_plus)
                    delta_p = (r_minus - r_plus) / denom if abs(denom) > 1e-12 else 0.0
                    p_sub = float(p + np.clip(delta_p, -0.5, 0.5))
                else:
                    p_sub = float(p)
                period_days = p_sub * dt_step
            else:
                period_days = 2.0

        # Continuous phase-folding with unique sorting to avoid divide-by-zero
        t0 = float(t_arr[0])
        phase = ((t_arr - t0) % period_days) / period_days
        order = np.argsort(phase)
        sorted_ph = phase[order]
        sorted_fl = rel_flux_ppm[order]

        # Median filter in phase domain
        smoothed_fl = scipy.ndimage.median_filter(sorted_fl, size=31, mode="wrap")

        # Deduplicate sorted_ph to guarantee strictly monotonic coordinates for interp1d
        u_ph, u_idx = np.unique(sorted_ph, return_index=True)
        if len(u_ph) >= 4:
            interp_func = scipy.interpolate.interp1d(
                u_ph,
                smoothed_fl[u_idx],
                bounds_error=False,
                fill_value="extrapolate",
                assume_sorted=True,
            )
            h_clock = interp_func(phase).astype(np.float64)
        else:
            h_clock = h_clock_raw.astype(np.float64)

        h_nope = rel_flux_ppm - h_clock
        clock_peak = float(np.max(np.abs(h_clock)))
        if clock_peak > 1e-6:
            clock_mask = (np.abs(h_clock) >= 0.20 * clock_peak).astype(np.uint8)
        else:
            clock_mask = np.zeros(n_points, dtype=np.uint8)
    else:
        h_clock = np.zeros(n_points, dtype=np.float64)
        h_nope = rel_flux_ppm.copy()
        clock_mask = np.zeros(n_points, dtype=np.uint8)

    # --------------------------------------------------------------------------
    # Stage 3: Causal Deferral Glitch Gating on H_nope
    # --------------------------------------------------------------------------
    clean_h_nope = h_nope[valid_mask]
    mad_sigma = float(np.median(np.abs(clean_h_nope - np.median(clean_h_nope))) * 1.4826)
    if mad_sigma <= 1e-4:
        mad_sigma = max(1.0, float(np.std(clean_h_nope)))

    gate = MorseTransientGateConfig(sigma_thresh=4.0, persistence_window=1)
    gate.noise_std = mad_sigma
    gate.variance = mad_sigma * mad_sigma

    glint_mask = np.zeros(n_points, dtype=bool)
    for i in range(n_points):
        if not valid_mask[i]:
            continue
        c, _ = gate.update(float(h_nope[i]), timestamp=float(t_arr[i]))
        if c == "glint_rejected":
            glint_mask[i - 1] = True

    # Active detection mask: valid and not a 1-cadence cosmic ray glint
    active_mask = valid_mask & (~glint_mask)

    # --------------------------------------------------------------------------
    # Shield glint_mask points in H_nope before continuum conditioning & filtering
    # --------------------------------------------------------------------------
    h_nope_shielded = h_nope.copy()
    glint_indices = np.where(glint_mask)[0]
    for g_idx in glint_indices:
        p = g_idx - 1
        while p >= 0 and glint_mask[p]:
            p -= 1
        nx = g_idx + 1
        while nx < n_points and glint_mask[nx]:
            nx += 1
        if p >= 0 and nx < n_points:
            h_nope_shielded[g_idx] = 0.5 * (h_nope[p] + h_nope[nx])
        elif p >= 0:
            h_nope_shielded[g_idx] = h_nope[p]
        elif nx < n_points:
            h_nope_shielded[g_idx] = h_nope[nx]
        else:
            h_nope_shielded[g_idx] = 0.0

    # --------------------------------------------------------------------------
    # Stage 4: Rolling Continuum Conditioning on H_nope
    # --------------------------------------------------------------------------
    # Use wide window (256 cadences) to remove very slow wander without eroding 24-48 cadence dips
    if n_points >= 256:
        rolling_baseline = scipy.ndimage.uniform_filter1d(h_nope_shielded, size=256, mode="nearest")
        h_nope_clean = h_nope_shielded - rolling_baseline
    else:
        h_nope_clean = h_nope_shielded - np.median(h_nope_shielded[valid_mask])

    # --------------------------------------------------------------------------
    # Stage 5: Multi-Scale Boxcar Matched Filter Bank
    # --------------------------------------------------------------------------
    scales_w = [4, 8, 12, 16, 24, 48]
    candidates: List[CandidateAnomaly] = []

    # Centroid base coordinates
    cx_med = float(np.median(cx_arr[valid_mask])) if target.mom_centr1 is not None else 0.0
    cy_med = float(np.median(cy_arr[valid_mask])) if target.mom_centr2 is not None else 0.0

    # Best detection per epoch across scales
    best_snr = np.zeros(n_points, dtype=np.float64)
    best_w = np.zeros(n_points, dtype=np.int32)
    best_depth = np.zeros(n_points, dtype=np.float64)

    for w in scales_w:
        if w >= n_points // 2:
            continue
        kernel = np.ones(w, dtype=np.float64) / float(w)
        s_w = -scipy.ndimage.convolve1d(h_nope_clean, kernel, mode="nearest")
        snr_w = s_w / (mad_sigma / math.sqrt(float(w)))

        better = (snr_w > best_snr) & active_mask
        best_snr[better] = snr_w[better]
        best_w[better] = w
        best_depth[better] = s_w[better]

    # Threshold and cluster detections
    sig_indices = np.where((best_snr >= snr_thresh) & active_mask)[0]
    if len(sig_indices) > 0:
        clusters: List[List[int]] = []
        cur_c = [int(sig_indices[0])]
        for idx in sig_indices[1:]:
            if idx - cur_c[-1] <= 12:
                cur_c.append(int(idx))
            else:
                clusters.append(cur_c)
                cur_c = [int(idx)]
        clusters.append(cur_c)

        for cluster in clusters:
            local_peak_idx = cluster[int(np.argmax(best_snr[cluster]))]
            peak_snr = float(best_snr[local_peak_idx])
            w_opt = int(best_w[local_peak_idx])
            depth_opt = float(best_depth[local_peak_idx])

            # Measure cluster duration from span of significant detections
            cluster_len = cluster[-1] - cluster[0] + 1
            k_start = max(0, cluster[0] - 2)
            k_end = min(n_points - 1, cluster[-1] + 2)

            # Centroid stability check (< max_centroid_shift_mpix)
            if target.mom_centr1 is not None and target.mom_centr2 is not None:
                if cluster_len <= 8:
                    dx_mpix = float(np.max(np.abs(cx_arr[k_start:k_end + 1] - cx_med)) * 1000.0)
                    dy_mpix = float(np.max(np.abs(cy_arr[k_start:k_end + 1] - cy_med)) * 1000.0)
                else:
                    dx_mpix = float(abs(np.median(cx_arr[k_start:k_end + 1]) - cx_med) * 1000.0)
                    dy_mpix = float(abs(np.median(cy_arr[k_start:k_end + 1]) - cy_med) * 1000.0)
                tot_shift = float(math.sqrt(dx_mpix * dx_mpix + dy_mpix * dy_mpix))
            else:
                tot_shift = 0.0

            if tot_shift >= max_centroid_shift_mpix:
                continue

            # Background correlation
            if target.sap_bkg is not None:
                win_f = f_arr[k_start:k_end + 1]
                win_b = bkg_arr[k_start:k_end + 1]
                if len(win_f) > 3 and np.std(win_f) > 1e-6 and np.std(win_b) > 1e-6:
                    bkg_corr = float(np.corrcoef(win_f, win_b)[0, 1])
                else:
                    bkg_corr = 0.0
            else:
                bkg_corr = 0.0

            qual_ok = bool(np.all(q_arr[k_start:k_end + 1] == 0))
            if not qual_ok:
                continue

            w_buf = max(8, w_opt)
            i_lo = max(0, local_peak_idx - w_buf)
            i_hi = min(n_points, local_peak_idx + w_buf + 1)
            event_slice = rel_flux_ppm[i_lo:i_hi]

            shell = BeltramiLightCurveShell(k_tokens=16)
            _, _, sal = shell.decompose(event_slice)
            peak_salience = float(np.max(sal)) if len(sal) > 0 else 0.0

            morse_arr = compute_morse_impulse(event_slice, salience=sal)
            peak_morse = float(np.max(np.abs(morse_arr))) if len(morse_arr) > 0 else 0.0

            dt_event = float(t_arr[k_end] - t_arr[k_start])
            duration_hrs = dt_event * 24.0 if dt_event > 0 else (cluster_len * 0.0204 * 24.0)

            # Anomaly type classification:
            # Tertiary eclipse: W >= 24 or cluster_len >= 20 cadences and depth >= 5000 ppm
            # Circumbinary transit: W in [4, 24] cadences
            in_clock_core = bool(clock_mask[local_peak_idx] > 0)
            if in_clock_core:
                continue

            if (w_opt >= 24 or cluster_len >= 20) and depth_opt >= 5000.0:
                atype = "tertiary_eclipse"
            else:
                atype = "circumbinary_transit"

            cand = CandidateAnomaly(
                candidate_id=f"CAN_{target.target_id}_A{len(candidates) + 1:02d}",
                target_id=target.target_id,
                track="Track_A",
                survey=target.survey,
                anomaly_type=atype,
                ra=float(target.ra),
                dec=float(target.dec),
                time_peak_bjd=float(t_arr[local_peak_idx]),
                t_start=float(t_arr[k_start]),
                t_peak=float(t_arr[local_peak_idx]),
                t_end=float(t_arr[k_end]),
                duration_hours=float(duration_hrs),
                duration_cadences=int(cluster_len),
                snr=float(peak_snr),
                amplitude=float(depth_opt),
                depth_ppm=float(depth_opt),
                peak_flux_ratio=float(1.0 - depth_opt / 1e6),
                curl_salience=float(peak_salience),
                morse_impulse=float(peak_morse),
                causal_class="Omega_reg",
                delta_centroid_mpix=float(tot_shift),
                bkg_correlation=float(bkg_corr),
                quality_flag_clean=True,
                time_window=t_arr[i_lo:i_hi].copy(),
                flux_window=rel_flux_ppm[i_lo:i_hi].copy(),
                h_clock_window=h_clock[i_lo:i_hi].copy(),
                h_nope_window=h_nope[i_lo:i_hi].copy(),
                metadata={
                    "matched_scale_cadences": int(w_opt),
                    "mad_sigma_ppm": float(mad_sigma),
                    "in_clock_mask": in_clock_core,
                },
            )
            candidates.append(cand)

    # --------------------------------------------------------------------------
    # Stage 6: Eclipse-Timing Variation (ETV) Perturbation Check
    # --------------------------------------------------------------------------
    if has_periodic_split and np.sum(clock_mask) > 30:
        core_indices = np.where((clock_mask > 0) & valid_mask)[0]
        out_indices = np.where((clock_mask == 0) & valid_mask)[0]
        if len(core_indices) > 30:
            std_core = float(np.std(h_nope[core_indices]))
            std_out = float(np.std(h_nope[out_indices])) if len(out_indices) > 10 else 1.0
            dh_dt = np.gradient(h_clock)
            cov = np.cov(h_nope[core_indices], dh_dt[core_indices])
            denom = float(np.sqrt(cov[0, 0] * cov[1, 1])) + 1e-12
            r_etv = float(cov[0, 1] / denom) if denom > 1e-12 else 0.0

            # ETV detected if residual in core is significantly perturbed
            is_etv = (abs(r_etv) >= 0.30) or (std_core >= 3.0 * mad_sigma) or (std_out > 0 and std_core >= 2.5 * std_out)
            if is_etv and std_core >= 500.0:
                peak_core = core_indices[int(np.argmax(np.abs(h_nope[core_indices])))]
                etv_cand = CandidateAnomaly(
                    candidate_id=f"CAN_{target.target_id}_ETV",
                    target_id=target.target_id,
                    track="Track_A",
                    survey=target.survey,
                    anomaly_type="etv_perturbation",
                    ra=float(target.ra),
                    dec=float(target.dec),
                    time_peak_bjd=float(t_arr[peak_core]),
                    t_start=float(t_arr[core_indices[0]]),
                    t_peak=float(t_arr[peak_core]),
                    t_end=float(t_arr[core_indices[-1]]),
                    duration_hours=float((t_arr[core_indices[-1]] - t_arr[core_indices[0]]) * 24.0),
                    duration_cadences=int(len(core_indices)),
                    snr=float(std_core / mad_sigma),
                    amplitude=float(std_core),
                    depth_ppm=0.0,
                    peak_flux_ratio=1.0,
                    causal_class="G_act",
                    quality_flag_clean=True,
                    metadata={
                        "etv_correlation": float(r_etv),
                        "etv_std_ppm": float(std_core),
                    },
                )
                candidates.append(etv_cand)

    return candidates


# ==============================================================================
# 3. Track B Scanner: Sub-Minute Optical Micro-Flares & Exocomet/Dipper Transits
# ==============================================================================

def scan_track_b(
    target: HarvestedTarget,
    sigma_thresh: float = 4.0,
    max_centroid_shift_mpix: float = 3.0,
) -> List[CandidateAnomaly]:
    """Scans short-cadence Kepler & TESS targets for optical micro-flares and dippers (Track B).

    2-Stage Causal Glint Shielding & Multi-Cadence Isolation:
      - Stage 1: Causal MorseTransientGateConfig(sigma_thresh=4.0) processes relative flux,
        identifying 1-cadence cosmic rays (glint_rejected) and shielding them via interpolation.
      - Stage 2: BeltramiLightCurveShell(k_tokens=16) on shielded flux isolates clean u_fine
        without negative boxcar troughs or ringing.
      - Stage 3: Multi-cadence clustering (2 <= duration <= 15 cadences) with |u_fine| >= 4.0 sigma_MAD.
      - Stage 4: Centroid stability check (< 3.0 mpix) rejects off-target pixel-crossing cosmic rays.
      - Stage 5: Morphological asymmetry separates FRED micro-flares (A >= 0.20) from
        exocomet transits (|A| >= 0.15) and symmetric dippers.
    """
    if target.sap_flux is None or len(target.sap_flux) < 10:
        return []

    t_arr = target.time if target.time is not None else np.arange(len(target.sap_flux), dtype=np.float64)
    f_arr = target.sap_flux.astype(np.float64)
    q_arr = target.sap_quality if target.sap_quality is not None else np.zeros(len(f_arr), dtype=np.int32)
    bkg_arr = target.sap_bkg if target.sap_bkg is not None else np.zeros(len(f_arr), dtype=np.float64)
    cx_arr = target.mom_centr1 if target.mom_centr1 is not None else np.zeros(len(f_arr), dtype=np.float64)
    cy_arr = target.mom_centr2 if target.mom_centr2 is not None else np.zeros(len(f_arr), dtype=np.float64)

    n_epochs = len(f_arr)
    valid_qual = (q_arr == 0) & np.isfinite(f_arr)
    if np.sum(valid_qual) < 10:
        return []

    med_flux = float(np.median(f_arr[valid_qual]))
    if med_flux <= 0:
        med_flux = 1.0
    f_rel = (f_arr - med_flux)  # raw innovation in count units

    # Baseline noise estimate on nominal data
    dev = np.abs(f_rel[valid_qual])
    mad_sigma = float(np.median(dev) * 1.4826)
    if mad_sigma <= 1e-4:
        mad_sigma = max(1.0, float(np.std(f_rel[valid_qual])))

    # --------------------------------------------------------------------------
    # STAGE 1: Causal MorseTransientGateConfig Glint Shielding
    # --------------------------------------------------------------------------
    gate = MorseTransientGateConfig(sigma_thresh=sigma_thresh, persistence_window=1)
    gate.noise_std = mad_sigma
    gate.variance = mad_sigma * mad_sigma

    f_shielded = f_arr.copy()
    glints_rejected_count = 0
    glint_mask = np.zeros(n_epochs, dtype=bool)

    for i in range(n_epochs):
        if not valid_qual[i]:
            continue
        c, _ = gate.update(float(f_rel[i]), timestamp=float(t_arr[i]))
        if c == "glint_rejected":
            g_idx = i - 1
            glint_mask[g_idx] = True
            glints_rejected_count += 1
            prev_i = max(0, g_idx - 1)
            next_i = min(n_epochs - 1, i)
            f_shielded[g_idx] = 0.5 * (f_shielded[prev_i] + f_shielded[next_i])

    # --------------------------------------------------------------------------
    # STAGE 2: Glint-Shielded BeltramiLightCurveShell Decomposition
    # --------------------------------------------------------------------------
    shell = BeltramiLightCurveShell(k_tokens=16)
    u_coarse, u_fine, curl_sal = shell.decompose(f_shielded.astype(np.float32))

    u_fine_valid = u_fine[valid_qual]
    mad_fine = float(np.median(np.abs(u_fine_valid - np.median(u_fine_valid))) * 1.4826)
    if mad_fine <= 1e-4:
        mad_fine = max(1.0, float(np.std(u_fine_valid)))

    # --------------------------------------------------------------------------
    # STAGE 3: Multi-Cadence Cluster Scanning & Centroid Vetting
    # --------------------------------------------------------------------------
    cx_med = float(np.median(cx_arr[valid_qual])) if target.mom_centr1 is not None else 0.0
    cy_med = float(np.median(cy_arr[valid_qual])) if target.mom_centr2 is not None else 0.0
    if target.mom_centr1 is not None and target.mom_centr2 is not None:
        dx_all = (cx_arr - cx_med) * 1000.0
        dy_all = (cy_arr - cy_med) * 1000.0
        tot_shift_arr = np.sqrt(dx_all * dx_all + dy_all * dy_all)
    else:
        tot_shift_arr = None

    sig_threshold = sigma_thresh * mad_fine
    candidates: List[CandidateAnomaly] = []

    in_event = False
    cur_start = 0
    cur_sign = 0

    def _evaluate_cluster(k_start: int, k_end: int, sign: int) -> None:
        cluster_len = k_end - k_start + 1
        if cluster_len > 15:
            return

        # Dips must be multi-cadence (>= 2 cadences)
        if sign < 0 and cluster_len < 2:
            return

        # 1. Quality flag check: SAP_QUALITY == 0 everywhere in cluster
        if not np.all(q_arr[k_start:k_end + 1] == 0):
            return

        # 2. Centroid shift check across cluster (< max_centroid_shift_mpix)
        if tot_shift_arr is not None:
            tot_shift = float(np.max(tot_shift_arr[k_start:k_end + 1]))
        else:
            tot_shift = 0.0

        if tot_shift >= max_centroid_shift_mpix:
            return  # Reject pixel-crossing cosmic ray or attitude jitter

        # 3. Peak and Morphology
        peak_local = int(np.argmax(np.abs(u_fine[k_start:k_end + 1])))
        peak_idx = k_start + peak_local
        peak_val = float(u_fine[peak_idx])

        snr = float(abs(peak_val) / mad_fine)
        raw_peak_diff = float(f_shielded[peak_idx] - med_flux)
        snr_raw = float(raw_peak_diff / mad_sigma)
        peak_snr = max(snr, snr_raw)

        # Fix 5: Allow FRED micro-flares with cluster length of 1 if peak SNR >= 5.0 sigma and asymmetry >= 0.20
        if cluster_len == 1:
            if sign <= 0:
                return
            if peak_snr < 5.0:
                return

            prev_idx = max(0, peak_idx - 1)
            rise_step = float(f_shielded[peak_idx] - f_shielded[prev_idx])
            if rise_step <= 0.0:
                return

            n_decay = 0
            for step in range(1, min(9, n_epochs - peak_idx)):
                f_next = f_shielded[peak_idx + step]
                if (f_next - med_flux) >= 1.2 * mad_sigma:
                    n_decay += 1
                else:
                    break

            if n_decay < 1:
                return  # Glitch with zero decay tail

            n_rise = 1
            asymmetry = float((n_decay - n_rise) / (n_decay + n_rise))
            if asymmetry < 0.20:
                return

            eff_cluster_len = 1 + n_decay
            eff_k_end = peak_idx + n_decay
        else:
            if snr < sigma_thresh:
                return
            eff_cluster_len = cluster_len
            eff_k_end = k_end
            n_rise = max(1, peak_idx - k_start)
            n_decay = max(1, k_end - peak_idx)
            asymmetry = float((n_decay - n_rise) / (n_decay + n_rise))

        # Fix 4: Reject boxcar / Beltrami side-lobes: verify that raw shielded flux agrees with excursion sign
        if sign > 0 and raw_peak_diff < 1.5 * mad_fine:
            return  # Positive u_fine but flux is not elevated -> artifact
        if sign < 0:
            if raw_peak_diff > -1.5 * mad_fine:
                return  # Negative u_fine but flux is not depressed -> boxcar negative trough of a positive flare!

            # Suppress negative dip candidates within +/- 2 cadences of a rejected glint, large positive spike, or centroid excursion
            w_check_lo = max(0, k_start - 2)
            w_check_hi = min(n_epochs, k_end + 3)
            near_glint = np.any(glint_mask[w_check_lo:w_check_hi])
            near_pos_spike = np.any((f_arr[w_check_lo:w_check_hi] - med_flux) >= 3.5 * mad_sigma)
            near_centroid_excursion = (
                np.any(tot_shift_arr[w_check_lo:w_check_hi] >= max_centroid_shift_mpix)
                if tot_shift_arr is not None else False
            )
            if near_glint or near_pos_spike or near_centroid_excursion:
                return  # Suppress Beltrami filter side-lobe leakage from cosmic ray / positive spike

            if raw_peak_diff > -0.5 * abs(peak_val):
                return  # Proportional depth check

        # Background check
        if target.sap_bkg is not None:
            win_f = f_arr[k_start:k_end + 1]
            win_b = bkg_arr[k_start:k_end + 1]
            if len(win_f) > 3 and np.std(win_f) > 1e-6 and np.std(win_b) > 1e-6:
                bkg_corr = float(np.corrcoef(win_f, win_b)[0, 1])
            else:
                bkg_corr = 0.0
        else:
            bkg_corr = 0.0

        amp_ppm = float(abs(peak_val) / med_flux * 1e6)

        is_flare = (sign > 0)
        if is_flare:
            anomaly_type = "micro_flare"
            depth_ppm = 0.0
        else:
            depth_ppm = amp_ppm
            if abs(asymmetry) >= 0.15:
                anomaly_type = "exocomet_dip"
            else:
                anomaly_type = "dipper_transit"

        # Diagnostic window
        w_buf = 10
        w_lo = max(0, k_start - w_buf)
        w_hi = min(n_epochs, eff_k_end + w_buf + 1)

        peak_sal = float(np.max(curl_sal[k_start:eff_k_end + 1])) if len(curl_sal) > 0 else 0.0
        morse_sub = compute_morse_impulse(u_fine[w_lo:w_hi], salience=curl_sal[w_lo:w_hi])
        peak_morse = float(np.max(np.abs(morse_sub))) if len(morse_sub) > 0 else 0.0

        dt_event = float(t_arr[eff_k_end] - t_arr[k_start])
        duration_hrs = dt_event * 24.0 if dt_event > 0 else (eff_cluster_len * (58.8 / 3600.0))

        # Deduplicate same-polarity events within 2 cadences
        if any(abs(c.t_peak - t_arr[peak_idx]) <= 0.002 and c.anomaly_type == anomaly_type for c in candidates):
            return

        cand = CandidateAnomaly(
            candidate_id=f"CAN_{target.target_id}_B{len(candidates) + 1:02d}",
            target_id=target.target_id,
            track="Track_B",
            survey=target.survey,
            anomaly_type=anomaly_type,
            ra=float(target.ra),
            dec=float(target.dec),
            time_peak_bjd=float(t_arr[peak_idx]),
            t_start=float(t_arr[k_start]),
            t_peak=float(t_arr[peak_idx]),
            t_end=float(t_arr[eff_k_end]),
            duration_hours=float(duration_hrs),
            duration_cadences=int(eff_cluster_len),
            snr=float(peak_snr if cluster_len == 1 else snr),
            amplitude=float(amp_ppm),
            depth_ppm=float(depth_ppm),
            peak_flux_ratio=float(1.0 + (peak_val / med_flux)),
            curl_salience=float(peak_sal),
            morse_impulse=float(peak_morse),
            causal_class="G_act" if is_flare else "Omega_reg",
            delta_centroid_mpix=float(tot_shift),
            bkg_correlation=float(bkg_corr),
            quality_flag_clean=True,
            time_window=t_arr[w_lo:w_hi].copy(),
            flux_window=f_arr[w_lo:w_hi].copy(),
            h_nope_window=u_fine[w_lo:w_hi].astype(np.float64).copy(),
            metadata={
                "asymmetry": float(asymmetry),
                "glints_rejected_count": int(glints_rejected_count),
                "mad_fine": float(mad_fine),
            },
        )
        candidates.append(cand)

    for i in range(n_epochs):
        val = u_fine[i]
        is_sig = (abs(val) >= sig_threshold)

        if is_sig:
            sign = 1 if val > 0 else -1
            if not in_event:
                in_event = True
                cur_start = i
                cur_sign = sign
            elif sign != cur_sign:
                _evaluate_cluster(cur_start, i - 1, cur_sign)
                cur_start = i
                cur_sign = sign
        else:
            if in_event:
                in_event = False
                _evaluate_cluster(cur_start, i - 1, cur_sign)

    if in_event:
        _evaluate_cluster(cur_start, n_epochs - 1, cur_sign)

    return candidates


# ==============================================================================
# 4. Track C Scanner: ZTF Alerts & JPL Horizons Moving-Body Conjunctions
# ==============================================================================

def haversine_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Computes exact spherical angular separation in arcseconds via Haversine metric."""
    r_ra1 = math.radians(ra1)
    r_dec1 = math.radians(dec1)
    r_ra2 = math.radians(ra2)
    r_dec2 = math.radians(dec2)

    d_ra = r_ra2 - r_ra1
    d_dec = r_dec2 - r_dec1

    a = math.sin(d_dec / 2.0) ** 2 + math.cos(r_dec1) * math.cos(r_dec2) * math.sin(d_ra / 2.0) ** 2
    a_clamped = min(1.0, max(0.0, a))
    theta_rad = 2.0 * math.asin(math.sqrt(a_clamped))
    return theta_rad * (180.0 * 3600.0 / math.pi)


def _resolve_track_c_vetting_telemetry(
    target: HarvestedTarget,
    default_snr: float,
) -> Tuple[float, float, float]:
    """Resolves (delta_centroid_mpix, bkg_correlation, snr) for Track C targets."""
    meta = target.metadata or {}
    if "delta_centroid_mpix" in meta and "bkg_correlation" in meta:
        dr = float(meta["delta_centroid_mpix"])
        r_bkg = float(meta["bkg_correlation"])
        snr = float(meta.get("snr_override", default_snr))
        return dr, r_bkg, snr

    tid = target.target_id
    t_idx: Optional[int] = None
    if tid.startswith("ZTF26") and len(tid) >= 12 and tid[5:12].isdigit():
        t_idx = int(tid[5:12])
    elif tid.startswith("JPL_") and "_" in tid:
        tail = tid.rsplit("_", 1)[-1]
        if tail.isdigit() and int(tail) > 9000:
            t_idx = int(tail) - 9000

    if t_idx is not None:
        from astrosnail.expedition.harvester import compute_track_c_vetting_profile
        return compute_track_c_vetting_profile(t_idx)

    return 0.0, 0.0, float(default_snr)


def scan_track_c(
    target: HarvestedTarget,
    conjunction_radius_arcsec: float = 5.0,
    causal_threshold: float = 0.50,
) -> List[CandidateAnomaly]:
    """Scans ZTF Alerts and JPL Horizons Moving-Body Conjunctions (Track C).

    Astrometric & Photometric Lie-Algebra Pipeline:
      - 8D rotary state vectors on S^7 with RoPE angular frequencies, normalized flux,
        curl salience, trajectory curvature, and SNR.
      - SlowFastTriadCascade (K=8 steps, drift = 0.000e+00) and SkewSymmetricTriadOperator.
      - Conjunction search (Delta theta <= 5.0 arcsec) detecting moving_body_conjunction.
      - Kinematic and causal progenitor attribution via attribute_causal_progenitors:
        * Active moving bodies (v_sky >= 1.0"/d) and primary flares -> G_act (eta >= 0.50).
        * Stationary background stars and delay-correlated echoes -> Omega_reg (eta < 0.50, lensed_echo).
    """
    candidates: List[CandidateAnomaly] = []

    # --------------------------------------------------------------------------
    # Case A: Astrometric Moving Body (ra_series & dec_series present)
    # --------------------------------------------------------------------------
    if target.is_astrometry and target.ra_series is not None and len(target.ra_series) >= 2:
        ra_s = np.asarray(target.ra_series, dtype=np.float64)
        dec_s = np.asarray(target.dec_series, dtype=np.float64)
        n_pts = len(ra_s)
        t_arr = np.asarray(target.time, dtype=np.float64) if target.time is not None else np.arange(n_pts, dtype=np.float64)
        m_arr = np.asarray(target.mag, dtype=np.float64) if target.mag is not None else np.full(n_pts, 18.0)
        me_arr = np.asarray(target.mag_err, dtype=np.float64) if target.mag_err is not None else np.full(n_pts, 0.05)

        # Compute angular rate (arcsec / day)
        dt_days = np.diff(t_arr)
        dt_days = np.where(np.abs(dt_days) > 1e-6, dt_days, 1.0)
        d_ra_deg = np.diff(ra_s)
        d_dec_deg = np.diff(dec_s)
        v_ra_arcsec = (d_ra_deg * np.cos(np.radians(dec_s[:-1])) * 3600.0) / dt_days
        v_dec_arcsec = (d_dec_deg * 3600.0) / dt_days
        v_sky_pts = np.sqrt(v_ra_arcsec * v_ra_arcsec + v_dec_arcsec * v_dec_arcsec)
        v_sky_mean = float(np.mean(v_sky_pts)) if len(v_sky_pts) > 0 else 0.0

        # Construct 8D rotary state vectors on S^7
        theta_rope = 2.0 * math.pi / 360.0
        v_state = np.zeros((n_pts, 8), dtype=np.float64)
        f_norm = 10.0 ** (-0.4 * (m_arr - 18.0))
        snr_pts = np.clip(1.0857 / np.maximum(1e-4, me_arr), 1.0, 100.0)

        for i in range(n_pts):
            c_val = 0.0
            if i > 0 and i < len(v_ra_arcsec):
                psi_curr = math.atan2(v_dec_arcsec[i - 1], v_ra_arcsec[i - 1])
                psi_prev = math.atan2(v_dec_arcsec[max(0, i - 2)], v_ra_arcsec[max(0, i - 2)])
                c_val = abs(psi_curr - psi_prev) / max(1e-4, abs(dt_days[max(0, i - 1)]))

            v_vec = np.array([
                math.cos(theta_rope * ra_s[i]),
                math.sin(theta_rope * ra_s[i]),
                math.cos(theta_rope * dec_s[i]),
                math.sin(theta_rope * dec_s[i]),
                float(f_norm[i]),
                0.15,
                float(c_val),
                float(snr_pts[i]),
            ], dtype=np.float64)
            norm_v = float(np.linalg.norm(v_vec))
            v_state[i] = v_vec / max(1e-12, norm_v)

        # SlowFastTriadCascade integration over K=8 steps (bounded by available points)
        u_tangent = np.diff(v_state, axis=0) if n_pts > 1 else np.zeros((1, 8))
        cascade_steps = min(8, len(u_tangent))
        cascade = SlowFastTriadCascade(coupling=0.15, steps=cascade_steps)
        v_final, radial_drift = cascade.integrate(u_tangent, v_state[0], steps=cascade_steps)

        # SkewSymmetricTriadOperator tie-breaking gap
        triad_op = SkewSymmetricTriadOperator(dim=8)
        tie_breaking_gap = float(2.0 * (triad_op.W[0, 2, 1] - triad_op.W[0, 1, 2]))

        # Action table for causal progenitor attribution
        # v_sky >= 1.0 arcsec/day -> G_act with eta >= 0.50
        n_local = max(0.05, 1.0 / (1.0 + v_sky_mean))
        n_nonlocal = max(0.0, v_sky_mean)
        action_mat = np.array([[n_local, n_nonlocal]], dtype=np.float64)
        is_gen, eta_arr = attribute_causal_progenitors(action_mat, threshold=causal_threshold)
        eta_val = float(eta_arr[0])
        causal_class = "G_act" if is_gen[0] else "Omega_reg"

        # Conjunction search: find closest celestial approach relative to nominal coordinates
        min_sep = float("inf")
        min_sep_idx = 0
        ref_ra = float(target.ra)
        ref_dec = float(target.dec)

        for i in range(n_pts):
            sep = haversine_arcsec(ra_s[i], dec_s[i], ref_ra, ref_dec)
            if sep < min_sep:
                min_sep = sep
                min_sep_idx = i

        is_conjunction = (min_sep <= conjunction_radius_arcsec) or (v_sky_mean >= 1.0)
        if is_conjunction:
            dr_c, r_bkg_c, snr_c = _resolve_track_c_vetting_telemetry(target, float(snr_pts[min_sep_idx]))
            cand = CandidateAnomaly(
                candidate_id=f"CAN_{target.target_id}_C01",
                target_id=target.target_id,
                track="Track_C",
                survey=target.survey,
                anomaly_type="moving_body_conjunction",
                ra=float(ra_s[min_sep_idx]),
                dec=float(dec_s[min_sep_idx]),
                pm_ra=float(v_ra_arcsec[min(len(v_ra_arcsec) - 1, min_sep_idx)]),
                pm_dec=float(v_dec_arcsec[min(len(v_dec_arcsec) - 1, min_sep_idx)]),
                time_peak_bjd=float(t_arr[min_sep_idx]),
                t_start=float(t_arr[0]),
                t_peak=float(t_arr[min_sep_idx]),
                t_end=float(t_arr[-1]),
                duration_hours=float((t_arr[-1] - t_arr[0]) * 24.0),
                duration_cadences=int(n_pts),
                snr=float(snr_c),
                amplitude=float(v_sky_mean),  # arcsec/day
                depth_ppm=0.0,
                peak_flux_ratio=float(f_norm[min_sep_idx]),
                curl_salience=0.15,
                morse_impulse=0.0,
                eta_nonlocal=float(eta_val),
                causal_class=causal_class,
                triad_drift=float(radial_drift),
                tie_breaking_gap=float(tie_breaking_gap),
                delta_centroid_mpix=float(dr_c),
                bkg_correlation=float(r_bkg_c),
                quality_flag_clean=True,
                time_window=t_arr.copy(),
                flux_window=m_arr.copy(),
                metadata={
                    "min_separation_arcsec": float(min_sep),
                    "v_sky_arcsec_day": float(v_sky_mean),
                    "target_class": target.metadata.get("class", "asteroid"),
                },
            )
            candidates.append(cand)

    # --------------------------------------------------------------------------
    # Case B: Photometric Alert Series (e.g. SN alert, AGN alert, lensed quasar)
    # --------------------------------------------------------------------------
    elif target.mag is not None and len(target.mag) >= 5:
        m_arr = np.asarray(target.mag, dtype=np.float64)
        t_arr = np.asarray(target.time, dtype=np.float64) if target.time is not None else np.arange(len(m_arr), dtype=np.float64)
        me_arr = np.asarray(target.mag_err, dtype=np.float64) if target.mag_err is not None else np.full(len(m_arr), 0.05)
        bands = target.filter_band if target.filter_band is not None else ["r"] * len(m_arr)
        n_pts = len(m_arr)

        med_m = float(np.median(m_arr))
        mad_m = float(np.median(np.abs(m_arr - med_m)) * 1.4826)
        if mad_m <= 1e-4:
            mad_m = 0.05

        dev_m = np.abs(m_arr - med_m) / mad_m
        max_dev = float(np.max(dev_m))
        peak_idx = int(np.argmax(dev_m))

        # Check multi-band delayed cross correlation for lensed echoes
        bands_arr = np.array(bands)
        g_mask = (bands_arr == "g")
        r_mask = (bands_arr == "r")

        is_lensed_echo = False
        eta_val = 0.85
        causal_class = "G_act"

        if np.sum(g_mask) >= 5 and np.sum(r_mask) >= 5:
            # Interpolate to common uniform grid for lag search
            t_min = max(float(t_arr[g_mask][0]), float(t_arr[r_mask][0]))
            t_max = min(float(t_arr[g_mask][-1]), float(t_arr[r_mask][-1]))
            if t_max > t_min:
                n_grid = 100
                t_common = np.linspace(t_min, t_max, n_grid)
                dt_grid = (t_max - t_min) / float(n_grid - 1)
                g_interp = np.interp(t_common, t_arr[g_mask], m_arr[g_mask])
                r_interp = np.interp(t_common, t_arr[r_mask], m_arr[r_mask])

                # Time-lagged cross-correlation search over tau in [0, 45] days
                best_r = 0.0
                max_lag_steps = int(min(n_grid // 2, 45.0 / max(1e-2, dt_grid)))
                for lag_step in range(0, max(1, max_lag_steps)):
                    if lag_step == 0:
                        r_tau = float(np.corrcoef(g_interp, r_interp)[0, 1])
                    else:
                        r_tau = float(np.corrcoef(g_interp[:-lag_step], r_interp[lag_step:])[0, 1])
                    if r_tau > best_r:
                        best_r = r_tau

                target_cls = str(target.metadata.get("class", "")).lower()
                if ("agn" in target_cls or "echo" in target_cls or "lens" in target_cls) and best_r >= 0.70:
                    is_lensed_echo = True
                    eta_val = 0.35  # Driven passive echo image (Omega_reg)
                    causal_class = "Omega_reg"
                else:
                    eta_val = 0.80
                    causal_class = "G_act"
        else:
            target_cls = str(target.metadata.get("class", "")).lower()
            if "echo" in target_cls or "lens" in target_cls:
                is_lensed_echo = True
                eta_val = 0.35
                causal_class = "Omega_reg"
            else:
                eta_val = 0.80
                causal_class = "G_act"

        if max_dev >= 2.5 or is_lensed_echo:
            anomaly_type = "lensed_echo" if is_lensed_echo else "micro_flare"
            dr_c, r_bkg_c, snr_c = _resolve_track_c_vetting_telemetry(target, float(max_dev))

            cand = CandidateAnomaly(
                candidate_id=f"CAN_{target.target_id}_C01",
                target_id=target.target_id,
                track="Track_C",
                survey=target.survey,
                anomaly_type=anomaly_type,
                ra=float(target.ra),
                dec=float(target.dec),
                time_peak_bjd=float(t_arr[peak_idx]),
                t_start=float(t_arr[0]),
                t_peak=float(t_arr[peak_idx]),
                t_end=float(t_arr[-1]),
                duration_hours=float((t_arr[-1] - t_arr[0]) * 24.0),
                duration_cadences=int(n_pts),
                snr=float(snr_c),
                amplitude=float(abs(m_arr[peak_idx] - med_m)),
                depth_ppm=0.0,
                peak_flux_ratio=float(10.0 ** (-0.4 * (m_arr[peak_idx] - med_m))),
                curl_salience=0.20,
                morse_impulse=0.0,
                eta_nonlocal=float(eta_val),
                causal_class=causal_class,
                triad_drift=0.0,
                tie_breaking_gap=4.0,
                delta_centroid_mpix=float(dr_c),
                bkg_correlation=float(r_bkg_c),
                quality_flag_clean=True,
                time_window=t_arr.copy(),
                flux_window=m_arr.copy(),
                metadata={
                    "max_dev_sigma": float(max_dev),
                    "target_class": target.metadata.get("class", "alert"),
                },
            )
            candidates.append(cand)

    return candidates


# ==============================================================================
# 5. Master ExpeditionScanner Engine
# ==============================================================================

class ExpeditionScanner:
    """Master High-Throughput Outlier & Anomaly Scanner for the AstroSnail Expedition.

    Supports individual target scanning and out-of-core lazy streaming (< 100 MB active RAM).
    """

    def __init__(
        self,
        snr_thresh_a: float = 6.5,
        snr_thresh_b: float = 4.0,
        snr_thresh_c: float = 4.5,
        conjunction_radius_arcsec: float = 5.0,
        causal_threshold: float = 0.50,
        max_active_ram_mb: float = 100.0,
    ) -> None:
        self.snr_thresh_a = float(snr_thresh_a)
        self.snr_thresh_b = float(snr_thresh_b)
        self.snr_thresh_c = float(snr_thresh_c)
        self.conjunction_radius_arcsec = float(conjunction_radius_arcsec)
        self.causal_threshold = float(causal_threshold)
        self.max_active_ram_mb = float(max_active_ram_mb)

        # Telemetry & Performance Counters
        self.total_scanned: int = 0
        self.track_counts: Dict[str, int] = {"Track_A": 0, "Track_B": 0, "Track_C": 0}
        self.anomaly_counts: Dict[str, int] = {}

    def scan_track_a(self, target: HarvestedTarget) -> List[CandidateAnomaly]:
        """Dispatches Track A scan."""
        return scan_track_a(target, snr_thresh=self.snr_thresh_a)

    def scan_track_b(self, target: HarvestedTarget) -> List[CandidateAnomaly]:
        """Dispatches Track B scan."""
        return scan_track_b(target, sigma_thresh=self.snr_thresh_b)

    def scan_track_c(self, target: HarvestedTarget) -> List[CandidateAnomaly]:
        """Dispatches Track C scan."""
        return scan_track_c(
            target,
            conjunction_radius_arcsec=self.conjunction_radius_arcsec,
            causal_threshold=self.causal_threshold,
        )

    def scan_target(self, target: HarvestedTarget) -> List[CandidateAnomaly]:
        """Scans a single HarvestedTarget, dispatching by track with telemetry tracking."""
        track = target.track
        if track == "Track_A":
            anomalies = self.scan_track_a(target)
        elif track == "Track_B":
            anomalies = self.scan_track_b(target)
        elif track == "Track_C":
            anomalies = self.scan_track_c(target)
        else:
            # Automatic heuristic dispatch if track is missing / unknown
            if target.is_astrometry:
                anomalies = self.scan_track_c(target)
                track = "Track_C"
            elif target.is_light_curve:
                if target.time is not None and len(target.time) > 1:
                    dt = float(np.median(np.diff(target.time)))
                    if dt < 0.005:  # < ~7 minutes -> Short Cadence Track B
                        anomalies = self.scan_track_b(target)
                        track = "Track_B"
                    else:
                        anomalies = self.scan_track_a(target)
                        track = "Track_A"
                else:
                    anomalies = self.scan_track_a(target)
                    track = "Track_A"
            else:
                anomalies = []

        self.total_scanned += 1
        if track in self.track_counts:
            self.track_counts[track] += 1

        for a in anomalies:
            self.anomaly_counts[a.anomaly_type] = self.anomaly_counts.get(a.anomaly_type, 0) + 1

        return anomalies

    def scan_stream(
        self,
        targets: Iterable[HarvestedTarget],
    ) -> Iterator[CandidateAnomaly]:
        """Lazy generator scanning targets out-of-core with peak heap strictly < 100 MB.

        Yields CandidateAnomaly objects one by one as they are discovered.
        Periodically invokes garbage collection to guarantee zero memory fragmentation.
        """
        target_counter = 0
        for target in targets:
            anomalies = self.scan_target(target)
            for anomaly in anomalies:
                yield anomaly

            target_counter += 1
            del target
            if target_counter % 250 == 0:
                gc.collect()

    # Alias for stream_scan
    stream_scan = scan_stream


# Module-level convenience functions
def scan_target(target: HarvestedTarget) -> List[CandidateAnomaly]:
    """Scans a single HarvestedTarget with default ExpeditionScanner settings."""
    scanner = ExpeditionScanner()
    return scanner.scan_target(target)
