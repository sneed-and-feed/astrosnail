"""
AstroSnail Ephemeris Cross-Matching & False-Positive Vetting Engine (`astrosnail.expedition.vetter`)
=====================================================================================================
Milestone M7.3 Production Implementation

Implements:
  1. `CatalogEphemerisEntry`: Compact representation of known NASA KOI, TOI, and Kepler/TESS
     Eclipsing Binary ephemerides.
  2. `InstrumentalVettingReceipt`: Sub-pixel centroid shift, background flux Pearson correlation,
     and spacecraft telemetry quality bitmask vetting.
  3. `PhysicalVettingReceipt`: Odd-even depth parity, Keplerian stellar density duration bounds,
     and transit profile morphology (U-shape vs V-shape).
  4. `VettedCandidate`: Unified container (slots=True) with 32+ attributes preserving all operator
     receipts, physical attribution parameters, discovery tiers, and export methods (.lc, .fits).
  5. `EphemerisMatcher` / `EphemerisCrossMatcher`: 3-Tier catalog cross-matching engine with dynamic
     tolerance matching window, integer cycle matching, and multi-planet / novel taxonomy.
  6. `InstrumentalFalsePositiveFilter` / `InstrumentalTriageEngine`: Centroid pull (>= 3.0 mpix),
     stray light background coupling (|r| >= 0.80), and momentum dump / Argabrightening triage.
  7. `AstrophysicalConsistencyVetter` / `PhysicalConsistencyEngine`: Odd-even depth parity (Delta d >= 3.0 sigma),
     duration ratio (R_dur > 3.0), and profile morphology triage.
  8. `ExpeditionVetter`: Master vetting coordinator with out-of-core streaming (< 25 MB active heap).
"""

from __future__ import annotations

import gc
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np

from astrosnail.expedition.harvester import HarvestedTarget, write_fits_bintable
from astrosnail.expedition.scanner import CandidateAnomaly

# ==============================================================================
# Spacecraft Quality Mask Definitions (NASA Kepler & TESS)
# ==============================================================================

FATAL_QUALITY_MASK: int = (
    (1 << 0)   # Bit 1: Attitude tweak
    | (1 << 1) # Bit 2: Safe mode
    | (1 << 2) # Bit 3: Spacecraft recovery
    | (1 << 3) # Bit 4: Earth point
    | (1 << 4) # Bit 5: Pointing excursion (TESS) / Zero crossing (Kepler)
    | (1 << 5) # Bit 6: Desaturation event (Thruster firing / Momentum dump)
    | (1 << 6) # Bit 7: Raw data anomalous / Missing data
    | (1 << 8) # Bit 9: Manual exclude
    | (1 << 9) # Bit 10: Sudden sensitivity dropout (Kepler) / Stray light (TESS)
    | (1 << 10)# Bit 11: Impulsive outlier / Instrument anomaly
    | (1 << 11)# Bit 12: Argabrightening event (Kepler) / Planet search exclude (TESS)
    | (1 << 12)# Bit 13: Cosmic ray in optimal aperture / Bad calibration
    | (1 << 13)# Bit 14: Detector anomaly / Cadence not used
    | (1 << 14)# Bit 15: Coarse pointing (no fine point)
    | (1 << 15)# Bit 16: No data (gap)
    | (1 << 19)# Bit 20: Severe pointing excursion (> 0.5 px)
)


# ==============================================================================
# 1. Dataclasses: Catalog, Receipts, and VettedCandidate
# ==============================================================================

@dataclass(slots=True)
class CatalogEphemerisEntry:
    """Compact catalog ephemeris entry for known planets and eclipsing binaries."""

    target_id: str
    survey: str = "Kepler"
    catalog_name: str = "KOI"   # "KOI" | "TOI" | "KEBC" | "TEBC" | "CUSTOM"
    object_name: str = ""
    period_days: float = 0.0
    t0_bjd: float = 0.0
    duration_hours: float = 0.0
    depth_ppm: float = 0.0
    disposition: str = "CONFIRMED"  # "CONFIRMED" | "CANDIDATE" | "FALSE_POSITIVE"
    ra: float = 0.0
    dec: float = 0.0
    period_err: float = 0.0
    t0_err: float = 0.0
    impact_parameter: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Aliases for compatibility
    @property
    def catalog_type(self) -> str:
        return self.catalog_name

    @property
    def period(self) -> float:
        return self.period_days

    @property
    def epoch_bjd(self) -> float:
        return self.t0_bjd

    @property
    def epoch_err(self) -> float:
        return self.t0_err

    @property
    def impact_param(self) -> float:
        return self.impact_parameter


@dataclass(slots=True)
class InstrumentalVettingReceipt:
    """Quantitative receipt for instrumental false-positive triage."""

    delta_centroid_mpix: float
    centroid_snr: float
    delta_x_mpix: float
    delta_y_mpix: float
    centroid_sigma_mpix: float
    centroid_passed: bool
    bkg_correlation: float
    bkg_relative_amp: float
    bkg_passed: bool
    quality_flag_clean: bool
    fatal_quality_bits: int
    min_dump_distance_hours: float
    thruster_veto_passed: bool
    argabrightening_detected: bool
    instrumental_score: float
    is_instrumental_fp: bool
    veto_reasons: List[str]


@dataclass(slots=True)
class PhysicalVettingReceipt:
    """Quantitative receipt for astrophysical consistency triage."""

    odd_even_sigma: float
    odd_even_passed: bool
    depth_odd_ppm: float
    depth_even_ppm: float
    duration_ratio: float
    duration_passed: bool
    morphology_ratio: float
    morphology_shape: str
    morphology_score: float
    physical_score: float
    is_physical_fp: bool
    veto_reasons: List[str]


@dataclass(slots=True)
class VettedCandidate:
    """Unified container for fully vetted astrophysical candidate anomalies across Tracks A, B, and C."""

    # 1. Target Identity & Survey Provenance
    candidate_id: str
    target_id: str
    track: str
    survey: str
    anomaly_type: str

    # 2. Celestial Coordinates & Kinematics
    ra: float
    dec: float
    pm_ra: float = 0.0
    pm_dec: float = 0.0

    # 3. Detection Epochs & Orbital Timing
    time_peak_bjd: float = 0.0
    t_start: float = 0.0
    t_peak: float = 0.0
    t_end: float = 0.0
    duration_hours: float = 0.0
    duration_cadences: int = 0
    n_epochs: int = 0
    period_days: Optional[float] = None

    # 4. Significance & Photometric Properties
    snr: float = 0.0
    amplitude: float = 0.0
    depth_ppm: float = 0.0
    peak_flux_ratio: float = 1.0

    # 5. Core AstroSnail Operator Receipts & Lie Metrics
    curl_salience: float = 0.0
    morse_impulse: float = 0.0
    eta_nonlocal: float = 0.0
    causal_class: str = "Omega_reg"
    triad_drift: float = 0.0
    tie_breaking_gap: float = 0.0

    # 6. Physical Transit / Eclipse Consistency Receipts
    odd_even_sigma: float = 0.0
    odd_even_passed: bool = True
    depth_odd_ppm: float = 0.0
    depth_even_ppm: float = 0.0
    duration_ratio: float = 1.0
    duration_passed: bool = True
    morphology_ratio: float = 0.10
    morphology_shape: str = "U-shape"

    # 7. Instrumental & Centroid Stability Receipts
    delta_centroid_mpix: float = 0.0
    centroid_snr: float = 0.0
    centroid_passed: bool = True
    bkg_correlation: float = 0.0
    bkg_passed: bool = True
    quality_flag_clean: bool = True

    # 8. Ephemeris Cross-Match & Discovery Taxonomy
    catalog_match: Optional[str] = None
    catalog_id: Optional[str] = None
    disposition: str = "NOVEL_CANDIDATE"
    taxonomy_class: str = "genuine_novel_discovery"

    # 9. Discovery Tier Ranking & Vetting Score
    vetting_score: float = 0.0
    discovery_tier: str = "Tier_1_Gold"
    rejection_reason: Optional[str] = None

    # 10. Physical Attribution Parameters
    rp_rstar: float = 0.0
    rp_earth: float = 0.0
    semi_major_axis_au: float = 0.0
    a_rstar: float = 0.0
    impact_parameter_b: float = 0.0
    stellar_teff_k: float = 5778.0
    stellar_radius_solar: float = 1.0
    stellar_mass_solar: float = 1.0
    stellar_density_g_cm3: float = 1.408

    # 11. Diagnostic Sub-Arrays (local event window)
    time_window: Optional[np.ndarray] = None
    flux_window: Optional[np.ndarray] = None
    h_clock_window: Optional[np.ndarray] = None
    h_nope_window: Optional[np.ndarray] = None
    model_window: Optional[np.ndarray] = None

    # 12. Receipts & Context Metadata
    instrumental_receipt: Optional[InstrumentalVettingReceipt] = None
    physical_receipt: Optional[PhysicalVettingReceipt] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Derive physical parameters if not already populated."""
        # Radius ratio Rp / R*
        if self.depth_ppm > 0 and self.rp_rstar == 0.0:
            self.rp_rstar = float(math.sqrt(max(0.0, self.depth_ppm * 1e-6)))

        # Planet radius in Earth radii (R_sun / R_earth ~ 109.076)
        if self.rp_rstar > 0 and self.rp_earth == 0.0:
            self.rp_earth = float((self.rp_rstar * self.stellar_radius_solar) / 0.009167)

        # Semi-major axis and a/R*
        if self.period_days is not None and self.period_days > 0:
            if self.semi_major_axis_au == 0.0:
                p_yr = self.period_days / 365.25
                self.semi_major_axis_au = float((self.stellar_mass_solar * p_yr * p_yr) ** (1.0 / 3.0))
            if self.a_rstar == 0.0:
                density_ratio = self.stellar_density_g_cm3 / 1.408
                self.a_rstar = float(4.269 * (density_ratio ** (1.0 / 3.0)) * (self.period_days ** (2.0 / 3.0)))

        # Impact parameter estimation if flat-bottomed
        if self.impact_parameter_b == 0.0:
            if self.morphology_shape == "U-shape":
                self.impact_parameter_b = min(0.50, max(0.0, self.morphology_ratio * 2.0))
            elif self.morphology_shape == "V-shape":
                self.impact_parameter_b = min(0.95, max(0.85, 0.85 + (self.morphology_ratio - 0.38) * 0.5))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes vetted candidate to a JSON-serializable dictionary."""
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
            "n_epochs": int(self.n_epochs),
            "period_days": float(self.period_days) if self.period_days is not None else None,
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
            "odd_even_sigma": float(self.odd_even_sigma),
            "odd_even_passed": bool(self.odd_even_passed),
            "depth_odd_ppm": float(self.depth_odd_ppm),
            "depth_even_ppm": float(self.depth_even_ppm),
            "duration_ratio": float(self.duration_ratio),
            "duration_passed": bool(self.duration_passed),
            "morphology_ratio": float(self.morphology_ratio),
            "morphology_shape": self.morphology_shape,
            "delta_centroid_mpix": float(self.delta_centroid_mpix),
            "centroid_snr": float(self.centroid_snr),
            "centroid_passed": bool(self.centroid_passed),
            "bkg_correlation": float(self.bkg_correlation),
            "bkg_passed": bool(self.bkg_passed),
            "quality_flag_clean": bool(self.quality_flag_clean),
            "catalog_match": self.catalog_match,
            "catalog_id": self.catalog_id,
            "disposition": self.disposition,
            "taxonomy_class": self.taxonomy_class,
            "vetting_score": float(self.vetting_score),
            "discovery_tier": self.discovery_tier,
            "rejection_reason": self.rejection_reason,
            "rp_rstar": float(self.rp_rstar),
            "rp_earth": float(self.rp_earth),
            "semi_major_axis_au": float(self.semi_major_axis_au),
            "a_rstar": float(self.a_rstar),
            "impact_parameter_b": float(self.impact_parameter_b),
            "stellar_teff_k": float(self.stellar_teff_k),
            "stellar_radius_solar": float(self.stellar_radius_solar),
            "stellar_mass_solar": float(self.stellar_mass_solar),
            "stellar_density_g_cm3": float(self.stellar_density_g_cm3),
            "metadata": dict(self.metadata),
        }

    def to_dossier_row(self) -> Dict[str, str]:
        """Formats candidate attributes into human-readable strings for dossier tables."""
        p_str = f"{self.period_days:.4f} d" if self.period_days is not None else "N/A"
        oe_str = f"{self.odd_even_sigma:.2f}σ" if self.period_days is not None else "N/A"
        return {
            "Candidate ID": self.candidate_id,
            "Target ID": self.target_id,
            "Track": self.track,
            "Survey": self.survey,
            "Anomaly Type": self.anomaly_type.replace("_", " ").title(),
            "Peak BJD": f"{self.time_peak_bjd:.5f}",
            "Period": p_str,
            "Depth": f"{self.depth_ppm:,.0f} ppm",
            "Duration": f"{self.duration_hours:.2f} h",
            "SNR": f"{self.snr:.1f}σ",
            "Centroid Shift": f"{self.delta_centroid_mpix:.2f} mpix",
            "Odd-Even": oe_str,
            "Morphology": f"{self.morphology_shape} (τ/T={self.morphology_ratio:.2f})",
            "Vetting Score": f"{self.vetting_score:.3f}",
            "Discovery Tier": self.discovery_tier.replace("_", " "),
            "Disposition": self.disposition.replace("_", " "),
        }

    def export_studio_lc(self, filepath: Union[str, Path]) -> Path:
        """Exports diagnostic light curve window to exact 7-column .lc format for portable studio.

        Columns: time flux flux_err bkg centr1 centr2 flag
        """
        dest_path = Path(filepath)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        t = self.time_window if self.time_window is not None else np.array([self.time_peak_bjd], dtype=np.float64)
        f = self.flux_window if self.flux_window is not None else np.array([1.0], dtype=np.float64)
        n = len(t)
        err = np.full(n, 0.00085, dtype=np.float64)
        bkg = self.metadata.get("bkg_window") if "bkg_window" in self.metadata and len(self.metadata["bkg_window"]) == n else np.full(n, 120.0, dtype=np.float64)
        centr1 = self.metadata.get("centr1_window") if "centr1_window" in self.metadata and len(self.metadata["centr1_window"]) == n else np.full(n, self.delta_centroid_mpix * 0.001, dtype=np.float64)
        centr2 = self.metadata.get("centr2_window") if "centr2_window" in self.metadata and len(self.metadata["centr2_window"]) == n else np.zeros(n, dtype=np.float64)
        flag = self.metadata.get("flag_window") if "flag_window" in self.metadata and len(self.metadata["flag_window"]) == n else np.zeros(n, dtype=np.int32)

        lines = [
            f"# AstroSnail Discovery Light Curve: {self.candidate_id}",
            f"# Target: {self.target_id} | Track: {self.track} | Survey: {self.survey}",
            f"# Discovery Tier: {self.discovery_tier} | Vetting Score: {self.vetting_score:.3f}",
            f"# Disposition: {self.disposition} | Anomaly Type: {self.anomaly_type}",
            f"# Depth: {self.depth_ppm:.1f} ppm | Duration: {self.duration_hours:.2f} h | SNR: {self.snr:.1f} sigma",
            f"# Centroid Shift: {self.delta_centroid_mpix:.2f} mpix | Odd-Even: {self.odd_even_sigma:.2f} sigma",
            f"# Physical Attribution: Rp/R* = {self.rp_rstar:.6f} | Rp = {self.rp_earth:.2f} R_earth | a = {self.semi_major_axis_au:.4f} AU | b = {self.impact_parameter_b:.3f}",
            f"# Columns: time flux flux_err bkg centr1 centr2 flag",
        ]
        for i in range(n):
            lines.append(f"{t[i]:.6f} {f[i]:.6f} {err[i]:.6f} {bkg[i]:.3f} {centr1[i]:.4f} {centr2[i]:.4f} {int(flag[i])}")

        dest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return dest_path

    def export_cat(self, filepath: Union[str, Path]) -> Path:
        """Exports astrometric point-source catalog to exact 6-column .cat format.

        Columns: ra dec mag pm_ra pm_dec tracklet_id
        """
        dest_path = Path(filepath)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# AstroSnail Astrometric Discovery Catalog",
            f"# Candidate ID: {self.candidate_id}",
            f"# Target ID: {self.target_id} | Track: {self.track} | Survey: {self.survey}",
            f"# Anomaly Type: {self.anomaly_type} | Discovery Tier: {self.discovery_tier} | Vetting Score: {self.vetting_score:.3f}",
            f"# Centroid Stability: {self.delta_centroid_mpix:.3f} mpix | so(8) Drift: {self.triad_drift:.3e}",
            f"# Columns: ra dec mag pm_ra pm_dec tracklet_id",
        ]
        tracklet_id = int(self.metadata.get("tracklet_id", 1001))
        mag = float(self.metadata.get("mag", 18.5))
        ra_pts = self.metadata.get("ra_series")
        dec_pts = self.metadata.get("dec_series")
        if ra_pts is not None and dec_pts is not None and len(ra_pts) == len(dec_pts) and len(ra_pts) > 1:
            for i in range(len(ra_pts)):
                lines.append(f"{float(ra_pts[i]):.6f} {float(dec_pts[i]):.6f} {mag:.4f} {self.pm_ra:.3f} {self.pm_dec:.3f} {tracklet_id}")
        else:
            lines.append(f"{self.ra:.6f} {self.dec:.6f} {mag:.4f} {self.pm_ra:.3f} {self.pm_dec:.3f} {tracklet_id}")

        dest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return dest_path

    def export_fits(self, filepath: Union[str, Path]) -> Path:
        """Exports diagnostic light curve window to standard FITS table with comprehensive header cards."""
        dest_path = Path(filepath)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        t = self.time_window if self.time_window is not None else np.array([self.time_peak_bjd], dtype=np.float64)
        f = self.flux_window if self.flux_window is not None else np.array([1.0], dtype=np.float64)
        err = np.full_like(f, 0.001)

        cols: Dict[str, np.ndarray] = {
            "TIME": t.astype(">f8"),
            "FLUX": f.astype(">f4"),
            "FLUX_ERR": err.astype(">f4"),
        }
        if self.h_clock_window is not None and len(self.h_clock_window) == len(t):
            cols["H_CLOCK"] = self.h_clock_window.astype(">f4")
        if self.h_nope_window is not None and len(self.h_nope_window) == len(t):
            cols["H_NOPE"] = self.h_nope_window.astype(">f4")

        p_cards = {
            "CAND_ID": self.candidate_id,
            "TARG_ID": self.target_id,
            "TRACK": self.track,
            "SURVEY": self.survey,
            "ANOM_TYP": self.anomaly_type,
            "VETSCORE": float(self.vetting_score),
            "TIER": self.discovery_tier,
            "DISPOSIT": self.disposition,
            "REJECT": self.rejection_reason or "NONE",
            "ODDEVEN": float(self.odd_even_sigma),
            "CENTRSH": float(self.delta_centroid_mpix),
            "BKGCORR": float(self.bkg_correlation),
            "RP_RSTAR": float(self.rp_rstar),
            "RP_EARTH": float(self.rp_earth),
        }
        write_fits_bintable(dest_path, cols, primary_cards=p_cards)
        return dest_path


# ==============================================================================
# 2. EphemerisMatcher: 3-Tier Catalog Ingestion & Phase Concordance
# ==============================================================================

class EphemerisMatcher:
    """High-throughput ephemeris cross-matching engine against KOI, TOI, and EB catalogs."""

    def __init__(
        self,
        alpha_dur: float = 0.50,
        beta_period: float = 0.02,
        dt_floor_days: float = 0.0408,
        catalog_entries: Optional[List[CatalogEphemerisEntry]] = None,
    ) -> None:
        self.alpha_dur = alpha_dur
        self.beta_period = beta_period
        self.dt_floor_days = dt_floor_days
        self._id_index: Dict[str, List[CatalogEphemerisEntry]] = {}
        self._spatial_grid: Dict[Tuple[int, int], List[CatalogEphemerisEntry]] = {}
        self._known_multi_hosts: set[str] = set()

        # Load built-in canonical benchmark systems
        self._load_canonical_benchmarks()

        # Add user-provided catalog entries
        if catalog_entries:
            for entry in catalog_entries:
                self.add_catalog_entry(entry)

    def _normalize_id(self, target_id: str) -> str:
        """Normalizes target identifiers (e.g., 'KIC_010666592' -> 'KIC_10666592')."""
        s = str(target_id).strip().upper().replace(" ", "_")
        for pfx in ("KIC_", "TIC_", "KIC", "TIC"):
            if s.startswith(pfx):
                num_part = s[len(pfx):].lstrip("0")
                canonical_pfx = "KIC_" if "KIC" in pfx else "TIC_"
                return f"{canonical_pfx}{num_part or '0'}"
        return s

    def _load_canonical_benchmarks(self) -> None:
        """Loads canonical Kepler and TESS benchmark reference systems."""
        benchmarks = [
            # HAT-P-7 b / Kepler-2 b (KOI-2.01)
            CatalogEphemerisEntry(
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
            ),
            # Kepler-8 b (KOI-10.01)
            CatalogEphemerisEntry(
                target_id="KIC_6922244",
                survey="Kepler",
                catalog_name="KOI",
                object_name="Kepler-8 b",
                period_days=3.5225,
                t0_bjd=122.045,
                duration_hours=3.2,
                depth_ppm=9400.0,
                disposition="CONFIRMED",
                ra=285.679,
                dec=42.452,
                period_err=2e-6,
                t0_err=1e-4,
            ),
            # TrES-2 b / Kepler-1 b (KOI-1.01)
            CatalogEphemerisEntry(
                target_id="KIC_11446443",
                survey="Kepler",
                catalog_name="KOI",
                object_name="TrES-2 b",
                period_days=2.47063,
                t0_bjd=121.250,
                duration_hours=1.8,
                depth_ppm=16000.0,
                disposition="CONFIRMED",
                ra=286.808,
                dec=49.316,
                period_err=1e-6,
                t0_err=1e-4,
            ),
            # Kepler-11 multi-planet system (KIC 6541920)
            CatalogEphemerisEntry(
                target_id="KIC_6541920",
                survey="Kepler",
                catalog_name="KOI",
                object_name="Kepler-11 b",
                period_days=10.30375,
                t0_bjd=125.500,
                duration_hours=4.0,
                depth_ppm=1200.0,
                disposition="CONFIRMED",
                ra=296.068,
                dec=41.908,
            ),
            CatalogEphemerisEntry(
                target_id="KIC_6541920",
                survey="Kepler",
                catalog_name="KOI",
                object_name="Kepler-11 c",
                period_days=13.02502,
                t0_bjd=127.100,
                duration_hours=4.5,
                depth_ppm=2100.0,
                disposition="CONFIRMED",
                ra=296.068,
                dec=41.908,
            ),
            # Kepler-16 circumbinary system (KIC 12644769)
            CatalogEphemerisEntry(
                target_id="KIC_12644769",
                survey="Kepler",
                catalog_name="KEBC",
                object_name="Kepler-16 EB",
                period_days=41.0792,
                t0_bjd=130.000,
                duration_hours=15.0,
                depth_ppm=130000.0,
                disposition="CONFIRMED",
                ra=289.075,
                dec=51.756,
            ),
            # TOI-700 multi-planet system (TIC 150428135)
            CatalogEphemerisEntry(
                target_id="TIC_150428135",
                survey="TESS",
                catalog_name="TOI",
                object_name="TOI-700 d",
                period_days=37.426,
                t0_bjd=1350.000,
                duration_hours=3.5,
                depth_ppm=850.0,
                disposition="CONFIRMED",
                ra=97.239,
                dec=-65.579,
            ),
            # Canonical benchmark Eclipsing Binary
            CatalogEphemerisEntry(
                target_id="KIC_5871143",
                survey="Kepler",
                catalog_name="KEBC",
                object_name="KIC 5871143 EB",
                period_days=4.2500,
                t0_bjd=120.000,
                duration_hours=5.0,
                depth_ppm=50000.0,
                disposition="CONFIRMED",
                ra=290.0,
                dec=41.0,
            ),
        ]
        for b in benchmarks:
            self.add_catalog_entry(b)

    def add_catalog_entry(self, entry: CatalogEphemerisEntry) -> None:
        """Registers a catalog ephemeris entry into ID and spatial indices."""
        norm_id = self._normalize_id(entry.target_id)
        entries = self._id_index.setdefault(norm_id, [])
        entries.append(entry)

        # Track multi-planet host status if multiple planet entries exist
        planet_entries = [e for e in entries if "eb" not in e.catalog_name.lower()]
        if len(planet_entries) > 1:
            self._known_multi_hosts.add(norm_id)

        # Spatial indexing
        cell = (int(entry.ra / 0.1), int(entry.dec / 0.1))
        self._spatial_grid.setdefault(cell, []).append(entry)

    def mark_multi_planet_host(self, target_id: str) -> None:
        """Explicitly tags a target as a known multi-planet host."""
        self._known_multi_hosts.add(self._normalize_id(target_id))

    def _get_target_ephemerides(
        self,
        candidate: CandidateAnomaly,
        target: Optional[HarvestedTarget] = None,
    ) -> List[CatalogEphemerisEntry]:
        """Gathers all candidate ephemerides from indices and target metadata."""
        ephemerides: List[CatalogEphemerisEntry] = []
        norm_id = self._normalize_id(candidate.target_id)

        # 1. Check ID index
        if norm_id in self._id_index:
            ephemerides.extend(self._id_index[norm_id])

        # 2. Check Target metadata
        t_meta = target.metadata if target is not None else {}
        c_meta = candidate.metadata

        # Extract multi-planet metadata
        if t_meta.get("category") == "multi_planet" or c_meta.get("category") == "multi_planet":
            self._known_multi_hosts.add(norm_id)
            periods = t_meta.get("periods") or c_meta.get("periods") or []
            depths = t_meta.get("depths_ppm") or c_meta.get("depths_ppm") or []
            t0s = t_meta.get("t0s") or c_meta.get("t0s") or [120.0] * len(periods)
            for idx, p in enumerate(periods):
                d = depths[idx] if idx < len(depths) else 1000.0
                t0 = t0s[idx] if idx < len(t0s) else 120.0
                ephemerides.append(
                    CatalogEphemerisEntry(
                        target_id=norm_id,
                        survey=candidate.survey,
                        catalog_name="KOI" if candidate.survey == "Kepler" else "TOI",
                        object_name=f"{candidate.target_id}_planet_{idx + 1}",
                        period_days=float(p),
                        t0_bjd=float(t0),
                        depth_ppm=float(d),
                    )
                )

        # Extract EB metadata
        if "eclipsing_binary" in str(t_meta.get("category")) or "eb" in str(t_meta.get("category")):
            eb_p = t_meta.get("period") or c_meta.get("catalog_period")
            if eb_p is not None:
                p_depth = float(t_meta.get("primary_depth_ppm") or t_meta.get("depth_ppm") or 50000.0)
                ephemerides.append(
                    CatalogEphemerisEntry(
                        target_id=norm_id,
                        survey=candidate.survey,
                        catalog_name="KEBC" if candidate.survey == "Kepler" else "TEBC",
                        object_name=f"{candidate.target_id}_EB",
                        period_days=float(eb_p),
                        t0_bjd=float(t_meta.get("t0", 120.0)),
                        depth_ppm=p_depth,
                    )
                )

        # 3. Spatial Cone Search (if ID not found and coordinates valid)
        if not ephemerides and (candidate.ra != 0.0 or candidate.dec != 0.0):
            c_ra, c_dec = candidate.ra, candidate.dec
            c_x, c_y = int(c_ra / 0.1), int(c_dec / 0.1)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cell = (c_x + dx, c_y + dy)
                    for entry in self._spatial_grid.get(cell, []):
                        # Approximate angular separation in arcseconds
                        dra = (entry.ra - c_ra) * math.cos(math.radians(c_dec)) * 3600.0
                        ddec = (entry.dec - c_dec) * 3600.0
                        dist_arcsec = math.sqrt(dra * dra + ddec * ddec)
                        if dist_arcsec <= 5.0:
                            ephemerides.append(entry)

        return ephemerides

    def cross_match(
        self,
        candidate: CandidateAnomaly,
        target: Optional[HarvestedTarget] = None,
    ) -> Dict[str, Any]:
        """Cross-matches candidate peak against known ephemerides."""
        ephemerides = self._get_target_ephemerides(candidate, target)
        norm_id = self._normalize_id(candidate.target_id)
        is_known_multi = norm_id in self._known_multi_hosts

        t_peak = candidate.time_peak_bjd if candidate.time_peak_bjd != 0.0 else candidate.t_peak
        d_cand = candidate.depth_ppm

        for eph in ephemerides:
            P = eph.period_days
            t0 = eph.t0_bjd
            if P <= 0.0:
                continue

            # Integer cycle matching
            N = round((t_peak - t0) / P)
            delta_t = t_peak - (t0 + N * P)

            # Dynamic tolerance matching window
            dur_hrs = eph.duration_hours if eph.duration_hours > 0 else candidate.duration_hours
            dur_days = dur_hrs / 24.0 if dur_hrs > 0 else 0.10
            sigma_eph = math.sqrt(eph.t0_err**2 + (N * eph.period_err)**2)
            dt_window = max(self.alpha_dur * dur_days, self.beta_period * P, self.dt_floor_days) + 3.0 * sigma_eph
            dt_window = min(dt_window, 0.20 * P, 1.0)

            # Primary transit concordance check
            if abs(delta_t) <= dt_window:
                depth_ratio = d_cand / max(eph.depth_ppm, 100.0)
                # Primary match
                is_eb = "eb" in eph.catalog_name.lower() or "binary" in eph.catalog_name.lower()
                cls_name = "catalog_confirmed_eb" if is_eb else "catalog_confirmed_planet"
                return {
                    "is_match": True,
                    "catalog_match": eph.catalog_name,
                    "catalog_id": eph.object_name or eph.target_id,
                    "matched_period": P,
                    "matched_epoch": t0,
                    "cycle_index": N,
                    "phase_offset_days": delta_t,
                    "phase_normalized": delta_t / P,
                    "delta_t_window": dt_window,
                    "is_secondary": False,
                    "depth_ratio": depth_ratio,
                    "disposition": "CONFIRMED_KNOWN",
                    "taxonomy_class": cls_name,
                }

            # Secondary eclipse concordance check near phase 0.5
            N_sec = round((t_peak - (t0 + 0.5 * P)) / P)
            delta_t_sec = t_peak - (t0 + (N_sec + 0.5) * P)
            if abs(delta_t_sec) <= dt_window:
                depth_ratio = d_cand / max(eph.depth_ppm, 100.0)
                return {
                    "is_match": True,
                    "catalog_match": eph.catalog_name,
                    "catalog_id": eph.object_name or eph.target_id,
                    "matched_period": P,
                    "matched_epoch": t0,
                    "cycle_index": N_sec,
                    "phase_offset_days": delta_t_sec,
                    "phase_normalized": delta_t_sec / P,
                    "delta_t_window": dt_window,
                    "is_secondary": True,
                    "depth_ratio": depth_ratio,
                    "disposition": "CONFIRMED_KNOWN",
                    "taxonomy_class": "catalog_confirmed_eb",
                }

        # No match found -> Discovery Taxonomy
        if is_known_multi and d_cand > 0:
            tax_class = "multi_planet_additional_candidate"
            disp = "NOVEL_CANDIDATE"
        else:
            tax_class = "genuine_novel_discovery"
            disp = "NOVEL_CANDIDATE"

        return {
            "is_match": False,
            "catalog_match": None,
            "catalog_id": None,
            "matched_period": None,
            "matched_epoch": None,
            "cycle_index": 0,
            "phase_offset_days": 0.0,
            "phase_normalized": 0.0,
            "delta_t_window": 0.0,
            "is_secondary": False,
            "depth_ratio": 1.0,
            "disposition": disp,
            "taxonomy_class": tax_class,
        }


EphemerisCrossMatcher = EphemerisMatcher  # Exact synonym


# ==============================================================================
# 3. InstrumentalFalsePositiveFilter: Centroid & Quality Triage
# ==============================================================================

class InstrumentalFalsePositiveFilter:
    """Triage engine for centroid pull, stray light, and spacecraft instrumental artifacts."""

    MAX_CENTROID_SHIFT_MPIX: float = 3.0
    MAX_CENTROID_SNR: float = 3.5
    MAX_BKG_CORRELATION: float = 0.80
    MIN_DUMP_BUFFER_DAYS: float = 0.025  # ~36 minutes
    MIN_ARG_BUFFER_DAYS: float = 0.025   # ~36 minutes

    @classmethod
    def evaluate(
        cls,
        cand: CandidateAnomaly,
        target: Optional[HarvestedTarget] = None,
    ) -> InstrumentalVettingReceipt:
        """Evaluates centroid stability, background decorrelation, and quality bitmasks."""
        t_peak = cand.time_peak_bjd if cand.time_peak_bjd != 0.0 else cand.t_peak
        veto_reasons: List[str] = []

        # 1. Spacecraft Telemetry & Quality Masks
        clean_qual = bool(cand.quality_flag_clean)
        fatal_bits = 0
        min_dt_dump = 999.0
        min_dt_arg = 999.0

        if target is not None and target.time is not None and len(target.time) > 0:
            t_arr = target.time
            n_pts = len(t_arr)
            k_start = int(np.searchsorted(t_arr, cand.t_start))
            k_end = int(np.searchsorted(t_arr, cand.t_end))
            k_start = max(0, min(k_start, n_pts - 1))
            k_end = max(k_start, min(k_end, n_pts - 1))

            q_arr = target.sap_quality if target.sap_quality is not None else np.zeros(n_pts, dtype=np.int32)
            event_q = q_arr[k_start:k_end + 1]
            fatal_bits = int(np.bitwise_or.reduce(event_q)) & FATAL_QUALITY_MASK
            if fatal_bits != 0:
                clean_qual = False

            # Desaturation (bit 32) proximity buffer
            dump_cadences = np.where((q_arr & 32) != 0)[0]
            if len(dump_cadences) > 0:
                min_dt_dump = float(np.min(np.abs(t_arr[dump_cadences] - t_peak)))

            # Argabrightening (bit 2048) proximity buffer
            arg_cadences = np.where((q_arr & 2048) != 0)[0]
            if len(arg_cadences) > 0:
                min_dt_arg = float(np.min(np.abs(t_arr[arg_cadences] - t_peak)))
        else:
            # Fallback to candidate metadata
            if cand.metadata.get("momentum_dump", False) or cand.metadata.get("is_momentum_dump", False):
                min_dt_dump = 0.005
            if cand.metadata.get("argabrightening", False):
                min_dt_arg = 0.005

        # 2. Sub-Pixel Centroid Shift Evaluation
        dx_mpix, dy_mpix = 0.0, 0.0
        dr_eff = float(cand.delta_centroid_mpix)
        sig_r = 0.1
        cent_snr = 0.0

        if target is not None and target.mom_centr1 is not None and target.mom_centr2 is not None:
            cx = target.mom_centr1
            cy = target.mom_centr2
            t_arr = target.time
            if t_arr is not None and len(t_arr) > 0:
                n_pts = len(t_arr)
                k_start = int(np.searchsorted(t_arr, cand.t_start))
                k_end = int(np.searchsorted(t_arr, cand.t_end))
                k_start = max(0, min(k_start, n_pts - 1))
                k_end = max(k_start, min(k_end, n_pts - 1))
                n_in = k_end - k_start + 1

                w_base = max(24, 3 * n_in)
                b_lo = max(0, k_start - w_base)
                b_hi = min(n_pts, k_end + w_base + 1)
                q_arr = target.sap_quality if target.sap_quality is not None else np.zeros(n_pts, dtype=np.int32)

                base_mask = np.ones(b_hi - b_lo, dtype=bool)
                base_mask[k_start - b_lo:k_end - b_lo + 1] = False
                base_mask &= (q_arr[b_lo:b_hi] == 0) & np.isfinite(cx[b_lo:b_hi]) & np.isfinite(cy[b_lo:b_hi])

                cx_base = cx[b_lo:b_hi][base_mask]
                cy_base = cy[b_lo:b_hi][base_mask]

                if len(cx_base) >= 8:
                    cx_med = float(np.median(cx_base))
                    cy_med = float(np.median(cy_base))
                    sig_x = float(1.4826 * np.median(np.abs(cx_base - cx_med)))
                    sig_y = float(1.4826 * np.median(np.abs(cy_base - cy_med)))
                else:
                    cx_med, cy_med = float(np.nanmedian(cx)), float(np.nanmedian(cy))
                    sig_x, sig_y = 0.001, 0.001

                cx_in = cx[k_start:k_end + 1]
                cy_in = cy[k_start:k_end + 1]
                dx_mpix = float(abs(np.nanmedian(cx_in) - cx_med) * 1000.0)
                dy_mpix = float(abs(np.nanmedian(cy_in) - cy_med) * 1000.0)
                dr_eff = float(math.sqrt(dx_mpix * dx_mpix + dy_mpix * dy_mpix))

                sig_r = float(math.sqrt((sig_x**2 + sig_y**2) / max(1, n_in)) * 1000.0)
                sig_r = max(0.05, sig_r)
                cent_snr = float(dr_eff / sig_r)

        # 3. Sky Background Correlation Evaluation
        r_val = float(cand.bkg_correlation)
        amp_b = 0.0

        if target is not None and target.sap_bkg is not None and target.sap_flux is not None and target.time is not None:
            t_arr = target.time
            n_pts = len(t_arr)
            k_start = int(np.searchsorted(t_arr, cand.t_start))
            k_end = int(np.searchsorted(t_arr, cand.t_end))
            n_in = max(1, k_end - k_start + 1)
            w_diag = max(24, 3 * n_in)
            d_lo = max(0, k_start - w_diag)
            d_hi = min(n_pts, k_end + w_diag + 1)
            q_arr = target.sap_quality if target.sap_quality is not None else np.zeros(n_pts, dtype=np.int32)

            flux = target.sap_flux
            bkg = target.sap_bkg
            valid_d = np.isfinite(flux[d_lo:d_hi]) & np.isfinite(bkg[d_lo:d_hi]) & (q_arr[d_lo:d_hi] == 0)
            f_win = flux[d_lo:d_hi][valid_d]
            b_win = bkg[d_lo:d_hi][valid_d]
            if len(f_win) >= 8 and np.std(f_win) > 1e-6 and np.std(b_win) > 1e-6:
                r_val = float(np.corrcoef(f_win, b_win)[0, 1])
                med_b = float(np.median(b_win))
                amp_b = float(np.max(np.abs(b_win - med_b)) / max(1.0, med_b))

        # 4. Instrumental Vetoes
        if dr_eff >= cls.MAX_CENTROID_SHIFT_MPIX:
            veto_reasons.append("CENTROID_PULL")
        elif dr_eff >= 2.0 and cent_snr >= cls.MAX_CENTROID_SNR:
            veto_reasons.append("CENTROID_PULL")

        if abs(r_val) >= cls.MAX_BKG_CORRELATION:
            veto_reasons.append("STRAY_LIGHT")

        if min_dt_dump <= cls.MIN_DUMP_BUFFER_DAYS:
            veto_reasons.append("MOMENTUM_DUMP")

        if min_dt_arg <= cls.MIN_ARG_BUFFER_DAYS:
            veto_reasons.append("ARGABRIGHTENING")

        if not clean_qual and fatal_bits != 0 and "MOMENTUM_DUMP" not in veto_reasons and "ARGABRIGHTENING" not in veto_reasons:
            veto_reasons.append("QUALITY_CORRUPTED")

        is_fp = len(veto_reasons) > 0

        # 5. Continuous Sub-Scores
        # Centroid sub-score
        if dr_eff <= 0.5:
            s_cent = 1.0
        elif dr_eff < 3.0:
            s_cent = 1.0 - ((dr_eff - 0.5) / 2.5) ** 1.5
        else:
            s_cent = 0.0
        if cent_snr > 2.5 and dr_eff > 1.5:
            s_cent *= max(0.0, 1.0 - 0.25 * (cent_snr - 2.5))

        # Background sub-score
        abs_r = abs(r_val)
        if abs_r <= 0.20:
            s_bkg = 1.0
        elif abs_r < 0.80:
            s_bkg = 1.0 - ((abs_r - 0.20) / 0.60) ** 2
        else:
            s_bkg = 0.0

        # Spacecraft quality sub-score
        dump_hours = min_dt_dump * 24.0
        if not clean_qual or dump_hours < 0.5:
            s_qual = 0.0
        elif dump_hours < 2.0:
            s_qual = (dump_hours - 0.5) / 1.5
        else:
            s_qual = 1.0

        s_inst = 0.0 if is_fp else float(0.45 * s_cent + 0.35 * s_bkg + 0.20 * s_qual)

        return InstrumentalVettingReceipt(
            delta_centroid_mpix=dr_eff,
            centroid_snr=cent_snr,
            delta_x_mpix=dx_mpix,
            delta_y_mpix=dy_mpix,
            centroid_sigma_mpix=sig_r,
            centroid_passed=("CENTROID_PULL" not in veto_reasons),
            bkg_correlation=r_val,
            bkg_relative_amp=amp_b,
            bkg_passed=("STRAY_LIGHT" not in veto_reasons),
            quality_flag_clean=clean_qual,
            fatal_quality_bits=fatal_bits,
            min_dump_distance_hours=dump_hours,
            thruster_veto_passed=("MOMENTUM_DUMP" not in veto_reasons),
            argabrightening_detected=("ARGABRIGHTENING" in veto_reasons),
            instrumental_score=s_inst,
            is_instrumental_fp=is_fp,
            veto_reasons=veto_reasons,
        )


InstrumentalTriageEngine = InstrumentalFalsePositiveFilter  # Exact synonym


# ==============================================================================
# 4. AstrophysicalConsistencyVetter: Odd-Even, Duration & Morphology
# ==============================================================================

class AstrophysicalConsistencyVetter:
    """Vetting engine for odd-even parity, Keplerian duration bounds, and profile morphology."""

    @classmethod
    def evaluate(
        cls,
        cand: CandidateAnomaly,
        target: Optional[HarvestedTarget] = None,
        period_days: Optional[float] = None,
    ) -> PhysicalVettingReceipt:
        """Evaluates physical transit consistency receipts."""
        veto_reasons: List[str] = []
        d_cand = float(cand.depth_ppm)
        dur_hrs = float(cand.duration_hours)

        # 1. Odd-Even Depth Parity Test
        # Check if odd-even metadata is already provided
        p_val = period_days or cand.metadata.get("catalog_period") or cand.metadata.get("period")
        d_odd = cand.metadata.get("depth_odd_ppm", d_cand)
        d_even = cand.metadata.get("depth_even_ppm", d_cand)
        oe_sig = cand.metadata.get("odd_even_sigma")

        evaluable_oe = False
        if oe_sig is not None:
            oe_sigma_val = float(oe_sig)
            evaluable_oe = True
        elif target is not None and p_val is not None and p_val > 0 and cand.track == "Track_A":
            # Compute odd-even parity from target light curve
            t_arr = target.time
            f_arr = target.sap_flux
            if t_arr is not None and f_arr is not None and len(t_arr) > 0:
                t0 = cand.time_peak_bjd or cand.t_peak
                half_dur = (dur_hrs / 24.0) * 0.5
                odd_depths: List[float] = []
                even_depths: List[float] = []

                n_min = int(math.floor((t_arr[0] - t0) / p_val))
                n_max = int(math.ceil((t_arr[-1] - t0) / p_val))

                for n in range(n_min, n_max + 1):
                    tn = t0 + n * p_val
                    in_transit = np.abs(t_arr - tn) <= half_dur
                    base = (np.abs(t_arr - tn) > half_dur * 1.5) & (np.abs(t_arr - tn) <= half_dur * 3.5)
                    if np.sum(in_transit) >= 1 and np.sum(base) >= 4:
                        d_meas = (np.median(f_arr[base]) - np.median(f_arr[in_transit])) / np.median(f_arr[base]) * 1e6
                        if n % 2 == 1:
                            odd_depths.append(d_meas)
                        else:
                            even_depths.append(d_meas)

                if len(odd_depths) >= 1 and len(even_depths) >= 1:
                    d_odd = float(np.mean(odd_depths))
                    d_even = float(np.mean(even_depths))
                    err_odd = max(100.0, float(np.std(odd_depths) if len(odd_depths) > 1 else 300.0))
                    err_even = max(100.0, float(np.std(even_depths) if len(even_depths) > 1 else 300.0))
                    oe_sigma_val = float(abs(d_odd - d_even) / math.sqrt(err_odd**2 + err_even**2))
                    evaluable_oe = True
                else:
                    oe_sigma_val = 0.0
            else:
                oe_sigma_val = 0.0
        else:
            oe_sigma_val = 0.0

        oe_passed = True
        if evaluable_oe and oe_sigma_val >= 3.0:
            oe_passed = False
            veto_reasons.append("BEB")

        # 2. Keplerian Duration Bounds
        dur_ratio = 1.0
        dur_passed = True
        if p_val is not None and p_val > 0 and cand.depth_ppm > 0:
            delta = cand.depth_ppm * 1e-6
            # Solar density circular duration bound
            t_circ = 1.791 * (1.0 + math.sqrt(max(0.0, delta))) * (p_val ** (1.0 / 3.0))
            dur_ratio = dur_hrs / max(0.1, t_circ)
            if dur_ratio > 3.0:
                dur_passed = False
                veto_reasons.append("UNPHYSICAL_DURATION")

        # 3. Profile Morphology Ratio (tau / T)
        # Evaluate xi_shape = (T10 - T90) / (2 * T10)
        morph_ratio = 0.12
        shape_type = "U-shape"

        if cand.flux_window is not None and len(cand.flux_window) >= 8 and d_cand > 0:
            f_win = cand.flux_window
            t_win = cand.time_window if cand.time_window is not None else np.linspace(0, 1, len(f_win))
            med_f = float(np.median(f_win))
            min_f = float(np.min(f_win))
            dip_depth = med_f - min_f
            if dip_depth > 1e-6:
                d10 = dip_depth * 0.10
                d90 = dip_depth * 0.90
                pts_10 = np.where((med_f - f_win) >= d10)[0]
                pts_90 = np.where((med_f - f_win) >= d90)[0]
                if len(pts_10) > 0:
                    t10 = float(t_win[pts_10[-1]] - t_win[pts_10[0]])
                    t90 = float(t_win[pts_90[-1]] - t_win[pts_90[0]]) if len(pts_90) > 1 else 0.0
                    if t10 > 0:
                        morph_ratio = max(0.01, min(0.50, (t10 - t90) / (2.0 * t10)))
        elif cand.metadata.get("morphology_ratio") is not None:
            morph_ratio = float(cand.metadata["morphology_ratio"])
        elif cand.anomaly_type == "tertiary_eclipse":
            morph_ratio = 0.32
        elif cand.anomaly_type == "micro_flare":
            # Asymmetry calculation for flares
            shape_type = "asymmetric_flare"
            morph_ratio = 0.25

        if shape_type != "asymmetric_flare":
            if morph_ratio <= 0.25:
                shape_type = "U-shape"
            elif morph_ratio <= 0.38:
                shape_type = "intermediate"
            else:
                shape_type = "V-shape"

        # Pure V-shape with centroid shift >= 1.5 mpix -> BEB
        if morph_ratio >= 0.45 and cand.delta_centroid_mpix >= 1.5:
            if "BEB" not in veto_reasons:
                veto_reasons.append("BEB")

        # 4. Continuous Sub-Scores
        # Odd-even sub-score
        if not evaluable_oe:
            s_oe = 1.0
        else:
            s_oe = max(0.0, min(1.0, 1.0 - oe_sigma_val / 3.0))

        # Morphology sub-score
        if shape_type == "U-shape":
            s_morph = 1.0
        elif shape_type == "intermediate":
            s_morph = max(0.0, min(1.0, 1.0 - (morph_ratio - 0.20) / 0.25))
        elif shape_type == "asymmetric_flare":
            s_morph = 1.0
        else:  # V-shape
            s_morph = max(0.0, 1.0 - (morph_ratio - 0.20) / 0.25)

        # Duration multiplier
        if dur_ratio <= 1.5:
            m_dur = 1.0
        elif dur_ratio <= 3.0:
            m_dur = 1.0 - 0.15 * (dur_ratio - 1.5) / 1.5
        elif dur_ratio <= 8.0:
            m_dur = 0.50
        else:
            m_dur = 0.10

        is_fp = len(veto_reasons) > 0
        s_phys = 0.0 if is_fp else float(m_dur * (0.60 * s_oe + 0.40 * s_morph))

        return PhysicalVettingReceipt(
            odd_even_sigma=oe_sigma_val,
            odd_even_passed=oe_passed,
            depth_odd_ppm=float(d_odd),
            depth_even_ppm=float(d_even),
            duration_ratio=dur_ratio,
            duration_passed=dur_passed,
            morphology_ratio=morph_ratio,
            morphology_shape=shape_type,
            morphology_score=s_morph,
            physical_score=s_phys,
            is_physical_fp=is_fp,
            veto_reasons=veto_reasons,
        )


PhysicalConsistencyEngine = AstrophysicalConsistencyVetter  # Exact synonym


# ==============================================================================
# 5. Composite Scoring & Discovery Tier Ranking Engine
# ==============================================================================

def compute_composite_vetting_score(
    cand: CandidateAnomaly,
    inst_receipt: InstrumentalVettingReceipt,
    phys_receipt: PhysicalVettingReceipt,
) -> Tuple[float, str, Optional[str]]:
    """Computes composite vetting score S_vet in [0.0, 1.0], discovery tier, and rejection reason."""
    all_vetoes = inst_receipt.veto_reasons + phys_receipt.veto_reasons
    if len(all_vetoes) > 0:
        return 0.0, "Rejected", all_vetoes[0]

    # Sub-score 1: SNR (calibrated to astronomical thresholds: 5.0 Bronze, 6.0 Silver, 7.1 Gold)
    snr = cand.snr
    if snr < 5.0:
        s_snr = max(0.0, snr / 10.0)
    elif snr < 7.1:
        s_snr = 0.50 + 0.35 * ((snr - 5.0) / 2.1)
    elif snr < 10.0:
        s_snr = 0.85 + 0.15 * ((snr - 7.1) / 2.9)
    else:
        s_snr = 1.00

    # Sub-score 2: Centroid stability (Explorer 2 formulation)
    dr = inst_receipt.delta_centroid_mpix
    if dr <= 0.5:
        s_centr = 1.0
    elif dr < 3.0:
        s_centr = 1.0 - ((dr - 0.5) / 2.5) ** 1.5
    else:
        s_centr = 0.0

    # Sub-score 3: Background decorrelation (Explorer 2 formulation)
    abs_r = abs(inst_receipt.bkg_correlation)
    if abs_r <= 0.20:
        s_bkg = 1.0
    elif abs_r < 0.80:
        s_bkg = 1.0 - ((abs_r - 0.20) / 0.60) ** 2
    else:
        s_bkg = 0.0

    # Sub-score 4: Odd-Even parity
    if phys_receipt.odd_even_sigma == 0.0:
        s_oe = 1.0
    elif phys_receipt.odd_even_sigma < 3.0:
        s_oe = max(0.0, 1.0 - (phys_receipt.odd_even_sigma / 3.0) ** 1.5)
    else:
        s_oe = 0.0

    # Sub-score 5: Morphology
    s_morph = phys_receipt.morphology_score

    # Convex combination
    s_raw = 0.25 * s_snr + 0.25 * s_centr + 0.15 * s_bkg + 0.20 * s_oe + 0.15 * s_morph

    # Multipliers
    m_qual = 1.0 if inst_receipt.quality_flag_clean else 0.0
    dur_ratio = phys_receipt.duration_ratio
    if dur_ratio <= 1.5:
        m_dur = 1.0
    elif dur_ratio <= 3.0:
        m_dur = 1.0 - 0.15 * (dur_ratio - 1.5) / 1.5
    elif dur_ratio <= 8.0:
        m_dur = 0.50
    else:
        m_dur = 0.10

    s_vet = float(m_qual * m_dur * s_raw)

    # Discovery Tier Gates
    # Tier 1 (Gold Discovery)
    is_gold = (
        s_vet >= 0.85
        and dr < 1.5
        and abs(inst_receipt.bkg_correlation) < 0.20
        and phys_receipt.odd_even_sigma < 2.0
        and (phys_receipt.morphology_shape == "U-shape" or cand.track in ("Track_B", "Track_C"))
        and cand.snr >= 7.1
        and inst_receipt.quality_flag_clean
    )
    if is_gold:
        return s_vet, "Tier_1_Gold", None

    # Tier 2 (Silver Candidate)
    is_silver = (
        s_vet >= 0.70
        and dr < 3.0
        and abs(inst_receipt.bkg_correlation) < 0.50
        and phys_receipt.odd_even_sigma < 3.0
        and phys_receipt.morphology_ratio <= 0.38
        and cand.snr >= 6.0
    )
    if is_silver:
        return s_vet, "Tier_2_Silver", None

    # Tier 3 (Bronze Candidate)
    is_bronze = (
        s_vet >= 0.50
        and dr < 3.0
        and abs(inst_receipt.bkg_correlation) < 0.80
        and cand.snr >= 5.0
    )
    if is_bronze:
        return s_vet, "Tier_3_Bronze", None

    return s_vet, "Rejected", "LOW_VETTING_SCORE"



# ==============================================================================
# 6. ExpeditionVetter Master Class
# ==============================================================================

class ExpeditionVetter:
    """Master false-positive vetting and ephemeris cross-matching engine."""

    def __init__(
        self,
        ephemeris_matcher: Optional[EphemerisMatcher] = None,
        fp_filter: Optional[InstrumentalFalsePositiveFilter] = None,
        phys_vetter: Optional[AstrophysicalConsistencyVetter] = None,
    ) -> None:
        self.matcher = ephemeris_matcher or EphemerisMatcher()
        self.fp_filter = fp_filter or InstrumentalFalsePositiveFilter()
        self.phys_vetter = phys_vetter or AstrophysicalConsistencyVetter()

    def vet_candidate(
        self,
        candidate: CandidateAnomaly,
        target: Optional[HarvestedTarget] = None,
    ) -> VettedCandidate:
        """Executes full ephemeris cross-matching, instrumental triage, and physical consistency vetting."""
        # 1. Ephemeris Cross-Matching
        eph_res = self.matcher.cross_match(candidate, target)
        matched_p = eph_res.get("matched_period")
        p_eff = matched_p or candidate.metadata.get("catalog_period") or candidate.metadata.get("period")

        # 2. Instrumental Triage
        inst_receipt = self.fp_filter.evaluate(candidate, target)

        # 3. Physical Consistency Vetting
        phys_receipt = self.phys_vetter.evaluate(candidate, target, period_days=p_eff)

        # 4. Composite Scoring & Tier Ranking
        vet_score, tier, rej_reason = compute_composite_vetting_score(candidate, inst_receipt, phys_receipt)

        # Reconcile taxonomy disposition with vetting result
        if tier == "Rejected":
            disp = "FALSE_POSITIVE"
            tax_class = "instrumental_false_positive" if inst_receipt.is_instrumental_fp else "astrophysical_false_positive"
        else:
            disp = eph_res["disposition"]
            tax_class = eph_res["taxonomy_class"]

        # Build candidate ID
        vet_id = f"VET_{candidate.candidate_id}" if not candidate.candidate_id.startswith("VET_") else candidate.candidate_id

        # Instantiate unified VettedCandidate
        vetted = VettedCandidate(
            candidate_id=vet_id,
            target_id=candidate.target_id,
            track=candidate.track,
            survey=candidate.survey,
            anomaly_type=candidate.anomaly_type,
            ra=candidate.ra,
            dec=candidate.dec,
            pm_ra=candidate.pm_ra,
            pm_dec=candidate.pm_dec,
            time_peak_bjd=candidate.time_peak_bjd or candidate.t_peak,
            t_start=candidate.t_start,
            t_peak=candidate.t_peak,
            t_end=candidate.t_end,
            duration_hours=candidate.duration_hours,
            duration_cadences=candidate.duration_cadences,
            n_epochs=candidate.n_epochs,
            period_days=p_eff,
            snr=candidate.snr,
            amplitude=candidate.amplitude,
            depth_ppm=candidate.depth_ppm,
            peak_flux_ratio=candidate.peak_flux_ratio,
            curl_salience=candidate.curl_salience,
            morse_impulse=candidate.morse_impulse,
            eta_nonlocal=candidate.eta_nonlocal,
            causal_class=candidate.causal_class,
            triad_drift=candidate.triad_drift,
            tie_breaking_gap=candidate.tie_breaking_gap,
            odd_even_sigma=phys_receipt.odd_even_sigma,
            odd_even_passed=phys_receipt.odd_even_passed,
            depth_odd_ppm=phys_receipt.depth_odd_ppm,
            depth_even_ppm=phys_receipt.depth_even_ppm,
            duration_ratio=phys_receipt.duration_ratio,
            duration_passed=phys_receipt.duration_passed,
            morphology_ratio=phys_receipt.morphology_ratio,
            morphology_shape=phys_receipt.morphology_shape,
            delta_centroid_mpix=inst_receipt.delta_centroid_mpix,
            centroid_snr=inst_receipt.centroid_snr,
            centroid_passed=inst_receipt.centroid_passed,
            bkg_correlation=inst_receipt.bkg_correlation,
            bkg_passed=inst_receipt.bkg_passed,
            quality_flag_clean=inst_receipt.quality_flag_clean,
            catalog_match=eph_res.get("catalog_match"),
            catalog_id=eph_res.get("catalog_id"),
            disposition=disp,
            taxonomy_class=tax_class,
            vetting_score=vet_score,
            discovery_tier=tier,
            rejection_reason=rej_reason,
            time_window=candidate.time_window,
            flux_window=candidate.flux_window,
            h_clock_window=candidate.h_clock_window,
            h_nope_window=candidate.h_nope_window,
            model_window=candidate.model_window,
            instrumental_receipt=inst_receipt,
            physical_receipt=phys_receipt,
            metadata=dict(candidate.metadata),
        )

        return vetted

    def vet_target(
        self,
        target: HarvestedTarget,
        candidates: List[CandidateAnomaly],
    ) -> List[VettedCandidate]:
        """Vets all candidates identified on a single harvested target."""
        return [self.vet_candidate(cand, target) for cand in candidates]

    def vet_stream(
        self,
        candidate_stream: Iterable[Union[CandidateAnomaly, Tuple[CandidateAnomaly, Optional[HarvestedTarget]]]],
    ) -> Generator[VettedCandidate, None, None]:
        """Streams vetted candidates out-of-core with periodic memory reclamation."""
        count = 0
        for item in candidate_stream:
            if isinstance(item, tuple):
                cand, target = item
            else:
                cand, target = item, None
            yield self.vet_candidate(cand, target)
            count += 1
            if count % 250 == 0:
                gc.collect()


# ==============================================================================
# Functional Helpers
# ==============================================================================

def vet_candidate(
    candidate: CandidateAnomaly,
    target: Optional[HarvestedTarget] = None,
) -> VettedCandidate:
    """Convenience functional interface for vetting a single candidate anomaly."""
    vetter = ExpeditionVetter()
    return vetter.vet_candidate(candidate, target)


def vet_target(
    target: HarvestedTarget,
    candidates: List[CandidateAnomaly],
) -> List[VettedCandidate]:
    """Convenience functional interface for vetting candidates on a single target."""
    vetter = ExpeditionVetter()
    return vetter.vet_target(target, candidates)
