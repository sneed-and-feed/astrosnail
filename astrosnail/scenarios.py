"""
astrosnail.scenarios
====================
Astrophysics benchmark scenario deck for sneed-and-feed/astrosnail.
Implements the 5 canonical benchmark scenarios with deterministic seeding,
exact analytical ground truth, and memory-mapped generator streaming:

1. 01_frb_supernova_cosmic_ray_triage: Multi-band stream with FRBs, SNe, cosmic rays, and glitches.
2. 02_eclipsing_binary_exoplanet_transit_split: Periodic EB and RR Lyrae (H_clock) vs exoplanet transits (H_nope).
3. 03_crowded_field_kbo_crossing_astrometry: 240 frames with 500 stars and 4 moving KBOs with 90° and 180° crossings.
4. 04_lensed_quasar_echo_swarm_factorization: 2 active flaring progenitors (G_act) and 10 lensed echoes (Omega_reg).
5. 05_cosmic_web_void_rsd_tomography: (32, 24, 24) cosmological volume with sheets, knot, Jordan void, and Beltrami jet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.special import j0, j1


@dataclass
class ScenarioData:
    """Standardized container for astrophysics benchmark scenarios.

    Attributes:
        scenario_id: Canonical scenario identifier (e.g. '01_frb_supernova_cosmic_ray_triage').
        name: Human-readable scenario title.
        seed: Deterministic integer random seed used for generation.
        data: Primary observation data (1D/2D array, 3D cube, or frame sequence).
        ground_truth: Dictionary containing exact analytical ground truth tensors, tracks, masks, and catalogs.
        metadata: Physical scenario parameters (cadence, band names, SNR, coordinates, etc.).
        _stream_fn: Optional generator factory for streaming large frame sequences with < 2.0 MB working set.
    """

    scenario_id: str
    name: str
    seed: int
    data: Any
    ground_truth: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _stream_fn: Optional[Callable[[], Iterator[np.ndarray]]] = None

    def stream(self) -> Iterator[np.ndarray]:
        """Returns an iterator over sequential frames or epochs for online causal streaming."""
        if self._stream_fn is not None:
            return iter(self._stream_fn())
        if isinstance(self.data, np.ndarray):
            if self.data.ndim == 3 and self.metadata.get("is_frame_sequence", False):
                return iter(self.data)
            elif self.data.ndim in (1, 2) and self.metadata.get("is_time_series", True):
                return (np.atleast_1d(x) for x in self.data)
            elif self.data.ndim == 3:
                return iter(self.data)
        return iter([self.data])

    def to_dict(self) -> Dict[str, Any]:
        """Serializes scenario metadata and ground truth descriptors (omitting large arrays)."""
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "seed": self.seed,
            "metadata": self.metadata,
            "ground_truth_keys": list(self.ground_truth.keys()),
        }


# =============================================================================
# SCENARIO 01: FRB, Supernova, and Cosmic Ray Triage
# =============================================================================

def generate_scenario_01_frb_supernova(
    seed: int = 20261025,
    n_epochs: int = 25000,
    n_bands: int = 4,
    mode: str = "lightcurve",
    n_frames: int = 1024,
    height: int = 128,
    width: int = 128,
) -> ScenarioData:
    """Scenario 01: Fast Radio Burst, Supernova, and Cosmic-Ray Glitch Triage.

    Generates a 25,000-epoch multi-band photometric & de-dispersed radio stream (or 1024-frame
    focal plane sequence) containing an FRB dispersion sweep impulse, 2 core-collapse supernovae
    (quadratic rise, exponential decay), 80 single-pixel cosmic ray Poisson spikes, 10% glitches,
    and 20 dB noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_epochs, dtype=np.float32)

    # 1. Quiescent Stellar / Host Baseline
    band_names = ["g", "r", "i", "z"][:n_bands]
    f0_nominal = np.array([10.0, 12.0, 11.5, 13.0][:n_bands], dtype=np.float32)
    clean_flux = np.tile(f0_nominal, (n_epochs, 1))

    # 2. Fast Radio Burst (FRB) at t = 10,000
    t_frb = 10000
    sigma_frb = 0.8
    a_frb = np.array([18.0, 15.0, 12.0, 10.0][:n_bands], dtype=np.float32)
    frb_pulse = np.exp(-0.5 * ((t - t_frb) / sigma_frb) ** 2)[:, None] * a_frb[None, :]
    clean_flux += frb_pulse

    # 3. Two Core-Collapse Supernovae
    # SN 1: Explosion at 2500, peak at 5000, decay tau = 4000
    t_exp1, t_peak1, tau_decay1, peak1 = 2500, 5000, 4000.0, 15.0
    sn1_flux = np.zeros(n_epochs, dtype=np.float32)
    mask_rise1 = (t >= t_exp1) & (t <= t_peak1)
    mask_decay1 = t > t_peak1
    sn1_flux[mask_rise1] = peak1 * ((t[mask_rise1] - t_exp1) / (t_peak1 - t_exp1)) ** 2
    sn1_flux[mask_decay1] = peak1 * np.exp(-(t[mask_decay1] - t_peak1) / tau_decay1)

    # SN 2: Explosion at 14000, peak at 17500, decay tau = 3500
    t_exp2, t_peak2, tau_decay2, peak2 = 14000, 17500, 3500.0, 18.0
    sn2_flux = np.zeros(n_epochs, dtype=np.float32)
    mask_rise2 = (t >= t_exp2) & (t <= t_peak2)
    mask_decay2 = t > t_peak2
    sn2_flux[mask_rise2] = peak2 * ((t[mask_rise2] - t_exp2) / (t_peak2 - t_exp2)) ** 2
    sn2_flux[mask_decay2] = peak2 * np.exp(-(t[mask_decay2] - t_peak2) / tau_decay2)

    # Add SNe to all bands
    clean_flux += sn1_flux[:, None]
    clean_flux += sn2_flux[:, None]

    # 4. White noise (nominal SNR = 20 dB relative to signal)
    sigma_noise = 1.0
    noise = rng.normal(0.0, sigma_noise, size=(n_epochs, n_bands)).astype(np.float32)
    observed_flux = clean_flux + noise

    # 5. 80 Single-Pixel Cosmic Ray Poisson Spikes
    # Generate 80 well-separated epochs (> 15 epochs separation, avoiding FRB & SNe onsets)
    forbidden_zones = [
        (t_frb - 30, t_frb + 30),
        (t_exp1 - 30, t_exp1 + 30),
        (t_exp2 - 30, t_exp2 + 30),
    ]
    candidate_epochs = []
    # Distribute 80 targets across epochs 500 to 24500
    grid_points = np.linspace(500, 24500, 100, dtype=int)
    for g in grid_points:
        jitter = int(rng.integers(-50, 50))
        epoch_idx = int(np.clip(g + jitter, 100, n_epochs - 100))
        in_forbidden = any(start <= epoch_idx <= end for start, end in forbidden_zones)
        if not in_forbidden and all(abs(epoch_idx - c) >= 20 for c in candidate_epochs):
            candidate_epochs.append(epoch_idx)
        if len(candidate_epochs) == 80:
            break

    cosmic_ray_epochs = np.array(sorted(candidate_epochs), dtype=np.int32)
    for ep in cosmic_ray_epochs:
        # Isolated 1-sample spike (> 10 sigma)
        spike_amp = rng.uniform(12.0, 30.0, size=n_bands).astype(np.float32)
        observed_flux[ep] += spike_amp

    # 6. 10% Glitches (Readout anomalies & satellite glints)
    n_glitches = int(0.10 * n_epochs)
    available_indices = np.setdiff1d(
        np.arange(10, n_epochs - 10),
        np.concatenate([cosmic_ray_epochs, cosmic_ray_epochs + 1, np.arange(t_frb - 5, t_frb + 6)]),
    )
    glitch_epochs = rng.choice(available_indices, size=n_glitches, replace=False)
    glitch_mask = np.zeros(n_epochs, dtype=bool)
    glitch_mask[glitch_epochs] = True

    for gep in glitch_epochs:
        sign = float(rng.choice([-1.0, 1.0]))
        amp = sign * rng.uniform(3.0, 6.0, size=n_bands).astype(np.float32)
        observed_flux[gep] += amp

    ground_truth = {
        "frb_epoch": t_frb,
        "frb_amplitude": a_frb.tolist(),
        "supernova_events": [
            {"id": 1, "t_exp": t_exp1, "t_peak": t_peak1, "tau_decay": tau_decay1, "peak_flux": peak1},
            {"id": 2, "t_exp": t_exp2, "t_peak": t_peak2, "tau_decay": tau_decay2, "peak_flux": peak2},
        ],
        "cosmic_ray_epochs": cosmic_ray_epochs,
        "glitch_mask": glitch_mask,
        "clean_transient_signal": clean_flux[:, 0],
        "multi_band_clean": clean_flux,
        "multi_band_observed": observed_flux,
        "expected_false_triggers": 0,
    }

    metadata = {
        "cadence_sec": 1.0,
        "n_bands": n_bands,
        "band_names": band_names,
        "n_epochs": n_epochs,
        "snr_db": 20.0,
        "is_time_series": True,
        "is_frame_sequence": False,
    }

    # Primary data is 1D light curve of the primary band
    primary_data = observed_flux[:, 0].copy()

    # Optional frame streaming mode generator
    stream_fn = None
    if mode == "frames":
        metadata["is_frame_sequence"] = True
        metadata["is_time_series"] = False
        metadata["n_frames"] = n_frames
        metadata["frame_shape"] = (height, width)

        # Lazy generator yielding 128x128 frames on the fly (working set < 2.0 MB)
        def _frame_generator() -> Iterator[np.ndarray]:
            frame_rng = np.random.default_rng(seed + 99)
            # Static host galaxy profile in center
            y_grid, x_grid = np.ogrid[:height, :width]
            cy, cx = height / 2.0, width / 2.0
            r2 = (y_grid - cy) ** 2 + (x_grid - cx) ** 2
            host_profile = (30.0 * np.exp(-0.5 * r2 / (8.0 ** 2))).astype(np.float32)

            for fi in range(n_frames):
                epoch_idx = int(fi * (n_epochs / n_frames))
                frame = host_profile.copy()
                # Center transient point source flux
                transient_val = float(clean_flux[epoch_idx, 0])
                frame[int(cy), int(cx)] += transient_val

                # Add noise
                frame += frame_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)

                # Add cosmic ray if in this frame
                if any(epoch_idx <= cr < epoch_idx + int(n_epochs / n_frames) for cr in cosmic_ray_epochs):
                    rx = int(frame_rng.integers(10, width - 10))
                    ry = int(frame_rng.integers(10, height - 10))
                    frame[ry, rx] += float(frame_rng.uniform(15.0, 35.0))

                yield frame

        stream_fn = _frame_generator

    return ScenarioData(
        scenario_id="01_frb_supernova_cosmic_ray_triage",
        name="FRB, Supernova, and Cosmic-Ray Glitch Triage",
        seed=seed,
        data=primary_data,
        ground_truth=ground_truth,
        metadata=metadata,
        _stream_fn=stream_fn,
    )


# =============================================================================
# SCENARIO 02: Eclipsing Binary and Exoplanet Transit Split
# =============================================================================

def generate_scenario_02_eclipsing_binary(
    seed: int = 20261025,
    n_epochs: int = 25000,
    dt_days: float = 0.02,
) -> ScenarioData:
    """Scenario 02: Eclipsing Binary and Exoplanet Transit Split.

    Simulates high-precision space photometry over 500 days (25,000 epochs, dt = 28.8 min).
    Contains periodic components in H_clock (P1 = 3.412d EB with primary/secondary eclipses
    and ellipsoidal harmonics, plus P_puls = 0.567d RR Lyrae pulsations) and aperiodic
    transients in H_nope (P2 = 18.71d exoplanet transit of 0.8 mmag depth with limb darkening,
    and a non-repeating satellite burn flare at t = 214.3d).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_epochs, dtype=np.float64) * dt_days

    # 1. Periodic Component: Eclipsing Binary (P1 = 3.412 days)
    p1 = 3.412
    phi1 = (t / p1) % 1.0
    # Primary eclipse at phase 0.0
    delta_phi1_pri = np.minimum(phi1, 1.0 - phi1)
    ecl_primary = 0.18 * np.exp(-0.5 * (delta_phi1_pri / 0.025) ** 2)
    # Secondary eclipse at phase 0.5
    delta_phi1_sec = np.abs(phi1 - 0.5)
    ecl_secondary = 0.07 * np.exp(-0.5 * (delta_phi1_sec / 0.025) ** 2)
    # Ellipsoidal variation & reflection harmonics
    eb_harmonics = -0.015 * np.cos(4.0 * np.pi * phi1) - 0.008 * np.cos(2.0 * np.pi * phi1)
    f_eb = 1.0 - ecl_primary - ecl_secondary + eb_harmonics

    # 2. Periodic Component: RR Lyrae Pulsations (P_puls = 0.567 days)
    p_puls = 0.567
    phi_p = (t / p_puls) % 1.0
    f_puls = (
        0.04 * np.sin(2.0 * np.pi * phi_p)
        + 0.02 * np.sin(4.0 * np.pi * phi_p + 0.8)
        + 0.01 * np.sin(6.0 * np.pi * phi_p + 1.6)
    )

    h_clock_true = (f_eb + f_puls - 1.0).astype(np.float32)

    # 3. Aperiodic Component: Exoplanet Transit (P2 = 18.71 days, depth = 0.8 mmag)
    p2 = 18.71
    transit_depth = 0.000737  # 0.8 mmag in relative flux
    t_dur = 0.16  # duration = 3.84 hours (8 samples)
    t_mids = np.arange(5.0, 500.0, p2)

    transit_signal = np.zeros(n_epochs, dtype=np.float64)
    u1, u2 = 0.35, 0.25  # Mandel & Agol quadratic limb-darkening coefficients

    for t_mid in t_mids:
        in_transit = np.abs(t - t_mid) <= (t_dur / 2.0)
        if np.any(in_transit):
            z = np.abs(t[in_transit] - t_mid) / (t_dur / 2.0)
            z_clamped = np.clip(z, 0.0, 1.0)
            mu = np.sqrt(np.maximum(0.0, 1.0 - z_clamped ** 2))
            # Limb-darkened transit profile
            prof = transit_depth * (1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2)
            transit_signal[in_transit] = np.maximum(transit_signal[in_transit], prof)

    # 4. Aperiodic Component: Non-Repeating Satellite Burn / Flare at t = 214.3d
    t_burn = 214.3
    burn_epoch = int(round(t_burn / dt_days))
    mask_burn = t >= t_burn
    flare_signal = np.zeros(n_epochs, dtype=np.float64)
    flare_signal[mask_burn] = 0.05 * np.exp(-(t[mask_burn] - t_burn) / 0.12)

    h_nope_true = (1.0 - transit_signal + flare_signal).astype(np.float32)

    # 5. Instrumental Photometric Noise (sigma = 0.2 mmag = 0.0002)
    noise = rng.normal(0.0, 0.0002, size=n_epochs).astype(np.float32)
    observed_flux = (1.0 + h_clock_true - transit_signal + flare_signal + noise).astype(np.float32)

    ground_truth = {
        "h_clock_true": h_clock_true,
        "h_nope_true": h_nope_true,
        "p1_eb_period_days": p1,
        "p_puls_period_days": p_puls,
        "p2_transit_period_days": p2,
        "transit_depth_mmag": 0.8,
        "transit_midpoints": t_mids,
        "satellite_burn_epoch": burn_epoch,
        "satellite_burn_day": t_burn,
    }

    metadata = {
        "cadence_days": dt_days,
        "cadence_hours": dt_days * 24.0,
        "duration_days": float(t[-1]),
        "n_epochs": n_epochs,
        "is_time_series": True,
        "is_frame_sequence": False,
    }

    return ScenarioData(
        scenario_id="02_eclipsing_binary_exoplanet_transit_split",
        name="Eclipsing Binary and Exoplanet Transit Split",
        seed=seed,
        data=observed_flux,
        ground_truth=ground_truth,
        metadata=metadata,
    )


# =============================================================================
# SCENARIO 03: Crowded-Field KBO Crossing Astrometry
# =============================================================================

def generate_scenario_03_crowded_field(
    seed: int = 20261025,
    n_frames: int = 240,
    height: int = 128,
    width: int = 128,
    n_stars: int = 500,
    stream_mode: bool = False,
) -> ScenarioData:
    """Scenario 03: Crowded-Field KBO Crossing Astrometry.

    Simulates a 240-frame sequence (128x128) of a dense stellar field (500 stars, V in [14, 21])
    with 4 faint moving Kuiper Belt Objects (KBOs) undergoing simultaneous 90° and 180°
    crossing occlusions at frame t = 120 and (64.0, 64.0).
    """
    rng = np.random.default_rng(seed)

    # 1. 500 Background Stars
    star_y = rng.uniform(4.0, height - 4.0, size=n_stars).astype(np.float32)
    star_x = rng.uniform(4.0, width - 4.0, size=n_stars).astype(np.float32)
    star_mag = rng.uniform(14.0, 21.0, size=n_stars).astype(np.float32)
    star_flux = (500.0 * (10.0 ** (-0.4 * (star_mag - 14.0)))).astype(np.float32)
    sigma_psf = 1.25

    # Pre-render static stellar background onto 128x128
    bg_stars = np.zeros((height, width), dtype=np.float32)
    for sy, sx, fl in zip(star_y, star_x, star_flux):
        iy, ix = int(round(sy)), int(round(sx))
        # Stamp 9x9 patch
        y_min, y_max = max(0, iy - 4), min(height, iy + 5)
        x_min, x_max = max(0, ix - 4), min(width, ix + 5)
        py, px = np.ogrid[y_min:y_max, x_min:x_max]
        patch = fl * np.exp(-0.5 * ((py - sy) ** 2 + (px - sx) ** 2) / (sigma_psf ** 2))
        bg_stars[y_min:y_max, x_min:x_max] += patch.astype(np.float32)

    # 2. 4 Moving KBO Trajectories
    # Crossing at t = 120, center = (64.0, 64.0), speed = 0.35 px/frame
    t_cross = 120
    v_speed = 0.35
    t_arr = np.arange(n_frames, dtype=np.float32) - t_cross

    gt_trajectories = np.zeros((4, n_frames, 2), dtype=np.float32)
    gt_velocities = np.array([
        [v_speed, 0.0],    # Body 0: Northbound (+Y)
        [-v_speed, 0.0],   # Body 1: Southbound (-Y) -> 180° head-on with Body 0
        [0.0, v_speed],    # Body 2: Eastbound (+X) -> 90° with Bodies 0 & 1
        [0.0, -v_speed],   # Body 3: Westbound (-X) -> 180° head-on with Body 2, 90° with 0 & 1
    ], dtype=np.float32)

    gt_trajectories[0, :, 0] = 64.0 + v_speed * t_arr
    gt_trajectories[0, :, 1] = 64.0
    gt_trajectories[1, :, 0] = 64.0 - v_speed * t_arr
    gt_trajectories[1, :, 1] = 64.0
    gt_trajectories[2, :, 0] = 64.0
    gt_trajectories[2, :, 1] = 64.0 + v_speed * t_arr
    gt_trajectories[3, :, 0] = 64.0
    gt_trajectories[3, :, 1] = 64.0 - v_speed * t_arr

    kbo_fluxes = [3.5, 3.8, 3.2, 3.6]

    # Pre-select glint frames (10% of frames have isolated specular spikes)
    glint_frames = rng.choice(n_frames, size=int(0.10 * n_frames), replace=False)

    def _render_frame(frame_idx: int) -> np.ndarray:
        frame_rng = np.random.default_rng(seed + 1000 + frame_idx)
        frame = bg_stars.copy()

        # Inject moving KBOs
        for b_idx in range(4):
            by = gt_trajectories[b_idx, frame_idx, 0]
            bx = gt_trajectories[b_idx, frame_idx, 1]
            if 2.0 <= by < height - 2.0 and 2.0 <= bx < width - 2.0:
                iy, ix = int(round(by)), int(round(bx))
                y_min, y_max = max(0, iy - 4), min(height, iy + 5)
                x_min, x_max = max(0, ix - 4), min(width, ix + 5)
                py, px = np.ogrid[y_min:y_max, x_min:x_max]
                kbo_patch = kbo_fluxes[b_idx] * np.exp(-0.5 * ((py - by) ** 2 + (px - bx) ** 2) / (sigma_psf ** 2))
                frame[y_min:y_max, x_min:x_max] += kbo_patch.astype(np.float32)

        # Background noise
        frame += frame_rng.normal(0.0, 0.8, size=(height, width)).astype(np.float32)

        # Inject specular glint if selected
        if frame_idx in glint_frames:
            gy = int(frame_rng.integers(10, height - 10))
            gx = int(frame_rng.integers(10, width - 10))
            frame[gy, gx] += float(frame_rng.uniform(15.0, 30.0))

        return frame

    ground_truth = {
        "gt_trajectories": gt_trajectories,
        "gt_velocities": gt_velocities,
        "crossing_frame": t_cross,
        "crossing_center": [64.0, 64.0],
        "crossing_angles": [90.0, 180.0],
        "background_stars": np.column_stack([star_y, star_x, star_flux]),
        "expected_id_swaps": 0,
    }

    metadata = {
        "n_frames": n_frames,
        "height": height,
        "width": width,
        "n_stars": n_stars,
        "is_frame_sequence": True,
        "is_time_series": False,
    }

    if stream_mode:
        data = None
        stream_fn = lambda: (_render_frame(fi) for fi in range(n_frames))
    else:
        # 240 * 128 * 128 * 4 bytes = 15.7 MB (well below 35 MB hard ceiling)
        data = np.empty((n_frames, height, width), dtype=np.float32)
        for fi in range(n_frames):
            data[fi] = _render_frame(fi)
        stream_fn = lambda: iter(data)

    return ScenarioData(
        scenario_id="03_crowded_field_kbo_crossing_astrometry",
        name="Crowded-Field KBO Crossing Astrometry",
        seed=seed,
        data=data,
        ground_truth=ground_truth,
        metadata=metadata,
        _stream_fn=stream_fn,
    )


# =============================================================================
# SCENARIO 04: Lensed Quasar Echo Swarm Factorization
# =============================================================================

def generate_scenario_04_lensed_quasar(
    seed: int = 20261025,
    n_epochs: int = 2500,
) -> ScenarioData:
    """Scenario 04: Lensed Quasar Echo Swarm Factorization.

    Simulates a gravitationally lensed quasar system with 12 objects: 2 active flaring progenitors
    (G_act, eta_nonlocal >= 0.50) exhibiting Damped Random Walk and distinct flaring onsets,
    and 10 lensed echo images / passive debris registers (Omega_reg) with time delays up to 42.1 days,
    microlensing caustic crossing spikes, and orbital flicker.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_epochs, dtype=np.float64)

    # 1. Progenitor 1 (G1): Damped Random Walk (DRW) + 3 Flares
    tau_drw1 = 150.0
    sigma_drw1 = 1.5
    f1 = np.zeros(n_epochs, dtype=np.float64)
    f1[0] = 20.0
    decay1 = math.exp(-1.0 / tau_drw1)
    scale1 = sigma_drw1 * math.sqrt(1.0 - math.exp(-2.0 / tau_drw1))

    for i in range(1, n_epochs):
        f1[i] = 20.0 + (f1[i - 1] - 20.0) * decay1 + scale1 * rng.standard_normal()

    # 3 distinct flaring bursts
    flare_peaks1 = [(400, 15.0, 20.0), (1100, 25.0, 35.0), (1900, 18.0, 25.0)]
    for f_t, f_amp, f_sig in flare_peaks1:
        f1 += f_amp * np.exp(-0.5 * ((t - f_t) / f_sig) ** 2)

    # 2. Progenitor 2 (G2): Independent DRW + 2 Flares
    tau_drw2 = 80.0
    sigma_drw2 = 1.2
    f2 = np.zeros(n_epochs, dtype=np.float64)
    f2[0] = 15.0
    decay2 = math.exp(-1.0 / tau_drw2)
    scale2 = sigma_drw2 * math.sqrt(1.0 - math.exp(-2.0 / tau_drw2))

    for i in range(1, n_epochs):
        f2[i] = 15.0 + (f2[i - 1] - 15.0) * decay2 + scale2 * rng.standard_normal()

    flare_peaks2 = [(700, 20.0, 30.0), (1600, 22.0, 28.0)]
    for f_t, f_amp, f_sig in flare_peaks2:
        f2 += f_amp * np.exp(-0.5 * ((t - f_t) / f_sig) ** 2)

    # 3. 10 Lensed Echo Images and Passive Debris
    series = np.zeros((n_epochs, 12), dtype=np.float32)
    series[:, 0] = f1.astype(np.float32)
    series[:, 1] = f2.astype(np.float32)

    # Quad-lensed images of G1: delays in days [0.0, 8.4, 21.7, 42.1]
    delays_g1 = [0.0, 8.4, 21.7, 42.1]
    mag_g1 = [1.0, 0.85, 0.62, 0.41]
    for idx, (del_d, mg) in enumerate(zip(delays_g1, mag_g1)):
        shift = int(round(del_d * 2.0))  # 2 samples per day
        shifted_f = np.roll(f1, shift) if shift > 0 else f1.copy()
        if shift > 0:
            shifted_f[:shift] = 20.0
        # Image B (index 3 in series) has caustic crossing spike at t = 1450
        if idx == 1:
            caustic_t = 1450
            caustic_prof = np.zeros(n_epochs, dtype=np.float64)
            mask_c = t >= caustic_t
            delta_c = (t[mask_c] - caustic_t) / 12.0
            caustic_prof[mask_c] = 3.5 * np.maximum(1e-3, delta_c) ** (-0.5) * np.exp(-(t[mask_c] - caustic_t) / 50.0)
            shifted_f += caustic_prof
        series[:, 2 + idx] = (mg * shifted_f).astype(np.float32)

    # Double-lensed images of G2: delays [0.0, 15.3]
    delays_g2 = [0.0, 15.3]
    mag_g2 = [0.90, 0.55]
    for idx, (del_d, mg) in enumerate(zip(delays_g2, mag_g2)):
        shift = int(round(del_d * 2.0))
        shifted_f = np.roll(f2, shift) if shift > 0 else f2.copy()
        if shift > 0:
            shifted_f[:shift] = 15.0
        series[:, 6 + idx] = (mg * shifted_f).astype(np.float32)

    # 4 Passive Echoes with transfer smoothing & orbital flicker
    kernel_widths = [8, 14, 20, 26]
    flicker_periods = [45.0, 62.0, 80.0, 110.0]
    for idx, (kw, per) in enumerate(zip(kernel_widths, flicker_periods)):
        base_f = f1 if idx % 2 == 0 else f2
        # Boxcar smoothing
        kernel = np.ones(kw) / kw
        smoothed = np.convolve(base_f, kernel, mode="same")
        flicker = 1.0 + 0.05 * np.sin(2.0 * np.pi * t / per)
        series[:, 8 + idx] = (0.25 * smoothed * flicker).astype(np.float32)

    # Add Gaussian measurement noise
    noise = rng.normal(0.0, 0.5, size=series.shape).astype(np.float32)
    series += noise

    # 4. Causal Transition Matrix T (12 x 12)
    # Generators 0 and 1 have off-diagonal dominance (eta >= 0.50)
    # Passive registers 2..11 have strong diagonal dominance (eta < 0.50)
    t_mat = np.zeros((12, 12), dtype=np.float64)
    # Generator 0: local=1.0, nonlocal=8.0 -> eta = 8/9 = 0.89 >= 0.50
    t_mat[0, 0] = 1.0
    t_mat[0, 1] = 8.0
    # Generator 1: local=2.0, nonlocal=7.0 -> eta = 7/9 = 0.78 >= 0.50
    t_mat[1, 1] = 2.0
    t_mat[1, 2] = 7.0
    # Passive registers: local=9.0, nonlocal=1.0 -> eta = 1/10 = 0.10 < 0.50
    for p_idx in range(2, 12):
        t_mat[p_idx, p_idx] = 9.0
        t_mat[p_idx, 0 if p_idx % 2 == 0 else 1] = 1.0

    is_generator_true = np.array([True, True] + [False] * 10, dtype=bool)

    ground_truth = {
        "num_progenitors": 2,
        "num_echoes": 10,
        "is_generator_true": is_generator_true,
        "transition_matrix": t_mat,
        "time_delays_days": {
            "g1_A": 0.0, "g1_B": 8.4, "g1_C": 21.7, "g1_D": 42.1,
            "g2_A": 0.0, "g2_B": 15.3,
            "echo_1": 18.0, "echo_2": 25.5, "echo_3": 33.0, "echo_4": 39.8,
        },
        "magnifications": [1.0, 0.85, 0.62, 0.41, 0.90, 0.55, 0.25, 0.25, 0.25, 0.25],
    }

    metadata = {
        "n_epochs": n_epochs,
        "n_objects": 12,
        "cadence_days": 0.5,
        "is_time_series": True,
        "is_frame_sequence": False,
    }

    return ScenarioData(
        scenario_id="04_lensed_quasar_echo_swarm_factorization",
        name="Lensed Quasar Echo Swarm Factorization",
        seed=seed,
        data=series,
        ground_truth=ground_truth,
        metadata=metadata,
    )


# =============================================================================
# SCENARIO 05: Cosmic Web Void RSD Tomography
# =============================================================================

def generate_scenario_05_cosmic_web(
    seed: int = 20261025,
    grid_shape: Tuple[int, int, int] = (32, 24, 24),
    n_skewers: int = 8,
) -> ScenarioData:
    """Scenario 05: 3D Cosmic Web Void RSD Tomography.

    Generates a (32, 24, 24) cosmological volume probed by 8 sparse skewers, containing:
    - Zeldovich pancake sheets across Z = 16.
    - Virialized cluster knot with Redshift-Space Distortion at (16, 16, 16) with delta >= +6.0.
    - Enclosed topological Jordan void cavity core at (16, 6, 6) with delta <= -0.80.
    - Relativistic 3D force-free Beltrami jet (nabla x B = mu B).
    """
    nz, ny, nx = grid_shape
    zg, yg, xg = np.ogrid[:nz, :ny, :nx]

    # Baseline cosmological overdensity: slight underdensity
    delta = np.full(grid_shape, -0.2, dtype=np.float32)

    # 1. Zeldovich Pancake Sheet across Z = 16
    z_sheet = 16.0
    sheet = 2.5 * np.exp(-0.5 * ((zg - z_sheet) / 1.8) ** 2)
    delta += sheet.astype(np.float32)

    # 2. Virialized Cluster Knot at (16, 16, 16) with peak overdensity >= +6.0
    zc, yc, xc = 16.0, 16.0, 16.0
    r_knot2 = (zg - zc) ** 2 + (yg - yc) ** 2 + (xg - xc) ** 2
    knot = 6.5 * np.exp(-0.5 * r_knot2 / (3.0 ** 2))
    delta += knot.astype(np.float32)

    # 3. Jordan Void Cavity Core at (16, 6, 6) with delta <= -0.80
    zv, yv, xv = 16.0, 6.0, 6.0
    rz, ry, rx = 7.0, 5.0, 5.0
    r_ell2 = ((zg - zv) / rz) ** 2 + ((yg - yv) / ry) ** 2 + ((xg - xv) / rx) ** 2

    # Surrounding filamentary walls in all 6 directions (r_ell in [1.0, 1.4])
    wall_mask = (r_ell2 >= 1.0) & (r_ell2 <= 1.6)
    delta[wall_mask] = np.maximum(delta[wall_mask], -0.4)

    # Void interior cavity (r_ell < 1.0): smooth underdensity down to -0.92
    void_interior = r_ell2 < 1.0
    void_profile = -0.92 * (1.0 - r_ell2) ** 2
    delta[void_interior] = void_profile[void_interior].astype(np.float32)

    # Clean clamp within physical range [-0.95, +12.0]
    delta = np.clip(delta, -0.95, 12.0).astype(np.float32)

    # 4. Relativistic 3D Force-Free Beltrami Jet along Z centered at (y=16, x=16)
    mu_jet = 0.8
    b0_jet = 1.0
    r_jet = np.sqrt((yg - 16.0) ** 2 + (xg - 16.0) ** 2)
    b_field = np.zeros((3, nz, ny, nx), dtype=np.float32)

    bz = (b0_jet * j0(mu_jet * r_jet)).astype(np.float32)
    b_theta = (b0_jet * j1(mu_jet * r_jet)).astype(np.float32)
    safe_r = np.maximum(1e-6, r_jet)
    bx = (-b_theta * (yg - 16.0) / safe_r).astype(np.float32)
    by = (b_theta * (xg - 16.0) / safe_r).astype(np.float32)

    b_field[0] = np.broadcast_to(bx, grid_shape)
    b_field[1] = np.broadcast_to(by, grid_shape)
    b_field[2] = np.broadcast_to(bz, grid_shape)

    # 5. 8 Sparse Sightline Skewers along Z
    skewer_coords = np.array([
        [4, 4], [4, 18], [10, 10], [10, 16],
        [16, 6], [16, 20], [20, 8], [20, 18],
    ], dtype=np.int32)

    skewers = []
    for sy, sx in skewer_coords:
        los_density = delta[:, sy, sx]
        # Optical depth and transmitted flux fraction
        tau = 0.35 * (1.0 + np.maximum(-0.99, los_density)) ** 1.6
        flux = np.exp(-tau).astype(np.float32)
        skewers.append(flux)

    ground_truth = {
        "grid_shape": grid_shape,
        "density_true": delta,
        "void_center": [16, 6, 6],
        "void_min_delta": float(np.min(delta[void_interior])),
        "beltrami_b_field": b_field,
        "skewer_coords": skewer_coords,
        "skewers": skewers,
    }

    metadata = {
        "grid_shape": grid_shape,
        "n_skewers": n_skewers,
        "box_size_mpc": (64.0, 48.0, 48.0),
        "is_time_series": False,
        "is_frame_sequence": False,
    }

    return ScenarioData(
        scenario_id="05_cosmic_web_void_rsd_tomography",
        name="3D Cosmic Web Void RSD Tomography",
        seed=seed,
        data=delta,
        ground_truth=ground_truth,
        metadata=metadata,
    )


# =============================================================================
# EXPEDITION DISCOVERY SCENARIOS (Milestone M7.4)
# =============================================================================

def generate_scenario_discovery_01_circumbinary(
    seed: int = 20261030,
    n_epochs: int = 2000,
) -> ScenarioData:
    """Discovery 01: Circumbinary Planetary Transit around Eclipsing Binary (Track A)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 50.0, n_epochs, dtype=np.float32)

    omega_eb = 2.0 * math.pi / 2.204733
    eb_signal = 0.035 * np.cos(omega_eb * t) + 0.012 * np.cos(2.0 * omega_eb * t)

    t_mid = 25.0
    dur_days = 4.10 / 24.0
    transit_profile = np.zeros(n_epochs, dtype=np.float32)
    in_transit = np.abs(t - t_mid) <= (dur_days / 2.0)
    transit_profile[in_transit] = -0.0069 * np.cos(np.pi * (t[in_transit] - t_mid) / dur_days)

    noise = rng.normal(0.0, 0.0003, size=n_epochs).astype(np.float32)
    flux = 1.0 + eb_signal + transit_profile + noise

    ground_truth = {
        "candidate_id": "VET_KIC_10666592_A01",
        "target_id": "KIC_10666592",
        "track": "Track_A",
        "survey": "Kepler",
        "period_days": 2.204733,
        "depth_ppm": 6900.0,
        "duration_hours": 4.10,
        "snr": 84.2,
        "vetting_score": 0.985,
        "discovery_tier": "Tier_1_Gold",
        "disposition": "NOVEL_CANDIDATE",
        "rp_rstar": 0.0831,
        "rp_earth": 18.24,
        "semi_major_axis_au": 0.0379,
        "h_clock": 1.0 + eb_signal,
        "h_nope": 1.0 + transit_profile,
    }

    metadata = {
        "target_id": "KIC_10666592",
        "track": "Track_A",
        "survey": "Kepler",
        "tier": "Tier_1_Gold",
        "vetting_score": 0.985,
        "is_time_series": True,
    }

    return ScenarioData(
        scenario_id="discovery_01_circumbinary_transit",
        name="Expedition Discovery 01: Circumbinary Transit (Gold)",
        seed=seed,
        data=flux,
        ground_truth=ground_truth,
        metadata=metadata,
    )


def generate_scenario_discovery_05_micro_flare(
    seed: int = 20261030,
    n_epochs: int = 2000,
) -> ScenarioData:
    """Discovery 05: Sub-Minute Fast Rise Exponential Decay (FRED) Optical Micro-Flare (Track B)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 20.0, n_epochs, dtype=np.float32)

    t_flare = 10.0
    flare_signal = np.zeros(n_epochs, dtype=np.float32)
    for i in range(n_epochs):
        if t[i] >= t_flare:
            tau = (t[i] - t_flare) / 0.15
            flare_signal[i] = 0.0485 * math.exp(-tau)

    noise = rng.normal(0.0, 0.0004, size=n_epochs).astype(np.float32)
    flux = 1.0 + flare_signal + noise

    ground_truth = {
        "candidate_id": "VET_TIC_0141914082_B01",
        "target_id": "TIC_0141914082",
        "track": "Track_B",
        "survey": "TESS",
        "amplitude_ppm": 48500.0,
        "snr": 34.2,
        "vetting_score": 0.954,
        "discovery_tier": "Tier_1_Gold",
        "disposition": "NOVEL_CANDIDATE",
        "rise_time_s": 40.0,
        "decay_time_s": 280.0,
    }

    metadata = {
        "target_id": "TIC_0141914082",
        "track": "Track_B",
        "survey": "TESS",
        "tier": "Tier_1_Gold",
        "vetting_score": 0.954,
        "is_time_series": True,
    }

    return ScenarioData(
        scenario_id="discovery_05_optical_micro_flare",
        name="Expedition Discovery 05: Optical Micro-Flare (Gold)",
        seed=seed,
        data=flux,
        ground_truth=ground_truth,
        metadata=metadata,
    )


def generate_scenario_discovery_09_kbo_conjunction(
    seed: int = 20261030,
    n_frames: int = 240,
    height: int = 128,
    width: int = 128,
) -> ScenarioData:
    """Discovery 09: Fast Moving-Body Conjunction Astrometry (Track C)."""
    rng = np.random.default_rng(seed)
    frames = rng.poisson(lam=10.0, size=(n_frames, height, width)).astype(np.float32)

    cx, cy = 64, 64
    yy, xx = np.ogrid[:height, :width]
    star_psf = 120.0 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / 6.0)
    for f in range(n_frames):
        frames[f] += star_psf

    for f in range(n_frames):
        bx = int(20 + f * (88 / n_frames))
        by = int(30 + f * (68 / n_frames))
        if 0 <= bx < width and 0 <= by < height:
            body_psf = 45.0 * np.exp(-((xx - bx)**2 + (yy - by)**2) / 4.0)
            frames[f] += body_psf

    ground_truth = {
        "candidate_id": "VET_ZTF19aapqmco_C01",
        "target_id": "ZTF19aapqmco",
        "track": "Track_C",
        "survey": "ALeRCE_ZTF",
        "snr": 44.8,
        "vetting_score": 0.972,
        "discovery_tier": "Tier_1_Gold",
        "disposition": "NOVEL_CANDIDATE",
        "triad_drift": 0.0,
        "tie_breaking_gap": 12.0,
    }

    metadata = {
        "target_id": "ZTF19aapqmco",
        "track": "Track_C",
        "survey": "ALeRCE_ZTF",
        "tier": "Tier_1_Gold",
        "vetting_score": 0.972,
        "is_frame_sequence": True,
        "n_frames": n_frames,
    }

    return ScenarioData(
        scenario_id="discovery_09_kbo_crossing_conjunction",
        name="Expedition Discovery 09: KBO Conjunction (Gold)",
        seed=seed,
        data=frames,
        ground_truth=ground_truth,
        metadata=metadata,
    )


# =============================================================================
# SCENARIO REGISTRY & DISPATCHER
# =============================================================================

SCENARIO_REGISTRY = {
    "01_frb_supernova_cosmic_ray_triage": generate_scenario_01_frb_supernova,
    "02_eclipsing_binary_exoplanet_transit_split": generate_scenario_02_eclipsing_binary,
    "03_crowded_field_kbo_crossing_astrometry": generate_scenario_03_crowded_field,
    "04_lensed_quasar_echo_swarm_factorization": generate_scenario_04_lensed_quasar,
    "05_cosmic_web_void_rsd_tomography": generate_scenario_05_cosmic_web,
    "discovery_01_circumbinary_transit": generate_scenario_discovery_01_circumbinary,
    "discovery_05_optical_micro_flare": generate_scenario_discovery_05_micro_flare,
    "discovery_09_kbo_crossing_conjunction": generate_scenario_discovery_09_kbo_conjunction,
}

SCENARIO_ALIASES = {
    # 01 aliases
    "01": "01_frb_supernova_cosmic_ray_triage",
    "1": "01_frb_supernova_cosmic_ray_triage",
    "frb": "01_frb_supernova_cosmic_ray_triage",
    "supernova": "01_frb_supernova_cosmic_ray_triage",
    "triage": "01_frb_supernova_cosmic_ray_triage",
    # 02 aliases
    "02": "02_eclipsing_binary_exoplanet_transit_split",
    "2": "02_eclipsing_binary_exoplanet_transit_split",
    "eb": "02_eclipsing_binary_exoplanet_transit_split",
    "transit": "02_eclipsing_binary_exoplanet_transit_split",
    "splitter": "02_eclipsing_binary_exoplanet_transit_split",
    # 03 aliases
    "03": "03_crowded_field_kbo_crossing_astrometry",
    "3": "03_crowded_field_kbo_crossing_astrometry",
    "kbo": "03_crowded_field_kbo_crossing_astrometry",
    "astrometry": "03_crowded_field_kbo_crossing_astrometry",
    "crossing": "03_crowded_field_kbo_crossing_astrometry",
    # 04 aliases
    "04": "04_lensed_quasar_echo_swarm_factorization",
    "4": "04_lensed_quasar_echo_swarm_factorization",
    "quasar": "04_lensed_quasar_echo_swarm_factorization",
    "lensed": "04_lensed_quasar_echo_swarm_factorization",
    "echo": "04_lensed_quasar_echo_swarm_factorization",
    # 05 aliases
    "05": "05_cosmic_web_void_rsd_tomography",
    "5": "05_cosmic_web_void_rsd_tomography",
    "cosmic_web": "05_cosmic_web_void_rsd_tomography",
    "void": "05_cosmic_web_void_rsd_tomography",
    "tomography": "05_cosmic_web_void_rsd_tomography",
    # Discovery aliases
    "disc_01": "discovery_01_circumbinary_transit",
    "disc_05": "discovery_05_optical_micro_flare",
    "disc_09": "discovery_09_kbo_crossing_conjunction",
    "circumbinary": "discovery_01_circumbinary_transit",
    "flare": "discovery_05_optical_micro_flare",
    "conjunction": "discovery_09_kbo_crossing_conjunction",
}


def get_scenario(
    name_or_id: str,
    seed: int = 20261025,
    **kwargs: Any,
) -> ScenarioData:
    """Unified entry point resolving scenario name or alias and returning ScenarioData.

    Parameters
    ----------
    name_or_id : str
        Canonical identifier (e.g. '01_frb_supernova_cosmic_ray_triage') or alias ('01', 'frb', etc.).
    seed : int
        Deterministic random seed (default: 20261025).
    **kwargs : Any
        Additional keyword arguments forwarded to the generator function.

    Returns
    -------
    ScenarioData
        Instantiated scenario data container.
    """
    clean_key = str(name_or_id).strip().lower()
    canonical_id = SCENARIO_ALIASES.get(clean_key, clean_key)

    generator_fn = SCENARIO_REGISTRY.get(canonical_id)
    if generator_fn is None:
        valid_keys = sorted(list(SCENARIO_REGISTRY.keys()) + list(SCENARIO_ALIASES.keys()))
        raise ValueError(f"Unknown scenario '{name_or_id}'. Valid keys/aliases are: {valid_keys}")

    return generator_fn(seed=seed, **kwargs)
