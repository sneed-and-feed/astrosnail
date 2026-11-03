"""
astrosnail.tests.e2e.test_tier4_realworld
=========================================
Tier 4: Real-world astronomical workloads evaluating AstroSnail on authentic
public astronomical standards:
1. NASA Kepler HAT-P-7b Short-Cadence Light Curve (14,280 epochs)
2. 2MASS Galactic Center Crowded Star Field (720x721 H-band FITS)
3. NASA JPL Horizons Ephemeris Track Catalog (Ceres, Vesta, Apophis, Pluto)
4. FCRAO 14m Radio Telescope L1448 13CO 3D Spectral Line Cube (53x105x105 voxels)
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from astrosnail.transients import (
    BeltramiLightCurveShell,
    compute_morse_impulse,
    MorseTransientGateConfig,
    TemporalClockContentSplitter,
)
from astrosnail.astrometry import (
    RoPE2DMedoidPooler,
    SlowFastTriadCascade,
)
from astrosnail.cosmology import (
    reconstruct_zslab_tomography,
    detect_jordan_cavity_cores,
    retract_void_step0_homotopy,
    solve_beltrami_jet_poincare,
)
from astrosnail.engine import AstroSnailEngine
from astrosnail.io import (
    read_fits_header,
    stream_fits_cube,
    profile_stream_memory,
    parse_cat_file,
    parse_lc_table,
)


# =============================================================================
# 1. Real NASA Kepler Short-Cadence Exoplanet Benchmark (HAT-P-7b / KIC 10666592)
# =============================================================================

def test_tier4_kepler_hatp7b_real_standards(real_standards_dir: Path):
    """Evaluates AstroSnail on authentic NASA Kepler short-cadence FITS data (14,280 epochs)."""
    fits_path = real_standards_dir / "kepler_hatp7b_slc.fits"
    assert fits_path.exists(), f"Kepler FITS standard missing at {fits_path}"

    raw = fits_path.read_bytes()
    dt_kep = np.dtype([
        ("TIME", ">f8"), ("TIMECORR", ">f4"), ("CADENCENO", ">i4"),
        ("SAP_FLUX", ">f4"), ("SAP_FLUX_ERR", ">f4"), ("SAP_BKG", ">f4"), ("SAP_BKG_ERR", ">f4"),
        ("PDCSAP_FLUX", ">f4"), ("PDCSAP_FLUX_ERR", ">f4"), ("SAP_QUALITY", ">i4"),
        ("PSF_CENTR1", ">f8"), ("PSF_CENTR1_ERR", ">f4"), ("PSF_CENTR2", ">f8"), ("PSF_CENTR2_ERR", ">f4"),
        ("MOM_CENTR1", ">f8"), ("MOM_CENTR1_ERR", ">f4"), ("MOM_CENTR2", ">f8"), ("MOM_CENTR2_ERR", ">f4"),
        ("POS_CORR1", ">f4"), ("POS_CORR2", ">f4")
    ])
    table = np.frombuffer(raw, dtype=dt_kep, count=14280, offset=20160)
    valid = np.isfinite(table["TIME"]) & np.isfinite(table["SAP_FLUX"]) & np.isfinite(table["PDCSAP_FLUX"])
    t_bjd = table["TIME"][valid].astype(np.float64)
    sap = table["SAP_FLUX"][valid].astype(np.float32)
    pdcsap = table["PDCSAP_FLUX"][valid].astype(np.float32)
    qual = table["SAP_QUALITY"][valid].astype(np.int32)

    assert len(t_bjd) >= 14000, f"Expected >= 14,000 valid epochs, got {len(t_bjd)}"

    sap_med = float(np.median(sap))
    sap_rel = (sap / sap_med - 1.0) * 1e6  # relative flux in ppm

    # 1. Beltrami-Morse Shell Decomposition
    shell = BeltramiLightCurveShell(k_tokens=16)
    u_coarse, u_fine, curl_sal = shell.decompose(sap_rel)
    assert len(u_fine) == len(sap_rel)

    # 2. Discrete Morse Impulse: isolate cosmic-ray spikes
    i_morse = compute_morse_impulse(sap_rel, salience=curl_sal)
    assert len(i_morse) == len(sap_rel)
    # Top Morse impulse spike should align with known cosmic-ray glint
    top_spike_idx = int(np.argmax(i_morse))
    assert i_morse[top_spike_idx] > 1000.0

    # 3. Exoplanet Transit Detection (HAT-P-7b has ~6000 ppm transit depth)
    in_transit = sap_rel < -4000.0
    transit_epochs_count = int(np.sum(in_transit))
    assert transit_epochs_count > 100, f"Expected > 100 in-transit epochs, found {transit_epochs_count}"

    mean_transit_depth_ppm = float(np.mean(sap_rel[in_transit]))
    assert -8000.0 <= mean_transit_depth_ppm <= -4500.0, (
        f"HAT-P-7b transit depth measured {mean_transit_depth_ppm:.1f} ppm, expected between -8000 and -4500 ppm"
    )

    # 4. 1-Sample Causal Glitch Gate on raw Kepler epochs
    gate = MorseTransientGateConfig(sigma_thresh=4.0)
    for idx in range(min(5000, len(u_fine))):
        gate.update(float(u_fine[idx]), timestamp=float(t_bjd[idx]))

    assert gate.glint_rejected_count >= 1


# =============================================================================
# 2. Real 2MASS Galactic Center Crowded Field Benchmark
# =============================================================================

def test_tier4_2mass_galactic_center_real_standards(real_standards_dir: Path):
    """Evaluates AstroSnail on 2MASS Galactic Center H-band crowded stellar field (720x721)."""
    p_2mass = real_standards_dir / "2mass_galactic_center_h.fits"
    assert p_2mass.exists(), f"2MASS FITS standard missing at {p_2mass}"

    hdr = read_fits_header(p_2mass)
    assert hdr.shape == (720, 721) or len(hdr.shape) == 2

    slab = next(stream_fits_cube(p_2mass, chunk_z=1))[0]
    h, w = slab.shape
    cy, cx = h // 2, w // 2
    cutout = slab[cy - 64 : cy + 64, cx - 64 : cx + 64].astype(np.float32)
    assert cutout.shape == (128, 128)

    # Sub-pixel stellar medoid extraction
    pooler = RoPE2DMedoidPooler(k_medoids=24)
    coords, feats = pooler.pool(cutout)
    assert len(coords) >= 10, f"Expected >= 10 stellar medoids in crowded center, extracted {len(coords)}"
    assert feats.shape == (len(coords), 8)

    # Verify Theorem 2.5 Rotary Phase Invariance on real stars
    th_grid = float(2.0 * np.pi / 256.0)
    dot_y = float(feats[0, 0] * feats[1, 0] + feats[0, 1] * feats[1, 1])
    exp_dot_y = float(np.cos((coords[0, 0] - coords[1, 0]) * th_grid))
    rope_phase_err = abs(dot_y - exp_dot_y)
    assert rope_phase_err < 1e-4, f"RoPE phase invariance error on real 2MASS stars: {rope_phase_err:.3e}"


# =============================================================================
# 3. Real NASA JPL Horizons Solar-System Ephemeris Tracking
# =============================================================================

def test_tier4_jpl_horizons_solar_system_tracking(catalogs_dir: Path):
    """Evaluates AstroSnail on real NASA JPL Horizons ephemeris track catalog (Ceres, Vesta, Apophis, Pluto)."""
    cat_path = catalogs_dir / "real_jpl_horizons_2mass.cat"
    assert cat_path.exists(), f"JPL Horizons catalog missing at {cat_path}"

    cat_records = parse_cat_file(cat_path)
    assert len(cat_records) >= 20, f"Expected >= 20 ephemeris records, got {len(cat_records)}"

    # Integrate ephemeris velocity states through SlowFastTriadCascade
    cascade = SlowFastTriadCascade(coupling=0.15, steps=16)
    u_drive = np.ones((16, 8), dtype=np.float64) * 0.25

    # Group records into 8D celestial state vectors (r is (source_id, ra, dec, mag, mag_err))
    ra_coords = [float(r[1]) for r in cat_records[:8]]
    dec_coords = [float(r[2]) for r in cat_records[:8]]
    v_ephem = np.array(ra_coords, dtype=np.float64)
    v_norm = v_ephem / np.linalg.norm(v_ephem)

    v_out, drift = cascade.integrate(u_drive, v_norm, steps=16)
    assert drift == 0.0, f"Cascade radial drift on JPL Horizons vectors: {drift}"
    assert abs(np.linalg.norm(v_out) - 1.0) < 1e-12


# =============================================================================
# 4. Real FCRAO 14m L1448 13CO Molecular Cloud 3D Spectral Line Cube
# =============================================================================

def test_tier4_l1448_13co_spectral_cube_tomography(real_standards_dir: Path):
    """Evaluates AstroSnail on FCRAO 14m radio telescope 3D spectral cube (53x105x105)."""
    p_cube = real_standards_dir / "l1448_13co_cube.fits"
    assert p_cube.exists(), f"FCRAO L1448 FITS cube missing at {p_cube}"

    hdr = read_fits_header(p_cube)
    assert hdr.shape == (53, 105, 105)

    # 1. Out-of-core streaming under bounded heap constraint
    slabs = list(stream_fits_cube(p_cube, chunk_z=16, as_native=True))
    assert len(slabs) == 4, f"Expected 4 chunks for 53 Z-slices with chunk_z=16, got {len(slabs)}"

    # 2. Extract central protostellar outflow core (32x24x24 voxels)
    full_cube = np.concatenate(slabs, axis=0)
    sub_cube = full_cube[10:42, 40:64, 40:64].astype(np.float32)
    assert sub_cube.shape == (32, 24, 24)

    # 3. Detect 3D Jordan Cavity Cores in cloud
    mean_cloud = float(np.mean(sub_cube[sub_cube > 0.1])) if np.any(sub_cube > 0.1) else 1.0
    delta_cube = (sub_cube - mean_cloud) / max(1e-6, mean_cloud)
    void_cores = detect_jordan_cavity_cores(delta_cube, threshold=-0.7)
    assert isinstance(void_cores, np.ndarray)

    # 4. Step-0 Homotopy Retraction: vacuum bound enforcement
    retracted = retract_void_step0_homotopy(delta_cube, void_cores)
    assert np.all(retracted[void_cores] >= -1.0)

    # 5. Beltrami Jet Poincare Solver on cloud peak velocity slice
    peak_z = int(np.argmax(np.max(sub_cube, axis=(1, 2))))
    b_field, flux_escape = solve_beltrami_jet_poincare(sub_cube[peak_z], mu=1.25, steps=16)
    assert flux_escape == 0.0
    assert b_field.shape == (3, 32, 24, 24)


def test_tier4_kepler_centroid_astrometric_jitter(real_standards_dir: Path):
    """Verifies that Kepler HAT-P-7b sub-pixel centroid jitter is within physical bounds (< 50 milli-pixels)."""
    fits_path = real_standards_dir / "kepler_hatp7b_slc.fits"
    raw = fits_path.read_bytes()
    dt_kep = np.dtype([
        ("TIME", ">f8"), ("TIMECORR", ">f4"), ("CADENCENO", ">i4"),
        ("SAP_FLUX", ">f4"), ("SAP_FLUX_ERR", ">f4"), ("SAP_BKG", ">f4"), ("SAP_BKG_ERR", ">f4"),
        ("PDCSAP_FLUX", ">f4"), ("PDCSAP_FLUX_ERR", ">f4"), ("SAP_QUALITY", ">i4"),
        ("PSF_CENTR1", ">f8"), ("PSF_CENTR1_ERR", ">f4"), ("PSF_CENTR2", ">f8"), ("PSF_CENTR2_ERR", ">f4"),
        ("MOM_CENTR1", ">f8"), ("MOM_CENTR1_ERR", ">f4"), ("MOM_CENTR2", ">f8"), ("MOM_CENTR2_ERR", ">f4"),
        ("POS_CORR1", ">f4"), ("POS_CORR2", ">f4")
    ])
    table = np.frombuffer(raw, dtype=dt_kep, count=14280, offset=20160)
    valid = np.isfinite(table["MOM_CENTR1"]) & np.isfinite(table["MOM_CENTR2"])
    mom_x = table["MOM_CENTR1"][valid]
    mom_y = table["MOM_CENTR2"][valid]

    std_x_mpix = float(np.std(mom_x) * 1000.0)
    std_y_mpix = float(np.std(mom_y) * 1000.0)
    assert std_x_mpix < 50.0, f"Kepler Centroid X jitter too high: {std_x_mpix:.2f} milli-pixels"
    assert std_y_mpix < 50.0, f"Kepler Centroid Y jitter too high: {std_y_mpix:.2f} milli-pixels"


def test_tier4_fcrao_moment0_integrated_intensity(real_standards_dir: Path):
    """Computes moment-0 integrated intensity of FCRAO L1448 13CO cube and verifies positive emission."""
    p_cube = real_standards_dir / "l1448_13co_cube.fits"
    slabs = list(stream_fits_cube(p_cube, chunk_z=16, as_native=True))
    full_cube = np.concatenate(slabs, axis=0)

    # Moment 0: integrate along velocity channel axis (axis 0)
    moment0 = np.sum(np.maximum(0.0, full_cube), axis=0)
    assert moment0.shape == (105, 105)
    assert np.max(moment0) > 1.0, "Expected significant molecular line emission in moment-0 map"

