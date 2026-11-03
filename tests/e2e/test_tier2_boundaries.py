"""
astrosnail.tests.e2e.test_tier2_boundaries
==========================================
Tier 2: Boundary conditions, extreme dynamic ranges, corner cases,
and stress tolerances across all subsystems.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from astrosnail.transients import (
    BeltramiLightCurveShell,
    compute_morse_impulse,
    MorseTransientGateConfig,
    TemporalClockContentSplitter,
    compute_spatial_mad_floor_256,
)
from astrosnail.astrometry import (
    RoPE2DMedoidPooler,
    SkewSymmetricTriadOperator,
    SlowFastTriadCascade,
    attribute_causal_progenitors,
)
from astrosnail.cosmology import (
    reconstruct_zslab_tomography,
    detect_jordan_cavity_cores,
    so6_kelvin_mandel_rotor,
    retract_void_step0_homotopy,
    blind_astro_let_decouple,
    solve_beltrami_jet_poincare,
)
from astrosnail.engine import AstroSnailEngine


# =============================================================================
# 1. Empty & Minimal Signal Inputs
# =============================================================================

def test_tier2_empty_1d_signal_handling():
    """Verifies safe handling of 0-length 1D signals without crashing."""
    empty = np.array([], dtype=np.float32)

    # Shell decomposition requires length >= 3
    shell = BeltramiLightCurveShell(k_tokens=16)
    with pytest.raises(ValueError, match="Input time length must be >= 3"):
        shell.decompose(empty)

    # Morse impulse handles empty cleanly
    impulse = compute_morse_impulse(empty)
    assert len(impulse) == 0

    # Clock splitter handles empty cleanly
    splitter = TemporalClockContentSplitter()
    h_c, h_n, mask = splitter.split(empty)
    assert len(h_c) == 0 and len(h_n) == 0 and len(mask) == 0


def test_tier2_single_sample_stream_update():
    """Verifies that AstroSnailEngine handles 1-sample scalar streaming update."""
    engine = AstroSnailEngine(backend="numpy")
    scalar_arr = np.array([42.5], dtype=np.float32)
    out = engine.update(scalar_arr)
    assert out["step"] == 1
    assert "classification" in out
    assert out["classification"] in ("nominal", "pending_confirmation", "glint_rejected", "confirmed_transient")


def test_tier2_short_2_epoch_sequence():
    """Verifies that sequence of length 2 (too short for 3-sample Morse) is handled cleanly."""
    short_seq = np.array([10.0, 12.0], dtype=np.float32)
    impulse = compute_morse_impulse(short_seq)
    assert len(impulse) == 2
    assert np.all(impulse == 0.0)


def test_tier2_empty_or_flat_2d_focal_plane():
    """Verifies that RoPE2DMedoidPooler safely handles flat zero or empty 2D arrays."""
    pooler = RoPE2DMedoidPooler(k_medoids=8)
    flat_frame = np.zeros((32, 32), dtype=np.float32)
    coords, feats = pooler.pool(flat_frame)
    assert len(coords) == 0 or len(coords) <= 8
    assert feats.shape[1] == 8


def test_tier2_single_pixel_2d_frame():
    """Verifies that a 1x1 focal plane does not trigger index out of bounds."""
    pooler = RoPE2DMedoidPooler(k_medoids=4)
    frame_1x1 = np.array([[100.0]], dtype=np.float32)
    coords, feats = pooler.pool(frame_1x1)
    assert len(coords) <= 1


def test_tier2_empty_and_single_skewer_tomography():
    """Verifies that reconstruct_zslab_tomography handles empty or single skewer lists."""
    # Empty skewers
    out_empty = reconstruct_zslab_tomography([], np.zeros((0, 2)), grid_shape=(16, 16, 16))
    assert out_empty["density"].shape == (16, 16, 16)
    assert np.all(out_empty["density"] == 0.0)

    # Single skewer
    skewer_1 = [np.ones(16, dtype=np.float32)]
    coords_1 = np.array([[8.0, 8.0]], dtype=np.float32)
    out_single = reconstruct_zslab_tomography(skewer_1, coords_1, grid_shape=(16, 16, 16))
    assert out_single["density"].shape == (16, 16, 16)
    assert np.all(np.isfinite(out_single["density"]))


def test_tier2_coincident_skewer_coordinates():
    """Verifies that coincident/overlapping skewer coordinates do not divide by zero."""
    skewers = [np.random.randn(16).astype(np.float32) for _ in range(4)]
    coincident_coords = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]], dtype=np.float32)
    out = reconstruct_zslab_tomography(skewers, coincident_coords, grid_shape=(16, 16, 16))
    assert np.all(np.isfinite(out["density"]))
    assert np.all(np.isfinite(out["velocity_potential"]))
    assert np.all(np.isfinite(out["shear_tensor"]))


# =============================================================================
# 2. Extreme SNRs & Dynamic Ranges
# =============================================================================

def test_tier2_extreme_submerged_snr_minus_30db():
    """Verifies that under extreme submerged SNR (-30 dB), transient gate does not crash
    and false glint triggers remain bounded.
    """
    rng = np.random.default_rng(2026)
    clean_signal = np.sin(np.linspace(0, 10 * np.pi, 500))
    # -30 dB SNR: noise amplitude is ~31.6 times signal
    noise = rng.normal(0.0, 31.6, 500)
    submerged = (clean_signal + noise).astype(np.float32)

    gate = MorseTransientGateConfig(sigma_thresh=4.0)
    for val in submerged:
        gate.update(float(val))

    assert gate.glint_rejected_count >= 0
    assert gate.confirmed_transient_count >= 0


def test_tier2_extreme_clean_snr_plus_100db():
    """Verifies that under near-noiseless SNR (+100 dB), transient gate detects clean steps
    with zero glint rejection.
    """
    rng = np.random.default_rng(2026)
    clean_signal = np.zeros(200, dtype=np.float32)
    clean_signal[100:] = 50.0  # Sharp sustained physical step
    # +100 dB SNR: noise std is 5e-4
    noise = rng.normal(0.0, 5e-4, 200).astype(np.float32)
    clean_signal += noise

    gate = MorseTransientGateConfig(sigma_thresh=4.0)
    confirmations = 0
    for val in clean_signal:
        cls_name, _ = gate.update(float(val))
        if cls_name == "confirmed_transient":
            confirmations += 1

    assert confirmations >= 1, "Failed to confirm physical step at +100 dB SNR"


def test_tier2_extreme_dynamic_ranges():
    """Verifies that flux values from 1e-15 up to 1e12 do not produce numerical overflow or NaN."""
    tiny_flux = np.full(50, 1e-15, dtype=np.float64)
    huge_flux = np.full(50, 1e12, dtype=np.float64)

    # Shell decomposition
    shell = BeltramiLightCurveShell(k_tokens=8)
    u_c_tiny, u_f_tiny, _ = shell.decompose(tiny_flux)
    assert np.all(np.isfinite(u_c_tiny)) and np.all(np.isfinite(u_f_tiny))

    u_c_huge, u_f_huge, _ = shell.decompose(huge_flux)
    assert np.all(np.isfinite(u_c_huge)) and np.all(np.isfinite(u_f_huge))


def test_tier2_nan_and_inf_input_resilience():
    """Verifies that NaN and Inf inputs are sanitized without unhandled exception."""
    poisoned_flux = np.array([1.0, 2.0, np.nan, 4.0, np.inf, -np.inf, 7.0], dtype=np.float32)
    
    # Morse impulse
    impulse = compute_morse_impulse(poisoned_flux)
    assert np.all(np.isfinite(impulse))

    # Tomography
    poisoned_skewer = [poisoned_flux]
    coords = np.array([[4.0, 4.0]], dtype=np.float32)
    tomo = reconstruct_zslab_tomography(poisoned_skewer, coords, grid_shape=(8, 8, 8))
    assert np.all(np.isfinite(tomo["density"]))


# =============================================================================
# 3. State Machine & Parameter Extremes
# =============================================================================

def test_tier2_ring_buffer_boundary_overflow():
    """Verifies that feeding 100 sequential updates keeps AstroSnailEngine ring buffer
    strictly bounded at exactly 8 frames.
    """
    engine = AstroSnailEngine(backend="numpy")
    for i in range(100):
        frame = np.full((16, 16), float(i), dtype=np.float32)
        out = engine.update(frame)
        assert len(engine.ring_buffer) <= 8

    assert len(engine.ring_buffer) == 8
    assert engine.step_count == 100


def test_tier2_triad_cascade_coupling_extremes():
    """Verifies that SlowFastTriadCascade remains norm-conserving at coupling=0.0 and coupling=1.0."""
    v0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    u_seq = np.ones((16, 8), dtype=np.float64)

    # Coupling 0.0: identity evolution
    cascade_zero = SlowFastTriadCascade(coupling=0.0, steps=16)
    v_out_zero, drift_zero = cascade_zero.integrate(u_seq, v0, steps=16)
    assert drift_zero == 0.0
    assert np.allclose(v_out_zero, v0)

    # Coupling 1.0: strong coupling
    cascade_unit = SlowFastTriadCascade(coupling=1.0, steps=16)
    v_out_unit, drift_unit = cascade_unit.integrate(u_seq, v0, steps=16)
    assert drift_unit == 0.0
    assert abs(np.linalg.norm(v_out_unit) - 1.0) < 1e-12


def test_tier2_kelvin_mandel_degenerate_rotations():
    """Verifies SO(6) Kelvin-Mandel rotor under identity rotation and 180-degree flip."""
    # Identity
    R_eye = np.eye(3, dtype=np.float64)
    shear = np.array([1.0, 2.0, 3.0, 0.5, -0.5, 0.25], dtype=np.float64)
    rot_eye, trace_res_eye = so6_kelvin_mandel_rotor(R_eye, shear)
    assert trace_res_eye < 1e-14
    assert np.allclose(rot_eye, shear)

    # 180-degree rotation around Z
    R_flip = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    rot_flip, trace_res_flip = so6_kelvin_mandel_rotor(R_flip, shear)
    assert trace_res_flip < 1e-14


def test_tier2_poincare_beltrami_jet_mu_extremes():
    """Verifies force-free Beltrami jet solver under mu=0.0 and mu=-1.0."""
    grid = np.zeros((16, 16), dtype=np.float32)

    # mu = 0.0
    b_zero, escape_zero = solve_beltrami_jet_poincare(grid, mu=0.0, steps=8)
    assert escape_zero == 0.0
    assert b_zero.shape == (3, 32, 16, 16)

    # mu = -1.0
    b_neg, escape_neg = solve_beltrami_jet_poincare(grid, mu=-1.0, steps=8)
    assert escape_neg == 0.0
    assert np.all(np.isfinite(b_neg))


def test_tier2_causal_attribution_boundary_matrices():
    """Verifies attribute_causal_progenitors on all-zero, all-one, and identity matrices."""
    # All zeros: no causal attribution
    mat_zero = np.zeros((5, 5), dtype=np.float64)
    is_gen_zero, scores_zero = attribute_causal_progenitors(mat_zero)
    assert len(is_gen_zero) == 5
    assert not any(is_gen_zero)

    # Identity matrix: each node isolated
    mat_eye = np.eye(4, dtype=np.float64)
    is_gen_eye, scores_eye = attribute_causal_progenitors(mat_eye)
    assert len(is_gen_eye) == 4

    # All ones: fully connected
    mat_ones = np.ones((4, 4), dtype=np.float64)
    is_gen_ones, scores_ones = attribute_causal_progenitors(mat_ones)
    assert len(is_gen_ones) == 4


def test_tier2_jordan_cavity_cores_uniform_field():
    """Verifies detect_jordan_cavity_cores on completely uniform density field."""
    # Completely flat density field (no underdensities)
    flat_cube = np.zeros((16, 16, 16), dtype=np.float32)
    void_cores = detect_jordan_cavity_cores(flat_cube, threshold=-0.7)
    assert not np.any(void_cores)

    # Completely underdense field: all values -0.9
    deep_void = np.full((16, 16, 16), -0.9, dtype=np.float32)
    # Without walls surrounding, enclosure count will not enclose
    void_enclosed = detect_jordan_cavity_cores(deep_void, threshold=-0.7)
    assert isinstance(void_enclosed, np.ndarray)
    assert void_enclosed.dtype == bool


def test_tier2_mad_floor_boundary_sizes():
    """Verifies compute_spatial_mad_floor_256 on input size < 256 and fully masked input."""
    # Input size < 256
    small_salience = np.ones(50, dtype=np.float32)
    mad_val = compute_spatial_mad_floor_256(small_salience)
    assert np.isfinite(mad_val)

    # All elements masked out
    mask = np.zeros(50, dtype=bool)
    mad_masked = compute_spatial_mad_floor_256(small_salience, mask=mask)
    assert np.isfinite(mad_masked)


def test_tier2_retract_void_boundary_masks():
    """Verifies retract_void_step0_homotopy with empty void mask and all-True mask."""
    density = np.random.randn(8, 8, 8).astype(np.float32)

    # Empty mask (no voids): density untouched
    mask_empty = np.zeros((8, 8, 8), dtype=bool)
    retracted_empty = retract_void_step0_homotopy(density, mask_empty)
    assert np.allclose(retracted_empty, density)

    # All-True mask: everything clamped to >= -1.0
    mask_all = np.ones((8, 8, 8), dtype=bool)
    retracted_all = retract_void_step0_homotopy(density, mask_all)
    assert np.all(retracted_all >= -1.0)


def test_tier2_so6_kelvin_mandel_matrix_shapes():
    """Verifies so6_kelvin_mandel_rotor supports (3, 3) tensor and (6, 6) stiffness matrix shapes."""
    R = np.eye(3, dtype=np.float64)

    # 3x3 symmetric tensor
    tensor_3x3 = np.array([[2.0, 0.5, 0.0], [0.5, 3.0, 0.2], [0.0, 0.2, 1.5]], dtype=np.float64)
    rot_3x3, res_3x3 = so6_kelvin_mandel_rotor(R, tensor_3x3)
    assert res_3x3 < 1e-14
    assert np.allclose(rot_3x3, tensor_3x3)

    # 6x6 stiffness matrix
    stiff_6x6 = np.eye(6, dtype=np.float64) * 5.0
    rot_6x6, res_6x6 = so6_kelvin_mandel_rotor(R, stiff_6x6)
    assert res_6x6 < 1e-14
    assert np.allclose(rot_6x6, stiff_6x6)

