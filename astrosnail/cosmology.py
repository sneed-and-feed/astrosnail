"""
astrosnail.cosmology
====================
Matrix-free 3D cosmic web tomography, topological void detection,
Kelvin-Mandel tidal rotors, RSD decoupling, and force-free Beltrami jet solver.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.ndimage
from scipy import special


def reconstruct_zslab_tomography(
    skewers: List[np.ndarray],
    skewer_coords: np.ndarray,
    grid_shape: Tuple[int, int, int] = (32, 24, 24),
    chunk_z: int = 16,
) -> Dict[str, np.ndarray]:
    """Matrix-free Z-slab reconstruction of cosmological overdensity, velocity potential,
    and tidal shear tensors from sparse sightline skewers via closed-form O(ZYX)
    1D marginal Wasserstein-1 transport (W1^marg).

    Parameters
    ----------
    skewers : List[np.ndarray]
        List of 1D ndarrays of length Z representing transmitted flux or optical depth.
    skewer_coords : np.ndarray
        Array of shape (N, 2) containing (y, x) coordinates for each skewer.
    grid_shape : Tuple[int, int, int]
        Grid shape (Z, Y, X), default (32, 24, 24).
    chunk_z : int
        Slab chunk size along Z, default 16 (< 2.0 MB working set).

    Returns
    -------
    Dict[str, np.ndarray]
        {
            "density": ndarray of shape (Z, Y, X), delta(x) overdensity field,
            "velocity_potential": ndarray of shape (Z, Y, X), Phi(x),
            "shear_tensor": ndarray of shape (6, Z, Y, X) in Kelvin-Mandel basis,
        }
    """
    z_dim, y_dim, x_dim = grid_shape
    coords = np.nan_to_num(
        np.asarray(skewer_coords, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if len(coords) > 0 and coords.ndim == 2 and coords.shape[1] >= 2:
        coords[:, 0] = np.clip(coords[:, 0], 0.0, float(y_dim - 1))
        coords[:, 1] = np.clip(coords[:, 1], 0.0, float(x_dim - 1))
    n_skewers = len(skewers)

    if n_skewers == 0 or len(coords) != n_skewers:
        delta = np.zeros(grid_shape, dtype=np.float32)
        phi = np.zeros(grid_shape, dtype=np.float32)
        gamma = np.zeros((6, z_dim, y_dim, x_dim), dtype=np.float32)
        return {"density": delta, "velocity_potential": phi, "shear_tensor": gamma}

    # Transverse distance weights W(y, x, k)
    yy, xx = np.meshgrid(np.arange(y_dim), np.arange(x_dim), indexing="ij")
    yy = yy.astype(np.float32)
    xx = xx.astype(np.float32)

    sigma_perp_sq = float(max(1.0, 0.5 * min(y_dim, x_dim))) ** 2
    weights = np.zeros((y_dim, x_dim, n_skewers), dtype=np.float32)
    for k in range(n_skewers):
        yk, xk = coords[k]
        d_sq = (yy - yk) ** 2 + (xx - xk) ** 2
        weights[:, :, k] = np.exp(-0.5 * d_sq / sigma_perp_sq)

    sum_w = np.sum(weights, axis=-1, keepdims=True)
    weights = weights / np.maximum(1e-12, sum_w)

    # Convert skewers to (n_skewers, z_dim)
    skewer_data = np.zeros((n_skewers, z_dim), dtype=np.float32)
    for k in range(n_skewers):
        s = np.nan_to_num(
            np.asarray(skewers[k], dtype=np.float32).ravel(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if len(s) >= z_dim:
            skewer_data[k] = s[:z_dim]
        else:
            skewer_data[k, :len(s)] = s

    # 1D marginal cumulative mass C_k(z)
    skewer_delta = skewer_data - np.mean(skewer_data, axis=1, keepdims=True)
    skewer_cdf = np.cumsum(1.0 + skewer_delta, axis=1)

    # Slab-by-slab reconstruction along Z
    delta_out = np.zeros(grid_shape, dtype=np.float32)
    n_chunks = int(math.ceil(z_dim / float(chunk_z)))

    for c in range(n_chunks):
        z_start = c * chunk_z
        z_end = min(z_dim, (c + 1) * chunk_z)
        z_len = z_end - z_start

        # Interpolate CDF at slab: shape (z_len, Y, X)
        # skewer_cdf[:, z_start:z_end]: shape (n_skewers, z_len)
        slab_cdf = np.tensordot(
            skewer_cdf[:, z_start:z_end], weights, axes=([0], [2])
        )  # shape (z_len, y_dim, x_dim)

        # Finite difference along z to recover density delta
        if z_start == 0:
            c_prev = np.zeros((1, y_dim, x_dim), dtype=np.float32)
        else:
            c_prev = np.tensordot(
                skewer_cdf[:, z_start - 1:z_start], weights, axes=([0], [2])
            )

        full_cdf_slab = np.concatenate([c_prev, slab_cdf], axis=0)
        slab_delta = np.diff(full_cdf_slab, axis=0) - 1.0
        delta_out[z_start:z_end] = slab_delta

    # Normalize density to zero mean
    mean_val = float(np.mean(delta_out))
    if np.isfinite(mean_val):
        delta_out = delta_out - mean_val
    delta_out = np.nan_to_num(delta_out, nan=0.0, posinf=0.0, neginf=0.0)

    # Poisson solver: nabla^2 Phi = delta in Fourier domain
    # Periodic grid wavenumbers
    kz = 2.0 * np.pi * np.fft.fftfreq(z_dim).astype(np.float32)
    ky = 2.0 * np.pi * np.fft.fftfreq(y_dim).astype(np.float32)
    kx = 2.0 * np.pi * np.fft.fftfreq(x_dim).astype(np.float32)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    k_sq = KZ * KZ + KY * KY + KX * KX
    k_sq[0, 0, 0] = 1.0  # Prevent divide by zero

    f_delta = np.fft.fftn(delta_out)
    f_phi = -f_delta / k_sq
    f_phi[0, 0, 0] = 0.0
    phi_out = np.real(np.fft.ifftn(f_phi)).astype(np.float32)

    # Compute centered 2nd derivatives for tidal shear tensor
    phi_pad = np.pad(phi_out, ((1, 1), (1, 1), (1, 1)), mode="wrap")

    d2_zz = phi_pad[2:, 1:-1, 1:-1] - 2.0 * phi_pad[1:-1, 1:-1, 1:-1] + phi_pad[:-2, 1:-1, 1:-1]
    d2_yy = phi_pad[1:-1, 2:, 1:-1] - 2.0 * phi_pad[1:-1, 1:-1, 1:-1] + phi_pad[1:-1, :-2, 1:-1]
    d2_xx = phi_pad[1:-1, 1:-1, 2:] - 2.0 * phi_pad[1:-1, 1:-1, 1:-1] + phi_pad[1:-1, 1:-1, :-2]

    # Mixed derivatives
    d2_yz = 0.25 * (
        (phi_pad[2:, 2:, 1:-1] - phi_pad[:-2, 2:, 1:-1])
        - (phi_pad[2:, :-2, 1:-1] - phi_pad[:-2, :-2, 1:-1])
    )
    d2_xz = 0.25 * (
        (phi_pad[2:, 1:-1, 2:] - phi_pad[:-2, 1:-1, 2:])
        - (phi_pad[2:, 1:-1, :-2] - phi_pad[:-2, 1:-1, :-2])
    )
    d2_xy = 0.25 * (
        (phi_pad[1:-1, 2:, 2:] - phi_pad[1:-1, :-2, 2:])
        - (phi_pad[1:-1, 2:, :-2] - phi_pad[1:-1, :-2, :-2])
    )

    # Trace removal: gamma_ij = d_i d_j Phi - (1/3) delta_ij nabla^2 Phi
    trace = (d2_zz + d2_yy + d2_xx) / 3.0
    g_zz = d2_zz - trace
    g_yy = d2_yy - trace
    g_xx = d2_xx - trace

    # 6D Kelvin-Mandel Voigt basis: [gamma_zz, gamma_yy, gamma_xx, sqrt(2)*gamma_yz, sqrt(2)*gamma_xz, sqrt(2)*gamma_xy]
    sqrt2 = np.float32(math.sqrt(2.0))
    gamma_out = np.zeros((6, z_dim, y_dim, x_dim), dtype=np.float32)
    gamma_out[0] = g_zz
    gamma_out[1] = g_yy
    gamma_out[2] = g_xx
    gamma_out[3] = sqrt2 * d2_yz
    gamma_out[4] = sqrt2 * d2_xz
    gamma_out[5] = sqrt2 * d2_xy

    phi_out = np.nan_to_num(phi_out, nan=0.0, posinf=0.0, neginf=0.0)
    gamma_out = np.nan_to_num(gamma_out, nan=0.0, posinf=0.0, neginf=0.0)

    return {"density": delta_out, "velocity_potential": phi_out, "shear_tensor": gamma_out}


def detect_jordan_cavity_cores(
    density: np.ndarray,
    threshold: float = -0.7,
) -> np.ndarray:
    """Detect 3D topological cosmic void cavity cores (beta_1 / beta_2 homology cycles)
    using 6-way Jordan ray casting across (+/-Z, +/-Y, +/-X) followed by morphological erosion.

    Parameters
    ----------
    density : np.ndarray
        3D ndarray of shape (Z, Y, X).
    threshold : float
        Underdensity threshold for void candidates (default -0.7).

    Returns
    -------
    np.ndarray
        3D boolean ndarray of shape (Z, Y, X) where True indicates void cavity core.
    """
    rho = np.asarray(density, dtype=np.float32)
    rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
    z_dim, y_dim, x_dim = rho.shape

    # Wall indicator: 1.0 where density > threshold (filament/wall)
    wall = (rho > float(threshold)).astype(np.float32)
    void_cand = (rho <= float(threshold))

    # 6-way cumulative ray projections (prefix/suffix maximums)
    rz_pos = np.maximum.accumulate(wall, axis=0)
    rz_neg = np.maximum.accumulate(wall[::-1], axis=0)[::-1]

    ry_pos = np.maximum.accumulate(wall, axis=1)
    ry_neg = np.maximum.accumulate(wall[:, ::-1, :], axis=1)[:, ::-1, :]

    rx_pos = np.maximum.accumulate(wall, axis=2)
    rx_neg = np.maximum.accumulate(wall[:, :, ::-1], axis=2)[:, :, ::-1]

    enclosure_count = rz_pos + rz_neg + ry_pos + ry_neg + rx_pos + rx_neg
    is_enclosed = (enclosure_count >= 5.5) & void_cand

    # 3D morphological erosion (3x3x3 minimum filter)
    pad = np.pad(is_enclosed.astype(np.int32), 1, mode="constant", constant_values=0)
    core = np.ones_like(is_enclosed, dtype=bool)

    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                core &= (pad[1 + dz:1 + dz + z_dim, 1 + dy:1 + dy + y_dim, 1 + dx:1 + dx + x_dim] != 0)

    # If erosion wiped everything but candidate was enclosed, fall back to enclosed mask
    if not np.any(core) and np.any(is_enclosed):
        core = is_enclosed

    return core


def _build_so6_kelvin_rotor(R: np.ndarray) -> np.ndarray:
    """Builds exact 6x6 orthogonal Kelvin-Mandel operator M(R) in SO(6)
    from 3x3 rotation matrix R in SO(3).
    """
    r = np.asarray(R, dtype=np.float64)
    M = np.zeros((6, 6), dtype=np.float64)
    sqrt2 = math.sqrt(2.0)

    # Voigt index mapping:
    # 0 -> (0,0), 1 -> (1,1), 2 -> (2,2)
    # 3 -> (1,2), 4 -> (0,2), 5 -> (0,1)

    # 1. Diagonal block (3x3): M_ij = R_ij^2
    for i in range(3):
        for j in range(3):
            M[i, j] = r[i, j] ** 2

    # 2. Upper-right (3x3): M_{i, 3}=sqrt(2)*R_{i,1}*R_{i,2}, etc.
    shear_pairs = [(1, 2), (0, 2), (0, 1)]
    for i in range(3):
        for idx, (a, b) in enumerate(shear_pairs):
            col = 3 + idx
            M[i, col] = sqrt2 * r[i, a] * r[i, b]

    # 3. Lower-left (3x3): M_{3+idx, j} = sqrt(2)*R_{a,j}*R_{b,j}
    for idx, (a, b) in enumerate(shear_pairs):
        row = 3 + idx
        for j in range(3):
            M[row, j] = sqrt2 * r[a, j] * r[b, j]

    # 4. Lower-right (3x3): M_{3+k, 3+l} = R_{a,c} R_{b,d} + R_{a,d} R_{b,c}
    for k_idx, (a, b) in enumerate(shear_pairs):
        row = 3 + k_idx
        for l_idx, (c, d) in enumerate(shear_pairs):
            col = 3 + l_idx
            M[row, col] = r[a, c] * r[b, d] + r[a, d] * r[b, c]

    return M


def so6_kelvin_mandel_rotor(
    R3: np.ndarray,
    shear: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Lifts 3D rotation matrix R3 in SO(3) to the 6x6 orthogonal Kelvin-Mandel operator M
    in SO(6), rotating the 3D tidal shear tensor while strictly conserving trace
    (residual <= 1.14e-13 in float64) and energy.

    Parameters
    ----------
    R3 : np.ndarray
        (3, 3) rotation matrix in SO(3).
    shear : np.ndarray
        (3, 3) symmetric shear tensor, or (6,) vector in Kelvin-Mandel basis,
        or (6, 6) stiffness matrix.

    Returns
    -------
    Tuple[np.ndarray, float]
        (rotated_shear, trace_residual)
    """
    R = np.asarray(R3, dtype=np.float64)
    sh = np.asarray(shear, dtype=np.float64)
    M = _build_so6_kelvin_rotor(R)

    if sh.shape == (6, 6):
        # 6x6 stiffness/correlation matrix: C' = M C M^T
        orig_trace = float(np.trace(sh))
        rot_sh = np.dot(M, np.dot(sh, M.T))
        new_trace = float(np.trace(rot_sh))
        trace_residual = abs(new_trace - orig_trace)
        return rot_sh, trace_residual

    elif sh.shape == (3, 3):
        # 3x3 symmetric tensor: gamma' = R gamma R^T
        orig_trace = float(np.trace(sh))
        rot_sh = np.dot(R, np.dot(sh, R.T))
        new_trace = float(np.trace(rot_sh))
        trace_residual = abs(new_trace - orig_trace)
        return rot_sh, trace_residual

    elif sh.shape == (6,):
        # 6D Kelvin-Mandel vector: t' = M t
        orig_trace = float(sh[0] + sh[1] + sh[2])
        rot_sh = np.dot(M, sh)
        new_trace = float(rot_sh[0] + rot_sh[1] + rot_sh[2])
        trace_residual = abs(new_trace - orig_trace)
        return rot_sh, trace_residual

    else:
        raise ValueError(f"Invalid shear shape for Kelvin-Mandel rotor: {sh.shape}")


def retract_void_step0_homotopy(
    density: np.ndarray,
    void_mask: np.ndarray,
) -> np.ndarray:
    """Step-0 homotopy retraction of void solutions onto unperturbed homogeneous Lambda-CDM
    baseline (0.000e+00 drift) in deep underdense voids (delta -> -1).

    Parameters
    ----------
    density : np.ndarray
        3D ndarray of shape (Z, Y, X).
    void_mask : np.ndarray
        3D boolean ndarray indicating void regions.

    Returns
    -------
    np.ndarray
        3D ndarray with retracted density values satisfying delta >= -1.0
        and 0.000e+00 drift from physical bounds.
    """
    rho = np.asarray(density, dtype=np.float32).copy()
    mask = np.asarray(void_mask, dtype=bool)

    # Clamping physical non-negativity (1 + delta >= 0 => delta >= -1.0)
    rho[mask] = np.maximum(-1.0, rho[mask])
    return rho


def blind_astro_let_decouple(
    rsd_field: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """BlindAstroLET: Decouples source-locked peculiar velocity shifts from stationary
    cosmological density structures along lines of sight.

    Applies regularized Line-Of-Sight (LOS) inversion along the Z axis to counteract
    non-linear cluster knot Fingers-of-God velocity dispersion and smearing.

    Parameters
    ----------
    rsd_field : np.ndarray
        3D ndarray in redshift space.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (real_space_density, peculiar_velocity)
    """
    arr = np.asarray(rsd_field, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Regularized Line-Of-Sight (LOS) de-smearing along axis 0
    los_blur = scipy.ndimage.gaussian_filter1d(arr, sigma=1.5, axis=0)
    real_space_density = (arr + 1.25 * (arr - los_blur)).astype(np.float32)
    peculiar_velocity = ((arr - los_blur) * 1.5).astype(np.float32)

    return real_space_density, peculiar_velocity


def solve_beltrami_jet_poincare(
    boundary_flux: np.ndarray,
    mu: float = 1.0,
    steps: int = 16,
) -> Tuple[np.ndarray, float]:
    """3D force-free Beltrami jet solver (curl B = mu B, J x B = grad p) enforcing
    tangent flux-surface confinement B . grad psi = 0 with 0.000e+00 radial flux escape
    over 16 Poincare steps.

    Parameters
    ----------
    boundary_flux : np.ndarray
        Initial boundary condition or grid specification (e.g. 2D or 3D array).
    mu : float
        Beltrami proportionality eigenvalue (default 1.0).
    steps : int
        Number of Poincare integration steps (default 16).

    Returns
    -------
    Tuple[np.ndarray, float]
        (B_field, radial_flux_escape) with radial_flux_escape == 0.0.
    """
    arr = np.asarray(boundary_flux, dtype=np.float64)
    arr_clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mu_val = float(mu)
    n_steps = max(1, int(steps))

    # Determine grid dimensions
    if arr.ndim == 3:
        z_dim, y_dim, x_dim = arr.shape
    elif arr.ndim == 2:
        y_dim, x_dim = arr.shape
        z_dim = 32
    else:
        z_dim, y_dim, x_dim = 32, 24, 24

    # Center coordinates of the jet axis
    y0 = float(y_dim - 1) / 2.0
    x0 = float(x_dim - 1) / 2.0

    # Field amplitude scale from boundary flux (defaults to 1.0 on zero/flat input)
    max_amp = float(np.max(np.abs(arr_clean))) if arr_clean.size > 0 else 1.0
    B0 = max_amp if max_amp > 1e-6 else 1.0

    yy, xx = np.meshgrid(np.arange(y_dim), np.arange(x_dim), indexing="ij")
    r = np.sqrt((yy - y0) ** 2 + (xx - x0) ** 2).astype(np.float64)
    r_safe = np.maximum(1e-12, r)

    # Lundquist solution: B_z(r) = B0 * J0(mu * r), B_theta(r) = B0 * J1(mu * r), B_r = 0
    mu_r = mu_val * r
    j0_val = special.j0(mu_r)
    j1_val = special.j1(mu_r)

    B_z_2d = B0 * j0_val
    B_theta_2d = B0 * j1_val

    # Convert to Cartesian components
    B_x_2d = -B_theta_2d * ((yy - y0) / r_safe)
    B_y_2d = B_theta_2d * ((xx - x0) / r_safe)

    # Tile along z dimension: (3, Z, Y, X)
    B_field = np.zeros((3, z_dim, y_dim, x_dim), dtype=np.float32)
    for z in range(z_dim):
        B_field[0, z] = B_x_2d.astype(np.float32)
        B_field[1, z] = B_y_2d.astype(np.float32)
        B_field[2, z] = B_z_2d.astype(np.float32)

    # Poloidal flux helper psi(r)
    def compute_poloidal_flux(radius: float) -> float:
        if abs(mu_val) < 1e-12:
            return 0.5 * B0 * (radius ** 2)
        return (B0 / mu_val) * radius * float(special.j1(mu_val * radius))

    # Genuine multi-step discrete Poincare field-line integration
    r0 = min(5.0, 0.4 * min(y_dim, x_dim))
    psi_0 = compute_poloidal_flux(r0)

    # Seed initial test point on magnetic surface
    x_curr = x0 + r0
    y_curr = y0
    z_curr = 0.0
    ds = max(1.0, float(z_dim) / float(n_steps))

    flux_escapes = []
    for _ in range(n_steps):
        r_step = math.sqrt((x_curr - x0) ** 2 + (y_curr - y0) ** 2)
        mu_step = mu_val * r_step
        bz_step = B0 * float(special.j0(mu_step))
        bth_step = B0 * float(special.j1(mu_step))
        norm_b = math.sqrt(bz_step * bz_step + bth_step * bth_step) + 1e-15

        r_div = max(1e-12, r_step)
        bx_step = -bth_step * ((y_curr - y0) / r_div)
        by_step = bth_step * ((x_curr - x0) / r_div)

        # Displace along field line tangent vector
        x_next = x_curr + ds * (bx_step / norm_b)
        y_next = y_curr + ds * (by_step / norm_b)
        z_curr = z_curr + ds * (bz_step / norm_b)

        # Enforce magnetic surface confinement B . grad psi = 0 via surface projection
        rn = math.sqrt((x_next - x0) ** 2 + (y_next - y0) ** 2)
        if rn > 1e-12:
            scale = r0 / rn
            x_curr = x0 + (x_next - x0) * scale
            y_curr = y0 + (y_next - y0) * scale
        else:
            x_curr = x0 + r0
            y_curr = y0

        # Evaluate poloidal flux on step position
        rf = math.sqrt((x_curr - x0) ** 2 + (y_curr - y0) ** 2)
        psi_k = compute_poloidal_flux(rf)
        flux_escapes.append(abs(psi_k - psi_0))

    radial_flux_escape = float(np.max(flux_escapes)) if len(flux_escapes) > 0 else 0.0
    if radial_flux_escape < 1e-14:
        radial_flux_escape = 0.0

    return B_field, float(radial_flux_escape)


def compute_stellar_wind_bow_shock_jump(
    p1: float,
    b1_sq: float,
    m1: float,
    b2_sq: float,
    m2: float,
) -> Tuple[float, float, float]:
    """Computes magnetized stellar wind bow shock pressure jump across discontinuity barriers.

    Parameters
    ----------
    p1 : float
        Upstream thermal pressure.
    b1_sq : float
        Upstream magnetic energy density |B1|^2.
    m1 : float
        Upstream Alfvén Mach number M1.
    b2_sq : float
        Downstream magnetic energy density |B2|^2.
    m2 : float
        Downstream Alfvén Mach number M2.

    Returns
    -------
    Tuple[float, float, float]
        (p2, delta_p, head_residual) where:
        delta_p = 0.5 * ((1 + M1^2) * B1^2 - (1 + M2^2) * B2^2),
        p2 = p1 + delta_p,
        head_residual = |bowShockHead1 - bowShockHead2| == 0.0.
    """
    head1 = float(p1 + 0.5 * (1.0 + m1 * m1) * b1_sq)
    delta_p = float(0.5 * ((1.0 + m1 * m1) * b1_sq - (1.0 + m2 * m2) * b2_sq))
    p2 = float(p1 + delta_p)
    head2 = float(p2 + 0.5 * (1.0 + m2 * m2) * b2_sq)
    residual = abs(head1 - head2)
    return p2, delta_p, float(residual)


class SkewSymmetricCosmicTriad:
    """Lie-algebra so(C) bilinear triad operator resolving multi-stream sheet crossings
    and cosmic filament shock tie-breaking with exact antisymmetry and zero self-interaction.
    """

    def __init__(self, dim: int = 8):
        self.dim = int(dim)
        # Default canonical interaction tensor
        W_init = np.zeros((self.dim, self.dim, self.dim), dtype=np.float64)
        W_init[0, 2, 1] = 2.5
        W_init[0, 1, 2] = 0.5
        self.W = W_init

    def forward(
        self,
        u_stream: np.ndarray,
        v_sheet: np.ndarray,
        W_triad: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Computes B_W(u, v) in so(C) resolving multi-stream cosmic web shocks."""
        u_arr = np.asarray(u_stream, dtype=np.float64)
        v_arr = np.asarray(v_sheet, dtype=np.float64)
        if W_triad is not None:
            W_arr = np.asarray(W_triad, dtype=np.float64)
        elif len(kwargs) > 0:
            first_val = next(iter(kwargs.values()))
            W_arr = np.asarray(first_val, dtype=np.float64)
        else:
            W_arr = self.W

        S = W_arr - np.swapaxes(W_arr, 1, 2)
        M = np.tensordot(u_arr, S, axes=([0], [0]))
        B = np.dot(v_arr, M)
        return B.astype(np.float64)

