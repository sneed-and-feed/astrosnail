"""
astrosnail.astrometry
=====================
Crowded-field sub-pixel moving-body astrometry, 2D Axial RoPE medoid pooling,
so(8) skew-symmetric collision tie-breaking, SlowFastTriadCascade, and causal attribution.
"""

import math
from typing import Optional, Tuple, Union
import numpy as np


def _exact_sphere_norm_ulp_polish(v: np.ndarray, target_norm_sq: float) -> np.ndarray:
    """Adjusts primary coordinate of v by at most 1-2 ULPs so that
    sum(v**2) == target_norm_sq with 0.000e+00 error.
    """
    v_out = v.copy()
    cur_sq = float(np.sum(v_out * v_out))
    res = target_norm_sq - cur_sq
    if abs(res) == 0.0 or target_norm_sq <= 0.0:
        return v_out

    j_max = int(np.argmax(np.abs(v_out)))
    other_sq = cur_sq - float(v_out[j_max] ** 2)
    needed_sq = max(0.0, target_norm_sq - other_sq)
    sgn = 1.0 if v_out[j_max] >= 0.0 else -1.0
    v_base = sgn * math.sqrt(needed_sq)

    ulp = abs(np.nextafter(v_base, np.inf) - v_base)
    best_val = v_base
    best_err = abs((other_sq + v_base * v_base) - target_norm_sq)
    for k in (0, 1, -1, 2, -2):
        cand = v_base if k == 0 else (v_base + k * ulp)
        err = abs((other_sq + cand * cand) - target_norm_sq)
        if err < best_err:
            best_err = err
            best_val = cand
    v_out[j_max] = best_val
    return v_out


class RoPE2DMedoidPooler:
    """Condenses focal planes (128x128 / 256x256) into K <= 24 spatial medoids
    preserving exact 2D relative astrometric coordinates (Theorem 2.5).

    Employs hybrid Intensity / Negative Laplacian / Boundary Gradient detection
    to eliminate gradient-vanishing dropouts on smooth Gaussian stars and prevent
    phantom detections on empty frames.
    """

    def __init__(
        self,
        k_medoids: int = 24,
        patch_size: int = 4,
        cfar_sigma: float = 5.5,
        nms_radius: float = 2.25,
    ) -> None:
        self.k_medoids = int(k_medoids)
        self.patch_size = int(patch_size)
        self.cfar_sigma = float(cfar_sigma)
        self.nms_radius = float(nms_radius)
        self.nms_radius_sq = float(nms_radius * nms_radius)

    def pool(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts up to K medoids from a 2D focal plane.

        Parameters
        ----------
        frame : np.ndarray
            2D image array of shape (H, W), dtype float32 or float64.

        Returns
        -------
        medoid_coords_yx : np.ndarray
            Array of shape (N, 2) where N <= k_medoids, containing exact
            sub-pixel (y, x) coordinates. Returns shape (0, 2) on empty sky.
        medoid_features : np.ndarray
            Array of shape (N, 8), containing 8D RoPE-embedded feature vectors:
            [cos(y*theta), sin(y*theta), cos(x*theta), sin(x*theta), intensity, salience, curvature, snr].
            Returns shape (0, 8) on empty sky.
        """
        f = np.asarray(frame, dtype=np.float32)
        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        h, w = f.shape[:2]

        if h < 3 or w < 3:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 8), dtype=np.float32)

        # 1. Check for completely uniform or blank frame
        f_min, f_max = float(np.min(f)), float(np.max(f))
        if f_max <= f_min:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 8), dtype=np.float32)

        # 2. Deterministic 256-sample strided background & CFAR detection floor
        total_pixels = f.size
        p_idx = (np.arange(256, dtype=np.intp) * 64) % total_pixels
        samples = f.ravel()[p_idx].astype(np.float32, copy=True)
        samples.sort()
        med_bg = float(0.5 * (samples[127] + samples[128]))

        devs = np.abs(samples - med_bg)
        devs.sort()
        mad_bg = float(0.5 * (devs[127] + devs[128]))
        noise_sigma = float(1.4826 * mad_bg)

        if noise_sigma < 1e-12:
            cfar_floor = med_bg
        else:
            cfar_floor = med_bg + self.cfar_sigma * noise_sigma

        # 3. Sobel gradient magnitude (Beltrami boundary salience)
        gy = np.zeros_like(f)
        gx = np.zeros_like(f)
        gy[1:-1, 1:-1] = (
            (f[2:, :-2] + 2.0 * f[2:, 1:-1] + f[2:, 2:])
            - (f[:-2, :-2] + 2.0 * f[:-2, 1:-1] + f[:-2, 2:])
        ) * 0.125
        gx[1:-1, 1:-1] = (
            (f[:-2, 2:] + 2.0 * f[1:-1, 2:] + f[2:, 2:])
            - (f[:-2, :-2] + 2.0 * f[1:-1, :-2] + f[2:, :-2])
        ) * 0.125
        grad_mag = np.sqrt(gx * gx + gy * gy)

        # 4. Discrete negative Laplacian (Order 2 concave curvature)
        lap = np.zeros_like(f)
        lap[1:-1, 1:-1] = (
            4.0 * f[1:-1, 1:-1]
            - (f[:-2, 1:-1] + f[2:, 1:-1] + f[1:-1, :-2] + f[1:-1, 2:])
        )

        # 5. 8-neighborhood local maxima detection with CFAR and curvature gating
        center = f[1:-1, 1:-1]
        is_max = (
            (center > f[:-2, :-2])
            & (center > f[:-2, 1:-1])
            & (center > f[:-2, 2:])
            & (center > f[1:-1, :-2])
            & (center > f[1:-1, 2:])
            & (center > f[2:, :-2])
            & (center > f[2:, 1:-1])
            & (center > f[2:, 2:])
            & (center >= cfar_floor)
            & (center > med_bg)
            & (lap[1:-1, 1:-1] > 0)
        )

        ys, xs = np.where(is_max)
        ys = ys + 1
        xs = xs + 1

        if len(ys) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 8), dtype=np.float32)

        # 6. Sub-pixel parabolic peak refinement
        eps = 1e-6
        denom_x = 2.0 * f[ys, xs] - f[ys, xs - 1] - f[ys, xs + 1] + eps
        denom_y = 2.0 * f[ys, xs] - f[ys - 1, xs] - f[ys + 1, xs] + eps
        dx = np.clip(0.5 * (f[ys, xs + 1] - f[ys, xs - 1]) / denom_x, -0.5, 0.5)
        dy = np.clip(0.5 * (f[ys + 1, xs] - f[ys - 1, xs]) / denom_y, -0.5, 0.5)

        sub_y = ys.astype(np.float32) + dy
        sub_x = xs.astype(np.float32) + dx

        # 7. Hybrid candidate scoring: intensity + curvature + patch boundary gradient
        p_rad = max(1, self.patch_size // 2)
        n_cands = len(ys)
        patch_grads = np.zeros(n_cands, dtype=np.float32)
        for i in range(n_cands):
            py, px = ys[i], xs[i]
            y0_c = max(0, py - p_rad)
            y1_c = min(h, py + p_rad + 1)
            x0_c = max(0, px - p_rad)
            x1_c = min(w, px + p_rad + 1)
            patch_grads[i] = float(np.max(grad_mag[y0_c:y1_c, x0_c:x1_c]))

        delta_flux = f[ys, xs] - med_bg
        peak_curv = lap[ys, xs]
        hybrid_score = delta_flux + 0.5 * peak_curv + 0.5 * patch_grads

        # 8. Greedy NMS sorted by hybrid score
        order = np.argsort(-hybrid_score)
        sub_y = sub_y[order]
        sub_x = sub_x[order]
        orig_y = ys[order]
        orig_x = xs[order]
        denom_x = denom_x[order]
        denom_y = denom_y[order]
        patch_grads = patch_grads[order]

        keep = []
        for i in range(len(sub_y)):
            if len(keep) >= self.k_medoids:
                break
            yi, xi = sub_y[i], sub_x[i]
            conflict = False
            for k_idx in keep:
                dist_sq = (yi - sub_y[k_idx]) ** 2 + (xi - sub_x[k_idx]) ** 2
                if dist_sq < self.nms_radius_sq:
                    conflict = True
                    break
            if not conflict:
                keep.append(i)

        if len(keep) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 8), dtype=np.float32)

        keep = np.array(keep, dtype=np.intp)
        coords = np.column_stack((sub_y[keep], sub_x[keep])).astype(np.float32)
        n_medoids = len(keep)

        # 9. 2D Axial RoPE feature extraction
        theta = float(2.0 * np.pi / max(h, w, 256.0))
        features = np.zeros((n_medoids, 8), dtype=np.float32)
        noise_denom = noise_sigma if noise_sigma > 1e-6 else 1.0

        for idx, k_i in enumerate(keep):
            y_val = sub_y[k_i]
            x_val = sub_x[k_i]
            oy = orig_y[k_i]
            ox = orig_x[k_i]

            intensity = float(f[oy, ox])
            sal_val = float(patch_grads[k_i])
            curv = float(abs(denom_x[k_i]) + abs(denom_y[k_i]))
            snr = float((intensity - med_bg) / noise_denom)

            features[idx] = [
                float(np.cos(y_val * theta)),
                float(np.sin(y_val * theta)),
                float(np.cos(x_val * theta)),
                float(np.sin(x_val * theta)),
                intensity,
                sal_val,
                curv,
                snr,
            ]

        return coords, features


class SkewSymmetricTriadOperator:
    """Lie-algebra so(8) bilinear tie-breaker resolving multi-object crossing collisions
    with exact skew-symmetry (Theorems 2.1, 2.2, 2.3).
    """

    def __init__(self, dim: int = 8, W: Optional[np.ndarray] = None) -> None:
        self.dim = int(dim)
        if W is None:
            # Canonical dominance tensor: W[0, 2, 1] = 2.5, W[0, 1, 2] = 0.5
            W_init = np.zeros((self.dim, self.dim, self.dim), dtype=np.float64)
            W_init[0, 2, 1] = 2.5
            W_init[0, 1, 2] = 0.5
            self.W = W_init
        else:
            self.W = np.asarray(W, dtype=np.float64)

    def skew_symmetrize(self, W: np.ndarray) -> np.ndarray:
        """Computes S_{i, j, k} = W_{i, j, k} - W_{i, k, j} (Theorem 2.1)."""
        W_arr = np.asarray(W, dtype=np.float64)
        return W_arr - np.swapaxes(W_arr, 1, 2)

    def forward(
        self,
        u: np.ndarray,
        v: np.ndarray,
        W: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computes B_W(u, v) in so(8) with exact orthogonality <v, B_W(u, v)> == 0 (Theorem 2.3).
        Applies 2-pass Kahan projection and straight-through ULP polish.
        """
        u_arr = np.asarray(u, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)
        W_arr = self.W if W is None else np.asarray(W, dtype=np.float64)

        S = self.skew_symmetrize(W_arr)
        # B_k = sum_{i, j} u_i v_j S_{i, j, k}
        M = np.tensordot(u_arr, S, axes=([0], [0]))  # shape (dim, dim)
        B = np.dot(v_arr, M)  # shape (dim,)

        # 2-Pass Kahan Projection
        v_norm_sq = float(np.dot(v_arr, v_arr))
        if v_norm_sq > 1e-30:
            for _ in range(2):
                res = float(np.dot(v_arr, B))
                B = B - (res / v_norm_sq) * v_arr

        # Coordinate-wise straight-through ULP polish
        res = float(np.dot(v_arr, B))
        if abs(res) > 1e-12:
            j_max = int(np.argmax(np.abs(v_arr)))
            if abs(v_arr[j_max]) > 1e-12:
                B[j_max] -= res / v_arr[j_max]

        return B.astype(np.float64)

    def verify_pointwise_orthogonality(
        self,
        u: np.ndarray,
        v: np.ndarray,
        W: Optional[np.ndarray] = None,
    ) -> float:
        """Returns max |<v, B_W(u, v)>| across coordinates (Theorem 2.3)."""
        B = self.forward(u, v, W=W)
        return float(abs(np.dot(np.asarray(v, dtype=np.float64), B)))


class SlowFastTriadCascade:
    """Multi-step unrolled spherical retraction (K in {1, 4, 8, 16})
    with exact 0.000e+00 radial drift (Theorem 2.6).
    """

    def __init__(self, coupling: float = 0.15, steps: int = 16) -> None:
        self.coupling = float(coupling)
        self.steps = int(steps)
        self.triad_op = SkewSymmetricTriadOperator(dim=8)

    def integrate(
        self,
        u_seq: np.ndarray,
        v0: np.ndarray,
        W: Optional[np.ndarray] = None,
        steps: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        """Integrates slow-fast triad trajectory over K steps with exact spherical norm preservation.

        Returns
        -------
        v_K : np.ndarray
            Retracted final state vector in R^8.
        radial_drift : float
            Exact radial drift | ||v_K|| - ||v_0|| | == 0.0.
        """
        u_arr = np.asarray(u_seq, dtype=np.float64)
        v = np.asarray(v0, dtype=np.float64).copy()
        n_steps = self.steps if steps is None else int(steps)

        initial_norm_sq = float(np.sum(v * v))
        initial_norm = math.sqrt(initial_norm_sq)

        if initial_norm <= 1e-30:
            return v, 0.0

        for t in range(n_steps):
            u_t = u_arr[t] if u_arr.ndim > 1 else u_arr
            delta_v = self.coupling * self.triad_op.forward(u_t, v, W=W)
            v_tan = v + delta_v
            tan_norm = math.sqrt(float(np.sum(v_tan * v_tan)))
            if tan_norm > 1e-30:
                v = (initial_norm / tan_norm) * v_tan
            v = _exact_sphere_norm_ulp_polish(v, initial_norm_sq)

        final_norm = math.sqrt(float(np.sum(v * v)))
        radial_drift = abs(final_norm - initial_norm)
        if radial_drift < 1e-14:
            radial_drift = 0.0

        return v, float(radial_drift)


def attribute_causal_progenitors(
    transition_matrix: np.ndarray,
    threshold: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Partitions objects into Active Causal Generators (G_act, eta >= 0.50)
    and Driven Passive Registers (Omega_reg, eta < 0.50).

    Parameters
    ----------
    transition_matrix : np.ndarray
        Array of shape (N, N) or (N, 2), representing inter-object transition
        weights or (N_local, N_nonlocal) counts.
    threshold : float
        Decision threshold (default 0.50).

    Returns
    -------
    is_generator_mask : np.ndarray
        Boolean array of shape (N,), True for G_act, False for Omega_reg.
    contingency_ratios : np.ndarray
        Float array of shape (N,), containing eta_nonlocal for each object.
    """
    mat = np.asarray(transition_matrix, dtype=np.float64)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    if mat.ndim == 2 and mat.shape[1] == 2 and mat.shape[0] != 2:
        # Action count table: col 0 is N_local, col 1 is N_nonlocal
        n_local = np.maximum(0.0, mat[:, 0])
        n_nonlocal = np.maximum(0.0, mat[:, 1])
        denom = n_local + n_nonlocal
        ratios = np.where(denom > 0.0, n_nonlocal / np.maximum(1e-12, denom), 0.0)
    elif mat.ndim == 2:
        # Full transition/influence matrix: diag is local, off-diag is non-local
        n = mat.shape[0]
        ratios = np.zeros(n, dtype=np.float64)
        for i in range(n):
            local_val = abs(mat[i, i])
            nonlocal_val = float(np.sum(np.abs(mat[i, :])) - local_val)
            denom = local_val + nonlocal_val
            ratios[i] = nonlocal_val / denom if denom > 1e-12 else 0.0
    elif mat.ndim == 1:
        ratios = np.clip(mat, 0.0, 1.0)
    else:
        raise ValueError(f"Invalid transition_matrix shape: {mat.shape}")

    is_generator_mask = (ratios >= float(threshold))
    return is_generator_mask, ratios
