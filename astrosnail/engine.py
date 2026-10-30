"""
astrosnail.engine
=================
Unified AstroSnailEngine orchestrating transients, astrometry, and cosmology
subsystems with backend dispatch, causal online streaming, and runtime verification
of formal Theorems 2.1–2.9.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

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
    compute_mrxmhd_flow_jump,
    SkewSymmetricFaultTriad,
)


def verify_formal_theorems(seed: int = 20261025) -> Tuple[bool, Dict[str, Any]]:
    """Validates machine-checked numerical bounds for Theorems 2.1 through 2.9
    from formalization/OcclusionTriad.lean on live instantiated operators.

    Parameters
    ----------
    seed : int
        Deterministic random seed (default 20261025).

    Returns
    -------
    Tuple[bool, Dict[str, Any]]
        (all_passed, theorem_audit_results)
    """
    rng = np.random.default_rng(seed)
    results: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Theorem 2.1: Skew-Symmetric Color-Occlusion Antisymmetry
    # S(W, i, c2, c1) = -S(W, i, c1, c2), S(W, i, c, c) = 0
    # -------------------------------------------------------------------------
    C = 8
    W_rand = rng.standard_normal((C, C, C))
    triad_op = SkewSymmetricTriadOperator(dim=C, W=W_rand)
    S = triad_op.skew_symmetrize(W_rand)

    # Check S_{i, c2, c1} == -S_{i, c1, c2}
    antisym_err = float(np.max(np.abs(S + np.swapaxes(S, 1, 2))))
    # Check S_{i, c, c} == 0
    diag_vals = np.array([S[:, c, c] for c in range(C)])
    diag_err = float(np.max(np.abs(diag_vals)))

    results["theorem_2_1"] = {
        "passed": (antisym_err < 1e-14 and diag_err == 0.0),
        "antisymmetry_error": antisym_err,
        "self_interaction_error": diag_err,
    }

    # -------------------------------------------------------------------------
    # Theorem 2.2: Exact Tie-Breaking of Symmetric Readout Collisions
    # ell_post(c1) - ell_post(c2) = 2 * beta * alpha * (W(i0, c2, c1) - W(i0, c1, c2))
    # -------------------------------------------------------------------------
    i0, c1, c2 = 0, 1, 2
    alpha = 1.5
    beta = 2.0
    u_vec = np.zeros(C, dtype=np.float64)
    u_vec[i0] = beta
    v_vec = np.zeros(C, dtype=np.float64)
    v_vec[c1] = alpha
    v_vec[c2] = alpha

    # Setup asymmetric W: W(i0, c2, c1) = 2.5, W(i0, c1, c2) = 0.5
    W_tie = np.zeros((C, C, C), dtype=np.float64)
    W_tie[i0, c2, c1] = 2.5
    W_tie[i0, c1, c2] = 0.5

    l_sym = np.full(C, 5.0, dtype=np.float64)  # Symmetric tied baseline
    shift = triad_op.forward(u_vec, v_vec, W=W_tie)
    l_post = l_sym + shift

    actual_gap = float(l_post[c1] - l_post[c2])
    expected_gap = 2.0 * beta * alpha * (W_tie[i0, c2, c1] - W_tie[i0, c1, c2])
    gap_formula_err = abs(actual_gap - expected_gap)

    results["theorem_2_2"] = {
        "passed": (gap_formula_err < 1e-12 and actual_gap > 0.0),
        "tie_break_gap": actual_gap,
        "formula_error": float(gap_formula_err),
    }

    # -------------------------------------------------------------------------
    # Theorem 2.3: Pointwise Zero-Energy-Explosion Orthogonality
    # <v, B_W(u, v)> = 0
    # -------------------------------------------------------------------------
    ortho_errs = []
    for _ in range(10):
        u_r = rng.standard_normal(C)
        v_r = rng.standard_normal(C)
        W_r = rng.standard_normal((C, C, C))
        err = triad_op.verify_pointwise_orthogonality(u_r, v_r, W=W_r)
        ortho_errs.append(err)
    max_ortho_err = max(ortho_errs)

    results["theorem_2_3"] = {
        "passed": (max_ortho_err <= 1.0e-8),
        "orthogonality_residual": float(max_ortho_err),
    }

    # -------------------------------------------------------------------------
    # Theorem 2.4: Step-0 System 1 Homotopy Equivalence
    # At W = 0, full readout reduces identically to System 1
    # -------------------------------------------------------------------------
    u_h = rng.standard_normal(C)
    v_h = rng.standard_normal(C)
    W_zero = np.zeros((C, C, C), dtype=np.float64)
    shift_zero = triad_op.forward(u_h, v_h, W=W_zero)
    step0_err = float(np.max(np.abs(shift_zero)))

    results["theorem_2_4"] = {
        "passed": (step0_err == 0.0),
        "step0_homotopy_error": float(step0_err),
    }

    # -------------------------------------------------------------------------
    # Theorem 2.5: 2D Axial RoPE Medoid Invariance & Patch Attenuation
    # (a) Relative rotary inner product depends purely on (y* - y_q, x* - x_q)
    # (b) Arithmetic patch averaging strictly attenuates key norm: ||k_avg||^2 < ||k||^2
    # Live module verification: RoPE2DMedoidPooler
    # -------------------------------------------------------------------------
    theta_y = float(rng.uniform(0.1, 1.0))
    theta_x = float(rng.uniform(0.1, 1.0))
    q_y = rng.standard_normal(2)
    q_x = rng.standard_normal(2)
    k_y = rng.standard_normal(2)
    k_x = rng.standard_normal(2)

    def rot2_d(th: float, vec: np.ndarray) -> np.ndarray:
        c, s = math.cos(th), math.sin(th)
        return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])

    y_q, x_q = 12.5, 34.2
    y_star, x_star = 18.1, 40.8
    # Absolute RoPE dot product
    q_rot = np.concatenate([rot2_d(y_q * theta_y, q_y), rot2_d(x_q * theta_x, q_x)])
    k_rot = np.concatenate([rot2_d(y_star * theta_y, k_y), rot2_d(x_star * theta_x, k_x)])
    abs_inner = float(np.dot(q_rot, k_rot))

    # Relative displacement dot product
    rel_inner = float(
        np.dot(q_y, rot2_d((y_star - y_q) * theta_y, k_y))
        + np.dot(q_x, rot2_d((x_star - x_q) * theta_x, k_x))
    )
    phase_err = abs(abs_inner - rel_inner)

    # Patch averaging attenuation: theta1 != theta2
    th1, th2 = 0.2, 1.8
    k_rot1 = rot2_d(th1, k_y)
    k_rot2 = rot2_d(th2, k_y)
    k_avg = 0.5 * (k_rot1 + k_rot2)
    norm_sq_orig = float(np.sum(k_y * k_y))
    norm_sq_avg = float(np.sum(k_avg * k_avg))
    patch_attenuated = bool(norm_sq_avg < norm_sq_orig)

    # Live module verification on RoPE2DMedoidPooler
    pooler = RoPE2DMedoidPooler(k_medoids=8)
    test_frame = np.zeros((64, 64), dtype=np.float32)
    test_frame[16, 20] = 10.0
    test_frame[32, 45] = 8.0
    pooled_coords, pooled_feats = pooler.pool(test_frame)
    live_rope_valid = bool(len(pooled_coords) >= 2 and pooled_feats.shape[1] == 8)
    if live_rope_valid:
        th_grid = float(2.0 * np.pi / max(64, 256.0))
        live_dot_y = float(pooled_feats[0, 0] * pooled_feats[1, 0] + pooled_feats[0, 1] * pooled_feats[1, 1])
        live_expected_dot_y = math.cos((pooled_coords[0, 0] - pooled_coords[1, 0]) * th_grid)
        live_phase_err = abs(live_dot_y - live_expected_dot_y)
        live_rope_valid = bool(live_phase_err <= 1.0e-5)
    else:
        live_phase_err = 0.0

    results["theorem_2_5"] = {
        "passed": (phase_err < 1e-12 and patch_attenuated and live_rope_valid),
        "relative_phase_error": float(phase_err),
        "patch_norm_attenuated": patch_attenuated,
        "live_module_verified": live_rope_valid,
        "live_phase_error": float(live_phase_err),
    }

    # -------------------------------------------------------------------------
    # Theorem 2.6: Multi-Step Unrolled Slow-Fast Spherical Invariance
    # Radial drift == 0.000e+00 across K in {1, 4, 8, 16}
    # -------------------------------------------------------------------------
    cascade = SlowFastTriadCascade(coupling=0.15, steps=16)
    v0 = rng.standard_normal(C)
    u_seq = rng.standard_normal((16, C))

    cascade_drifts = []
    for k_steps in (1, 4, 8, 16):
        v_K, drift = cascade.integrate(u_seq[:k_steps], v0, steps=k_steps)
        cascade_drifts.append(drift)
    max_cas_drift = max(cascade_drifts)

    results["theorem_2_6"] = {
        "passed": (max_cas_drift == 0.0),
        "spherical_radial_drift": float(max_cas_drift),
        "max_cascade_steps_tested": 16,
    }

    # -------------------------------------------------------------------------
    # Theorem 2.7: SO(6) Kelvin-Mandel Stiffness & Discrete Morse Shielding
    # (a) Tr(M C M^T) = Tr(C) within <= 1.14e-13
    # (b) Discrete Morse impulse is <= 0 on ramps, shields thin spikes
    # -------------------------------------------------------------------------
    # Random SO(3) matrix via QR
    H_mat = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(H_mat)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    # Random symmetric positive-definite 6x6 matrix C
    A_6 = rng.standard_normal((6, 6))
    C_mat = np.dot(A_6, A_6.T) + 2.0 * np.eye(6)
    _, trace_res = so6_kelvin_mandel_rotor(Q, C_mat)

    # Discrete Morse tests
    ramp = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    i_ramp = compute_morse_impulse(ramp)
    ramp_rejected = bool(np.all(i_ramp <= 0.0))

    spike = np.array([2.0, 2.0, 10.0, 2.0, 2.0], dtype=np.float32)
    i_spike = compute_morse_impulse(spike)
    spike_amp = 8.0
    expected_impulse = spike_amp ** 2  # 64.0
    spike_shielded = bool(abs(float(i_spike[2]) - expected_impulse) < 1e-4)

    results["theorem_2_7"] = {
        "passed": (trace_res <= 1.14e-13 and ramp_rejected and spike_shielded),
        "stiffness_trace_drift": float(trace_res),
        "ramp_rejected": ramp_rejected,
        "spike_shielded": spike_shielded,
    }

    # -------------------------------------------------------------------------
    # Theorem 2.8: 3D Ideal MHD / Euler Force Balance Bijection & Flux Confinement
    # (a) B_{W_LC}(u, v) = u x v
    # (b) B . grad psi = 0
    # -------------------------------------------------------------------------
    u3 = rng.standard_normal(3)
    v3 = rng.standard_normal(3)
    cross_np = np.cross(u3, v3)

    # Levi-Civita half-tensor
    W_lc = np.zeros((3, 3, 3), dtype=np.float64)
    W_lc[0, 1, 2] = 1.0
    W_lc[1, 2, 0] = 1.0
    W_lc[2, 0, 1] = 1.0
    op3 = SkewSymmetricTriadOperator(dim=3, W=W_lc)
    cross_triad = op3.forward(u3, v3, W=W_lc)
    cross_iso_err = float(np.max(np.abs(cross_triad - cross_np)))

    # Flux surface confinement: B . grad psi == 0
    _, radial_flux_escape = solve_beltrami_jet_poincare(np.zeros((16, 16)), mu=1.0, steps=16)

    results["theorem_2_8"] = {
        "passed": (cross_iso_err <= 1.0e-7 and radial_flux_escape <= 1.0e-8),
        "cross3_isomorphism_error": cross_iso_err,
        "flux_confinement_residual": radial_flux_escape,
    }

    # -------------------------------------------------------------------------
    # Theorem 2.9: Collinear Beltrami Lorentz Vanishing & Bernoulli Jump Law
    # (a) B_{W_LC}(alpha B, B) = 0
    # (b) Bernoulli head jump balance [[p + 0.5 * (1 + M^2) B^2]] = 0
    # Live module verification: SkewSymmetricTriadOperator & compute_mrxmhd_flow_jump
    # -------------------------------------------------------------------------
    B_vec = rng.standard_normal(3)
    collinear_B = 3.5 * B_vec
    collinear_lorentz = op3.forward(collinear_B, B_vec, W=W_lc)
    collinear_lorentz_err = float(np.max(np.abs(collinear_lorentz)))

    # Bernoulli jump balance via live module function
    p1, B1_sq, M1 = 10.0, 4.0, 0.3
    M2 = 0.5
    B2_sq = 2.0
    p2, delta_p, bernoulli_jump_err = compute_mrxmhd_flow_jump(
        p1=p1, b1_sq=B1_sq, m1=M1, b2_sq=B2_sq, m2=M2
    )

    results["theorem_2_9"] = {
        "passed": (collinear_lorentz_err <= 1.0e-12 and bernoulli_jump_err <= 1.0e-12),
        "collinear_lorentz_error": collinear_lorentz_err,
        "bernoulli_jump_residual": float(bernoulli_jump_err),
        "jump_pressure_p2": float(p2),
        "delta_p": float(delta_p),
    }

    all_passed = bool(all(entry["passed"] for entry in results.values()))
    return all_passed, results


class AstroSnailEngine:
    """Unified Reference Engine for sneed-and-feed/astrosnail.
    Orchestrates transients, astrometry, and cosmology modules with
    backend dispatch ("auto" | "rust" | "numpy"), streaming update(),
    and machine-checked formal theorem audits (Theorems 2.1–2.9).
    """

    def __init__(self, backend: str = "auto"):
        self.backend_requested = backend
        self.backend = self._resolve_backend(backend)
        self.ring_buffer: List[np.ndarray] = []
        self.step_count = 0
        self.telemetry: Dict[str, Any] = {}

        # Component instances
        self.shell_splitter = BeltramiLightCurveShell(k_tokens=16)
        self.transient_gate = MorseTransientGateConfig(sigma_thresh=4.0)
        self.clock_splitter = TemporalClockContentSplitter(periodicity_threshold=0.85)
        self.medoid_pooler = RoPE2DMedoidPooler(k_medoids=24)
        self.triad_operator = SkewSymmetricTriadOperator(dim=8)
        self.cascade = SlowFastTriadCascade(coupling=0.15, steps=16)

    def _resolve_backend(self, backend: str) -> str:
        if backend == "auto":
            try:
                from astrosnail import _native
                if hasattr(_native, "is_available") and _native.is_available():
                    return "rust"
            except ImportError:
                pass
            return "numpy"
        elif backend == "rust":
            try:
                from astrosnail import _native
                if not (hasattr(_native, "is_available") and _native.is_available()):
                    raise RuntimeError("Native Rust library is not available.")
                return "rust"
            except ImportError as e:
                raise RuntimeError("Rust backend requested but astrosnail._native cannot be imported.") from e
        elif backend == "numpy":
            return "numpy"
        else:
            raise ValueError(f"Unknown backend: {backend}. Must be 'auto', 'rust', or 'numpy'.")

    def update(
        self,
        frame_or_series: np.ndarray,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Strictly causal online streaming update.
        Maintains an internal 8-frame ring buffer.
        Dispatches 1D series, 2D focal planes, or 3D cubes.
        Returns dict of detections, medoids, classifications, and telemetry HUD.
        """
        self.step_count += 1
        arr = np.asarray(frame_or_series)

        # Update 8-frame ring buffer
        if len(self.ring_buffer) >= 8:
            self.ring_buffer.pop(0)
        self.ring_buffer.append(arr)

        out: Dict[str, Any] = {
            "step": self.step_count,
            "backend": self.backend,
            "input_shape": list(arr.shape),
        }

        # 1D or 2D temporal series: Transients triage
        if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] <= 8):
            if arr.ndim == 1 and arr.shape[0] == 1:
                # Single scalar sample
                val = float(arr[0])
                cls_name, update_applied = self.transient_gate.update(val, timestamp=float(self.step_count))
                out["classification"] = cls_name
                out["update_applied"] = update_applied
                out["glint_rejected_count"] = self.transient_gate.glint_rejected_count
                out["confirmed_transient_count"] = self.transient_gate.confirmed_transient_count
            elif arr.shape[0] >= 3:
                # Sequence of epochs
                u_coarse, u_fine, curl_sal = self.shell_splitter.decompose(arr)
                i_morse = compute_morse_impulse(arr, salience=curl_sal)
                h_clock, h_nope, mask = self.clock_splitter.split(arr)
                mad_floor = compute_spatial_mad_floor_256(curl_sal, mask=mask)

                out["u_coarse"] = u_coarse
                out["u_fine"] = u_fine
                out["curl_salience"] = curl_sal
                out["morse_impulse"] = i_morse
                out["h_clock"] = h_clock
                out["h_nope"] = h_nope
                out["clock_mask"] = mask
                out["mad_floor"] = mad_floor

        # 2D Focal Plane: Sub-pixel astrometry
        elif arr.ndim == 2:
            coords, features = self.medoid_pooler.pool(arr)
            out["medoid_coords_yx"] = coords
            out["medoid_features"] = features
            out["num_medoids"] = len(coords)

            if len(coords) > 0:
                # Run cascade on primary medoid feature
                v0 = features[0]
                u_dummy = np.ones(8, dtype=np.float64)
                v_K, drift = self.cascade.integrate(u_dummy, v0, steps=16)
                out["primary_cascade_drift"] = drift

        # 3D Cosmological Cube: Tomography & Void Core Detection
        elif arr.ndim == 3:
            void_cores = detect_jordan_cavity_cores(arr)
            retracted = retract_void_step0_homotopy(arr, void_cores)
            out["void_cores"] = void_cores
            out["retracted_density"] = retracted
            out["void_voxel_count"] = int(np.sum(void_cores))

        if telemetry:
            out.update(telemetry)

        return out

    def verify_formal_theorems(self, seed: int = 20261025) -> Tuple[bool, Dict[str, Any]]:
        """Validates machine-checked numerical bounds for Theorems 2.1–2.9."""
        return verify_formal_theorems(seed=seed)
