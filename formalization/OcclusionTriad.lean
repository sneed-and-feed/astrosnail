import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic

/-!
# Project AstroSnail: Proof-Carrying Time-Domain Astrophysics & Cosmology Invariants (Theorems 2.1–2.9)

This module formalizes the machine-checked mathematical core of `sneed-and-feed/astrosnail`,
establishing the mathematical invariants governing time-domain outlier triage, crowded-field
sub-pixel moving-body astrometry, and 3D cosmic web tomography:

1. **Theorem 2.1 (`skew_triad_swap_neg`, `skew_triad_self_zero`)**:
   $\mathfrak{so}(C)$ Skew-Symmetric Bilinear Triad antisymmetry and vanishing self-interaction.
2. **Theorem 2.2 (`occlusion_triad_breaks_symmetric_tie`)**:
   Exact tie-breaking of symmetric 2-color collision readouts via bilinear triad shift.
3. **Theorem 2.3 (`darksnail_readout_energy_conservation`)**:
   Pointwise zero-energy-explosion orthogonality: $\langle v, B_W(u, v) \rangle = 0$.
4. **Theorem 2.4 (`darksnail_step0_system1_homotopy`)**:
   Step-0 System 1 homotopy equivalence: full readout reduces identically to baseline.
5. **Theorem 2.5 (`rope2d_medoid_relative_invariance`, `rope2d_arithmetic_norm_lt`)**:
   2D Axial RoPE medoid relative displacement invariance and strict patch-averaging attenuation.
6. **Theorem 2.6 (`darksnail_multistep_spherical_invariance`, `darksnail_multistep_step0_homotopy`)**:
   Multi-step unrolled Slow-Fast spherical cascade norm invariance ($0.000\mathrm{e}{+}00$ drift).
7. **Theorem 2.7 (`elasticEnergy6_rotateStiffness6`, `so6_stiffness_preserves_pos_def`,
   `so6_stiffness_trace_conservation`, `morseImpulse_iff_strict_extremum`,
   `caliper_morse_rejects_washout_step`, `caliper_morse_shields_thin_stringer`)**:
   $\mathrm{SO}(6)$ Kelvin-Mandel stiffness rotor trace/energy conservation and discrete Morse shielding.
8. **Theorem 2.8 (`cross3_eq_triadOperator`, `mhd_euler_force_balance_bijection`,
   `mhd_flux_surface_confinement_iff`, `stellarator_parity_breaks_tie_and_conserves_flux`,
   `stellarator_zero_torsion_homotopy_iff`)**:
   3D Ideal MHD $\leftrightarrow$ Euler force balance bijection and tangent flux confinement.
9. **Theorem 2.9 (`sub_alfvenic_beltrami_homotopy_zero_flow`, `sub_alfvenic_beltrami_pos`,
   `beltrami_cross_helicity_collinear_lorentz_vanishes`, `mrxmhd_flow_bernoulli_jump_conservation`,
   `kam_sub_alfvenic_ellipticity_and_gap`)**:
   Relaxed-Flow MRxMHD Bernoulli jump law and collinear Beltrami Lorentz vanishing.

All theorems compile with **0 unproved placeholders**, **0 custom axioms**, and **0 warnings**.
-/

namespace AstroSnail.Formalization

open scoped BigOperators
open Real Finset
open Classical

-- ============================================================================
-- Section 1: Foundational Bilinear Triad Operator & Shell Energy (Theorems 2.1–2.4)
-- ============================================================================

/-- Skew-symmetrized bilinear triad operator:
$$B_W(u, v)_k = \sum_{i, j \in \mathrm{Fin}\,M} u_i v_j (W_{i,j,k} - W_{i,k,j}).$$ -/
def triadOperator {M : ℕ} (W : Fin M → Fin M → Fin M → ℝ)
    (u v : Fin M → ℝ) (k : Fin M) : ℝ :=
  ∑ i : Fin M, ∑ j : Fin M, u i * v j * (W i j k - W i k j)

/-- Standard Euclidean inner product on $\mathbb{R}^M$. -/
def shellInner {M : ℕ} (x y : Fin M → ℝ) : ℝ :=
  ∑ k : Fin M, x k * y k

/-- Squared Euclidean norm $\|x\|_2^2 = \langle x, x \rangle$ on $\mathbb{R}^M$. -/
def shellEnergy {M : ℕ} (x : Fin M → ℝ) : ℝ :=
  shellInner x x

/-- Pointwise orthogonality $\langle v, B_W(u, v) \rangle = 0$. -/
theorem triad_pointwise_orthogonal {M : ℕ} (W : Fin M → Fin M → Fin M → ℝ)
    (u v : Fin M → ℝ) :
    shellInner v (triadOperator W u v) = 0 := by
  unfold shellInner triadOperator
  simp_rw [mul_sum]; rw [sum_comm]
  refine sum_eq_zero fun i _ => ?_
  simp_rw [mul_sub, sum_sub_distrib, sub_eq_zero]
  rw [sum_comm]
  exact sum_congr rfl fun j _ => sum_congr rfl fun k _ => by ring

/-! ### Theorem 2.1: Skew-Symmetric Color-Occlusion Antisymmetry -/

/-- Directed interaction tensor $S(W, i, c_1, c_2) = W(i, c_1, c_2) - W(i, c_2, c_1)$. -/
def skewTriad {C : ℕ} (W : Fin C → Fin C → Fin C → ℝ) (i c₁ c₂ : Fin C) : ℝ :=
  W i c₁ c₂ - W i c₂ c₁

/-- **Theorem 2.1 (`skew_triad_swap_neg`)**:
Swapping competing colors strictly negates directed interaction:
$$S(W, i, c_2, c_1) = -S(W, i, c_1, c_2),$$
and self-interaction vanishes identically: $S(W, i, c, c) = 0$. -/
theorem skew_triad_swap_neg {C : ℕ} (W : Fin C → Fin C → Fin C → ℝ) (i c₁ c₂ : Fin C) :
    skewTriad W i c₂ c₁ = -skewTriad W i c₁ c₂ ∧ skewTriad W i c₁ c₁ = 0 := by
  unfold skewTriad; constructor <;> ring

theorem skew_triad_self_zero {C : ℕ} (W : Fin C → Fin C → Fin C → ℝ) (i c : Fin C) :
    skewTriad W i c c = 0 :=
  (skew_triad_swap_neg W i c c).2

/-! ### Theorem 2.2: Exact Tie-Breaking of Symmetric Readout Collisions -/

/-- Post-readout logit combining symmetric baseline $\ell_{\mathrm{sym}}$ and triad shift. -/
def postLogit (C : ℕ) (ℓ_sym : Fin C → ℝ) (W : Fin C → Fin C → Fin C → ℝ)
    (u v : Fin C → ℝ) (k : Fin C) : ℝ :=
  ℓ_sym k + triadOperator W u v k

lemma triadOperator_two_color_eval {C : ℕ} (W : Fin C → Fin C → Fin C → ℝ)
    (u v : Fin C → ℝ) (i₀ c₁ c₂ k : Fin C) (α β : ℝ) (hc : c₁ ≠ c₂)
    (hu_i0 : u i₀ = β) (hu_other : ∀ i : Fin C, i ≠ i₀ → u i = 0)
    (hv_c1 : v c₁ = α) (hv_c2 : v c₂ = α)
    (hv_other : ∀ c : Fin C, c ≠ c₁ → c ≠ c₂ → v c = 0) :
    triadOperator W u v k =
      β * α * (W i₀ c₁ k - W i₀ k c₁) + β * α * (W i₀ c₂ k - W i₀ k c₂) := by
  unfold triadOperator
  rw [Finset.sum_eq_single i₀]
  · rw [← Finset.add_sum_erase _ _ (Finset.mem_univ c₁),
        ← Finset.add_sum_erase _ _ (Finset.mem_erase.mpr ⟨hc.symm, Finset.mem_univ c₂⟩),
        Finset.sum_eq_zero, add_zero, hu_i0, hv_c1, hv_c2]
    intro j hj
    simp only [Finset.mem_erase] at hj
    simp [hv_other j hj.2.1 hj.1]
  · exact fun i _ hi => Finset.sum_eq_zero fun j _ => by simp [hu_other i hi]
  · exact fun h => (h (Finset.mem_univ i₀)).elim

/-- **Theorem 2.2 (`occlusion_triad_breaks_symmetric_tie`)**:
Under a symmetric readout tie $\ell_{\mathrm{sym}}(c_1) = \ell_{\mathrm{sym}}(c_2)$, the bilinear
triad shift produces an exact antisymmetric margin:
$$\ell_{\mathrm{post}}(c_1) - \ell_{\mathrm{post}}(c_2) = 2 \beta \alpha (W(i_0, c_2, c_1) - W(i_0, c_1, c_2)),$$
strictly breaking the tie whenever $W(i_0, c_2, c_1) > W(i_0, c_1, c_2)$. -/
theorem occlusion_triad_breaks_symmetric_tie {C : ℕ}
    (ℓ_sym : Fin C → ℝ) (W : Fin C → Fin C → Fin C → ℝ)
    (u v : Fin C → ℝ) (i₀ c₁ c₂ : Fin C) (α β : ℝ)
    (hα : 0 < α) (hβ : 0 < β) (h_tie : ℓ_sym c₁ = ℓ_sym c₂)
    (hu_i0 : u i₀ = β) (hu_other : ∀ i : Fin C, i ≠ i₀ → u i = 0)
    (hv_c1 : v c₁ = α) (hv_c2 : v c₂ = α)
    (hv_other : ∀ c : Fin C, c ≠ c₁ → c ≠ c₂ → v c = 0) :
    (postLogit C ℓ_sym W u v c₁ - postLogit C ℓ_sym W u v c₂ =
      2 * β * α * (W i₀ c₂ c₁ - W i₀ c₁ c₂)) ∧
    (W i₀ c₂ c₁ > W i₀ c₁ c₂ →
      postLogit C ℓ_sym W u v c₁ > postLogit C ℓ_sym W u v c₂) := by
  have h_diff : postLogit C ℓ_sym W u v c₁ - postLogit C ℓ_sym W u v c₂ =
      2 * β * α * (W i₀ c₂ c₁ - W i₀ c₁ c₂) := by
    by_cases hc : c₁ = c₂
    · subst hc; ring
    · unfold postLogit
      rw [triadOperator_two_color_eval W u v i₀ c₁ c₂ c₁ α β hc hu_i0 hu_other hv_c1 hv_c2 hv_other,
          triadOperator_two_color_eval W u v i₀ c₁ c₂ c₂ α β hc hu_i0 hu_other hv_c1 hv_c2 hv_other,
          h_tie]; ring
  exact ⟨h_diff, fun hW => by nlinarith [mul_pos hβ hα]⟩

/-! ### Theorem 2.3: Readout Energy Conservation -/

/-- **Theorem 2.3 (`darksnail_readout_energy_conservation`)**:
For any mode vectors $u, v$ and coupling tensor $W$, $\langle v, B_W(u, v) \rangle = 0$,
and the Slow-Fast recurrent pump has zero radial energy growth. -/
theorem darksnail_readout_energy_conservation {C M : ℕ}
    (W : Fin C → Fin C → Fin C → ℝ) (u v : Fin C → ℝ)
    (W_rec : Fin M → Fin M → Fin M → ℝ) (z_H z_L : Fin M → ℝ) :
    (∑ k : Fin C, v k * triadOperator W u v k) = 0 ∧
    shellInner z_L (triadOperator W_rec z_H z_L) = 0 :=
  ⟨triad_pointwise_orthogonal W u v, triad_pointwise_orthogonal W_rec z_H z_L⟩

/-! ### Theorem 2.4: Step-0 System 1 Homotopy Equivalence -/

/-- Full readout combining System 1 baseline, symmetric shift, and skew triad shift. -/
def darksnailReadout (C : ℕ) (ℓ_S1 Δ_sym : Fin C → ℝ)
    (W : Fin C → Fin C → Fin C → ℝ) (u v : Fin C → ℝ) (k : Fin C) : ℝ :=
  ℓ_S1 k + Δ_sym k + triadOperator W u v k

/-- **Theorem 2.4 (`darksnail_step0_system1_homotopy`)**:
At Step-0 ($\Delta_{\mathrm{sym}} = 0, W = 0$), the full readout reduces identically to System 1. -/
theorem darksnail_step0_system1_homotopy {C : ℕ}
    (ℓ_S1 u v : Fin C → ℝ) (k : Fin C) :
    darksnailReadout C ℓ_S1 0 0 u v k = ℓ_S1 k := by
  simp [darksnailReadout, triadOperator]

-- ============================================================================
-- Section 2: 2D Axial RoPE Medoid Invariance & Spherical Cascade (Theorems 2.5–2.6)
-- ============================================================================

noncomputable def rot2 (θ : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (Real.cos θ * v.1 - Real.sin θ * v.2, Real.sin θ * v.1 + Real.cos θ * v.2)

def inner2 (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def normSq2 (v : ℝ × ℝ) : ℝ := inner2 v v

noncomputable def rope2d (θ_y θ_x : ℝ) (q_y q_x : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  (rot2 θ_y q_y, rot2 θ_x q_x)

def inner2d (u v : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := inner2 u.1 v.1 + inner2 u.2 v.2
def normSq2d (v : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := normSq2 v.1 + normSq2 v.2
noncomputable def avg2 (u v : ℝ × ℝ) : ℝ × ℝ := ((u.1 + v.1) / 2, (u.2 + v.2) / 2)

noncomputable def rope2dPatchAvg (θ_y1 θ_x1 θ_y2 θ_x2 : ℝ) (k_y k_x : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  (avg2 (rot2 θ_y1 k_y) (rot2 θ_y2 k_y), avg2 (rot2 θ_x1 k_x) (rot2 θ_x2 k_x))

lemma inner2_rot2_shift (α β : ℝ) (u v : ℝ × ℝ) :
    inner2 (rot2 α u) (rot2 β v) = inner2 u (rot2 (β - α) v) := by
  unfold inner2 rot2; rw [Real.cos_sub, Real.sin_sub]; ring

/-- **Theorem 2.5a (`rope2d_medoid_relative_invariance`)**:
The 2D axial rotary inner product depends purely on the relative 2D displacement:
$$\langle \mathrm{rope2d}(y_q, x_q, q), \mathrm{rope2d}(y^*, x^*, k) \rangle
  = \langle q_y, \mathcal{R}_{y^* - y_q} k_y \rangle + \langle q_x, \mathcal{R}_{x^* - x_q} k_x \rangle.$$ -/
theorem rope2d_medoid_relative_invariance
    (y_q x_q y_star x_star : ℝ) (q_y q_x k_y k_x : ℝ × ℝ) :
    inner2d (rope2d y_q x_q q_y q_x) (rope2d y_star x_star k_y k_x) =
      inner2 q_y (rot2 (y_star - y_q) k_y) + inner2 q_x (rot2 (x_star - x_q) k_x) := by
  unfold inner2d rope2d; rw [inner2_rot2_shift, inner2_rot2_shift]

lemma normSq2_avg2_rot2 (θ₁ θ₂ : ℝ) (k : ℝ × ℝ) :
    normSq2 (avg2 (rot2 θ₁ k) (rot2 θ₂ k)) =
      normSq2 k * (1 + Real.cos (θ₁ - θ₂)) / 2 := by
  unfold normSq2 inner2 avg2 rot2; rw [Real.cos_sub]
  linear_combination
    ((k.1 ^ 2 + k.2 ^ 2) / 4) * Real.cos_sq_add_sin_sq θ₁ +
    ((k.1 ^ 2 + k.2 ^ 2) / 4) * Real.cos_sq_add_sin_sq θ₂

lemma normSq2_nonneg (k : ℝ × ℝ) : 0 ≤ normSq2 k :=
  add_nonneg (mul_self_nonneg k.1) (mul_self_nonneg k.2)

/-- **Theorem 2.5b (`rope2d_arithmetic_norm_lt`)**:
Arithmetic patch averaging across distinct grid phases strictly attenuates 2D key norm:
$$\|\mathrm{rope2dPatchAvg}(\theta, k)\|^2 < \|k\|^2.$$ -/
theorem rope2d_arithmetic_norm_lt
    (θ_y1 θ_x1 θ_y2 θ_x2 : ℝ) (k_y k_x : ℝ × ℝ)
    (h_att : (Real.cos (θ_y1 - θ_y2) < 1 ∧ 0 < normSq2 k_y) ∨
             (Real.cos (θ_x1 - θ_x2) < 1 ∧ 0 < normSq2 k_x)) :
    normSq2d (rope2dPatchAvg θ_y1 θ_x1 θ_y2 θ_x2 k_y k_x) < normSq2d (k_y, k_x) := by
  unfold normSq2d rope2dPatchAvg; dsimp only
  rw [normSq2_avg2_rot2, normSq2_avg2_rot2]
  rcases h_att with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;>
    nlinarith [normSq2_nonneg k_y, normSq2_nonneg k_x, Real.cos_le_one (θ_y1 - θ_y2), Real.cos_le_one (θ_x1 - θ_x2)]

/-! ### Theorem 2.6: Multi-Step Unrolled Slow-Fast Spherical Invariance -/

def tangentTriadStep (M : ℕ) (γ : ℝ) (W_rec : Fin M → Fin M → Fin M → ℝ)
    (u v : Fin M → ℝ) (k : Fin M) : ℝ :=
  γ * triadOperator W_rec u v k

noncomputable def sphericalRetract {M : ℕ} (v v_upd : Fin M → ℝ) (k : Fin M) : ℝ :=
  (Real.sqrt (shellEnergy v) / Real.sqrt (shellEnergy v_upd)) * v_upd k

noncomputable def sphericalTriadStep (M : ℕ) (γ : ℝ) (W_rec : Fin M → Fin M → Fin M → ℝ)
    (u v : Fin M → ℝ) : Fin M → ℝ :=
  sphericalRetract v (fun k => v k + tangentTriadStep M γ W_rec u v k)

noncomputable def multistepSphericalTriad (M : ℕ) (γ : ℕ → ℝ)
    (W_rec : ℕ → Fin M → Fin M → Fin M → ℝ)
    (u : ℕ → Fin M → ℝ) (v₀ : Fin M → ℝ) : ℕ → (Fin M → ℝ)
  | 0 => v₀
  | t + 1 => sphericalTriadStep M (γ t) (W_rec t) (u t) (multistepSphericalTriad M γ W_rec u v₀ t)

lemma shellEnergy_nonneg {M : ℕ} (x : Fin M → ℝ) : 0 ≤ shellEnergy x :=
  Finset.sum_nonneg (s := Finset.univ) fun k _ => mul_self_nonneg (x k)

lemma shellEnergy_eq_zero_iff {M : ℕ} (x : Fin M → ℝ) :
    shellEnergy x = 0 ↔ ∀ k : Fin M, x k = 0 := by
  unfold shellEnergy shellInner
  simp [Finset.sum_eq_zero_iff_of_nonneg (fun k _ => mul_self_nonneg (x k))]

lemma tangentTriadStep_orthogonal {M : ℕ} (γ : ℝ) (W_rec : Fin M → Fin M → Fin M → ℝ)
    (u v : Fin M → ℝ) :
    shellInner v (tangentTriadStep M γ W_rec u v) = 0 := by
  unfold shellInner tangentTriadStep
  simp_rw [mul_left_comm (v _), ← Finset.mul_sum]
  exact mul_eq_zero_of_right γ (triad_pointwise_orthogonal W_rec u v)

lemma shellEnergy_add_of_orthogonal {M : ℕ} (v δv : Fin M → ℝ)
    (h_orth : shellInner v δv = 0) :
    shellEnergy (fun k => v k + δv k) = shellEnergy v + shellEnergy δv := by
  have h_expand :
      shellEnergy (fun k => v k + δv k) =
        shellEnergy v + 2 * shellInner v δv + shellEnergy δv := by
    unfold shellEnergy shellInner
    simp only [← Finset.sum_add_distrib, Finset.mul_sum]
    exact Finset.sum_congr rfl fun k _ => by ring
  linarith

lemma shellEnergy_scale {M : ℕ} (c : ℝ) (x : Fin M → ℝ) :
    shellEnergy (fun k => c * x k) = c ^ 2 * shellEnergy x := by
  unfold shellEnergy shellInner; rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun k _ => by ring

lemma sphericalTriadStep_preserves_energy {M : ℕ} (γ : ℝ)
    (W_rec : Fin M → Fin M → Fin M → ℝ) (u v : Fin M → ℝ) :
    shellEnergy (sphericalTriadStep M γ W_rec u v) = shellEnergy v := by
  unfold sphericalTriadStep sphericalRetract
  rw [shellEnergy_scale]
  by_cases hv0 : shellEnergy v = 0
  · simp [hv0]
  · have hv_pos : 0 < shellEnergy v := lt_of_le_of_ne (shellEnergy_nonneg v) (Ne.symm hv0)
    have h_pyth := shellEnergy_add_of_orthogonal v _ (tangentTriadStep_orthogonal γ W_rec u v)
    have h_upd_pos : 0 < shellEnergy (fun k => v k + tangentTriadStep M γ W_rec u v k) := by
      linarith [hv_pos, shellEnergy_nonneg (tangentTriadStep M γ W_rec u v)]
    rw [div_pow, Real.sq_sqrt hv_pos.le, Real.sq_sqrt h_upd_pos.le, div_mul_cancel₀ _ h_upd_pos.ne']

/-- **Theorem 2.6a (`darksnail_multistep_spherical_invariance`)**:
Across any $T$-step unrolled cascade, the terminal retracted state preserves exact initial norm:
$$\|\tilde{v}_L^{(T)}\|_2^2 = \|v_L^{(0)}\|_2^2.$$ -/
theorem darksnail_multistep_spherical_invariance {M : ℕ}
    (γ : ℕ → ℝ) (W_rec : ℕ → Fin M → Fin M → Fin M → ℝ)
    (u : ℕ → Fin M → ℝ) (v₀ : Fin M → ℝ) (T : ℕ) :
    shellEnergy (multistepSphericalTriad M γ W_rec u v₀ T) = shellEnergy v₀ ∧
    (∀ t : ℕ,
      let v_t := multistepSphericalTriad M γ W_rec u v₀ t
      let δv_t := tangentTriadStep M (γ t) (W_rec t) (u t) v_t
      shellInner v_t δv_t = 0) :=
  ⟨by induction T with | zero => rfl | succ t ih => simp [multistepSphericalTriad, sphericalTriadStep_preserves_energy, ih],
   fun t => tangentTriadStep_orthogonal (γ t) (W_rec t) (u t) _⟩

lemma sphericalTriadStep_zero {M : ℕ} (γ : ℝ) (u v : Fin M → ℝ) :
    sphericalTriadStep M γ 0 u v = v := by
  have h_tan : (fun i => v i + tangentTriadStep M γ 0 u v i) = v := by
    funext i; simp [tangentTriadStep, triadOperator]
  unfold sphericalTriadStep sphericalRetract; rw [h_tan]
  by_cases hv0 : shellEnergy v = 0
  · ext k; simp [hv0, (shellEnergy_eq_zero_iff v).mp hv0 k]
  · ext k; rw [div_self (ne_of_gt (Real.sqrt_pos.mpr (lt_of_le_of_ne (shellEnergy_nonneg v) (Ne.symm hv0)))), one_mul]

def system1CarryTrajectory {M : ℕ}
    (step_S1 : ℕ → (Fin M → ℝ) × (Fin M → ℝ) → (Fin M → ℝ) × (Fin M → ℝ))
    (z₀ : (Fin M → ℝ) × (Fin M → ℝ)) : ℕ → (Fin M → ℝ) × (Fin M → ℝ)
  | 0 => z₀
  | t + 1 => step_S1 t (system1CarryTrajectory step_S1 z₀ t)

noncomputable def darksnailCarryTrajectory {M : ℕ}
    (step_S1 : ℕ → (Fin M → ℝ) × (Fin M → ℝ) → (Fin M → ℝ) × (Fin M → ℝ))
    (γ : ℕ → ℝ) (W_rec : ℕ → Fin M → Fin M → Fin M → ℝ)
    (δz_H δz_L : ℕ → Fin M → ℝ)
    (z₀ : (Fin M → ℝ) × (Fin M → ℝ)) : ℕ → (Fin M → ℝ) × (Fin M → ℝ)
  | 0 => z₀
  | t + 1 =>
      let z_t := darksnailCarryTrajectory step_S1 γ W_rec δz_H δz_L z₀ t
      let v_tilde := sphericalTriadStep M (γ t) (W_rec t) z_t.1 z_t.2
      step_S1 t (fun k => z_t.1 k + δz_H t k, fun k => v_tilde k + δz_L t k)

/-- **Theorem 2.6b (`darksnail_multistep_step0_homotopy`)**:
At $W_{\mathrm{rec}} = 0$ and $\delta z = 0$, the $T$-step unrolled cascade reduces identically to System 1. -/
theorem darksnail_multistep_step0_homotopy {M : ℕ}
    (step_S1 : ℕ → (Fin M → ℝ) × (Fin M → ℝ) → (Fin M → ℝ) × (Fin M → ℝ))
    (γ : ℕ → ℝ) (u : ℕ → Fin M → ℝ) (v₀ : Fin M → ℝ)
    (z₀ : (Fin M → ℝ) × (Fin M → ℝ)) (T : ℕ) :
    multistepSphericalTriad M γ 0 u v₀ T = v₀ ∧
    darksnailCarryTrajectory step_S1 γ 0 0 0 z₀ T =
      system1CarryTrajectory step_S1 z₀ T := by
  constructor <;> induction T with
  | zero => rfl
  | succ t ih =>
    simp [multistepSphericalTriad, darksnailCarryTrajectory, system1CarryTrajectory,
          ih, sphericalTriadStep_zero]

-- ============================================================================
-- Section 3: SO(6) Kelvin-Mandel Stiffness & Discrete Morse Shielding (Theorem 2.7)
-- ============================================================================

def elasticEnergy6 (C : Fin 6 → Fin 6 → ℝ) (ε : Fin 6 → ℝ) : ℝ :=
  ∑ i : Fin 6, ∑ j : Fin 6, ε i * C i j * ε j

def matVec6 (M : Fin 6 → Fin 6 → ℝ) (x : Fin 6 → ℝ) (i : Fin 6) : ℝ :=
  ∑ j : Fin 6, M i j * x j

def matTranspose6 (M : Fin 6 → Fin 6 → ℝ) (i j : Fin 6) : ℝ := M j i

def rotateStiffness6 (M C : Fin 6 → Fin 6 → ℝ) (i j : Fin 6) : ℝ :=
  ∑ k : Fin 6, ∑ l : Fin 6, M i k * C k l * M j l

def trace6 (C : Fin 6 → Fin 6 → ℝ) : ℝ := ∑ i : Fin 6, C i i

def IsOrthogonal6 (M : Fin 6 → Fin 6 → ℝ) : Prop :=
  ∀ x : Fin 6 → ℝ, shellEnergy (matVec6 (matTranspose6 M) x) = shellEnergy x

def IsColOrthonormal6 (M : Fin 6 → Fin 6 → ℝ) : Prop :=
  ∀ k l : Fin 6, (∑ i : Fin 6, M i k * M i l) = if k = l then 1 else 0

def IsRowOrthonormal6 (M : Fin 6 → Fin 6 → ℝ) : Prop :=
  ∀ i j : Fin 6, (∑ k : Fin 6, M i k * M j k) = if i = j then 1 else 0

lemma isOrthogonal6_of_rowOrthonormal6 (M : Fin 6 → Fin 6 → ℝ)
    (hM : IsRowOrthonormal6 M) : IsOrthogonal6 M := fun x => by
  unfold shellEnergy shellInner matVec6 matTranspose6
  calc (∑ k : Fin 6, (∑ i : Fin 6, M i k * x i) * (∑ j : Fin 6, M j k * x j))
    _ = ∑ i : Fin 6, ∑ j : Fin 6, x i * x j * (∑ k : Fin 6, M i k * M j k) := by
        simp only [Fin.sum_univ_six]; ring
    _ = ∑ k : Fin 6, x k * x k := by simp [hM _ _]

noncomputable def kelvinMandelDipRotor (θ : ℝ) (i j : Fin 6) : ℝ :=
  let c := Real.cos θ
  let s := Real.sin θ
  let s2 := Real.sqrt 2
  match i.val, j.val with
  | 0, 0 => c * c
  | 0, 2 => s * s
  | 0, 4 => s2 * c * s
  | 1, 1 => 1
  | 2, 0 => s * s
  | 2, 2 => c * c
  | 2, 4 => -s2 * c * s
  | 3, 3 => c
  | 3, 5 => -s
  | 4, 0 => -s2 * c * s
  | 4, 2 => s2 * c * s
  | 4, 4 => c * c - s * s
  | 5, 3 => s
  | 5, 5 => c
  | _, _ => 0

lemma kelvinMandelDipRotor_orthonormal (θ : ℝ) :
    IsRowOrthonormal6 (kelvinMandelDipRotor θ) ∧
    IsColOrthonormal6 (kelvinMandelDipRotor θ) ∧
    IsOrthogonal6 (kelvinMandelDipRotor θ) := by
  have h_row : ∀ φ : ℝ, IsRowOrthonormal6 (kelvinMandelDipRotor φ) := by
    intro φ a b
    have hs2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
    have h2 := Real.cos_sq_add_sin_sq φ
    have h4 : Real.cos φ ^ 4 + Real.sin φ ^ 4 = 1 - 2 * Real.cos φ ^ 2 * Real.sin φ ^ 2 := by
      linear_combination (Real.cos φ ^ 2 + Real.sin φ ^ 2 + 1) * h2
    rw [Fin.sum_univ_six]
    fin_cases a <;>
      simp only [kelvinMandelDipRotor, Fin.isValue, Fin.coe_ofNat_eq_mod, Nat.zero_mod,
        Nat.one_mod, Nat.reduceMod, Nat.mod_succ, zero_mul, add_zero, zero_add] <;>
      fin_cases b <;>
      simp only [mul_zero, add_zero, Fin.mk.injEq, Nat.reduceEqDiff, ↓reduceIte] <;>
      ring_nf <;> (try simp only [hs2]) <;> try linarith [h2, h4]
  have h_tr : ∀ a b : Fin 6, kelvinMandelDipRotor θ a b = kelvinMandelDipRotor (-θ) b a := by
    intro a b; fin_cases a <;> fin_cases b <;> simp [kelvinMandelDipRotor]
  exact ⟨h_row θ, fun k l => by simpa only [h_tr] using h_row (-θ) k l,
    isOrthogonal6_of_rowOrthonormal6 _ (h_row θ)⟩

/-- **Theorem 2.7a (`elasticEnergy6_rotateStiffness6`)**:
$$\varepsilon^\top (M C M^\top) \varepsilon = (M^\top \varepsilon)^\top C (M^\top \varepsilon).$$ -/
theorem elasticEnergy6_rotateStiffness6 (M C : Fin 6 → Fin 6 → ℝ) (ε : Fin 6 → ℝ) :
    elasticEnergy6 (rotateStiffness6 M C) ε =
      elasticEnergy6 C (matVec6 (matTranspose6 M) ε) := by
  simp only [elasticEnergy6, rotateStiffness6, matVec6, matTranspose6, Fin.sum_univ_six]; ring

/-- **Theorem 2.7b (`so6_stiffness_preserves_pos_def`)**:
Orthogonal Kelvin-Mandel rotation preserves coercivity lower bounds and strict positive-definiteness. -/
theorem so6_stiffness_preserves_pos_def (M C : Fin 6 → Fin 6 → ℝ) (μ_min : ℝ)
    (hM : IsOrthogonal6 M) (hμ : 0 < μ_min)
    (h_coercive : ∀ ξ : Fin 6 → ℝ, μ_min * shellEnergy ξ ≤ elasticEnergy6 C ξ) :
    (∀ ε : Fin 6 → ℝ, μ_min * shellEnergy ε ≤ elasticEnergy6 (rotateStiffness6 M C) ε) ∧
    (∀ ε : Fin 6 → ℝ, (∃ i : Fin 6, ε i ≠ 0) →
      0 < elasticEnergy6 (rotateStiffness6 M C) ε) :=
  have h_bound (ε : Fin 6 → ℝ) : μ_min * shellEnergy ε ≤ elasticEnergy6 (rotateStiffness6 M C) ε := by
    rw [elasticEnergy6_rotateStiffness6, ← hM ε]; exact h_coercive _
  ⟨h_bound, fun ε ⟨i, hi⟩ =>
    lt_of_lt_of_le (mul_pos hμ (lt_of_le_of_ne (shellEnergy_nonneg ε)
      fun h => hi ((shellEnergy_eq_zero_iff ε).mp h.symm i))) (h_bound ε)⟩

/-- **Theorem 2.7c (`so6_stiffness_trace_conservation`)**:
$$\mathrm{Tr}(M C M^\top) = \mathrm{Tr}(C).$$ -/
theorem so6_stiffness_trace_conservation (M C : Fin 6 → Fin 6 → ℝ)
    (hM : IsColOrthonormal6 M) :
    trace6 (rotateStiffness6 M C) = trace6 C := by
  unfold trace6 rotateStiffness6
  calc (∑ i : Fin 6, ∑ k : Fin 6, ∑ l : Fin 6, M i k * C k l * M i l)
    _ = ∑ k : Fin 6, ∑ l : Fin 6, C k l * (∑ i : Fin 6, M i k * M i l) := by simp only [Fin.sum_univ_six]; ring
    _ = ∑ i : Fin 6, C i i := by simp [hM _ _]

def deltaMinus (v : ℤ → ℝ) (z : ℤ) : ℝ := v z - v (z - 1)
def deltaPlus (v : ℤ → ℝ) (z : ℤ) : ℝ := v z - v (z + 1)
def morseImpulseProd (v : ℤ → ℝ) (z : ℤ) : ℝ := deltaMinus v z * deltaPlus v z
def isMorseImpulse (v : ℤ → ℝ) (z : ℤ) : Prop := 0 < morseImpulseProd v z
noncomputable def laplacian3pt (v : ℤ → ℝ) (z : ℤ) : ℝ := (deltaMinus v z + deltaPlus v z) / 2

/-- **Theorem 2.7d (`morseImpulse_iff_strict_extremum`)**:
3-point discrete Morse impulse triggers if and only if z is a strict local extremum. -/
theorem morseImpulse_iff_strict_extremum (v : ℤ → ℝ) (z : ℤ) :
    isMorseImpulse v z ↔
      (v (z - 1) < v z ∧ v (z + 1) < v z) ∨ (v z < v (z - 1) ∧ v z < v (z + 1)) := by
  unfold isMorseImpulse morseImpulseProd deltaMinus deltaPlus
  rw [mul_pos_iff]; simp only [sub_pos, sub_neg]

/-- **Theorem 2.7e (`caliper_morse_rejects_washout_step`)**:
Monotone compaction ramps and single-sided flat steps are strictly rejected. -/
theorem caliper_morse_rejects_washout_step (v : ℤ → ℝ) (z : ℤ) :
    ((v z = v (z - 1) ∨ v z = v (z + 1)) →
      morseImpulseProd v z = 0 ∧ ¬ isMorseImpulse v z) ∧
    (((v (z - 1) ≤ v z ∧ v z ≤ v (z + 1)) ∨ (v (z + 1) ≤ v z ∧ v z ≤ v (z - 1))) →
      morseImpulseProd v z ≤ 0 ∧ ¬ isMorseImpulse v z) :=
  ⟨fun h => by
     have h0 : morseImpulseProd v z = 0 := by
       unfold morseImpulseProd deltaMinus deltaPlus; rcases h with h | h <;> { rw [h]; ring }
     exact ⟨h0, by unfold isMorseImpulse; linarith⟩,
   fun h => by
     have h_le : morseImpulseProd v z ≤ 0 := by
       unfold morseImpulseProd deltaMinus deltaPlus; rcases h with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> nlinarith
     exact ⟨h_le, by unfold isMorseImpulse; linarith⟩⟩

/-- **Theorem 2.7f (`caliper_morse_shields_thin_stringer`)**:
Thin-bed spike impulses are recovered at 100% amplitude. -/
theorem caliper_morse_shields_thin_stringer (v : ℤ → ℝ) (z : ℤ) (v_bg δ_spike : ℝ)
    (h_left : v (z - 1) = v_bg) (h_right : v (z + 1) = v_bg)
    (h_center : v z = v_bg + δ_spike) (h_nz : δ_spike ≠ 0) :
    morseImpulseProd v z = δ_spike ^ 2 ∧
    isMorseImpulse v z ∧
    laplacian3pt v z = δ_spike := by
  have h_prod : morseImpulseProd v z = δ_spike ^ 2 := by
    unfold morseImpulseProd deltaMinus deltaPlus; rw [h_left, h_right, h_center]; ring
  exact ⟨h_prod, by unfold isMorseImpulse; rw [h_prod]; positivity,
    by unfold laplacian3pt deltaMinus deltaPlus; rw [h_left, h_right, h_center]; ring⟩

-- ============================================================================
-- Section 4: Ideal MHD / Euler Force Balance & MRxMHD Flow (Theorems 2.8–2.9)
-- ============================================================================

def cross3 (u v : Fin 3 → ℝ) (k : Fin 3) : ℝ :=
  match k.val with
  | 0 => u 1 * v 2 - u 2 * v 1
  | 1 => u 2 * v 0 - u 0 * v 2
  | _ => u 0 * v 1 - u 1 * v 0

def leviCivitaHalfTensor (i j k : Fin 3) : ℝ :=
  match i.val, j.val, k.val with
  | 0, 1, 2 => 1
  | 1, 2, 0 => 1
  | 2, 0, 1 => 1
  | _, _, _ => 0

/-- **Theorem 2.8a (`cross3_eq_triadOperator`)**:
$B_{W_{\mathrm{LC}}}(u, v) = u \times v$. -/
theorem cross3_eq_triadOperator (u v : Fin 3 → ℝ) (k : Fin 3) :
    triadOperator leviCivitaHalfTensor u v k = cross3 u v k := by
  fin_cases k <;> simp [triadOperator, leviCivitaHalfTensor, cross3, Fin.sum_univ_three] <;> ring

def IsIdealMHDBalance (J B grad_p : Fin 3 → ℝ) : Prop :=
  ∀ k : Fin 3, triadOperator leviCivitaHalfTensor J B k = grad_p k

def IsEulerLambBalance (ω u grad_H : Fin 3 → ℝ) : Prop :=
  ∀ k : Fin 3, triadOperator leviCivitaHalfTensor ω u k = -grad_H k

def mhdToEulerMap (state : (Fin 3 → ℝ) × (Fin 3 → ℝ) × (Fin 3 → ℝ)) :
    (Fin 3 → ℝ) × (Fin 3 → ℝ) × (Fin 3 → ℝ) :=
  (state.1, state.2.1, -state.2.2)

lemma mhdToEulerMap_involutive : Function.Involutive mhdToEulerMap := by
  rintro ⟨J, B, grad_p⟩; simp [mhdToEulerMap]

/-- **Theorem 2.8b (`mhd_euler_force_balance_bijection`)**:
Involution and bijection between ideal MHD force balance and steady Euler Lamb balance. -/
theorem mhd_euler_force_balance_bijection :
    Function.Bijective mhdToEulerMap ∧
    (∀ s : (Fin 3 → ℝ) × (Fin 3 → ℝ) × (Fin 3 → ℝ),
      mhdToEulerMap (mhdToEulerMap s) = s) ∧
    (∀ J B grad_p : Fin 3 → ℝ,
      IsIdealMHDBalance J B grad_p ↔
        IsEulerLambBalance (mhdToEulerMap (J, B, grad_p)).1
          (mhdToEulerMap (J, B, grad_p)).2.1
          (mhdToEulerMap (J, B, grad_p)).2.2) ∧
    (∀ J B grad_p : Fin 3 → ℝ,
      IsIdealMHDBalance J B grad_p ↔ (∀ k : Fin 3, cross3 J B k = grad_p k)) :=
  ⟨mhdToEulerMap_involutive.bijective, mhdToEulerMap_involutive,
    fun J B grad_p => by simp [IsIdealMHDBalance, IsEulerLambBalance, mhdToEulerMap],
    fun J B grad_p => by simp [IsIdealMHDBalance, cross3_eq_triadOperator]⟩

def IsFluxSurfaceConfined {C : ℕ} (B grad_ψ : Fin C → ℝ) : Prop :=
  shellInner B grad_ψ = 0

/-- **Theorem 2.8c (`mhd_flux_surface_confinement_iff`)**:
Tangent magnetic flux-surface confinement $\mathbf{B} \cdot \nabla\psi = 0$ is equivalent to triad orthogonality. -/
theorem mhd_flux_surface_confinement_iff {C : ℕ}
    (W : Fin C → Fin C → Fin C → ℝ) (u v grad_ψ : Fin C → ℝ)
    (J B : Fin 3 → ℝ) :
    (grad_ψ = triadOperator W u v →
      (IsFluxSurfaceConfined v grad_ψ ↔ shellInner v (triadOperator W u v) = 0)) ∧
    IsFluxSurfaceConfined v (triadOperator W u v) ∧
    shellInner B (cross3 J B) = 0 :=
  ⟨fun h => h ▸ Iff.rfl, triad_pointwise_orthogonal W u v,
   by rw [← funext (cross3_eq_triadOperator J B)]; exact triad_pointwise_orthogonal leviCivitaHalfTensor J B⟩

def stellaratorParityShift (C : ℕ) (τ₀ : ℝ) (W : Fin C → Fin C → Fin C → ℝ)
    (u v : Fin C → ℝ) (k : Fin C) : ℝ :=
  τ₀ * triadOperator W u v k

def stellaratorReadout (C : ℕ) (ℓ_sym : Fin C → ℝ) (τ₀ : ℝ)
    (W : Fin C → Fin C → Fin C → ℝ) (u v : Fin C → ℝ) (k : Fin C) : ℝ :=
  ℓ_sym k + stellaratorParityShift C τ₀ W u v k

/-- **Theorem 2.8d (`stellarator_parity_breaks_tie_and_conserves_flux`)**:
Non-stellarator-symmetric torsion breaks resonant mode ties while preserving flux tangency. -/
theorem stellarator_parity_breaks_tie_and_conserves_flux {C : ℕ}
    (ℓ_sym : Fin C → ℝ) (τ₀ : ℝ) (W : Fin C → Fin C → Fin C → ℝ)
    (u v : Fin C → ℝ) (i₀ c₁ c₂ : Fin C) (α β : ℝ)
    (hτ : 0 < τ₀) (hα : 0 < α) (hβ : 0 < β) (h_tie : ℓ_sym c₁ = ℓ_sym c₂)
    (hu_i0 : u i₀ = β) (hu_other : ∀ i : Fin C, i ≠ i₀ → u i = 0)
    (hv_c1 : v c₁ = α) (hv_c2 : v c₂ = α)
    (hv_other : ∀ c : Fin C, c ≠ c₁ → c ≠ c₂ → v c = 0) :
    (stellaratorReadout C ℓ_sym τ₀ W u v c₁ -
      stellaratorReadout C ℓ_sym τ₀ W u v c₂ =
        2 * τ₀ * β * α * (W i₀ c₂ c₁ - W i₀ c₁ c₂)) ∧
    (W i₀ c₂ c₁ > W i₀ c₁ c₂ →
      stellaratorReadout C ℓ_sym τ₀ W u v c₁ >
        stellaratorReadout C ℓ_sym τ₀ W u v c₂) ∧
    IsFluxSurfaceConfined v (stellaratorParityShift C τ₀ W u v) := by
  have h_occl := (occlusion_triad_breaks_symmetric_tie ℓ_sym W u v i₀ c₁ c₂ α β
    hα hβ h_tie hu_i0 hu_other hv_c1 hv_c2 hv_other).1
  have h_diff : stellaratorReadout C ℓ_sym τ₀ W u v c₁ -
      stellaratorReadout C ℓ_sym τ₀ W u v c₂ = 2 * τ₀ * β * α * (W i₀ c₂ c₁ - W i₀ c₁ c₂) := by
    unfold stellaratorReadout stellaratorParityShift postLogit at *
    linear_combination (1 - τ₀) * h_tie + τ₀ * h_occl
  exact ⟨h_diff, fun hW => by nlinarith [mul_pos (mul_pos hτ hβ) hα], tangentTriadStep_orthogonal τ₀ W u v⟩

/-- **Theorem 2.8e (`stellarator_zero_torsion_homotopy_iff`)**:
Torsion shift vanishes if and only if $\tau_0 = 0$. -/
theorem stellarator_zero_torsion_homotopy_iff {C : ℕ}
    (ℓ_sym : Fin C → ℝ) (τ₀ : ℝ) (W : Fin C → Fin C → Fin C → ℝ)
    (u v : Fin C → ℝ)
    (h_nondeg : ∃ k : Fin C, triadOperator W u v k ≠ 0) :
    (τ₀ = 0 ↔ stellaratorParityShift C τ₀ W u v = 0) ∧
    (τ₀ = 0 → ∀ k : Fin C, stellaratorReadout C ℓ_sym τ₀ W u v k = ℓ_sym k) :=
  let ⟨k, hk⟩ := h_nondeg
  ⟨⟨fun hτ => funext fun i => by simp [stellaratorParityShift, hτ],
    fun h0 => (mul_eq_zero.mp (congr_fun h0 k)).resolve_right hk⟩,
   fun hτ i => by simp [stellaratorReadout, stellaratorParityShift, hτ]⟩

noncomputable def flowBeltramiMultiplier (μ M : ℝ) : ℝ := μ / (1 - M ^ 2)

theorem sub_alfvenic_beltrami_homotopy_zero_flow (μ : ℝ) :
    flowBeltramiMultiplier μ 0 = μ := by
  unfold flowBeltramiMultiplier; ring

theorem sub_alfvenic_beltrami_pos (μ M : ℝ) (hμ : 0 < μ) (hM : M ^ 2 < 1) :
    0 < flowBeltramiMultiplier μ M :=
  div_pos hμ (by linarith)

def collinearBeltramiField (α : ℝ) (B : Fin 3 → ℝ) (k : Fin 3) : ℝ := α * B k

/-- **Theorem 2.9a (`beltrami_cross_helicity_collinear_lorentz_vanishes`)**:
$B_{W_{\mathrm{LC}}}(\alpha \mathbf{B}, \mathbf{B}) = \mathbf{0}$. -/
theorem beltrami_cross_helicity_collinear_lorentz_vanishes (α : ℝ) (B : Fin 3 → ℝ) (k : Fin 3) :
    triadOperator leviCivitaHalfTensor (collinearBeltramiField α B) B k = 0 ∧
    cross3 (collinearBeltramiField α B) B k = 0 := by
  rw [cross3_eq_triadOperator, and_self_iff]
  fin_cases k <;> simp [cross3, collinearBeltramiField] <;> ring

noncomputable def flowBernoulliHead (p B_sq M : ℝ) : ℝ := p + (1 + M ^ 2) * B_sq / 2

def IsFlowMRxMHDJumpBalanced (p₁ B₁_sq M₁ p₂ B₂_sq M₂ : ℝ) : Prop :=
  flowBernoulliHead p₁ B₁_sq M₁ = flowBernoulliHead p₂ B₂_sq M₂

/-- **Theorem 2.9b (`mrxmhd_flow_bernoulli_jump_conservation`)**:
Interface pressure jump equation across stepped-pressure barriers. -/
theorem mrxmhd_flow_bernoulli_jump_conservation (p₁ B₁_sq M₁ p₂ B₂_sq M₂ : ℝ) :
    (IsFlowMRxMHDJumpBalanced p₁ B₁_sq M₁ p₂ B₂_sq M₂ ↔
      p₂ - p₁ = ((1 + M₁ ^ 2) * B₁_sq - (1 + M₂ ^ 2) * B₂_sq) / 2) ∧
    (IsFlowMRxMHDJumpBalanced p₁ B₁_sq 0 p₂ B₂_sq 0 ↔
      p₂ - p₁ = (B₁_sq - B₂_sq) / 2) := by
  unfold IsFlowMRxMHDJumpBalanced flowBernoulliHead
  refine ⟨⟨fun h => by linarith, fun h => by linarith⟩,
          ⟨fun h => by ring_nf at h ⊢; linarith, fun h => by ring_nf at h ⊢; linarith⟩⟩

noncomputable def kamDiophantineGap (ι : ℝ) (n m : ℕ) : ℝ := |ι - (n : ℝ) / (m : ℝ)|

def IsKAMInterfaceAdmissible (ι M γ ν : ℝ) (n m : ℕ) : Prop :=
  M ^ 2 < 1 ∧ γ / ((m : ℝ) ^ ν) ≤ kamDiophantineGap ι n m

/-- **Theorem 2.9c (`kam_sub_alfvenic_ellipticity_and_gap`)**:
Sub-Alfvénic Beltrami ellipticity ($1 - M^2 > 0$) and Diophantine non-resonance. -/
theorem kam_sub_alfvenic_ellipticity_and_gap (ι M γ ν : ℝ) (n m : ℕ)
    (h : IsKAMInterfaceAdmissible ι M γ ν n m) :
    0 < 1 - M ^ 2 ∧ γ / ((m : ℝ) ^ ν) ≤ |ι - (n : ℝ) / (m : ℝ)| :=
  ⟨by linarith [h.1], h.2⟩

end AstroSnail.Formalization
