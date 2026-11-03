"""
tests/empirical_m3_2_stress_runner.py
=====================================
Quantitative empirical stress harness for Challenger M3-2:
1. Scenario 01 Glint Stress & Continuous Stream Invariance
2. Scenario 05 RSD Smearing Error Boundary Challenge
"""

import json
import math
import sys
import time
import numpy as np
import scipy.ndimage

from astrosnail.transients import (
    MorseTransientGateConfig,
    compute_morse_impulse,
)
from astrosnail.benchmark import (
    UniformBinningSigmaClip,
    StandardDampedLSQR_RSD,
    evaluate_transient_recall,
    evaluate_glint_false_triggers,
    evaluate_rsd_smearing_error,
)
from astrosnail.cosmology import blind_astro_let_decouple
from astrosnail.scenarios import (
    generate_scenario_01_frb_supernova,
    generate_scenario_05_cosmic_web,
)
from astrosnail.engine import AstroSnailEngine


def run_scenario_01_glint_fuzzing(n_glints_list=(1000, 2500, 5000)):
    print("\n" + "=" * 70)
    print("1. SCENARIO 01: GLINT STRESS & CONTINUOUS STREAM FUZZING")
    print("=" * 70)

    results = {}

    for n_glints in n_glints_list:
        rng = np.random.default_rng(20261103 + n_glints)
        n_epochs = 50000
        noise_std = 1.0
        stream = rng.normal(0.0, noise_std, size=n_epochs).astype(np.float32)

        glint_indices = rng.choice(n_epochs, size=n_glints, replace=False)
        glint_indices.sort()
        separated = []
        last = -10
        for idx in glint_indices:
            if idx > last + 2 and idx < n_epochs - 2:
                separated.append(idx)
                last = idx

        actual_glints = len(separated)
        for idx in separated:
            amp = float(rng.uniform(8.0, 50.0) * rng.choice([-1.0, 1.0]))
            stream[idx] += amp

        # AstroSnail Gate
        as_gate = MorseTransientGateConfig(sigma_thresh=4.0, persistence_window=1)
        as_false = 0
        as_glint_rejected = 0
        for val in stream:
            c, _ = as_gate.update(float(val))
            if c == "confirmed_transient":
                as_false += 1
            elif c == "glint_rejected":
                as_glint_rejected += 1

        # Baseline Clip
        base_clip = UniformBinningSigmaClip(window_size=16, sigma_thresh=4.0)
        base_false = 0
        for val in stream:
            st, _ = base_clip.update(float(val))
            if st == "spike_detected":
                base_false += 1

        rate_as = evaluate_glint_false_triggers(as_false, actual_glints)
        rate_base = evaluate_glint_false_triggers(base_false, actual_glints)

        print(f"\n[Spike Density: {actual_glints} Poisson glints over 50,000 samples]")
        print(f"  AstroSnail False Triggers: {as_false} (rate per 10k: {rate_as:.1f})")
        print(f"  AstroSnail Glint Rejections: {as_glint_rejected} ({as_glint_rejected / actual_glints * 100:.1f}%)")
        print(f"  Baseline False Triggers:   {base_false} (rate per 10k: {rate_base:.1f})")
        print(f"  False Trigger Suppression: {(1.0 - as_false / max(1, base_false)) * 100:.2f}%")

        results[f"glints_{actual_glints}"] = {
            "actual_glints": actual_glints,
            "as_false": as_false,
            "as_rate_per_10k": rate_as,
            "as_glint_rejected": as_glint_rejected,
            "base_false": base_false,
            "base_rate_per_10k": rate_base,
            "suppression_pct": (1.0 - as_false / max(1, base_false)) * 100,
        }

    return results


def run_scenario_01_pathological_streams():
    print("\n--- Pathological Signals (Zero, Constant, NaNs, Dropouts) ---")
    results = {}

    # All zeros
    zeros = np.zeros(25000, dtype=np.float32)
    as_g = MorseTransientGateConfig(4.0)
    as_zeros = sum(1 for v in zeros if as_g.update(float(v))[0] == "confirmed_transient")
    print(f"  All-zero stream (25k samples): confirmed transients = {as_zeros}, noise_std = {as_g.noise_std}")

    # Constant
    consts = np.full(25000, 42.0, dtype=np.float32)
    as_g = MorseTransientGateConfig(4.0)
    as_const = sum(1 for v in consts if as_g.update(float(v))[0] == "confirmed_transient")
    print(f"  Constant 42.0 stream (25k samples): confirmed transients = {as_const}, noise_std = {as_g.noise_std}")

    # NaNs and recovery
    nans = np.random.default_rng(123).normal(0.0, 1.0, size=5000).astype(np.float32)
    nans[500:550] = np.nan
    nans[2000:2020] = np.inf
    as_g = MorseTransientGateConfig(4.0)
    nan_classes = [as_g.update(float(v))[0] for v in nans]
    nan_healthy = np.isfinite(as_g.noise_std) and as_g.noise_std > 0
    print(f"  NaN/Inf stream (5k samples with 50 NaNs & 20 Infs): healthy = {nan_healthy}, final noise_std = {as_g.noise_std:.4f}")

    results["zeros_triggers"] = as_zeros
    results["const_triggers"] = as_const
    results["nan_healthy"] = nan_healthy
    return results


def run_scenario_01_blind_frb_recall(n_trials=30):
    print("\n--- Genuine FRB Recall without Ground Truth Peeking ---")
    engine = AstroSnailEngine(backend="numpy")
    snr_levels = [20.0, 18.0, 15.0, 12.0, 10.0]
    results = {}

    for snr in snr_levels:
        as_recalls = []
        base_recalls = []
        for tr in range(n_trials):
            seed = 20261103 + tr * 100 + int(snr * 10)
            rng = np.random.default_rng(seed)
            sc = generate_scenario_01_frb_supernova(seed=seed, n_epochs=25000)
            flux = sc.data.copy()

            if snr < 20.0:
                p_sig = float(np.var(sc.ground_truth["clean_transient_signal"]))
                p_noise = p_sig * (10.0 ** (-snr / 10.0))
                flux += rng.normal(0.0, math.sqrt(max(1e-8, p_noise)), size=flux.shape).astype(np.float32)

            # Blind AstroSnail Morse impulse detection
            u_c, u_f, sal = engine.shell_splitter.decompose(flux)
            i_morse = compute_morse_impulse(flux, salience=sal)
            th_morse = np.median(i_morse) + 3.0 * np.std(i_morse)
            as_det = np.where(i_morse > th_morse)[0]
            r_as = evaluate_transient_recall(as_det, [sc.ground_truth["frb_epoch"]], tolerance=15)

            # Baseline sigma clipping
            base_det = np.where(flux > np.mean(flux) + 4.0 * np.std(flux))[0]
            r_base = evaluate_transient_recall(base_det, [sc.ground_truth["frb_epoch"]], tolerance=15)

            as_recalls.append(r_as)
            base_recalls.append(r_base)

        m_as = float(np.mean(as_recalls))
        m_base = float(np.mean(base_recalls))
        win = m_as >= m_base
        print(f"  SNR {snr:4.1f} dB ({n_trials} trials): AstroSnail Recall = {m_as * 100:.1f}%, Baseline Recall = {m_base * 100:.1f}% -> {'AS WIN' if win else 'BASE WIN'}")
        results[f"{snr}_db"] = {"as_recall": m_as, "base_recall": m_base, "win": win}

    return results


def run_scenario_05_rsd_boundary_sweep():
    print("\n" + "=" * 70)
    print("2. SCENARIO 05: RSD SMEARING ERROR & MASS CONSERVATION BOUNDARY CHALLENGE")
    print("=" * 70)

    results = {}

    # 2.1 Velocity dispersion sweep sigma in [0.5, 3.0]
    print("\n--- 2.1 Line-of-Sight Velocity Dispersion Sweep sigma in [0.5, 3.0] ---")
    sigmas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    sc_data = generate_scenario_05_cosmic_web(seed=20261025, grid_shape=(32, 24, 24))
    gt_delta = sc_data.ground_truth["density_true"]
    gt_clean = gt_delta - float(np.mean(gt_delta))

    dispersion_results = []
    for sig in sigmas:
        rsd_distorted = scipy.ndimage.gaussian_filter1d(gt_clean, sigma=sig, axis=0)
        real_as, v_as = blind_astro_let_decouple(rsd_distorted)
        real_base, v_base = StandardDampedLSQR_RSD(damping=0.15).decouple(rsd_distorted)

        smear_as = evaluate_rsd_smearing_error(real_as, gt_clean)
        smear_base = evaluate_rsd_smearing_error(real_base, gt_clean)
        as_wins = smear_as < smear_base

        print(f"  sigma={sig:4.2f}: Smear AS = {smear_as:.4f}, Smear Base = {smear_base:.4f} -> {'AS WIN' if as_wins else 'BASE WIN'}")
        dispersion_results.append({
            "sigma": sig,
            "smear_as": smear_as,
            "smear_base": smear_base,
            "as_wins": as_wins,
        })

    results["dispersion_sweep"] = dispersion_results

    # 2.2 Strict Mass Conservation \int \rho dV
    print("\n--- 2.2 Strict Mass Conservation Verification (\int rho dV) ---")
    shapes = [(32, 24, 24), (64, 16, 16), (20, 20, 20), (128, 8, 8), (17, 19, 23)]
    mass_results = []
    for sh in shapes:
        rng = np.random.default_rng(42)
        delta = np.maximum(-0.95, rng.normal(0.0, 1.2, size=sh).astype(np.float32))
        rho = 1.0 + delta
        real_rho, _ = blind_astro_let_decouple(rho)

        m_init = float(np.sum(rho))
        m_rec = float(np.sum(real_rho))
        rel_diff = abs(m_rec - m_init) / max(1e-6, abs(m_init))
        conserved = rel_diff < 1e-5

        print(f"  Grid {sh}: Initial Mass = {m_init:.4f}, Reconstructed = {m_rec:.4f}, Rel Diff = {rel_diff:.2e} -> {'CONSERVED' if conserved else 'VIOLATED'}")
        mass_results.append({
            "shape": sh,
            "rel_diff": rel_diff,
            "conserved": conserved,
        })

    results["mass_conservation"] = mass_results

    # 2.3 Diverse grid geometries and degenerate shapes
    print("\n--- 2.3 Diverse Geometries & Boundary Shapes ---")
    geom_shapes = [(1, 1, 1), (2, 2, 2), (5, 7, 11), (64, 32, 16), (128, 8, 8)]
    geom_ok = True
    for sh in geom_shapes:
        arr = np.random.default_rng(7).normal(0.0, 1.0, size=sh).astype(np.float32)
        r, v = blind_astro_let_decouple(arr)
        if not (r.shape == sh and v.shape == sh and np.all(np.isfinite(r)) and np.all(np.isfinite(v))):
            geom_ok = False
            print(f"  FAIL on geometry {sh}")
        else:
            print(f"  Geometry {sh}: PASS (finite, shape preserved)")

    results["geom_ok"] = geom_ok
    return results


def main_suite():
    t0 = time.time()
    res1 = run_scenario_01_glint_fuzzing()
    res2 = run_scenario_01_pathological_streams()
    res3 = run_scenario_01_blind_frb_recall()
    res4 = run_scenario_05_rsd_boundary_sweep()
    t1 = time.time()

    print("\n" + "=" * 70)
    print(f"CHALLENGER M3-2 EMPIRICAL STRESS COMPLETE IN {t1 - t0:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main_suite()
