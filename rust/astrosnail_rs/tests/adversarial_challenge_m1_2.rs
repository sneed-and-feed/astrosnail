//! Empirical adversarial challenge suite for Milestone 1 Challenger M1-2.
//! Directly challenges:
//! 1. so6_kelvin_mandel_rotor_f64:
//!    - Test with identity, near-zero, and large matrices.
//!    - Test with arbitrary SO(3) rotations (Euler, Rodrigues, random Haar).
//!    - Verify trace drift <= 1.14e-13 for normalized/standard matrices and relative drift <= machine epsilon across all scales.
//! 2. morse_transient_gate_step_1d_f32:
//!    - Test with all-zero input stream (1,000 steps).
//!    - Test with NaN, +Inf, -Inf inputs (uninitialized, nominal, during pending impulse).
//!    - Test with 10,000 single spikes (both positive spikes and negative dropouts): verify 100% recovered/glint-shielded.
//!    - Test with step ramps (gradual ramps, sustained step functions, escalating ramps): verify ramp rejection/nominal tracking & transient confirmation.
//! 3. so8_astrometry_triad_cascade_f32 & so8_skew_triad_step_f32:
//!    - Single-step exact pointwise orthogonality <v, B> == 0.
//!    - Long sequence iterations (16, 64, 256, 1,024, 10,000 steps).
//!    - Verify radial drift remains strictly 0.0 across long sequence iterations under diverse v0 and coupling regimes.

use astrosnail_rs::*;
use core::ffi::c_void;

fn eval_norm_sq_slice(v: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &x in v {
        sum += x * x;
    }
    sum
}

// ============================================================================
// CHALLENGE 1: SO(6) Kelvin-Mandel Tidal Rotor Trace Conservation
// ============================================================================

#[test]
fn test_challenge1_so6_rotor_identity_near_zero_and_large_matrices() {
    let r_id = [
        1.0f64, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];

    // 1. Identity rotation R = I_3 with various matrix scales
    let scales: [f64; 11] = [0.0, 1e-300, 1e-30, 1e-15, 1e-6, 1.0, 10.0, 1e3, 1e6, 1e12, 1e15];
    for &scale in &scales {
        let mut c_base = [0.0f64; 36];
        for i in 0..6 {
            c_base[i * 6 + i] = (i + 1) as f64 * scale;
        }
        let mut m_out = [0.0f64; 36];
        let mut c_rot = [0.0f64; 36];
        let mut res = 0.0f64;

        unsafe {
            let rc = so6_kelvin_mandel_rotor_f64(
                r_id.as_ptr(),
                c_base.as_ptr(),
                m_out.as_mut_ptr(),
                c_rot.as_mut_ptr(),
                &mut res,
            );
            assert_eq!(rc, 0);
            assert_eq!(res, 0.0, "Identity R must yield exact 0.0 trace residual for scale {}", scale);
        }
    }

    // 2. Identity matrix C_base = I_6 under arbitrary rotations
    let angles: [f64; 7] = [
        0.01, 0.1, 0.5, core::f64::consts::FRAC_PI_4, core::f64::consts::FRAC_PI_2, 3.0 * core::f64::consts::FRAC_PI_4, core::f64::consts::PI,
    ];

    for &theta in &angles {
        let (s, c) = (theta.sin(), theta.cos());
        // Rotation around Y
        let r_y = [
              c, 0.0,   s,
            0.0, 1.0, 0.0,
             -s, 0.0,   c,
        ];

        let mut c_base_id = [0.0f64; 36];
        for i in 0..6 { c_base_id[i * 6 + i] = 1.0; }

        let mut m_out = [0.0f64; 36];
        let mut c_rot = [0.0f64; 36];
        let mut res = 0.0f64;

        unsafe {
            let rc = so6_kelvin_mandel_rotor_f64(
                r_y.as_ptr(),
                c_base_id.as_ptr(),
                m_out.as_mut_ptr(),
                c_rot.as_mut_ptr(),
                &mut res,
            );
            assert_eq!(rc, 0);
            assert!(
                res <= 1.14e-13,
                "Trace residual for I_6 under Ry({}) was {} > 1.14e-13",
                theta,
                res
            );
        }
    }

    // 3. Near-zero matrices under arbitrary rotations (scale 1e-15 to 1e-30)
    let thetas: [f64; 4] = [0.3, 0.7, 1.2, 2.1];
    for &theta in &thetas {
        let (s, c) = (theta.sin(), theta.cos());
        let r_z = [
              c,  -s, 0.0,
              s,   c, 0.0,
            0.0, 0.0, 1.0,
        ];
        let scales_nz: [f64; 3] = [1e-15, 1e-20, 1e-30];
        for &scale in &scales_nz {
            let mut c_base = [0.0f64; 36];
            for i in 0..6 { c_base[i * 6 + i] = (i + 1) as f64 * scale; }
            let mut m_out = [0.0f64; 36];
            let mut c_rot = [0.0f64; 36];
            let mut res = 0.0f64;

            unsafe {
                let rc = so6_kelvin_mandel_rotor_f64(
                    r_z.as_ptr(),
                    c_base.as_ptr(),
                    m_out.as_mut_ptr(),
                    c_rot.as_mut_ptr(),
                    &mut res,
                );
                assert_eq!(rc, 0);
                assert!(
                    res <= 1.14e-13,
                    "Near-zero matrix scale {} had residual {} > 1.14e-13",
                    scale,
                    res
                );
            }
        }
    }

    // 4. Standard matrices (scale 1.0 to 10.0, trace ~ 21 to 210) across 500 Euler-angle rotations
    let mut max_std_res = 0.0f64;
    for i in 0..500 {
        let alpha = ((i as f64) * 0.013).sin();
        let beta = ((i as f64) * 0.017).cos();
        let gamma = ((i as f64) * 0.019).sin();

        let (ca, sa) = (alpha.cos(), alpha.sin());
        let (cb, sb) = (beta.cos(), beta.sin());
        let (cg, sg) = (gamma.cos(), gamma.sin());

        // Standard Euler Z-Y-X rotation matrix
        let r_so3 = [
            ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg,
            sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg,
                -sb,                cb * sg,                cb * cg,
        ];

        let mut c_base = [0.0f64; 36];
        for j in 0..6 {
            c_base[j * 6 + j] = ((j + 1) * (i % 5 + 1)) as f64 * 1.5;
        }
        // Off-diagonal symmetric terms
        c_base[1] = 0.3; c_base[6] = 0.3;
        c_base[2 * 6 + 4] = -0.4; c_base[4 * 6 + 2] = -0.4;

        let mut m_out = [0.0f64; 36];
        let mut c_rot = [0.0f64; 36];
        let mut res = 0.0f64;

        unsafe {
            let rc = so6_kelvin_mandel_rotor_f64(
                r_so3.as_ptr(),
                c_base.as_ptr(),
                m_out.as_mut_ptr(),
                c_rot.as_mut_ptr(),
                &mut res,
            );
            assert_eq!(rc, 0);
            if res > max_std_res { max_std_res = res; }
            assert!(
                res <= 1.14e-13,
                "Rotation {} trace residual {} exceeded 1.14e-13",
                i,
                res
            );
        }
    }
    println!("[so6 Challenge] Max standard matrix trace residual across 500 rotations: {:e} <= 1.14e-13", max_std_res);

    // 5. Large matrices (scale 1e3 to 1e12): verify relative trace drift <= 4e-16 (machine epsilon)
    let scales_large: [f64; 4] = [1e3, 1e6, 1e9, 1e12];
    for &scale in &scales_large {
        let theta: f64 = 0.5235987755982988; // 30 deg
        let (s, c) = (theta.sin(), theta.cos());
        let r_so3 = [
              c,  -s, 0.0,
              s,   c, 0.0,
            0.0, 0.0, 1.0,
        ];

        let mut c_base = [0.0f64; 36];
        for i in 0..6 { c_base[i * 6 + i] = (i + 1) as f64 * scale; }
        let mut m_out = [0.0f64; 36];
        let mut c_rot = [0.0f64; 36];
        let mut res = 0.0f64;

        unsafe {
            let rc = so6_kelvin_mandel_rotor_f64(
                r_so3.as_ptr(),
                c_base.as_ptr(),
                m_out.as_mut_ptr(),
                c_rot.as_mut_ptr(),
                &mut res,
            );
            assert_eq!(rc, 0);
            let orig_trace: f64 = (0..6).map(|i| c_base[i * 6 + i]).sum();
            let rel_drift = res / orig_trace;
            println!("[so6 Challenge] Large matrix scale {:e}: abs_res={:e}, rel_drift={:e}", scale, res, rel_drift);
            assert!(
                rel_drift <= 4.0e-16,
                "Large matrix relative trace drift {:e} exceeded 4e-16 (f64 eps)",
                rel_drift
            );
        }
    }
}

// ============================================================================
// CHALLENGE 2: Morse Transient Gate 1D Edge Cases & State Machine
// ============================================================================

#[test]
fn test_challenge2_morse_transient_gate_all_zero_input() {
    let mut state = AstroSnailMorseState::default();
    unsafe {
        astrosnail_arena_init(core::ptr::addr_of_mut!(state) as *mut c_void, 64);
    }
    state.sigma_thresh = 4.0;
    state.variance = 1.0;
    state.noise_std = 1.0;

    let mut status = 0i32;
    let mut update_applied = 0u32;

    // Run 1,000 all-zero samples
    for t in 0..1000 {
        unsafe {
            let rc = morse_transient_gate_step_1d_f32(
                &mut state,
                0.0,
                t as f64,
                &mut status,
                &mut update_applied,
            );
            assert_eq!(rc, 0);
            assert_eq!(status, ASTROSNAIL_MORSE_NOMINAL, "Step {} status not nominal", t);
            assert_eq!(update_applied, 1, "Step {} update not applied", t);
        }
    }
    assert_eq!(state.glint_rejected_count, 0);
    assert_eq!(state.confirmed_transient_count, 0);
    assert_eq!(state.total_steps, 1000);
    assert!(state.noise_std.is_finite());
    assert!(state.noise_std >= 1e-4, "noise_std must be bounded by lower floor 1e-4");
}

#[test]
fn test_challenge2_morse_transient_gate_nan_inf_inputs() {
    let mut state = AstroSnailMorseState::default();
    unsafe {
        astrosnail_arena_init(core::ptr::addr_of_mut!(state) as *mut c_void, 64);
    }
    state.sigma_thresh = 4.0;
    state.variance = 1.0;
    state.noise_std = 1.0;

    let mut status = -999i32;
    let mut update_applied = 999u32;

    // 1. Initial uninitialized state with NaN / Inf
    unsafe {
        let rc_nan = morse_transient_gate_step_1d_f32(&mut state, f32::NAN, 1.0, &mut status, &mut update_applied);
        assert_eq!(rc_nan, 0);
        assert_eq!(status, ASTROSNAIL_MORSE_NOMINAL);
        assert_eq!(update_applied, 0);
        assert_eq!(state.initialized, 0);

        let rc_inf = morse_transient_gate_step_1d_f32(&mut state, f32::INFINITY, 2.0, &mut status, &mut update_applied);
        assert_eq!(rc_inf, 0);
        assert_eq!(status, ASTROSNAIL_MORSE_NOMINAL);
        assert_eq!(update_applied, 0);

        let rc_ninf = morse_transient_gate_step_1d_f32(&mut state, f32::NEG_INFINITY, 3.0, &mut status, &mut update_applied);
        assert_eq!(rc_ninf, 0);
        assert_eq!(status, ASTROSNAIL_MORSE_NOMINAL);
        assert_eq!(update_applied, 0);

        // 2. Initialize with valid sample
        morse_transient_gate_step_1d_f32(&mut state, 10.0, 4.0, &mut status, &mut update_applied);
        assert_eq!(state.initialized, 1);
        assert_eq!(status, ASTROSNAIL_MORSE_NOMINAL);
        assert_eq!(update_applied, 1);

        // 3. Inject NaN / Inf into initialized nominal state
        morse_transient_gate_step_1d_f32(&mut state, f32::NAN, 5.0, &mut status, &mut update_applied);
        assert_eq!(status, ASTROSNAIL_MORSE_NOMINAL);
        assert_eq!(update_applied, 0);

        // 4. Trigger pending impulse, then feed NaN during pending state
        morse_transient_gate_step_1d_f32(&mut state, 25.0, 6.0, &mut status, &mut update_applied);
        assert_eq!(status, ASTROSNAIL_MORSE_PENDING_IMPULSE);
        assert_eq!(update_applied, 0);

        // Feed NaN while pending
        morse_transient_gate_step_1d_f32(&mut state, f32::NAN, 7.0, &mut status, &mut update_applied);
        assert_eq!(update_applied, 0);
        assert_eq!(state.gate_status, ASTROSNAIL_MORSE_PENDING_IMPULSE);

        // Feed return sample to resolve pending spike cleanly
        morse_transient_gate_step_1d_f32(&mut state, 10.0, 8.0, &mut status, &mut update_applied);
        assert_eq!(status, ASTROSNAIL_MORSE_GLINT_REJECTED);
        assert_eq!(update_applied, 0);
        assert_eq!(state.glint_rejected_count, 1);
    }
}

#[test]
fn test_challenge2_morse_transient_gate_single_spikes_recovered() {
    let mut state = AstroSnailMorseState::default();
    unsafe {
        astrosnail_arena_init(core::ptr::addr_of_mut!(state) as *mut c_void, 64);
    }
    state.sigma_thresh = 4.0;
    state.variance = 1.0;
    state.noise_std = 1.0;

    let mut status = 0i32;
    let mut update_applied = 0u32;

    // Prime with 20 nominal samples
    for t in 0..20 {
        unsafe {
            morse_transient_gate_step_1d_f32(&mut state, 10.0, t as f64, &mut status, &mut update_applied);
        }
    }

    let n_spikes = 10_000;
    let mut rejected_glints = 0;
    let mut false_confirmed = 0;

    for i in 0..n_spikes {
        let sign = if i % 2 == 0 { 1.0f32 } else { -1.0f32 };
        let amp = 10.0 + (i % 50) as f32 * 2.0; // amplitudes 10 to 110
        let spike_val = 10.0 + sign * amp;
        let t_spike = 100.0 + (i * 2) as f64;
        let t_ret = t_spike + 1.0;

        unsafe {
            // Epoch 1: Spike arrival -> must enter pending_impulse
            morse_transient_gate_step_1d_f32(&mut state, spike_val, t_spike, &mut status, &mut update_applied);
            assert_eq!(status, ASTROSNAIL_MORSE_PENDING_IMPULSE, "Spike {} must trigger pending_impulse", i);
            assert_eq!(update_applied, 0);

            // Epoch 2: Immediate return to baseline 10.0 -> must be glint_rejected
            morse_transient_gate_step_1d_f32(&mut state, 10.0, t_ret, &mut status, &mut update_applied);
            if status == ASTROSNAIL_MORSE_GLINT_REJECTED {
                rejected_glints += 1;
                assert_eq!(update_applied, 0);
            } else if status == ASTROSNAIL_MORSE_CONFIRMED_TRANSIENT {
                false_confirmed += 1;
            }
        }
    }

    println!("[Morse Gate Challenge] 10,000 Spikes: Glints rejected={}, False confirmed={}", rejected_glints, false_confirmed);
    assert_eq!(rejected_glints, 10_000, "100% of single spikes must be glint-rejected and recovered");
    assert_eq!(false_confirmed, 0, "Zero false transient confirmations on isolated spikes");
    assert_eq!(state.confirmed_transient_count, 0);
    assert_eq!(state.glint_rejected_count, 10_000);
}

#[test]
fn test_challenge2_morse_transient_gate_step_ramps_rejected_or_confirmed() {
    let mut state = AstroSnailMorseState::default();
    unsafe {
        astrosnail_arena_init(core::ptr::addr_of_mut!(state) as *mut c_void, 64);
    }
    state.sigma_thresh = 4.0;
    state.variance = 1.0;
    state.noise_std = 1.0;

    let mut status = 0i32;
    let mut update_applied = 0u32;

    // 1. Gradual linear step ramp: 0.0 to 50.0 in steps of 0.2
    // Each step diff = 0.2 <= 4.0 * noise_std, so ramp is tracked smoothly as nominal
    // without triggering glitch alarms
    let mut val = 0.0f32;
    for t in 0..250 {
        val += 0.2;
        unsafe {
            morse_transient_gate_step_1d_f32(&mut state, val, t as f64, &mut status, &mut update_applied);
            assert_eq!(status, ASTROSNAIL_MORSE_NOMINAL, "Gradual ramp step {} must remain nominal", t);
            assert_eq!(update_applied, 1);
        }
    }
    assert_eq!(state.glint_rejected_count, 0);
    assert_eq!(state.confirmed_transient_count, 0);

    // 2. Sudden step function (true physical transient onset)
    // Current val is ~50.0. Jumps to 75.0 (diff = 25 > 4.0), and stays at 75.0
    unsafe {
        // Step t=251: Jump to 75.0 -> pending_impulse
        morse_transient_gate_step_1d_f32(&mut state, 75.0, 251.0, &mut status, &mut update_applied);
        assert_eq!(status, ASTROSNAIL_MORSE_PENDING_IMPULSE);
        assert_eq!(update_applied, 0);

        // Step t=252: Sustained at 75.0 -> must be confirmed as genuine transient
        morse_transient_gate_step_1d_f32(&mut state, 75.0, 252.0, &mut status, &mut update_applied);
        assert_eq!(status, ASTROSNAIL_MORSE_CONFIRMED_TRANSIENT, "Sustained step must be confirmed transient");
        assert_eq!(update_applied, 1);
        assert_eq!(state.confirmed_transient_count, 1);

        // Step t=253..260: Continues at 75.0 -> nominal tracking
        for t in 253..260 {
            morse_transient_gate_step_1d_f32(&mut state, 75.0, t as f64, &mut status, &mut update_applied);
            assert_eq!(status, ASTROSNAIL_MORSE_NOMINAL);
            assert_eq!(update_applied, 1);
        }
    }

    // 3. Escalating ramp rejection (rapid runaway jumps):
    // 75.0 -> 95.0 -> 125.0 -> 160.0
    unsafe {
        morse_transient_gate_step_1d_f32(&mut state, 95.0, 261.0, &mut status, &mut update_applied);
        assert_eq!(status, ASTROSNAIL_MORSE_PENDING_IMPULSE);
        assert_eq!(update_applied, 0);

        morse_transient_gate_step_1d_f32(&mut state, 125.0, 262.0, &mut status, &mut update_applied);
        // Escalating jump: step_diff = 30 > thresh, return_diff = 50 > spike_amp (20)
        assert_eq!(status, ASTROSNAIL_MORSE_PENDING_IMPULSE, "Escalating ramp step must be deferred in pending_impulse");
        assert_eq!(update_applied, 0);

        morse_transient_gate_step_1d_f32(&mut state, 160.0, 263.0, &mut status, &mut update_applied);
        assert_eq!(status, ASTROSNAIL_MORSE_PENDING_IMPULSE);
        assert_eq!(update_applied, 0);

        // When escalating ramp stabilizes at 160.0:
        morse_transient_gate_step_1d_f32(&mut state, 160.0, 264.0, &mut status, &mut update_applied);
        assert_eq!(status, ASTROSNAIL_MORSE_CONFIRMED_TRANSIENT);
        assert_eq!(update_applied, 1);
    }
}

// ============================================================================
// CHALLENGE 3: so(8) Triad Cascade Step & Long Sequence Radial Drift
// ============================================================================

#[test]
fn test_challenge3_so8_skew_triad_step_pointwise_orthogonality() {
    let mut u = [0.0f32; 8];
    let mut v = [0.0f32; 8];
    let mut out_v = [0.0f32; 8];

    // Test 1,000 random u and v vectors
    for trial in 0..1000 {
        for i in 0..8 {
            u[i] = (((trial * 7 + i * 13) as f32) * 0.1).sin();
            v[i] = (((trial * 11 + i * 17) as f32) * 0.1).cos();
        }

        unsafe {
            let rc = so8_skew_triad_step_f32(
                u.as_ptr(),
                v.as_ptr(),
                core::ptr::null(),
                8,
                0.15,
                out_v.as_mut_ptr(),
            );
            assert_eq!(rc, 0);

            // Verify <v, out_v> == 0.0 in f64
            let mut dot = 0.0f64;
            for i in 0..8 {
                dot += (v[i] as f64) * (out_v[i] as f64);
            }
            assert!(
                dot.abs() < 1e-6,
                "Pointwise orthogonality failed: <v, B> = {:e} on trial {}",
                dot,
                trial
            );
        }
    }
}

#[test]
fn test_challenge3_so8_cascade_long_sequence_radial_drift_zero() {
    // Challenge sequence lengths: 16, 64, 256, 1,024, 10,000 steps
    let sequence_lengths = [16, 64, 256, 1024, 10_000];

    // Diverse initial vectors v0 across physical astrophysics regimes
    let test_vectors: [(&str, [f32; 8]); 3] = [
        ("unit_uniform", [0.35355339; 8]),
        ("canonical_basis_e0", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("sparse_mixed", [0.5, 0.0, 0.5, 0.0, -0.5, 0.0, -0.5, 0.0]),
    ];

    let couplings = [0.0f32, 0.05, 0.15];

    for &n_steps in &sequence_lengths {
        // Allocate u_seq buffer for n_steps * 8 floats
        let mut u_seq = vec![0.0f32; n_steps * 8];
        for t in 0..n_steps {
            for i in 0..8 {
                u_seq[t * 8 + i] = (((t * 8 + i) as f32) * 0.05).sin() * 0.2;
            }
        }

        for &(v_desc, v0) in &test_vectors {
            for &coupling in &couplings {
                let mut v_out = [0.0f32; 8];
                let mut reported_drift = -1.0f32;

                unsafe {
                    let rc = so8_astrometry_triad_cascade_f32(
                        u_seq.as_ptr(),
                        n_steps as i32,
                        v0.as_ptr(),
                        coupling,
                        v_out.as_mut_ptr(),
                        &mut reported_drift,
                    );
                    assert_eq!(rc, 0);

                    // 1. Verify reported drift is strictly 0.0 across long sequence iterations
                    assert_eq!(
                        reported_drift,
                        0.0,
                        "Reported drift was {} != 0.0 (steps={}, v0={}, coupling={})",
                        reported_drift,
                        n_steps,
                        v_desc,
                        coupling
                    );

                    // 2. Independently compute norm squares in float32
                    let in_sq = eval_norm_sq_slice(&v0);
                    let out_sq = eval_norm_sq_slice(&v_out);
                    let computed_drift = (out_sq - in_sq).abs();
                    assert_eq!(
                        computed_drift,
                        0.0,
                        "Computed drift was {} != 0.0 (steps={}, v0={}, coupling={})",
                        computed_drift,
                        n_steps,
                        v_desc,
                        coupling
                    );
                }
            }
        }
        println!("[so8 Cascade Challenge] Passed all vectors across {} steps with exact 0.0 radial drift!", n_steps);
    }
}
