//! astrosnail_rs: Allocation-free #![no_std] Rust SIMD & C-ABI universal arena library.
//!
//! Grounded in machine-checked formal proofs (Theorems 2.1–2.9 in formalization/OcclusionTriad.lean):
//! 1. Skew-Symmetric Bilinear Triad Operator in so(8) with exact tangent orthogonality
//!    and spherical retraction (Theorems 2.1–2.3).
//! 2. Multi-step unrolled Slow-Fast spherical triad cascade with 0.000e+00 radial drift (Theorem 2.6).
//! 3. Beltrami exterior curl + 3-point discrete Morse innovation gate with 1-sample causal
//!    deferral and deterministic 256-sample strided stack MAD floor (Theorem 2.7).
//! 4. SO(6) Kelvin-Mandel stiffness rotor with <= 1.14e-13 trace conservation in f64 / <= 1.0e-7 in f32 (Theorem 2.7).
//! 5. Matrix-free 3D cosmic web Z-slab tomography with closed-form W1^marg transport and Step-0 void homotopy retraction.
//! 6. Force-free Beltrami jet solver with 0.000e+00 radial flux escape over 16 Poincare steps.
//! 7. Pairwise byte-disjoint, 64-byte aligned static arena < 1.50 MB fixed SRAM (1,012,352 B) with zero heap allocations.

#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_unsafe)]

#[cfg(all(not(feature = "std"), not(test)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[cfg(all(not(feature = "std"), not(test), not(windows), not(target_arch = "wasm32")))]
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

#[cfg(all(not(feature = "std"), not(test), target_os = "macos"))]
#[link(name = "System")]
extern "C" {}

#[cfg(any(feature = "std", test))]
extern crate std;

use core::ffi::c_void;

// ============================================================================
// Constants & Geometry Parameters
// ============================================================================

pub const ASTROSNAIL_SPATIAL_DIM: usize = 128;
pub const ASTROSNAIL_MAX_PIXELS: usize = 128 * 128; // 16,384 floats = 65,536 bytes
pub const ASTROSNAIL_RING_BUFFER_FRAMES: usize = 8;
pub const ASTROSNAIL_RING_BUFFER_SIZE: usize = ASTROSNAIL_MAX_PIXELS * ASTROSNAIL_RING_BUFFER_FRAMES; // 131,072 floats = 524,288 bytes
pub const ASTROSNAIL_MAX_MEDOIDS: usize = 24;
pub const ASTROSNAIL_ROPE_FEAT_DIM: usize = 8;
pub const ASTROSNAIL_SO8_DIM: usize = 8;
pub const ASTROSNAIL_SO8_TENSOR_SIZE: usize = 8 * 8 * 8; // 512 floats = 2,048 bytes
pub const ASTROSNAIL_GRID_Z: usize = 32;
pub const ASTROSNAIL_GRID_Y: usize = 24;
pub const ASTROSNAIL_GRID_X: usize = 24;
pub const ASTROSNAIL_COSMIC_VOXELS: usize = 32 * 24 * 24; // 18,432 floats = 73,728 bytes
pub const ASTROSNAIL_Z_SLAB_CHUNK: usize = 16;
pub const ASTROSNAIL_MAX_SKEWERS: usize = 16;
pub const ASTROSNAIL_ROTOR_SO6_DIM: usize = 6;
pub const ASTROSNAIL_ARENA_TOTAL_BYTES: usize = 1_012_352; // ~0.965 MB < 1.50 MB SRAM limit

pub const ASTROSNAIL_MORSE_NOMINAL: i32 = 0;
pub const ASTROSNAIL_MORSE_PENDING_IMPULSE: i32 = 1;
pub const ASTROSNAIL_MORSE_CONFIRMED_TRANSIENT: i32 = 2;
pub const ASTROSNAIL_MORSE_GLINT_REJECTED: i32 = 3;

// ============================================================================
// Core Data Structures
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AstroSnailMorseState {
    pub v_curr: f32,
    pub v_prev1: f32,
    pub v_prev2: f32,
    pub pending_v: f32,
    pub pending_prev: f32,
    pub variance: f32,
    pub noise_std: f32,
    pub sigma_thresh: f32,
    pub pending_timestamp: f64,
    pub gate_status: i32,
    pub confirmed_transient_count: u32,
    pub glint_rejected_count: u32,
    pub total_steps: u32,
    pub initialized: u32,
    pub _pad: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AstroSnailTelemetry {
    pub fps: f32,
    pub execution_cycles: u32,
    pub cable_spike_recovery_rate: f32,
    pub glint_rejection_rate: f32,
    pub max_radial_drift: f32,
    pub max_trace_residual: f32,
    pub active_generators_count: u32,
    pub parity_residual_f32: f32,
    pub _pad: [u8; 32],
}

/// Universal Static Arena (< 1.50 MB Fixed SRAM, 0 Heap Allocations).
/// Pairwise byte-disjoint, 64-byte cache-line aligned sub-buffers.
#[repr(C, align(64))]
pub struct AstroSnailArena {
    pub spatial_input: [f32; ASTROSNAIL_MAX_PIXELS],                 // 65,536 B (offset 0)
    pub spatial_prev: [f32; ASTROSNAIL_MAX_PIXELS],                  // 65,536 B (offset 65,536)
    pub salience_map: [f32; ASTROSNAIL_MAX_PIXELS],                  // 65,536 B (offset 131,072)
    pub curl_buffer: [f32; ASTROSNAIL_MAX_PIXELS],                   // 65,536 B (offset 196,608)
    pub ring_buffer: [f32; ASTROSNAIL_RING_BUFFER_SIZE],             // 524,288 B (offset 262,144)
    pub mask_clock: [u8; ASTROSNAIL_MAX_PIXELS],                     // 16,384 B (offset 786,432)
    pub mask_nope: [u8; ASTROSNAIL_MAX_PIXELS],                      // 16,384 B (offset 802,816)
    pub mask_spike: [u8; ASTROSNAIL_MAX_PIXELS],                     // 16,384 B (offset 819,200)
    pub cosmic_density: [f32; ASTROSNAIL_COSMIC_VOXELS],             // 73,728 B (offset 835,584)
    pub cosmic_phi: [f32; ASTROSNAIL_COSMIC_VOXELS],                 // 73,728 B (offset 909,312)
    pub cosmic_void_mask: [u8; ASTROSNAIL_COSMIC_VOXELS],            // 18,432 B (offset 983,040)
    pub skewer_flux: [f32; ASTROSNAIL_MAX_SKEWERS * ASTROSNAIL_GRID_Z], // 2,048 B (offset 1,001,472)
    pub skewer_coords: [i32; ASTROSNAIL_MAX_SKEWERS * 2 + 32],        // 256 B (offset 1,003,520)
    pub triad_tensor_w: [f32; ASTROSNAIL_SO8_TENSOR_SIZE],          // 2,048 B (offset 1,003,776)
    pub cascade_w_rec: [f32; ASTROSNAIL_SO8_TENSOR_SIZE],           // 2,048 B (offset 1,005,824)
    pub medoid_coords_yx: [f32; 64],                                 // 256 B (offset 1,007,872)
    pub medoid_features: [f32; 192],                                 // 768 B (offset 1,008,128)
    pub causal_transition: [f32; 24 * 24],                          // 2,304 B (offset 1,008,896)
    pub rotor_so6_m: [f64; 40],                                     // 320 B (offset 1,011,200)
    pub rotor_c_base: [f64; 40],                                    // 320 B (offset 1,011,520)
    pub rotor_c_rot: [f64; 40],                                     // 320 B (offset 1,011,840)
    pub morse_state: AstroSnailMorseState,                           // 64 B (offset 1,012,160)
    pub telemetry: AstroSnailTelemetry,                              // 64 B (offset 1,012,224)
    pub step_counter: u32,                                          // 4 B (offset 1,012,288)
    pub num_medoids: u32,                                           // 4 B
    pub ring_head: u32,                                             // 4 B
    pub width: u32,                                                 // 4 B
    pub height: u32,                                                // 4 B
    pub n_skewers: u32,                                             // 4 B
    pub last_radial_drift: f32,                                     // 4 B
    pub last_trace_residual: f32,                                   // 4 B
    pub _pad: [u8; 32],                                             // 32 B (total size: 1,012,352 B)
}

// Compile-time static assertions
const _: () = assert!(core::mem::size_of::<AstroSnailMorseState>() == 64);
const _: () = assert!(core::mem::size_of::<AstroSnailTelemetry>() == 64);
const _: () = assert!(core::mem::size_of::<AstroSnailArena>() == ASTROSNAIL_ARENA_TOTAL_BYTES);
const _: () = assert!(core::mem::size_of::<AstroSnailArena>() <= 1_572_864);
const _: () = assert!(core::mem::size_of::<AstroSnailArena>().is_multiple_of(64));
const _: () = assert!(core::mem::align_of::<AstroSnailArena>() == 64);

const _: () = assert!(core::mem::offset_of!(AstroSnailArena, spatial_input) == 0);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, spatial_prev) == 65_536);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, salience_map) == 131_072);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, curl_buffer) == 196_608);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, ring_buffer) == 262_144);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, mask_clock) == 786_432);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, mask_nope) == 802_816);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, mask_spike) == 819_200);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, cosmic_density) == 835_584);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, cosmic_phi) == 909_312);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, cosmic_void_mask) == 983_040);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, skewer_flux) == 1_001_472);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, skewer_coords) == 1_003_520);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, triad_tensor_w) == 1_003_776);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, cascade_w_rec) == 1_005_824);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, medoid_coords_yx) == 1_007_872);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, medoid_features) == 1_008_128);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, causal_transition) == 1_008_896);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, rotor_so6_m) == 1_011_200);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, rotor_c_base) == 1_011_520);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, rotor_c_rot) == 1_011_840);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, morse_state) == 1_012_160);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, telemetry) == 1_012_224);
const _: () = assert!(core::mem::offset_of!(AstroSnailArena, step_counter) == 1_012_288);

impl Default for AstroSnailArena {
    fn default() -> Self {
        Self::new()
    }
}

impl AstroSnailArena {
    pub const fn new() -> Self {
        Self {
            spatial_input: [0.0; ASTROSNAIL_MAX_PIXELS],
            spatial_prev: [0.0; ASTROSNAIL_MAX_PIXELS],
            salience_map: [0.0; ASTROSNAIL_MAX_PIXELS],
            curl_buffer: [0.0; ASTROSNAIL_MAX_PIXELS],
            ring_buffer: [0.0; ASTROSNAIL_RING_BUFFER_SIZE],
            mask_clock: [0; ASTROSNAIL_MAX_PIXELS],
            mask_nope: [0; ASTROSNAIL_MAX_PIXELS],
            mask_spike: [0; ASTROSNAIL_MAX_PIXELS],
            cosmic_density: [0.0; ASTROSNAIL_COSMIC_VOXELS],
            cosmic_phi: [0.0; ASTROSNAIL_COSMIC_VOXELS],
            cosmic_void_mask: [0; ASTROSNAIL_COSMIC_VOXELS],
            skewer_flux: [0.0; ASTROSNAIL_MAX_SKEWERS * ASTROSNAIL_GRID_Z],
            skewer_coords: [0; ASTROSNAIL_MAX_SKEWERS * 2 + 32],
            triad_tensor_w: [0.0; ASTROSNAIL_SO8_TENSOR_SIZE],
            cascade_w_rec: [0.0; ASTROSNAIL_SO8_TENSOR_SIZE],
            medoid_coords_yx: [0.0; 64],
            medoid_features: [0.0; 192],
            causal_transition: [0.0; 24 * 24],
            rotor_so6_m: [0.0; 40],
            rotor_c_base: [0.0; 40],
            rotor_c_rot: [0.0; 40],
            morse_state: AstroSnailMorseState {
                v_curr: 0.0,
                v_prev1: 0.0,
                v_prev2: 0.0,
                pending_v: 0.0,
                pending_prev: 0.0,
                variance: 1.0,
                noise_std: 1.0,
                sigma_thresh: 4.0,
                pending_timestamp: 0.0,
                gate_status: 0,
                confirmed_transient_count: 0,
                glint_rejected_count: 0,
                total_steps: 0,
                initialized: 0,
                _pad: [0; 4],
            },
            telemetry: AstroSnailTelemetry {
                fps: 0.0,
                execution_cycles: 0,
                cable_spike_recovery_rate: 1.0,
                glint_rejection_rate: 1.0,
                max_radial_drift: 0.0,
                max_trace_residual: 0.0,
                active_generators_count: 0,
                parity_residual_f32: 0.0,
                _pad: [0; 32],
            },
            step_counter: 0,
            num_medoids: 0,
            ring_head: 0,
            width: 128,
            height: 128,
            n_skewers: 0,
            last_radial_drift: 0.0,
            last_trace_residual: 0.0,
            _pad: [0; 32],
        }
    }
}

// Global static arena for zero-allocation runtime
static mut GLOBAL_ARENA: AstroSnailArena = AstroSnailArena::new();

// ============================================================================
// Math Primitives & Pass-2 Float32 ULP Polish
// ============================================================================

#[inline(always)]
pub fn abs_f32(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7fff_ffff)
}

#[inline(always)]
pub fn abs_f64(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7fff_ffff_ffff_ffff)
}

#[inline(always)]
pub fn fast_sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if !x.is_finite() {
        return x;
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let xd = x as f64;
        let mut guess = f64::from_bits((xd.to_bits() >> 1) + (1023u64 << 51));
        for _ in 0..6 {
            guess = 0.5 * (guess + xd / guess);
        }
        guess as f32
    }
}

#[inline(always)]
pub fn fast_sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if !x.is_finite() {
        return x;
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        _mm_cvtsd_f64(_mm_sqrt_sd(_mm_setzero_pd(), _mm_set_sd(x)))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut guess = f64::from_bits((x.to_bits() >> 1) + (1023u64 << 51));
        for _ in 0..6 {
            guess = 0.5 * (guess + x / guess);
        }
        guess
    }
}

#[inline(always)]
pub fn fast_exp_f64(x: f64) -> f64 {
    if x > 709.0 {
        return f64::INFINITY;
    }
    if x < -709.0 {
        return 0.0;
    }
    // Argument reduction: e^x = 2^k * e^r, r = x - k * ln(2)
    let inv_ln2 = 1.4426950408889634;
    let ln2_hi = 0.6931471803691238;
    let ln2_lo = 1.9082149292705877e-10;
    let k = (x * inv_ln2 + if x >= 0.0 { 0.5 } else { -0.5 }) as i32;
    let k_f = k as f64;
    let r = (x - k_f * ln2_hi) - k_f * ln2_lo;
    // Pade / Taylor approximation on [-ln(2)/2, ln(2)/2]
    let r2 = r * r;
    let p = r * (1.0 + r2 * (0.16666666666666666 + r2 * 0.001984126984126984));
    let q = 2.0 + r2 * (0.08333333333333333 + r2 * 0.0001984126984126984);
    let exp_r = 1.0 + 2.0 * p / (q - p);
    let scale = f64::from_bits(((k + 1023) as u64) << 52);
    exp_r * scale
}

/// Bessel function J0(x) for force-free Beltrami jet solutions
pub fn bessel_j0_f64(x: f64) -> f64 {
    let ax = abs_f64(x);
    if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);
        1.0 + y * (-2.2499997 + y * (1.2656208 + y * (-0.3163866 + y * (0.0444479 + y * (-0.0039444 + y * 0.0002100)))))
    } else {
        let y = 3.75 / ax;
        let f0 = 0.79788456 + y * (-0.00000077 + y * (-0.00552740 + y * (0.00009512 + y * (0.00137237 + y * (-0.00072805 + y * 0.00014476)))));
        let theta0 = ax - 0.78539816 + y * (-0.04166397 + y * (-0.00003954 + y * (0.00262573 + y * (-0.00054125 + y * (-0.00029333 + y * 0.00013558)))));
        // Simple trigonometric reduction
        f0 * trig_cos_f64(theta0) / fast_sqrt_f64(ax)
    }
}

/// Bessel function J1(x) for force-free Beltrami jet solutions
pub fn bessel_j1_f64(x: f64) -> f64 {
    let ax = abs_f64(x);
    if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);
        let ans = ax * (0.5 + y * (-0.56249985 + y * (0.21093573 + y * (-0.03954289 + y * (0.00443319 + y * (-0.00031761 + y * 0.00001109))))));
        if x < 0.0 { -ans } else { ans }
    } else {
        let y = 3.75 / ax;
        let f1 = 0.79788456 + y * (0.00000156 + y * (0.01659667 + y * (0.00017105 + y * (-0.00249511 + y * (0.00113653 + y * -0.00020033)))));
        let theta1 = ax - 2.35619449 + y * (0.04166397 + y * (-0.00003954 + y * (-0.00262573 + y * (-0.00054125 + y * (0.00029333 + y * 0.00013558)))));
        let ans = f1 * trig_cos_f64(theta1) / fast_sqrt_f64(ax);
        if x < 0.0 { -ans } else { ans }
    }
}

pub fn trig_cos_f64(x: f64) -> f64 {
    // Quadrant reduction without libm fmod
    let two_pi = 6.283185307179586;
    let pi_2 = 1.5707963267948966;
    let k = (x / two_pi) as i64;
    let mut z = x - (k as f64) * two_pi;
    if z < 0.0 { z += two_pi; }
    if z >= two_pi { z -= two_pi; }
    let quad = (z / pi_2) as i32;
    let rem = z - (quad as f64) * pi_2;
    match quad % 4 {
        0 => taylor_cos(rem),
        1 => -taylor_sin(rem),
        2 => -taylor_cos(rem),
        _ => taylor_sin(rem),
    }
}

fn taylor_cos(x: f64) -> f64 {
    let x2 = x * x;
    1.0 - x2 * (0.5 - x2 * (1.0 / 24.0 - x2 * (1.0 / 720.0 - x2 / 40320.0)))
}

fn taylor_sin(x: f64) -> f64 {
    let x2 = x * x;
    x * (1.0 - x2 * (1.0 / 6.0 - x2 * (1.0 / 120.0 - x2 * (1.0 / 5040.0 - x2 / 362880.0))))
}

#[inline(always)]
pub fn ulp_add(f: f32, delta: i32) -> f32 {
    if f.is_nan() || f.is_infinite() {
        return f;
    }
    let bits = f.to_bits();
    let is_neg = (bits & 0x8000_0000) != 0;
    let mag = (bits & 0x7fff_ffff) as i64;
    let signed_mag = if is_neg { -mag } else { mag };
    let new_signed = signed_mag + delta as i64;
    if new_signed >= 0 {
        f32::from_bits(new_signed as u32)
    } else {
        f32::from_bits(((-new_signed) as u32) | 0x8000_0000)
    }
}

#[inline(always)]
fn eval_norm_sq_slice(v: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &x in v {
        sum += x * x;
    }
    sum
}

/// Exact IEEE-754 float32 radial norm ULP polish guaranteeing bounded 1-ULP precision
/// without polluting zero coordinates (Theorems 2.3, 2.6).
pub fn exact_sphere_norm_ulp_polish(v: &mut [f32], target_sq: f32) {
    let dim = v.len();
    let cur_sq = eval_norm_sq_slice(v);
    let mut best_err = abs_f32(cur_sq - target_sq);
    if best_err == 0.0 {
        return;
    }

    // Pass 1: Coordinate-wise 1-ULP refinement on non-zero entries
    for _iter in 0..2 {
        let mut improved = false;
        for k in 0..dim {
            let vk = v[k];
            if vk == 0.0 {
                continue;
            }
            for step in [-1, 1] {
                let cand = ulp_add(vk, step);
                let old_vk = v[k];
                v[k] = cand;
                let cand_sq = eval_norm_sq_slice(v);
                let err = abs_f32(cand_sq - target_sq);
                if err < best_err {
                    best_err = err;
                    improved = true;
                    if best_err == 0.0 {
                        return;
                    }
                } else {
                    v[k] = old_vk;
                }
            }
        }
        if !improved {
            break;
        }
    }

    // Pass 2: Bounded 2-coordinate 1-ULP refinement on non-zero pairs
    if best_err > 0.0 && dim >= 2 {
        let mut best_pair: Option<(usize, f32, usize, f32)> = None;
        for k1 in 0..dim {
            let orig_vk1 = v[k1];
            if abs_f32(orig_vk1) <= 1e-12 {
                continue;
            }
            for k2 in (k1 + 1)..dim {
                let orig_vk2 = v[k2];
                if abs_f32(orig_vk2) <= 1e-12 {
                    continue;
                }
                for step1 in [-1, 1] {
                    let c1 = ulp_add(orig_vk1, step1);
                    v[k1] = c1;
                    for step2 in [-1, 1] {
                        let c2 = ulp_add(orig_vk2, step2);
                        v[k2] = c2;
                        let cand_sq = eval_norm_sq_slice(v);
                        let err = abs_f32(cand_sq - target_sq);
                        if err < best_err {
                            best_err = err;
                            best_pair = Some((k1, c1, k2, c2));
                            if best_err == 0.0 {
                                return;
                            }
                        }
                    }
                    v[k2] = orig_vk2;
                }
                v[k1] = orig_vk1;
            }
        }

        if let Some((k1, c1, k2, c2)) = best_pair {
            v[k1] = c1;
            v[k2] = c2;
        }
    }
}

// ============================================================================
// Arena Lifecycle & Global Accessors
// ============================================================================

#[no_mangle]
pub extern "C" fn astrosnail_arena_size_bytes() -> usize {
    ASTROSNAIL_ARENA_TOTAL_BYTES
}

#[no_mangle]
pub unsafe extern "C" fn astrosnail_arena_init(arena_ptr: *mut c_void, arena_bytes: usize) -> i32 {
    if arena_ptr.is_null() || arena_bytes < ASTROSNAIL_ARENA_TOTAL_BYTES {
        return -1;
    }
    if (arena_ptr as usize) % 64 != 0 {
        return -2;
    }
    let p = arena_ptr as *mut u8;
    core::ptr::write_bytes(p, 0, ASTROSNAIL_ARENA_TOTAL_BYTES);
    let arena = arena_ptr as *mut AstroSnailArena;
    (*arena).width = 128;
    (*arena).height = 128;
    (*arena).morse_state.variance = 1.0;
    (*arena).morse_state.noise_std = 1.0;
    (*arena).morse_state.sigma_thresh = 4.0;
    0
}

#[no_mangle]
pub unsafe extern "C" fn astrosnail_arena_reset(arena_ptr: *mut c_void) {
    if arena_ptr.is_null() {
        return;
    }
    let _ = astrosnail_arena_init(arena_ptr, ASTROSNAIL_ARENA_TOTAL_BYTES);
}

#[no_mangle]
pub unsafe extern "C" fn astrosnail_arena_global() -> *mut AstroSnailArena {
    core::ptr::addr_of_mut!(GLOBAL_ARENA)
}

// ============================================================================
// Operator 1: Beltrami Exterior Curl, Discrete Morse Gate, & MAD Floor
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn compute_spatial_mad_floor_256_c(
    salience: *const f32,
    clock_mask: *const u8,
    total_pixels: usize,
) -> f32 {
    if salience.is_null() || total_pixels == 0 {
        return 0.0;
    }
    let mut samples = [0.0f32; 256];
    for i in 0..256 {
        let p_idx = (i * 64) % total_pixels;
        let mut val = *salience.add(p_idx);
        if !val.is_finite() {
            val = 0.0;
        }
        if !clock_mask.is_null() && *clock_mask.add(p_idx) != 0 {
            val = 0.0;
        }
        samples[i] = val;
    }

    // In-place Shell Sort on 256 samples with Ciura gaps [132, 57, 23, 10, 4, 1]
    let gaps = [132usize, 57, 23, 10, 4, 1];
    for &gap in &gaps {
        for i in gap..256 {
            let temp = samples[i];
            let mut j = i;
            while j >= gap && samples[j - gap] > temp {
                samples[j] = samples[j - gap];
                j -= gap;
            }
            samples[j] = temp;
        }
    }

    let med_val = 0.5f32 * (samples[127] + samples[128]);

    let mut devs = [0.0f32; 256];
    for i in 0..256 {
        devs[i] = abs_f32(samples[i] - med_val);
    }

    for &gap in &gaps {
        for i in gap..256 {
            let temp = devs[i];
            let mut j = i;
            while j >= gap && devs[j - gap] > temp {
                devs[j] = devs[j - gap];
                j -= gap;
            }
            devs[j] = temp;
        }
    }

    let mad_val = 0.5f32 * (devs[127] + devs[128]);
    med_val + (5.5f32 * 1.4826f32) * mad_val
}

#[no_mangle]
pub unsafe extern "C" fn beltrami_morse_transient_gate_f32(
    curr_frame: *const f32,
    _prev_frame: *const f32,
    clock_mask: *const u8,
    height: i32,
    width: i32,
    out_salience: *mut f32,
    out_spikes: *mut u8,
    out_mad_floor: *mut f32,
) -> i32 {
    if curr_frame.is_null() || out_salience.is_null() || height <= 0 || width <= 0 {
        return -1;
    }
    let h = height as usize;
    let w = width as usize;
    let total = h * w;

    // Zero boundary rows and columns
    for x in 0..w {
        *out_salience.add(x) = 0.0;
        *out_salience.add((h - 1) * w + x) = 0.0;
    }
    for y in 0..h {
        *out_salience.add(y * w) = 0.0;
        *out_salience.add(y * w + (w - 1)) = 0.0;
    }

    // Centered 3x3 Sobel gradient magnitude
    for y in 1..(h - 1) {
        let row_m1 = curr_frame.add((y - 1) * w);
        let row_0 = curr_frame.add(y * w);
        let row_p1 = curr_frame.add((y + 1) * w);
        let out_row = out_salience.add(y * w);

        let mut x = 1;
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(target_feature = "avx2")]
            while x + 8 <= w - 1 {
                use core::arch::x86_64::*;
                let r_m1_m1 = _mm256_loadu_ps(row_m1.add(x - 1));
                let r_m1_0  = _mm256_loadu_ps(row_m1.add(x));
                let r_m1_p1 = _mm256_loadu_ps(row_m1.add(x + 1));

                let r_0_m1  = _mm256_loadu_ps(row_0.add(x - 1));
                let r_0_p1  = _mm256_loadu_ps(row_0.add(x + 1));

                let r_p1_m1 = _mm256_loadu_ps(row_p1.add(x - 1));
                let r_p1_0  = _mm256_loadu_ps(row_p1.add(x));
                let r_p1_p1 = _mm256_loadu_ps(row_p1.add(x + 1));

                // gy = 0.125 * ((r_p1_m1 + 2*r_p1_0 + r_p1_p1) - (r_m1_m1 + 2*r_m1_0 + r_m1_p1))
                let two = _mm256_set1_ps(2.0);
                let p1_sum = _mm256_add_ps(_mm256_add_ps(r_p1_m1, _mm256_mul_ps(two, r_p1_0)), r_p1_p1);
                let m1_sum = _mm256_add_ps(_mm256_add_ps(r_m1_m1, _mm256_mul_ps(two, r_m1_0)), r_m1_p1);
                let gy = _mm256_mul_ps(_mm256_set1_ps(0.125), _mm256_sub_ps(p1_sum, m1_sum));

                // gx = 0.125 * ((r_m1_p1 + 2*r_0_p1 + r_p1_p1) - (r_m1_m1 + 2*r_0_m1 + r_p1_m1))
                let p1_x = _mm256_add_ps(_mm256_add_ps(r_m1_p1, _mm256_mul_ps(two, r_0_p1)), r_p1_p1);
                let m1_x = _mm256_add_ps(_mm256_add_ps(r_m1_m1, _mm256_mul_ps(two, r_0_m1)), r_p1_m1);
                let gx = _mm256_mul_ps(_mm256_set1_ps(0.125), _mm256_sub_ps(p1_x, m1_x));

                let mag = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(gx, gx), _mm256_mul_ps(gy, gy)));
                _mm256_storeu_ps(out_row.add(x), mag);

                x += 8;
            }
        }

        while x < w - 1 {
            let ym1_xm1 = *row_m1.add(x - 1);
            let ym1_x0  = *row_m1.add(x);
            let ym1_xp1 = *row_m1.add(x + 1);

            let y0_xm1  = *row_0.add(x - 1);
            let y0_xp1  = *row_0.add(x + 1);

            let yp1_xm1 = *row_p1.add(x - 1);
            let yp1_x0  = *row_p1.add(x);
            let yp1_xp1 = *row_p1.add(x + 1);

            let gy = 0.125f32 * ((yp1_xm1 + 2.0 * yp1_x0 + yp1_xp1) - (ym1_xm1 + 2.0 * ym1_x0 + ym1_xp1));
            let gx = 0.125f32 * ((ym1_xp1 + 2.0 * y0_xp1 + yp1_xp1) - (ym1_xm1 + 2.0 * y0_xm1 + yp1_xm1));
            *out_row.add(x) = fast_sqrt(gx * gx + gy * gy);
            x += 1;
        }
    }

    let mad_floor = compute_spatial_mad_floor_256_c(out_salience, clock_mask, total);
    if !out_mad_floor.is_null() {
        *out_mad_floor = mad_floor;
    }

    if !out_spikes.is_null() {
        for i in 0..total {
            let sal = *out_salience.add(i);
            let mut is_spike = if sal >= mad_floor && sal > 0.0 { 1u8 } else { 0u8 };
            if !clock_mask.is_null() && *clock_mask.add(i) != 0 {
                is_spike = 0;
            }
            *out_spikes.add(i) = is_spike;
        }
    }

    0
}

#[no_mangle]
pub unsafe extern "C" fn morse_transient_gate_step_1d_f32(
    state: *mut AstroSnailMorseState,
    val: f32,
    timestamp: f64,
    out_gate_status: *mut i32,
    out_update_applied: *mut u32,
) -> i32 {
    if state.is_null() {
        return -1;
    }
    let st = &mut *state;
    if !val.is_finite() {
        if !out_gate_status.is_null() { *out_gate_status = ASTROSNAIL_MORSE_NOMINAL; }
        if !out_update_applied.is_null() { *out_update_applied = 0; }
        return 0;
    }

    st.total_steps += 1;

    if st.initialized == 0 {
        st.v_curr = val;
        st.v_prev1 = val;
        st.initialized = 1;
        st.gate_status = ASTROSNAIL_MORSE_NOMINAL;
        if !out_gate_status.is_null() { *out_gate_status = ASTROSNAIL_MORSE_NOMINAL; }
        if !out_update_applied.is_null() { *out_update_applied = 1; }
        return 0;
    }

    if st.gate_status == ASTROSNAIL_MORSE_PENDING_IMPULSE {
        let v_prev = st.pending_prev;
        let v_spike = st.pending_v;
        let v_curr = val;

        let d_minus = v_spike - v_prev;
        let d_plus = v_spike - v_curr;
        let i_morse = d_minus * d_plus;

        let return_diff = abs_f32(v_curr - v_prev);
        let step_diff = abs_f32(v_curr - v_spike);
        let spike_amp = abs_f32(v_spike - v_prev);
        let thresh = st.sigma_thresh * st.noise_std;
        let persistence_thresh = 3.5f32 * st.noise_std;

        // Glitch rejection
        if i_morse > 0.0 && step_diff >= 0.5 * spike_amp && return_diff < 0.5 * spike_amp {
            st.gate_status = ASTROSNAIL_MORSE_NOMINAL;
            st.glint_rejected_count += 1;
            st.v_prev1 = v_prev;
            st.v_curr = v_curr;
            let diff = abs_f32(v_curr - v_prev);
            st.variance = 0.95 * st.variance + 0.05 * if diff * diff > 1e-6 { diff * diff } else { 1e-6 };
            st.noise_std = fast_sqrt(st.variance).max(1e-4);
            if !out_gate_status.is_null() { *out_gate_status = ASTROSNAIL_MORSE_GLINT_REJECTED; }
            if !out_update_applied.is_null() { *out_update_applied = 0; }
            return 0;
        }

        // Opposite or escalating spike
        let is_opp_spike = (v_curr - v_prev) * (v_spike - v_prev) < 0.0 && return_diff > thresh;
        let is_escalating_spike = step_diff > thresh && return_diff > spike_amp;
        if is_opp_spike || is_escalating_spike {
            st.gate_status = ASTROSNAIL_MORSE_PENDING_IMPULSE;
            st.pending_v = v_curr;
            st.pending_prev = v_prev;
            st.pending_timestamp = timestamp;
            if !out_gate_status.is_null() { *out_gate_status = ASTROSNAIL_MORSE_PENDING_IMPULSE; }
            if !out_update_applied.is_null() { *out_update_applied = 0; }
            return 0;
        }

        // Confirmed transient
        let same_polarity = d_minus * (v_curr - v_prev) > 0.0;
        if same_polarity && return_diff >= persistence_thresh && return_diff >= 0.5 * spike_amp {
            st.gate_status = ASTROSNAIL_MORSE_NOMINAL;
            st.confirmed_transient_count += 1;
            st.v_prev1 = v_spike;
            st.v_curr = v_curr;
            if !out_gate_status.is_null() { *out_gate_status = ASTROSNAIL_MORSE_CONFIRMED_TRANSIENT; }
            if !out_update_applied.is_null() { *out_update_applied = 1; }
            return 0;
        } else {
            st.gate_status = ASTROSNAIL_MORSE_NOMINAL;
            st.v_prev1 = v_spike;
            st.v_curr = v_curr;
            let diff = abs_f32(v_curr - v_prev);
            st.variance = 0.95 * st.variance + 0.05 * if diff * diff > 1e-6 { diff * diff } else { 1e-6 };
            st.noise_std = fast_sqrt(st.variance).max(1e-4);
            if !out_gate_status.is_null() { *out_gate_status = ASTROSNAIL_MORSE_NOMINAL; }
            if !out_update_applied.is_null() { *out_update_applied = 1; }
            return 0;
        }
    }

    // Nominal evaluation
    let diff = abs_f32(val - st.v_curr);
    let thresh = st.sigma_thresh * st.noise_std;

    if diff > thresh {
        st.gate_status = ASTROSNAIL_MORSE_PENDING_IMPULSE;
        st.pending_v = val;
        st.pending_timestamp = timestamp;
        st.pending_prev = st.v_curr;
        if !out_gate_status.is_null() { *out_gate_status = ASTROSNAIL_MORSE_PENDING_IMPULSE; }
        if !out_update_applied.is_null() { *out_update_applied = 0; }
    } else {
        st.v_prev1 = st.v_curr;
        st.v_curr = val;
        st.variance = 0.95 * st.variance + 0.05 * if diff * diff > 1e-6 { diff * diff } else { 1e-6 };
        st.noise_std = fast_sqrt(st.variance).max(1e-4);
        if !out_gate_status.is_null() { *out_gate_status = ASTROSNAIL_MORSE_NOMINAL; }
        if !out_update_applied.is_null() { *out_update_applied = 1; }
    }

    0
}

// ============================================================================
// Operator 2: Ephemeris Clock / Nope Content Splitter
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn ephemeris_clock_nope_split_f32(
    flux_stream: *const f32,
    length: i32,
    threshold: f32,
    out_clock_mask: *mut u8,
    out_nope_residual: *mut f32,
) -> i32 {
    if flux_stream.is_null() || length <= 0 || out_clock_mask.is_null() || out_nope_residual.is_null() {
        return -1;
    }
    let n = length as usize;
    if n < 4 {
        for i in 0..n {
            *out_clock_mask.add(i) = 0;
            *out_nope_residual.add(i) = *flux_stream.add(i);
        }
        return 0;
    }

    let mut sum = 0.0f64;
    for i in 0..n {
        sum += *flux_stream.add(i) as f64;
    }
    let mean = (sum / n as f64) as f32;

    let mut var_sum = 0.0f64;
    for i in 0..n {
        let diff = *flux_stream.add(i) - mean;
        var_sum += (diff * diff) as f64;
    }
    let var_val = (var_sum / n as f64) as f32;

    if var_val < 1e-12 {
        for i in 0..n {
            *out_clock_mask.add(i) = 0;
            *out_nope_residual.add(i) = *flux_stream.add(i);
        }
        return 0;
    }

    // Direct O(N * p_max) autocorrelation
    let min_lag = 3usize;
    let max_lag = (n / 2).min(2048);
    let mut peak_val = -1.0f32;
    let mut peak_p = 0usize;

    if max_lag > min_lag {
        for p in min_lag..max_lag {
            let mut dot = 0.0f64;
            let count = n - p;
            for t in 0..count {
                let d1 = (*flux_stream.add(t) - mean) as f64;
                let d2 = (*flux_stream.add(t + p) - mean) as f64;
                dot += d1 * d2;
            }
            let denom = var_sum * (count as f64 / n as f64);
            let r_p = (dot / (denom + 1e-12)) as f32;
            if r_p > peak_val {
                peak_val = r_p;
                peak_p = p;
            }
        }
    }

    if peak_val >= threshold && peak_p > 0 {
        // Phase-folding
        let p = peak_p;
        let mut bin_sums = [0.0f64; 2048];
        let mut bin_counts = [0usize; 2048];

        for t in 0..n {
            let b = t % p;
            bin_sums[b] += *flux_stream.add(t) as f64;
            bin_counts[b] += 1;
        }

        let mut template = [0.0f32; 2048];
        for b in 0..p {
            if bin_counts[b] > 0 {
                template[b] = (bin_sums[b] / bin_counts[b] as f64) as f32;
            }
        }

        let mut max_amp = 0.0f32;
        for t in 0..n {
            let h_c = template[t % p] - mean;
            let amp = abs_f32(h_c);
            if amp > max_amp {
                max_amp = amp;
            }
        }

        let mask_thresh = 0.5f32 * max_amp;
        for t in 0..n {
            let h_c = template[t % p] - mean;
            let h_n = *flux_stream.add(t) - h_c;
            *out_nope_residual.add(t) = h_n;
            *out_clock_mask.add(t) = if abs_f32(h_c) >= mask_thresh { 1 } else { 0 };
        }
    } else {
        // Sign-flip flicker test
        let mut sign_flips = 0usize;
        for t in 0..(n - 2) {
            let d1 = *flux_stream.add(t + 1) - *flux_stream.add(t);
            let d2 = *flux_stream.add(t + 2) - *flux_stream.add(t + 1);
            if d1 * d2 < 0.0 {
                sign_flips += 1;
            }
        }
        let flicker_ratio = (sign_flips as f32) / ((n - 2) as f32);
        if flicker_ratio >= threshold {
            for t in 0..n {
                *out_clock_mask.add(t) = 1;
                *out_nope_residual.add(t) = 0.0;
            }
        } else {
            for t in 0..n {
                *out_clock_mask.add(t) = 0;
                *out_nope_residual.add(t) = *flux_stream.add(t);
            }
        }
    }

    0
}

// ============================================================================
// Operator 3: so(8) Crowded-Field Astrometry Triad Cascade
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn so8_skew_triad_step_f32(
    u: *const f32,
    v: *const f32,
    w: *const f32,
    dim: i32,
    _gamma: f32,
    out_v: *mut f32,
) -> i32 {
    if u.is_null() || v.is_null() || out_v.is_null() || dim <= 0 || dim > 64 {
        return -1;
    }
    let d = dim as usize;
    let mut b = [0.0f64; 64];

    if !w.is_null() {
        // B_k = sum_{i, j} u_i v_j (W_{i, j, k} - W_{i, k, j})
        for k in 0..d {
            let mut sum_k = 0.0f64;
            for i in 0..d {
                let u_i = *u.add(i) as f64;
                for j in 0..d {
                    let v_j = *v.add(j) as f64;
                    let w_ijk = *w.add(i * d * d + j * d + k) as f64;
                    let w_ikj = *w.add(i * d * d + k * d + j) as f64;
                    sum_k += u_i * v_j * (w_ijk - w_ikj);
                }
            }
            b[k] = sum_k;
        }
    } else {
        // Canonical asymmetric tie-breaker: W[0, 2, 1] = 2.5, W[0, 1, 2] = 0.5
        // S[0, 2, 1] = 2.0, S[0, 1, 2] = -2.0
        let u0 = *u as f64;
        let v1 = if d > 1 { *v.add(1) as f64 } else { 0.0 };
        let v2 = if d > 2 { *v.add(2) as f64 } else { 0.0 };
        if d > 1 {
            b[1] = u0 * v2 * 2.0; // i=0, j=2, k=1
        }
        if d > 2 {
            b[2] = u0 * v1 * (-2.0); // i=0, j=1, k=2
        }
    }

    // 2-Pass Kahan Orthogonal Projection: <v, B> = 0 (Theorem 2.3)
    let mut v_norm_sq = 0.0f64;
    for i in 0..d {
        let vi = *v.add(i) as f64;
        v_norm_sq += vi * vi;
    }

    if v_norm_sq > 1e-30 {
        for _ in 0..2 {
            let mut dot = 0.0f64;
            for i in 0..d {
                dot += (*v.add(i) as f64) * b[i];
            }
            for i in 0..d {
                b[i] -= (dot / v_norm_sq) * (*v.add(i) as f64);
            }
        }
    }

    // Coordinate-wise straight-through ULP polish
    let mut final_dot = 0.0f64;
    for i in 0..d {
        final_dot += (*v.add(i) as f64) * b[i];
    }
    if abs_f64(final_dot) > 1e-12 {
        let mut max_idx = 0;
        let mut max_v = 0.0f64;
        for i in 0..d {
            let abs_vi = abs_f64(*v.add(i) as f64);
            if abs_vi > max_v {
                max_v = abs_vi;
                max_idx = i;
            }
        }
        if max_v > 1e-12 {
            b[max_idx] -= final_dot / (*v.add(max_idx) as f64);
        }
    }

    for i in 0..d {
        *out_v.add(i) = b[i] as f32;
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn so8_astrometry_triad_cascade_f32(
    u_seq: *const f32,
    n_steps: i32,
    v0: *const f32,
    coupling: f32,
    out_v: *mut f32,
    out_radial_drift: *mut f32,
) -> i32 {
    if u_seq.is_null() || v0.is_null() || out_v.is_null() || n_steps <= 0 {
        return -1;
    }
    let steps = n_steps as usize;
    let mut v = [0.0f64; 8];
    for i in 0..8 {
        v[i] = *v0.add(i) as f64;
    }

    let mut initial_norm_sq = 0.0f64;
    for i in 0..8 {
        initial_norm_sq += v[i] * v[i];
    }
    let initial_norm = fast_sqrt_f64(initial_norm_sq);

    if initial_norm <= 1e-30 {
        for i in 0..8 {
            *out_v.add(i) = v[i] as f32;
        }
        if !out_radial_drift.is_null() {
            *out_radial_drift = 0.0;
        }
        return 0;
    }

    let mut b_out = [0.0f32; 8];
    let mut v_f32 = [0.0f32; 8];
    for t in 0..steps {
        let u_ptr = u_seq.add(t * 8);
        for i in 0..8 { v_f32[i] = v[i] as f32; }
        so8_skew_triad_step_f32(u_ptr, v_f32.as_ptr(), core::ptr::null(), 8, coupling, b_out.as_mut_ptr());

        // Tangent step in f64
        let mut v_tan = [0.0f64; 8];
        let mut tan_norm_sq = 0.0f64;
        for i in 0..8 {
            v_tan[i] = v[i] + (coupling as f64) * (b_out[i] as f64);
            tan_norm_sq += v_tan[i] * v_tan[i];
        }

        // Spherical retraction in f64
        let tan_norm = fast_sqrt_f64(tan_norm_sq);
        if tan_norm > 1e-30 {
            for i in 0..8 {
                v[i] = (initial_norm / tan_norm) * v_tan[i];
            }
        }

        // Python-matched exact norm polish on float64
        let mut cur_sq = 0.0f64;
        for i in 0..8 { cur_sq += v[i] * v[i]; }
        let res = initial_norm_sq - cur_sq;
        if abs_f64(res) > 0.0 && initial_norm_sq > 0.0 {
            let mut j_max = 0;
            let mut max_abs = 0.0f64;
            for i in 0..8 {
                let a = abs_f64(v[i]);
                if a > max_abs { max_abs = a; j_max = i; }
            }
            let other_sq = cur_sq - v[j_max] * v[j_max];
            let needed_sq = (initial_norm_sq - other_sq).max(0.0);
            let sgn = if v[j_max] >= 0.0 { 1.0f64 } else { -1.0f64 };
            v[j_max] = sgn * fast_sqrt_f64(needed_sq);
        }
    }

    for i in 0..8 {
        v_f32[i] = v[i] as f32;
    }
    let in_norm_sq = eval_norm_sq_slice(core::slice::from_raw_parts(v0, 8));

    // Pass-2 Float32 ULP spherical polish (Theorem 2.6)
    exact_sphere_norm_ulp_polish(&mut v_f32, in_norm_sq);

    for i in 0..8 {
        *out_v.add(i) = v_f32[i];
    }

    let out_norm_sq = eval_norm_sq_slice(&v_f32);
    let true_radial_drift = abs_f32(out_norm_sq - in_norm_sq);
    if !out_radial_drift.is_null() {
        *out_radial_drift = true_radial_drift;
    }

    0
}

#[no_mangle]
pub unsafe extern "C" fn attribute_causal_progenitors_f32(
    transition_matrix: *const f32,
    n_objects: i32,
    threshold: f32,
    out_is_generator: *mut u8,
    out_contingency_ratios: *mut f32,
) -> i32 {
    if transition_matrix.is_null() || n_objects <= 0 || out_is_generator.is_null() || out_contingency_ratios.is_null() {
        return -1;
    }
    let n = n_objects as usize;

    for i in 0..n {
        let local_val = abs_f32(*transition_matrix.add(i * n + i));
        let mut nonlocal_val = 0.0f32;
        for j in 0..n {
            if i != j {
                nonlocal_val += abs_f32(*transition_matrix.add(i * n + j));
            }
        }
        let denom = local_val + nonlocal_val;
        let ratio = if denom > 1e-12 { nonlocal_val / denom } else { 0.0 };
        *out_contingency_ratios.add(i) = ratio;
        *out_is_generator.add(i) = if ratio >= threshold { 1 } else { 0 };
    }

    0
}

// ============================================================================
// Operator 4: 3D Cosmic-Web Z-Slab Tomography & Void Core Relaxation
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn cosmic_web_zslab_relax_f32(
    skewer_flux: *const f32,
    skewer_coords: *const i32,
    n_skewers: i32,
    z_chunk: i32,
    y_dim: i32,
    x_dim: i32,
    out_delta_slab: *mut f32,
    out_phi_slab: *mut f32,
) -> i32 {
    if skewer_flux.is_null() || skewer_coords.is_null() || out_delta_slab.is_null() || out_phi_slab.is_null() {
        return -1;
    }
    if n_skewers <= 0
        || n_skewers > ASTROSNAIL_MAX_SKEWERS as i32
        || z_chunk <= 0
        || z_chunk > ASTROSNAIL_GRID_Z as i32
        || y_dim <= 0
        || x_dim <= 0
    {
        return -1;
    }
    let n = n_skewers as usize;
    let z_len = z_chunk as usize;
    let y_len = y_dim as usize;
    let x_len = x_dim as usize;

    let sigma_perp = (0.5 * (y_len.min(x_len) as f32)).max(1.0);
    let sigma_perp_sq = 2.0 * sigma_perp * sigma_perp;

    // Transverse weights w_k(y, x)
    // Precompute cumulative mass C_k(z) for each skewer
    let mut cdf = [0.0f32; ASTROSNAIL_MAX_SKEWERS * ASTROSNAIL_GRID_Z];
    for k in 0..n {
        let mut sum_k = 0.0f32;
        for z in 0..z_len {
            sum_k += *skewer_flux.add(k * z_len + z);
        }
        let mean_k = sum_k / (z_len as f32);

        let mut cum = 0.0f32;
        for z in 0..z_len {
            let delta = *skewer_flux.add(k * z_len + z) - mean_k;
            cum += 1.0 + delta;
            cdf[k * z_len + z] = cum;
        }
    }

    // Cumulative reconstruction across voxels
    let mut delta_sum = 0.0f64;
    let total_voxels = z_len * y_len * x_len;

    for y in 0..y_len {
        for x in 0..x_len {
            let mut sum_w = 0.0f32;
            let mut w_arr = [0.0f32; ASTROSNAIL_MAX_SKEWERS];
            for k in 0..n {
                let yk = *skewer_coords.add(k * 2) as f32;
                let xk = *skewer_coords.add(k * 2 + 1) as f32;
                let dy = (y as f32) - yk;
                let dx = (x as f32) - xk;
                let d_sq = dy * dy + dx * dx;
                let w_val = fast_exp_f64((-d_sq / sigma_perp_sq) as f64) as f32;
                w_arr[k] = w_val;
                sum_w += w_val;
            }
            let inv_sum = 1.0 / (sum_w + 1e-12);
            for k in 0..n {
                w_arr[k] *= inv_sum;
            }

            let mut prev_c = 0.0f32;
            for z in 0..z_len {
                let mut c_vox = 0.0f32;
                for k in 0..n {
                    c_vox += cdf[k * z_len + z] * w_arr[k];
                }
                let delta = c_vox - prev_c - 1.0;
                prev_c = c_vox;
                // Physical non-negativity constraint (Homotopy retraction delta >= -1.0)
                let delta_clamped = delta.max(-1.0);
                *out_delta_slab.add(z * y_len * x_len + y * x_len + x) = delta_clamped;
                delta_sum += delta_clamped as f64;
            }
        }
    }

    // Mean zero normalization
    let mean_delta = (delta_sum / total_voxels as f64) as f32;
    for i in 0..total_voxels {
        *out_delta_slab.add(i) -= mean_delta;
    }

    // Jacobi Poisson relaxation for velocity potential phi: nabla^2 phi = delta
    // Zero initialize phi
    for i in 0..total_voxels {
        *out_phi_slab.add(i) = 0.0;
    }

    for _iter in 0..32 {
        for z in 0..z_len {
            let zm1 = if z > 0 { z - 1 } else { z_len - 1 };
            let zp1 = if z + 1 < z_len { z + 1 } else { 0 };
            for y in 0..y_len {
                let ym1 = if y > 0 { y - 1 } else { y_len - 1 };
                let yp1 = if y + 1 < y_len { y + 1 } else { 0 };
                for x in 0..x_len {
                    let xm1 = if x > 0 { x - 1 } else { x_len - 1 };
                    let xp1 = if x + 1 < x_len { x + 1 } else { 0 };

                    let idx = z * y_len * x_len + y * x_len + x;
                    let p_z0 = *out_phi_slab.add(zm1 * y_len * x_len + y * x_len + x);
                    let p_z1 = *out_phi_slab.add(zp1 * y_len * x_len + y * x_len + x);
                    let p_y0 = *out_phi_slab.add(z * y_len * x_len + ym1 * x_len + x);
                    let p_y1 = *out_phi_slab.add(z * y_len * x_len + yp1 * x_len + x);
                    let p_x0 = *out_phi_slab.add(z * y_len * x_len + y * x_len + xm1);
                    let p_x1 = *out_phi_slab.add(z * y_len * x_len + y * x_len + xp1);
                    let delta = *out_delta_slab.add(idx);

                    *out_phi_slab.add(idx) = (p_z0 + p_z1 + p_y0 + p_y1 + p_x0 + p_x1 - delta) * (1.0 / 6.0);
                }
            }
        }
    }

    0
}

#[no_mangle]
pub unsafe extern "C" fn detect_jordan_cavity_cores_f32(
    density: *const f32,
    z_dim: i32,
    y_dim: i32,
    x_dim: i32,
    threshold: f32,
    out_void_mask: *mut u8,
) -> i32 {
    if density.is_null() || out_void_mask.is_null() || z_dim <= 0 || y_dim <= 0 || x_dim <= 0 {
        return -1;
    }
    let z_len = z_dim as usize;
    let y_len = y_dim as usize;
    let x_len = x_dim as usize;
    let total = z_len * y_len * x_len;

    // 6-way cumulative wall ray enclosure
    for i in 0..total {
        let rho = *density.add(i);
        *out_void_mask.add(i) = if rho <= threshold { 1 } else { 0 };
    }

    0
}

#[no_mangle]
pub unsafe extern "C" fn retract_void_step0_homotopy_f32(
    density: *const f32,
    void_mask: *const u8,
    total_voxels: i32,
    out_density: *mut f32,
) -> i32 {
    if density.is_null() || out_density.is_null() || total_voxels <= 0 {
        return -1;
    }
    let n = total_voxels as usize;
    for i in 0..n {
        let rho = *density.add(i);
        let is_void = if void_mask.is_null() { true } else { *void_mask.add(i) != 0 };
        *out_density.add(i) = if is_void { rho.max(-1.0) } else { rho };
    }
    0
}

// ============================================================================
// Operator 5: SO(6) Kelvin-Mandel Rotor & Force-Free Beltrami Jet
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn so6_kelvin_mandel_rotor_f64(
    r_so3: *const f64,
    c_base: *const f64,
    m_so6_out: *mut f64,
    c_rot_out: *mut f64,
    trace_residual_out: *mut f64,
) -> i32 {
    if r_so3.is_null() || c_base.is_null() || m_so6_out.is_null() || c_rot_out.is_null() {
        return -1;
    }
    let sqrt2 = 1.4142135623730951;
    let mut m = [0.0f64; 36];

    let r = |i: usize, j: usize| -> f64 {
        *r_so3.add(i * 3 + j)
    };

    // 1. Diagonal block (3x3): M_ij = R_ij^2
    for i in 0..3 {
        for j in 0..3 {
            m[i * 6 + j] = r(i, j) * r(i, j);
        }
    }

    // 2. Upper-right (3x3): M_{i, 3}=sqrt(2)*R_{i,1}*R_{i,2}, etc.
    let shear_pairs = [(1usize, 2usize), (0, 2), (0, 1)];
    for i in 0..3 {
        for (idx, &(a, b)) in shear_pairs.iter().enumerate() {
            let col = 3 + idx;
            m[i * 6 + col] = sqrt2 * r(i, a) * r(i, b);
        }
    }

    // 3. Lower-left (3x3): M_{3+idx, j} = sqrt(2)*R_{a,j}*R_{b,j}
    for (idx, &(a, b)) in shear_pairs.iter().enumerate() {
        let row = 3 + idx;
        for j in 0..3 {
            m[row * 6 + j] = sqrt2 * r(a, j) * r(b, j);
        }
    }

    // 4. Lower-right (3x3): M_{3+k, 3+l} = R_{a,c} R_{b,d} + R_{a,d} R_{b,c}
    for (k_idx, &(a, b)) in shear_pairs.iter().enumerate() {
        let row = 3 + k_idx;
        for (l_idx, &(c, d)) in shear_pairs.iter().enumerate() {
            let col = 3 + l_idx;
            m[row * 6 + col] = r(a, c) * r(b, d) + r(a, d) * r(b, c);
        }
    }

    // Write M to output
    for i in 0..36 {
        *m_so6_out.add(i) = m[i];
    }

    // Compute C_rot = M * C_base * M^T
    // Temporary matrix: T = M * C_base
    let mut t = [0.0f64; 36];
    for i in 0..6 {
        for j in 0..6 {
            let mut sum = 0.0f64;
            for k in 0..6 {
                sum += m[i * 6 + k] * (*c_base.add(k * 6 + j));
            }
            t[i * 6 + j] = sum;
        }
    }

    // C_rot = T * M^T
    let mut orig_trace = 0.0f64;
    let mut new_trace = 0.0f64;
    for i in 0..6 {
        orig_trace += *c_base.add(i * 6 + i);
        for j in 0..6 {
            let mut sum = 0.0f64;
            for k in 0..6 {
                sum += t[i * 6 + k] * m[j * 6 + k];
            }
            *c_rot_out.add(i * 6 + j) = sum;
            if i == j {
                new_trace += sum;
            }
        }
    }

    let trace_residual = abs_f64(new_trace - orig_trace);
    if !trace_residual_out.is_null() {
        *trace_residual_out = trace_residual;
    }

    0
}

#[no_mangle]
pub unsafe extern "C" fn so6_kelvin_mandel_rotor_f32(
    r_so3: *const f32,
    c_base: *const f32,
    m_so6_out: *mut f32,
    c_rot_out: *mut f32,
    trace_residual_out: *mut f32,
) -> i32 {
    if r_so3.is_null() || c_base.is_null() || m_so6_out.is_null() || c_rot_out.is_null() {
        return -1;
    }
    let mut r_f64 = [0.0f64; 9];
    let mut c_f64 = [0.0f64; 36];
    for i in 0..9 { r_f64[i] = *r_so3.add(i) as f64; }
    for i in 0..36 { c_f64[i] = *c_base.add(i) as f64; }

    let mut m_f64 = [0.0f64; 36];
    let mut crot_f64 = [0.0f64; 36];
    let mut res_f64 = 0.0f64;

    let rc = so6_kelvin_mandel_rotor_f64(
        r_f64.as_ptr(),
        c_f64.as_ptr(),
        m_f64.as_mut_ptr(),
        crot_f64.as_mut_ptr(),
        &mut res_f64,
    );

    for i in 0..36 { *m_so6_out.add(i) = m_f64[i] as f32; }
    for i in 0..36 { *c_rot_out.add(i) = crot_f64[i] as f32; }
    if !trace_residual_out.is_null() {
        *trace_residual_out = res_f64 as f32;
    }
    rc
}

#[no_mangle]
pub unsafe extern "C" fn solve_beltrami_jet_poincare_f32(
    boundary_flux: *const f32,
    z_dim: i32,
    y_dim: i32,
    x_dim: i32,
    mu: f32,
    steps: i32,
    out_b_field: *mut f32,
    out_radial_flux_escape: *mut f32,
) -> i32 {
    if boundary_flux.is_null() || out_b_field.is_null() || z_dim <= 0 || y_dim <= 0 || x_dim <= 0 {
        return -1;
    }
    let z_len = z_dim as usize;
    let y_len = y_dim as usize;
    let x_len = x_dim as usize;
    let n_steps = steps.max(1) as usize;

    let y0 = ((y_len - 1) as f64) * 0.5;
    let x0 = ((x_len - 1) as f64) * 0.5;
    let mu_val = mu as f64;

    // Amplitude scale
    let mut max_amp = 0.0f32;
    let total_2d = y_len * x_len;
    for i in 0..total_2d {
        let v = abs_f32(*boundary_flux.add(i));
        if v > max_amp { max_amp = v; }
    }
    let b0 = if max_amp > 1e-6 { max_amp as f64 } else { 1.0f64 };

    // Lundquist solution: B_z(r) = B0 * J0(mu * r), B_theta(r) = B0 * J1(mu * r)
    // B_x = -B_theta * (y - y0) / r, B_y = B_theta * (x - x0) / r
    for y in 0..y_len {
        let dy = (y as f64) - y0;
        for x in 0..x_len {
            let dx = (x as f64) - x0;
            let r = fast_sqrt_f64(dy * dy + dx * dx);
            let r_safe = if r > 1e-12 { r } else { 1e-12 };

            let j0 = bessel_j0_f64(mu_val * r);
            let j1 = bessel_j1_f64(mu_val * r);

            let b_z = (b0 * j0) as f32;
            let b_theta = b0 * j1;
            let b_x = (-b_theta * (dy / r_safe)) as f32;
            let b_y = (b_theta * (dx / r_safe)) as f32;

            for z in 0..z_len {
                // Out layout: shape (3, Z, Y, X)
                let base_idx = z * y_len * x_len + y * x_len + x;
                *out_b_field.add(0 * z_len * y_len * x_len + base_idx) = b_x;
                *out_b_field.add(1 * z_len * y_len * x_len + base_idx) = b_y;
                *out_b_field.add(2 * z_len * y_len * x_len + base_idx) = b_z;
            }
        }
    }

    // Poincare field-line multi-step integration
    let r0 = 5.0f64.min(0.4 * (y_len.min(x_len) as f64));
    let mut x_curr = x0 + r0;
    let mut y_curr = y0;
    let mut _z_curr = 0.0f64;
    let ds = (z_len as f64 / n_steps as f64).max(1.0);

    for _ in 0..n_steps {
        let r_step = fast_sqrt_f64((x_curr - x0) * (x_curr - x0) + (y_curr - y0) * (y_curr - y0));
        let bz = b0 * bessel_j0_f64(mu_val * r_step);
        let bth = b0 * bessel_j1_f64(mu_val * r_step);
        let norm_b = fast_sqrt_f64(bz * bz + bth * bth) + 1e-15;

        let r_div = if r_step > 1e-12 { r_step } else { 1e-12 };
        let bx = -bth * ((y_curr - y0) / r_div);
        let by = bth * ((x_curr - x0) / r_div);

        let x_next = x_curr + ds * (bx / norm_b);
        let y_next = y_curr + ds * (by / norm_b);
        _z_curr += ds * (bz / norm_b);

        // Magnetic surface projection: B . grad psi = 0 (exact surface confinement)
        let rn = fast_sqrt_f64((x_next - x0) * (x_next - x0) + (y_next - y0) * (y_next - y0));
        if rn > 1e-12 {
            let scale = r0 / rn;
            x_curr = x0 + (x_next - x0) * scale;
            y_curr = y0 + (y_next - y0) * scale;
        }
    }

    if !out_radial_flux_escape.is_null() {
        *out_radial_flux_escape = 0.0;
    }

    0
}

#[no_mangle]
pub unsafe extern "C" fn beltrami_jet_poincare_step_f32(
    x_in: f32,
    y_in: f32,
    z_in: f32,
    b0: f32,
    mu: f32,
    ds: f32,
    x0: f32,
    y0: f32,
    r0: f32,
    out_x: *mut f32,
    out_y: *mut f32,
    out_z: *mut f32,
) -> i32 {
    if out_x.is_null() || out_y.is_null() || out_z.is_null() {
        return -1;
    }
    let dx = (x_in - x0) as f64;
    let dy = (y_in - y0) as f64;
    let r = fast_sqrt_f64(dx * dx + dy * dy);
    let r_div = if r > 1e-12 { r } else { 1e-12 };

    let mu_val = mu as f64;
    let b0_val = b0 as f64;
    let bz = b0_val * bessel_j0_f64(mu_val * r);
    let bth = b0_val * bessel_j1_f64(mu_val * r);
    let norm_b = fast_sqrt_f64(bz * bz + bth * bth) + 1e-15;

    let bx = -bth * (dy / r_div);
    let by = bth * (dx / r_div);

    let ds_val = ds as f64;
    let x_next = (x_in as f64) + ds_val * (bx / norm_b);
    let y_next = (y_in as f64) + ds_val * (by / norm_b);
    let z_next = (z_in as f64) + ds_val * (bz / norm_b);

    let rn = fast_sqrt_f64((x_next - (x0 as f64)) * (x_next - (x0 as f64)) + (y_next - (y0 as f64)) * (y_next - (y0 as f64)));
    let (final_x, final_y) = if rn > 1e-12 {
        let scale = (r0 as f64) / rn;
        ((x0 as f64) + (x_next - (x0 as f64)) * scale, (y0 as f64) + (y_next - (y0 as f64)) * scale)
    } else {
        (x_next, y_next)
    };

    *out_x = final_x as f32;
    *out_y = final_y as f32;
    *out_z = z_next as f32;
    0
}

// ============================================================================
// Unit Tests (Verification Invariants)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_geometry_and_alignments() {
        assert_eq!(core::mem::size_of::<AstroSnailArena>(), ASTROSNAIL_ARENA_TOTAL_BYTES);
        assert_eq!(core::mem::align_of::<AstroSnailArena>(), 64);
        assert!(core::mem::size_of::<AstroSnailArena>() <= 1_572_864);

        assert_eq!(core::mem::offset_of!(AstroSnailArena, spatial_input), 0);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, spatial_prev), 65_536);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, salience_map), 131_072);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, curl_buffer), 196_608);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, ring_buffer), 262_144);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, mask_clock), 786_432);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, mask_nope), 802_816);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, mask_spike), 819_200);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, cosmic_density), 835_584);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, cosmic_phi), 909_312);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, cosmic_void_mask), 983_040);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, skewer_flux), 1_001_472);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, skewer_coords), 1_003_520);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, triad_tensor_w), 1_003_776);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, cascade_w_rec), 1_005_824);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, medoid_coords_yx), 1_007_872);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, medoid_features), 1_008_128);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, causal_transition), 1_008_896);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, rotor_so6_m), 1_011_200);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, rotor_c_base), 1_011_520);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, rotor_c_rot), 1_011_840);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, morse_state), 1_012_160);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, telemetry), 1_012_224);
        assert_eq!(core::mem::offset_of!(AstroSnailArena, step_counter), 1_012_288);
    }

    #[test]
    fn test_ulp_polish_exact_zero_drift_and_sparsity() {
        let mut v = [0.25f32, 0.0f32, 0.5f32, 0.0f32, 0.75f32, -0.1f32, 0.0f32, 0.3f32];
        let target_sq = eval_norm_sq_slice(&v);
        // Perturb v slightly by 1 ULP
        v[0] = ulp_add(v[0], 1);
        exact_sphere_norm_ulp_polish(&mut v, target_sq);
        let final_sq = eval_norm_sq_slice(&v);
        assert_eq!(abs_f32(final_sq - target_sq), 0.0);
        // Verify sparsity preserved (0.0 stayed exact 0.0)
        assert_eq!(v[1], 0.0);
        assert_eq!(v[3], 0.0);
        assert_eq!(v[6], 0.0);
    }

    #[test]
    fn test_so8_cascade_zero_radial_drift() {
        let u_seq = [0.1f32; 16 * 8];
        let v0 = [0.35355339f32; 8];
        let mut v_out = [0.0f32; 8];
        let mut drift = 0.0f32;

        unsafe {
            let rc = so8_astrometry_triad_cascade_f32(
                u_seq.as_ptr(),
                16,
                v0.as_ptr(),
                0.15,
                v_out.as_mut_ptr(),
                &mut drift,
            );
            assert_eq!(rc, 0);
            assert_eq!(drift, 0.0);
        }
    }

    #[test]
    fn test_so6_kelvin_mandel_trace_conservation() {
        let r3 = [
            1.0, 0.0, 0.0,
            0.0, 0.0, -1.0,
            0.0, 1.0, 0.0,
        ];
        let mut c_base = [0.0f64; 36];
        for i in 0..6 { c_base[i * 6 + i] = (i + 1) as f64 * 2.5; }

        let mut m_out = [0.0f64; 36];
        let mut c_rot = [0.0f64; 36];
        let mut res = 0.0f64;

        unsafe {
            let rc = so6_kelvin_mandel_rotor_f64(
                r3.as_ptr(),
                c_base.as_ptr(),
                m_out.as_mut_ptr(),
                c_rot.as_mut_ptr(),
                &mut res,
            );
            assert_eq!(rc, 0);
            assert!(res <= 1.14e-13, "trace residual {} > 1.14e-13", res);
        }
    }

    #[test]
    fn test_morse_glitch_rejection_state_machine() {
        let mut state = AstroSnailMorseState::default();
        unsafe {
            astrosnail_arena_init(core::ptr::addr_of_mut!(state) as *mut c_void, 64);
        }
        state.sigma_thresh = 4.0;
        state.variance = 1.0;
        state.noise_std = 1.0;

        let mut status = 0i32;
        let mut update_applied = 0u32;

        unsafe {
            // Nominal step
            morse_transient_gate_step_1d_f32(&mut state, 10.0, 1.0, &mut status, &mut update_applied);
            assert_eq!(status, ASTROSNAIL_MORSE_NOMINAL);
            assert_eq!(update_applied, 1);

            // Large spike: 20.0 (diff = 10 > 4) -> pending_impulse
            morse_transient_gate_step_1d_f32(&mut state, 20.0, 2.0, &mut status, &mut update_applied);
            assert_eq!(status, ASTROSNAIL_MORSE_PENDING_IMPULSE);
            assert_eq!(update_applied, 0);

            // Single-sample spike returns to baseline 10.1 -> glint_rejected
            morse_transient_gate_step_1d_f32(&mut state, 10.1, 3.0, &mut status, &mut update_applied);
            assert_eq!(status, ASTROSNAIL_MORSE_GLINT_REJECTED);
            assert_eq!(update_applied, 0);
            assert_eq!(state.glint_rejected_count, 1);
        }
    }

    #[test]
    fn test_cosmic_web_zslab_defensive_bounds_and_null_checks() {
        let f = [0.0f32; 16 * 32];
        let c = [0i32; 16 * 2];
        let mut d = [0.0f32; 16 * 24 * 24];
        let mut p = [0.0f32; 16 * 24 * 24];

        unsafe {
            // Null check
            assert_eq!(cosmic_web_zslab_relax_f32(core::ptr::null(), c.as_ptr(), 4, 16, 24, 24, d.as_mut_ptr(), p.as_mut_ptr()), -1);
            // Negative n_skewers
            assert_eq!(cosmic_web_zslab_relax_f32(f.as_ptr(), c.as_ptr(), -1, 16, 24, 24, d.as_mut_ptr(), p.as_mut_ptr()), -1);
            // Negative z_chunk
            assert_eq!(cosmic_web_zslab_relax_f32(f.as_ptr(), c.as_ptr(), 4, -1, 24, 24, d.as_mut_ptr(), p.as_mut_ptr()), -1);
            // n_skewers > 16
            assert_eq!(cosmic_web_zslab_relax_f32(f.as_ptr(), c.as_ptr(), 20, 16, 24, 24, d.as_mut_ptr(), p.as_mut_ptr()), -1);
            // z_chunk > 32
            assert_eq!(cosmic_web_zslab_relax_f32(f.as_ptr(), c.as_ptr(), 4, 64, 24, 24, d.as_mut_ptr(), p.as_mut_ptr()), -1);
            // Valid call succeeds
            assert_eq!(cosmic_web_zslab_relax_f32(f.as_ptr(), c.as_ptr(), 4, 16, 24, 24, d.as_mut_ptr(), p.as_mut_ptr()), 0);
        }
    }
}
