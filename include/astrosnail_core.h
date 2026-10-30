/**
 * @file astrosnail_core.h
 * @brief Unified ANSI C-ABI Header for sneed-and-feed/astrosnail.
 *
 * Exposes allocation-free static AstroSnailArena geometry (< 1.50 MB fixed SRAM),
 * SIMD-accelerated astrophysics and cosmology kernels, Pass-2 Float32 ULP
 * spherical polish, and zero-copy ctypes FFI bindings.
 *
 * Synced to triad-core and formalization/OcclusionTriad.lean (0 sorry).
 * License: CC0-1.0 (Public Domain)
 */

#ifndef ASTROSNAIL_CORE_H
#define ASTROSNAIL_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Alignment and API macros for MSVC, GCC, and Clang */
#if defined(_MSC_VER)
  #define ASTROSNAIL_ALIGN64 __declspec(align(64))
  #define ASTROSNAIL_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
  #define ASTROSNAIL_ALIGN64 __attribute__((aligned(64)))
  #define ASTROSNAIL_API __attribute__((visibility("default")))
#else
  #define ASTROSNAIL_ALIGN64
  #define ASTROSNAIL_API
#endif

/* ========================================================================= */
/* Geometry Constants                                                       */
/* ========================================================================= */

#define ASTROSNAIL_SPATIAL_DIM 128
#define ASTROSNAIL_MAX_PIXELS (128 * 128)
#define ASTROSNAIL_RING_BUFFER_FRAMES 8
#define ASTROSNAIL_RING_BUFFER_SIZE (ASTROSNAIL_MAX_PIXELS * ASTROSNAIL_RING_BUFFER_FRAMES)
#define ASTROSNAIL_MAX_MEDOIDS 24
#define ASTROSNAIL_ROPE_FEAT_DIM 8
#define ASTROSNAIL_SO8_DIM 8
#define ASTROSNAIL_SO8_TENSOR_SIZE (8 * 8 * 8)
#define ASTROSNAIL_GRID_Z 32
#define ASTROSNAIL_GRID_Y 24
#define ASTROSNAIL_GRID_X 24
#define ASTROSNAIL_COSMIC_VOXELS (32 * 24 * 24)
#define ASTROSNAIL_Z_SLAB_CHUNK 16
#define ASTROSNAIL_MAX_SKEWERS 16
#define ASTROSNAIL_ROTOR_SO6_DIM 6
#define ASTROSNAIL_ARENA_TOTAL_BYTES 1012352

/* Discrete Morse Classification Codes */
#define ASTROSNAIL_MORSE_NOMINAL 0
#define ASTROSNAIL_MORSE_PENDING_IMPULSE 1
#define ASTROSNAIL_MORSE_CONFIRMED_TRANSIENT 2
#define ASTROSNAIL_MORSE_GLINT_REJECTED 3

/* ========================================================================= */
/* Core Subsystem Data Structures                                           */
/* ========================================================================= */

/**
 * @brief 1-Sample Causal Deferral Discrete Morse Gate State (64 bytes).
 */
typedef struct {
    float v_curr;
    float v_prev1;
    float v_prev2;
    float pending_v;
    float pending_prev;
    float variance;
    float noise_std;
    float sigma_thresh;
    double pending_timestamp;
    int32_t gate_status;
    uint32_t confirmed_transient_count;
    uint32_t glint_rejected_count;
    uint32_t total_steps;
    uint32_t initialized;
    uint8_t _pad[4];
} AstroSnailMorseState;

/**
 * @brief Real-Time Telemetry Metrics HUD (64 bytes).
 */
typedef struct {
    float fps;
    uint32_t execution_cycles;
    float cable_spike_recovery_rate;
    float glint_rejection_rate;
    float max_radial_drift;
    float max_trace_residual;
    uint32_t active_generators_count;
    float parity_residual_f32;
    uint8_t _pad[32];
} AstroSnailTelemetry;

/**
 * @brief Universal Static Arena (< 1.50 MB fixed SRAM, 0 heap allocations).
 * Pairwise byte-disjoint, 64-byte aligned sub-buffers.
 * Total size: exactly 1,012,352 bytes.
 */
typedef struct ASTROSNAIL_ALIGN64 {
    float spatial_input[ASTROSNAIL_MAX_PIXELS];                 /* 65,536 B (offset 0) */
    float spatial_prev[ASTROSNAIL_MAX_PIXELS];                  /* 65,536 B (offset 65,536) */
    float salience_map[ASTROSNAIL_MAX_PIXELS];                  /* 65,536 B (offset 131,072) */
    float curl_buffer[ASTROSNAIL_MAX_PIXELS];                   /* 65,536 B (offset 196,608) */
    float ring_buffer[ASTROSNAIL_RING_BUFFER_SIZE];             /* 524,288 B (offset 262,144) */
    uint8_t mask_clock[ASTROSNAIL_MAX_PIXELS];                  /* 16,384 B (offset 786,432) */
    uint8_t mask_nope[ASTROSNAIL_MAX_PIXELS];                   /* 16,384 B (offset 802,816) */
    uint8_t mask_spike[ASTROSNAIL_MAX_PIXELS];                  /* 16,384 B (offset 819,200) */
    float cosmic_density[ASTROSNAIL_COSMIC_VOXELS];             /* 73,728 B (offset 835,584) */
    float cosmic_phi[ASTROSNAIL_COSMIC_VOXELS];                 /* 73,728 B (offset 909,312) */
    uint8_t cosmic_void_mask[ASTROSNAIL_COSMIC_VOXELS];         /* 18,432 B (offset 983,040) */
    float skewer_flux[ASTROSNAIL_MAX_SKEWERS * ASTROSNAIL_GRID_Z]; /* 2,048 B (offset 1,001,472) */
    int32_t skewer_coords[ASTROSNAIL_MAX_SKEWERS * 2 + 32];     /* 256 B (offset 1,003,520) */
    float triad_tensor_w[ASTROSNAIL_SO8_TENSOR_SIZE];           /* 2,048 B (offset 1,003,776) */
    float cascade_w_rec[ASTROSNAIL_SO8_TENSOR_SIZE];            /* 2,048 B (offset 1,005,824) */
    float medoid_coords_yx[64];                                 /* 256 B (offset 1,007,872) */
    float medoid_features[192];                                 /* 768 B (offset 1,008,128) */
    float causal_transition[24 * 24];                           /* 2,304 B (offset 1,008,896) */
    double rotor_so6_m[40];                                     /* 320 B (offset 1,011,200) */
    double rotor_c_base[40];                                    /* 320 B (offset 1,011,520) */
    double rotor_c_rot[40];                                     /* 320 B (offset 1,011,840) */
    AstroSnailMorseState morse_state;                           /* 64 B (offset 1,012,160) */
    AstroSnailTelemetry telemetry;                              /* 64 B (offset 1,012,224) */
    uint32_t step_counter;                                      /* 4 B (offset 1,012,288) */
    uint32_t num_medoids;                                       /* 4 B */
    uint32_t ring_head;                                         /* 4 B */
    uint32_t width;                                             /* 4 B */
    uint32_t height;                                            /* 4 B */
    uint32_t n_skewers;                                         /* 4 B */
    float last_radial_drift;                                    /* 4 B */
    float last_trace_residual;                                  /* 4 B */
    uint8_t _pad[32];                                           /* 32 B (total: 1,012,352 B) */
} AstroSnailArena;

/* Compile-time static assertions for C11 and C++11 */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
  _Static_assert(sizeof(AstroSnailMorseState) == 64, "AstroSnailMorseState must be 64 bytes");
  _Static_assert(sizeof(AstroSnailTelemetry) == 64, "AstroSnailTelemetry must be 64 bytes");
  _Static_assert(sizeof(AstroSnailArena) == ASTROSNAIL_ARENA_TOTAL_BYTES, "AstroSnailArena size mismatch");
  _Static_assert(offsetof(AstroSnailArena, spatial_input) == 0, "spatial_input offset");
  _Static_assert(offsetof(AstroSnailArena, spatial_prev) == 65536, "spatial_prev offset");
  _Static_assert(offsetof(AstroSnailArena, salience_map) == 131072, "salience_map offset");
  _Static_assert(offsetof(AstroSnailArena, curl_buffer) == 196608, "curl_buffer offset");
  _Static_assert(offsetof(AstroSnailArena, ring_buffer) == 262144, "ring_buffer offset");
  _Static_assert(offsetof(AstroSnailArena, mask_clock) == 786432, "mask_clock offset");
  _Static_assert(offsetof(AstroSnailArena, mask_nope) == 802816, "mask_nope offset");
  _Static_assert(offsetof(AstroSnailArena, mask_spike) == 819200, "mask_spike offset");
  _Static_assert(offsetof(AstroSnailArena, cosmic_density) == 835584, "cosmic_density offset");
  _Static_assert(offsetof(AstroSnailArena, cosmic_phi) == 909312, "cosmic_phi offset");
  _Static_assert(offsetof(AstroSnailArena, cosmic_void_mask) == 983040, "cosmic_void_mask offset");
  _Static_assert(offsetof(AstroSnailArena, skewer_flux) == 1001472, "skewer_flux offset");
  _Static_assert(offsetof(AstroSnailArena, skewer_coords) == 1003520, "skewer_coords offset");
  _Static_assert(offsetof(AstroSnailArena, triad_tensor_w) == 1003776, "triad_tensor_w offset");
  _Static_assert(offsetof(AstroSnailArena, cascade_w_rec) == 1005824, "cascade_w_rec offset");
  _Static_assert(offsetof(AstroSnailArena, medoid_coords_yx) == 1007872, "medoid_coords_yx offset");
  _Static_assert(offsetof(AstroSnailArena, medoid_features) == 1008128, "medoid_features offset");
  _Static_assert(offsetof(AstroSnailArena, causal_transition) == 1008896, "causal_transition offset");
  _Static_assert(offsetof(AstroSnailArena, rotor_so6_m) == 1011200, "rotor_so6_m offset");
  _Static_assert(offsetof(AstroSnailArena, rotor_c_base) == 1011520, "rotor_c_base offset");
  _Static_assert(offsetof(AstroSnailArena, rotor_c_rot) == 1011840, "rotor_c_rot offset");
  _Static_assert(offsetof(AstroSnailArena, morse_state) == 1012160, "morse_state offset");
  _Static_assert(offsetof(AstroSnailArena, telemetry) == 1012224, "telemetry offset");
  _Static_assert(offsetof(AstroSnailArena, step_counter) == 1012288, "step_counter offset");
#elif defined(__cplusplus) && __cplusplus >= 201103L
  static_assert(sizeof(AstroSnailMorseState) == 64, "AstroSnailMorseState must be 64 bytes");
  static_assert(sizeof(AstroSnailTelemetry) == 64, "AstroSnailTelemetry must be 64 bytes");
  static_assert(sizeof(AstroSnailArena) == ASTROSNAIL_ARENA_TOTAL_BYTES, "AstroSnailArena size mismatch");
#endif

/* ========================================================================= */
/* Arena Lifecycle & Accessors                                              */
/* ========================================================================= */

ASTROSNAIL_API size_t astrosnail_arena_size_bytes(void);
ASTROSNAIL_API int32_t astrosnail_arena_init(void *arena_ptr, size_t arena_bytes);
ASTROSNAIL_API void astrosnail_arena_reset(void *arena_ptr);
ASTROSNAIL_API AstroSnailArena *astrosnail_arena_global(void);

/* ========================================================================= */
/* Operator 1: Beltrami Exterior Curl + Discrete Morse Gate + MAD Floor     */
/* ========================================================================= */

ASTROSNAIL_API int32_t beltrami_morse_transient_gate_f32(
    const float *curr_frame,
    const float *prev_frame,
    const uint8_t *clock_mask,
    int32_t height,
    int32_t width,
    float *out_salience,
    uint8_t *out_spikes,
    float *out_mad_floor
);

ASTROSNAIL_API float compute_spatial_mad_floor_256_c(
    const float *salience,
    const uint8_t *clock_mask,
    size_t total_pixels
);

ASTROSNAIL_API int32_t morse_transient_gate_step_1d_f32(
    AstroSnailMorseState *state,
    float val,
    double timestamp,
    int32_t *out_gate_status,
    uint32_t *out_update_applied
);

/* ========================================================================= */
/* Operator 2: Ephemeris Clock / Nope Content Splitter                      */
/* ========================================================================= */

ASTROSNAIL_API int32_t ephemeris_clock_nope_split_f32(
    const float *flux_stream,
    int32_t length,
    float threshold,
    uint8_t *out_clock_mask,
    float *out_nope_residual
);

/* ========================================================================= */
/* Operator 3: so(8) Crowded-Field Astrometry Triad Cascade                 */
/* ========================================================================= */

ASTROSNAIL_API int32_t so8_skew_triad_step_f32(
    const float *u,
    const float *v,
    const float *w,
    int32_t dim,
    float gamma,
    float *out_v
);

ASTROSNAIL_API int32_t so8_astrometry_triad_cascade_f32(
    const float *u_seq,
    int32_t n_steps,
    const float *v0,
    float coupling,
    float *out_v,
    float *out_radial_drift
);

ASTROSNAIL_API int32_t attribute_causal_progenitors_f32(
    const float *transition_matrix,
    int32_t n_objects,
    float threshold,
    uint8_t *out_is_generator,
    float *out_contingency_ratios
);

/* ========================================================================= */
/* Operator 4: 3D Cosmic-Web Z-Slab Tomography & Void Core Relaxation       */
/* ========================================================================= */

ASTROSNAIL_API int32_t cosmic_web_zslab_relax_f32(
    const float *skewer_flux,
    const int32_t *skewer_coords,
    int32_t n_skewers,
    int32_t z_chunk,
    int32_t y_dim,
    int32_t x_dim,
    float *out_delta_slab,
    float *out_phi_slab
);

ASTROSNAIL_API int32_t detect_jordan_cavity_cores_f32(
    const float *density,
    int32_t z_dim,
    int32_t y_dim,
    int32_t x_dim,
    float threshold,
    uint8_t *out_void_mask
);

ASTROSNAIL_API int32_t retract_void_step0_homotopy_f32(
    const float *density,
    const uint8_t *void_mask,
    int32_t total_voxels,
    float *out_density
);

/* ========================================================================= */
/* Operator 5: SO(6) Kelvin-Mandel Rotor & 3D Force-Free Beltrami Jet       */
/* ========================================================================= */

ASTROSNAIL_API int32_t so6_kelvin_mandel_rotor_f64(
    const double *r_so3,
    const double *c_base,
    double *m_so6_out,
    double *c_rot_out,
    double *trace_residual_out
);

ASTROSNAIL_API int32_t so6_kelvin_mandel_rotor_f32(
    const float *r_so3,
    const float *c_base,
    float *m_so6_out,
    float *c_rot_out,
    float *trace_residual_out
);

ASTROSNAIL_API int32_t solve_beltrami_jet_poincare_f32(
    const float *boundary_flux,
    int32_t z_dim,
    int32_t y_dim,
    int32_t x_dim,
    float mu,
    int32_t steps,
    float *out_b_field,
    float *out_radial_flux_escape
);

ASTROSNAIL_API int32_t beltrami_jet_poincare_step_f32(
    float x_in,
    float y_in,
    float z_in,
    float b0,
    float mu,
    float ds,
    float x0,
    float y0,
    float r0,
    float *out_x,
    float *out_y,
    float *out_z
);

#ifdef __cplusplus
}
#endif

#endif /* ASTROSNAIL_CORE_H */
