//! Empirical allocation audit integration test.
//! Uses a custom GlobalAlloc to trap and count all heap allocation attempts.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAlloc;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static REALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        REALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

use astrosnail_rs::*;

#[test]
fn test_zero_heap_allocations_across_10000_iterations() {
    let layout = Layout::new::<AstroSnailArena>();
    let raw_ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut AstroSnailArena };
    assert!(!raw_ptr.is_null(), "Failed to allocate AstroSnailArena on the heap");
    let mut arena: Box<AstroSnailArena> = unsafe { Box::from_raw(raw_ptr) };
    let arena_ptr = &mut *arena as *mut AstroSnailArena as *mut core::ffi::c_void;

    unsafe {
        assert_eq!(astrosnail_arena_init(arena_ptr, ASTROSNAIL_ARENA_TOTAL_BYTES), 0);
    }

    let input_frame = [1.5f32; 128 * 128];
    let clock_mask = [0u8; 128 * 128];
    let u_seq = [0.15f32; 16 * 8];
    let v0 = [0.35355339f32; 8];
    let mut v_out = [0.0f32; 8];
    let mut drift = 0.0f32;
    let r3 = [
        1.0, 0.0, 0.0,
        0.0, 0.0, -1.0,
        0.0, 1.0, 0.0,
    ];
    let mut c_base = [0.0f64; 36];
    for i in 0..6 { c_base[i * 6 + i] = (i + 1) as f64 * 3.0; }
    let mut m_out = [0.0f64; 36];
    let mut c_rot = [0.0f64; 36];
    let mut trace_res = 0.0f64;
    let boundary = [1.0f32; 24 * 24];
    let mut b_field = [0.0f32; 3 * 32 * 24 * 24];
    let mut flux_esc = 0.0f32;

    // Reset counters before benchmark loop
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    DEALLOC_COUNT.store(0, Ordering::SeqCst);
    REALLOC_COUNT.store(0, Ordering::SeqCst);

    let start_allocs = ALLOC_COUNT.load(Ordering::SeqCst);

    for _iter in 0..10_000 {
        unsafe {
            // Operator 1: Beltrami Morse gate
            let rc1 = beltrami_morse_transient_gate_f32(
                input_frame.as_ptr(),
                core::ptr::null(),
                clock_mask.as_ptr(),
                128,
                128,
                arena.salience_map.as_mut_ptr(),
                arena.mask_spike.as_mut_ptr(),
                core::ptr::null_mut(),
            );
            assert_eq!(rc1, 0);

            // Operator 1b: MAD floor calculation
            let _mad = compute_spatial_mad_floor_256_c(
                arena.salience_map.as_ptr(),
                clock_mask.as_ptr(),
                128 * 128,
            );

            // Operator 2: Ephemeris Clock / Nope split
            let rc2 = ephemeris_clock_nope_split_f32(
                input_frame.as_ptr(),
                256,
                0.85,
                arena.mask_clock.as_mut_ptr(),
                arena.spatial_prev.as_mut_ptr(),
            );
            assert_eq!(rc2, 0);

            // Operator 3: so(8) triad cascade
            let rc3 = so8_astrometry_triad_cascade_f32(
                u_seq.as_ptr(),
                16,
                v0.as_ptr(),
                0.15,
                v_out.as_mut_ptr(),
                &mut drift,
            );
            assert_eq!(rc3, 0);
            assert_eq!(drift, 0.0);

            // Operator 4: SO(6) Kelvin-Mandel rotor
            let rc4 = so6_kelvin_mandel_rotor_f64(
                r3.as_ptr(),
                c_base.as_ptr(),
                m_out.as_mut_ptr(),
                c_rot.as_mut_ptr(),
                &mut trace_res,
            );
            assert_eq!(rc4, 0);

            // Operator 5: Beltrami jet solver
            let rc5 = solve_beltrami_jet_poincare_f32(
                boundary.as_ptr(),
                32,
                24,
                24,
                1.0,
                16,
                b_field.as_mut_ptr(),
                &mut flux_esc,
            );
            assert_eq!(rc5, 0);
            assert_eq!(flux_esc, 0.0);
        }
    }

    let end_allocs = ALLOC_COUNT.load(Ordering::SeqCst);
    let end_deallocs = DEALLOC_COUNT.load(Ordering::SeqCst);
    let end_reallocs = REALLOC_COUNT.load(Ordering::SeqCst);

    println!(
        "ALLOCATION AUDIT RESULTS (10,000 iterations): allocs={}, deallocs={}, reallocs={}",
        end_allocs - start_allocs,
        end_deallocs,
        end_reallocs
    );

    assert_eq!(end_allocs - start_allocs, 0, "Heap allocations detected during 10,000 iterations!");
    assert_eq!(end_deallocs, 0, "Heap deallocations detected during 10,000 iterations!");
    assert_eq!(end_reallocs, 0, "Heap reallocations detected during 10,000 iterations!");
}
