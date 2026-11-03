"""
astrosnail: Time-domain sky-survey outlier engine, sub-pixel moving-body astrometry tracker,
and 3D cosmic-web tomography workstation synced to triad-core and formalization/OcclusionTriad.lean.
"""

__version__ = "0.2.0"

# Transients exports
from astrosnail.transients import (
    BeltramiLightCurveShell,
    compute_morse_impulse,
    MorseTransientGateConfig,
    TemporalClockContentSplitter,
    compute_spatial_mad_floor_256,
)

# Astrometry exports
from astrosnail.astrometry import (
    RoPE2DMedoidPooler,
    SkewSymmetricTriadOperator,
    SlowFastTriadCascade,
    attribute_causal_progenitors,
)

# Cosmology exports
from astrosnail.cosmology import (
    reconstruct_zslab_tomography,
    detect_jordan_cavity_cores,
    so6_kelvin_mandel_rotor,
    retract_void_step0_homotopy,
    blind_astro_let_decouple,
    solve_beltrami_jet_poincare,
    compute_stellar_wind_bow_shock_jump,
    SkewSymmetricCosmicTriad,
)

# Scenarios exports
from astrosnail.scenarios import (
    ScenarioData,
    get_scenario,
    generate_scenario_01_frb_supernova,
    generate_scenario_02_eclipsing_binary,
    generate_scenario_03_crowded_field,
    generate_scenario_04_lensed_quasar,
    generate_scenario_05_cosmic_web,
    SCENARIO_REGISTRY,
    SCENARIO_ALIASES,
)

# I/O exports
from astrosnail.io import (
    FITSHeader,
    read_fits_header,
    stream_fits_cube,
    write_fits_cube,
    stream_lc_records,
    stream_lc_chunks,
    parse_lc_table,
    write_lc_file,
    write_lc_table,
    stream_cat_records,
    stream_cat_chunks,
    parse_cat_file,
    write_cat_file,
    write_cat_table,
    generate_standard_catalogs,
    profile_stream_memory,
)

# Engine exports
from astrosnail.engine import (
    AstroSnailEngine,
    verify_formal_theorems,
)

# Benchmark exports
from astrosnail.benchmark import (
    UniformBinningSigmaClip,
    NaiveFrameDiffCentroid,
    PatchAvgKalmanIoU,
    NaiveIsotropicRBF,
    TunedAnisotropicMaternKriging,
    StandardDampedLSQR_RSD,
    metric_subcadence_transient_recall,
    metric_glint_false_trigger_rate,
    metric_crowded_track_id_swaps,
    metric_subpixel_astrometric_rmse,
    metric_causal_attribution_f1,
    metric_leave_one_skewer_out_rmse,
    metric_void_boundary_sharpness,
    metric_rsd_smearing_error,
    metric_single_core_fps,
    metric_peak_active_heap,
    evaluate_scenario_head_to_head,
    run_full_benchmark_suite,
)

__all__ = [
    "__version__",
    # Transients
    "BeltramiLightCurveShell",
    "compute_morse_impulse",
    "MorseTransientGateConfig",
    "TemporalClockContentSplitter",
    "compute_spatial_mad_floor_256",
    # Astrometry
    "RoPE2DMedoidPooler",
    "SkewSymmetricTriadOperator",
    "SlowFastTriadCascade",
    "attribute_causal_progenitors",
    # Cosmology
    "reconstruct_zslab_tomography",
    "detect_jordan_cavity_cores",
    "so6_kelvin_mandel_rotor",
    "retract_void_step0_homotopy",
    "blind_astro_let_decouple",
    "solve_beltrami_jet_poincare",
    "compute_stellar_wind_bow_shock_jump",
    "SkewSymmetricCosmicTriad",
    # Scenarios
    "ScenarioData",
    "get_scenario",
    "generate_scenario_01_frb_supernova",
    "generate_scenario_02_eclipsing_binary",
    "generate_scenario_03_crowded_field",
    "generate_scenario_04_lensed_quasar",
    "generate_scenario_05_cosmic_web",
    "SCENARIO_REGISTRY",
    "SCENARIO_ALIASES",
    # I/O
    "FITSHeader",
    "read_fits_header",
    "stream_fits_cube",
    "write_fits_cube",
    "stream_lc_records",
    "stream_lc_chunks",
    "parse_lc_table",
    "write_lc_file",
    "write_lc_table",
    "stream_cat_records",
    "stream_cat_chunks",
    "parse_cat_file",
    "write_cat_file",
    "write_cat_table",
    "generate_standard_catalogs",
    "profile_stream_memory",
    # Engine
    "AstroSnailEngine",
    "verify_formal_theorems",
    # Benchmark
    "UniformBinningSigmaClip",
    "NaiveFrameDiffCentroid",
    "PatchAvgKalmanIoU",
    "NaiveIsotropicRBF",
    "TunedAnisotropicMaternKriging",
    "StandardDampedLSQR_RSD",
    "metric_subcadence_transient_recall",
    "metric_glint_false_trigger_rate",
    "metric_crowded_track_id_swaps",
    "metric_subpixel_astrometric_rmse",
    "metric_causal_attribution_f1",
    "metric_leave_one_skewer_out_rmse",
    "metric_void_boundary_sharpness",
    "metric_rsd_smearing_error",
    "metric_single_core_fps",
    "metric_peak_active_heap",
    "evaluate_scenario_head_to_head",
    "run_full_benchmark_suite",
]
