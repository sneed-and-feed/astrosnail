"""
AstroSnail Discovery Expedition Package (`astrosnail.expedition`)
================================================================
Modules:
  - `harvester`: Multi-survey target query, caching, manifest, and out-of-core streaming.
  - `scanner`: Anomaly scanner via Beltrami-Morse shell and H_clock / H_nope content splitting (M7.2).
  - `vetter`: Ephemeris cross-matching and false-positive vetting engine (M7.3).
  - `atlas`: Expedition dossier, studio catalog exporter, and report generator (M7.4).
"""

from astrosnail.expedition.harvester import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SEED,
    ExpeditionHarvester,
    FitsBinTableContext,
    HarvestedTarget,
    TargetManifest,
    TargetManifestEntry,
    get_target_by_id,
    harvest_expedition_targets,
    parse_bintable_descriptor,
    stream_harvested_targets,
    write_fits_bintable,
)
from astrosnail.expedition.scanner import (
    CandidateAnomaly,
    ExpeditionScanner,
    scan_target,
    scan_track_a,
    scan_track_b,
    scan_track_c,
)
from astrosnail.expedition.vetter import (
    AstrophysicalConsistencyVetter,
    CatalogEphemerisEntry,
    EphemerisCrossMatcher,
    EphemerisMatcher,
    ExpeditionVetter,
    InstrumentalFalsePositiveFilter,
    InstrumentalTriageEngine,
    InstrumentalVettingReceipt,
    PhysicalConsistencyEngine,
    PhysicalVettingReceipt,
    VettedCandidate,
    compute_composite_vetting_score,
    vet_candidate,
    vet_target,
)
from astrosnail.expedition.atlas import (
    ExpeditionAtlas,
    ExpeditionReport,
    create_curated_reference_portfolio,
    run_discovery_expedition,
)

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_SEED",
    "ExpeditionHarvester",
    "FitsBinTableContext",
    "HarvestedTarget",
    "TargetManifest",
    "TargetManifestEntry",
    "get_target_by_id",
    "harvest_expedition_targets",
    "parse_bintable_descriptor",
    "stream_harvested_targets",
    "write_fits_bintable",
    "CandidateAnomaly",
    "ExpeditionScanner",
    "scan_target",
    "scan_track_a",
    "scan_track_b",
    "scan_track_c",
    "AstrophysicalConsistencyVetter",
    "CatalogEphemerisEntry",
    "EphemerisCrossMatcher",
    "EphemerisMatcher",
    "ExpeditionVetter",
    "InstrumentalFalsePositiveFilter",
    "InstrumentalTriageEngine",
    "InstrumentalVettingReceipt",
    "PhysicalConsistencyEngine",
    "PhysicalVettingReceipt",
    "VettedCandidate",
    "compute_composite_vetting_score",
    "vet_candidate",
    "vet_target",
    "ExpeditionAtlas",
    "ExpeditionReport",
    "create_curated_reference_portfolio",
    "run_discovery_expedition",
]

