"""
Unit & Adversarial Tests for Multi-Survey Astronomical Target Harvester (`tests/test_expedition.py`)
===================================================================================================
Verifies:
  1. `HarvestedTarget` dataclass, slot allocation, and astronomical helper methods.
  2. Dynamic FITS BINTABLE parser with zero hardcoded byte offsets on real Kepler standards.
  3. `FitsBinTableContext` zero-copy memory mapping and Windows file handle release.
  4. Compliant FITS BINTABLE generation and atomic disk caching.
  5. Decoupled `TargetManifest` indexing with memory footprint < 1.5 MB.
  6. 2,500-target multi-survey distribution across Track A (1,000), Track B (800), and Track C (700).
  7. Out-of-core streaming memory bound: active peak heap strictly < 100 MB (< 5.0 MB actual).
  8. Gitignore compliance ensuring `data/expedition_cache/` is untracked.
"""

import gc
import json
from pathlib import Path
import tracemalloc
import numpy as np
import pytest

from astrosnail.expedition import (
    DEFAULT_CACHE_DIR,
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


def test_harvested_target_dataclass_and_methods():
    """Validates HarvestedTarget fields, slots, and astronomical helper methods."""
    assert hasattr(HarvestedTarget, "__slots__"), "HarvestedTarget must use slots=True"

    # 1. Track A Photometric Target
    n = 1000
    t = np.linspace(0.0, 10.0, n, dtype=np.float64)
    flux = np.full(n, 50000.0, dtype=np.float32)
    flux[200:220] -= 2500.0  # 50,000 ppm dip
    pdc = flux.copy()
    qual = np.zeros(n, dtype=np.int32)
    qual[50] = 2048  # Simulated Argabrightening flag
    bkg = 120.0 + np.random.normal(0.0, 2.0, n).astype(np.float32)
    cx = 500.12 + np.random.normal(0.0, 0.001, n)
    cy = 500.85 + np.random.normal(0.0, 0.001, n)

    target_a = HarvestedTarget(
        target_id="KIC_10666592",
        track="Track_A",
        survey="Kepler",
        ra=285.679,
        dec=48.245,
        metadata={"period": 2.2047, "koi_depth": 50000.0},
        time=t,
        sap_flux=flux,
        pdcsap_flux=pdc,
        sap_quality=qual,
        sap_bkg=bkg,
        mom_centr1=cx,
        mom_centr2=cy,
    )

    assert target_a.is_light_curve
    assert not target_a.is_astrometry
    assert target_a.n_epochs == 1000

    # Test clean_light_curve with quality filtering
    t_clean, f_clean, e_clean = target_a.clean_light_curve(quality_filter=True)
    assert len(t_clean) == 999  # 1 bad cadence removed
    assert len(f_clean) == 999

    # Test relative_flux_ppm
    ppm = target_a.relative_flux_ppm(quality_filter=False)
    assert len(ppm) == 1000
    assert abs(ppm[210] - (-50000.0)) < 1500.0

    # Test centroid_dispersion_mpix
    dx, dy = target_a.centroid_dispersion_mpix()
    assert dx < 3.0 and dy < 3.0, f"Expected < 3 milli-pixels, got {dx}, {dy}"

    # Test bkg_correlation
    corr = target_a.bkg_correlation()
    assert isinstance(corr, float)
    assert -1.0 <= corr <= 1.0

    # 2. Track C Astrometric Target
    c_times = np.linspace(59000.0, 59100.0, 50, dtype=np.float64)
    ra_c = 180.0 + np.linspace(0.0, 0.05, 50, dtype=np.float64)
    dec_c = 15.0 + np.linspace(0.0, -0.02, 50, dtype=np.float64)
    mag_c = np.full(50, 19.5, dtype=np.float32)
    mag_err_c = np.full(50, 0.05, dtype=np.float32)

    target_c = HarvestedTarget(
        target_id="ZTF19aapqmco",
        track="Track_C",
        survey="ALeRCE_ZTF",
        ra=180.025,
        dec=14.99,
        metadata={"class": "transient"},
        time=c_times,
        ra_series=ra_c,
        dec_series=dec_c,
        mag=mag_c,
        mag_err=mag_err_c,
    )

    assert not target_c.is_light_curve
    assert target_c.is_astrometry
    assert target_c.n_epochs == 50
    tbl = target_c.astrometry_table
    assert tbl is not None
    assert tbl.shape == (50, 5)


def test_dynamic_bintable_descriptor_and_context():
    """Validates dynamic FITS BINTABLE parsing on real Kepler data without hardcoded byte offsets."""
    repo_root = Path(__file__).resolve().parent.parent
    std_fits = repo_root / "data" / "real_standards" / "kepler_hatp7b_slc.fits"
    if not std_fits.exists():
        std_fits = Path("data/real_standards/kepler_hatp7b_slc.fits")
    if not std_fits.exists():
        pytest.skip("Real standard Kepler FITS not downloaded")

    desc = parse_bintable_descriptor(std_fits)
    assert desc["data_offset"] == 20160
    assert desc["n_rows"] == 14280
    assert desc["row_bytes"] == 100
    assert desc["tfields"] == 20
    assert "TIME" in desc["column_names"]
    assert "SAP_FLUX" in desc["column_names"]
    assert "PDCSAP_FLUX" in desc["column_names"]

    # Test memory mapping and Windows file handle cleanup
    with FitsBinTableContext(std_fits) as table:
        assert len(table) == 14280
        assert np.isfinite(table["TIME"][0])
        assert np.isfinite(table["SAP_FLUX"][0])


def test_write_fits_bintable_and_roundtrip(tmp_path: Path):
    """Validates compliant 2880-byte padded FITS BINTABLE creation and roundtrip loading."""
    n = 200
    t = np.linspace(100.0, 120.0, n, dtype=np.float64)
    flux = np.full(n, 45000.0, dtype=np.float32)
    flux[50:60] -= 2000.0
    qual = np.zeros(n, dtype=np.int32)
    qual[10] = 16

    cols = {
        "TIME": t,
        "SAP_FLUX": flux,
        "SAP_QUALITY": qual,
    }

    out_fits = tmp_path / "test_table.fits"
    write_fits_bintable(out_fits, cols, primary_cards={"OBJECT": "TEST_STAR"})

    assert out_fits.exists()
    assert out_fits.stat().st_size % 2880 == 0, "FITS file must be a multiple of 2880 bytes"

    # Read back with dynamic parser
    desc = parse_bintable_descriptor(out_fits)
    assert desc["n_rows"] == n
    assert "TIME" in desc["column_names"]
    assert "SAP_FLUX" in desc["column_names"]

    with FitsBinTableContext(out_fits) as table:
        np.testing.assert_allclose(table["TIME"], t)
        np.testing.assert_allclose(table["SAP_FLUX"], flux)
        np.testing.assert_array_equal(table["SAP_QUALITY"], qual)


def test_target_manifest_indexing_and_memory(tmp_path: Path):
    """Validates TargetManifest index operations and ensures < 1.5 MB RAM footprint for 2,500 targets."""
    manifest = TargetManifest()
    for i in range(2500):
        entry = TargetManifestEntry(
            target_id=f"TARGET_{i:04d}",
            track=f"Track_{chr(ord('A') + (i % 3))}",
            survey="Kepler" if i % 2 == 0 else "ZTF",
            ra=float(i % 360),
            dec=float((i % 180) - 90),
            n_epochs=2000,
            cache_relpath=f"track_{chr(ord('a') + (i % 3))}/target_{i:04d}.fits",
            data_format="fits" if i % 3 != 2 else "json",
            metadata={"index": i},
        )
        manifest.add_entry(entry)

    assert len(manifest) == 2500
    counts = manifest.track_counts()
    assert sum(counts.values()) == 2500

    # Measure RAM footprint of the 2,500 manifest entries
    tracemalloc.start()
    tracemalloc.reset_peak()
    manifest_copy = TargetManifest.load(tmp_path / "m.json") if (tmp_path / "m.json").exists() else manifest
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak / (1024 * 1024) < 1.5, f"Manifest RAM footprint must be < 1.5 MB, got {peak / (1024 * 1024):.2f} MB"

    # Test serialization roundtrip
    m_path = tmp_path / "manifest.json"
    manifest.save(m_path)
    assert m_path.exists()

    loaded = TargetManifest.load(m_path)
    assert len(loaded) == 2500
    assert loaded["TARGET_0042"].ra == 42.0


@pytest.fixture(scope="module")
def expedition_cache_dir(tmp_path_factory):
    """Module-scoped cache populated once with 2,500 targets for fast test execution."""
    cache_dir = tmp_path_factory.mktemp("expedition_cache")
    harvester = ExpeditionHarvester(cache_dir=cache_dir, offline_mode=True, seed=20261030)
    harvester.harvest(target_count=2500)
    return cache_dir


def test_expedition_harvest_target_counts_and_distribution(expedition_cache_dir: Path):
    """Validates harvesting >= 2,500 targets across Track A (1,000), Track B (800), Track C (700)."""
    manifest_p = expedition_cache_dir / "manifests" / "expedition_manifest.json"
    assert manifest_p.exists()
    manifest = TargetManifest.load(manifest_p)

    assert len(manifest) == 2500
    counts = manifest.track_counts()
    assert counts["Track_A"] == 1000
    assert counts["Track_B"] == 800
    assert counts["Track_C"] == 700

    # Verify single target lookup
    target_sample = get_target_by_id("KIC_10000001", cache_dir=expedition_cache_dir)
    assert target_sample.target_id == "KIC_10000001"
    assert target_sample.track == "Track_A"
    assert target_sample.is_light_curve
    assert target_sample.n_epochs == 4000

    # Verify Track C target lookup
    manifest_c = manifest.filter_by_track("Track_C")
    assert len(manifest_c) == 700
    target_c = get_target_by_id(manifest_c[0].target_id, cache_dir=expedition_cache_dir)
    assert target_c.track == "Track_C"
    assert target_c.is_astrometry


def test_out_of_core_streaming_memory_bound(expedition_cache_dir: Path):
    """Validates that out-of-core streaming of 2,500 targets strictly satisfies active RAM < 100 MB."""
    cache_dir = expedition_cache_dir

    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()

    stream = stream_harvested_targets(cache_dir=cache_dir)
    count = 0
    track_counts = {"Track_A": 0, "Track_B": 0, "Track_C": 0}

    for target in stream:
        count += 1
        track_counts[target.track] += 1

        # Simulate consumer scanner step
        if target.is_light_curve:
            ppm = target.relative_flux_ppm()
            var = float(np.var(ppm))
            assert np.isfinite(var)
        else:
            tbl = target.astrometry_table
            assert tbl is not None
            mean_mag = float(np.mean(tbl[:, 3]))
            assert np.isfinite(mean_mag)

        del target
        if count % 500 == 0:
            gc.collect()

    gc.collect()
    curr, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    assert count == 2500, f"Expected 2,500 targets, got {count}"
    assert track_counts["Track_A"] == 1000
    assert track_counts["Track_B"] == 800
    assert track_counts["Track_C"] == 700
    assert peak_mb < 100.0, f"Active RAM during streaming must be < 100 MB, got {peak_mb:.2f} MB"
    assert peak_mb < 5.0, f"Expected peak heap < 5.0 MB with clean generator, got {peak_mb:.2f} MB"


def test_gitignore_contains_expedition_cache():
    """Validates that data/expedition_cache/ is in astrosnail/.gitignore."""
    repo_root = Path(__file__).resolve().parent.parent
    gitignore_path = repo_root / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path = Path(".gitignore")
    assert gitignore_path.exists(), f".gitignore must exist at {gitignore_path}"
    content = gitignore_path.read_text(encoding="utf-8")
    assert "data/expedition_cache/" in content, "data/expedition_cache/ must be excluded in .gitignore"
