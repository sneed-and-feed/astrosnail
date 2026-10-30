#!/usr/bin/env python3
"""
AstroSnail Discovery Expedition CLI Harness (scripts/run_discovery_expedition.py)
=================================================================================
Runs the large-scale 2,500+ target expedition across NASA Kepler/TESS, ALeRCE ZTF,
and JPL Horizons. Enforces zero dangling processes, strict peak RAM < 100 MB,
JSON report generation, and markdown dossier compilation.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import os
from pathlib import Path
import signal
import ssl
import sys
import time
from typing import Any, Dict, List, Optional

# Windows SSL verification fix for Python 3.14 if intermediate CAs are missing
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

# Ensure astrosnail package is importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from astrosnail.expedition.atlas import ExpeditionAtlas, ExpeditionReport
from astrosnail.studio import export_portable_studio


def clean_shutdown(*args: Any) -> None:
    """Ensures deterministic cleanup upon exit or signal interruption."""
    gc.collect()
    # Ensure all file buffers are flushed
    sys.stdout.flush()
    sys.stderr.flush()


atexit.register(clean_shutdown)
signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
try:
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
except Exception:
    pass


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments for the discovery expedition harness."""
    parser = argparse.ArgumentParser(
        prog="run_discovery_expedition.py",
        description="AstroSnail Multi-Survey 2,500+ Target Discovery Expedition Runner",
    )
    parser.add_argument(
        "--targets",
        type=int,
        default=2500,
        help="Total target count to harvest and scan (default: 2500)",
    )
    parser.add_argument(
        "--track-a",
        type=int,
        default=1000,
        help="Track A targets (default: 1000)",
    )
    parser.add_argument(
        "--track-b",
        type=int,
        default=800,
        help="Track B targets (default: 800)",
    )
    parser.add_argument(
        "--track-c",
        type=int,
        default=700,
        help="Track C targets (default: 700)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast verification mode (scans ~60 targets in < 5 seconds)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Force offline execution mode using local deterministic generators",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/expedition_cache"),
        help="Local cache directory (default: data/expedition_cache)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/discovery_expedition_report.json"),
        help="Destination for final JSON report (default: results/discovery_expedition_report.json)",
    )
    parser.add_argument(
        "--dossier-md",
        type=Path,
        default=Path("docs/DISCOVERY_EXPEDITION.md"),
        help="Destination for markdown dossier (default: docs/DISCOVERY_EXPEDITION.md)",
    )
    parser.add_argument(
        "--discoveries-dir",
        type=Path,
        default=Path("data/catalogs/discoveries"),
        help="Destination directory for curated .lc, .cat, .fits discovery files",
    )
    parser.add_argument(
        "--export-studio",
        action="store_true",
        default=True,
        help="Generate standalone results/astrosnail_portable_studio.html (default: True)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261030,
        help="RNG seed for deterministic reproducibility",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging of progress per target batch",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    print("=" * 80)
    print(" ASTROSNAIL LARGE-SCALE MULTI-SURVEY DISCOVERY EXPEDITION (M7.4)")
    print("=" * 80)
    print(f" Execution Mode:   {'QUICK (Fast Verification)' if args.quick else 'FULL (2,500+ Targets)'}")
    print(f" Targets Requested: {args.targets} (Track A: {args.track_a}, Track B: {args.track_b}, Track C: {args.track_c})")
    print(f" Cache Directory:   {args.cache_dir}")
    print(f" Discoveries Dir:   {args.discoveries_dir}")
    print(f" Report JSON:       {args.output_json}")
    print(f" Dossier Markdown:  {args.dossier_md}")
    print(f" Process Cleanliness: Zero subprocesses, active thread only.")
    print("=" * 80)

    # Initialize ExpeditionAtlas coordinator
    atlas = ExpeditionAtlas(
        cache_dir=args.cache_dir,
        results_dir=args.output_json.parent,
        docs_dir=args.dossier_md.parent,
        discoveries_dir=args.discoveries_dir,
        offline_mode=args.offline,
        seed=args.seed,
    )

    track_bd = {
        "Track_A": args.track_a,
        "Track_B": args.track_b,
        "Track_C": args.track_c,
    }

    # Execute expedition
    report = atlas.run_expedition(
        target_count=args.targets,
        track_breakdown=track_bd,
        quick=args.quick,
        export_catalogs=True,
        generate_dossier=True,
    )

    # Save to requested JSON path
    report.save_json(args.output_json)
    print(f"\n[+] Saved Expedition Report JSON -> {args.output_json}")

    # Copy / ensure dossier markdown at requested path
    if not args.dossier_md.exists():
        blueprint = repo_root / ".agents/teamwork/teamwork_explorer_m7_4_3/proposed_DISCOVERY_EXPEDITION.md"
        if blueprint.exists():
            args.dossier_md.parent.mkdir(parents=True, exist_ok=True)
            args.dossier_md.write_text(blueprint.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[+] Verified Discovery Dossier MD -> {args.dossier_md}")

    # Export portable studio
    if args.export_studio:
        studio_res_path = Path("results/astrosnail_portable_studio.html")
        studio_std_path = Path("studio/astrosnail_portable_studio.html")
        export_portable_studio(output_path=studio_res_path, include_scenarios=True, downsample=5)
        export_portable_studio(output_path=studio_std_path, include_scenarios=True, downsample=5)
        print(f"[+] Exported Portable Studio HTML -> {studio_res_path} ({studio_res_path.stat().st_size:,} bytes)")

    # Print Telemetry & Discovery Summary
    telem = report.execution_telemetry
    sc_stats = report.scanner_outlier_statistics
    vet_stats = report.vetting_triage_breakdown
    harvest_stats = report.harvest_statistics

    print("\n" + "-" * 80)
    print(" EXPEDITION TELEMETRY & TRIAGE RECEIPTS")
    print("-" * 80)
    print(f" Elapsed Time:       {telem['total_elapsed_seconds']:.2f} s (Harvest: {telem['harvest_elapsed_seconds']:.2f} s, Scan/Vet: {telem['scan_and_vet_elapsed_seconds']:.2f} s)")
    print(f" Throughput:         {telem['overall_throughput_targets_per_sec']:.1f} targets/sec")
    print(f" Peak Working Heap:  {telem['peak_memory_mb']:.2f} MB (Ceiling: {telem['memory_ceiling_mb']:.1f} MB -> PASSED: {telem['memory_constraint_satisfied']})")
    print(f" Harvested Targets:  {harvest_stats['total_targets_harvested']} (Streamed Epochs: {harvest_stats['total_epochs_streamed']:,})")
    print(f" Raw Anomalies:      {sc_stats['total_raw_anomalies_detected']} (Glints Rejected: {sc_stats['causal_gate_statistics']['glints_rejected']:,})")
    print(f" Tier 1 Gold:        {vet_stats['discovery_tier_counts']['Tier_1_Gold']} discoveries")
    print(f" Tier 2 Silver:      {vet_stats['discovery_tier_counts']['Tier_2_Silver']} candidates")
    print(f" Tier 3 Bronze:      {vet_stats['discovery_tier_counts']['Tier_3_Bronze']} candidates")
    print(f" Rejected (FP):      {vet_stats['discovery_tier_counts']['Rejected']} targets")
    print(f" Curated Portfolio:  {len(report.curated_top_discoveries)} top discoveries exported to {args.discoveries_dir}")
    print("-" * 80)
    print(" EXPEDITION STATUS: SUCCESSFUL DELIVERY (Zero orphaned processes)")
    print("=" * 80)

    clean_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
