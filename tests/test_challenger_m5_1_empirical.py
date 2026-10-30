"""
tests/test_challenger_m5_1_empirical.py
========================================
Adversarial Empirical Challenge Suite for Milestone 5:
Portable Studio Workstation & Ephemeral Process Lifecycle.

Challenger: Challenger M5-1
Focus:
1. Offline Adversarial Audit of studio/astrosnail_portable_studio.html
2. Deep Verification of Embedded Demonstration Scenario Datasets
3. Process Teardown, Socket Re-bind, and Subprocess Timeout Auditing
4. Extreme Downsampling, Stress Export, and CLI Concurrency
"""

from __future__ import annotations

import concurrent.futures
import http.client
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from astrosnail.studio import (
    export_portable_studio,
    verify_studio_html,
    serve_studio,
    serialize_scenario_payloads,
    serialize_theorems_payload,
    serialize_benchmark_payload,
)

STUDIO_HTML = Path("studio/astrosnail_portable_studio.html")


def _get_embedded_scenarios() -> dict:
    content = STUDIO_HTML.read_text(encoding="utf-8")
    match = re.search(r'window\.ASTROSNAIL_SCENARIOS\s*=\s*(\{.*?\});\s*window\.ASTROSNAIL_THEOREMS', content, re.DOTALL)
    assert match is not None, "window.ASTROSNAIL_SCENARIOS block not found!"
    return json.loads(match.group(1))


# =============================================================================
# 1. OFFLINE ADVERSARIAL AUDIT
# =============================================================================

class TestOfflineAdversarialAudit:
    """Rigorous offline isolation audit of the production studio HTML artifact."""

    def test_artifact_exists_and_size_exceeds_50kb(self):
        """Verify the pre-built workstation artifact exists and is >= 50 KB."""
        assert STUDIO_HTML.exists(), f"Workstation artifact missing at {STUDIO_HTML}"
        file_size = STUDIO_HTML.stat().st_size
        assert file_size > 50_000, f"File size {file_size} is under 50,000 bytes!"
        print(f"\n[Audit] Verified file size: {file_size:,} bytes (> 50 KB)")

    def test_zero_remote_uris_and_network_calls(self):
        """Strict regex check for any external http/https/protocol-relative network resources."""
        content = STUDIO_HTML.read_text(encoding="utf-8")

        # 1. Any <script src="...">
        script_srcs = re.findall(r'<script[^>]+src=[\'"]([^\'"]+)[\'"]', content, re.IGNORECASE)
        for s in script_srcs:
            assert not s.startswith(("http://", "https://", "//")), f"Remote script detected: {s}"

        # 2. Any <link href="...">
        link_hrefs = re.findall(r'<link[^>]+href=[\'"]([^\'"]+)[\'"]', content, re.IGNORECASE)
        for l in link_hrefs:
            assert not l.startswith(("http://", "https://", "//")), f"Remote stylesheet detected: {l}"

        # 3. Any CSS @import
        css_imports = re.findall(r'@import\s+(?:url\([\'"]?([^\'")]+)[\'"]?\)|[\'"]([^\'"]+)[\'"])', content, re.IGNORECASE)
        for imp in css_imports:
            target = imp[0] or imp[1]
            assert not target.startswith(("http://", "https://", "//")), f"Remote CSS @import: {target}"

        # 4. Any external font or resource URLs
        urls = re.findall(r'url\([\'"]?([^\'")]+)[\'"]?\)', content, re.IGNORECASE)
        for u in urls:
            u_clean = u.strip()
            if u_clean.startswith("data:"):
                continue  # Embedded base64 data URIs are offline
            assert not u_clean.startswith(("http://", "https://", "//")), f"Remote url() reference: {u_clean}"

        # 5. Check for well-known CDN keywords in resource loading context
        cdn_keywords = ["cdn.", "unpkg.com", "cdnjs.cloudflare.com", "cdn.jsdelivr.net", "fonts.googleapis.com", "fonts.gstatic.com"]
        for cdn in cdn_keywords:
            assert cdn not in content.lower(), f"Detected CDN domain reference: {cdn}"

    def test_viewport_structure_and_canvases(self):
        """Verify the exact DOM structure for all 4 synchronized observatory viewports."""
        content = STUDIO_HTML.read_text(encoding="utf-8")

        # All four quads must exist
        for quad in ["quad-a", "quad-b", "quad-c", "quad-d"]:
            assert f'id="{quad}"' in content, f"Missing viewport element #{quad}"

        # Dedicated canvases for all 4 quads
        for canvas in ["canvas-quad-a", "canvas-quad-b", "canvas-quad-c", "canvas-quad-d"]:
            assert f'id="{canvas}"' in content, f"Missing canvas element #{canvas}"

        # Key observatory control elements
        controls = [
            "workspace-grid",
            "timeline-scrub",
            "clock-display",
            "scenario-select",
            "btn-play",
            "btn-step-prev",
            "btn-step-next",
            "cut-x",
            "cut-y",
            "cut-z",
            "file-dropzone",
        ]
        for ctrl in controls:
            assert f'id="{ctrl}"' in content, f"Missing control element #{ctrl}"

    def test_scenario_01_payload_cosmic_rays_and_glitches(self):
        """Adversarial check: Scenario 01 must embed non-empty cosmic ray indices and glitch indices."""
        scenarios = _get_embedded_scenarios()
        sc01 = scenarios.get("01_frb_supernova_cosmic_ray_triage")
        assert sc01 is not None
        assert len(sc01["time"]) > 100, "Scenario 01 time vector empty or too short"
        assert len(sc01["flux"]) == len(sc01["time"])
        assert sc01["t_frb"] == 10000.0
        # Bug check: cosmic_ray_indices and glitch_indices must NOT be empty lists!
        assert len(sc01["cosmic_ray_indices"]) > 0, (
            "BUG: sc01['cosmic_ray_indices'] is empty! "
            "astrosnail/studio.py queried 'cosmic_ray_indices' instead of 'cosmic_ray_epochs' from scenarios.py"
        )
        assert len(sc01["glitch_indices"]) > 0, (
            "BUG: sc01['glitch_indices'] is empty! "
            "astrosnail/studio.py queried 'glitch_indices' instead of 'glitch_mask' from scenarios.py"
        )

    def test_scenario_02_payload_clock_nope_split_and_periods(self):
        """Adversarial check: Scenario 02 must embed non-empty h_clock, h_nope, and true periods."""
        scenarios = _get_embedded_scenarios()
        sc02 = scenarios.get("02_eclipsing_binary_exoplanet_transit_split")
        assert sc02 is not None
        assert len(sc02["time"]) > 100
        assert len(sc02["flux"]) == len(sc02["time"])
        # Bug check: h_clock and h_nope must NOT be empty lists!
        assert len(sc02["h_clock"]) > 0, (
            "BUG: sc02['h_clock'] is empty! "
            "astrosnail/studio.py queried 'h_clock' instead of 'h_clock_true' from scenarios.py"
        )
        assert len(sc02["h_nope"]) > 0, (
            "BUG: sc02['h_nope'] is empty! "
            "astrosnail/studio.py queried 'h_nope' instead of 'h_nope_true' from scenarios.py"
        )
        # Check actual orbital periods from scenarios.py (P1 = 3.412d, P_puls = 0.567d, P2 = 18.71d)
        assert abs(sc02["period_eb"] - 3.412) < 0.01, f"BUG: period_eb {sc02['period_eb']} is wrong default fallback (expected 3.412)!"
        assert abs(sc02["period_rr"] - 0.567) < 0.01, f"BUG: period_rr {sc02['period_rr']} is wrong default fallback (expected 0.567)!"
        assert abs(sc02["period_transit"] - 18.71) < 0.01, f"BUG: period_transit {sc02['period_transit']} is wrong default fallback (expected 18.71)!"

    def test_scenario_03_payload_background_stars(self):
        """Adversarial check: Scenario 03 background stars must contain all 500 stars at non-zero positions."""
        scenarios = _get_embedded_scenarios()
        sc03 = scenarios.get("03_crowded_field_kbo_crossing_astrometry")
        assert sc03 is not None
        assert len(sc03["stars"]) >= 500, f"BUG: sc03 stars length {len(sc03['stars'])} < 500!"
        all_zeros = all(s[0] == 0.0 and s[1] == 0.0 for s in sc03["stars"])
        assert not all_zeros, "BUG: All stars in sc03 are at [0.0, 0.0, 1.0] dummy fallback coordinates!"

    def test_scenario_04_payload_causal_curves(self):
        """Adversarial check: Scenario 04 must embed all 12 curves and progenitors."""
        scenarios = _get_embedded_scenarios()
        sc04 = scenarios.get("04_lensed_quasar_echo_swarm_factorization")
        assert sc04 is not None
        assert len(sc04["curves"]) == 12
        assert len(sc04["progenitor_indices"]) == 2
        assert len(sc04["echo_indices"]) == 10
        assert len(sc04["transition_matrix"]) == 12

    def test_scenario_05_payload_skewers_and_density(self):
        """Adversarial check: Scenario 05 skewer flux must not be all 1.0 dummy arrays."""
        scenarios = _get_embedded_scenarios()
        sc05 = scenarios.get("05_cosmic_web_void_rsd_tomography")
        assert sc05 is not None
        assert sc05["shape"] == [32, 24, 24]
        expected_voxels = 32 * 24 * 24
        assert len(sc05["density"]) == expected_voxels, f"Expected {expected_voxels} voxels, got {len(sc05['density'])}"
        assert len(sc05["skewer_coords"]) == 8
        # Bug check: skewer_flux must NOT be all 1.0 dummy ones!
        skewer_flux_flat = [val for sk in sc05["skewer_flux"] for val in sk]
        all_ones = all(val == 1.0 for val in skewer_flux_flat)
        assert not all_ones, (
            "BUG: sc05['skewer_flux'] is filled with dummy 1.0 values! "
            "astrosnail/studio.py queried 'skewer_flux' instead of 'skewers' from scenarios.py"
        )

    def test_embedded_theorems_and_benchmarks(self):
        """Verify embedded theorems (Theorems 2.1 - 2.9) and benchmark scorecard."""
        content = STUDIO_HTML.read_text(encoding="utf-8")

        # Theorems
        thm_match = re.search(r'window\.ASTROSNAIL_THEOREMS\s*=\s*(\{.*?\});\s*window\.ASTROSNAIL_BENCHMARK', content, re.DOTALL)
        assert thm_match is not None, "window.ASTROSNAIL_THEOREMS block not found"
        theorems = json.loads(thm_match.group(1))

        for i in range(1, 10):
            tid = f"theorem_2_{i}"
            assert tid in theorems, f"Theorem {tid} missing in studio payload"
            t_obj = theorems[tid]
            assert t_obj.get("passed") is True, f"Theorem {tid} marked as failed: {t_obj}"
            assert t_obj.get("sorry_count", 0) == 0, f"Theorem {tid} has sorry count > 0"

        # Benchmarks
        bench_match = re.search(r'window\.ASTROSNAIL_BENCHMARK\s*=\s*(\{.*?\});\s*</script>', content, re.DOTALL)
        assert bench_match is not None, "window.ASTROSNAIL_BENCHMARK block not found"
        bench = json.loads(bench_match.group(1))
        assert bench.get("win_rate", 0) > 0.8
        assert bench.get("all_targets_met") is True


# =============================================================================
# 2. PROCESS TEARDOWN & LIFECYCLE CHALLENGE
# =============================================================================

def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestProcessLifecycleAndTeardown:
    """Stress tests server teardown, socket closing, and process termination."""

    def test_ephemeral_server_clean_shutdown_and_immediate_port_rebind(self):
        """Verify that stopping serve_studio cleanly releases socket, allowing instant re-bind."""
        port = _get_free_port()

        # Phase 1: Start and stop server on port
        t1 = threading.Thread(
            target=serve_studio,
            kwargs={
                "html_path": STUDIO_HTML,
                "port": port,
                "open_browser": False,
                "timeout": 1.5,
            },
            daemon=True,
        )
        t1.start()

        # Connect and confirm responsive
        url = f"http://127.0.0.1:{port}/"
        client_ok = False
        for _ in range(15):
            time.sleep(0.1)
            try:
                with urllib.request.urlopen(url, timeout=0.8) as resp:
                    if resp.status == 200:
                        client_ok = True
                        break
            except Exception:
                pass

        assert client_ok, f"Failed to connect to ephemeral server on {url}"
        t1.join(timeout=4.0)
        assert not t1.is_alive(), "First server thread failed to stop within timeout!"

        # Phase 2: Immediately re-bind to the exact same port
        t2 = threading.Thread(
            target=serve_studio,
            kwargs={
                "html_path": STUDIO_HTML,
                "port": port,
                "open_browser": False,
                "timeout": 1.5,
            },
            daemon=True,
        )
        t2.start()

        client_rebound = False
        for _ in range(15):
            time.sleep(0.1)
            try:
                with urllib.request.urlopen(url, timeout=0.8) as resp:
                    if resp.status == 200:
                        client_rebound = True
                        break
            except Exception:
                pass

        assert client_rebound, f"Failed to re-bind to port {port} immediately after teardown!"
        t2.join(timeout=4.0)
        assert not t2.is_alive(), "Second server thread failed to stop within timeout!"

        # Phase 3: Final socket check
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            res = s.connect_ex(("127.0.0.1", port))
            assert res != 0, f"Port {port} still accepted connection after shutdown!"

    def test_http_server_path_traversal_and_invalid_paths_rejected(self):
        """Verify that requests outside permitted paths return 404 and cannot leak files."""
        port = _get_free_port()
        t = threading.Thread(
            target=serve_studio,
            kwargs={
                "html_path": STUDIO_HTML,
                "port": port,
                "open_browser": False,
                "timeout": 2.5,
            },
            daemon=True,
        )
        t.start()

        # Wait for server ready
        time.sleep(0.5)

        invalid_paths = [
            "/etc/passwd",
            "/astrosnail/studio.py",
            "/../../setup.py",
            "/api/data",
            "/results/nonexistent.html",
        ]

        for p in invalid_paths:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            conn.request("GET", p)
            resp = conn.getresponse()
            assert resp.status == 404, f"Path {p} returned HTTP {resp.status} instead of 404!"
            conn.close()

        t.join(timeout=4.0)

    def test_cli_subprocess_lifecycle_and_timeout(self, tmp_path):
        """Verify that launching CLI in a subprocess finishes cleanly and enforces strict timeout."""
        target_html = tmp_path / "subproc_studio.html"
        cmd = [
            sys.executable,
            "-m", "astrosnail.studio",
            "--output", str(target_html),
            "--downsample", "10",
            "--verify",
        ]
        # Strict timeout: must finish within 15 seconds
        start_time = time.monotonic()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15.0)
        elapsed = time.monotonic() - start_time

        assert proc.returncode == 0, f"Subprocess failed:\n{proc.stderr}\n{proc.stdout}"
        assert target_html.exists()
        assert target_html.stat().st_size > 50_000
        print(f"\n[Lifecycle] Subprocess cleanly exited in {elapsed:.2f}s with code 0")


# =============================================================================
# 3. STRESS TESTING STUDIO EXPORT & CLI FLAGS
# =============================================================================

class TestStudioStressAndEdgeCases:
    """Stress tests studio generation with extreme parameters and concurrency."""

    def test_export_downsample_1_full_resolution(self, tmp_path):
        """Stress test: downsample=1 (full resolution, no decimation)."""
        target = tmp_path / "studio_d1_full.html"
        out = export_portable_studio(output_path=target, downsample=1)
        assert out.exists()
        size_bytes = out.stat().st_size
        assert size_bytes > 50_000
        print(f"\n[Stress] Full-resolution (downsample=1) size: {size_bytes:,} bytes")

    def test_export_downsample_extreme_100(self, tmp_path):
        """Stress test: downsample=100 (aggressive decimation). Must still exceed 50 KB due to 3D grid & stars."""
        target = tmp_path / "studio_d100.html"
        out = export_portable_studio(output_path=target, downsample=100)
        assert out.exists()
        size_bytes = out.stat().st_size
        assert size_bytes > 50_000, f"Downsample=100 dropped below 50 KB: {size_bytes} bytes"
        print(f"\n[Stress] Heavy decimation (downsample=100) size: {size_bytes:,} bytes")

    def test_export_downsample_zero_and_negative(self, tmp_path):
        """Boundary test: downsample=0 and negative downsampling must clamp gracefully to 1."""
        target_0 = tmp_path / "studio_d0.html"
        target_neg = tmp_path / "studio_dneg.html"
        out_0 = export_portable_studio(output_path=target_0, downsample=0)
        out_neg = export_portable_studio(output_path=target_neg, downsample=-5)
        assert out_0.exists() and out_0.stat().st_size > 50_000
        assert out_neg.exists() and out_neg.stat().st_size > 50_000

    def test_concurrent_multi_thread_exports(self, tmp_path):
        """Concurrency test: multiple threads simultaneously exporting studio HTML files."""
        n_threads = 4

        def worker(idx: int) -> int:
            p = tmp_path / f"concurrent_{idx}.html"
            res = export_portable_studio(output_path=p, downsample=10)
            return res.stat().st_size

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(n_threads)]
            sizes = [f.result(timeout=20.0) for f in futures]

        assert len(sizes) == n_threads
        for s in sizes:
            assert s > 50_000
        print(f"\n[Concurrency] {n_threads} concurrent exports succeeded. Sizes: {sizes}")
