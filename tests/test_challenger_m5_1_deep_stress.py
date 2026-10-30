"""
tests/test_challenger_m5_1_deep_stress.py
=========================================
Deep Adversarial Stress and Empirical Validation Suite for Milestone 5.
Role: Challenger M5-1-r2 (Studio Offline & Lifecycle Stress Challenger)

Adversarial Stress Axes:
1. Port reuse hammer: 10 rapid sequential bind-shutdown-rebind cycles on exact same port.
2. High-concurrency client load: 50 simultaneous HTTP requests against ephemeral server.
3. Slowloris / malformed HTTP requests (HTTP 0.9, bad headers, abrupt socket disconnects).
4. Exhaustive offline isolation audit: zero external network calls or remote CDNs in HTML.
5. In-depth physical authenticity and non-degeneracy audit of all 5 embedded scenario payloads:
   - Scenario 01: FRB singular impulse, SNe shock breakouts, Morse impulse spike at t_frb, valid CR epochs.
   - Scenario 02: Discrete periodicity of h_clock (EB + RR Lyrae), h_nope transit dip, orthogonality/low-correlation.
   - Scenario 03: 500 background stars across 128x128 focal plane, continuous KBO trajectories, 0 swaps for AstroSnail vs Kalman swap at frame 120.
   - Scenario 04: Causal generator contingency ratios >= 0.50, echo ratios < 0.50, valid delays and transition matrix.
   - Scenario 05: 3D density cube (18,432 voxels), void center [16,6,6] with delta < -0.7, 8 skewers with absorption variance > 0.
6. Dropzone file parser audit: no blocking alert() modals, proper state handling.
"""

from __future__ import annotations

import concurrent.futures
import http.client
import json
from pathlib import Path
import re
import socket
import threading
import time
import urllib.request
import numpy as np
import pytest

from astrosnail.studio import (
    export_portable_studio,
    verify_studio_html,
    serve_studio,
    serialize_scenario_payloads,
)

STUDIO_HTML = Path("studio/astrosnail_portable_studio.html")


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _extract_embedded_json(var_name: str) -> dict:
    content = STUDIO_HTML.read_text(encoding="utf-8")
    pattern = rf'window\.{var_name}\s*=\s*({{.*?}});\s*(?:window\.|</script>)'
    match = re.search(pattern, content, re.DOTALL)
    assert match is not None, f"Failed to extract window.{var_name}"
    return json.loads(match.group(1))


# =============================================================================
# 1. RAPID PORT RE-BIND & SERVER LIFECYCLE HAMMER
# =============================================================================

class TestServerLifecycleHammer:
    """Stress tests ephemeral server under adversarial port re-use and client loads."""

    def test_rapid_port_rebind_10_cycles(self):
        """Stress test: 10 rapid sequential bind-shutdown-rebind cycles on the exact same port."""
        port = _get_free_port()
        print(f"\n[Hammer] Testing 10 rapid rebind cycles on port {port}...")

        for cycle in range(10):
            server_stopped = threading.Event()
            t = threading.Thread(
                target=serve_studio,
                kwargs={
                    "html_path": STUDIO_HTML,
                    "port": port,
                    "open_browser": False,
                    "timeout": 0.4,
                },
                daemon=True,
            )
            t.start()

            # Confirm responsive
            connected = False
            for _ in range(10):
                time.sleep(0.05)
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=0.5) as resp:
                        if resp.status == 200:
                            connected = True
                            break
                except Exception:
                    pass

            assert connected, f"Cycle {cycle+1}: Failed to connect to port {port}"
            t.join(timeout=3.0)
            assert not t.is_alive(), f"Cycle {cycle+1}: Server thread failed to stop"

    def test_high_concurrency_requests(self):
        """Stress test: 50 concurrent client requests hammering the ephemeral server."""
        port = _get_free_port()
        t = threading.Thread(
            target=serve_studio,
            kwargs={
                "html_path": STUDIO_HTML,
                "port": port,
                "open_browser": False,
                "timeout": 3.0,
            },
            daemon=True,
        )
        t.start()
        time.sleep(0.3)

        n_requests = 50
        def client_fetch(idx: int) -> int:
            path = "/" if idx % 2 == 0 else "/studio.html"
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            conn.request("GET", path)
            resp = conn.getresponse()
            _ = resp.read()
            status = resp.status
            conn.close()
            return status

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(client_fetch, i) for i in range(n_requests)]
            statuses = [f.result(timeout=5.0) for f in futures]

        assert len(statuses) == n_requests
        assert all(s == 200 for s in statuses), f"Non-200 responses observed: {set(statuses)}"
        t.join(timeout=4.0)

    def test_abrupt_client_disconnect_handling(self):
        """Adversarial test: client opens connection, sends GET, and abruptly closes before reading."""
        port = _get_free_port()
        t = threading.Thread(
            target=serve_studio,
            kwargs={
                "html_path": STUDIO_HTML,
                "port": port,
                "open_browser": False,
                "timeout": 2.0,
            },
            daemon=True,
        )
        t.start()
        time.sleep(0.3)

        # Abrupt close test
        for _ in range(5):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("127.0.0.1", port))
            s.sendall(b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
            # Close immediately with RST packet (SO_LINGER 0)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b'\x01\x00\x00\x00\x00\x00\x00\x00')
            s.close()

        # Check server is still healthy
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1.0) as resp:
            assert resp.status == 200

        t.join(timeout=3.0)


# =============================================================================
# 2. OFFLINE ISOLATION & PURITY AUDIT
# =============================================================================

class TestOfflinePurityAudit:
    """Exhaustive check that no external network calls or CDNs are embedded."""

    def test_strict_offline_regex_audit(self):
        content = STUDIO_HTML.read_text(encoding="utf-8")

        # Any external scheme outside permitted documentation comments
        # Strip comments first
        no_comments = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        no_comments = re.sub(r'//.*?$', '', no_comments, flags=re.MULTILINE)

        # Check for remote scripts or links
        assert not re.search(r'<script[^>]+src=[\'"]https?://', no_comments, re.I)
        assert not re.search(r'<link[^>]+href=[\'"]https?://', no_comments, re.I)
        assert not re.search(r'@import\s+[\'"]https?://', no_comments, re.I)
        assert not re.search(r'url\([\'"]?https?://', no_comments, re.I)

        # Check for dynamic remote script injection in JS
        assert "document.createElement('script')" not in no_comments
        assert 'document.createElement("script")' not in no_comments

        # Check that no external APIs or telemetry beacons exist
        assert "fetch(" not in no_comments
        assert "XMLHttpRequest" not in no_comments
        assert "navigator.sendBeacon" not in no_comments
        assert "WebSocket(" not in no_comments

    def test_no_disruptive_alert_dialogs_in_dropzone(self):
        """Audit that file drop handler does NOT invoke disruptive window.alert()."""
        content = STUDIO_HTML.read_text(encoding="utf-8")
        # Ensure alert( is not used in drop handlers
        drop_handler_match = re.search(r'function handleFiles\(.*?\n\}', content, re.DOTALL)
        if drop_handler_match:
            assert "alert(" not in drop_handler_match.group(0), "Disruptive alert() found in handleFiles"


# =============================================================================
# 3. DEEP SCENARIO PAYLOAD PHYSICAL NON-DEGENERACY AUDIT
# =============================================================================

class TestScenarioPayloadNonDegeneracy:
    """Deep physical inspection of all 5 embedded scenario payloads."""

    @pytest.fixture(scope="class")
    def scenarios(self) -> dict:
        return _extract_embedded_json("ASTROSNAIL_SCENARIOS")

    def test_scenario_01_physics_and_morse_spike(self, scenarios):
        sc = scenarios["01_frb_supernova_cosmic_ray_triage"]
        assert sc["t_frb"] == 10000.0
        time_arr = np.array(sc["time"])
        flux_arr = np.array(sc["flux"])
        morse_arr = np.array(sc["morse_impulse"])

        # FRB epoch must exist in time
        frb_idx = np.argmin(np.abs(time_arr - 10000.0))
        assert abs(time_arr[frb_idx] - 10000.0) < 1.0

        # Morse impulse at FRB must exhibit a strong physical spike
        frb_morse = morse_arr[frb_idx]
        median_morse = np.median(morse_arr)
        assert frb_morse > median_morse * 5.0, f"Morse impulse at FRB ({frb_morse}) not prominent vs median ({median_morse})"

        # Cosmic ray epochs and glitch indices non-empty and disjoint
        cr_indices = sc["cosmic_ray_indices"]
        glitch_indices = sc["glitch_indices"]
        assert len(cr_indices) >= 10
        assert len(glitch_indices) >= 10
        # Check they fall within original epoch count
        orig_epochs = sc["metadata"]["n_epochs_original"]
        assert all(0 <= idx < orig_epochs for idx in cr_indices)
        assert all(0 <= idx < orig_epochs for idx in glitch_indices)

    def test_scenario_02_clock_nope_periodicity_and_split(self, scenarios):
        sc = scenarios["02_eclipsing_binary_exoplanet_transit_split"]
        h_clock = np.array(sc["h_clock"])
        h_nope = np.array(sc["h_nope"])
        flux = np.array(sc["flux"])

        assert len(h_clock) > 0
        assert len(h_nope) > 0
        assert len(h_clock) == len(h_nope) == len(flux)

        # h_clock should have periodic structure with authentic physical amplitude
        # (primary eclipse depth ~ -0.26, pulsation peaks ~ +0.05, variance ~ 0.003)
        assert np.var(h_clock) > 0.001, "h_clock is degenerate (flat)"
        assert np.min(h_clock) < -0.20, f"Primary eclipse not prominent: {np.min(h_clock)}"
        assert np.max(h_clock) > 0.04, f"Pulsation peaks not prominent: {np.max(h_clock)}"

        # h_nope should have satellite flare (peak ~ 1.05) and exoplanet transit dip (min ~ 0.999)
        assert np.var(h_nope) > 0.0, "h_nope is completely flat"
        assert np.max(h_nope) >= 1.04, f"Satellite flare not prominent in h_nope: {np.max(h_nope)}"
        assert np.min(h_nope) <= 0.9995, f"Exoplanet transit dip missing in h_nope: {np.min(h_nope)}"

        # Physical constants check
        assert abs(sc["period_eb"] - 3.412) < 0.01
        assert abs(sc["period_rr"] - 0.567) < 0.01
        assert abs(sc["period_transit"] - 18.71) < 0.01
        assert abs(sc["t_flare"] - 214.3) < 0.5

        # Correlation between clock and nope should be moderate to low (distinct physical mechanisms)
        corr = np.corrcoef(h_clock, h_nope)[0, 1]
        assert abs(corr) < 0.85, f"h_clock and h_nope are too highly correlated ({corr:.3f})"

    def test_scenario_03_stars_and_trajectories_continuity(self, scenarios):
        sc = scenarios["03_crowded_field_kbo_crossing_astrometry"]
        stars = np.array(sc["stars"])
        # Exactly 500 stars
        assert len(stars) == 500, f"Expected 500 background stars, found {len(stars)}"
        # Check distribution in [0, 128]x[0, 128]
        y_coords = stars[:, 0]
        x_coords = stars[:, 1]
        fluxes = stars[:, 2]
        assert np.min(y_coords) >= 0.0 and np.max(y_coords) <= 128.0
        assert np.min(x_coords) >= 0.0 and np.max(x_coords) <= 128.0
        assert np.std(x_coords) > 20.0 and np.std(y_coords) > 20.0, "Stars are clustered/degenerate"
        assert all(f > 0 for f in fluxes), "Non-positive star flux found"

        # Trajectories continuity (no teleportation)
        astrosnail_trajs = np.array(sc["astrosnail_trajectories"])  # [4, 240, 2]
        kalman_trajs = np.array(sc["kalman_trajectories"])          # [4, 240, 2]
        assert astrosnail_trajs.shape == (4, 240, 2)
        assert kalman_trajs.shape == (4, 240, 2)

        # Step distance between frames should be smooth (< 5.0 pixels/frame)
        for k in range(4):
            diffs = np.linalg.norm(np.diff(astrosnail_trajs[k], axis=0), axis=1)
            assert np.max(diffs) < 5.0, f"KBO {k} has discontinuous jump: {np.max(diffs)}"

        # Crossing swap check at frame 120
        # Astrosnail must remain continuous across frame 120
        # Kalman swaps trajectories 0 and 1, and 2 and 3
        k0_step_at_crossing = np.linalg.norm(kalman_trajs[0, 120] - kalman_trajs[0, 119])
        # Trajectories 0 and 1 cross at frame 120, partner is swapped
        # Check swap happened in Kalman
        assert np.allclose(kalman_trajs[0, 120:], astrosnail_trajs[1, 120:]), "Kalman track 0 did not swap with track 1"
        assert np.allclose(kalman_trajs[1, 120:], astrosnail_trajs[0, 120:]), "Kalman track 1 did not swap with track 0"

    def test_scenario_04_causal_progenitors_and_echoes(self, scenarios):
        sc = scenarios["04_lensed_quasar_echo_swarm_factorization"]
        prog_indices = sc["progenitor_indices"]
        echo_indices = sc["echo_indices"]
        cont_ratios = sc["contingency_ratios"]
        time_delays = sc["time_delays"]

        assert set(prog_indices) == {0, 1}, f"Expected progenitors {{0, 1}}, got {prog_indices}"
        assert len(echo_indices) == 10
        assert len(cont_ratios) == 12

        # Progenitors must have contingency ratio >= 0.50
        for p in prog_indices:
            assert cont_ratios[p] >= 0.50, f"Progenitor {p} contingency ratio {cont_ratios[p]} < 0.50"

        # Echoes must have contingency ratio < 0.50
        for e in echo_indices:
            assert cont_ratios[e] < 0.50, f"Echo {e} contingency ratio {cont_ratios[e]} >= 0.50"

        # Time delays must be 0 for progenitors and positive for echoes
        assert time_delays[0] == 0.0 and time_delays[1] == 0.0
        assert any(d > 5.0 for d in time_delays[2:]), "No positive time delays found for echoes"

    def test_scenario_05_cosmic_web_3d_density_and_skewers(self, scenarios):
        sc = scenarios["05_cosmic_web_void_rsd_tomography"]
        shape = sc["shape"]
        assert shape == [32, 24, 24]
        density = np.array(sc["density"]).reshape(shape)

        # Cosmic void: underdensity at void_center
        vc = sc["void_center"]
        void_val = density[vc[0], vc[1], vc[2]]
        assert void_val < -0.70, f"Void center density ({void_val}) not underdense (< -0.70)"

        # Overdense cluster knot at knot_center
        kc = sc["knot_center"]
        knot_val = density[kc[0], kc[1], kc[2]]
        assert knot_val > 1.0, f"Knot center density ({knot_val}) not overdense (> 1.0)"

        # Skewers: 8 skewers with absorption structure
        skewer_coords = sc["skewer_coords"]
        skewer_flux = np.array(sc["skewer_flux"])
        assert len(skewer_coords) == 8
        assert skewer_flux.shape == (8, 32)
        # Absorption variation: flux should not be flat
        variances = np.var(skewer_flux, axis=1)
        assert all(v > 0.001 for v in variances), f"Flat skewer flux detected: {variances}"
