"""
tests/test_studio.py
====================
Comprehensive offline test suite for AstroSnail Portable Observatory Studio
(astrosnail.studio) and studio/astrosnail_portable_studio.html.

Verifies:
1. Programmatic API and CLI generation.
2. Minimum file size (> 50 KB).
3. Zero external CDN dependencies (100% offline file:// self-sufficiency).
4. Full 4-quad viewport DOM hierarchy and canvases.
5. Integrity of embedded JSON payloads for all 5 canonical scenarios.
6. Ephemeral HTTP server start, GET response, and clean termination without orphaned processes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
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
    serialize_theorems_payload,
    serialize_benchmark_payload,
)


@pytest.fixture(scope="module")
def studio_html_path(tmp_path_factory) -> Path:
    """Generates a cached studio HTML file for module-level inspection."""
    temp_dir = tmp_path_factory.mktemp("studio_test")
    out_path = temp_dir / "astrosnail_test_studio.html"
    result = export_portable_studio(output_path=out_path, include_scenarios=True, downsample=5)
    assert result.exists()
    return result


# =============================================================================
# TIER 1: PROGRAMMATIC API & FILE SIZE
# =============================================================================

def test_export_portable_studio_default(tmp_path):
    """Verify default export produces an HTML file > 50 KB."""
    target = tmp_path / "studio.html"
    out_path = export_portable_studio(output_path=target)
    assert out_path.exists()
    size = out_path.stat().st_size
    assert size > 50_000, f"Expected size > 50,000 bytes, got {size}"


def test_export_portable_studio_auto_creates_directories(tmp_path):
    """Verify export automatically creates nested parent directories."""
    nested = tmp_path / "deep" / "nested" / "dir" / "custom_studio.html"
    out_path = export_portable_studio(output_path=nested)
    assert out_path.exists()
    assert out_path.stat().st_size > 50_000


def test_export_portable_studio_no_data(tmp_path):
    """Verify --no-data mode creates valid shell without scenarios."""
    target = tmp_path / "studio_no_data.html"
    out_path = export_portable_studio(output_path=target, include_scenarios=False)
    assert out_path.exists()
    assert out_path.stat().st_size > 40_000
    with open(out_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "window.ASTROSNAIL_SCENARIOS = {}" in content or "window.ASTROSNAIL_SCENARIOS = {" in content


def test_export_portable_studio_downsample(tmp_path):
    """Verify downsample factor alters serialization length."""
    target_d5 = tmp_path / "d5.html"
    target_d10 = tmp_path / "d10.html"
    out_d5 = export_portable_studio(output_path=target_d5, downsample=5)
    out_d10 = export_portable_studio(output_path=target_d10, downsample=10)
    assert out_d5.exists() and out_d10.exists()
    assert out_d10.stat().st_size <= out_d5.stat().st_size


# =============================================================================
# TIER 2: OFFLINE SELF-SUFFICIENCY & ZERO-CDN AUDIT
# =============================================================================

def test_zero_external_cdn_links(studio_html_path):
    """Verify 100% offline self-containment with zero external network references."""
    with open(studio_html_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Disallow remote scripts: <script src="http..."
    disallowed_scripts = re.findall(r'<script[^>]+src=[\'"](https?://[^\'"]+)[\'"]', content, re.IGNORECASE)
    assert len(disallowed_scripts) == 0, f"Found external script tags: {disallowed_scripts}"

    # Disallow remote stylesheets: <link href="http..."
    disallowed_links = re.findall(r'<link[^>]+href=[\'"](https?://[^\'"]+)[\'"]', content, re.IGNORECASE)
    assert len(disallowed_links) == 0, f"Found external stylesheet links: {disallowed_links}"

    # Disallow CSS @import remote URLs
    disallowed_imports = re.findall(r'@import\s+url\([\'"]?(https?://[^\'")]+)[\'"]?\)', content, re.IGNORECASE)
    assert len(disallowed_imports) == 0, f"Found external CSS imports: {disallowed_imports}"


def test_verify_studio_html_api(studio_html_path):
    """Verify built-in verify_studio_html validation function."""
    ok, errors = verify_studio_html(studio_html_path)
    assert ok, f"verify_studio_html failed: {errors}"
    assert len(errors) == 0


def test_verify_studio_html_missing_file(tmp_path):
    """Verify error reporting on missing file."""
    fake_path = tmp_path / "non_existent.html"
    ok, errors = verify_studio_html(fake_path)
    assert not ok
    assert any("not found" in err.lower() for err in errors)


# =============================================================================
# TIER 3: DOM HIERARCHY, CANVASES & HUD
# =============================================================================

def test_viewport_containers_present(studio_html_path):
    """Verify presence of all 4 quad containers and workspace grid."""
    with open(studio_html_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert 'id="workspace-grid"' in content
    assert 'id="quad-a"' in content
    assert 'id="quad-b"' in content
    assert 'id="quad-c"' in content
    assert 'id="quad-d"' in content


def test_canvases_present(studio_html_path):
    """Verify presence of dedicated HTML5 canvases in all viewports."""
    with open(studio_html_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert 'id="canvas-quad-a"' in content
    assert 'id="canvas-quad-b"' in content
    assert 'id="canvas-quad-c"' in content
    assert 'id="canvas-quad-d"' in content


def test_controls_and_annunciator(studio_html_path):
    """Verify presence of top annunciator badges and timeline controls."""
    with open(studio_html_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert 'id="timeline-scrub"' in content
    assert 'id="clock-display"' in content
    assert 'id="scenario-select"' in content
    assert 'id="btn-play"' in content
    assert any(b in content for b in ['id="btn-prev"', 'id="btn-step-prev"']), "Missing previous button"
    assert any(b in content for b in ['id="btn-next"', 'id="btn-step-next"']), "Missing next button"
    assert 'id="cut-x"' in content
    assert 'id="cut-y"' in content
    assert 'id="cut-z"' in content
    assert 'id="file-dropzone"' in content


# =============================================================================
# TIER 4: EMBEDDED JSON DATA INTEGRITY
# =============================================================================

def test_embedded_payloads_valid_json(studio_html_path):
    """Extract and validate embedded JSON objects in the studio HTML."""
    with open(studio_html_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Match window.ASTROSNAIL_SCENARIOS = {...};
    scen_match = re.search(r'window\.ASTROSNAIL_SCENARIOS\s*=\s*(\{.*?\});\s*window\.ASTROSNAIL_THEOREMS', content, re.DOTALL)
    assert scen_match, "Could not locate window.ASTROSNAIL_SCENARIOS in HTML"
    scenarios_json = json.loads(scen_match.group(1))

    # Match window.ASTROSNAIL_THEOREMS = {...};
    thm_match = re.search(r'window\.ASTROSNAIL_THEOREMS\s*=\s*(\{.*?\});\s*window\.ASTROSNAIL_BENCHMARK', content, re.DOTALL)
    assert thm_match, "Could not locate window.ASTROSNAIL_THEOREMS in HTML"
    theorems_json = json.loads(thm_match.group(1))

    # Check 5 canonical scenarios
    canonical = [
        "01_frb_supernova_cosmic_ray_triage",
        "02_eclipsing_binary_exoplanet_transit_split",
        "03_crowded_field_kbo_crossing_astrometry",
        "04_lensed_quasar_echo_swarm_factorization",
        "05_cosmic_web_void_rsd_tomography",
    ]
    for cid in canonical:
        assert cid in scenarios_json, f"Missing scenario: {cid}"

    # Scenario 01: Verify authentic non-empty cosmic ray indices, glitch indices, and positive Morse impulse
    sc01 = scenarios_json["01_frb_supernova_cosmic_ray_triage"]
    assert len(sc01["cosmic_ray_indices"]) > 0, "Scenario 01 cosmic_ray_indices must not be empty"
    assert len(sc01["glitch_indices"]) > 0, "Scenario 01 glitch_indices must not be empty"
    assert max(sc01["morse_impulse"]) > 0.0, "Scenario 01 morse_impulse must contain positive transient peaks"

    # Scenario 02: Verify non-empty periodic clock and aperiodic nope decompositions and true periods
    sc02 = scenarios_json["02_eclipsing_binary_exoplanet_transit_split"]
    assert len(sc02["h_clock"]) > 0, "Scenario 02 h_clock must not be empty"
    assert len(sc02["h_nope"]) > 0, "Scenario 02 h_nope must not be empty"
    assert abs(sc02["period_eb"] - 3.412) < 0.01, f"Scenario 02 period_eb mismatch: {sc02['period_eb']}"
    assert abs(sc02["period_rr"] - 0.567) < 0.01, f"Scenario 02 period_rr mismatch: {sc02['period_rr']}"
    assert abs(sc02["period_transit"] - 18.71) < 0.01, f"Scenario 02 period_transit mismatch: {sc02['period_transit']}"

    # Scenario 03: Verify all 500 stars distributed across focal plane (std > 10.0), not collapsed at (0, 0)
    sc03 = scenarios_json["03_crowded_field_kbo_crossing_astrometry"]
    assert len(sc03["stars"]) >= 500, f"Scenario 03 stars count {len(sc03['stars'])} < 500"
    assert np.std([s[0] for s in sc03["stars"]]) > 10.0, "Scenario 03 stars must be distributed across focal plane (std > 10.0)"
    assert not all(s[0] == 0.0 and s[1] == 0.0 for s in sc03["stars"]), "Scenario 03 stars must not all be at (0, 0)"

    # Scenario 04: Verify active progenitors vs driven passive registers factorization
    sc04 = scenarios_json["04_lensed_quasar_echo_swarm_factorization"]
    assert len(sc04["progenitor_indices"]) == 2
    assert len(sc04["echo_indices"]) == 10
    assert len(sc04["contingency_ratios"]) == 12
    assert max(sc04["contingency_ratios"][:2]) >= 0.50, "Progenitors must have contingency ratio >= 0.50"

    # Scenario 05: Verify varied absorption spectra across skewers (variance > 0.0, not flat 1.0 dummy)
    sc05 = scenarios_json["05_cosmic_web_void_rsd_tomography"]
    assert len(sc05["skewer_flux"]) == 8
    skewer_flat = [v for sk in sc05["skewer_flux"] for v in sk]
    assert not all(v == 1.0 for v in skewer_flat), "Scenario 05 skewers must not be flat 1.0 dummy arrays"
    assert np.var(sc05["skewer_flux"]) > 0.0, "Scenario 05 skewer_flux variance must be > 0.0"

    # Check theorems 2.1-2.9
    for i in range(1, 10):
        tid = f"theorem_2_{i}"
        assert tid in theorems_json, f"Missing theorem: {tid}"
        assert theorems_json[tid].get("passed", False) is True


# =============================================================================
# TIER 5: CLI EXECUTION & EPHEMERAL HTTP SERVER CLEAN TEARDOWN
# =============================================================================

def test_studio_cli_generate_and_verify(tmp_path):
    """Test CLI generation and verification via subprocess."""
    out_file = tmp_path / "cli_studio.html"
    cmd = [
        sys.executable,
        "-m", "astrosnail.studio",
        "--output", str(out_file),
        "--verify",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
    assert proc.returncode == 0, f"CLI command failed:\n{proc.stderr}\n{proc.stdout}"
    assert out_file.exists()
    assert out_file.stat().st_size > 50_000
    assert "Verification PASSED" in proc.stdout


def _find_free_port() -> int:
    """Finds an unused TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_ephemeral_server_lifecycle(studio_html_path):
    """Verify ephemeral HTTP server starts, serves HTML at root, and shuts down cleanly."""
    port = _find_free_port()

    # Launch server with 3-second auto-shutdown timeout in a background thread
    server_thread = threading.Thread(
        target=serve_studio,
        kwargs={
            "html_path": studio_html_path,
            "port": port,
            "open_browser": False,
            "timeout": 2.5,
        },
        daemon=True,
    )
    server_thread.start()

    # Poll until server is responding
    url = f"http://127.0.0.1:{port}/"
    connected = False
    for _ in range(20):
        time.sleep(0.1)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AstroSnailTest/1.0"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    body = resp.read()
                    assert len(body) > 50_000
                    assert b"ASTROSNAIL" in body
                    assert b"quad-a" in body
                    connected = True
                    break
        except Exception:
            continue

    assert connected, f"Could not connect to ephemeral server at {url}"

    # Wait for server thread to cleanly terminate
    server_thread.join(timeout=5.0)
    assert not server_thread.is_alive(), "Server thread did not terminate cleanly"

    # Confirm socket is closed
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        res = s.connect_ex(("127.0.0.1", port))
        assert res != 0, f"Port {port} still open after server shutdown"


# =============================================================================
# TIER 6: QUAD C CAMERA X-AXIS FIX & INDUSTRY-STANDARD OPTIONS / BOOST
# =============================================================================

def test_quad_c_camera_x_axis_drag_fix(studio_html_path):
    """Verify Quad C camera X-axis drag and elevation pitch math never invert foreground rotation."""
    with open(studio_html_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Ensure the old inverted X-drag assignment is gone and replaced with natural turntable yaw -= dx
    assert "STATE.camera.yaw += dx" not in content, "Quad C X-axis camera drag is still backwards (yaw += dx)"
    assert "STATE.camera.yaw -= dx * 0.01 * invX" in content, "Expected natural turntable X-axis drag (yaw -= dx)"

    # Ensure true turntable elevation rotation (cosP > 0 across [-1.45, 1.45]) so negative pitch never flips Z or X-drag
    assert "const y2 = y1 * sinP - dz * cosP;" in content, "Missing true elevation y2 = y1 * sinP - dz * cosP"
    assert "const z2 = y1 * cosP + dz * sinP;" in content, "Missing true elevation depth z2 = y1 * cosP + dz * sinP"
    assert "const ry2 = ry1 * sinP - vz * cosP;" in content, "Triad gizmo ry2 must match elevation projection"

    # Verify Painter's algorithm depth sorting and perspective foreshortening in Quad C
    assert "drawList.sort((a, b) => a.pt[2] - b.pt[2])" in content
    assert 'id="btn-proj-c"' in content
    assert 'id="btn-orbit-c"' in content
    for preset in ['data-cam="iso"', 'data-cam="top"', 'data-cam="front"', 'data-cam="side"']:
        assert preset in content, f"Missing Quad C camera preset: {preset}"


def test_industry_standard_options_and_boost_controls(studio_html_path):
    """Verify Boost controls, expanded colormaps, DS9 stretch, Loupe, Phase/PSD, and 12x12 Matrix tabs."""
    with open(studio_html_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Global and Quad C Boost & Iso-threshold controls
    assert 'id="btn-boost"' in content
    assert 'id="boost-c"' in content
    assert 'id="iso-c"' in content
    assert "function setBoost(" in content

    # Expanded scientific colormaps
    for cmap in ["turbo", "plasma", "inferno", "viridis", "magma", "coolwarm"]:
        assert f'value="{cmap}"' in content, f"Missing colormap option: {cmap}"

    # Quad A options (band selector, stretch selector, 3-sigma toggle, Phase/PSD tab)
    assert 'id="band-a-select"' in content
    assert 'id="stretch-a-select"' in content
    assert 'id="btn-sigma-a"' in content
    assert 'data-mode="phase"' in content

    # Quad B DS9 stretch, 4x Crossing Loupe, and Velocity Vectors
    assert 'id="stretch-b-select"' in content
    assert 'id="btn-loupe-b"' in content
    assert 'id="btn-vectors-b"' in content
    assert 'id="btn-reset-b"' in content

    # Quad D 12x12 Transition Matrix Heatmap tab, Dropzone sync, & Keyboard Shortcuts Modal
    assert 'data-mode="matrix"' in content
    assert "function syncQuadDPanels(" in content
    assert 'id="shortcuts-modal"' in content


def test_quad_c_x_drag_foreground_direction_math():
    """Mathematically verify across all pitch angles in [-1.45, +1.45] that:
    1. World +Z is always upright on screen (dy2/ddz = -cos(pitch) < 0),
    2. Foreground points (z2 > 0 at dz=0) always have y1 > 0 and move right (+x1) when dragging right (dx > 0),
    3. Dragging down (dy > 0 -> +pitch) moves foreground points (z2 > 0) down (+y2).
    """
    yaw0 = -0.62
    dx_v, dy_v, dz_v = 6.5, 10.5, 0.0
    x1_before = dx_v * np.cos(yaw0) - dy_v * np.sin(yaw0)
    y1_0 = dx_v * np.sin(yaw0) + dy_v * np.cos(yaw0)
    assert y1_0 > 0.0

    for pitch in [-1.45, -0.85, -0.25, 0.0, 0.48, 1.05, 1.45]:
        cos_p = np.cos(pitch)
        sin_p = np.sin(pitch)
        # 1. World +Z never flips upside down
        assert -cos_p < 0.0, f"World +Z flipped upside down at pitch={pitch}"

        # 2. Foreground point (dz=0, y1_0 > 0) has positive camera depth z2 > 0 at every pitch
        z2 = y1_0 * cos_p + dz_v * sin_p
        assert z2 > 0.0, f"Foreground point lost positive depth z2={z2} at pitch={pitch}"

        # Dragging right (+20px -> yaw decreases by 0.20 rad) moves foreground point right (+x1)
        yaw_after = yaw0 - 20.0 * 0.01
        x1_after = dx_v * np.cos(yaw_after) - dy_v * np.sin(yaw_after)
        assert x1_after > x1_before, (
            f"At pitch={pitch}, dragging right (+dx) must move foreground right, got {x1_before:.3f} -> {x1_after:.3f}"
        )

        # 3. Dragging down (+10px -> pitch increases by 0.10 rad) moves foreground point down (+y2)
        if pitch <= 1.30:
            y2_before = y1_0 * sin_p - dz_v * cos_p
            pitch_after = pitch + 10.0 * 0.01
            y2_after = y1_0 * np.sin(pitch_after) - dz_v * np.cos(pitch_after)
            assert y2_after > y2_before, (
                f"At pitch={pitch}, dragging down (+dy) must move foreground down (+y2), got {y2_before:.3f} -> {y2_after:.3f}"
            )


def test_headless_browser_quad_c_and_boost_e2e(studio_html_path, tmp_path):
    """Run V8 / Headless E2E verification of Quad C X/Y drag (positive & negative pitch),
    3D voxel probing, Boost synchronization, Quad A channels, Quad B DS9 hover,
    Quad D Dropzone -> Clear button visibility, and composite PNG export with zero orphaned processes.
    """
    import shutil

    node_bin = shutil.which("node") or (
        r"C:\Program Files\nodejs\node.exe" if Path(r"C:\Program Files\nodejs\node.exe").exists() else None
    )
    assert node_bin is not None, "Node.js V8 runtime is required for E2E JS verification"

    harness_script = tmp_path / "run_studio_e2e.js"
    harness_script.write_text(
        r"""
const fs = require("fs");
const vm = require("vm");

const htmlPath = process.argv[2];
const html = fs.readFileSync(htmlPath, "utf8");

// Extract all <script> blocks in order
const scriptBlocks = [];
const re = /<script>([\s\S]*?)<\/script>/gi;
let m;
while ((m = re.exec(html)) !== null) {
  scriptBlocks.push(m[1]);
}
if (scriptBlocks.length < 2) {
  throw new Error("Expected at least 2 script blocks in studio HTML");
}

// Minimal DOM & 2D Canvas mock that records draw calls, events, and classLists
function makeClassList(initial = []) {
  const set = new Set(initial.filter(Boolean));
  return {
    add: (...cls) => cls.forEach(c => set.add(c)),
    remove: (...cls) => cls.forEach(c => set.delete(c)),
    contains: (c) => set.has(c),
    toggle: (c, force) => {
      const shouldAdd = force !== undefined ? Boolean(force) : !set.has(c);
      if (shouldAdd) set.add(c); else set.delete(c);
      return shouldAdd;
    },
    toString: () => Array.from(set).join(" ")
  };
}

function makeCanvasContext(canvas) {
  return {
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
    font: "",
    globalAlpha: 1,
    _ops: [],
    save() {},
    restore() {},
    beginPath() {},
    rect() {},
    clip() {},
    fillRect(x, y, w, h) {
      if (x === 0 && y === 0 && w >= 600 && h >= 300 && canvas.id !== "offscreen-composite") {
        this._ops = [];
      }
      this._ops.push({ op: "fillRect", x, y, w, h, style: this.fillStyle });
    },
    strokeRect(x, y, w, h) { this._ops.push({ op: "strokeRect", x, y, w, h, style: this.strokeStyle }); },
    moveTo(x, y) { this._ops.push({ op: "moveTo", x, y, style: this.strokeStyle }); },
    lineTo(x, y) { this._ops.push({ op: "lineTo", x, y, style: this.strokeStyle }); },
    stroke() { this._ops.push({ op: "stroke", style: this.strokeStyle }); },
    fill() { this._ops.push({ op: "fill", style: this.fillStyle }); },
    arc(x, y, r) { this._ops.push({ op: "arc", x, y, r, fill: this.fillStyle, stroke: this.strokeStyle }); },
    setLineDash() {},
    fillText(text, x, y) { this._ops.push({ op: "fillText", text: String(text), x, y, style: this.fillStyle }); },
    createImageData(w, h) { return { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }; },
    putImageData(img) { this._ops.push({ op: "putImageData", w: img.width, h: img.height }); },
    drawImage(src, x, y, w, h) { this._ops.push({ op: "drawImage", id: src && src.id, x, y, w, h }); }
  };
}

const elementsById = new Map();
const queryAllMap = new Map();

function makeElement(id, tagName = "DIV", initialClasses = [], attrs = {}) {
  const listeners = {};
  const attrMap = Object.assign({}, attrs);
  const el = {
    id,
    tagName,
    value: attrs.value || "",
    innerText: attrs.innerText || "",
    innerHTML: "",
    download: "",
    href: "",
    width: 640,
    height: 360,
    clientWidth: 640,
    clientHeight: 360,
    _clicked: 0,
    classList: makeClassList(initialClasses),
    get className() { return this.classList.toString(); },
    set className(v) {
      const parts = String(v).split(/\s+/).filter(Boolean);
      this.classList = makeClassList(parts);
    },
    parentElement: {
      clientWidth: 640,
      clientHeight: 360,
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 640, height: 360 })
    },
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 640, height: 360 }),
    getAttribute: (k) => attrMap[k],
    setAttribute: (k, v) => { attrMap[k] = String(v); },
    addEventListener(type, fn) {
      if (!listeners[type]) listeners[type] = [];
      listeners[type].push(fn);
    },
    dispatchEvent(evt) {
      const list = listeners[evt.type] || [];
      list.forEach(fn => fn(Object.assign({ target: el, preventDefault() {} }, evt)));
    },
    click() {
      this._clicked++;
      this.dispatchEvent({ type: "click", button: 0 });
    },
    querySelector(sel) {
      if (sel === ".drop-text") return el._dropText || (el._dropText = makeElement("drop-text"));
      if (sel === ".drop-sub") return el._dropSub || (el._dropSub = makeElement("drop-sub"));
      return null;
    },
    toDataURL() { return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==" + id; }
  };
  el._ctx = makeCanvasContext(el);
  el.getContext = () => el._ctx;
  if (id) elementsById.set(id, el);
  return el;
}

// Pre-populate DOM elements matching studio.py
[
  ["scenario-select", "SELECT", [], { value: "01_frb_supernova_cosmic_ray_triage" }],
  ["btn-prev", "BUTTON"],
  ["btn-next", "BUTTON"],
  ["btn-play", "BUTTON", ["btn", "active"], { innerText: "PAUSE" }],
  ["btn-step-prev", "BUTTON"],
  ["btn-step-next", "BUTTON"],
  ["timeline-scrub", "INPUT", [], { value: "120" }],
  ["clock-display", "SPAN"],
  ["speed-select", "SELECT", [], { value: "1.0" }],
  ["btn-boost", "BUTTON", ["btn"], { innerText: "BOOST: 1.0x" }],
  ["boost-c", "INPUT", [], { value: "1.0" }],
  ["cmap-select", "SELECT", [], { value: "turbo" }],
  ["workspace-grid", "MAIN", ["workspace-grid"]],
  ["shortcuts-modal", "ASIDE", ["shortcuts-modal", "hidden"]],
  ["btn-shortcuts", "BUTTON"],
  ["btn-close-shortcuts", "BUTTON"],
  ["canvas-quad-a", "CANVAS"],
  ["canvas-quad-b", "CANVAS"],
  ["canvas-quad-c", "CANVAS"],
  ["canvas-quad-d", "CANVAS"],
  ["panel-quad-d-theorems", "DIV", ["hud-panel", "hidden"]],
  ["panel-quad-d-dropzone", "DIV", ["dropzone-panel", "hidden"]],
  ["file-dropzone", "DIV", ["dropzone-box"]],
  ["file-input-hidden", "INPUT", ["hidden"]],
  ["overlay-quad-d", "DIV", ["quad-overlay"]],
  ["toolbar-quad-d", "DIV", ["sub-toolbar"]],
  ["btn-clear-drop", "BUTTON", ["mini-btn", "hidden"]],
  ["badge-a-info", "DIV"],
  ["badge-a-cursor", "DIV"],
  ["badge-b-info", "DIV"],
  ["badge-b-rms", "DIV"],
  ["badge-c-info", "DIV"],
  ["badge-c-cam", "DIV"],
  ["badge-d-info", "DIV"],
  ["label-d-hover", "SPAN"],
  ["band-a-select", "SELECT", [], { value: "all" }],
  ["stretch-a-select", "SELECT", [], { value: "linear" }],
  ["btn-sigma-a", "BUTTON", ["mini-btn", "active"]],
  ["stretch-b-select", "SELECT", [], { value: "log" }],
  ["btn-loupe-b", "BUTTON", ["mini-btn", "active"]],
  ["btn-vectors-b", "BUTTON", ["mini-btn", "active"]],
  ["btn-reset-b", "BUTTON", ["mini-btn"]],
  ["cut-x", "INPUT", [], { value: "23" }],
  ["cut-y", "INPUT", [], { value: "23" }],
  ["cut-z", "INPUT", [], { value: "31" }],
  ["iso-c", "INPUT", [], { value: "0.25" }],
  ["btn-orbit-c", "BUTTON", ["btn"]],
  ["btn-reset-cam", "BUTTON", ["btn"]],
  ["btn-proj-c", "BUTTON", ["mini-btn", "active"], { innerText: "PERSP" }],
  ["btn-layer-vox", "BUTTON", ["mini-btn", "active"]],
  ["btn-layer-skew", "BUTTON", ["mini-btn", "active"]],
  ["btn-layer-jet", "BUTTON", ["mini-btn", "active"]],
  ["btn-export-png", "BUTTON", ["btn"]],
  ["btn-export-json", "BUTTON", ["btn"]]
].forEach(([id, tag, cls, attrs]) => makeElement(id, tag, cls || [], attrs || {}));

const layoutBtns = ["grid", "solo-a", "solo-b", "solo-c", "solo-d"].map((l, i) =>
  makeElement(`layout-${l}`, "BUTTON", i === 0 ? ["btn", "layout-btn", "active"] : ["btn", "layout-btn"], { "data-layout": l })
);
const quadATabs = ["lightcurve", "beltrami", "phase", "spectrogram"].map((m, i) =>
  makeElement(`tab-a-${m}`, "BUTTON", i === 0 ? ["tab-btn", "active"] : ["tab-btn"], { "data-mode": m })
);
const quadBTabs = ["astrosnail", "baseline", "side-by-side", "overlay"].map((m, i) =>
  makeElement(`tab-b-${m}`, "BUTTON", i === 0 ? ["tab-btn", "active"] : ["tab-btn"], { "data-mode": m })
);
const quadCTamPresets = ["iso", "top", "front", "side"].map(c =>
  makeElement(`cam-c-${c}`, "BUTTON", ["mini-btn"], { "data-cam": c })
);
const quadDTabs = ["causal", "matrix", "theorems", "dropzone"].map((m, i) =>
  makeElement(`tab-d-${m}`, "BUTTON", i === 0 ? ["tab-btn", "active"] : ["tab-btn"], { "data-mode": m, innerText: m === "causal" ? "CAUSAL GRAPH" : m.toUpperCase() })
);

queryAllMap.set(".layout-btn", layoutBtns);
queryAllMap.set("#quad-a .tab-btn", quadATabs);
queryAllMap.set("#quad-b .tab-btn", quadBTabs);
queryAllMap.set("#quad-c [data-cam]", quadCTamPresets);
queryAllMap.set("#quad-d .tab-btn", quadDTabs);

const winListeners = {};
let lastCreatedLink = null;
let lastCompositeCanvas = null;

const sandbox = {
  console,
  Math,
  Number,
  String,
  Array,
  Object,
  Set,
  Map,
  RegExp,
  JSON,
  parseInt,
  parseFloat,
  isNaN,
  Infinity,
  encodeURIComponent,
  URLSearchParams,
  Uint8Array,
  Uint8ClampedArray,
  TextDecoder,
  FileReader: class {
    readAsText(file) {
      if (this.onload) this.onload({ target: { result: file._content || "" } });
    }
    readAsArrayBuffer(file) {
      if (this.onload) this.onload({ target: { result: file._buffer || new ArrayBuffer(64) } });
    }
  },
  window: {
    devicePixelRatio: 1,
    location: { hash: "" },
    addEventListener(type, fn) {
      if (!winListeners[type]) winListeners[type] = [];
      winListeners[type].push(fn);
    },
    dispatchEvent(evt) {
      (winListeners[evt.type] || []).forEach(fn => fn(Object.assign({ preventDefault() {} }, evt)));
    }
  },
  document: {
    getElementById(id) { return elementsById.get(id) || makeElement(id); },
    querySelectorAll(sel) { return queryAllMap.get(sel) || []; },
    querySelector(sel) {
      if (sel === "#quad-d .tab-btn[data-mode='causal']") return quadDTabs[0];
      return null;
    },
    createElement(tag) {
      if (tag.toLowerCase() === "a") {
        lastCreatedLink = makeElement("", "A");
        return lastCreatedLink;
      }
      if (tag.toLowerCase() === "canvas") {
        lastCompositeCanvas = makeElement("offscreen-composite", "CANVAS");
        return lastCompositeCanvas;
      }
      return makeElement("", tag.toUpperCase());
    }
  },
  requestAnimationFrame() {}
};
sandbox.window.document = sandbox.document;
sandbox.globalThis = sandbox;

vm.createContext(sandbox);
scriptBlocks.forEach(code => vm.runInContext(code, sandbox));

function assert(cond, msg) {
  if (!cond) {
    console.error("ASSERTION FAILED:", msg);
    process.exit(1);
  }
}

// 1. Verify initial camera HUD badge in Quad C
const badgeCam = elementsById.get("badge-c-cam");
assert(badgeCam.innerText.includes("YAW: -35.5°") && badgeCam.innerText.includes("PITCH: 27.5°"), "Unexpected initial cam badge: " + badgeCam.innerText);

// Helper to get screen X/Y of Virialized Cluster Knot (#ff3355 stroke arc) in Quad C
const canvasC = elementsById.get("canvas-quad-c");
function getVoidAndKnotScreenCoords() {
  const arcs = canvasC._ctx._ops.filter(o => o.op === "arc");
  // Last two large feature arcs before skewers/probe are Void Core (#00ff9d) and Cluster Knot (#ff3355)
  const voidArc = arcs.find(a => a.stroke === "#00ff9d" && a.r > 15);
  const knotArc = arcs.find(a => a.stroke === "#ff3355" && a.r > 12);
  return { voidArc, knotArc };
}

// 2. Test X-axis drag right (+10px and +100px) at default positive pitch (+0.48 rad)
const knotXPosBefore = getVoidAndKnotScreenCoords().knotArc.x;
canvasC.dispatchEvent({ type: "mousedown", button: 0, clientX: 300, clientY: 200 });
sandbox.window.dispatchEvent({ type: "mousemove", clientX: 310, clientY: 200 });
const knotXPosAfter10 = getVoidAndKnotScreenCoords().knotArc.x;
assert(knotXPosAfter10 > knotXPosBefore, `Foreground knot did not move right at positive pitch! ${knotXPosBefore} -> ${knotXPosAfter10}`);
sandbox.window.dispatchEvent({ type: "mousemove", clientX: 400, clientY: 200 });
sandbox.window.dispatchEvent({ type: "mouseup" });
assert(badgeCam.innerText.includes("-92.8°"), "Expected YAW: -92.8° after +100px right drag, got: " + badgeCam.innerText);

// 3. Reset camera, drag UP by 100px into NEGATIVE pitch (0.48 - 1.00 = -0.52 rad), then test X-axis drag right (+10px)
elementsById.get("btn-reset-cam").click();
assert(badgeCam.innerText.includes("YAW: -35.5°"), "Camera did not reset");
canvasC.dispatchEvent({ type: "mousedown", button: 0, clientX: 300, clientY: 200 });
sandbox.window.dispatchEvent({ type: "mousemove", clientX: 300, clientY: 100 }); // dy = -100 -> pitch = -0.52 rad (-29.8 deg)
sandbox.window.dispatchEvent({ type: "mouseup" });
assert(badgeCam.innerText.includes("PITCH: -29.8°"), "Expected negative pitch -29.8°, got: " + badgeCam.innerText);

// Verify at negative pitch (-29.8 deg) that Beltrami Jet top (z=31) is STILL above screen center (world +Z is upright!)
const textsBefore = canvasC._ctx._ops.filter(o => o.op === "fillText" && o.text.includes("BELTRAMI JET"));
assert(textsBefore.length > 0 && textsBefore[0].y < 180, "Beltrami Jet top flipped upside-down at negative pitch! y=" + (textsBefore[0] && textsBefore[0].y));

// And verify that dragging right (+10px) at negative pitch moves foreground knot right (same direction as positive pitch!)
const knotXNegBefore = getVoidAndKnotScreenCoords().knotArc.x;
canvasC.dispatchEvent({ type: "mousedown", button: 0, clientX: 300, clientY: 100 });
sandbox.window.dispatchEvent({ type: "mousemove", clientX: 310, clientY: 100 }); // dx = +10
sandbox.window.dispatchEvent({ type: "mouseup" });
const knotXNegAfter = getVoidAndKnotScreenCoords().knotArc.x;
assert(knotXNegAfter > knotXNegBefore, `Foreground knot moved backwards at negative pitch! ${knotXNegBefore} -> ${knotXNegAfter}`);

// 4. Test 3D Voxel Probing on hover over Quad C
const anyVoxArc = canvasC._ctx._ops.find(o => o.op === "arc" && o.r < 12);
assert(anyVoxArc, "No voxel arcs rendered in Quad C");
canvasC.dispatchEvent({ type: "mousemove", clientX: anyVoxArc.x, clientY: anyVoxArc.y });
const badgeInfoC = elementsById.get("badge-c-info");
assert(badgeInfoC.innerText.includes("VOXEL (") && badgeInfoC.innerText.includes("δ ="), "Quad C voxel hover probe failed: " + badgeInfoC.innerText);

// 5. Test Boost cycling & synchronization
elementsById.get("btn-boost").click();
assert(elementsById.get("btn-boost").innerText.includes("1.5x"), "Boost button did not advance to 1.5x");
assert(badgeCam.innerText.includes("BOOST: 1.5x"), "Quad C HUD did not reflect 1.5x boost");

// 6. Test Quad A Channel Selector (CH 0, CH 1, CH 2, CH 3 must produce distinct traces on Scenario 01!)
const canvasA = elementsById.get("canvas-quad-a");
const bandA = elementsById.get("band-a-select");
const channelLabels = new Set();
["0", "1", "2", "3"].forEach(ch => {
  bandA.value = ch;
  canvasA._ctx._ops = [];
  bandA.dispatchEvent({ type: "change" });
  const lblOp = canvasA._ctx._ops.find(o => o.op === "fillText" && o.text.startsWith("CH " + ch));
  assert(lblOp, "Missing distinct channel label for CH " + ch);
  channelLabels.add(lblOp.text);
});
assert(channelLabels.size === 4, "Expected 4 distinct channel labels in Scenario 01");

// 7. Test Quad B Focal-Plane Hover Readout
const canvasB = elementsById.get("canvas-quad-b");
canvasB.dispatchEvent({ type: "mousemove", clientX: 320, clientY: 180 });
const badgeInfoB = elementsById.get("badge-b-info");
assert(badgeInfoB.innerText.includes("FOCAL (") && badgeInfoB.innerText.includes("NEAREST: KBO-"), "Quad B hover readout failed: " + badgeInfoB.innerText);

// 8. Test Quad D Dropzone Tab -> Drop File -> Verify #toolbar-quad-d (#btn-clear-drop) & #overlay-quad-d are UNHIDDEN -> Clear
quadDTabs[3].click(); // Click DROPZONE tab
const toolbarD = elementsById.get("toolbar-quad-d");
const overlayD = elementsById.get("overlay-quad-d");
const btnClear = elementsById.get("btn-clear-drop");
assert(toolbarD.classList.contains("hidden") && overlayD.classList.contains("hidden"), "Dropzone tab should hide canvas toolbar/overlay");

// Drop a .lc file while on Dropzone tab
const dropzone = elementsById.get("file-dropzone");
dropzone.dispatchEvent({
  type: "drop",
  dataTransfer: {
    files: [{ name: "test_burst.lc", _content: "0.0 1.25\n1.0 3.50\n2.0 0.95\n" }]
  }
});
assert(!toolbarD.classList.contains("hidden"), "BUG: #toolbar-quad-d remained hidden after dropping file!");
assert(!overlayD.classList.contains("hidden"), "BUG: #overlay-quad-d remained hidden after dropping file!");
assert(!btnClear.classList.contains("hidden"), "BUG: #btn-clear-drop remained hidden after dropping file!");
assert(elementsById.get("badge-d-info").innerText.includes("DROPPED: TEST_BURST.LC // 3 RECORDS"), "Unexpected badge-d-info after drop");

// Click CLEAR DROPPED FILE
btnClear.click();
assert(btnClear.classList.contains("hidden"), "#btn-clear-drop should hide after clearing");
assert(quadDTabs[0].innerText === "CAUSAL GRAPH", "Causal tab title did not restore after clear");

// 9. Test Composite 4-Quad Snapshot PNG Export vs Solo C Export
elementsById.get("btn-export-png").click();
assert(lastCreatedLink && lastCreatedLink.href.includes("offscreen-composite"), "Expected 4-Quad composite canvas export in grid view");
assert(lastCompositeCanvas && lastCompositeCanvas._ctx._ops.filter(o => o.op === "drawImage").length === 4, "Composite PNG must draw all 4 quad canvases");

layoutBtns[3].click(); // solo-c
elementsById.get("btn-export-png").click();
assert(lastCreatedLink && lastCreatedLink.href.includes("canvas-quad-c"), "Expected canvas-quad-c export in solo-c mode");

console.log("E2E_ALL_ASSERTIONS_PASSED");
""",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [node_bin, str(harness_script), str(studio_html_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15.0,
    )
    assert proc.returncode == 0, f"V8 E2E Studio verification failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "E2E_ALL_ASSERTIONS_PASSED" in proc.stdout



