"""
astrosnail.tests.e2e.conftest
=============================
Shared fixtures, mathematical reference oracles, and filesystem utilities
for the 4-tier requirement-driven E2E test suite.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Tuple, Union

import numpy as np
import pytest

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
ASTROSNAIL_DIR = REPO_ROOT / "astrosnail"
FORMALIZATION_DIR = REPO_ROOT / "formalization"
DATA_DIR = REPO_ROOT / "data"
REAL_STANDARDS_DIR = DATA_DIR / "real_standards"
CATALOGS_DIR = DATA_DIR / "catalogs"
RUST_DIR = REPO_ROOT / "rust" / "astrosnail_rs"
SCRIPTS_DIR = REPO_ROOT / "scripts"
DOCS_DIR = REPO_ROOT / "docs"

PRUNE_DIRS = {
    ".git",
    ".lake",
    "target",
    "__pycache__",
    ".pytest_cache",
    "build",
    "astrosnail.egg-info",
    ".system_generated",
    ".venv",
    "node_modules",
}


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def astrosnail_dir() -> Path:
    return ASTROSNAIL_DIR


@pytest.fixture(scope="session")
def formalization_dir() -> Path:
    return FORMALIZATION_DIR


@pytest.fixture(scope="session")
def real_standards_dir() -> Path:
    required_files = [
        "kepler_hatp7b_slc.fits",
        "2mass_galactic_center_h.fits",
        "l1448_13co_cube.fits",
    ]
    if any(not (REAL_STANDARDS_DIR / f).exists() for f in required_files):
        try:
            import sys
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            if str(SCRIPTS_DIR) not in sys.path:
                sys.path.insert(0, str(SCRIPTS_DIR))
            try:
                from scripts.run_astronomical_smoke_test import ensure_real_standards
            except ImportError:
                from run_astronomical_smoke_test import ensure_real_standards
            ensure_real_standards(REAL_STANDARDS_DIR)
        except Exception as exc:
            print(f"Warning: Failed to auto-seed real standards: {exc}")
    return REAL_STANDARDS_DIR


@pytest.fixture(scope="session")
def catalogs_dir() -> Path:
    return CATALOGS_DIR


@pytest.fixture(scope="session")
def rust_dir() -> Path:
    return RUST_DIR


@pytest.fixture(scope="session")
def scripts_dir() -> Path:
    return SCRIPTS_DIR


@pytest.fixture(scope="session")
def docs_dir() -> Path:
    return DOCS_DIR


def walk_clean_files(
    root: Path,
    extensions: Optional[Tuple[str, ...]] = None,
) -> Iterator[Path]:
    """Iterates through all files under root while strictly pruning build/cache/git directories."""
    for dirpath, dirnames, filenames in os.walk(str(root)):
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        for f in filenames:
            p = Path(dirpath) / f
            if extensions is None or p.suffix.lower() in extensions:
                yield p


@pytest.fixture(scope="session")
def clean_file_scanner():
    """Provides a safe generator to walk files without scanning huge dependencies."""
    return walk_clean_files


# =============================================================================
# Pure Mathematical Reference Oracles (Independent Authoritative Specifications)
# =============================================================================

class MathOracles:
    """Independent pure-Python reference implementations of mathematical theorems
    and invariants derived from theoretical specifications.
    """

    @staticmethod
    def skew_triad_contract(
        W: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Independent implementation of triadOperator:
        B_W(u, v)_k = sum_{i, j} u_i * v_j * (W_{i, j, k} - W_{i, k, j})
        """
        dim = len(u)
        result = np.zeros(dim, dtype=np.float64)
        for k in range(dim):
            s = 0.0
            for i in range(dim):
                for j in range(dim):
                    s += float(u[i]) * float(v[j]) * (float(W[i, j, k]) - float(W[i, k, j]))
            result[k] = s
        return result

    @staticmethod
    def discrete_morse_contract(
        flux: np.ndarray,
        salience: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Independent implementation of Discrete Morse impulse:
        I[i] = (flux[i] - flux[i-1]) * (flux[i] - flux[i+1]) * (salience[i] if salience is not None else 1.0)
        Shields ramp/monotonic segments (<= 0) and isolates local extrema (> 0).
        """
        arr = np.asarray(flux, dtype=np.float64)
        n = len(arr)
        impulse = np.zeros(n, dtype=np.float64)
        if n < 3:
            return impulse

        sal = np.ones(n, dtype=np.float64) if salience is None else np.asarray(salience, dtype=np.float64)
        for i in range(1, n - 1):
            d_left = arr[i] - arr[i - 1]
            d_right = arr[i] - arr[i + 1]
            # Both sides must drop (strict local maximum) or both sides rise (strict local minimum)
            prod = d_left * d_right
            if prod > 0.0:
                impulse[i] = prod * sal[i]
            else:
                impulse[i] = 0.0
        return impulse

    @staticmethod
    def kelvin_mandel_trace_contract(
        R: np.ndarray,
        shear_6: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Independent implementation of SO(6) Kelvin-Mandel rotor trace conservation:
        Tr(C') = Tr(C)
        """
        r = np.asarray(R, dtype=np.float64)
        M = np.zeros((6, 6), dtype=np.float64)
        sqrt2 = math.sqrt(2.0)

        # Diagonal block
        for i in range(3):
            for j in range(3):
                M[i, j] = r[i, j] ** 2

        # Shear pairs
        shear_pairs = [(1, 2), (0, 2), (0, 1)]
        for i in range(3):
            for idx, (a, b) in enumerate(shear_pairs):
                col = 3 + idx
                M[i, col] = sqrt2 * r[i, a] * r[i, b]

        for idx, (a, b) in enumerate(shear_pairs):
            row = 3 + idx
            for j in range(3):
                M[row, j] = sqrt2 * r[a, j] * r[b, j]

        for k_idx, (a, b) in enumerate(shear_pairs):
            row = 3 + k_idx
            for l_idx, (c, d) in enumerate(shear_pairs):
                col = 3 + l_idx
                M[row, col] = r[a, c] * r[b, d] + r[a, d] * r[b, c]

        rotated = np.dot(M, shear_6)
        orig_trace = float(shear_6[0] + shear_6[1] + shear_6[2])
        new_trace = float(rotated[0] + rotated[1] + rotated[2])
        return rotated, abs(new_trace - orig_trace)

    @staticmethod
    def transit_depth_geometry_contract(
        radius_ratio: float,
        impact_param: float = 0.0,
    ) -> float:
        """Geometric transit depth for exoplanet of radius R_p across star R_*
        with radius_ratio k = R_p / R_*:
        delta = k^2 for central transit (b = 0).
        """
        k = float(radius_ratio)
        b = float(impact_param)
        if b + k <= 1.0:
            # Full interior transit (no limb grazing)
            return k * k
        elif b - k >= 1.0:
            # No occultation
            return 0.0
        else:
            # Geometric chord overlap
            return float(max(0.0, min(1.0, k * k)))


@pytest.fixture(scope="session")
def math_oracle() -> MathOracles:
    return MathOracles()
