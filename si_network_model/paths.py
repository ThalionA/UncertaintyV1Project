"""Filesystem paths for the SI-framework network model."""

from __future__ import annotations

import pathlib

PKG_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
FIGURES_DIR = PKG_DIR / "figures"
RESULTS_DIR = PKG_DIR / "results"
IO_HMM_DIR = REPO_ROOT / "ideal_observer" / "io_hmm"


def ensure_dirs() -> None:
    """Create the figures/ and results/ output directories if missing."""
    FIGURES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
