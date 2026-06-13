"""Filesystem paths for the RNN+RL similarity-framework model."""

from __future__ import annotations

import pathlib

PKG_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
FIGURES_DIR = PKG_DIR / "figures"
RESULTS_DIR = PKG_DIR / "results"
NN_DECODER_DIR = REPO_ROOT / "nn_decoder"


def ensure_dirs() -> None:
    """Create the figures/ and results/ output directories if missing."""
    FIGURES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
