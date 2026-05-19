# -*- coding: utf-8 -*-
"""Unified training infrastructure for the V1 decoder.

Single source of truth for:

  - ``Config`` dataclass (all training-time decisions, with documented
    per-target defaults)
  - ``make_target`` (per-target construction of the supervised label
    array, used by ``run_experiment.run_animal_decoder``)
  - ``run_config`` (driver that runs one Config across (mouse x split)
    and writes a structured ``results/<run_name>/<slug>/<split>.mat``
    tree alongside a serialised ``config.yaml`` for provenance)

Existing analysis modules are not yet moved here — that's a separate
refactor pass once the fresh production run lands.
"""

from .config import Config, default_config_for_target  # noqa: F401
from .targets import make_target  # noqa: F401
from .run import run_config  # noqa: F401
