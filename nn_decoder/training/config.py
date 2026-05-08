# -*- coding: utf-8 -*-
"""Single Config dataclass for V1-decoder training. Replaces the
ad-hoc dicts each ``run_fixed_hyperparams_*.py`` script used to build.

Per-target defaults live in :func:`default_config_for_target` with
documented reasoning for every value that differs across targets.
After per-target Optuna sweeps land, those defaults will be updated
in this single place.

Short-name target convention (Config-facing):

  ``Q``           — perceptual posterior, IO Q(theta)        (91 bins, PCA loss)
  ``L``           — marginalised likelihood L(theta)         (91 bins, PCA loss)
  ``d``           — IO decision posterior [P(Go), P(NoGo)]   (2 bins,  MSE loss)
  ``choice``      — animal's actual goChoice, one-hot        (2 bins,  CE loss)
  ``stim_kernel`` — Gaussian-smoothed delta at true theta    (91 bins, PCA loss)
  ``stim_cat``    — one-hot at true theta bin                (91 bins, CE loss)

These short names are translated to the legacy ``which_model`` strings
that ``run_experiment_v26.run_animal_decoder`` consumes via
:meth:`Config.to_legacy_dict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


# Short-name -> legacy which_model. Keep in sync with the branches in
# run_experiment_v26.run_animal_decoder (real-targets block).
TARGET_TO_WHICH_MODEL = {
    'Q':           'perception',
    'L':           'likelihood',
    'd':           'decision',
    'choice':      'true_choice',
    'stim_kernel': 'stim_kernel',
    'stim_cat':    'stim_cat',
}


VALID_LOSSES = ('PCA', 'MSE', 'CE', 'KL', 'JS', 'Wasserstein')
VALID_TIME_WINDOWS = ('full', 'half', 'last_quarter')
VALID_BIN_SIZES = (50, 100, 250)


@dataclass
class Config:
    # ----- Target -----
    target_type: str                            # short name (see module docstring)
    loss_func: str                              # 'PCA' | 'MSE' | 'CE' | 'KL' | 'JS' | 'Wasserstein'

    # ----- Data window -----
    time_window: str = 'half'                   # 'full' | 'half' | 'last_quarter'
    bin_size_ms: int = 100                      # 50 | 100 | 250

    # ----- Architecture (both PPC and SBC trained per call) -----
    hidden_sizes: List[int] = field(default_factory=lambda: [32])
    activation_function: str = 'tanh'
    weight_initialization: str = 'xavier_uniform'

    # ----- Optimiser -----
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer_type: str = 'adam'
    momentum: float = 0.9

    # ----- Schedule -----
    num_epochs: int = 30
    minibatch_size: int = 16
    REP: int = 5

    # ----- SBC sharpness penalty -----
    # 3e-3 across all targets after the lambda=3e3 bug was identified
    # (Session 2026-05-06). The high lambda forced per-bin SBC outputs to
    # near-{0,1}, which combined with one-hot CE targets produced a
    # degenerate gradient (NLL > log 2 on true_choice_SBC).
    entropy_lambda: float = 3e-3

    # ----- Split -----
    split_type: str = 'stratified_balanced'
    random_state: int = 42

    # ----- Output / provenance -----
    run_name: str = 'default'

    # ----- Optional metadata -----
    notes: Optional[str] = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self):
        if self.target_type not in TARGET_TO_WHICH_MODEL:
            raise ValueError(
                f"Unknown target_type {self.target_type!r}; "
                f"valid: {tuple(TARGET_TO_WHICH_MODEL)}"
            )
        if self.loss_func not in VALID_LOSSES:
            raise ValueError(
                f"Unknown loss_func {self.loss_func!r}; valid: {VALID_LOSSES}"
            )
        if self.time_window not in VALID_TIME_WINDOWS:
            raise ValueError(
                f"Unknown time_window {self.time_window!r}; valid: {VALID_TIME_WINDOWS}"
            )
        if self.bin_size_ms not in VALID_BIN_SIZES:
            raise ValueError(
                f"Unknown bin_size_ms {self.bin_size_ms}; valid: {VALID_BIN_SIZES}"
            )

    # ------------------------------------------------------------------
    # Translations
    # ------------------------------------------------------------------

    def to_legacy_dict(self) -> dict:
        """Translate to the dict shape ``run_animal_decoder`` consumes.

        Maps the short-name ``target_type`` to the legacy ``which_model``
        string and inlines every other field with the legacy key name.
        """
        return {
            "target_source":         "real",
            "time_window":           self.time_window,
            "bin_size_ms":           self.bin_size_ms,
            "split_type":            self.split_type,
            "which_model":           TARGET_TO_WHICH_MODEL[self.target_type],
            "hidden_sizes":          list(self.hidden_sizes),
            "activation_function":   self.activation_function,
            "weight_initialization": self.weight_initialization,
            "custom_loss_func":      self.loss_func,
            "entropy_lambda":        self.entropy_lambda,
            "learning_rate":         self.learning_rate,
            "optimizer_type":        self.optimizer_type,
            "momentum":              self.momentum,
            "num_epochs":            self.num_epochs,
            "minibatch_size":        self.minibatch_size,
            "REP":                   self.REP,
        }

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------

    def slug(self) -> str:
        """Directory slug encoding (target, loss, window, bin_size)."""
        return f"{self.target_type}_{self.loss_func}_{self.time_window}_{self.bin_size_ms}ms"

    def output_dir(self, results_root='results') -> Path:
        """Run-name-prefixed nested tree:
        ``<results_root>/<run_name>/<slug>/``."""
        return Path(results_root) / self.run_name / self.slug()

    # ------------------------------------------------------------------
    # YAML provenance
    # ------------------------------------------------------------------

    def save_yaml(self, path: Path) -> None:
        """Serialise to YAML for run provenance. Requires PyYAML."""
        import yaml
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: Path) -> 'Config':
        """Load a saved Config back from disk."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


# ----------------------------------------------------------------------
# Per-target presets
# ----------------------------------------------------------------------

# Each entry documents WHY the values differ from the global defaults,
# so future readers don't strip the differences as drift. Optuna sweeps
# overwrite these as evidence comes in.
_PRESETS = {
    'Q': dict(
        loss_func='PCA',
        hidden_sizes=[32],
        learning_rate=1e-3,
        num_epochs=30,
        # WHY PCA: leading PCs of per-condition averaged Q encode peak
        # position; choice prediction depends on peak position relative
        # to 45 deg, so PCA-weighted Euclidean is best for downstream
        # behavioural prediction. Confirmed via Wasserstein/KL/JS sweep
        # (Session 2026-05-06): all losses sit ~0.05 NLL apart on Q.
    ),
    'L': dict(
        loss_func='PCA',
        hidden_sizes=[32],
        learning_rate=1e-3,
        num_epochs=30,
        # Same family as Q (91-bin distributional). Likelihood is the
        # prior-free version; everything else identical.
    ),
    'd': dict(
        loss_func='MSE',
        hidden_sizes=[32],
        learning_rate=1e-3,
        num_epochs=30,
        # WHY MSE: PCA is undefined on a 2-D target. MSE is the
        # straightforward squared-error fit on the soft [P(Go), P(NoGo)]
        # IO output.
    ),
    'choice': dict(
        loss_func='CE',
        hidden_sizes=[16],
        learning_rate=5e-3,
        num_epochs=50,
        # WHY 16/5e-3/50:
        #   smaller net — output is binary, less capacity needed.
        #   higher LR  — CE has a sharper loss landscape than PCA-Euclid.
        #   more epochs — CE convergence on noisy binary targets benefits
        #                  from more iterations.
        # These differ from Q/L/d. Documented as drift-suspicious until
        # the per-target Optuna sweep validates them.
    ),
    'stim_kernel': dict(
        loss_func='PCA',
        hidden_sizes=[32],
        learning_rate=1e-3,
        num_epochs=30,
        # Gaussian-smoothed delta at true orientation, 91 bins. Same
        # family as Q so the comparison of neural-state-space dimensions
        # used by the orientation decoder vs the Q decoder is
        # apples-to-apples (same loss, same architecture, same target
        # shape).
    ),
    'stim_cat': dict(
        loss_func='CE',
        hidden_sizes=[32],
        learning_rate=1e-3,
        num_epochs=30,
        # One-hot at true orientation bin, 91 bins. Categorical decoding
        # via CE. Larger net than the binary 'choice' case because the
        # output is 91-D (much more capacity needed for one-hot
        # discrimination).
    ),
}


def default_config_for_target(target: str, **overrides) -> Config:
    """Per-target default Config with documented reasoning. Pass keyword
    overrides for any field (e.g. ``run_name='production_2026_05_06'``,
    ``bin_size_ms=50``)."""
    if target not in _PRESETS:
        raise ValueError(
            f"Unknown target {target!r}; valid: {tuple(_PRESETS)}"
        )
    preset = dict(_PRESETS[target])
    preset['target_type'] = target
    preset.update(overrides)
    return Config(**preset)
