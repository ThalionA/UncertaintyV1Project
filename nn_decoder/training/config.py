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
that ``run_experiment.run_animal_decoder`` consumes via
:meth:`Config.to_legacy_dict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


# Short-name -> legacy which_model. Keep in sync with the branches in
# run_experiment.run_animal_decoder (real-targets block).
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
VALID_PCA_BASES = ('condition_mean', 'residual')


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

    # ----- PCA loss basis (PCA-loss targets only) -----
    # 'condition_mean': PCA is fit on per-(o,c,d)-cell averaged training
    #   targets. The dominant PCs are then the across-condition Q axes;
    #   the closed-form loss minimum is "predict the per-condition
    #   training mean", which stim_mean_baseline.py provides directly.
    #   This is the historical production basis (and the bias source
    #   flagged in GOTCHAS: "PCA-weighted Euclidean loss measures
    #   across-condition variation only").
    # 'residual': PCA is fit on per-trial (target - cond_mean_train_target),
    #   isolating within-cell deviations. The dominant PCs are then the
    #   within-condition trial-to-trial axes, and the loss scores
    #   trial-level information that stim_mean cannot.
    # Ignored when custom_loss_func is not 'PCA' (CE/MSE branches use the
    # raw target directly). Kept on Config for schema consistency so
    # provenance YAMLs always record the intent.
    pca_basis: str = 'condition_mean'

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
        if self.pca_basis not in VALID_PCA_BASES:
            raise ValueError(
                f"Unknown pca_basis {self.pca_basis!r}; valid: {VALID_PCA_BASES}"
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
            "weight_decay":          self.weight_decay,
            "optimizer_type":        self.optimizer_type,
            "momentum":              self.momentum,
            "num_epochs":            self.num_epochs,
            "minibatch_size":        self.minibatch_size,
            "REP":                   self.REP,
            "pca_basis":             self.pca_basis,
        }

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------

    def slug(self) -> str:
        """Directory slug encoding (target, loss, window, bin_size,
        [pca_basis]). The pca_basis suffix is only appended for PCA-loss
        targets (it's a no-op for CE/MSE) so non-PCA slugs are unchanged
        and existing on-disk paths remain stable. Within PCA targets,
        condmean/residual write to different directories so the two
        bases can coexist."""
        base = f"{self.target_type}_{self.loss_func}_{self.time_window}_{self.bin_size_ms}ms"
        if self.loss_func == 'PCA':
            short = {'condition_mean': 'condmean', 'residual': 'residual'}[self.pca_basis]
            return f"{base}_{short}"
        return base

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
# Per-target presets, keyed on (target, bin_size_ms)
# ----------------------------------------------------------------------

# Optuna sweeps populate these entries as evidence comes in. A bin_size_ms
# of ``None`` means "this preset applies to any bin size that doesn't
# have its own entry yet" — used as a fallback for targets that haven't
# been per-bin-size tuned. When you Optuna-tune a target at a specific
# bin size, add the explicit (target, bin_size_ms) entry; the lookup
# in :func:`default_config_for_target` prefers exact matches over the
# fallback.
#
# WHY 50 ms vs 100 ms matters: at 50 ms there are 2x more time bins per
# trial (20 bins vs 10 in the 'half' window), so the gradient signal per
# trial is finer. Empirically the 50 ms optimum sits at a LOWER learning
# rate and MORE epochs than the 100 ms optimum, even though both end up
# at essentially the same validation score.

_PRESETS = {
    # ----- Q (perceptual posterior) -----
    # Tuned per bin size via Optuna sweep (2026-05-06).
    ('Q', 50):  dict(
        loss_func='PCA',
        hidden_sizes=[32],
        learning_rate=5.786e-4,
        weight_decay=1.309e-5,
        minibatch_size=16,
        num_epochs=50,
        entropy_lambda=9.891e-3,
        # Optuna best score = 0.3807 (PPC 0.367, SBC 0.394)
    ),
    ('Q', 100): dict(
        loss_func='PCA',
        hidden_sizes=[32],
        learning_rate=1.679e-3,
        weight_decay=1.388e-5,
        minibatch_size=16,
        num_epochs=30,
        entropy_lambda=5.293e-3,
        # Optuna best score = 0.3792 (PPC 0.375, SBC 0.383)
    ),

    # ----- L (marginalised likelihood) -----
    ('L', None): dict(
        loss_func='PCA',
        hidden_sizes=[32],
        learning_rate=0.0002765,
        weight_decay=3.96e-05,
        minibatch_size=8,
        num_epochs=75,
        entropy_lambda=0.00229,
        # Same family as Q (91-bin distributional). Likelihood is the
        # prior-free version; pre-Optuna default mirrors Q's old preset.
    ),

    # ----- d (decision posterior)  -----
    ('d', None): dict(
        loss_func='MSE',
        hidden_sizes=[16],
        learning_rate=0.0005643,
        weight_decay=2.946e-05,
        minibatch_size=16,
        num_epochs=75,
        entropy_lambda=0.0001687,
        # WHY MSE: PCA is undefined on a 2-D target. MSE on the soft
        # [P(Go), P(NoGo)] output.
    ),

    # ----- choice (animal goChoice) -----
    ('choice', None): dict(
        loss_func='CE',
        hidden_sizes=[32],
        learning_rate=0.0003677,
        weight_decay=4.096e-05,
        minibatch_size=16,
        num_epochs=30,
        entropy_lambda=0.0609,
    ),

    # ----- stim_kernel (Gaussian-smoothed delta at true theta) -----
    ('stim_kernel', None): dict(
        loss_func='PCA',
        hidden_sizes=[32],
        learning_rate=1e-3,
        num_epochs=30,
        # Same family as Q so that neural-state-space comparisons with
        # the Q decoder are apples-to-apples.
    ),

    # ----- stim_cat (one-hot at true theta bin) -----
    ('stim_cat', None): dict(
        loss_func='CE',
        hidden_sizes=[16, 16],
        learning_rate=0.001186,
        weight_decay=1.404e-05,
        minibatch_size=32,
        num_epochs=50,
        entropy_lambda=0.003337,
    ),
}


def _lookup_preset(target: str, bin_size_ms: int) -> dict:
    """Exact (target, bin_size_ms) match wins; otherwise fall back to
    (target, None). Raises ValueError if neither is present."""
    if (target, bin_size_ms) in _PRESETS:
        return dict(_PRESETS[(target, bin_size_ms)])
    if (target, None) in _PRESETS:
        return dict(_PRESETS[(target, None)])
    valid_targets = sorted({t for t, _ in _PRESETS})
    raise ValueError(
        f"No preset for target={target!r} bin_size_ms={bin_size_ms}; "
        f"valid targets: {tuple(valid_targets)}"
    )


def default_config_for_target(target: str, bin_size_ms: int = 100,
                                **overrides) -> Config:
    """Per-target default Config with documented reasoning. The lookup
    prefers a per-bin-size preset if one has been Optuna-tuned; otherwise
    falls back to the bin-size-agnostic default for that target.

    Pass keyword overrides for any other field (e.g.
    ``run_name='production_2026_05_06'``).
    """
    preset = _lookup_preset(target, bin_size_ms)
    preset['target_type'] = target
    preset['bin_size_ms'] = bin_size_ms
    preset.update(overrides)
    return Config(**preset)
