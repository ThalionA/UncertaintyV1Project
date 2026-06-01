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
VALID_PCA_BASES = ('all_trials', 'condition_mean', 'residual')


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
    # 'all_trials' (default): PCA is fit on the raw per-trial training
    #   targets -- no condition averaging, no cell-mean subtraction. The
    #   dominant PCs capture across- and within-condition variance
    #   together, in proportion to how much each contributes to the
    #   total trial-to-trial spread.
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
    pca_basis: str = 'all_trials'

    # ----- Split -----
    split_type: str = 'stratified_balanced'
    random_state: int = 42

    # ----- Output / provenance -----
    run_name: str = 'default'

    # ----- Diagnostic checkpointing (default off — production runs
    # are unchanged). When ``track_training_history`` is true,
    # ``fit_model`` populates a per-epoch history dict (train fit
    # loss, train total loss, entropy penalty mean, per-parameter
    # weight L2 norms, and a PCA-projected eval loss used as a
    # cross-loss yardstick) and ``train_and_select_best_model`` saves
    # the winning restart's history into the per-arch Checkpoints
    # bundle. ``weight_snapshot_every`` additionally writes a
    # CPU-deep-copied ``state_dict`` snapshot every N epochs (and at
    # the last epoch). Disk cost grows linearly with snapshots × mice
    # × archs × losses, so set this only when running an exploratory
    # sweep.
    track_training_history: bool = False
    weight_snapshot_every: int = 0

    # ----- Validation split (default 0 = no val). When > 0, this
    # fraction of the training trials is carved off as a validation
    # set, stratified on the stim cell category so the per-condition
    # composition matches train. The PCA basis is fit on the
    # remaining training trials only — val never leaks into the basis.
    # The val PCA-yardstick loss is logged per-epoch into the history
    # dict (when track_training_history is also on) so the train-vs-val
    # gap is readable directly off the curves; REP selection still
    # uses the training loss to keep results comparable across runs
    # without val. Typical exploratory value: 0.15.
    val_frac: float = 0.0

    # ----- Early stopping (default off — production runs unchanged).
    # patience=0 keeps the fixed-`num_epochs` schedule exactly. When >0,
    # fit_model monitors the held-out validation fit-loss and stops once
    # it has not improved for `patience` epochs (after `min_epochs`),
    # restoring the best weights. The stop signal reuses the `val_frac`
    # carve (stratified) when one exists; otherwise fit_model carves a
    # seeded `val_fraction` slice of the training trials. With early
    # stopping on, set `num_epochs` to the epoch cap (e.g. 200). The
    # stop signal is the fit-loss only — entropy_lambda never enters it.
    patience: int = 0
    min_epochs: int = 0
    val_fraction: float = 0.2

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
            "track_training_history": self.track_training_history,
            "weight_snapshot_every":  self.weight_snapshot_every,
            "val_frac":               self.val_frac,
            "patience":               self.patience,
            "min_epochs":             self.min_epochs,
            "val_fraction":           self.val_fraction,
        }

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------

    def slug(self) -> str:
        """Directory slug encoding (target, loss, window, bin_size,
        [pca_basis]). The pca_basis suffix is only appended for PCA-loss
        targets (it's a no-op for CE/MSE) so non-PCA slugs are unchanged
        and existing on-disk paths remain stable. Within PCA targets,
        all/condmean/residual write to different directories so the
        three bases can coexist."""
        base = f"{self.target_type}_{self.loss_func}_{self.time_window}_{self.bin_size_ms}ms"
        if self.loss_func == 'PCA':
            short = {'all_trials': 'all', 'condition_mean': 'condmean',
                     'residual': 'residual'}[self.pca_basis]
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
    ('Q', 50):  dict(
        loss_func='PCA',
        hidden_sizes=[32],
        learning_rate=5.786e-4,
        weight_decay=1.309e-5,
        minibatch_size=16,
        num_epochs=100,
        entropy_lambda=9.891e-3,
        # Optuna best score = 0.3807 (PPC 0.367, SBC 0.394). Swept
        # 2026-05-06 under the pre-vectorisation training loop.
    ),
    ('Q', 100): dict(
        loss_func='PCA',
        hidden_sizes=[32],
        learning_rate=0.001252,
        weight_decay=0.0001201,
        minibatch_size=16,
        num_epochs=100,
        entropy_lambda=0.001841,
        # Optuna best score = 0.3624 (PPC 0.355, SBC 0.370). Swept
        # 2026-05-24 under the all_trials PCA basis (the current default),
        # vectorised training loop, n_trials=100 (41 completed).
    ),

    # ----- L (marginalised likelihood) -----
    ('L', 100): dict(
        loss_func='PCA',
        hidden_sizes=[32],
        learning_rate=0.0002364,
        weight_decay=2.134e-05,
        minibatch_size=8,
        num_epochs=100,
        entropy_lambda=0.00269,
        # Optuna best score = 0.3576 (PPC 0.351, SBC 0.364). Swept
        # 2026-05-24 under the all_trials PCA basis (the current default),
        # vectorised training loop, n_trials=100 (41 completed).
    ),
    ('L', None): dict(
        loss_func='PCA',
        hidden_sizes=[32],
        learning_rate=0.0002765,
        weight_decay=3.96e-05,
        minibatch_size=8,
        num_epochs=75,
        entropy_lambda=0.00229,
        # Pre-vectorisation fallback (50/250 ms). Same family as Q
        # (91-bin distributional); pre-Optuna default mirrored Q's old
        # preset. 100 ms is now served by the explicit ('L', 100) entry.
    ),

    # ----- d (decision posterior)  -----
    ('d', 100): dict(
        loss_func='MSE',
        hidden_sizes=[32],
        learning_rate=0.0001637,
        weight_decay=1.648e-05,
        minibatch_size=16,
        num_epochs=100,
        entropy_lambda=0.001634,
        # Optuna best score = 0.4664 (PPC 0.492, SBC 0.441). Swept
        # 2026-05-23 under the vectorised training loop (43 completed).
        # WHY MSE: PCA is undefined on a 2-D target.
    ),
    ('d', None): dict(
        loss_func='MSE',
        hidden_sizes=[16],
        learning_rate=0.0005643,
        weight_decay=2.946e-05,
        minibatch_size=16,
        num_epochs=75,
        entropy_lambda=0.0001687,
        # Pre-vectorisation fallback (50/250 ms). WHY MSE: PCA is
        # undefined on a 2-D target. MSE on the soft [P(Go), P(NoGo)]
        # output. 100 ms is now served by the explicit ('d', 100) entry.
    ),

    # ----- choice (animal goChoice) -----
    ('choice', 100): dict(
        loss_func='CE',
        hidden_sizes=[32],
        learning_rate=0.0003177,
        weight_decay=0.0003159,
        minibatch_size=16,
        num_epochs=30,
        entropy_lambda=0.002488,
        # Optuna best score = 0.7067 (PPC 0.717, SBC 0.697). Swept
        # 2026-05-23 under the vectorised training loop. Only 23/100
        # trials completed (aggressive MedianPruner n_warmup_steps=1),
        # but the optimum matches the prior independent ('choice', None)
        # sweep on architecture and num_epochs -- accepted as stable.
    ),
    ('choice', None): dict(
        loss_func='CE',
        hidden_sizes=[32],
        learning_rate=0.0003677,
        weight_decay=4.096e-05,
        minibatch_size=16,
        num_epochs=30,
        entropy_lambda=0.0609,
        # Pre-vectorisation fallback (50/250 ms). 100 ms is now served
        # by the explicit ('choice', 100) entry.
    ),

    # ----- stim_kernel (Gaussian-smoothed delta at true theta) -----
    ('stim_kernel', 100): dict(
        loss_func='PCA',
        hidden_sizes=[32],
        learning_rate=0.0002094,
        weight_decay=0.0001171,
        minibatch_size=16,
        num_epochs=200,
        entropy_lambda=0.0005594,
        # Optuna best score = 0.4757 (PPC 0.489, SBC 0.462). Swept
        # 2026-05-24 under the all_trials PCA basis (the current default),
        # vectorised training loop, n_trials=100 (33 completed). Same
        # family as Q so neural-state-space comparisons with the Q
        # decoder stay apples-to-apples.
    ),
    ('stim_kernel', None): dict(
        loss_func='PCA',
        hidden_sizes=[32],
        learning_rate=1e-3,
        num_epochs=30,
        # Pre-vectorisation fallback (50/250 ms); hand-set default, never
        # Optuna-tuned. 100 ms is now served by ('stim_kernel', 100).
    ),

    # ----- stim_cat (one-hot at true theta bin) -----
    ('stim_cat', 100): dict(
        loss_func='CE',
        hidden_sizes=[16],
        learning_rate=0.0001107,
        weight_decay=3.198e-05,
        minibatch_size=8,
        num_epochs=200,
        entropy_lambda=0.03187,
        # Optuna best score = 0.6787 (PPC 0.674, SBC 0.683). Swept
        # 2026-05-23 under the vectorised training loop (52 completed).
    ),
    ('stim_cat', None): dict(
        loss_func='CE',
        hidden_sizes=[16, 16],
        learning_rate=0.001186,
        weight_decay=1.404e-05,
        minibatch_size=32,
        num_epochs=50,
        entropy_lambda=0.003337,
        # Pre-vectorisation fallback (50/250 ms). 100 ms is now served
        # by the explicit ('stim_cat', 100) entry.
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
