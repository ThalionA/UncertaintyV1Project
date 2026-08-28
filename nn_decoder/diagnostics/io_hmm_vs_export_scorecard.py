# -*- coding: utf-8 -*-
"""Paired old-vs-new scorecard: does swapping the target family change what the
decoder DOES? (2026-08-12, mouse 0 — the only animal with IO-HMM posteriors.)

Two arms, identical cells / seeds / splits / neural data / code path; the ONLY
difference is the target family:

  NEW  results/io_hmm_v1/<cell>/           collaborator's ideal-observer HMM
                                           posteriors, 72 CIRCULAR bins x 2.5 deg
                                           over [0, 180)
  OLD  results/io_hmm_v1_exportref/<cell>/ the export's post_s_marginal Q,
                                           91 LINEAR bins x 1 deg over 0-90

so every cell present in both arms is a strictly paired observation and the
replication unit is the CELL, not the animal (n = 1 mouse; no error bars across
animals are drawn or implied anywhere here).

THE FOUR SHARED (CROSS-ARM) AXES
--------------------------------
  KL skill        mean KL(tgt||dec) / mean KL(tgt||LOO predict-mean)   [1 = null]
  projection skill  same ratio under the projection (PCA) loss, arm-pinned evar,
                  recomputed on the HELD-OUT TEST rows
  s_hat           equivalent sharpening factor, calibrated per arm against that
                  arm's OWN targets (see below)                 [1 = calibrated]
  overfitting gap ABSOLUTE (val_fit - train_fit) at best_epoch, DIVIDED BY that
                  arm's own LOO predict-mean loss under the cell's own training
                  loss                                                [0 = none]

Secondary panels (diagnostics, not claims): s_hat's peak-vs-entropy agreement,
best_epoch, early_stopped_epoch.

THE PEAKINESS AXIS — SETTLED, DO NOT REOPEN
-------------------------------------------
Two earlier rounds each shipped a "dimensionless" peakiness metric that turned
out to have an arm-specific ceiling. A ground-truth test then settled it:
synthetic decoders sharpened by a KNOWN factor, rendered on BOTH supports, at
matched true sharpening. Every fixed width statistic disagreed across arms:

    max-prob ratio             22%      perplexity-width ratio       25%
    raw Shannon entropy ratio  15%      Mardia circular SD ratio     26%
    axial dispersion ratio     37%      differential-entropy delta   54%
    resultant (R) ratio        60%      linear SD ratio              69%

The cause is not bin width: the two target SHAPES differ (the export Q is a
bimodal cardinal-U at low contrast, the IO-HMM posterior is flat-with-a-tilt),
and every width statistic responds to sharpening in a shape-dependent way. No
rescaling of a width statistic can fix that.

The accepted replacement is ``nn_decoder/sharpening_calibration.py``: per arm,
build a calibration curve from THAT ARM'S OWN targets using
``sharpen(p, s) = p ** (s**2)`` renormalised (a Gaussian-exact width shrink by
``s``, defined support-freely — it is a pointwise map, no grid, no metric), then
invert the observed statistic back to the equivalent sharpening factor ``s_hat``.
It recovers a known ``s`` to a 0.00% cross-arm gap.

    s_hat > 1  over-sharpened    s_hat = 1  calibrated    s_hat < 1  too broad

s_hat is ALWAYS reported with two companions, because it describes sharpening and
real decoders also reshape:

  * ``s_hat_agreement`` — relative discrepancy between the peak-derived and the
    entropy-derived s_hat. Above ~0.10 the decoder is RESHAPING, not merely
    sharpening, and s_hat under-describes it (flagged with a ring in the figures
    and a ``!`` in the tables).
  * ``s_hat_clamped`` — the inversion hit a ladder endpoint (0.25 or 6.0), so the
    number is a BOUND, not an estimate (flagged with a cross / ``clamp``).

The eight rejected statistics survive ONLY as within-arm CSV columns, under
deliberately awkward names (``*_within_arm_only``). They appear in NO figure and
in NO paired comparison here. The single figure-free exception is ``--invariance``,
which is a SINGLE-ARM coarse-graining demonstration and says so.

GEOMETRY IS READ FROM PROVENANCE, NEVER GUESSED FROM THE BIN COUNT
-------------------------------------------------------------------
An earlier version special-cased ``n_bins == 91`` and treated every other grid as
circular [0,180) — so any future support (a 45-bin linear grid, a rebinned
export) would have been silently scored on the wrong geometry. The support now
comes from ``results[mouse].target_provenance``: ``target_source`` x ``n_bins``
is looked up in the explicit ``GEOMETRY`` table and an unrecognised combination
is a hard failure, not a guess. The provenance ``n_bins`` is cross-checked
against the actual array width.

THE NULL (say it in every axis label)
-------------------------------------
"predict-mean" here is the LEAVE-ONE-OUT marginal mean over the TEST trials:
trial i is predicted by the mean of the OTHER test targets. This is verbatim the
definition ``diagnostics/predict_mean_baseline.py`` (lines 116-125) settled on in
the 2026-07 audit, and the SAME definition is applied identically in both arms
and to BOTH skill axes (KL and projection) and to the overfitting-gap
denominator — so every number here is comparable with past project results.
It is NOT a train-half mean: the plain test-set mean fits the null's one free
parameter on the very trials it is scored against, understating the null loss by
O(1/n) and so making "worse than chance" claims too easy; the train-half mean is
not recoverable from the shards (they store test rows only). Skill < 1 beats the
strictest null. The shuffle-decoder reference (Dist.<arch>_shf) is recorded and
plotted as a secondary line, but predict-mean leads.

BOTH LOSSES, ALWAYS (CLAUDE.md house rule)
------------------------------------------
Every decoder is judged under KL skill AND projection-loss skill. They are not
redundant and are not expected to agree; where they disagree, that disagreement
is the finding, not a nuisance to be averaged away.

THE OVERFITTING GAP (renormalised)
----------------------------------
(val_fit - train_fit)/train_fit is dimensionless but its DENOMINATOR is the
absolute loss level, which is set by the target family (~2.8x different between
these two arms), so it manufactures a ~3x old->new drop at constant
generalisation. The headline gap is therefore the ABSOLUTE gap divided by the
same arm's LOO predict-mean loss under the cell's OWN training loss and OWN evar
weighting (flat for the pcaflat cells) — i.e. "the train->val degradation as a
fraction of what the trivial constant predictor costs on this target family".
Caveat, stated once: train/val fit losses live on the train half and its internal
val carve while the null lives on the test half, so the denominator is a scale
normaliser, not a matched-trial comparison. The train-relative form is kept in
the CSV as ``overfit_gap_over_train_within_arm_only``. Read at best_epoch (the
restored epoch, a 0-based index into history, MEASURED to equal
argmin(val_fit_loss) on 126/126 cell-arch histories across both arms), never at
the line end which sits ``patience`` epochs later. Missing history NaN-degrades.

THE PROJECTION BASIS (a trap)
-----------------------------
``pcs`` is bit-identical across all cells of an arm, but ``explained_var`` is
NOT: the ten ``pcaflat_*`` cells were trained with ``flat_evar=True`` and so store
the uniform 1/K vector, which makes their projection loss ~50x smaller in
absolute terms. Reading each cell's own ``explained_var`` would therefore compare
cells on different metrics. The projection SKILL axis PINS one evar per arm
(taken from a non-flat cell, i.e. the arm's true PCA spectrum) and scores all
cells of that arm on it, on the HELD-OUT TEST rows. For the same reason the saved
``*_pca_yardstick`` history is NOT used as the projection axis: it inherits each
cell's own weighting (flat for the 10 pcaflat cells) and only ever lives on the
train half and the internal val carve, never on the test half. (The cell's own
evar is used in exactly one place — the overfitting-gap denominator — because
there it must match the units of the cell's own train/val fit loss.)

Basis provenance: ``run_experiment.fit_pca_basis`` is called with
``training_posteriors = Y_train[...]`` (run_experiment.py line ~561), i.e. the
basis is fitted on the TRAIN half only and never sees the test rows scored here.
Its ``pca_basis='all_trials'`` default means "all TRAINING trials, unaveraged" —
not all trials of the dataset.

Wasserstein is deliberately absent: ``Wasserstein_calc_1D`` is a linear-cumsum
earth-mover distance and is origin-dependent (hence meaningless) on the 72-bin
circular support.

GUARDS (all fail loudly, none silently)
---------------------------------------
* the two arms must report DIFFERENT ``target_provenance.target_source``; pairing
  an arm against itself draws a perfect identity line and looks like a result.
  ``--self-check`` is the one explicit, labelled way to do that on purpose.
* an unrecognised (target_source, n_bins) combination aborts.
* the targets must be BIT-IDENTICAL across every cell of an arm (they are — that
  is what licenses building the calibration curves once per arm instead of 40
  times). Any drift aborts.
* decoded == target must invert to s_hat == 1.000 exactly, checked per arm at
  load time through this module's own code path.
* cells whose slug ``parse_cell`` cannot read are dropped BEFORE the plotters.
* the two arms' per-cell trial vectors (orientation / contrast / dispersion) must
  match exactly, as in the sister script ``io_hmm_decoded_comparison.py``;
  mismatched cells are dropped with a printed reason.

Handles a mid-flight export arm: it scores the intersection of the two arms and
prints, by name, which cells of the 40-cell grid are missing from each.

Outputs (PNG+SVG) under figures/io_hmm_vs_export/ plus a per-cell CSV:
  paired_scatter                 (a) new vs old, one panel per metric, identity line
  profile_<metric>               (b) metric vs cell, old and new lines, per arch
  family_summary                 (c) old -> new dumbbells per loss family x arch

Usage:
    python diagnostics/io_hmm_vs_export_scorecard.py
    python diagnostics/io_hmm_vs_export_scorecard.py --invariance   # single-arm proof
    python diagnostics/io_hmm_vs_export_scorecard.py --self-check   # pairing sanity
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.io import loadmat

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
import peakiness_style as ps                    # noqa: E402
import sharpening_calibration as sharp          # noqa: E402  (the peakiness axis)
from figsave import save_fig                    # noqa: E402
from cross_loss_eval import _eval_one           # noqa: E402  (loss maths == training)
from run_io_hmm_v1 import build_cells           # noqa: E402  (the expected grid)

ARCHS = (('spat', 'spatial'), ('temp', 'temporal'))
LOSS_ORDER = ('pca', 'pcaflat', 'kl', 'js')
MODEL_ORDER = ('h8', 'lin')
LAM_ORDER = ('lh0', 'lh1e-4', 'lh3e-4', 'lh1e-3', 'lh3e-3')

# the cell's OWN training loss — the units its train/val fit-loss curves live in,
# and hence the metric the overfitting-gap denominator must be computed under.
OWN_METRIC = {'pca': 'PCA', 'pcaflat': 'PCA', 'kl': 'KL', 'js': 'JS'}

FAMILY_COLOR = {'pca': ps.PCA_EVAR, 'pcaflat': ps.FLAT_EVAR,
                'kl': ps.KL, 'js': ps.JS}
FAMILY_LABEL = {'pca': 'projection (evar)', 'pcaflat': 'projection (flat)',
                'kl': 'KL', 'js': 'JS'}
ARCH_MARKER = {'spat': 'o', 'temp': '^'}

NULL_NAME = 'leave-one-out predict-mean (test trials)'

# Above this peak-vs-entropy discrepancy the decoder is reshaping rather than
# sharpening, and s_hat under-describes it. Flagged, never silently dropped.
AGREE_FLAG = 0.10


# ----------------------------------------------------------------------
# support geometry — READ FROM PROVENANCE, never inferred from the bin count
# ----------------------------------------------------------------------
class Geom:
    """One support: bin centres in degrees, bin width, and whether it wraps."""

    __slots__ = ('grid', 'delta', 'circular', 'label')

    def __init__(self, grid, delta, circular, label):
        self.grid = np.asarray(grid, float)
        self.delta = float(delta)
        self.circular = bool(circular)
        self.label = str(label)


# The ONLY recognised (target_source, n_bins) combinations. Anything else is a
# hard failure: silently defaulting an unknown grid to "circular [0,180)" is how
# a future 45-bin linear support would have been scored with the wrong geometry.
GEOMETRY = {
    ('export', 91): Geom(np.arange(91.0), 1.0, False,
                         'linear 0–90°, 91 bins × 1°'),
    ('io_hmm_pkl', 72): Geom(np.arange(72) * 2.5, 2.5, True,
                             'circular [0,180°), 72 bins × 2.5°'),
}


def geometry(target_source, n_bins, n_cols=None):
    """Look up the support for a saved posterior. Fails loudly on anything new."""
    key = (str(target_source), int(n_bins))
    if key not in GEOMETRY:
        raise SystemExit(
            f'\nABORT: unrecognised support (target_source={key[0]!r}, '
            f'n_bins={key[1]}).\n'
            '  The geometry of a posterior is read from '
            'results[mouse].target_provenance, never guessed from the bin count.\n'
            '  Known combinations: '
            + ', '.join(f'{s!r} x {n}' for s, n in sorted(GEOMETRY))
            + '\n  Add the new support to GEOMETRY (with its bin centres, bin '
              'width and wrap flag) before scoring it.')
    g = GEOMETRY[key]
    if n_cols is not None and int(n_cols) != int(n_bins):
        raise SystemExit(
            f'\nABORT: target_provenance says n_bins={n_bins} but the saved '
            f'posterior has {n_cols} columns ({key[0]!r}). The provenance and the '
            'data disagree; refusing to guess which is right.')
    return g


def coarsen_geom(g, factor):
    """The support you get by merging ``factor`` adjacent bins (for --invariance).

    Derived from the parent geometry, NOT re-looked-up by bin count — a merged
    grid is not one of the recognised supports and must never masquerade as one.
    """
    n = len(g.grid) // int(factor)
    delta = g.delta * int(factor)
    centres = np.arange(n) * delta + (int(factor) - 1) * g.delta / 2.0
    return Geom(centres, delta, g.circular, f'{g.label} merged ×{int(factor)}')


# ----------------------------------------------------------------------
# WITHIN-ARM-ONLY width statistics (all eight were rejected cross-arm)
# ----------------------------------------------------------------------
def posterior_sd(p, g):
    """Per-trial posterior SD in degrees on ONE arm's support. WITHIN-ARM ONLY —
    the linear-SD ratio missed matched true sharpening by 69% across arms and the
    Mardia circular-SD ratio by 26%. Never put this on a shared axis."""
    p = np.asarray(p, dtype=np.float64)
    s = p.sum(1, keepdims=True)
    p = np.divide(p, s, out=np.full_like(p, np.nan), where=s > 0)
    if not g.circular:
        mu = (p * g.grid).sum(1)
        return np.sqrt((p * (g.grid - mu[:, None]) ** 2).sum(1))
    z = (p * np.exp(2j * np.deg2rad(g.grid))).sum(1)
    R = np.clip(np.abs(z), 1e-12, 1.0 - 1e-15)
    return np.rad2deg(np.sqrt(-2.0 * np.log(R)) / 2.0)


def diff_entropy(p, g):
    """Per-trial differential entropy in nats: -sum p ln p + ln(delta).
    WITHIN-ARM ONLY — the differential-entropy DELTA, once thought to be the safe
    bin-width-invariant choice, missed matched true sharpening by 54%."""
    p = np.asarray(p, dtype=np.float64)
    s = p.sum(1, keepdims=True)
    p = np.divide(p, s, out=np.full_like(p, np.nan), where=s > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -(p * np.log(np.clip(p, 1e-300, None))).sum(1)
    return h + np.log(g.delta)


def _finite_mean(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else np.nan


def _finite_median(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


def within_arm_shape_metrics(dec, tgt, g):
    """The REJECTED width statistics, kept as within-arm CSV columns only.

    Every key here carries ``_within_arm_only`` in its name and the ground-truth
    cross-arm error that disqualified it. None of them reaches a figure or a
    paired comparison; the cross-arm peakiness axis is s_hat.
    """
    sd_d, sd_t = posterior_sd(dec, g), posterior_sd(tgt, g)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(sd_t > 0, sd_d / sd_t, np.nan)
    dH = diff_entropy(dec, g) - diff_entropy(tgt, g)
    return {
        # SD ratio: 69% cross-arm error (linear) / 26% (Mardia circular)
        'width_ratio_within_arm_only': _finite_mean(ratio),
        'width_ratio_med_within_arm_only': _finite_median(ratio),
        # differential-entropy delta: 54% cross-arm error
        'dH_nats_within_arm_only': _finite_mean(dH),
        'sd_dec_med_deg_within_arm_only': _finite_median(sd_d),
        'sd_tgt_med_deg_within_arm_only': _finite_median(sd_t),
        # max-prob ratio: 22% cross-arm error
        'peak_ratio_within_arm_only': float(dec.max(1).mean() / tgt.max(1).mean()),
        # raw Shannon entropy ratio: 15% cross-arm error (and an arm-specific
        # ceiling — a uniform decoder scores 1.021 new vs 1.116 old)
        'entropy_ratio_within_arm_only': float(
            _finite_mean(-(dec * np.log(np.clip(dec, 1e-300, None))).sum(1))
            / _finite_mean(-(tgt * np.log(np.clip(tgt, 1e-300, None))).sum(1))),
    }


# ----------------------------------------------------------------------
# the peakiness axis: equivalent sharpening factor
# ----------------------------------------------------------------------
def build_arm_curves(targets):
    """The per-arm calibration curves. Built ONCE per arm — the targets are
    bit-identical across every cell of an arm (asserted in ``score_arm``), so
    building them per cell would be ~40x wasted work for an identical answer."""
    t = np.asarray(targets, float)
    ok = np.isfinite(t).all(1)
    return {k: sharp.calibration_curve(t[ok], k) for k in sharp.BASE_STATS}


def sharpening_metrics(dec, tgt, curves):
    """s_hat + the two companions that must always travel with it."""
    r = sharp.equivalent_sharpening(dec, tgt, curves)
    return {'s_hat': r['s_hat'],
            's_hat_entropy': r['s_hat_entropy'],
            's_hat_agreement': r['agreement'],
            's_hat_clamped': float(bool(r['clamped']))}


# ----------------------------------------------------------------------
# metric registry.  (key, short name, axis label incl. the normalisation,
#                    reference value, log axis)
# EVERY key here goes on a SHARED old-vs-new axis, so every key here must be
# cross-arm licensed. The eight rejected width statistics are NOT in this list.
# ----------------------------------------------------------------------
METRICS = [
    ('kl_skill', 'KL skill',
     'KL skill\nmean KL(tgt||dec) ÷ mean KL(tgt||predict-mean)\n'
     f'null = {NULL_NAME}; <1 beats the strictest null', 1.0, False),
    ('proj_skill', 'projection skill',
     'projection-loss skill\nproj. loss ÷ proj. loss(predict-mean), arm-pinned '
     'evar weighting, held-out test rows\n'
     f'null = {NULL_NAME}; <1 beats the strictest null', 1.0, False),
    ('s_hat', 'equivalent sharpening ŝ',
     'equivalent sharpening ŝ (calibrated per arm on that arm’s OWN targets)\n'
     'ŝ = the s for which sharpen(target, s) = target^(s²), renormalised, '
     'reproduces the decoder’s peakiness\n'
     '1 = calibrated, >1 over-sharpened, <1 too broad. Ring = reshapes '
     f'(agreement > {AGREE_FLAG:.2f}); cross = clamped bound', 1.0, True),
    ('overfit_gap', 'overfitting gap',
     'overfitting gap at best epoch\n(val − train) fit-loss ÷ that arm’s '
     'predict-mean loss\n'
     f'same loss & evar as the cell trained on; null = {NULL_NAME}', 0.0, False),
]

# Diagnostics reported alongside, never as headline claims.
SECONDARY = [
    ('s_hat_agreement', 'ŝ agreement',
     'ŝ peak-vs-entropy discrepancy |ŝ_peak − ŝ_ent| ÷ max(ŝ)\n'
     f'>{AGREE_FLAG:.2f} = the decoder RESHAPES rather than sharpens, so ŝ '
     'under-describes it\n0 = the deviation really is sharpening-like',
     AGREE_FLAG, False),
    ('best_epoch', 'best epoch',
     'best epoch (0-based index into history; = argmin val fit loss)\n'
     'the restored epoch — raw epoch count, no normalisation applies',
     np.nan, False),
    ('early_stopped_epoch', 'early-stopped epoch',
     'early-stopped epoch (best epoch + patience, or the cap)\n'
     'raw epoch count, no normalisation applies', np.nan, False),
]

SKILL_KEYS = {'kl_skill': 'kl', 'proj_skill': 'proj'}     # metric -> loss tag


# ----------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------
def _loo_mean(tgt, ok):
    """Leave-one-out marginal-mean predictor for every row of ``tgt``.

    Verbatim in definition (not import — the collector there is hard-wired to the
    flat results layout) from ``diagnostics/predict_mean_baseline.py`` lines
    116-125: predicting trial i from the OTHER test trials only, so the null's
    single free parameter is never fitted on the trial it is scored against.
    This one definition serves BOTH skill axes and the overfitting-gap
    denominator, in BOTH arms.
    """
    n_ok = int(ok.sum())
    tot = tgt[ok].sum(axis=0)
    pm = np.tile((tot / n_ok)[None, :], (tgt.shape[0], 1))
    if n_ok > 1:
        pm[ok] = (tot[None, :] - tgt[ok]) / (n_ok - 1)
    return pm


def _hist_metrics(ckpt_path, arch):
    """best_epoch, early_stopped_epoch and the ABSOLUTE val-vs-train fit-loss gap
    READ AT best_epoch (the restored epoch) — never at the line end, which sits
    ``patience`` epochs later and measures the wrong thing (the 2026-08-04 fix in
    projflat_trainval_curves). ``best_epoch`` is a 0-based index into the history
    arrays and equals argmin(val_fit_loss); MEASURED on 126/126 cell-arch
    histories across both arms (max best_epoch 199 with 200 stored epochs).

    The caller divides the absolute gap by the arm's own predict-mean loss — see
    the module docstring for why the train-relative form is not the headline.

    Degrades gracefully: the ``lin`` cells run with weight_snapshot_every=0, so
    ``state_dicts``/``snapshot_epochs`` are empty there, but the loss curves,
    best_epoch and early_stopped_epoch are present for every cell. Missing
    history leaves every field NaN rather than silently scoring 0.
    """
    out = {'gap_abs': np.nan, 'train_fit_best': np.nan, 'val_fit_best': np.nan,
           'best_epoch': np.nan, 'early_stopped_epoch': np.nan,
           'best_epoch_is_argmin': np.nan,
           'censored': False, 'n_snapshots': 0}
    if not Path(ckpt_path).is_file():
        return out
    c = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    h = ((c.get(arch) or {}) if isinstance(c, dict) else {}).get('history') or {}
    t = np.asarray(h.get('train_fit_loss') or [], dtype=float)
    v = np.asarray(h.get('val_fit_loss') or [], dtype=float)
    out['n_snapshots'] = len(h.get('state_dicts') or [])
    be, es = h.get('best_epoch'), h.get('early_stopped_epoch')
    cap = h.get('epoch_cap')
    if be is not None:
        out['best_epoch'] = float(be)
    if es is not None:
        out['early_stopped_epoch'] = float(es)
    if be is not None and v.size:
        # the 0-based-index claim, re-verified on every shard rather than assumed
        out['best_epoch_is_argmin'] = float(int(be) == int(np.argmin(v)))
    if be is not None and cap is not None:
        # never early-stopped: the "how fast does it overfit" proxy is censored
        out['censored'] = bool(int(be) >= int(cap) - 1)
    if be is not None and t.size and v.size and 0 <= int(be) < min(t.size, v.size):
        b = int(be)
        if np.isfinite(t[b]) and np.isfinite(v[b]):
            out['train_fit_best'] = float(t[b])
            out['val_fit_best'] = float(v[b])
            out['gap_abs'] = float(v[b] - t[b])
    return out


def score_cell(cell_dir, mouse, pcs, evar, curves, ref_tgt, own_metric=None):
    """All metrics for one cell directory, per architecture.

    ``pcs``/``evar`` are the ARM-PINNED basis (see module docstring) — the cell's
    own ``explained_var`` is deliberately ignored on the projection SKILL axis so
    pcaflat cells land on the same projection metric as everything else. The
    cell's own evar IS used for the overfitting-gap denominator, where it must
    match the units of the cell's own train/val fit-loss curves.

    ``curves`` are the arm's calibration curves (built once); ``ref_tgt`` is the
    arm's reference target block, against which this cell's targets are checked
    for bit-identity — the assumption that licenses reusing ``curves``.
    """
    mat = cell_dir / 'stratified_balanced.mat'
    if not mat.is_file():
        return None
    m = loadmat(str(mat), squeeze_me=True, struct_as_record=False)
    r = getattr(m.get('results'), f'mouse_{mouse}', None)
    if r is None:
        return None
    ck = cell_dir / 'checkpoints' / f'mouse_{mouse}_stratified_balanced.pt'
    prov = getattr(r, 'target_provenance', None)
    if prov is None or not hasattr(prov, 'target_source') or not hasattr(prov, 'n_bins'):
        raise SystemExit(
            f'\nABORT: {mat} has no usable target_provenance (target_source / '
            'n_bins).\n  The support geometry is read from provenance and never '
            'guessed from the bin count; refusing to score this shard.')

    src, nb = str(prov.target_source), int(prov.n_bins)
    out = {'_n_bins': nb, '_target_source': src, '_pcs_drift': np.nan,
           '_tgt_drift': 0.0}
    own_pcs = np.asarray(getattr(r.Dist, 'pcs', np.empty(0)), float)
    own_evar = np.asarray(getattr(r.Dist, 'explained_var', np.empty(0)), float)
    if own_pcs.shape == np.asarray(pcs).shape:
        out['_pcs_drift'] = float(np.abs(own_pcs - np.asarray(pcs)).max())
    # trial fingerprint for the cross-arm pairing guard (as in the sister script
    # io_hmm_decoded_comparison.pair_cell)
    tr = getattr(r, 'trials', None)
    out['_trials'] = ({k: np.asarray(getattr(tr, k)).ravel().astype(float)
                       for k in ('orientation', 'contrast', 'dispersion')}
                      if tr is not None else None)

    for arch, _ in ARCHS:
        a = getattr(r.Dist, arch, None)
        if a is None:
            continue
        dec = np.asarray(a.decoded, float)
        tgt = np.asarray(a.target, float)
        g = geometry(src, nb, n_cols=tgt.shape[1])
        if ref_tgt is not None and ref_tgt.shape == tgt.shape:
            out['_tgt_drift'] = max(out['_tgt_drift'],
                                    float(np.nanmax(np.abs(tgt - ref_tgt))))
        ok = np.isfinite(tgt).all(1) & np.isfinite(dec).all(1)
        rec = {}
        if ok.any():
            pm = _loo_mean(tgt, ok)
            for tag, metric in (('kl', 'KL'), ('proj', 'PCA')):
                real = _eval_one(dec, tgt, metric, pcs, evar)
                null = _eval_one(pm, tgt, metric, pcs, evar)
                rec[f'{tag}_raw'] = real
                rec[f'{tag}_predmean'] = null
                rec[f'{tag}_skill'] = (real / null
                                       if np.isfinite(null) and null > 0 else np.nan)
            shf = getattr(r.Dist, f'{arch}_shf', None)
            for tag, metric in (('kl', 'KL'), ('proj', 'PCA')):
                s = np.nan
                if shf is not None:
                    s = _eval_one(np.asarray(shf.decoded, float),
                                  np.asarray(shf.target, float), metric, pcs, evar)
                null = rec.get(f'{tag}_predmean', np.nan)
                rec[f'{tag}_skill_shuffle'] = (s / null if np.isfinite(s)
                                               and np.isfinite(null) and null > 0
                                               else np.nan)
            # THE cross-arm peakiness axis
            rec.update(sharpening_metrics(dec[ok], tgt[ok], curves))
            # the eight rejected statistics, CSV only, within-arm only
            rec.update(within_arm_shape_metrics(dec[ok], tgt[ok], g))
            # gap denominator: the SAME leave-one-out predict-mean null, scored
            # under the cell's OWN training loss and OWN evar weighting, so it
            # carries the units of the train/val fit-loss curves.
            rec['own_predmean'] = (
                _eval_one(pm, tgt, own_metric, own_pcs, own_evar)
                if own_metric else np.nan)
        rec.update(_hist_metrics(ck, arch))
        nullv = rec.get('own_predmean', np.nan)
        gap = rec.get('gap_abs', np.nan)
        rec['overfit_gap'] = (gap / nullv if np.isfinite(gap) and np.isfinite(nullv)
                              and nullv > 0 else np.nan)
        tfb = rec.get('train_fit_best', np.nan)
        rec['overfit_gap_over_train_within_arm_only'] = (
            gap / tfb if np.isfinite(gap) and np.isfinite(tfb) and tfb > 0 else np.nan)
        out[arch] = rec
    return out


# ----------------------------------------------------------------------
# arm loading
# ----------------------------------------------------------------------
def _cell_dirs(root):
    """{cell_slug: cell_dir} for every finished cell under an arm root."""
    out = {}
    for shard in sorted(Path(root).glob('*/Q_*/stratified_balanced.mat')):
        out[shard.parent.parent.name] = shard.parent
    return out


def pin_basis(cell_dirs, mouse):
    """(pcs, evar) for the arm: the true PCA spectrum, taken from a NON-flat cell.

    Flat-evar cells store the uniform 1/K vector, so picking one of those would
    silently rescale the whole projection axis by ~50x. Detect by comparing the
    leading eigenvalue against uniform.
    """
    for cell in sorted(cell_dirs, key=lambda c: 0 if not c.startswith('pcaflat') else 1):
        mat = cell_dirs[cell] / 'stratified_balanced.mat'
        m = loadmat(str(mat), squeeze_me=True, struct_as_record=False)
        r = getattr(m.get('results'), f'mouse_{mouse}', None)
        if r is None:
            continue
        pcs = np.asarray(getattr(r.Dist, 'pcs', np.empty(0)), float)
        evar = np.asarray(getattr(r.Dist, 'explained_var', np.empty(0)), float)
        if pcs.size == 0 or evar.size == 0:
            continue
        if evar.max() <= 2.0 / evar.size:       # uniform => flat_evar cell
            continue
        return pcs, evar, cell
    return None, None, None


def arm_reference_target(cell_dirs, mouse):
    """(targets, target_source, n_bins) from ONE cell of the arm.

    Both architectures of a cell, and every cell of an arm, carry the same target
    block (measured: max |difference| = 0 across all 40 new and all available old
    cells, spat vs temp included). ``score_cell`` re-checks each cell against this
    reference and ``score_arm`` aborts on any drift.
    """
    for cell in sorted(cell_dirs):
        m = loadmat(str(cell_dirs[cell] / 'stratified_balanced.mat'),
                    squeeze_me=True, struct_as_record=False)
        r = getattr(m.get('results'), f'mouse_{mouse}', None)
        if r is None:
            continue
        prov = getattr(r, 'target_provenance', None)
        for arch, _ in ARCHS:
            a = getattr(r.Dist, arch, None)
            if a is None:
                continue
            t = np.asarray(a.target, float)
            if prov is None:
                raise SystemExit(f'ABORT: {cell} has no target_provenance.')
            return t, str(prov.target_source), int(prov.n_bins), cell
    return None, None, None, None


def score_arm(root, mouse, label):
    """({cell: metrics}, evar, info) for every finished cell of one arm.

    Builds the arm's sharpening calibration curves ONCE here (not per cell), and
    proves the two preconditions for doing so: the arm's targets are bit-identical
    across cells, and a decoder equal to its own target inverts to s_hat = 1.000
    through this module's own code path.
    """
    dirs = _cell_dirs(root)
    if not dirs:
        print(f"  {label:4s} {str(root):55s} 0 cells (arm not present)")
        return {}, None, {'sources': set(), 'n_bins': set(), 'root': str(root)}
    pcs, evar, src = pin_basis(dirs, mouse)
    if pcs is None:
        raise SystemExit(f'{label}: no cell with a non-flat PCA basis under {root}')

    ref_tgt, tgt_src, tgt_nb, ref_cell = arm_reference_target(dirs, mouse)
    geom = geometry(tgt_src, tgt_nb, n_cols=ref_tgt.shape[1])
    curves = build_arm_curves(ref_tgt)
    ident = sharpening_metrics(ref_tgt, ref_tgt, curves)
    if abs(ident['s_hat'] - 1.0) > 1e-9 or abs(ident['s_hat_entropy'] - 1.0) > 1e-9:
        raise SystemExit(
            f'\nABORT: {label} calibration is broken — a decoder identical to its '
            f'own target inverts to ŝ={ident["s_hat"]:.6f} '
            f'(entropy {ident["s_hat_entropy"]:.6f}), not 1.')

    scored, drift, tdrift = {}, 0.0, 0.0
    for cell, d in dirs.items():
        p = parse_cell(cell)
        s = score_cell(d, mouse, pcs, evar, curves, ref_tgt,
                       own_metric=OWN_METRIC.get(p[0]) if p else None)
        if s is not None:
            scored[cell] = s
            if np.isfinite(s['_pcs_drift']):
                drift = max(drift, s['_pcs_drift'])
            tdrift = max(tdrift, s['_tgt_drift'])
    nb = {s['_n_bins'] for s in scored.values()}
    ts = {s['_target_source'] for s in scored.values()}
    print(f"  {label:4s} {str(root):55s} {len(scored):2d} cells | "
          f"targets {sorted(ts)} | n_bins {sorted(nb)} | support {geom.label}")
    print(f"  {'':4s} evar pinned from {src} (lead {evar.max():.3f}) | "
          f"max |pcs − pinned| across cells {drift:.2e}")
    print(f"  {'':4s} ŝ calibration built ONCE from {ref_cell} "
          f"({ref_tgt.shape[0]} trials × {ref_tgt.shape[1]} bins); "
          f"max |target − reference| across all cells & archs {tdrift:.2e}; "
          f"identity check ŝ(target,target) = {ident['s_hat']:.6f}")
    if tdrift > 0:
        raise SystemExit(
            f'\nABORT: {label} targets are NOT identical across cells '
            f'(max |difference| = {tdrift:.3e}).\n'
            '  One calibration curve per arm is only valid if the arm has ONE '
            'target family; build per cell, or find out why the targets moved.')
    if drift > 1e-9:
        print(f"  {'':4s} WARNING: the PCA basis is NOT constant across this arm's "
              f"cells (drift {drift:.2e}); the common projection axis is only "
              f"approximate.")
    return scored, evar, {'sources': ts, 'n_bins': nb, 'root': str(root),
                          'geom': geom, 'curves': curves, 'ref_tgt': ref_tgt}


# ----------------------------------------------------------------------
# grid bookkeeping
# ----------------------------------------------------------------------
def parse_cell(name):
    """('pcaflat', 'lin', 'lh3e-4') or None."""
    parts = name.split('_')
    if len(parts) != 3:
        return None
    loss, model, lam = parts
    if loss in LOSS_ORDER and model in MODEL_ORDER and lam in LAM_ORDER:
        return loss, model, lam
    return None


def sort_key(name):
    p = parse_cell(name)
    if p is None:
        return (99, 99, 99, name)
    return (LOSS_ORDER.index(p[0]), MODEL_ORDER.index(p[1]),
            LAM_ORDER.index(p[2]), name)


def short_label(name):
    p = parse_cell(name)
    return name if p is None else f"{p[1]}  λ{p[2][2:]}"


def get(scored, cell, arch, key):
    return scored.get(cell, {}).get(arch, {}).get(key, np.nan)


def _flag(scored, cell, arch, key):
    v = get(scored, cell, arch, key)
    return bool(np.isfinite(v) and v > 0)


def _log_ticks(ax, lim, which='xy'):
    """Plain decimal ticks on a log axis. Matplotlib's default log locator labels
    only powers of ten, which on a ŝ axis spanning 0.25–5 leaves a single '10^0'
    and an unreadable panel."""
    cand = np.array([0.25, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    t = cand[(cand >= lim[0]) & (cand <= lim[1])]
    if t.size < 2:
        return
    lab = [f'{v:g}' for v in t]
    if 'x' in which:
        ax.set_xticks(t); ax.set_xticklabels(lab)
        ax.xaxis.set_minor_locator(plt.NullLocator())
    if 'y' in which:
        ax.set_yticks(t); ax.set_yticklabels(lab)
        ax.yaxis.set_minor_locator(plt.NullLocator())


def wrap_label(axlab, width):
    """Re-wrap an axis label to ``width`` characters per line, keeping its own
    line breaks. The normalisation statements here are long by design (house
    rule: every normalisation lives in the axis label), and an unwrapped 74-char
    line overruns a rotated y-axis or a 3-inch panel title."""
    out = []
    for line in axlab.split('\n'):
        out.extend(textwrap.wrap(line, width, break_long_words=False,
                                 break_on_hyphens=False) or [''])
    return '\n'.join(out)


# ----------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------
def guard_distinct_arms(old_info, new_info, old_root, new_root):
    """Abort unless the two arms really are two DIFFERENT target families.

    Self-pairing (io_hmm vs io_hmm) silently produces a perfect identity-line
    figure set that reads exactly like "the target family changed nothing".
    """
    if Path(old_root).resolve() == Path(new_root).resolve():
        raise SystemExit(
            '\nABORT: --old-root and --new-root are the SAME directory '
            f'({Path(old_root).resolve()}).\n'
            '  Every point would land on the identity line by construction. '
            'Pass --self-check if you meant to validate the pairing logic.')
    shared = old_info['sources'] & new_info['sources']
    if shared:
        raise SystemExit(
            '\nABORT: both arms report target_source '
            f'{sorted(shared)} — this is NOT a cross-family comparison.\n'
            f'  OLD {old_info["root"]}: sources {sorted(old_info["sources"])}, '
            f'n_bins {sorted(old_info["n_bins"])}\n'
            f'  NEW {new_info["root"]}: sources {sorted(new_info["sources"])}, '
            f'n_bins {sorted(new_info["n_bins"])}\n'
            '  The figures would show a perfect identity line and be read as '
            '"the target family changed nothing".\n'
            '  Point --old-root at the export-target arm '
            '(results/io_hmm_v1_exportref), or pass --self-check to validate the '
            'pairing/plotting logic on purpose.')


def guard_pairing(old, new, cells):
    """Drop unparseable slugs and cells whose two arms disagree on the test split.

    Returns (kept_cells, dropped) with dropped = [(cell, reason)].
    """
    kept, dropped = [], []
    for cell in cells:
        if parse_cell(cell) is None:
            dropped.append((cell, 'slug does not parse as <loss>_<arch>_<lambda>'))
            continue
        to, tn = old[cell].get('_trials'), new[cell].get('_trials')
        if to is None or tn is None:
            dropped.append((cell, 'no trials struct in one arm (cannot verify '
                                  'the test split matches)'))
            continue
        bad = [k for k in to if not np.array_equal(to[k], tn[k])]
        if bad:
            dropped.append((cell, f'test-split mismatch between arms ({", ".join(bad)})'))
            continue
        kept.append(cell)
    return kept, dropped


# ----------------------------------------------------------------------
# (a) paired scatters
# ----------------------------------------------------------------------
def fig_paired_scatter(old, new, cells, out_dir, tag, arm_labels, mouse=None):
    panels = METRICS + SECONDARY
    fig, axes = plt.subplots(3, 3, figsize=ps.figsize(3, 3))
    axes = axes.ravel()
    for i, (key, short, axlab, ref, logsc) in enumerate(panels):
        ax = axes[i]
        xs, ys = [], []
        for cell in cells:
            fam, model, _lam = parse_cell(cell)
            for arch, _ in ARCHS:
                x, y = get(old, cell, arch, key), get(new, cell, arch, key)
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                xs.append(x); ys.append(y)
                if key == 's_hat':
                    # the two companions s_hat must never be read without
                    if (_flag(old, cell, arch, 's_hat_clamped')
                            or _flag(new, cell, arch, 's_hat_clamped')):
                        ax.plot(x, y, marker='x', ms=8, ls='none', color='k',
                                mew=1.3, zorder=5)
                    ag = np.nanmax([get(old, cell, arch, 's_hat_agreement'),
                                    get(new, cell, arch, 's_hat_agreement')])
                    if np.isfinite(ag) and ag > AGREE_FLAG:
                        ax.plot(x, y, marker='o', ms=11, ls='none', mfc='none',
                                mec='0.2', mew=0.9, zorder=1)
                ax.plot(x, y, marker=ARCH_MARKER[arch], ms=5.5, ls='none',
                        mec=FAMILY_COLOR[fam], mew=1.2,
                        mfc=FAMILY_COLOR[fam] if model == 'h8' else 'none',
                        alpha=0.9, zorder=3)
        if xs:
            if logsc:
                lo, hi = min(min(xs), min(ys)), max(max(xs), max(ys))
                lim = (max(lo / 1.25, 1e-6), hi * 1.25)
                ax.set_xscale('log'); ax.set_yscale('log')
                _log_ticks(ax, lim)
            else:
                lo = min(min(xs), min(ys)); hi = max(max(xs), max(ys))
                pad = 0.06 * (hi - lo or 1.0)
                lim = (lo - pad, hi + pad)
            ax.plot(lim, lim, ls='--', lw=1.0, color='0.45', zorder=0)
            ax.set_xlim(lim); ax.set_ylim(lim)
            if np.isfinite(ref) and lim[0] < ref < lim[1]:
                ax.axhline(ref, ls=':', lw=0.8, color=ps.CHANCE_GREY, zorder=0)
                ax.axvline(ref, ls=':', lw=0.8, color=ps.CHANCE_GREY, zorder=0)
        # the full normalisation goes in the panel title rather than being
        # repeated verbatim on both axes (it is identical on x and y — this is a
        # paired plot — and twice over it is unreadable at this panel size).
        ax.set_xlabel(f'OLD: {arm_labels[0]}', fontsize=7.5)
        ax.set_ylabel(f'NEW: {arm_labels[1]}', fontsize=7.5)
        ax.set_title(wrap_label(axlab, 56), fontsize=6.4)
        ax.tick_params(labelsize=7.5)

    lax = axes[len(panels)]
    lax.axis('off')
    # only families actually paired: a legend entry with no points on the axes
    # sends the reader hunting for a family the export arm hasn't finished yet.
    present = [f for f in LOSS_ORDER if any(parse_cell(c)[0] == f for c in cells)]
    handles = [Line2D([0], [0], marker='s', ls='none', color=FAMILY_COLOR[f],
                      label=FAMILY_LABEL[f], ms=6) for f in present]
    handles += [Line2D([0], [0], marker=ARCH_MARKER[a], ls='none', color='0.3',
                       label=f'{lab} decoder', ms=6) for a, lab in ARCHS]
    handles += [Line2D([0], [0], marker='D', ls='none', mec='0.3', mfc='0.3',
                       label='filled = hidden [8]', ms=5),
                Line2D([0], [0], marker='D', ls='none', mec='0.3', mfc='none',
                       label='open = linear (no hidden)', ms=5),
                Line2D([0], [0], marker='o', ls='none', mfc='none', mec='0.2',
                       label=f'ring = ŝ agreement > {AGREE_FLAG:.2f}\n'
                             '(reshapes; ŝ under-describes)', ms=9),
                Line2D([0], [0], marker='x', ls='none', color='k',
                       label='cross = ŝ clamped at a ladder end\n(a bound, not an '
                             'estimate)', ms=7),
                Line2D([0], [0], ls='--', color='0.45', label='identity (no change)'),
                Line2D([0], [0], ls=':', color=ps.CHANCE_GREY,
                       label='calibrated / null reference')]
    lax.legend(handles=handles, loc='center', fontsize=7.5, ncol=1, frameon=False)
    for j in range(len(panels) + 1, len(axes)):
        axes[j].axis('off')

    ps.label_panels(axes[:len(panels)])
    fig.suptitle(
        f'Paired old-vs-new decoder behaviour, mouse {mouse} — one point per '
        f'(cell × decoder); {len(cells)} cells paired. Off the dashed identity '
        f'line = the target family changed that metric.\n'
        f'Peakiness is the calibrated equivalent sharpening ŝ; the eight fixed '
        f'width statistics (max-prob, entropy, SD, ΔH, …) were rejected against '
        f'ground truth and are within-arm CSV columns only.',
        fontsize=9)
    # max_px 1560, not the 1600 default: bbox_inches='tight' grows the saved
    # raster ~1% past the nominal figure size, which would put the PNG over
    # the house 1600 px ceiling.
    save_fig(fig, out_dir, f'paired_scatter{tag}', max_px=1560)


# ----------------------------------------------------------------------
# (b) per-metric cell profiles
# ----------------------------------------------------------------------
def fig_profiles(old, new, cells, out_dir, tag, arm_labels, shuffle=True, mouse=None):
    x = np.arange(len(cells))
    fams = [parse_cell(c)[0] for c in cells]
    for key, short, axlab, ref, logsc in METRICS + SECONDARY:
        fig, axes = plt.subplots(2, 1, figsize=(12.0, 6.8), sharex=True)
        for ri, (arch, alab) in enumerate(ARCHS):
            ax = axes[ri]
            # faint family bands, so the grouping reads without a busy top strip
            for f in LOSS_ORDER:
                idx = [i for i, ff in enumerate(fams) if ff == f]
                if idx:
                    ax.axvspan(min(idx) - 0.5, max(idx) + 0.5,
                               color=FAMILY_COLOR[f], alpha=0.055, lw=0, zorder=0)
            yo = np.array([get(old, c, arch, key) for c in cells], float)
            yn = np.array([get(new, c, arch, key) for c in cells], float)
            ax.plot(x, yo, 's--', ms=4, lw=1.1, color='0.35',
                    label=f'OLD — {arm_labels[0]}')
            ax.plot(x, yn, 'o-', ms=4.5, lw=1.4, color=ps.SHAPE,
                    label=f'NEW — {arm_labels[1]}')
            if key == 's_hat':
                ax.set_yscale('log')
                _log_ticks(ax, (np.nanmin([yo, yn]) / 1.25,
                                np.nanmax([yo, yn]) * 1.25), which='y')
                for arm, ys, mk in (('old', yo, 's'), ('new', yn, 'o')):
                    sc = old if arm == 'old' else new
                    cl = [i for i, c in enumerate(cells)
                          if _flag(sc, c, arch, 's_hat_clamped')]
                    ag = [i for i, c in enumerate(cells)
                          if get(sc, c, arch, 's_hat_agreement') > AGREE_FLAG]
                    if cl:
                        ax.plot(x[cl], ys[cl], 'x', ms=7, color='k', mew=1.2,
                                ls='none', zorder=5,
                                label='ŝ clamped (a bound)' if arm == 'old' else None)
                    if ag:
                        ax.plot(x[ag], ys[ag], 'o', ms=10, mfc='none', mec='0.2',
                                mew=0.9, ls='none', zorder=4,
                                label=f'ŝ agreement > {AGREE_FLAG:.2f} (reshapes)'
                                      if arm == 'old' else None)
            if shuffle and key in SKILL_KEYS:
                sk = f'{SKILL_KEYS[key]}_skill_shuffle'
                so = np.array([get(old, c, arch, sk) for c in cells], float)
                sn = np.array([get(new, c, arch, sk) for c in cells], float)
                ax.plot(x, so, '^:', ms=3, lw=0.8, color='0.65',
                        label='OLD shuffle decoder (2nd null)')
                ax.plot(x, sn, 'v:', ms=3, lw=0.8, color='#9ecae1',
                        label='NEW shuffle decoder (2nd null)')
            if np.isfinite(ref):
                ax.axhline(ref, ls=':', lw=1.0, color=ps.CHANCE_GREY)
            lo, hi = ax.get_ylim()
            # headroom for the family band labels, in the axis's own scale
            ax.set_ylim(lo, hi * 1.35 if logsc else hi + 0.16 * (hi - lo))
            if ri == 0:
                for f in LOSS_ORDER:
                    idx = [i for i, ff in enumerate(fams) if ff == f]
                    if idx:
                        ax.text(np.mean(idx), 0.985, FAMILY_LABEL[f],
                                transform=ax.get_xaxis_transform(), ha='center',
                                va='top', fontsize=9, color=FAMILY_COLOR[f])
            # short y-label: this figure plots ONE metric, so the full
            # normalisation lives in the suptitle, which IS this figure's axis
            # statement (a 4-line rotated label eats a third of the panel).
            ax.set_ylabel(f'{alab} decoder\n{short}', fontsize=8)
            ax.grid(axis='y', alpha=0.2, lw=0.5)
            handles = ax.get_legend_handles_labels()
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels([short_label(c) for c in cells], rotation=70,
                                 fontsize=6.5, ha='right')
        axes[-1].set_xlabel(
            'cell (grouped by loss family; hidden size and entropy penalty λ_H '
            'within group)\nλ_H acts on the TEMPORAL decoder only, so the five '
            'λ_H cells of a SPATIAL row are exact replicates, not 5 samples',
            fontsize=8.5)
        try:
            fig.legend(*handles, fontsize=7.5, ncol=4, frameon=False,
                       loc='outside lower center')
        except (ValueError, AttributeError):
            axes[-1].legend(*handles, fontsize=7, ncol=3, loc='best')
        # the suptitle carries the full normalisation for this one-metric figure;
        # drop axlab's first line when it merely repeats the short name
        norm = axlab.split('\n')
        if norm and norm[0].startswith(short):
            norm = norm[1:] or axlab.split('\n')
        fig.suptitle(f'{short} per cell, old vs new targets — mouse {mouse}\n'
                     + wrap_label('\n'.join(norm), 120), fontsize=9)
        save_fig(fig, out_dir, f'profile_{key}{tag}', max_px=1560)


# ----------------------------------------------------------------------
# (c) loss-family summary
# ----------------------------------------------------------------------
def fig_family_summary(old, new, cells, out_dir, tag, arm_labels, mouse=None):
    present = [f for f in LOSS_ORDER if any(parse_cell(c)[0] == f for c in cells)]
    rows = [(f, a, alab) for f in present for a, alab in ARCHS]
    ypos = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, len(METRICS),
                             figsize=(3.0 * len(METRICS) + 1.4, 5.6), sharey=True)
    for ci, (key, short, axlab, ref, logsc) in enumerate(METRICS):
        ax = axes[ci]
        for yi, (fam, arch, _alab) in zip(ypos, rows):
            cs = [c for c in cells if parse_cell(c)[0] == fam]
            vo = np.array([get(old, c, arch, key) for c in cs], float)
            vn = np.array([get(new, c, arch, key) for c in cs], float)
            ax.plot(vo, np.full(vo.size, yi), 's', ms=2.5, color='0.75',
                    alpha=0.8, zorder=1)
            ax.plot(vn, np.full(vn.size, yi), 'o', ms=2.5,
                    color=FAMILY_COLOR[fam], alpha=0.35, zorder=1)
            if np.isfinite(vo).any() and np.isfinite(vn).any():
                mo, mn = np.nanmedian(vo), np.nanmedian(vn)
                ax.plot([mo, mn], [yi, yi], '-', lw=1.6,
                        color=FAMILY_COLOR[fam], zorder=2)
                ax.plot(mo, yi, 's', ms=6, mfc='w',
                        mec=FAMILY_COLOR[fam], mew=1.5, zorder=3)
                ax.plot(mn, yi, 'o', ms=6.5, color=FAMILY_COLOR[fam], zorder=3)
        if logsc:
            ax.set_xscale('log')
        if np.isfinite(ref):
            ax.axvline(ref, ls=':', lw=1.0, color=ps.CHANCE_GREY)
        for yi in ypos[1::2]:
            ax.axhspan(yi - 0.5, yi + 0.5, color='0.95', zorder=0)
        if logsc:
            _log_ticks(ax, ax.get_xlim(), which='x')
        ax.set_xlabel(wrap_label(axlab, 54), fontsize=6.4)
        ax.set_title(short, fontsize=9.5)
        ax.tick_params(labelsize=7.5)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([f'{FAMILY_LABEL[f]}\n{alab}' for f, _a, alab in rows],
                            fontsize=7.5)
    axes[0].set_ylim(ypos.min() - 0.6, ypos.max() + 0.6)
    handles = [Line2D([0], [0], marker='s', ls='none', mfc='w', mec='0.3',
                      label=f'OLD median — {arm_labels[0]}', ms=6),
               Line2D([0], [0], marker='o', ls='none', color='0.3',
                      label=f'NEW median — {arm_labels[1]}', ms=6),
               Line2D([0], [0], marker='.', ls='none', color='0.6',
                      label='individual cells (hidden size x λ_H)', ms=6)]
    try:
        fig.legend(handles=handles, fontsize=8, ncol=3,
                   loc='outside lower center', frameon=False)
    except (ValueError, AttributeError):     # older mpl: no 'outside' locations
        axes[-1].legend(handles=handles, fontsize=7, loc='best')
    fig.suptitle('What changed when the ideal observer was swapped — median over '
                 f'the cells of each loss family (hidden size × 5 λ_H), mouse {mouse}\n'
                 'λ_H acts on the TEMPORAL decoder only: a SPATIAL row\'s five λ_H '
                 'cells are exact replicates, so its median is over 2 distinct '
                 'configurations (h8, lin), not 10',
                 fontsize=9)
    save_fig(fig, out_dir, f'family_summary{tag}', max_px=1560)


# ----------------------------------------------------------------------
def write_csv(old, new, cells, out_dir, tag):
    keys = ([k for k, _s, _l, _r, _g in METRICS + SECONDARY]
            + ['s_hat_entropy', 's_hat_clamped',
               # ---- the eight rejected cross-arm statistics: WITHIN-ARM ONLY ----
               'peak_ratio_within_arm_only',           # 22% cross-arm error
               'entropy_ratio_within_arm_only',        # 15%
               'width_ratio_within_arm_only',          # 69% linear / 26% circular
               'width_ratio_med_within_arm_only',
               'dH_nats_within_arm_only',              # 54%
               'sd_dec_med_deg_within_arm_only', 'sd_tgt_med_deg_within_arm_only',
               'overfit_gap_over_train_within_arm_only',
               # ---------------------------------------------------------------
               'kl_raw', 'kl_predmean', 'kl_skill_shuffle',
               'proj_raw', 'proj_predmean', 'proj_skill_shuffle',
               'own_predmean', 'gap_abs', 'train_fit_best', 'val_fit_best',
               'best_epoch_is_argmin', 'censored', 'n_snapshots'])
    lines = ['cell,loss_family,hidden,lambda_H,arch,arm,' + ','.join(keys)]
    for cell in cells:
        fam, model, lam = parse_cell(cell)
        for arch, _ in ARCHS:
            for arm, scored in (('old', old), ('new', new)):
                vals = [scored.get(cell, {}).get(arch, {}).get(k, np.nan)
                        for k in keys]
                txt = ','.join('' if (isinstance(v, float) and not np.isfinite(v))
                               else (f'{v:.6g}' if isinstance(v, float) else str(v))
                               for v in vals)
                lines.append(f'{cell},{fam},{model},{lam[2:]},{arch},{arm},{txt}')
    path = Path(out_dir) / f'io_hmm_vs_export_cells{tag}.csv'
    path.write_text('\n'.join(lines) + '\n')
    print(f'  -> {path.name}')


def print_summary(old, new, cells, missing_old, missing_new, extra):
    print(f'\npaired cells: {len(cells)} of 40 expected')
    if missing_old:
        print(f'  missing from OLD arm ({len(missing_old)}): '
              + ' '.join(sorted(missing_old, key=sort_key)))
    if missing_new:
        print(f'  missing from NEW arm ({len(missing_new)}): '
              + ' '.join(sorted(missing_new, key=sort_key)))
    if extra:
        print(f'  present but off-grid ({len(extra)}): ' + ' '.join(sorted(extra)))
    cens = [(c, a) for c in cells for a, _ in ARCHS
            for s in (old, new) if s.get(c, {}).get(a, {}).get('censored')]
    if cens:
        print(f'  never early-stopped (best_epoch == cap; the "how fast does it '
              f'overfit" proxy is censored there): {len(cens)} cell-arch-arm cases')
    # the 0-based-index claim, re-verified rather than asserted in prose
    chk = [get(s, c, a, 'best_epoch_is_argmin') for c in cells for a, _ in ARCHS
           for s in (old, new)]
    chk = [v for v in chk if np.isfinite(v)]
    if chk:
        print(f'  best_epoch == argmin(val_fit_loss): {int(sum(chk))}/{len(chk)} '
              f'cell-arch-arm histories (0-based index, verified here)')
    n_re = sum(get(s, c, a, 's_hat_agreement') > AGREE_FLAG
               for c in cells for a, _ in ARCHS for s in (old, new))
    n_cl = sum(_flag(s, c, a, 's_hat_clamped')
               for c in cells for a, _ in ARCHS for s in (old, new))
    print(f'  ŝ caveats: {n_re} cell-arch-arm cases reshape (agreement > '
          f'{AGREE_FLAG:.2f}, ŝ under-describes them); {n_cl} clamped at a ladder '
          f'end (bound, not estimate)')

    # pseudoreplication check, MEASURED not assumed: λ_H is documented to touch
    # the temporal decoder only, so a spatial row's five λ_H cells should be
    # exact replicates. Count the distinct KL-skill values per group and say so.
    print('\ndistinct KL-skill values within each (family × hidden × arch × arm) '
          'group of 5 λ_H cells\n  (1 = the five λ_H cells are exact replicates, '
          'so they are NOT 5 independent samples):')
    for arch, alab in ARCHS:
        n_uni = []
        for fam in LOSS_ORDER:
            for mod in MODEL_ORDER:
                cs = [c for c in cells if parse_cell(c)[:2] == (fam, mod)]
                for scored in (old, new):
                    v = [get(scored, c, arch, 'kl_skill') for c in cs]
                    v = [x for x in v if np.isfinite(x)]
                    if len(v) >= 2:
                        n_uni.append(len(set(np.round(v, 9))))
        if n_uni:
            print(f'  {alab:9s} distinct values per group: '
                  f'median {int(np.median(n_uni))}, range '
                  f'{min(n_uni)}–{max(n_uni)} over {len(n_uni)} groups')

    print('\nper-loss-family medians (over hidden size x 5 λ_H)')
    hdr = (f"{'family':20s}{'arch':9s}"
           + ''.join(f'{s:>19s}' for _k, s, _l, _r, _g in METRICS))
    print(hdr)
    print('-' * len(hdr))
    for fam in LOSS_ORDER:
        cs = [c for c in cells if parse_cell(c)[0] == fam]
        if not cs:
            continue
        for arch, alab in ARCHS:
            row = f'{FAMILY_LABEL[fam]:20s}{alab:9s}'
            for key, _s, _l, _r, _g in METRICS:
                vo = np.array([get(old, c, arch, key) for c in cs], float)
                vn = np.array([get(new, c, arch, key) for c in cs], float)
                mo = np.nanmedian(vo) if np.isfinite(vo).any() else np.nan
                mn = np.nanmedian(vn) if np.isfinite(vn).any() else np.nan
                row += f'{mo:8.3f}->{mn:<9.3f}'
            print(row)
    print(f'\n(old -> new; skills are ÷ the {NULL_NAME}, <1 beats the strictest '
          'null; ŝ = 1 is calibrated to that arm\'s own target family, >1 '
          'over-sharpened; gap read at best_epoch and divided by the same arm\'s '
          'predict-mean loss.')
    print(' No fixed width statistic is on this table: max-prob ratio, Shannon '
          'entropy ratio, SD ratio (linear and circular), axial dispersion, ΔH, '
          'resultant ratio and perplexity width were all rejected against ground '
          'truth (15-69% cross-arm error at matched true sharpening). They survive '
          'in the CSV as *_within_arm_only.)')


# ----------------------------------------------------------------------
# SINGLE-ARM bin-width demonstration
# ----------------------------------------------------------------------
def _coarsen(p, factor):
    """Merge ``factor`` adjacent bins (probability mass adds)."""
    p = np.asarray(p, float)
    n = p.shape[1] // factor
    return p[:, :n * factor].reshape(len(p), n, factor).sum(2)


def invariance_check(root, mouse, cells=None, factor=2):
    """SINGLE-ARM coarse-graining table: re-grid one arm's own posteriors and see
    which statistics move. Nothing here is compared across arms.

    ŝ is rebuilt from the coarsened targets (the calibration is part of the
    estimator, so it must be re-derived on the new grid) and should barely move;
    the rejected width statistics are shown moving, which is the point.
    """
    dirs = _cell_dirs(root)
    if not dirs:
        raise SystemExit(f'invariance check: no finished cells under {root}')
    if not cells:
        # span peaky -> broad: the projection losses over-sharpen, KL/JS do not
        cells = [c for c in ('pca_h8_lh0', 'pca_lin_lh0', 'pcaflat_h8_lh0',
                             'kl_h8_lh0', 'js_h8_lh0', 'kl_lin_lh3e-3')
                 if c in dirs]
    ref_tgt, tsrc, tnb, ref_cell = arm_reference_target(dirs, mouse)
    g = geometry(tsrc, tnb, n_cols=ref_tgt.shape[1])
    gc = coarsen_geom(g, factor)
    curves_fine = build_arm_curves(ref_tgt)
    curves_coarse = build_arm_curves(_coarsen(ref_tgt, factor))
    print(f'\nSINGLE-ARM BIN-WIDTH CHECK — {root} (mouse {mouse}), targets from '
          f'{ref_cell}.\n  {g.label}  vs  {gc.label}. No cross-arm quantity '
          'appears in this table.')
    print('  ŝ is re-calibrated on the coarsened targets; the *_within_arm_only '
          'columns are the rejected statistics, shown moving.')
    hdr = (f"{'cell':16s}{'arch':6s}{'bins':>10s} "
           f"{'ŝ (calibrated)':>17s} {'Δ%':>6s} {'agree':>6s} | "
           f"{'width ratio':>13s} {'Δ%':>6s} | {'ΔH nats':>13s} {'Δ':>6s} | "
           f"{'peak ratio':>11s} {'Δ%':>6s}")
    print(hdr)
    print('-' * len(hdr))
    worst_s, worst_w, worst_p, worst_s_ok = 0.0, 0.0, 0.0, 0.0
    for cell in cells:
        m = loadmat(str(dirs[cell] / 'stratified_balanced.mat'),
                    squeeze_me=True, struct_as_record=False)
        r = getattr(m.get('results'), f'mouse_{mouse}', None)
        if r is None:
            continue
        for arch, _ in ARCHS:
            a = getattr(r.Dist, arch, None)
            if a is None:
                continue
            dec = np.asarray(a.decoded, float)
            tgt = np.asarray(a.target, float)
            ok = np.isfinite(dec).all(1) & np.isfinite(tgt).all(1)
            dec, tgt = dec[ok], tgt[ok]
            cd, ct = _coarsen(dec, factor), _coarsen(tgt, factor)
            mf = sharpening_metrics(dec, tgt, curves_fine)
            sf, ag = mf['s_hat'], mf['s_hat_agreement']
            scr = sharpening_metrics(cd, ct, curves_coarse)['s_hat']
            fine = within_arm_shape_metrics(dec, tgt, g)
            coarse = within_arm_shape_metrics(cd, ct, gc)
            ds = 100.0 * (scr / sf - 1.0)
            dw = 100.0 * (coarse['width_ratio_within_arm_only']
                          / fine['width_ratio_within_arm_only'] - 1.0)
            dp = 100.0 * (coarse['peak_ratio_within_arm_only']
                          / fine['peak_ratio_within_arm_only'] - 1.0)
            dh = (coarse['dH_nats_within_arm_only']
                  - fine['dH_nats_within_arm_only'])
            worst_s = max(worst_s, abs(ds))
            if ag <= AGREE_FLAG:
                worst_s_ok = max(worst_s_ok, abs(ds))
            worst_w = max(worst_w, abs(dw))
            worst_p = max(worst_p, abs(dp))
            print(f'{cell:16s}{arch:6s}{dec.shape[1]:4d}->{cd.shape[1]:<4d} '
                  f'{sf:7.3f}->{scr:<8.3f}{ds:+6.1f} {ag:6.2f} | '
                  f'{fine["width_ratio_within_arm_only"]:6.3f}->'
                  f'{coarse["width_ratio_within_arm_only"]:<6.3f}{dw:+6.1f} | '
                  f'{fine["dH_nats_within_arm_only"]:+6.2f}->'
                  f'{coarse["dH_nats_within_arm_only"]:<+6.2f}{dh:+6.2f} | '
                  f'{fine["peak_ratio_within_arm_only"]:5.2f}->'
                  f'{coarse["peak_ratio_within_arm_only"]:<5.2f}{dp:+6.1f}')
    print(f'\nworst |Δ| across the table: ŝ {worst_s:.1f}% overall, but '
          f'{worst_s_ok:.1f}% among the cells whose agreement ≤ {AGREE_FLAG:.2f}; '
          f'width ratio {worst_w:.1f}%, peak ratio {worst_p:.1f}%.')
    print('READ THIS WITH THE AGREEMENT COLUMN. ŝ is regridding-stable exactly '
          'where its own companion says it is valid: every cell that moves more '
          'than a couple of percent is a cell already flagged as RESHAPING '
          f'(agreement > {AGREE_FLAG:.2f}), where ŝ is a summary of a deviation '
          'that is not a sharpening at all. That is the caveat the module states, '
          'measured here rather than assumed. The width-ratio and peak-ratio '
          'columns are the rejected statistics, shown for contrast — the '
          'peak-ratio column is the hazard being demonstrated, not a bug.')
    return worst_s, worst_w, worst_p


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--old-root', default='results/io_hmm_v1_exportref',
                    help='export-target reference arm (91 linear bins)')
    ap.add_argument('--new-root', default='results/io_hmm_v1',
                    help='ideal-observer HMM arm (72 circular bins)')
    ap.add_argument('--mouse', type=int, default=0)
    ap.add_argument('--out-root', default='figures/io_hmm_vs_export')
    ap.add_argument('--self-check', action='store_true',
                    help='pair the NEW arm against itself: every point must land '
                         'exactly on the identity line. Proves the pairing and '
                         'plotting logic, and is the ONLY way past the '
                         'different-target-family guard.')
    ap.add_argument('--invariance', action='store_true',
                    help='print the SINGLE-ARM bin-width table for the new arm '
                         '(native 72 bins vs a 2x bin merge) and exit')
    ap.add_argument('--invariance-factor', type=int, default=2)
    a = ap.parse_args()

    if a.invariance:
        invariance_check(Path(a.new_root), a.mouse, factor=a.invariance_factor)
        return

    old_root = a.new_root if a.self_check else a.old_root
    arm_labels = (('io_hmm 72 circular bins (self-check)', 'io_hmm 72 circular bins')
                  if a.self_check else
                  ('export Q, 91 linear bins', 'ideal-observer HMM, 72 circular bins'))
    tag = '_selfcheck' if a.self_check else ''

    ps.apply()
    print(f'Paired old-vs-new scorecard, mouse {a.mouse}'
          + ('   [SELF-CHECK: both arms = the io_hmm arm]' if a.self_check else ''))
    old, _, old_info = score_arm(Path(old_root), a.mouse, 'OLD')
    new, _, new_info = score_arm(Path(a.new_root), a.mouse, 'NEW')

    if not a.self_check and old and new:
        guard_distinct_arms(old_info, new_info, old_root, a.new_root)

    expected = {n for n, _l, _o, _w in build_cells()}
    candidates = sorted(set(old) & set(new), key=sort_key)
    cells, dropped = guard_pairing(old, new, candidates)
    if dropped:
        print(f'\ndropped {len(dropped)} paired cell(s) before plotting:')
        for c, why in dropped:
            print(f'  [drop] {c}: {why}')
    if cells:
        n_tr = len(new[cells[0]]['_trials']['orientation'])
        print(f'  pairing verified: {len(cells)} cell(s) parse, and both arms '
              f'report identical orientation/contrast/dispersion over the same '
              f'{n_tr} held-out trials')
    missing_old = expected - set(old)
    missing_new = expected - set(new)
    extra = (set(old) | set(new)) - expected

    if not cells:
        print(f'\nNO PAIRED CELLS: the OLD arm at {old_root} has '
              f'{len(old)} finished cells and the NEW arm has {len(new)}.')
        print('  Re-run once the export reference arm lands, or pass --self-check '
              'to validate the pairing/plotting logic in the meantime.')
        print(f'  expected grid ({len(expected)} cells): '
              + ' '.join(sorted(expected, key=sort_key)))
        return

    out_dir = Path(a.out_root)
    fig_paired_scatter(old, new, cells, out_dir, tag, arm_labels, mouse=a.mouse)
    fig_profiles(old, new, cells, out_dir, tag, arm_labels, mouse=a.mouse)
    fig_family_summary(old, new, cells, out_dir, tag, arm_labels, mouse=a.mouse)
    write_csv(old, new, cells, out_dir, tag)
    print_summary(old, new, cells, missing_old, missing_new, extra)

    if a.self_check:
        worst, nvals = 0.0, 0
        for cell in cells:
            for arch, _ in ARCHS:
                for key, _s, _l, _r, _g in METRICS + SECONDARY:
                    x, y = get(old, cell, arch, key), get(new, cell, arch, key)
                    if np.isfinite(x) and np.isfinite(y):
                        worst = max(worst, abs(x - y)); nvals += 1
        print(f'\nSELF-CHECK: {nvals} paired values, max |old - new| = {worst:.3e} '
              f'({"PASS — exactly on the identity line" if worst == 0 else "FAIL"})')
    print(f'\nDone. {out_dir.resolve()}')


if __name__ == '__main__':
    main()
