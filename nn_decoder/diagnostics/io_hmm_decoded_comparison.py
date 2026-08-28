# -*- coding: utf-8 -*-
"""Decoded (fitted) posteriors, old vs new target family — matched cells, mouse 0.

The paired arms train identical cells (same neurons, seeds, split, code path) and
differ ONLY in the target family:

  NEW  results/io_hmm_v1/<cell>/Q_<LOSS>_half_100ms/  — collaborator ideal-observer
       HMM posteriors, 72 circular bins x 2.5 deg over [0, 180).
  OLD  results/io_hmm_v1_exportref/<cell>/...          — the export's post_s_marginal
       Q, 91 linear bins over 0-90 deg.

Which support a shard lives on is READ FROM ``results[mouse].target_provenance``
(``target_source`` x ``n_bins``), never inferred from the array width, and an
unrecognised combination aborts instead of being guessed at.

X-AXIS MATCHING (stated on every axis, never hidden): the supports differ, so for
the POSTERIOR GALLERIES the new arm's rows — TARGET AND DECODED ALIKE — are folded
about 90 deg and resampled onto the old 1-deg grid with
``compare_io_hmm_vs_export_posteriors.fold_and_resample``. The old arm passes through
unchanged. Because target and decoded of a given arm go through the SAME transform,
the shapes drawn on one axis stay honest; but the fold is lossy (a median ~48% of
new-posterior mass sits beyond 90 deg) and it CHANGES WIDTH ESTIMATES, so it is used
for the galleries only. Scoring decoders through the fold would not be legitimate —
the skill ratios live in the paired scorecard, each arm on its own support.

THE CROSS-ARM PEAKINESS AXIS IS ŝ — SETTLED, DO NOT REOPEN
----------------------------------------------------------
Ground truth (synthetic decoders sharpened by a KNOWN factor, rendered on both
supports) rejected every fixed width statistic at matched true sharpening:

    raw Shannon entropy ratio  15%      Mardia circular SD ratio     26%
    max-prob ratio             22%      axial dispersion ratio       37%
    perplexity-width ratio     25%      differential-entropy delta   54%
    resultant (R) ratio        60%      linear SD ratio              69%

The cause is that the target SHAPES differ (the export Q is a bimodal cardinal-U
at low contrast; the IO-HMM posterior is flat-with-a-tilt), so no rescaling of a
width statistic can work. Cross-arm peakiness therefore comes from
``nn_decoder/sharpening_calibration.py``: per arm, calibrate that arm's OWN
targets under ``sharpen(p, s) = p ** (s**2)`` renormalised — a Gaussian-exact
width shrink by ``s``, defined support-freely — and invert the observed statistic
back to the equivalent sharpening factor ŝ (>1 over-sharpened, 1 calibrated,
<1 too broad). ŝ is always reported with ``agreement`` (>0.10 = the decoder
RESHAPES rather than sharpens, so ŝ under-describes it) and ``clamped`` (the
inversion hit a ladder end, so the value is a BOUND).

The calibration curves are built ONCE PER ARM: the targets are bit-identical
across every cell of an arm (measured — max |difference| = 0 over all 40 new and
all available old cells, spatial and temporal alike), and the code asserts it.

WHY THE WIDTH PANELS ARE SINGLE-ARM
-----------------------------------
The native width panels use ``native_sd``: an ordinary linear SD on the old
91-bin 0-90 grid, and the doubled-angle MARDIA circular SD ``sqrt(-2 ln R)/2`` on
the new 72-bin [0,180) grid. Those two estimators are not commensurable in level,
and — this is the point an earlier version of this docstring got wrong — the new
arm's estimator has NO finite ignorance level to normalise against: as R -> 0 the
Mardia SD DIVERGES (a uniform posterior returns 213 deg here, and that only
because R is floored at 1e-12; the true value is infinite). The "40.5 deg"
ignorance level quoted before belongs to ``_axial_dispersion``, the BOUNDED
sqrt(2(1-R))/2 statistic, which these panels never use. There is therefore no
normalisation that makes an old width and a new width mean the same thing.

So the width panels are now drawn ONE ARM PER FIGURE and say so in the title.
Within an arm they still answer a real question — does the decoder track the
target's trial-to-trial uncertainty? — read as the OLS slope, with both axes
divided by that arm's median target width purely to give a dimensionless axis
(dividing x and y by one constant cannot change the slope, the correlation, or
the meaning of the identity line). Comparing a slope or an offset across the two
figures is NOT licensed; the licensed cross-arm peakiness statement is the ŝ
figure and the ŝ table.

The FOLDED slope is quoted in each panel title as a within-arm cross-check,
because folding can flip its SIGN relative to the fold-free native slope
(measured: kl_lin_lh0, new arm, -0.18 folded vs +0.07 native). Where that happens
the panel frame turns red and the title says SIGN FLIP. The headline slope in
every title is the native, fold-free one, and it is the slope of the line drawn.

The width panels are within-mouse, within-cell: n = 470 held-out TRIALS, which are
not independent replicates (one animal, shared session structure). Slopes and r are
descriptive — no p-values are quoted and none should be inferred.

Figures (``spat`` = spatial/PPC, ``temp`` = temporal/sampling):
  a. gallery_trials_<arch>       — rows = cells, cols = representative held-out
     trials spanning contrast x orientation. Shaded = target, line = decoded;
     colour = arm. FOLDED (both arms on the 91-point 1-deg grid). The trial shown
     is the one CLOSEST TO ITS CONDITION MEAN (single trials are wobbly).
  b. gallery_condmean_<arch>     — same layout, averaged within stimulus condition.
  c. sharpening_<arch>           — CROSS-ARM: ŝ per cell, old vs new, with the
     reshape and clamp flags. This is the peakiness figure.
  d. width_tracking_<arch>_<arm> — SINGLE-ARM: per-trial decoded width vs target
     width on that arm's own native support, identity line + OLS slope.

Usage:
    python diagnostics/io_hmm_decoded_comparison.py            # default 8 cells
    python diagnostics/io_hmm_decoded_comparison.py --cells kl_h8_lh0 js_lin_lh0
    python diagnostics/io_hmm_decoded_comparison.py --old-root results/<other>
Cells absent from either arm (the export arm is still running) are skipped with a
printed warning and listed at the end.
Figures -> nn_decoder/figures/io_hmm_vs_export/decoded/ (png+svg via figsave.save_fig).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe      # noqa: E402
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
from matplotlib.patches import Patch     # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy.io import loadmat             # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))    # nn_decoder/
sys.path.insert(0, str(_HERE))           # nn_decoder/diagnostics/
import peakiness_style as ps             # noqa: E402
import sharpening_calibration as sharp   # noqa: E402  (the cross-arm peakiness axis)
from figsave import save_fig             # noqa: E402
from compare_io_hmm_vs_export_posteriors import (   # noqa: E402
    NEW_GRID_FULL, OLD_GRID, _circ_sd, _moments as _linear_moments, fold_and_resample)

ARCHS = ('spat', 'temp')
ARCH_LABEL = {'spat': 'spatial (PPC)', 'temp': 'temporal (sampling)'}

# Arm colours match diagnostics/compare_io_hmm_vs_export_posteriors.py so the
# target comparison and this decoded comparison read as one pair of figures.
C_OLD, C_NEW = 'tab:blue', 'tab:orange'
C_FLIP = 'crimson'                       # panel frame when the fold flips a slope sign
LBL_OLD = 'old (91-bin export Q)'
LBL_NEW = 'new (72-bin IO-HMM)'

# Above this peak-vs-entropy discrepancy the decoder is reshaping, not sharpening,
# and ŝ under-describes it. Same threshold as the paired scorecard.
AGREE_FLAG = 0.10

# The ONLY recognised (target_source, n_bins) supports — the same table as
# diagnostics/io_hmm_vs_export_scorecard.py. Geometry is read from provenance and
# never inferred from the array width; an unknown combination aborts.
# ``sd_estimator`` names the width statistic ``native_sd`` applies on that support,
# and ``ignorance`` is the value it takes on a UNIFORM posterior — None where the
# estimator has no finite ignorance level (which is why widths are single-arm).
GEOMETRY = {
    ('export', 91): {'family': 'export Q', 'label': 'linear 0–90°, 91 bins × 1°',
                     'sd_estimator': 'linear SD', 'ignorance': 26.3},
    ('io_hmm_pkl', 72): {'family': 'IO-HMM',
                         'label': 'circular [0,180°), 72 bins × 2.5°',
                         'sd_estimator': 'axial (doubled-angle) Mardia circular SD',
                         'ignorance': None},
}


def geometry(target_source, n_bins, n_cols=None):
    """Look up a shard's support from its provenance. Fails loudly on anything new."""
    key = (str(target_source), int(n_bins))
    if key not in GEOMETRY:
        raise SystemExit(
            f'\nABORT: unrecognised support (target_source={key[0]!r}, '
            f'n_bins={key[1]}).\n  Geometry is read from target_provenance, never '
            'guessed from the bin count. Known: '
            + ', '.join(f'{s!r} x {n}' for s, n in sorted(GEOMETRY)))
    if n_cols is not None and int(n_cols) != int(n_bins):
        raise SystemExit(
            f'\nABORT: target_provenance says n_bins={n_bins} but the saved '
            f'posterior has {n_cols} columns ({key[0]!r}); refusing to guess.')
    return GEOMETRY[key]


def _arm_label(tag, source, n_bins):
    """Legend text derived from the PROVENANCE actually loaded, so a self-paired
    rendering run (--old-root pointed at the io_hmm arm) is never mislabelled as
    the export reference."""
    return f'{tag} ({n_bins}-bin {geometry(source, n_bins)["family"]})'


DEFAULT_CELLS = ('kl_h8_lh0', 'js_h8_lh0', 'pca_h8_lh0', 'pcaflat_h8_lh0',
                 'kl_lin_lh0', 'js_lin_lh0', 'pca_lin_lh0', 'pcaflat_lin_lh0')

# (contrast, orientation): the orientation spread at full contrast plus the
# contrast ladder at 45 deg — the axis along which the two target families are
# known to diverge (old widths are strongly contrast-dependent, new are not).
CONDITIONS = ((1.0, 0), (1.0, 45), (1.0, 90), (0.25, 45), (0.01, 45), (0.01, 90))

TRIAL_KEYS = ('orientation', 'contrast', 'dispersion')


# ---------------------------------------------------- per-arm ŝ calibration ----
class ArmCalibration:
    """The sharpening calibration curves for ONE arm, built once and reused.

    Building them is the expensive part (a full ladder of sharpened target blocks)
    and they depend only on the arm's targets — which are bit-identical across
    every cell of an arm. That assumption is not taken on trust: every cell's
    target block is checked against the reference and any drift aborts, because a
    drifting target would silently invalidate every reused curve.
    """

    def __init__(self, name):
        self.name = name
        self.curves = None
        self._ref = None
        self.n_cells_checked = 0
        self.max_target_drift = 0.0
        self.identity_s_hat = np.nan

    def ensure(self, targets):
        t = np.asarray(targets, float)
        ok = np.isfinite(t).all(1)
        if self.curves is None:
            self._ref = t
            self.curves = {k: sharp.calibration_curve(t[ok], k)
                           for k in sharp.BASE_STATS}
            # end-to-end sanity: a decoder identical to its own target must
            # invert to exactly 1 through THIS code path, not just the module's
            # own self-test.
            self.identity_s_hat = self.s_hat(t[ok], t[ok])['s_hat']
            if abs(self.identity_s_hat - 1.0) > 1e-9:
                raise SystemExit(
                    f'\nABORT: {self.name} ŝ calibration is broken — a decoder '
                    f'identical to its own target inverts to '
                    f'{self.identity_s_hat:.6f}, not 1.')
        elif self._ref.shape == t.shape:
            d = float(np.nanmax(np.abs(t - self._ref)))
            self.max_target_drift = max(self.max_target_drift, d)
            if d > 0:
                raise SystemExit(
                    f'\nABORT: {self.name} targets are NOT identical across cells '
                    f'(max |difference| = {d:.3e}). One calibration per arm is '
                    'only valid for ONE target family.')
        else:
            raise SystemExit(
                f'\nABORT: {self.name} target block changed shape between cells '
                f'({self._ref.shape} vs {t.shape}).')
        self.n_cells_checked += 1
        return self.curves

    def s_hat(self, dec, tgt):
        r = sharp.equivalent_sharpening(dec, tgt, self.curves)
        return {'s_hat': r['s_hat'], 's_hat_entropy': r['s_hat_entropy'],
                'agreement': r['agreement'], 'clamped': bool(r['clamped'])}


# ------------------------------------------------------------------ loading ----
def _find_shard(root, cell):
    """The cell's shard, tolerating the PCA runs' 'Q_PCA_half_100ms_all' slug."""
    hits = sorted(Path(root).glob(f'{cell}/Q_*/stratified_balanced.mat'))
    return hits[0] if hits else None


def load_cell(root, cell, mouse):
    """{'trials', 'n_bins', 'source', 'geom', <arch>: {'dec', 'tgt'}} or None."""
    shard = _find_shard(root, cell)
    if shard is None:
        return None
    m = loadmat(str(shard), squeeze_me=True, struct_as_record=False)
    r = getattr(m['results'], f'mouse_{mouse}', None)
    if r is None:
        return None
    prov = getattr(r, 'target_provenance', None)
    if prov is None or not hasattr(prov, 'target_source') or not hasattr(prov, 'n_bins'):
        raise SystemExit(
            f'\nABORT: {shard} has no usable target_provenance (target_source / '
            'n_bins). The support is read from provenance, never guessed from the '
            'bin count; refusing to plot this shard.')
    out = {'shard': shard,
           'source': str(prov.target_source), 'n_bins': int(prov.n_bins),
           'trials': {k: np.asarray(getattr(r.trials, k)).ravel().astype(float)
                      for k in TRIAL_KEYS}}
    for arch in ARCHS:
        a = getattr(r.Dist, arch, None)
        if a is None:
            continue
        dec = np.asarray(a.decoded, dtype=np.float64)
        tgt = np.asarray(a.target, dtype=np.float64)
        if dec.ndim != 2 or dec.shape != tgt.shape:
            continue
        geometry(out['source'], out['n_bins'], n_cols=dec.shape[1])   # fails loudly
        out[arch] = {'dec': dec, 'tgt': tgt}
    out['geom'] = geometry(out['source'], out['n_bins'])
    return out


def to_common_grid(p):
    """(n, nbins) probability rows -> (n, 91) rows on OLD_GRID, + mass beyond 90.

    91 bins pass through (already the common grid, renormalised); 72 circular bins
    are folded about 90 deg and resampled. Applied identically to targets and
    decoded posteriors within an arm. GALLERIES ONLY — the fold is lossy and
    changes widths, so nothing scored goes through here.
    """
    p = np.asarray(p, dtype=np.float64)
    s = p.sum(1, keepdims=True)
    p = np.divide(p, s, out=np.full_like(p, np.nan), where=s > 0)
    if p.shape[1] == len(OLD_GRID):
        return p, np.zeros(len(p))
    if p.shape[1] == len(NEW_GRID_FULL):
        mass_beyond, _dens, on_old = fold_and_resample(p)
        return on_old, mass_beyond
    raise ValueError(f'unexpected bin count {p.shape[1]} (expected 91 or 72)')


def native_sd(p):
    """Posterior width in degrees on the arm's OWN support (no fold). WITHIN-ARM
    ONLY: linear SD on the 91-bin 0-90 grid (69% cross-arm error against ground
    truth), axial doubled-angle Mardia circular SD on the 72-bin [0,180) grid
    (26%). Never put two arms' values on one axis — use ŝ for that.

    CAVEAT on the circular branch: the new posteriors sit near R ~ 0.2-0.3, where
    sqrt(-2 ln R) is heavy-tailed, and it DIVERGES as R -> 0, so this estimator
    has no finite ignorance level to normalise against.
    """
    p = np.asarray(p, dtype=np.float64)
    if p.shape[1] == len(OLD_GRID):
        return _linear_moments(p, OLD_GRID)[1]
    return _circ_sd(p, NEW_GRID_FULL)


def pair_cell(old_root, new_root, cell, mouse, cal):
    """Both arms for one cell, or (None, reason).

    ``cal`` is {'old': ArmCalibration, 'new': ArmCalibration} — the per-arm ŝ
    calibration, built on the first cell seen and reused (and re-verified) after.
    """
    new = load_cell(new_root, cell, mouse)
    old = load_cell(old_root, cell, mouse)
    if new is None:
        return None, 'absent from the new (io_hmm) arm'
    if old is None:
        return None, 'absent from the old (export) arm — still running?'
    for k in TRIAL_KEYS:
        if not np.array_equal(old['trials'][k], new['trials'][k]):
            return None, f'test-split mismatch between arms ({k} differs)'
    out = {'trials': new['trials'], 'sources': (old['source'], new['source']),
           'n_bins': (old['n_bins'], new['n_bins']),
           'geom': (old['geom'], new['geom'])}
    for arch in ARCHS:
        if arch not in old or arch not in new:
            continue
        d = {}
        for tag, arm in (('old', old), ('new', new)):
            for what in ('tgt', 'dec'):
                common, mass = to_common_grid(arm[arch][what])
                d[f'{tag}_{what}'] = common
                d[f'{tag}_{what}_sd'] = _linear_moments(common, OLD_GRID)[1]
                d[f'{tag}_{what}_sd_native'] = native_sd(arm[arch][what])
                if what == 'tgt':
                    d[f'{tag}_mass_beyond'] = mass
            # THE cross-arm peakiness axis: calibrated on this arm's own targets,
            # on this arm's own NATIVE rows (never through the fold).
            dec_n, tgt_n = arm[arch]['dec'], arm[arch]['tgt']
            ok = np.isfinite(dec_n).all(1) & np.isfinite(tgt_n).all(1)
            cal[tag].ensure(tgt_n)
            d[f'{tag}_sharp'] = cal[tag].s_hat(dec_n[ok], tgt_n[ok])
        out[arch] = d
    return out, None


# ------------------------------------------------------------- trial picking ----
def condition_masks(trials):
    """[(contrast, orientation, mask)] for the CONDITIONS that exist in the split."""
    ct = np.round(trials['contrast'], 3)
    ori = trials['orientation']
    out = []
    for c, o in CONDITIONS:
        m = (ct == round(c, 3)) & (ori == o)
        if m.any():
            out.append((c, o, m))
    return out


def representative_trial(mask, tgt_arms):
    """Index of the trial closest to its condition mean, summed over the arms'
    TARGETS (single trials swing wrong-way often enough that a random pick
    misleads — see compare_io_hmm_vs_export_posteriors fig 3)."""
    idx = np.where(mask)[0]
    score = np.zeros(len(idx))
    for t in tgt_arms:
        sub = t[idx]
        score = score + np.abs(sub - sub.mean(0, keepdims=True)).sum(1)
    return int(idx[np.argmin(score)])


# -------------------------------------------------------------------- plots ----
def _panel(ax, old_t, old_d, new_t, new_d, ori):
    ax.fill_between(OLD_GRID, old_t, color=C_OLD, alpha=0.20, lw=0)
    ax.fill_between(OLD_GRID, new_t, color=C_NEW, alpha=0.20, lw=0)
    ax.plot(OLD_GRID, old_d, lw=1.4, color=C_OLD)
    ax.plot(OLD_GRID, new_d, lw=1.4, color=C_NEW)
    # cap BEFORE the reference lines: cap_posterior_ylim inspects ax.get_lines(),
    # and axhline/axvline carry list (not array) ydata.
    ps.cap_posterior_ylim(ax, max(np.nanmax(old_t), np.nanmax(new_t)), mult=3.0)
    ax.axhline(1.0 / len(OLD_GRID), color='0.6', lw=0.7, ls=':')
    ax.axvline(ori, color='k', lw=0.7, ls=':')
    ax.set_xlim(0, 90)
    ax.set_xticks([0, 45, 90])


def _finish(fig, handles, ncol, xlabel=None, bottom_in=0.55, top_in=0.32):
    """Park a figure legend (and optional shared x-label) below the axes and
    reserve room for them in INCHES, so the reservation holds whether the figure
    is one panel tall or eight."""
    h = fig.get_size_inches()[1]
    fig.legend(handles=handles, loc='lower center', ncol=ncol, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.04 / h))
    if xlabel:
        fig.text(0.5, (bottom_in - 0.22) / h, xlabel, ha='center', fontsize=8.5)
    fig.set_layout_engine('constrained')
    # constrained layout accounts for neither the figure legend nor the suptitle.
    # NB its rect is (left, bottom, WIDTH, HEIGHT) — not (left, bottom, right, top)
    # as tight_layout takes; passing a 'top' there silently overflows the figure.
    bottom, top = bottom_in / h, 1.0 - top_in / h
    fig.get_layout_engine().set(rect=(0, bottom, 1, top - bottom))


def _gallery_handles(lbl_old=LBL_OLD, lbl_new=LBL_NEW):
    return [Patch(facecolor=C_OLD, alpha=0.20, label=f'{lbl_old} — target'),
            Line2D([], [], color=C_OLD, lw=1.4, label=f'{lbl_old} — decoded'),
            Patch(facecolor=C_NEW, alpha=0.20, label=f'{lbl_new} — target'),
            Line2D([], [], color=C_NEW, lw=1.4, label=f'{lbl_new} — decoded'),
            Line2D([], [], color='0.6', lw=0.7, ls=':', label='uniform (1/91)'),
            Line2D([], [], color='k', lw=0.7, ls=':', label='true orientation')]


def gallery(paired, cells, arch, mouse, out_dir, condition_mean=False, labels=None):
    """Rows = cells, cols = stimulus conditions; one held-out trial per panel
    (or the condition mean). Returns the figure stem, or None if nothing to plot."""
    rows = [c for c in cells if c in paired and arch in paired[c]]
    if not rows:
        return None
    trials = paired[rows[0]]['trials']
    conds = condition_masks(trials)
    nrow, ncol = len(rows), len(conds)
    fig, axes = plt.subplots(nrow, ncol, squeeze=False, sharex=True,
                             figsize=(min(13.0, 2.15 * ncol),
                                      min(13.0, 1.55 * nrow + 1.5)))
    for i, cell in enumerate(rows):
        d = paired[cell][arch]
        for j, (c, o, mask) in enumerate(conds):
            ax = axes[i][j]
            if condition_mean:
                sl = mask
                tag = f'n={int(mask.sum())}'
            else:
                t = representative_trial(mask, (d['old_tgt'], d['new_tgt']))
                sl = slice(t, t + 1)
                tag = f'trial {t}'
            _panel(ax, d['old_tgt'][sl].mean(0), d['old_dec'][sl].mean(0),
                   d['new_tgt'][sl].mean(0), d['new_dec'][sl].mean(0), o)
            ax.tick_params(labelsize=6)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            if i == 0:
                ax.set_title(f'ctr {c:g}  ori {o:g}', fontsize=8)
            ax.annotate(tag, xy=(0.03, 0.95), xycoords='axes fraction',
                        fontsize=5.5, color='0.45', va='top')
            if j == 0:
                ax.set_ylabel(f'{cell}\nprob / deg', fontsize=7)
    what = 'condition-mean' if condition_mean else 'example held-out trials'
    fig.suptitle(f'Mouse {mouse}, {ARCH_LABEL[arch]} decoder — {what} posteriors, '
                 'old vs new target family (shaded = target, line = decoded)',
                 fontsize=10)
    _finish(fig, _gallery_handles(*(labels or (LBL_OLD, LBL_NEW))), ncol=3, bottom_in=0.95,
            xlabel='orientation (deg; the new arm is folded about 90 deg and '
                   'resampled onto the old 1-deg grid)')
    stem = f'gallery_{"condmean" if condition_mean else "trials"}_{arch}_m{mouse}'
    save_fig(fig, out_dir, stem, layout=None, max_px=1560)
    return stem


# ----------------------------------------------- (c) CROSS-ARM peakiness: ŝ ----
def sharpening_figure(paired, cells, archs, mouse, out_dir, labels=None):
    """ŝ per cell, old vs new — the one licensed cross-arm peakiness figure.

    One panel per architecture; cells on y, ŝ on a log x-axis centred on 1. The
    two companions are drawn, never dropped: a ring = the decoder reshapes rather
    than sharpens (agreement > AGREE_FLAG, so ŝ under-describes it), a cross = the
    inversion clamped at a ladder end, so the value is a bound.
    """
    lbl_old, lbl_new = labels or (LBL_OLD, LBL_NEW)
    archs = [a for a in archs if any(a in paired[c] for c in cells)]
    if not archs:
        return None
    rows = [c for c in cells if c in paired]
    ypos = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, len(archs), squeeze=False, sharey=True,
                             figsize=(min(13.0, 5.0 * len(archs) + 1.6),
                                      max(3.4, 0.42 * len(rows) + 2.6)))
    lo, hi = 1.0, 1.0
    for ci, arch in enumerate(archs):
        ax = axes[0][ci]
        for yi, cell in zip(ypos, rows):
            d = paired[cell].get(arch)
            if d is None:
                continue
            so, sn = d['old_sharp'], d['new_sharp']
            lo = min(lo, so['s_hat'], sn['s_hat'])
            hi = max(hi, so['s_hat'], sn['s_hat'])
            ax.plot([so['s_hat'], sn['s_hat']], [yi, yi], '-', lw=1.4,
                    color='0.6', zorder=1)
            for s, colr, mk in ((so, C_OLD, 's'), (sn, C_NEW, 'o')):
                ax.plot(s['s_hat'], yi, mk, ms=7, color=colr, zorder=3)
                if s['agreement'] > AGREE_FLAG:
                    ax.plot(s['s_hat'], yi, 'o', ms=13, mfc='none', mec='0.2',
                            mew=1.0, zorder=2)
                if s['clamped']:
                    ax.plot(s['s_hat'], yi, 'x', ms=9, color='k', mew=1.4, zorder=4)
        ax.axvline(1.0, ls=':', lw=1.0, color='0.4')
        ax.set_xscale('log')
        cand = np.array([0.25, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        t = cand[(cand >= lo / 1.3) & (cand <= hi * 1.3)]
        if t.size >= 2:
            ax.set_xticks(t)
            ax.set_xticklabels([f'{v:g}' for v in t])
            ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlim(lo / 1.3, hi * 1.3)
        ax.set_title(ARCH_LABEL[arch], fontsize=9.5)
        ax.set_xlabel('equivalent sharpening ŝ (log scale)\n'
                      'calibrated per arm on that arm’s OWN targets:\n'
                      'ŝ = the s for which target^(s²), renormalised, has the\n'
                      'decoder’s peakiness. 1 = calibrated, >1 over-sharpened',
                      fontsize=7.5)
        for yi in ypos[1::2]:
            ax.axhspan(yi - 0.5, yi + 0.5, color='0.95', zorder=0)
        ax.tick_params(labelsize=8)
    axes[0][0].set_yticks(ypos)
    axes[0][0].set_yticklabels(rows, fontsize=8)
    axes[0][0].set_ylim(ypos.min() - 0.6, ypos.max() + 0.6)
    handles = [Line2D([], [], color=C_OLD, marker='s', ls='none', ms=7,
                      label=f'{lbl_old} — ŝ'),
               Line2D([], [], color=C_NEW, marker='o', ls='none', ms=7,
                      label=f'{lbl_new} — ŝ'),
               Line2D([], [], color='0.4', ls=':', label='ŝ = 1 (calibrated)'),
               Line2D([], [], marker='o', ls='none', mfc='none', mec='0.2', ms=11,
                      label=f'ring: agreement > {AGREE_FLAG:.2f} — the decoder '
                            'RESHAPES, ŝ under-describes it'),
               Line2D([], [], marker='x', ls='none', color='k', ms=9,
                      label='cross: ŝ clamped at a ladder end — a bound, not an '
                            'estimate')]
    fig.suptitle(
        f'Mouse {mouse} — equivalent sharpening ŝ per cell, old vs new target '
        'family. THE cross-arm peakiness axis.\n'
        'Every fixed width statistic (max-prob 22%, entropy 15%, circular SD 26%, '
        'axial dispersion 37%, ΔH 54%, R 60%, linear SD 69%, perplexity 25%) was '
        'rejected against ground truth; ŝ recovers a known s to 0.00%.', fontsize=9)
    _finish(fig, handles, ncol=2, bottom_in=0.95, top_in=0.12)
    stem = f'sharpening_m{mouse}'
    save_fig(fig, out_dir, stem, layout=None, max_px=1560)
    return stem


# ------------------------------------------- (d) SINGLE-ARM width tracking ----
def _slope_r(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0:
        return np.nan, np.nan
    slope = float(np.polyfit(x[ok], y[ok], 1)[0])
    if np.ptp(y[ok]) == 0:            # a perfectly flat decoder: r undefined
        return slope, np.nan
    return slope, float(np.corrcoef(x[ok], y[ok])[0, 1])


def _sign_flip(a, b):
    """True when two slopes disagree in sign and both are big enough to mean it."""
    return bool(np.isfinite(a) and np.isfinite(b)
                and abs(a) > 0.02 and abs(b) > 0.02 and a * b < 0)


def width_tracking(paired, cells, arch, mouse, out_dir, tag, label):
    """SINGLE-ARM: decoded vs target posterior width per trial, one panel per cell.

    One arm per figure, on that arm's own native estimator (see the module
    docstring: the two arms' width estimators have incommensurable levels and the
    circular one has no finite ignorance level, so a shared axis would be a
    category error). Both axes are divided by this arm's median TARGET width
    purely to give the panel a dimensionless axis; that division cannot change the
    slope, the correlation, or the meaning of the identity line.

    Read the SLOPE (does the decoder track the target's trial-to-trial
    uncertainty?). Do NOT read the vertical offset against the other arm's figure
    — cross-arm peakiness is the ŝ figure.

    Returns (stem, stats).
    """
    rows = [c for c in cells if c in paired and arch in paired[c]]
    if not rows:
        return None, {}
    arm_i = 0 if tag == 'old' else 1
    geom = paired[rows[0]]['geom'][arm_i]
    nb = paired[rows[0]]['n_bins'][arm_i]
    folded = nb != len(OLD_GRID)
    colr = C_OLD if tag == 'old' else C_NEW
    # balanced grid: 4 columns max, then spread the cells evenly so a partial arm
    # (the export reference lands cell by cell) doesn't leave a row three-quarters
    # empty — 5 cells give 3x2, not 4x2.
    nrow = int(np.ceil(len(rows) / 4))
    ncol = int(np.ceil(len(rows) / nrow))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(min(13.0, 3.3 * ncol),
                                      min(13.0, 3.2 * nrow + 0.9)))
    stats, n_flip = {}, 0
    for k, cell in enumerate(rows):
        ax = axes[k // ncol][k % ncol]
        d = paired[cell][arch]
        s = {}
        xn, yn = d[f'{tag}_tgt_sd_native'], d[f'{tag}_dec_sd_native']
        ref = float(np.nanmedian(xn))                 # this arm's median target width
        x, y = xn / ref, yn / ref
        ax.scatter(x, y, s=7, alpha=0.45, color=colr, lw=0)
        # slope/r are invariant to the common divisor, so these ARE the native
        # numbers and they are the numbers the drawn fit line has.
        s['slope'], s['r'] = _slope_r(x, y)
        s['slope_fold'], s['r_fold'] = _slope_r(d[f'{tag}_tgt_sd'],
                                                d[f'{tag}_dec_sd'])
        s['flip'] = folded and _sign_flip(s['slope'], s['slope_fold'])
        s['tgt_sd_p10'] = float(np.nanpercentile(xn, 10))
        s['tgt_sd_p90'] = float(np.nanpercentile(xn, 90))
        s['tgt_sd_med'] = ref
        s['dec_sd_med'] = float(np.nanmedian(yn))
        hi = float(np.nanmax([x, y]))
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 3 and np.ptp(x[ok]) > 0:
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 20)
            b = np.polyfit(x[ok], y[ok], 1)
            # white halo: the new arm's target-width range is so narrow that its
            # fit is a short segment buried in its own point cloud.
            ax.plot(xs, np.polyval(b, xs), color=colr, lw=2.2, zorder=4,
                    solid_capstyle='butt',
                    path_effects=[pe.Stroke(linewidth=4.0, foreground='w'),
                                  pe.Normal()])
        lo, hi = 0.0, hi * 1.05
        ax.plot([lo, hi], [lo, hi], 'k:', lw=0.9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        if not folded:
            tail = 'fold n/a (already 91-bin)'
        elif s['flip']:
            tail = f'folded {s["slope_fold"]:+.2f}  ** SIGN FLIP **'
        else:
            tail = f'folded {s["slope_fold"]:+.2f}'
        n_flip += bool(s['flip'])
        ax.set_title(f'{cell}\nslope {s["slope"]:+.2f} (r {s["r"]:.2f})  |  {tail}',
                     fontsize=7.2, color=C_FLIP if s['flip'] else 'k')
        if s['flip']:
            # the house style hides the top/right spines, so re-show them: half a
            # red box reads as a rendering glitch, a whole one reads as a warning
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(C_FLIP)
                sp.set_linewidth(1.8)
        ax.tick_params(labelsize=7)
        if k % ncol == 0:
            ax.set_ylabel('decoded width / same divisor', fontsize=8)
        if k // ncol == nrow - 1:
            ax.set_xlabel("target width / this arm's median target width", fontsize=8)
        stats[cell] = s
    for k in range(len(rows), nrow * ncol):
        axes[k // ncol][k % ncol].set_axis_off()
    ign = ('uniform posterior = %.1f°' % geom['ignorance'] if geom['ignorance']
           else 'uniform posterior → ∞ (no finite ignorance level)')
    handles = [Line2D([], [], color=colr, marker='o', lw=2.2, ls='-',
                      label=f'{label} — native {geom["sd_estimator"]} on '
                            f'{geom["label"]}; {ign}'),
               Line2D([], [], color='k', lw=0.9, ls=':',
                      label='identity: decoded width = target width'),
               Patch(facecolor='none', edgecolor=C_FLIP, lw=1.8,
                     label='red frame: folding flips the slope sign — the headline '
                           'number is the fold-free one')]
    fig.suptitle(
        f'Mouse {mouse}, {ARCH_LABEL[arch]} decoder — SINGLE ARM: {label}. '
        "Does the decoded posterior track the target's width? (per held-out "
        'trial, OLS fit)\n'
        f'Native {geom["sd_estimator"]}, no fold; both axes ÷ this arm\'s median '
        'target width (a common divisor, so slope, r and the identity line are '
        'unchanged).\n'
        'WITHIN-ARM ONLY — this width estimator is not commensurable with the '
        'other arm\'s; for cross-arm peakiness see the calibrated ŝ figure.',
        fontsize=9)
    # top_in stays small: matplotlib's constrained layout already reserves room for
    # the suptitle, so adding a big rect margin on top of it just opens a gap.
    _finish(fig, handles, ncol=1, bottom_in=0.72, top_in=0.10)
    stem = f'width_tracking_{arch}_{tag}_m{mouse}'
    save_fig(fig, out_dir, stem, layout=None, max_px=1560)
    if n_flip:
        print(f'  [fold] {n_flip}/{len(rows)} {arch} {tag}-arm panels flagged: the '
              'folded slope disagrees in sign with the native one (red frame)')
    return stem, stats


# --------------------------------------------------------------------- main ----
def print_sharpening_table(paired, cells, arch, labels):
    """CROSS-ARM: the licensed peakiness comparison."""
    print(f'\n{arch} — EQUIVALENT SHARPENING ŝ (cross-arm; calibrated per arm on '
          "that arm's OWN targets)")
    print('  ŝ > 1 over-sharpened, 1 calibrated, < 1 too broad. "!" = agreement > '
          f'{AGREE_FLAG:.2f}: the decoder RESHAPES rather than sharpens, so ŝ '
          'under-describes it.')
    print('  "clamp" = the inversion hit a ladder end, so the value is a BOUND. '
          'Every fixed width statistic was rejected cross-arm (15-69% error at '
          'matched true sharpening).')
    print(f'{"cell":18s} | {"OLD ŝ":>7s} {"ŝ_ent":>7s} {"agree":>6s} {"":5s} | '
          f'{"NEW ŝ":>7s} {"ŝ_ent":>7s} {"agree":>6s} {"":5s} |  new÷old')
    for cell in cells:
        d = paired[cell].get(arch)
        if d is None:
            continue
        so, sn = d['old_sharp'], d['new_sharp']
        row = f'{cell:18s} |'
        for s in (so, sn):
            fl = ('!' if s['agreement'] > AGREE_FLAG else ' ') + \
                 ('clamp' if s['clamped'] else '     ')
            row += (f'{s["s_hat"]:8.3f}{s["s_hat_entropy"]:8.3f}'
                    f'{s["agreement"]:7.3f} {fl:5s} |')
        row += f' {sn["s_hat"] / so["s_hat"]:8.2f}'
        print(row)


def print_width_table(stats, arch, tag, geom, label):
    """WITHIN-ARM: slope only. No cross-arm quantity appears here."""
    if not stats:
        return
    print(f'\n{arch} / {tag} arm — WITHIN-ARM width tracking ({label})')
    print(f'  native {geom["sd_estimator"]} on {geom["label"]}; '
          + ('uniform posterior = %.1f°' % geom['ignorance'] if geom['ignorance']
             else 'uniform posterior → ∞, so this estimator has NO finite '
                  'ignorance level'))
    print('  "slope" is the fold-free slope the figure draws; "slope(fold)" is the '
          'same fit after folding onto the 91-bin grid, a cross-check only '
          '("*" = they disagree in SIGN).')
    print('  Widths and slopes here are NOT comparable with the other arm — that '
          'comparison is the ŝ table above.')
    print(f'{"cell":18s} {"tgt p10-p90 (deg)":>18s} {"tgt med":>8s} '
          f'{"dec med":>8s} {"slope":>7s} {"r":>6s} {"slope(fold)":>12s}')
    for cell, s in stats.items():
        print(f'{cell:18s} '
              f'{s["tgt_sd_p10"]:8.1f}-{s["tgt_sd_p90"]:<9.1f} '
              f'{s["tgt_sd_med"]:8.1f} {s["dec_sd_med"]:8.1f} '
              f'{s["slope"]:7.2f} {s["r"]:6.2f} {s["slope_fold"]:11.2f}'
              f'{"*" if s["flip"] else " "}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--new-root', type=Path,
                    default=_HERE.parent / 'results' / 'io_hmm_v1',
                    help='IO-HMM (72-bin circular) arm')
    ap.add_argument('--old-root', type=Path,
                    default=_HERE.parent / 'results' / 'io_hmm_v1_exportref',
                    help='export post_s_marginal (91-bin linear) reference arm')
    ap.add_argument('--cells', nargs='*', default=list(DEFAULT_CELLS))
    ap.add_argument('--archs', nargs='*', default=list(ARCHS), choices=list(ARCHS))
    ap.add_argument('--mouse', type=int, default=0)
    ap.add_argument('--out-dir', type=Path,
                    default=_HERE.parent / 'figures' / 'io_hmm_vs_export' / 'decoded')
    args = ap.parse_args()

    ps.apply()
    print(f'new arm: {args.new_root}\nold arm: {args.old_root}\nmouse {args.mouse}')

    cal = {'old': ArmCalibration('OLD arm'), 'new': ArmCalibration('NEW arm')}
    paired, skipped = {}, []
    for cell in args.cells:
        p, why = pair_cell(args.old_root, args.new_root, cell, args.mouse, cal)
        if p is None:
            skipped.append((cell, why))
            print(f'  [skip] {cell}: {why}')
            continue
        paired[cell] = p
        print(f'  [ok]   {cell}: bins old/new {p["n_bins"][0]}/{p["n_bins"][1]}, '
              f'sources {p["sources"][0]}/{p["sources"][1]}, '
              f'{len(p["trials"]["orientation"])} test trials')
    if not paired:
        print('\nno cell present in BOTH arms — nothing to plot. '
              'Rerun once the export reference arm has landed.')
        return

    for tag in ('old', 'new'):
        c = cal[tag]
        print(f'  ŝ calibration [{c.name}]: built ONCE, reused over '
              f'{c.n_cells_checked} cell-arch target blocks; max |target − '
              f'reference| {c.max_target_drift:.2e}; identity check '
              f'ŝ(target,target) = {c.identity_s_hat:.6f}')

    cells = [c for c in args.cells if c in paired]
    srcs = paired[cells[0]]['sources']
    if srcs[0] == srcs[1]:
        print(f"\n[warn] BOTH roots report target_source={srcs[0]!r} — this is not a "
              "cross-family comparison; check --old-root points at the export arm.")
    nb_old, nb_new = paired[cells[0]]['n_bins']
    labels = (_arm_label('old', srcs[0], nb_old), _arm_label('new', srcs[1], nb_new))
    med_mass = np.nanmedian(paired[cells[0]][args.archs[0]]['new_mass_beyond'])
    print(f'\nfold check: median new-target mass beyond 90 deg = {med_mass:.3f} '
          '(folded back onto the 0-90 grid for the GALLERIES only; the width panels '
          'and ŝ are fold-free)')

    written = []
    stem = sharpening_figure(paired, cells, args.archs, args.mouse, args.out_dir,
                             labels=labels)
    if stem:
        written.append(stem)
    for arch in args.archs:
        for cm in (False, True):
            stem = gallery(paired, cells, arch, args.mouse, args.out_dir,
                           condition_mean=cm, labels=labels)
            if stem:
                written.append(stem)
        print_sharpening_table(paired, cells, arch, labels)
        for ai, tag in enumerate(('old', 'new')):
            stem, stats = width_tracking(paired, cells, arch, args.mouse,
                                         args.out_dir, tag, labels[ai])
            if stem:
                written.append(stem)
            print_width_table(stats, arch, tag, paired[cells[0]]['geom'][ai],
                              labels[ai])
    n_tr = len(paired[cells[0]]['trials']['orientation'])
    print(f'\n(n = {n_tr} held-out trials, one mouse; trials are not independent '
          'replicates, so slopes and r are descriptive — no p-values.)')

    print(f'\n{len(written)} figures -> {args.out_dir}')
    if skipped:
        print(f'skipped {len(skipped)} cell(s): '
              + ', '.join(f'{c} ({w.split(" —")[0]})' for c, w in skipped))


if __name__ == '__main__':
    main()
