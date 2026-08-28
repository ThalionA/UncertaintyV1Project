# -*- coding: utf-8 -*-
"""Per-trial spatial-vs-temporal scatter with exemplar trials pulled out and their
TEMPORAL BIN posteriors shown as heatmaps — one figure per configuration
(2026-08-04, Theo's request).

Layout per configuration:
  left, tall   scatter of per-trial loss, spatial (x) vs temporal (y), each
               normalised to that mouse's leave-one-out predict-mean so 1 = chance
               on both axes. Below the diagonal = temporal better. Four exemplar
               trials are circled and numbered.
  top row      for each exemplar: IO target (grey fill) with the spatial and
               temporal decoded posteriors overlaid.
  bottom row   for each exemplar: the temporal decoder's INDIVIDUAL time-bin
               posteriors (`decoded_samp`, n_orientations x ~10 bins — 91 on the
               export grid, 72 on the IO-HMM one) as a heatmap — what the Jensen
               average is averaging over.

Exemplars are chosen to span the scatter, using
  d = log(temporal) - log(spatial)   (which architecture wins, and by how much)
  m = log(temporal) + log(spatial)   (overall difficulty of the trial)
picking: both good (lowest m), spatial much better (largest d), temporal much
better (smallest d), both bad (largest m).

Trials are pooled ACROSS mice after per-mouse normalisation, so an exemplar is
labelled with its mouse. Scored under the cell's OWN training weighting — spatial
and temporal share it within a cell, so the comparison is like-for-like.

`--by-mouse` splits that pooled scatter into ONE PANEL PER MOUSE and moves the
exemplar columns to their own figure (2026-08-28, Theo's request):

  <run>_trialsbymouse_<cell>[_<metric>]   6 scatter panels, shared axis limits,
                                          each with its own identity line, its own
                                          predict-mean lines, its trial count and a
                                          paired test over ITS trials. Points are
                                          COLOURED by how far that trial's individual
                                          time bins sit from the temporal average the
                                          y axis scores — mean-over-bins total
                                          variation, from
                                          `decoder_metrics.bin_divergence_from_mean`.
                                          0 = every bin says the same thing, so the
                                          Jensen average discards nothing. The colour is scaled PER
                                          PANEL (2026-08-28, Theo's ask): each mouse's
                                          bins disagree by its own amount, and a
                                          shared ramp buries that under the
                                          between-animal offset — so a colour means a
                                          different TV in each panel and the
                                          colourbars keep their tick labels. Each
                                          panel also prints the median TV either side
                                          of the identity and the rank correlation
                                          with log(temporal/spatial), so the colour is
                                          a claim, not decoration. Unlike the axes the
                                          TV is weighting-independent — it is computed
                                          on the raw posteriors — so it IS comparable
                                          across configs, panel-by-panel, by reading
                                          the colourbar numbers.
  <run>_trialexemplars_<cell>[_<metric>]  the same four exemplars (still picked from
                                          the POOLED scatter by `pick`, each ringed
                                          in the panel of the mouse it came from),
                                          drawn by the same `_draw_exemplars` body.

Pooling hides between-animal structure: each mouse is normalised to its own
predict-mean, so the six clouds are separately interpretable and the pooled cloud is
a mixture of six. The per-mouse panels default to `--trial-stat median` (Wilcoxon),
not the mean/paired-t that `projflat_spat_vs_temp_bymouse` defaults to — the
per-trial loss spans several decades (which is why these axes are logarithmic) and
its mean is carried by the top percent of trials. The terminal log always prints
BOTH p-values so the divergence is visible.

Outputs (PNG+SVG) under figures/projflat/.
Usage:  python diagnostics/projflat_trial_explorer.py
        python diagnostics/projflat_trial_explorer.py --run io_hmm_v3 \
            --cells pca_h8_lh0 --out-root figures/io_hmm_wide/spat_temp
            # ^ --cells is the ONE-OFF escape hatch. A recurring SET of cells belongs
            #   in projflat_cells.TABLES and is asked for with --configs; a literal
            #   list re-typed into a driver is exactly the drift this registry ended.
        python diagnostics/projflat_trial_explorer.py --configs io_hmm_proj \
            --by-mouse --out-root figures/io_hmm_wide/projection_configs

`--metric KL` exists (the projflat_v1 work judges under both metrics) but NO
documented command above uses it: the io_hmm_proj deliverable is PROJECTION-ONLY
(2026-08-28), and the KL-scored figures that used to sit beside these were deleted.

Titles carry two lines only — what the figure shows and how it is normalised. The
standing caveats (weighting, mean-vs-median, lambda_H being temporal-only) print
ONCE to the terminal at the start of a run; see `_preamble`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.stats as sstats
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import projflat_report as pr  # noqa: E402
from projflat_report import _res, _mice, have  # noqa: E402
# the per-trial paired test and its star mapping already live in the bar-chart
# sibling; imported rather than re-implemented so both deliverables report the
# SAME test for the same data
from projflat_spat_vs_temp_bymouse import _paired_p, _stars, _wrap  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nn_classifier import fit_loss_per_trial  # noqa: E402
from io_hmm_data import GRID_DEG_IO  # noqa: E402  (72-bin circular 2.5-deg grid)
# the colour metric: how far a trial's individual time bins sit from their own
# average, in total variation. ONE definition, in the row-wise-metrics module —
# scatter_spat_temp_by_mouse's `--colour l1` (where it started) now calls the same
# function, so the two figures colour by the identical number.
from decoder_metrics import bin_divergence_from_mean  # noqa: E402

import projflat_cells as pcells  # noqa: E402

# All nine headline cells, from the one shared table (was a local 4-cell literal
# that silently skipped rr8 and the KL anchors). Each figure's exemplars are picked
# from THAT cell's own scatter, so the nine figures show nine different trial sets —
# correct for "examples from this config's scatter", not for comparing decoders on a
# fixed trial.
CONFIGS = pcells.as_pairs()
MOUSE_CMAP = plt.get_cmap('tab10')

# --by-mouse colour: temporal-bin divergence from the trial's own temporal average.
# `viridis` is perceptually uniform and is what the sibling scatter
# (scatter_spat_temp_by_mouse, `--colour l1`) already uses for this same quantity.
DIV_CMAP = 'viridis'
# Per-panel colour limits are ROBUST percentiles, not min/max: a handful of
# sweeping trials (the rr8 cells reach TV ~0.8 while their median sits near 0.13)
# otherwise compress the whole cloud into the bottom fifth of the ramp. The
# colourbar is drawn with extend='both' so the clipping is visible, not hidden.
DIV_PCT = (2.0, 98.0)
# The Jensen identity `temp.decoded == temp.decoded_samp.mean(2)` is asserted per
# mouse before the colour is trusted. Measured max deviation over all twelve
# io_hmm_v3 projection cells x 6 mice (2026-08-28): 1.2e-07, i.e. float32
# round-trip. The tolerance is two decades above that, so a real change of
# convention (a weighted average, a renormalised mean) trips it.
JENSEN_TOL = 1e-5


def _theta(n):
    """Orientation grid for an n-bin posterior. 91 = the export runs' linear
    [0, 90] deg grid (the old hardcoded THETA); 72 = the IO-HMM circular
    2.5-deg grid [0, 177.5] (io_hmm_data.GRID_DEG_IO). Anything else is a
    support this script has never audited — refuse rather than mislabel."""
    if n == 91:
        return np.linspace(0, 90, 91)
    if n == len(GRID_DEG_IO):
        return np.asarray(GRID_DEG_IO, float)
    raise SystemExit(f'unrecognised posterior support ({n} bins): axis labels unknown')


def _axis_label(n):
    """Axis label for an n-bin posterior, SAYING whether the support wraps.

    The 72-bin IO-HMM grid is CIRCULAR and is drawn here on a straight axis, so
    the seam between the last bin and the first is invisible: a single unimodal
    posterior sitting on 175 deg renders as two humps, one at each end of the
    panel, and reads as bimodal. That is a reading trap, not a measurement error
    — the divergence metric and the projection loss are both geometry-free (see
    `decoder_metrics.bin_divergence_from_mean`) — so the fix belongs on the label,
    where it warns the eye, rather than in the numbers, which are already right.
    The 91-bin export grid is genuinely linear [0, 90] and gets no such note."""
    return ('orientation (deg; circular, 177.5 wraps to 0)' if n == len(GRID_DEG_IO)
            else 'orientation (deg)')


def _t(x):
    return torch.tensor(np.asarray(x, float))


def gather(results_root, cell, metric='PCA'):
    """Pooled per-trial normalised losses, the colour metric, an index back to
    (mouse, trial), and the Jensen-identity residual.

    `metric` is the per-trial scorer ('PCA' with the cell's own stored weighting,
    or 'KL'); the leave-one-out predict-mean divisor uses the same metric.

    `div` is the per-trial temporal-bin divergence from the trial's own temporal
    average (mean-over-bins TV, `decoder_metrics.bin_divergence_from_mean`) — the
    --by-mouse scatter colour. It is scored on the RAW posteriors, so unlike the
    axes it does not depend on the cell's projection weighting.

    Returns (sp, te, div, idx, jensen_err) where `jensen_err` is the worst
    max |decoded_samp.mean(bins) - decoded| over this cell's mice: the check that
    the average the divergence is measured against really is the decoder's own
    full-model posterior. Asserted, not merely reported."""
    r = _res(results_root, cell)
    sp, te, dv, idx = [], [], [], []
    jensen_err = 0.0
    for m in _mice(r):
        D = r[m]['Dist']
        pcs_t = evar_t = None
        if metric == 'PCA':
            pcs_t, evar_t = _t(D.get('pcs')), _t(D.get('explained_var'))
        tgt = np.asarray(D['spat']['target'], float)
        ds = np.asarray(D['spat']['decoded'], float)
        dt = np.asarray(D['temp']['decoded'], float)
        ok = np.isfinite(tgt).all(1) & np.isfinite(ds).all(1) & np.isfinite(dt).all(1)
        n = int(ok.sum()); tot = tgt[ok].sum(0)
        pm = np.tile((tot / n)[None, :], (tgt.shape[0], 1))
        pm[ok] = (tot[None, :] - tgt[ok]) / (n - 1)
        ch = np.nanmean(fit_loss_per_trial(_t(pm[ok]), _t(tgt[ok]), metric,
                                           pcs_t, evar_t).numpy())
        s = fit_loss_per_trial(_t(ds[ok]), _t(tgt[ok]), metric, pcs_t, evar_t).numpy() / ch
        t = fit_loss_per_trial(_t(dt[ok]), _t(tgt[ok]), metric, pcs_t, evar_t).numpy() / ch
        # colour: the per-bin posteriors against the average they collapse to.
        # `decoded` IS that average (the Jensen average), which is exactly what the
        # assert below checks — if it ever stopped being the plain mean, the colour
        # would be measuring divergence from something the figure does not plot.
        samp = np.asarray(D['temp'].get('decoded_samp'), float)
        if samp.ndim == 3 and samp.shape[2] > 1:
            err = float(np.nanmax(np.abs(samp.mean(2) - dt)))
            assert err < JENSEN_TOL, (
                f'{cell}/{m}: temp.decoded is not the mean over decoded_samp bins '
                f'(max |mean(bins) - decoded| = {err:.2e} > {JENSEN_TOL:g}); the '
                'divergence colour would not be measured against the plotted '
                'posterior')
            jensen_err = max(jensen_err, err)
            d = bin_divergence_from_mean(samp[ok])
        else:                              # no per-bin posteriors saved for this cell
            d = np.full(int(ok.sum()), np.nan)
        rows = np.where(ok)[0]
        sp.append(s); te.append(t); dv.append(d)
        idx += [(m, int(i)) for i in rows]
    return (np.concatenate(sp), np.concatenate(te), np.concatenate(dv), idx,
            jensen_err)


def _nearest(vals, target, pool=None):
    """Index of the trial whose value is closest to `target` (within `pool`)."""
    cand = np.arange(vals.size) if pool is None else np.asarray(pool)
    return int(cand[np.argmin(np.abs(vals[cand] - target))])


def pick(sp, te):
    """Four representative exemplars spanning the scatter.

    Chosen by PERCENTILE rather than absolute extreme: argmax/argmin lands on
    degenerate outliers (losses of ~0 or ~80) that illustrate nothing. Each pick is
    the trial nearest a chosen percentile, so it is typical of that region.
    """
    eps = 1e-12
    ls, lt = np.log(sp + eps), np.log(te + eps)
    d, m = lt - ls, lt + ls                     # who wins / how hard overall
    mid = np.where(np.abs(d) < np.percentile(np.abs(d), 40))[0]   # near-diagonal
    return [('both good',          _nearest(m, np.percentile(m, 12), mid)),
            ('spatial better',     _nearest(d, np.percentile(d, 97))),
            ('temporal better',    _nearest(d, np.percentile(d, 3))),
            ('both poor',          _nearest(m, np.percentile(m, 93), mid))]



# --------------------------------------------------------------- exemplar panels
def _draw_exemplars(fig, axes_pairs, r, idx, sp, te, picks, val_fmt='{:.2f}'):
    """Draw the four exemplar columns into the (top, bottom) axes pairs given.

    ONE body, two call sites: the pooled figure builds its pairs from its own
    GridSpec, the --by-mouse exemplar figure from a plain 2 x 4 grid. Top panel =
    IO target + the two decoded posteriors; bottom = the temporal decoder's
    per-time-bin posteriors (`decoded_samp`) before the Jensen average.

    `val_fmt` formats the two loss values in each column title. It stays at the
    historical two decimals for the pooled figure (whose output is unchanged) —
    but two decimals prints a genuinely good trial as 'spat 0.00' (seen on
    pca_rr8_lh0, where an exemplar scores 1.6e-3), so --by-mouse passes '{:.3g}'."""
    for j, (lab, i) in enumerate(picks):
        m, tr = idx[i]
        D = r[m]['Dist']
        tgt = np.asarray(D['spat']['target'], float)[tr]
        ds = np.asarray(D['spat']['decoded'], float)[tr]
        dt = np.asarray(D['temp']['decoded'], float)[tr]
        samp = np.asarray(D['temp']['decoded_samp'], float)
        theta = _theta(tgt.size)

        axt, axh = axes_pairs[j]
        axt.fill_between(theta, 0, tgt, color='0.75', lw=0,
                         label='IO target' if j == 0 else None)
        axt.plot(theta, ds, color=ps.SPATIAL, lw=1.3,
                 label='spatial' if j == 0 else None)
        axt.plot(theta, dt, color=ps.TEMPORAL, lw=1.3,
                 label='temporal' if j == 0 else None)
        axt.set_yticks([])
        axt.set_xlabel(_axis_label(tgt.size), fontsize=7)
        axt.set_title(f'{j + 1}. {lab}\n{m}, trial {tr}\n'
                      f'spat {val_fmt.format(sp[i])} · '
                      f'temp {val_fmt.format(te[i])}', fontsize=7)
        if j == 0:
            axt.legend(fontsize=5.5, frameon=True)

        if samp.ndim == 3 and samp.shape[0] > tr:
            bins = samp[tr]
            # interpolation='nearest' is REQUIRED, not cosmetic: matplotlib's
            # rcParam default is 'antialiased', which resamples the image through a
            # smoothing filter whenever it is minified on screen. `bins` is only
            # (91 orientations x n_bins) and n_bins is ~10, so the default silently
            # blurs the per-bin posteriors — exactly the structure this panel exists
            # to show. 'nearest' draws one rectangle per (orientation, bin) cell, so
            # what you see is the raw decoded value.
            im = axh.imshow(bins, aspect='auto', origin='lower', cmap='magma',
                            interpolation='nearest',
                            vmin=0, vmax=np.nanpercentile(bins, 99.5),
                            extent=[0, bins.shape[1], theta[0], theta[-1]])
            axh.plot(bins.shape[1] * tgt / max(tgt.max(), 1e-12) * 0.28,
                     theta, color='w', lw=1.1, alpha=0.9)
            fig.colorbar(im, ax=axh, fraction=0.045, pad=0.03).ax.tick_params(labelsize=5)
        axh.set_xlabel('time bin', fontsize=7)
        if j == 0:
            axh.set_ylabel(_axis_label(tgt.size), fontsize=6.5)
        else:
            axh.set_yticklabels([])   # the neighbouring colourbar owns that gutter
        axh.set_title('temporal bins (white = target)', fontsize=6.5)


def _load(results_root, cell, metric):
    """(sp, te, div, idx, picks, r, ring, mlab) or None if the cell is missing.

    The pooled figure and the --by-mouse pair open the SAME data the same way and
    pick the SAME exemplars; this is that shared preamble, so the two entry points
    cannot drift into scoring or picking differently. The Jensen-identity residual
    is printed here (once per cell) rather than drawn on the figure — it is a
    check, not a result."""
    if not have(results_root, cell):
        print(f'  [skip] {cell}')
        return None
    sp, te, div, idx, jerr = gather(results_root, cell, metric)
    print(f'  {cell}: Jensen check max|decoded_samp.mean(bins) - decoded| '
          f'= {jerr:.2e} (tol {JENSEN_TOL:g})')
    return (sp, te, div, idx, pick(sp, te), _res(results_root, cell),
            ps.color(metric), ps.loss_label(metric, short=True))


def make_figure(results_root, out_root, title, cell, metric='PCA'):
    loaded = _load(results_root, cell, metric)
    if loaded is None:
        return
    # `div` (the temporal-bin divergence) is deliberately UNUSED here: the pooled
    # scatter colours by MOUSE and carries a mouse legend, and one colour channel
    # cannot show both. The divergence colour is a --by-mouse feature, where each
    # panel is already a single mouse.
    sp, te, _div, idx, picks, r, ring, mlab = loaded

    ps.apply()
    fig = plt.figure(figsize=(15.5, 6.4))
    gs = fig.add_gridspec(2, 6, width_ratios=[1.55, 0.12, 1, 1, 1, 1],
                          hspace=0.42, wspace=0.32)

    # ---- scatter ----
    ax = fig.add_subplot(gs[:, 0])
    mm = np.array([m for m, _ in idx])
    mice_u = sorted(set(mm))
    mcol = {m: MOUSE_CMAP(k % 10) for k, m in enumerate(mice_u)}
    ax.scatter(sp, te, s=7, c=[mcol[m] for m in mm], alpha=0.45,
               edgecolor='none', zorder=1)
    lim = [min(sp.min(), te.min()) * 0.7, max(sp.max(), te.max()) * 1.4]
    ax.plot(lim, lim, color='0.3', ls=':', lw=1.4, zorder=2)
    ax.axhline(1.0, color='k', ls='--', lw=0.9, alpha=0.6)
    ax.axvline(1.0, color='k', ls='--', lw=0.9, alpha=0.6)
    for j, (lab, i) in enumerate(picks, 1):
        ax.plot(sp[i], te[i], 'o', ms=13, mfc='none', mec=ring, mew=2, zorder=5)
        ax.annotate(str(j), (sp[i], te[i]), fontsize=9, fontweight='bold',
                    color=ring, xytext=(9, 5), textcoords='offset points')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f'spatial {mlab} loss ÷ predict-mean', fontsize=8)
    ax.set_ylabel(f'temporal {mlab} loss ÷ predict-mean', fontsize=8)
    frac = float((te < sp).mean())
    # Panel title = DATA only (n, how many trials fall below the identity). The
    # config name is in the suptitle and does not need repeating here, and the
    # "dashed = chance" line went with it: both axes are divided by the
    # predict-mean, which the suptitle and the axis labels already say.
    ax.set_title(f'pooled over {len(mice_u)} mice, n={sp.size} trials\n'
                 f'below the identity = temporal lower ({frac:.0%} of trials)',
                 fontsize=8)
    ax.legend(handles=[Line2D([0], [0], marker='o', ls='none', ms=4,
                              color=mcol[m], label=f'{m} (n={int((mm == m).sum())})')
                       for m in mice_u],
              fontsize=5.5, frameon=True, loc='upper left', handletextpad=0.3)
    # paired test on per-mouse mean normalised losses (same stat family as
    # spat_temp_best_cell's across-animals panel)
    sp_m = np.array([sp[mm == m].mean() for m in mice_u])
    te_m = np.array([te[mm == m].mean() for m in mice_u])
    tt, pp = sstats.ttest_rel(te_m, sp_m)
    ax.text(0.97, 0.03,
            f'per-mouse means (n={len(mice_u)}):\nt({len(mice_u) - 1})={tt:+.2f}, p={pp:.3f}',
            transform=ax.transAxes, fontsize=6, va='bottom', ha='right',
            bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))

    _draw_exemplars(fig, [(fig.add_subplot(gs[0, 2 + j]), fig.add_subplot(gs[1, 2 + j]))
                          for j in range(len(picks))],
                    r, idx, sp, te, picks)
    # TWO LINES, the same contract as the --by-mouse figures: what the figure shows,
    # then the normalisation. `title` is FLATTENED because the shared table's labels
    # are stacked for bar-chart tick labels ('linear (0 hidden)\nflat-weighting') —
    # left in, that newline made this a three-line suptitle on every projflat_v1
    # default figure. The standing caveats print once in `_preamble`.
    nb = int(np.asarray(r[mice_u[0]]['Dist']['spat']['target'], float).shape[1])
    fig.suptitle(_wrap(
        f"{title.replace(chr(10), ', ')} — spatial vs temporal per trial, pooled, "
        'with four exemplar trials ringed\n'
        f"{_metric_note(cell, metric, nb)} ÷ that mouse's predict-mean; bottom row = "
        'the temporal decoder\'s individual 100 ms bin posteriors, before the Jensen '
        'average', fig.get_size_inches()[0], 8.5), y=1.03, fontsize=8.5)
    stem = _stem('trials', cell, metric)
    ps.save_fig(fig, Path(out_root), stem)
    print(f'  {stem}: ' + ' | '.join(
        f'{j+1}.{lab} {idx[i][0]}/{idx[i][1]} s={sp[i]:.2f} t={te[i]:.2f}'
        for j, (lab, i) in enumerate(picks)))



def _stem(kind, cell, metric):
    """Figure basename. The metric suffix is appended ONLY for a non-default metric,
    so the historical PCA filenames are unchanged — but `--metric KL` no longer
    silently overwrites the PCA figure of the same cell (it used to: the stem
    carried no metric tag)."""
    base = (cell.replace('_l0_d0_w0', '').replace('_raw', '')
            if pr.RUN == 'projflat_v1' else cell)
    pre = 'projflat' if pr.RUN == 'projflat_v1' else pr.RUN
    return f'{pre}_{kind}_{base}' + ('' if metric == 'PCA' else f'_{metric}')


def _metric_note(cell, metric, nbins=None):
    """How this cell's per-trial loss is actually computed, spelled out. Under the
    projection metric the weights are the cell's OWN stored ones (uniform for a flat
    cell, the eigenvalue spectrum for an evar cell), so loss values are on the same
    scale WITHIN a figure but not across the flat/evar figures.

    `nbins` is READ FROM THE DATA, not assumed: this script serves both supports
    (91 on the export grid, 72 on the IO-HMM one) and the flat clause used to say
    '72 bins' unconditionally, which printed the wrong bin count on every
    projflat_v1 figure. Omitted -> the count is left out rather than guessed.

    Kept SHORT (no ', per trial' tail — the line above it already says per trial):
    it is the first half of a two-line title, and the flat phrasing is the longer
    of the two, so it is what decides whether that line wraps to a third."""
    if metric != 'PCA':
        return f'{ps.loss_label(metric)} loss'
    w = pcells.weighting_of(cell)
    over = f'over {nbins} bins' if nbins else 'over bins'
    wl = (f'flat weights = MSE {over}' if w == 'flat'
          else 'eigenvalue-weighted' if w == 'evar' else 'stored weighting')
    return f'{ps.loss_label(metric)} loss ({wl})'


def _preamble(metric, stat, by_mouse):
    """The STANDING caveats — printed ONCE to the terminal, not onto every figure.

    These clauses used to be the suptitle of every panel grid, where they ran to
    five or six wrapped lines and dominated the frame (2026-08-28, Theo's request).
    They are properties of the RUN, identical on every config, so the terminal is
    their place. What stays on a figure is what it shows and how it is normalised;
    what stays IN a panel — n, the medians, the fraction below the diagonal, the
    test — is data, not caveat, and stays put.

    The four standing clauses, each of which stops a naive cross-figure reading:
      * normalisation. Both axes are divided by that mouse's leave-one-out
        predict-mean, so 1 = chance and the two axes are on one scale.
      * weighting. Under the projection metric each cell is scored with its OWN
        stored weights — the SAME convention the spatial-vs-temporal bar figures
        are delivered under (`--weighting own`, their default). Its price is that
        a flat figure is not comparable with an evar one, and the size of that
        price is measured: rescoring under one common evar anchor basis
        (`--weighting common`, a different question, not the delivered deck) moves
        pcaflat_rr8_lh0 / mouse_0's spatial loss 2.371 -> 1.008. The sentence
        itself comes from `projflat_cells.WEIGHT_NOTE`, so this script, the bar
        driver and the by-state driver say it in one wording rather than three —
        the third wording is how this clause previously drifted into claiming the
        bars used the common basis, which they have not since 2026-08-28.
        (KL is basis-free, so the clause does not apply to a KL figure.)
      * summary. These panels report a MEDIAN over trials; the bar figures report a
        MEAN. On the same cell and mouse that is 0.36 vs 0.84 — the heavy per-trial
        tail (which is why the axes are logarithmic), not a different result.
      * lambda_H. It penalises the TEMPORAL decoder only, so two cells differing
        only in lambda_H share their SPATIAL arrays bit-for-bit: within such a pair
        the x axis is a replicate and only the y axis can move.
    Spatial vs temporal WITHIN a panel is untouched by all of this: both share the
    weighting and the test is paired over the same trials."""
    test = 'Wilcoxon' if stat == 'median' else 'paired t'
    out = ['Standing notes for this run (deliberately NOT repeated on every figure):',
           f'  * normalisation: {ps.loss_label(metric)} loss per trial, each axis '
           "divided by THAT mouse's leave-one-out predict-mean, so 1 = chance."]
    if metric == 'PCA':
        # NB no bin count here. This preamble prints BEFORE any cell is loaded, and
        # this script serves both supports (91 bins on the export grid, 72 on the
        # IO-HMM one) — the same hardcoded '72' that was wrong in `_metric_note` was
        # still wrong here. Each figure's own title carries the measured count.
        # ONE wording, from the shared registry — the bar driver and the by-state
        # driver print this same string. A local paraphrase is what let this clause
        # go on claiming the bars used a common evar basis after they stopped.
        out.append('  * weighting: ' + pcells.WEIGHT_NOTE['own']
                   + ' Each figure title states its own bin count. The cost of that '
                     'choice is measured: rescoring under the one common evar anchor '
                     'basis instead moves pcaflat_rr8_lh0 / mouse_0 from 2.371 to '
                     '1.008.')
    if by_mouse:
        out.append(f'  * summary: each panel reports a {stat.upper()} over its trials '
                   f'({test}); the bar figures report a mean. Same data — the per-trial '
                   'loss spans several decades and its mean is carried by the top '
                   'percent (0.36 vs 0.84 on that same cell and mouse).')
    out.append('  * lambda_H is TEMPORAL-only: cells differing only in lambda_H share '
               'their SPATIAL arrays bit-for-bit, so within such a pair the x axis is '
               'a replicate and only the y axis can move.')
    print('\n'.join(out) + '\n')


# ------------------------------------------------------- (--by-mouse) two figures
def _div_norm(d):
    """Per-panel colour normalisation for the divergence, or None if unusable.

    PER PANEL is the point (Theo, 2026-08-28): each mouse's bins disagree by its
    own amount, and a shared scale buries the within-animal structure under the
    between-animal offset. The price is that a colour means a different TV in
    every panel — which is why the colourbars keep their tick labels and the
    suptitle says so."""
    d = np.asarray(d, float)
    if not np.isfinite(d).any():
        return None
    lo, hi = np.nanpercentile(d, DIV_PCT)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return plt.Normalize(vmin=float(lo), vmax=float(hi))


def _div_readout(s_m, t_m, d):
    """Two short lines: does the divergence track WHICH SIDE of the identity a
    trial falls on? Median TV split by side (with each subgroup's n), plus the
    rank correlation with the signed distance from the diagonal,
    log(temporal) - log(spatial). The colour is only decoration without it."""
    ok = np.isfinite(d) & np.isfinite(s_m) & np.isfinite(t_m) & (s_m > 0) & (t_m > 0)
    if ok.sum() < 10:
        return None
    d, s_m, t_m = d[ok], s_m[ok], t_m[ok]
    below = t_m < s_m                       # temporal lower = below the identity
    if below.sum() < 3 or (~below).sum() < 3:
        return None
    rho, pv = sstats.spearmanr(d, np.log(t_m) - np.log(s_m))
    return (f'TV median: below diag {np.median(d[below]):.3f} (n={int(below.sum())})'
            f' · above {np.median(d[~below]):.3f} (n={int((~below).sum())})\n'
            f'Spearman \u03c1(TV, log temp/spat) = {rho:+.2f}, p = {pv:.1e}')


def fig_by_mouse(results_root, out_root, title, cell, metric='PCA', stat='mean'):
    """One scatter panel PER MOUSE, plus the exemplar block as its own figure.

    The pooled figure (`make_figure`) hides the between-animal structure: every
    mouse is normalised to its own predict-mean, so the six clouds are separately
    interpretable and the pooled cloud is a mixture. Here each mouse gets its own
    panel with its own identity line, chance lines, trial count and paired test over
    ITS trials. Axis limits are SHARED across panels so the mice are comparable.

    Exemplars are still picked from the pooled scatter by the unchanged `pick()`,
    each ringed in the panel of the mouse it came from; the exemplar columns go to a
    separate figure so neither grid is cramped (six scatters plus eight exemplar
    panels in one frame breaches the 1600 px PNG cap at a legible panel size)."""
    loaded = _load(results_root, cell, metric)
    if loaded is None:
        return
    sp, te, div, idx, picks, r, ring, mlab = loaded
    test = 'Wilcoxon' if stat == 'median' else 'paired t'

    mm = np.array([m for m, _ in idx])
    mice_u = sorted(set(mm))
    # NB no per-mouse colour dict here any more: each panel IS one mouse, so the
    # colour channel is spent on the temporal-bin divergence instead (the pooled
    # `make_figure`, where mouse identity is the only way to read the mixture,
    # still uses MOUSE_CMAP).
    lim = [min(sp.min(), te.min()) * 0.7, max(sp.max(), te.max()) * 1.4]
    # posterior support READ FROM THE DATA (72 IO-HMM / 91 export), for the title's
    # flat-weighting clause — see `_metric_note`
    nbins = int(np.asarray(r[mice_u[0]]['Dist']['spat']['target'], float).shape[1])
    owner = {}                                   # exemplar number -> its mouse
    for j, (_, i) in enumerate(picks, 1):
        owner.setdefault(idx[i][0], []).append((j, i))

    # ---- figure 1: one scatter per mouse ----
    ps.apply()
    ncol = min(3, len(mice_u))
    nrow = int(np.ceil(len(mice_u) / ncol))
    # panel_w up from 3.0 to pay for the per-panel colourbar, which otherwise eats
    # the axes width (the panels are aspect='equal', so a narrower axes is a
    # smaller square, not a squashed one). The extra width also buys the suptitle
    # its character budget back — see `sup_fs` below.
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=ps.figsize(ncol, nrow, panel_w=3.4, panel_h=3.0),
                             constrained_layout=True)
    for k, m in enumerate(mice_u):
        ax = axes[k // ncol][k % ncol]
        sel = mm == m
        s_m, t_m, d_m = sp[sel], te[sel], div[sel]
        # Colour = how far this trial's individual time bins sit from the temporal
        # average that the y axis actually scores. Points are NOT re-ordered by
        # colour: drawing high-divergence trials last would put them on top of the
        # cloud and overstate how common they are.
        norm = _div_norm(d_m)
        if norm is None:                   # no per-bin posteriors saved for this cell
            ax.scatter(s_m, t_m, s=8, color='0.45', alpha=0.45,
                       edgecolor='none', zorder=1)
        else:
            sc = ax.scatter(s_m, t_m, s=9, c=d_m, cmap=DIV_CMAP, norm=norm,
                            alpha=0.85, linewidths=0.15, edgecolor='0.25', zorder=1)
            # extend='both': the limits are the 2nd/98th percentiles, so the
            # arrowheads are the honest statement that the tails are clipped
            cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, shrink=0.85,
                              aspect=26, extend='both')
            cb.set_label('temporal-bin divergence\nfrom the average (TV)', fontsize=6)
            cb.ax.tick_params(labelsize=5.5, length=2)
            cb.outline.set_linewidth(0.4)
        ax.plot(lim, lim, color='0.3', ls=':', lw=1.4, zorder=2)
        ax.axhline(1.0, color='k', ls='--', lw=0.9, alpha=0.6, zorder=2)
        ax.axvline(1.0, color='k', ls='--', lw=0.9, alpha=0.6, zorder=2)
        for j, i in owner.get(m, []):
            ax.plot(sp[i], te[i], 'o', ms=13, mfc='none', mec=ring, mew=2, zorder=5)
            # white halo: the exemplar numbers land inside the densest part of the
            # cloud, where bare coloured text is unreadable
            ax.annotate(str(j), (sp[i], te[i]), fontsize=10, fontweight='bold',
                        color=ring, xytext=(10, 5), textcoords='offset points',
                        zorder=6, annotation_clip=False,
                        path_effects=[pe.withStroke(linewidth=2.4, foreground='w')])
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal', adjustable='box')
        # seaborn's 'ticks' style draws full-length minor ticks; on a 4-decade log
        # axis that renders as a solid black band along the spine
        ax.tick_params(which='minor', length=0)
        pv = _paired_p(s_m, t_m, stat)
        frac = float((t_m < s_m).mean())
        summ = (np.median if stat == 'median' else np.mean)
        ax.set_title(f'{m}  (n = {int(sel.sum())} trials)\n'
                     f'{stat} spatial {summ(s_m):.2f} · temporal {summ(t_m):.2f}; '
                     f'temporal lower in {frac:.0%}\n'
                     f'{test} over trials: p = {pv:.1e} {_stars(pv)}', fontsize=7.5)
        # the colour, as a number: whether the divergence tracks the side of the
        # identity. In the panel (not the title, which is full) and not a prose
        # conclusion — two statistics with their subgroup n's.
        read = _div_readout(s_m, t_m, d_m)
        if read:
            ax.text(0.97, 0.03, read, transform=ax.transAxes, fontsize=5.5,
                    va='bottom', ha='right', zorder=7,
                    bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))
        if k // ncol == nrow - 1:
            ax.set_xlabel(f'spatial {mlab} loss ÷ predict-mean', fontsize=8)
        if k % ncol == 0:
            ax.set_ylabel(f'temporal {mlab} loss ÷ predict-mean', fontsize=8)
    for k in range(len(mice_u), nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    # TWO short lines: what the figure shows, then the normalisation. The standing
    # caveats that used to follow them (weighting, mean-vs-median, lambda_H being
    # temporal-only) now print once per run in `_preamble` — they were identical on
    # all twelve figures and left the panels a strip under a block of prose.
    # Still wrapped to the FIGURE's own width with the sibling's `_wrap` (imported,
    # not re-implemented): a long config label can still overrun the 3x2 grid, and
    # `bbox_inches='tight'` then grows the canvas to fit the title, capping the
    # PNG's dpi.
    # `title` flattened for the same reason as in `make_figure`: a projflat_v1
    # default run takes its labels from the stacked bar-chart table, and the
    # embedded newline turns a two-line contract into three.
    # TWO lines, still: line 1 says what the figure shows AND that the colour is
    # normalised per panel (the clause that stops a naive between-mouse reading of
    # the colours); line 2, how the axes are normalised and what the colour is.
    # `sup_fs` is 8.0 rather than the 8.5 the pooled figure uses, and that is what
    # keeps this to two lines: `_wrap`'s budget is ~0.94 * width * 72 / (0.62 * fs)
    # characters, and the longest of the twelve io_hmm_proj labels
    # ('reduced-rank 8 (linear), variance-weighting, lambda_H 1e-4', 52 chars) plus
    # the longest weighting note (54) overruns a 149-character line. Checked over
    # all twelve labels x both notes: worst line 159 of 165.
    sup_fs = 8.0
    fig.suptitle(_wrap(
        f"{title.replace(chr(10), ', ')} — spatial vs temporal per trial, one panel "
        'per mouse; PER-PANEL colour scale, not comparable between mice\n'
        + f"{_metric_note(cell, metric, nbins)} ÷ that mouse's predict-mean; below "
          'the identity = temporal lower; colour = temporal-bin divergence (TV)',
        fig.get_size_inches()[0], sup_fs), fontsize=sup_fs)
    stem = _stem('trialsbymouse', cell, metric)
    ps.save_fig(fig, Path(out_root), stem)
    print(f'  {stem}: ' + ' | '.join(
        f'{m} n={int((mm == m).sum())} '
        f'W p={_paired_p(sp[mm == m], te[mm == m], "median"):.1e} '
        f'/ t p={_paired_p(sp[mm == m], te[mm == m], "mean"):.1e} '
        f'TV med={np.nanmedian(div[mm == m]):.3f} '
        f'[{np.nanmin(div[mm == m]):.3f}, {np.nanmax(div[mm == m]):.3f}]'
        for m in mice_u))

    # ---- figure 2: the exemplar block, same picks, rings numbered as above ----
    ps.apply()
    fig2, axes2 = plt.subplots(2, len(picks), squeeze=False,
                               figsize=ps.figsize(len(picks), 2, panel_w=2.9,
                                                  panel_h=2.5),
                               constrained_layout=True)
    _draw_exemplars(fig2, [(axes2[0][j], axes2[1][j]) for j in range(len(picks))],
                    r, idx, sp, te, picks, val_fmt='{:.3g}')
    # Two lines here too; the per-column titles already carry the mouse, the trial
    # and both loss values, and the companion scatter is the same stem with
    # `trialsbymouse` in place of `trialexemplars` (printed below).
    fig2.suptitle(_wrap(
        f"{title.replace(chr(10), ', ')} — the four ringed exemplar trials from the "
        'per-mouse scatters, one per column\n'
        'Top: IO target with both decoded posteriors. Bottom: the temporal decoder\'s '
        'individual 100 ms bin posteriors, before the Jensen average',
        fig2.get_size_inches()[0], 8.5), fontsize=8.5)
    stem2 = _stem('trialexemplars', cell, metric)
    ps.save_fig(fig2, Path(out_root), stem2)
    print(f'  {stem2} (rings drawn in {stem}): ' + ' | '.join(
        f'{j+1}.{lab} {idx[i][0]}/{idx[i][1]} s={sp[i]:.2f} t={te[i]:.2f}'
        for j, (lab, i) in enumerate(picks)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/projflat')
    ap.add_argument('--run', default=None,
                    help='results run dir (default: projflat_v1 via projflat_report). '
                         'Cell dirs must hold one slug with stratified_balanced.mat, '
                         'e.g. io_hmm_v3.')
    ap.add_argument('--cells', nargs='+', default=None, metavar='CELL',
                    help='cell dir names under --run. Overrides --configs; prefer '
                         '--configs so the cell list stays in projflat_cells.')
    ap.add_argument('--configs', default=None, choices=list(pcells.TABLES),
                    help='a named cell table from projflat_cells.TABLES (each pins '
                         'its own run dir, used unless --run overrides it). Default: '
                         'the nine projflat headline cells.')
    ap.add_argument('--metric', default='PCA', choices=['PCA', 'KL'],
                    help="per-trial scoring metric; 'PCA' uses the cell's own stored "
                         'weighting. The predict-mean divisor matches the metric.')
    ap.add_argument('--by-mouse', action='store_true', dest='by_mouse',
                    help='one scatter panel PER MOUSE (plus the exemplar columns as '
                         'their own figure) instead of the pooled single scatter. '
                         'The pooled default is unchanged.')
    ap.add_argument('--trial-stat', choices=['mean', 'median'], default='median',
                    dest='trial_stat',
                    help='--by-mouse only: how each panel summarises its trials and '
                         "which paired test it reports ('median' -> Wilcoxon, "
                         "'mean' -> paired t). Defaults to median here (NOT to the "
                         'mean that projflat_spat_vs_temp_bymouse defaults to): the '
                         'per-trial loss spans several decades, which is why these '
                         'axes are logarithmic, and its mean is carried by the top '
                         'percent of trials.')
    a = ap.parse_args()
    # the standing caveats, once, before any figure — see `_preamble`
    _preamble(a.metric, a.trial_stat, a.by_mouse)
    tbl = pcells.table(a.configs) if a.configs else None
    # a table pins the run its cells live in; --run still wins if given explicitly
    run = a.run or (tbl['run'] if tbl else None)
    if run:
        pr.RUN = run  # projflat_report's loader keys the run off this module global
    if a.cells:
        # labels come from the shared registry when it knows the cell, so --cells
        # figures are titled like the table-driven ones instead of '<run> <cell>'
        configs = [((pcells.cell_label(c) if pcells.cell_label(c) != c
                     else f'{pr.RUN} {c}'), c) for c in a.cells]
    elif tbl:
        # the ONE-LINE label: a table's own label is stacked for bar-chart tick
        # labels, and three stacked lines ahead of this figure's four-line caption
        # is all header and no plot
        configs = [(pcells.cell_label(c, tbl['rows']), c) for _, c, _ in tbl['rows']]
    else:
        configs = CONFIGS
    for title, cell in configs:
        if a.by_mouse:
            fig_by_mouse(a.results_root, a.out_root, title, cell, metric=a.metric,
                         stat=a.trial_stat)
        else:
            make_figure(a.results_root, a.out_root, title, cell, metric=a.metric)


if __name__ == '__main__':
    main()
