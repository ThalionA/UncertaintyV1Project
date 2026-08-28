# -*- coding: utf-8 -*-
"""Shared visual language for the PCA-peakiness figure suite.

One palette, one save helper, one style call — so the ~26 figures that tell the
"why does the PCA loss go peaky" story cohere and build on each other instead of
each drifting its own colours, fonts and title conventions. Imported by every
``diagnostics/*`` plotter and the top-level ``compare_loss_variants`` /
``weight_evolution_variants`` scripts.

Semantics — WARM = the pathological decoders, COOL = the calibrated ones / fixes:
  * PCA (evar)   -> warm orange, the villain, identical in every figure
  * Wasserstein  -> warm red, the *other* loss whose basin drifts peaky
  * flat-evar    -> blue        (the unweighted-L2 control / fix)
  * PCA + shape  -> green       (the width-matched fix; a green ramp for the lam sweep)
  * KL / CE / JS -> purple / teal / steel-blue  (the calibrated divergences)
A reader flipping through sees the same orange villain and the same cool fixes
throughout, so the argument reads off the colours alone.

Conventions this module encodes (apply once per figure):
  * single-line, normal-weight titles — the *argument* lives in the note prose,
    not in two-line figure headers;
  * the IO target is ALWAYS a grey band (`target_band`) or black dashed line
    (`target_line`); chance is a dotted grey line (`chance_line`);
  * PNG is capped so the longest side stays <=1600 px (previewable), SVG keeps
    full vector detail — see `save_fig`.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

# The PNG+SVG/≤1600px save contract lives in figsave (dependency-light, no cycle);
# re-exported here so the ~11 callers that do `from peakiness_style import save_fig`
# are unchanged.
from figsave import save_fig  # noqa: F401

# ----------------------------------------------------------------------
# Canonical colours
# ----------------------------------------------------------------------
PCA_EVAR    = '#e6550d'   # warm orange  — the pathology, in EVERY figure
WASSERSTEIN = '#d7301f'   # warm red     — the other loss that drifts peaky
FLAT_EVAR   = '#2171b5'   # blue         — unweighted-L2 / Brier control & fix
SHAPE       = '#238b45'   # green        — the width-matched PCA+shape fix
KL          = '#6a51a3'   # purple       — calibrated divergence
CE          = '#35978f'   # teal         — calibrated divergence
JS          = '#4292c6'   # steel-blue   — calibrated divergence

TARGET_GREY = '0.75'      # IO-target band fill
TARGET_LINE = 'k'         # IO-target reference line
CHANCE_GREY = '0.5'       # uniform / chance reference line

# architecture colours (spatial vs temporal decoder), distinct from the loss
# palette so arch-comparison figures read consistently across the suite
SPATIAL  = '#e08214'      # amber  — spatial decoder
TEMPORAL = '#542788'      # violet — temporal decoder
ARCH = {'spat': SPATIAL, 'temp': TEMPORAL,
        'spatial': SPATIAL, 'temporal': TEMPORAL}

# consistent panel geometry: every panel is the SAME physical size across
# figures, so a 1x4 and a 2x3 figure are equally legible in a PDF.
PANEL_W, PANEL_H = 3.4, 2.8   # inches per panel (excluding shared margins)

# green ramp for the shape-lambda sweep (light -> dark as lambda grows)
SHAPE_GREENS = ['#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32']

# every key-string the scripts use, mapped to a canonical colour
_EXACT = {
    'PCA': PCA_EVAR, 'PCA (evar-weighted)': PCA_EVAR, 'PCA evar': PCA_EVAR,
    'PCA (evar)': PCA_EVAR, 'PCA_evar': PCA_EVAR,
    'PCA-flat': FLAT_EVAR, 'flat L2 (Brier)': FLAT_EVAR, 'flat-evar': FLAT_EVAR,
    'flat-evar (Brier)': FLAT_EVAR, 'flat L2': FLAT_EVAR, 'flat_evar': FLAT_EVAR,
    'KL': KL, 'CE': CE, 'JS': JS, 'Wasserstein': WASSERSTEIN,
}


def color(name):
    """Canonical colour for a loss/variant label, tolerant of the many aliases
    the scripts use. Any 'PCA+shape'/'PCA + shape' label resolves to the green."""
    if name in _EXACT:
        return _EXACT[name]
    key = name.replace('+', ' + ')
    if 'shape' in name.lower():
        return SHAPE
    return _EXACT.get(key, '0.3')


# ----------------------------------------------------------------------
# Display labels — the code/.mat keys stay 'PCA' etc. (data access unchanged),
# but figures show these friendlier names. The 'PCA' loss is a projection-based
# loss (weighted L2 in the PCA-projection subspace), so it reads 'Projection-
# based' rather than the method name; the divergences keep their standard acronyms.
# ----------------------------------------------------------------------
LOSS_LABEL = {
    'PCA': 'Projection-based',
    'CE': 'CE', 'KL': 'KL', 'JS': 'JS', 'Wasserstein': 'Wasserstein',
}


def loss_label(key, short=False):
    """Figure display name for a loss key (code/.mat keys stay 'PCA' etc.).
    ``short=True`` returns the compact 'Projection' for tight tick labels."""
    if key == 'PCA':
        return 'Projection' if short else 'Projection-based'
    return LOSS_LABEL.get(key, key)


def loss_labels(keys, short=False):
    """Map a sequence of loss keys to display labels (tick labels / legends)."""
    return [loss_label(k, short=short) for k in keys]


# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------
def apply():
    """Project style + decluttering defaults. Call once at the top of a plot
    script (replaces the old ``dpu.set_style()`` call). Keeps seaborn's ticks
    look but tames the 'talk' context that made two-line titles collide, and
    drops the top/right spines for a cleaner frame."""
    import decoder_plotting_utils as dpu
    dpu.set_style()
    plt.rcParams.update({
        'axes.titlesize': 11.5, 'axes.titleweight': 'normal', 'axes.titlepad': 6,
        'figure.titlesize': 12.5, 'figure.titleweight': 'normal',
        'axes.labelsize': 10.5,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 8.5, 'legend.frameon': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        # Never silently smooth an imshow. matplotlib's default is 'antialiased',
        # which for a small array (our per-bin posteriors are 91 orientations x ~10
        # time bins) resolves to a 'hanning' resampling filter whenever the panel
        # renders shorter than 3x the array's rows (~273 px for 91 rows) — i.e. the
        # heatmap gets blurred along the orientation axis, destroying exactly the
        # per-bin structure those panels exist to show, with no warning and no
        # visible cue. Whether it happened depended on panel height in pixels, so
        # some figures were smoothed and others weren't. Pinning it here fixes every
        # present and future imshow site at once (2026-08-12).
        'image.interpolation': 'nearest',
    })


# ----------------------------------------------------------------------
# Shared primitives
# ----------------------------------------------------------------------
def target_band(ax, x, t, label='IO target'):
    """Draw the IO target as a filled grey band (the canonical 'truth' motif)."""
    return ax.fill_between(x, t, color=TARGET_GREY, alpha=0.8, lw=0, label=label)


def target_line(ax, y, label=None, value_fmt='{:.3f}'):
    """Horizontal black-dashed IO-target reference (for peakiness / normalised-loss
    axes). House rule: never write "skill" on an axis, a title or a printout — say
    "normalised loss (/ predict-mean)"."""
    lab = label if label is not None else f'IO target ({value_fmt.format(y)})'
    return ax.axhline(y, ls='--', lw=1.5, color=TARGET_LINE, label=lab)


def chance_line(ax, y=1.0, label='chance'):
    """Dotted grey reference for the shuffle/uniform/chance level."""
    return ax.axhline(y, ls=':', lw=1.2, color=CHANCE_GREY, label=label)


def panel_label(ax, letter, dx=-0.5, dy=0.28):
    """Bold panel tag (a, b, c…) at the top-left of an axes, placed in offset
    points so it sits clear of the y-axis label and is identical across figures.
    `dx`/`dy` are in fontsize-fractions of nudge if a panel needs a tweak."""
    ax.annotate(letter, xy=(0.0, 1.0), xycoords='axes fraction',
                xytext=(-34 + dx * 10, 10 + dy * 10), textcoords='offset points',
                fontsize=13, fontweight='bold', va='bottom', ha='left',
                annotation_clip=False)


def figsize(ncol, nrow=1, panel_w=PANEL_W, panel_h=PANEL_H, mw=1.4, mh=1.3):
    """Figure size giving every panel the SAME physical size across the suite.
    mw/mh are generous shared-margin inches (room for y-labels, a colourbar, the
    suptitle and inter-panel breathing) — pair with ``constrained_layout=True`` so
    legends/titles/twin-axes don't collide. The margin also grows a little with the
    panel count, since more panels need more inter-panel gutter."""
    return (ncol * panel_w + mw + 0.25 * (ncol - 1),
            nrow * panel_h + mh + 0.2 * (nrow - 1))


def fig(ncol, nrow=1, **kw):
    """``plt.subplots`` with the suite's panel geometry AND constrained layout, so
    every figure breathes by default. Returns (fig, axes). Pass e.g. sharex=True."""
    return plt.subplots(nrow, ncol, figsize=figsize(ncol, nrow),
                        constrained_layout=True, squeeze=False if (ncol * nrow == 1) else True,
                        **kw)


def label_panels(axes, start=0):
    """Tag a flat/2-D array of axes a, b, c… in row-major order."""
    import numpy as _np
    flat = _np.atleast_1d(axes).ravel()
    for i, ax in enumerate(flat):
        panel_label(ax, chr(ord('a') + start + i))


def paired_bars(ax, xi, sp, te, w=0.38, labels=None, points=True,
                colors=(SPATIAL, TEMPORAL), point_color='0.25'):
    """One spatial/temporal bar PAIR at x=``xi``: mean ± SEM of each array, in the
    canonical architecture colours, with the paired unit's own points overlaid and
    JOINED spatial -> temporal.

    The join is the point: every test these bars carry is PAIRED, so the line shows
    which units move together and whether any one reverses — two clouds would not.
    ``sp``/``te`` are the paired values (one per mouse for an across-animals panel,
    one per trial for a within-animal one); pass ``points=False`` when there are
    hundreds of them and the overlay would just ink the panel black.

    ``labels`` is a (spatial, temporal) pair for the legend — pass it on the FIRST
    group only, else every group re-adds the same two entries. Returns the pair's
    top (highest bar+SEM, or highest point when ``points``), for placing a star
    clear of the error cap.

    Shared by diagnostics/projflat_spat_vs_temp_bymouse.py (bars per config) and
    diagnostics/spat_temp_by_state.py (bars per IO-HMM state) — keep the drawing
    here, do not copy it.
    """
    import numpy as _np
    sp, te = _np.asarray(sp, float), _np.asarray(te, float)
    top = 0.0
    for v, off, colr, lab in [(sp, -w / 2, colors[0], None if labels is None else labels[0]),
                              (te, +w / 2, colors[1], None if labels is None else labels[1])]:
        e = float(v.std(ddof=1) / _np.sqrt(v.size)) if v.size > 1 else 0.0
        ax.bar(xi + off, v.mean(), w, yerr=e, color=colr, edgecolor='k', linewidth=0.5,
               capsize=3, label=lab)
        top = max(top, v.mean() + e)
    if points:
        for si, ti in zip(sp, te):
            ax.plot([xi - w / 2, xi + w / 2], [si, ti], '-', lw=0.6, color='0.45',
                    alpha=0.75, zorder=3)
        for v, off in [(sp, -w / 2), (te, +w / 2)]:
            ax.plot(_np.full_like(v, xi + off), v, 'o', ms=2.5, color=point_color,
                    alpha=0.6, zorder=4)
        top = max(top, sp.max(), te.max())
    return float(top)


def cap_posterior_ylim(ax, ref_peak, mult=3.0, note=True):
    """Keep a broad target visible when overlaid with very peaky distributions.

    Posterior galleries plot the (broad) IO target together with decoded posteriors
    that can be many times peakier (PCA spikes to ~0.5 vs a target max-prob ~0.05),
    so the target gets squashed to the axis floor. Cap the y-axis at ``mult * ref_peak``
    (``ref_peak`` = that panel's IO-target max-prob) so the target occupies the lower
    ~1/mult of the panel; curves above the cap clip, flagged with a small corner note.
    Call AFTER plotting all curves. Returns the cap (or None if ref_peak is invalid)."""
    import numpy as _np
    if not _np.isfinite(ref_peak) or ref_peak <= 0:
        return None
    cap = float(mult) * float(ref_peak)
    clipped = any(ln.get_ydata().size and _np.nanmax(ln.get_ydata()) > cap * 1.02
                  for ln in ax.get_lines())
    ax.set_ylim(0, cap)
    if clipped and note:
        ax.annotate('peaks clipped', xy=(0.96, 0.96), xycoords='axes fraction',
                    ha='right', va='top', fontsize=6, color='0.55', style='italic')
    return cap


# Saving (SVG full detail + PNG capped ≤1600 px) is provided by figsave.save_fig,
# re-exported at the top of this module.
