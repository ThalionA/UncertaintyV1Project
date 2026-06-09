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
    })


# ----------------------------------------------------------------------
# Shared primitives
# ----------------------------------------------------------------------
def target_band(ax, x, t, label='IO target'):
    """Draw the IO target as a filled grey band (the canonical 'truth' motif)."""
    return ax.fill_between(x, t, color=TARGET_GREY, alpha=0.8, lw=0, label=label)


def target_line(ax, y, label=None, value_fmt='{:.3f}'):
    """Horizontal black-dashed IO-target reference (for peakiness/skill axes)."""
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


def figsize(ncol, nrow=1, panel_w=PANEL_W, panel_h=PANEL_H, mw=0.9, mh=0.8):
    """Figure size giving every panel the SAME physical size across the suite.
    mw/mh are the shared-margin inches (labels, suptitle, colourbar)."""
    return (ncol * panel_w + mw, nrow * panel_h + mh)


def label_panels(axes, start=0):
    """Tag a flat/2-D array of axes a, b, c… in row-major order."""
    import numpy as _np
    flat = _np.atleast_1d(axes).ravel()
    for i, ax in enumerate(flat):
        panel_label(ax, chr(ord('a') + start + i))


# Saving (SVG full detail + PNG capped ≤1600 px) is provided by figsave.save_fig,
# re-exported at the top of this module.
