# -*- coding: utf-8 -*-
"""The nine headline `projflat_v1` cells, in one place.

Three architectures x three loss weightings. The architecture axis is the point:

    lin   hidden_sizes=[]              ONE full-rank map, MORE parameters than h8
    rr8   [8] + identity activation    rank-<=8 affine logit map, NO non-linearity
    h8    [8] + tanh                   rank <=8 AND a non-linearity

so ``lin -> rr8`` isolates the RANK bottleneck and ``rr8 -> h8`` isolates the tanh
(same width, same parameter count). Crossed with the weighting the cell was trained
under: variance-weighted projection (EVAR), flat projection (= MSE over the 91
bins), and the KL reference.

WHY THIS MODULE EXISTS. Every plotting script used to carry its own literal copy of
this list, in four different shapes, and they had already drifted: on 2026-08-12 the
scatter script knew 6 cells, the trial explorer 4, and only one script knew all 9.
A missing cell does not raise — `have()` silently skips it — so drift shows up as a
figure that is quietly incomplete. One table here, adaptors for each call-site shape.

Deliberately NOT centralised here (these scripts use different sets ON PURPOSE, and
widening them breaks or empties their figures):
  * `diagnostics/projflat_report.py`        fig1 is dimensionality x weighting, and
                                            rr8 has no pc3/pc5/pc10 cells by design
  * `diagnostics/projflat_spat_vs_temp.py`  h8-only regularisation ladders
  * `diagnostics/projflat_config_axes.py`   has its own importer; keep `configs_for`
  * `projflat_posteriors` / `tail_diagnosis` / `lambda_diagnosis`  deliberate 3-4
                                            cell exemplar picks

Pure data: no numpy, matplotlib, torch or I/O at import, so any script can import it.
"""

from __future__ import annotations

# (token, display label)
ARCH = [
    ('lin', 'linear (0 hidden)'),
    ('rr8', 'reduced-rank 8'),
    ('h8',  '8 hidden units'),
]

# (token, display label, cell-name pattern)
WEIGHT = [
    ('evar',  'variance-weighting', '{a}_raw_EVAR'),
    ('flat',  'flat-weighting',     '{a}_raw_l0_d0_w0'),
    ('klref', 'KL-trained',         '{a}_raw_KLref'),
]

# (label, cell, short) — weighting-major so each block reads lin -> rr8 -> h8.
HEADLINE = [(f'{alab}\n{wlab}', pat.format(a=a), f'{a}_{w}')
            for w, wlab, pat in WEIGHT
            for a, alab in ARCH]


def as_pairs(rows=None):
    """[(label, cell)] — for scripts that want a title and a cell name."""
    return [(lab, cell) for lab, cell, _ in (rows or HEADLINE)]


def as_dict(rows=None):
    """{cell: one-line label} — for scripts keyed by cell name."""
    return {cell: lab.replace('\n', ', ') for lab, cell, _ in (rows or HEADLINE)}


def by_arch(rows=None):
    """[(weighting label, {arch token: cell})] — for the rank-vs-tanh figures."""
    out = []
    for w, wlab, pat in WEIGHT:
        out.append((wlab, {a: pat.format(a=a) for a, _ in ARCH}))
    return out


def is_klref(short_or_cell):
    """True for the KL-trained anchors. They are legitimately scored under the
    PROJECTION loss (the standing "judge under both metrics" rule) but that is NOT
    the metric they were trained on, so figures must say so."""
    return 'klref' in short_or_cell.lower() or 'KLref' in short_or_cell


def weighting_of(short_or_cell):
    """'flat' | 'evar' | 'klref' — what projection weighting the cell STORES.
    Verified from the saved `explained_var` (2026-08-12): flat cells hold uniform
    1/91 (= MSE over an orthonormal 91-PC basis); the EVAR *and* the KLref cells
    hold the eigenvalue spectrum. So a flat cell's loss is not on the same scale as
    an evar cell's — compare spatial vs temporal WITHIN a config, not across."""
    s = short_or_cell.lower()
    if 'klref' in s:
        return 'klref'
    return 'flat' if ('flat' in s or 'l0_d0_w0' in s) else 'evar'
