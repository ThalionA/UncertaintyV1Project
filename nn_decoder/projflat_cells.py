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


# ---------------------------------------------------------------- the io_hmm_v3 grid
# A SECOND table, same shape as HEADLINE, for the `io_hmm_v3` sweep (IO-HMM targets
# on the 72-bin circular support). Same reason as above: the twelve projection-loss
# cells were about to be re-literalled into another plotting script.
#
# Axes here are arch x projection weighting x lambda_H:
#   h8   [8] + tanh                  rank <=8 AND a non-linearity
#   rr8  [8] + identity activation   rank-<=8 affine logit map, NO non-linearity
# crossed with the weighting the cell was TRAINED under — `pca` = eigenvalue
# (EVAR) weighted projection, `pcaflat` = uniform 1/72 (= MSE over the 72 bins);
# verified from the saved `explained_var`, 2026-08-28 — and with lambda_H.
#
# lambda_H IS TEMPORAL-ONLY. It weights mean H(per-bin predicted posterior), which
# only exists for the per-bin ('sampling') temporal decoder, so ALL the lambda cells
# of a (weighting, arch) group are EXACT SPATIAL REPLICATES. Measured 2026-08-28 on
# all four groups, all 6 mice: max |spatial decoded difference| = 0 (bit-equal), while
# the temporal decoded arrays differ by up to 1.0. Any figure showing a lambda group
# must therefore say that the spatial difference is not a result.
IO_HMM_RUN = 'io_hmm_v3'
IO_HMM_ANCHOR = 'pca_h8_lh0'          # evar cell -> the common projection basis

IO_ARCH = [
    ('h8',  'H=8 + tanh'),
    ('rr8', 'reduced-rank 8 (linear)'),
]

# (token, display label, cell-name loss prefix)
IO_WEIGHT = [
    ('evar', 'variance-weighting', 'pca'),
    ('flat', 'flat-weighting',     'pcaflat'),
]

# (token, display label)
#
# WHY 1e-4 IS IN HERE (added 2026-08-28). The two weightings break down at
# DIFFERENT rates, and 1e-4 is the lambda that separates them: median temporal
# projection loss for h8 is 0.623 / 0.624 / 0.686 (evar) against 0.611 / 0.689 /
# 1.000 (flat) at lambda_H 0 / 1e-4 / 3e-3. So at 1e-4 the flat cell is ALREADY
# damaged while the evar cell is untouched, and at 3e-3 the flat cell has
# collapsed to chance. Two lambdas showed only the endpoints; three show the
# ordering.
IO_LAMBDA = [
    ('lh0',    '\u03bb_H 0'),
    ('lh1e-4', '\u03bb_H 1e-4'),
    ('lh3e-3', '\u03bb_H 3e-3'),
]

# (label, cell, short) — weighting-major, then arch, then lambda, so the cells of
# every lambda GROUP are ADJACENT columns and the spatial-replicate bracket the
# figures draw spans neighbours.
IO_PROJ = [(f'{alab}\n{wlab}\n{llab}', f'{pre}_{a}_{lt}', f'{a}_{w}_{lt}')
           for w, wlab, pre in IO_WEIGHT
           for a, alab in IO_ARCH
           for lt, llab in IO_LAMBDA]


# ------------------------------------------------------------------- the registry
# Named tables, so a driver takes `--configs <name>` instead of carrying a literal
# cell list. Each entry pins the run dir and the common-basis anchor that table
# needs, because those travel WITH the cell names (an io_hmm cell scored against
# projflat_v1's `h8_raw_EVAR` anchor is a 91-vs-72-bin category error).
TABLES = {
    'headline': dict(
        rows=HEADLINE, run='projflat_v1', anchor='h8_raw_EVAR',
        note='Within each block: lin -> rr8 adds the RANK bottleneck, rr8 -> h8 adds '
             'the tanh. KL-trained cells are scored on a metric they were NOT trained on.'),
    'io_hmm_proj': dict(
        rows=IO_PROJ, run=IO_HMM_RUN, anchor=IO_HMM_ANCHOR,
        note='IO-HMM targets, 72-bin circular support. rr8 -> h8 adds the tanh at the same '
             'rank-8 bottleneck. lambda_H is TEMPORAL-ONLY: within each lambda group the '
             'spatial fit is the SAME model (decoded arrays bit-equal), so only the temporal '
             'bar can move -- the spatial difference is not a result. lambda_H 1e-4 is in the '
             'set because it SEPARATES the two weightings: median temporal projection loss h8 '
             'runs 0.623/0.624/0.686 (evar) vs 0.611/0.689/1.000 (flat) at lambda_H 0/1e-4/3e-3, '
             'so at 1e-4 flat is already damaged while evar is untouched.'),
}


# ------------------------------------------------------- standing caveats (prose)
# Two sentences that are true of a whole RUN rather than of any one panel, so the
# drivers print them once to stdout instead of drawing them on every figure. They
# live here, in the pure-data module, because BOTH `diagnostics/
# projflat_spat_vs_temp_bymouse.py` (bars per config) and `diagnostics/
# spat_temp_by_state.py` (bars per IO-HMM state) say them over the same table —
# and two wordings of one caveat is exactly how a caveat drifts into being wrong.
#
# The lambda claim is MEASURED, not assumed: 2026-08-28, all four io_hmm_v3
# (weighting, arch) groups x 6 mice, max |spatial decoded difference| = 0 while the
# temporal arrays differ by up to 1.0.
LAMBDA_NOTE = ('lambda_H is TEMPORAL-ONLY: within a lambda group the spatial bars are '
               'the SAME fit (decoded arrays bit-equal) -- only the temporal bar can '
               'move, so a spatial difference across lambda is structurally zero and '
               'is never a result.')

WEIGHT_NOTE = {
    'own': ('Own stored weighting: flat cells are scored as MSE, evar AND KL-trained cells '
            'eigenvalue-weighted -- so compare spatial-vs-temporal WITHIN a config, never '
            'bar heights ACROSS configs (--weighting common does that).'),
    'common': ('Common evar basis: every cell is rescored under the one anchor basis, so bar '
               'heights ARE comparable across configs -- a different question from each '
               "cell's own training metric."),
}


def table(name):
    """Registry lookup with a listing error, so a typo names the alternatives."""
    if name not in TABLES:
        raise SystemExit(f'unknown config table {name!r}; have: {", ".join(TABLES)}')
    return TABLES[name]


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


def cell_label(cell, rows=None, sep=', '):
    """One-line display label for a cell name, looked up in a table.

    Searches every registered table by default, so a driver handed arbitrary
    `--cells` gets the table's own label without knowing which table the cell is in.
    Falls back to the raw cell name for an unknown cell, rather than inventing a
    wrong label. (Was a second io_hmm-only table + lookup; folded onto TABLES on
    2026-08-28 so there is one io_hmm cell list, not two.)"""
    tables = [rows] if rows is not None else [t['rows'] for t in TABLES.values()]
    for tbl in tables:
        for lab, c, _ in tbl:
            if c == cell:
                return lab.replace('\n', sep)
    return cell


def lambda_of(short_or_cell):
    """The lambda_H token ('0', '3e-3', ...) in an io_hmm cell/short name, else None.
    Figures use it to say, on the figure, that a lambda pair shares one spatial fit."""
    for part in str(short_or_cell).split('_'):
        if part.startswith('lh') and len(part) > 2:
            return part[2:]
    return None


def block_bounds(rows=None):
    """Indices where the WEIGHTING block changes — where a figure draws its faint
    separator rules. Derived from the table rather than assumed to be every third
    column, which was only true of the 3x3 HEADLINE grid."""
    rows = rows or HEADLINE
    w = [weighting_of(s) for _, _, s in rows]
    return [i for i in range(1, len(w)) if w[i] != w[i - 1]]


def replicate_groups(rows=None):
    """[(first, last)] index spans of consecutive columns that differ ONLY in
    lambda_H — i.e. share one spatial fit. Spans are arbitrary length (the io_hmm
    table went from 2 to 3 lambdas on 2026-08-28 and this needed no change). Empty
    for tables with no lambda axis (HEADLINE), so a figure brackets unconditionally."""
    rows = rows or HEADLINE
    keys = [tuple(p for p in s.split('_') if not p.startswith('lh')) if lambda_of(s)
            else object() for _, _, s in rows]
    out, i = [], 0
    while i < len(keys):
        j = i
        while j + 1 < len(keys) and keys[j + 1] == keys[i]:
            j += 1
        if j > i:
            out.append((i, j))
        i = j + 1
    return out
