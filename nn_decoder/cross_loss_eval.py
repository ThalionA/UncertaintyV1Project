# -*- coding: utf-8 -*-
"""Cross-loss test-time evaluation: score every trained decoder under EVERY
loss, regardless of which loss it was trained on, normalised to its own shuffle.

Motivation
----------
``plot_loss_sweep.py`` assesses every loss's decoder on a single yardstick (the
PCA-projected loss). But a model trained to minimise, say, JS is not being
scored on its own terms when we only read its PCA loss — and a model that looks
good under PCA may look bad under KL/Wasserstein/CE. The fair question the
2026-06 meeting asked is:

    For each *evaluation* metric, which *training* loss yields the best
    held-out posteriors?

That is a (training-loss x evaluation-loss) matrix. Both the decoded posterior
and the target are already saved per trial in each ``.mat`` (``dist[arch]
['decoded']`` / ``['target']``, plus ``pcs`` / ``explained_var`` for the PCA
branch), so the whole matrix is computable post-hoc with NO retraining — we just
re-apply each loss's definition to the saved (decoded, target) pairs.

Scale-free via shuffles
-----------------------
The losses live on wildly different scales (KL in nats, PCA x100, Wasserstein in
bin units, CE in nats), so raw values cannot be compared across columns. Every
``.mat`` also stores a shuffle control per arch (``dist[arch+'_shf']``) — the
decoder's posteriors under a label-shuffled fit, i.e. the chance floor for that
exact (metric, arch). We therefore report a **skill score**

    skill = mean_test_loss(real) / mean_test_loss(shuffle)

per mouse, then average across mice. ``skill < 1`` = better than chance,
``skill = 1`` = no better than shuffle, ``skill > 1`` = worse than chance. Skill
is dimensionless and comparable across every metric and both architectures, so
the cross-loss matrix and the spat-vs-temp difference are both meaningful without
any per-column rescaling. (Pass ``--value raw`` for the legacy column-normalised
view of absolute losses.)

The loss definitions are imported verbatim from ``nn_classifier`` via
``fit_loss_per_trial`` so eval-time numbers use exactly the same maths the
training loop used; the real diagonal equals the run's own held-out fit-loss.

Usage
-----
    python cross_loss_eval.py --run-name new_loss_sweep_may27
    python cross_loss_eval.py --run-name new_loss_sweep_may27 --value raw
    python cross_loss_eval.py --run-name <r> \
        --extra-mat Q_MSE_half_100ms_stratified_balanced.mat:MSE

Outputs land at ``figures/loss_sweep_plots/<run_name>/``:
    10_cross_loss_skill_<arch>.png/.svg   (or 10_cross_loss_matrix_* for raw)
    11_spat_vs_temp_diff.png/.svg
    10_cross_loss_matrix.csv              (real, shuffle, skill per cell)
    11_spat_vs_temp_diff.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import plot_loss_sweep as P
from nn_classifier import fit_loss_per_trial
import peakiness_style as ps  # loss display labels (PCA -> Projection-based)


# Evaluation metrics, in display order. These are the loss_func_type strings
# fit_loss_per_trial understands. MSE is deliberately excluded: it collapses to
# the marginal-mean baseline (degenerate), so it carries no signal as either a
# training loss or a test-time metric. Pass it explicitly via --extra-mat if you
# ever want it back.
EVAL_LOSSES = ('PCA', 'CE', 'KL', 'JS', 'Wasserstein')
ARCHS = ('spat', 'temp')
ARCH_LABEL = {'spat': 'spatial', 'temp': 'temporal'}


def _eval_one(decoded, target, eval_loss, pcs, evar):
    """Mean per-trial test loss of (decoded, target) under ``eval_loss``.

    Mirrors the training loop exactly by delegating to fit_loss_per_trial; NaN
    rows (unfit trials) are dropped before the mean.
    """
    dec = np.asarray(decoded, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if dec.ndim != 2 or dec.size == 0:
        return np.nan
    good = np.isfinite(dec).all(axis=1) & np.isfinite(tgt).all(axis=1)
    if not good.any():
        return np.nan
    pred_t = torch.tensor(dec[good])
    targ_t = torch.tensor(tgt[good])
    pcs_t = evar_t = None
    if eval_loss == 'PCA':
        if pcs is None or np.asarray(pcs).size == 0:
            return np.nan
        pcs_t = torch.tensor(np.asarray(pcs, dtype=np.float64))
        evar_t = torch.tensor(np.asarray(evar, dtype=np.float64))
    per_trial = fit_loss_per_trial(pred_t, targ_t, eval_loss, pcs_t, evar_t)
    return float(torch.mean(per_trial).item())


def _agg(vals):
    """(mean, sem) over a list of per-mouse scalars, NaNs dropped."""
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return (np.nan, np.nan)
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return (float(arr.mean()), sem)


def build_matrix(sweep, eval_losses=EVAL_LOSSES, exclude=None):
    """Return ``{arch: {train_loss: {eval_loss: cell}}}, train_losses``.

    Each ``cell`` is ``{'real': (m, sem), 'shuffle': (m, sem),
    'skill': (m, sem)}``. ``skill`` is computed per mouse (real_loss /
    shuffle_loss) then aggregated, so the chance normalisation is within-mouse.
    ``exclude`` (a mouse key, e.g. ``'mouse_2'``) drops that animal from every
    aggregate — the meeting's M2 leave-out.
    """
    # Only losses with a loaded .mat (load_loss_sweep returns None for a loss
    # whose cell hasn't finished training — common while a grid is mid-run).
    train_losses = [l for l in eval_losses
                    if sweep.get(l) is not None and 'results' in sweep[l]]
    out = {arch: {} for arch in ARCHS}
    for arch in ARCHS:
        shf = arch + '_shf'
        for tl in train_losses:
            mat = sweep[tl]
            cell = {}
            for el in eval_losses:
                reals, shufs, skills = [], [], []
                for mid, m in mat['results'].items():
                    if exclude and mid == exclude:
                        continue
                    d = m['Dist']
                    r = _eval_one(d[arch]['decoded'], d[arch]['target'], el,
                                  d.get('pcs'), d.get('explained_var'))
                    s = np.nan
                    if shf in d:
                        s = _eval_one(d[shf]['decoded'], d[shf]['target'], el,
                                      d.get('pcs'), d.get('explained_var'))
                    reals.append(r)
                    shufs.append(s)
                    if np.isfinite(r) and np.isfinite(s) and s > 0:
                        skills.append(r / s)
                cell[el] = {'real': _agg(reals), 'shuffle': _agg(shufs),
                            'skill': _agg(skills)}
            out[arch][tl] = cell
    return out, train_losses


# ----------------------------------------------------------------------
# Figure 10 — cross-loss matrix (skill, or raw column-normalised)
# ----------------------------------------------------------------------

def _skill_heatmap(ax, M, train_losses, eval_losses, title):
    """Heatmap of skill = real/shuffle. Diverging colour centred at 1 (chance):
    green = informative (<1), red = at/worse than chance (>=1). Green box marks
    the lowest-skill (best) training loss per evaluation metric."""
    import matplotlib
    skill = np.array([[M[tl][el]['skill'][0] for el in eval_losses]
                      for tl in train_losses])
    span = np.nanmax(np.abs(skill - 1.0))
    im = ax.imshow(skill, cmap='RdYlGn_r', aspect='auto',
                   vmin=1.0 - span, vmax=1.0 + span)
    ax.set_xticks(range(len(eval_losses)))
    ax.set_xticklabels(ps.loss_labels(eval_losses), rotation=20, ha='right')
    ax.set_yticks(range(len(train_losses)))
    ax.set_yticklabels(ps.loss_labels(train_losses))
    ax.set_xlabel('evaluation metric (applied to HELD-OUT test posteriors)')
    ax.set_ylabel('training objective (loss the net was fit with)')
    ax.set_title(title)
    win = np.nanargmin(skill, axis=0)
    for j in range(len(eval_losses)):
        for i, tl in enumerate(train_losses):
            v = skill[i, j]
            if not np.isfinite(v):
                ax.text(j, i, 'n/a', ha='center', va='center', fontsize=7,
                        color='0.4')
                continue
            best = (i == win[j])
            ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=8.5,
                    color='black', fontweight='bold' if best else 'normal')
            if best:
                ax.add_patch(matplotlib.patches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='#15561f',
                    lw=2.4))
    return im


def _raw_heatmap(ax, M, train_losses, eval_losses, title):
    """Legacy column-normalised heatmap of absolute losses (each column / its
    min). 1.0 = best training loss for that eval metric."""
    import matplotlib
    raw = np.array([[M[tl][el]['real'][0] for el in eval_losses]
                    for tl in train_losses])
    col_min = np.nanmin(raw, axis=0, keepdims=True)
    col_min = np.where(col_min > 0, col_min, np.nan)
    norm = raw / col_min
    im = ax.imshow(norm, cmap='YlOrRd', aspect='auto', vmin=1.0,
                   vmax=np.nanpercentile(norm, 95))
    ax.set_xticks(range(len(eval_losses)))
    ax.set_xticklabels(ps.loss_labels(eval_losses), rotation=20, ha='right')
    ax.set_yticks(range(len(train_losses)))
    ax.set_yticklabels(ps.loss_labels(train_losses))
    ax.set_xlabel('evaluation metric (applied to HELD-OUT test posteriors)')
    ax.set_ylabel('training objective (loss the net was fit with)')
    ax.set_title(title)
    win = np.nanargmin(raw, axis=0)
    for j in range(len(eval_losses)):
        for i in range(len(train_losses)):
            v = raw[i, j]
            if not np.isfinite(v):
                ax.text(j, i, 'n/a', ha='center', va='center', fontsize=7,
                        color='0.4')
                continue
            best = (i == win[j])
            ax.text(j, i, f"{v:.3g}", ha='center', va='center', fontsize=7.5,
                    color='black', fontweight='bold' if best else 'normal')
            if best:
                ax.add_patch(matplotlib.patches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='#1a7d1a',
                    lw=2.2))
    return im


def plot_matrix(matrix, train_losses, eval_losses, out_dir, value='skill'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    is_skill = value == 'skill'
    stem = '10_cross_loss_skill' if is_skill else '10_cross_loss_matrix'
    for arch in ARCHS:
        fig, ax = plt.subplots(figsize=(1.5 * len(eval_losses) + 2.6,
                                        0.75 * len(train_losses) + 2.4))
        if is_skill:
            im = _skill_heatmap(
                ax, matrix[arch], train_losses, eval_losses,
                f"Cross-loss normalised loss (test / shuffle) — {ARCH_LABEL[arch]}\n"
                f"<1 = beats chance, 1 = shuffle, green box = best training "
                f"loss for that metric")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label('normalised loss = test loss / shuffle loss  (1 = chance)')
        else:
            im = _raw_heatmap(
                ax, matrix[arch], train_losses, eval_losses,
                f"Cross-loss held-out loss — {ARCH_LABEL[arch]}\n"
                f"(cell = mean test loss; column-normalised colour)")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label('test loss / column best (1.0 = best)')
        fig.tight_layout()
        for ext in ('png', 'svg'):
            fig.savefig(out_dir / f"{stem}_{arch}.{ext}", dpi=140)
        plt.close(fig)
        print(f"  -> {stem}_{arch}.png/.svg")


# ----------------------------------------------------------------------
# Figure 11 — spat vs temp difference
# ----------------------------------------------------------------------

def _stars(p):
    """Significance stars from a p-value (paired t, n = mice)."""
    if not np.isfinite(p):
        return ''
    return '**' if p < 0.01 else ('*' if p < 0.05 else '')


def plot_diff_matrix(matrix, train_losses, eval_losses, out_dir, value='skill',
                     sweep=None, exclude=None):
    """spat-vs-temp dissociation in one figure. With ``value='skill'`` the cell
    is the skill difference ``skill_spat - skill_temp`` (both dimensionless, so
    directly subtractable); positive = temporal more informative. With ``value='raw'``
    it falls back to the relative gap ``(spat - temp)/temp`` in percent.

    If ``sweep`` is given, each cell is annotated with paired-t significance
    stars (n = mice, ``exclude`` dropped): * p<0.05, ** p<0.01 — computed on the
    skill difference (skill mode) or the raw difference (raw mode).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    is_skill = value == 'skill'
    D = np.full((len(train_losses), len(eval_losses)), np.nan)
    star = [['' for _ in eval_losses] for _ in train_losses]
    for i, tl in enumerate(train_losses):
        for j, el in enumerate(eval_losses):
            if is_skill:
                s = matrix['spat'][tl][el]['skill'][0]
                t = matrix['temp'][tl][el]['skill'][0]
                if np.isfinite(s) and np.isfinite(t):
                    D[i, j] = s - t
            else:
                s = matrix['spat'][tl][el]['real'][0]
                t = matrix['temp'][tl][el]['real'][0]
                if np.isfinite(s) and np.isfinite(t) and t > 0:
                    D[i, j] = (s - t) / t * 100.0
            if sweep is not None and tl in sweep:
                sr, tr, ss, ts = _per_mouse_spat_temp(sweep[tl], el, exclude=exclude)
                a, b = (ss, ts) if is_skill else (sr, tr)
                star[i][j] = _stars(_paired(a, b)[4])   # index 4 = paired-t p

    vmax = np.nanmax(np.abs(D))
    fig, ax = plt.subplots(figsize=(1.5 * len(eval_losses) + 2.6,
                                    0.75 * len(train_losses) + 2.4))
    im = ax.imshow(D, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(eval_losses)))
    ax.set_xticklabels(ps.loss_labels(eval_losses), rotation=20, ha='right')
    ax.set_yticks(range(len(train_losses)))
    ax.set_yticklabels(ps.loss_labels(train_losses))
    ax.set_xlabel('evaluation metric (applied to HELD-OUT test posteriors)')
    ax.set_ylabel('training objective (loss the net was fit with)')
    unit = 'normalised loss (spat - temp)' if is_skill else '(spat - temp)/temp [%]'
    nlab = '' if not exclude else f', {exclude} excluded'
    ax.set_title("Spatial vs temporal — architecture gap, shuffle-"
                 f"normalised{nlab}\n{unit}; green = temporal more informative, "
                 "red = spatial better;  * p<.05 ** p<.01 (paired t)")
    for i in range(len(train_losses)):
        for j in range(len(eval_losses)):
            v = D[i, j]
            if np.isfinite(v):
                txt = (f"{v:+.2f}" if is_skill else f"{v:+.0f}%") + star[i][j]
                ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                        color='black')
            else:
                ax.text(j, i, 'n/a', ha='center', va='center', fontsize=7,
                        color='0.4')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(unit)
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(out_dir / f"11_spat_vs_temp_diff.{ext}", dpi=140)
    plt.close(fig)
    print("  -> 11_spat_vs_temp_diff.png/.svg")


def write_csv(matrix, train_losses, eval_losses, out_dir: Path):
    rows = ["arch,train_loss,eval_loss,real,shuffle,norm_loss,norm_loss_sem,is_diagonal"]
    for arch in ARCHS:
        for tl in train_losses:
            for el in eval_losses:
                c = matrix[arch][tl][el]
                rows.append(
                    f"{arch},{tl},{el},{c['real'][0]:.6g},"
                    f"{c['shuffle'][0]:.6g},{c['skill'][0]:.6g},"
                    f"{c['skill'][1]:.4g},{int(tl == el)}")
    path = out_dir / "10_cross_loss_matrix.csv"
    path.write_text("\n".join(rows) + "\n")
    print(f"  -> {path.name}")

    drows = ["train_loss,eval_loss,norm_spat,norm_temp,norm_spat_minus_temp"]
    for tl in train_losses:
        for el in eval_losses:
            s = matrix['spat'][tl][el]['skill'][0]
            t = matrix['temp'][tl][el]['skill'][0]
            diff = s - t if (np.isfinite(s) and np.isfinite(t)) else float('nan')
            drows.append(f"{tl},{el},{s:.6g},{t:.6g},{diff:.6g}")
    dpath = out_dir / "11_spat_vs_temp_diff.csv"
    dpath.write_text("\n".join(drows) + "\n")
    print(f"  -> {dpath.name}")


def _per_mouse_spat_temp(mat, el, exclude=None):
    """Per-mouse paired (spat, temp) raw losses and skills under metric ``el``.

    Returns four equal-length lists aligned by mouse: spat_raw, temp_raw,
    spat_skill, temp_skill (skill = real/shuffle; NaN where the shuffle is
    missing/zero). Pairing is by mouse so a paired test is valid. ``exclude``
    drops one animal (the M2 leave-out).
    """
    sr, tr, ss, ts = [], [], [], []
    for mid, m in mat['results'].items():
        if exclude and mid == exclude:
            continue
        d = m['Dist']
        pcs, evar = d.get('pcs'), d.get('explained_var')
        vals = {}
        for arch in ARCHS:
            r = _eval_one(d[arch]['decoded'], d[arch]['target'], el, pcs, evar)
            shf = arch + '_shf'
            s = (_eval_one(d[shf]['decoded'], d[shf]['target'], el, pcs, evar)
                 if shf in d else np.nan)
            sk = r / s if (np.isfinite(r) and np.isfinite(s) and s > 0) else np.nan
            vals[arch] = (r, sk)
        sr.append(vals['spat'][0]); tr.append(vals['temp'][0])
        ss.append(vals['spat'][1]); ts.append(vals['temp'][1])
    return sr, tr, ss, ts


def _paired(spat, temp):
    """Paired spat-vs-temp summary over mice: means, mean diff (spat-temp),
    paired t p-value and Wilcoxon signed-rank p-value. NaN pairs dropped."""
    from scipy import stats as st
    a = np.asarray(spat, float)
    b = np.asarray(temp, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = a.size
    if n < 2:
        return n, np.nan, np.nan, np.nan, np.nan, np.nan
    t_p = float(st.ttest_rel(a, b).pvalue)
    try:
        w_p = float(st.wilcoxon(a, b).pvalue)
    except ValueError:          # all-zero differences etc.
        w_p = np.nan
    return n, float(a.mean()), float(b.mean()), float((a - b).mean()), t_p, w_p


def write_spat_temp_stats(sweep, train_losses, eval_losses, out_dir: Path,
                          exclude=None):
    """#2 — paired spat-vs-temp test per (training loss x eval metric), on BOTH
    raw test loss and shuffle-normalised skill. n = mice (paired). Reports a
    paired t and a Wilcoxon signed-rank p so the small-n nonparametric is there
    too. The diagonal (loss scored under its own training metric) is flagged.
    ``exclude`` drops one animal (M2 leave-out)."""
    rows = ["train_loss,eval_loss,n_mice,"
            "raw_spat,raw_temp,raw_diff,raw_ttest_p,raw_wilcoxon_p,"
            "norm_spat,norm_temp,norm_diff,norm_ttest_p,norm_wilcoxon_p,"
            "is_own_metric"]
    print("\n  spat-vs-temp paired stats (negative diff = temporal better):")
    for tl in train_losses:
        mat = sweep[tl]
        for el in eval_losses:
            sr, tr, ss, ts = _per_mouse_spat_temp(mat, el, exclude=exclude)
            n, rs, rt, rd, rtp, rwp = _paired(sr, tr)
            _, sks, skt, skd, stp, swp = _paired(ss, ts)
            rows.append(
                f"{tl},{el},{n},{rs:.6g},{rt:.6g},{rd:.6g},{rtp:.4g},{rwp:.4g},"
                f"{sks:.6g},{skt:.6g},{skd:.6g},{stp:.4g},{swp:.4g},"
                f"{int(tl == el)}")
            if tl == el:        # headline: each loss under its own metric
                star = '*' if (np.isfinite(stp) and stp < 0.05) else ' '
                print(f"    {tl:<11} [own metric] norm.loss spat={sks:.2f} "
                      f"temp={skt:.2f} Δ={skd:+.2f} (t p={stp:.3f}{star}, "
                      f"W p={swp:.3f})")
    path = out_dir / "12_spat_temp_paired_stats.csv"
    path.write_text("\n".join(rows) + "\n")
    print(f"  -> {path.name}")


def main(run_name, results_root=None, out_root='figures/loss_sweep_plots',
         extra_mats=(), value='skill', target='Q', window='half', bin_ms=100,
         split='stratified_balanced', exclude=None):
    # Point the shared plot_loss_sweep loader at the requested cell.
    P.TARGET, P.WINDOW, P.BIN_MS, P.SPLIT = target, window, bin_ms, split
    sweep = P.load_loss_sweep(run_name, results_root=results_root)
    # Optionally fold in stand-alone .mat files (e.g. the top-level Q_MSE that
    # the sweep driver dropped) as extra training-loss rows.
    for spec in extra_mats:
        mpath, label = spec.split(':')
        from scipy.io import loadmat
        raw = loadmat(mpath, simplify_cells=True)
        sweep[label] = raw
        print(f"  folded extra: {label} <- {mpath}")
    exc = f" (excluding {exclude})" if exclude else ""
    print(f"Cross-loss eval: {run_name} | cell {P.cell_tag()} (value={value}){exc}")
    print(f"  training losses present: {[l for l in EVAL_LOSSES if l in sweep]}")
    matrix, train_losses = build_matrix(sweep, exclude=exclude)
    cell = P.cell_tag() + (f"_no_{exclude}" if exclude else "")
    out_dir = Path(out_root) / run_name / cell
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_matrix(matrix, train_losses, list(EVAL_LOSSES), out_dir, value=value)
    plot_diff_matrix(matrix, train_losses, list(EVAL_LOSSES), out_dir,
                     value=value, sweep=sweep, exclude=exclude)
    write_csv(matrix, train_losses, list(EVAL_LOSSES), out_dir)
    write_spat_temp_stats(sweep, train_losses, list(EVAL_LOSSES), out_dir,
                          exclude=exclude)
    print(f"Done. {out_dir.resolve()}")
    return matrix


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run-name', default='new_loss_sweep_may27')
    ap.add_argument('--results-root', default=None)
    ap.add_argument('--out-root', default='figures/loss_sweep_plots')
    ap.add_argument('--value', choices=('skill', 'raw'), default='skill',
                    help="'skill' (default) = loss/shuffle, scale-free; "
                         "'raw' = absolute loss, column-normalised colour.")
    ap.add_argument('--extra-mat', action='append', default=[],
                    metavar='PATH:LABEL',
                    help='Fold a stand-alone .mat in as an extra training '
                         'loss, e.g. Q_MSE_half_100ms_stratified_balanced.mat:MSE')
    ap.add_argument('--target', default='Q')
    ap.add_argument('--window', default='half')
    ap.add_argument('--bin', type=int, default=100, dest='bin_ms')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--exclude', default=None,
                    help="mouse key to leave out of every aggregate, "
                         "e.g. mouse_2 (the meeting's M2 leave-out)")
    a = ap.parse_args()
    main(run_name=a.run_name, results_root=a.results_root, out_root=a.out_root,
         extra_mats=tuple(a.extra_mat), value=a.value,
         target=a.target, window=a.window, bin_ms=a.bin_ms, split=a.split,
         exclude=a.exclude)
