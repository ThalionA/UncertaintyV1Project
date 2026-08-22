"""Decoder PERFORMANCE on the IO-HMM targets — how well the networks actually decode.

Reads the per-mouse scorecard CSVs (figures/io_hmm_vs_export_v2/m<M>/) and plots
the two judging metrics directly, per mouse, per configuration:

  KL skill        = mean KL(target||decoded) / mean KL(target||leave-one-out
                    predict-mean), on held-out trials.  < 1 beats the null.
  projection skill= same ratio under the arm-pinned evar projection loss.

Both are dimensionless within an arm, so the NEW (IO-HMM, 72 circular bins) and
OLD (export Q, 91 linear bins) arms can be shown side by side.

lambda_H = 0 cells only: the entropy penalty is inert for KL/JS (<=4% over the
whole sweep) and destabilises the projection family, so lambda_H = 0 is the
canonical configuration. Usage: python diagnostics/io_hmm_performance.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from figsave import save_fig  # noqa: E402

IN = _HERE.parent / "figures" / "io_hmm_vs_export_v3"
OUT = IN / "performance"
MICE = [0, 1, 2, 3, 4, 5]
MK = ['o', 's', '^', 'v', 'D', 'P']
FAM = [('pca', 'projection\n(evar)'), ('pcaflat', 'projection\n(flat)'),
       ('kl', 'KL'), ('js', 'JS')]
COL = {'pca': 'tab:orange', 'pcaflat': 'tab:blue', 'kl': 'tab:purple', 'js': 'deepskyblue'}


def load():
    fr = []
    for m in MICE:
        f = IN / f"m{m}" / "io_hmm_vs_export_cells.csv"
        d = pd.read_csv(f); d['mouse'] = m; fr.append(d)
    c = pd.concat(fr, ignore_index=True)
    return c[c.lambda_H == 0].copy()


def _panel(ax, c, arm, arch, metric, title, ylab):
    xs, labs = [], []
    for i, (fam, fl) in enumerate(FAM):
        for j, hid in enumerate(['h8', 'lin']):
            x = i * 2 + j
            xs.append(x); labs.append(f"{fl}\n{'H=8' if hid=='h8' else 'linear'}")
            s = c[(c.loss_family == fam) & (c.hidden == hid) &
                  (c.arch == arch) & (c.arm == arm)]
            for _, r in s.iterrows():
                ax.plot(x, r[metric], MK[int(r.mouse)], ms=6.5, mfc=COL[fam],
                        mec='k', mew=.5, alpha=.85, zorder=3)
            if len(s):
                ax.hlines(s[metric].median(), x - .34, x + .34, color='k', lw=2.2, zorder=4)
    ax.axhline(1.0, color='k', ls=':', lw=1.2)
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.grid(axis='y', alpha=.25)


def fig_performance(c):
    for metric, mname, fn in (('kl_skill', 'KL skill', 'perf_kl'),
                              ('proj_skill', 'projection skill', 'perf_proj')):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharey='row')
        for r, arm in enumerate(['new', 'old']):
            for k, arch in enumerate(['spat', 'temp']):
                sub = c[(c.arm == arm)]
                _panel(axes[r, k], sub, arm, arch, metric,
                       f"{'IO-HMM targets' if arm=='new' else 'old export Q'} — "
                       f"{'spatial (PPC)' if arch=='spat' else 'temporal (SBC)'} decoder",
                       f"{mname}\n(<1 beats predict-mean null)")
        fig.suptitle(
            f"Decoder performance: {mname} on held-out trials, one point per mouse "
            f"(n=6), λ_H=0 cells\nblack bar = median across mice; dotted line = the "
            f"leave-one-out predict-mean null (1.0). Lower is better.", fontsize=10.5)
        handles = [plt.Line2D([], [], marker=MK[m], ls='', mfc='.75', mec='k',
                              ms=6.5, label=f'mouse {m}') for m in MICE]
        axes[0, 1].legend(handles=handles, loc='upper right', ncol=3, frameon=True,
                          framealpha=.9, fontsize=7.5, handletextpad=.3,
                          columnspacing=.9, borderpad=.4)
        fig.tight_layout(rect=[0, 0, 1, .94])
        save_fig(fig, OUT, fn, max_px=1540)


def fig_best(c):
    """Which configuration actually wins, per mouse, under each metric."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for k, (metric, mname) in enumerate((('kl_skill', 'KL skill'),
                                         ('proj_skill', 'projection skill'))):
        ax = axes[k]
        n = c[c.arm == 'new']
        for m in MICE:
            s = n[n.mouse == m]
            for j, arch in enumerate(['spat', 'temp']):
                ss = s[s.arch == arch]
                if not len(ss):
                    continue
                b = ss.loc[ss[metric].idxmin()]
                y = m + (j - .5) * .3
                ax.barh(y, b[metric], height=.26, color=COL[b.loss_family],
                        edgecolor='k', lw=.5,
                        hatch='' if arch == 'spat' else '///')
                ax.text(b[metric] + .012, y, f"{b.loss_family}/{b.hidden}",
                        va='center', fontsize=7)
        ax.axvline(1.0, color='k', ls=':', lw=1.2)
        ax.set_yticks(MICE); ax.set_yticklabels([f'mouse {m}' for m in MICE], fontsize=8.5)
        ax.set_xlabel(f'best {mname} achieved  (<1 beats the null)', fontsize=9)
        ax.set_title(f'best configuration per mouse — {mname}', fontsize=10)
        ax.grid(axis='x', alpha=.25)
        ax.set_xlim(0, max(1.15, ax.get_xlim()[1]))
    fig.suptitle('What actually wins on the IO-HMM targets (λ_H=0; solid = spatial, '
                 'hatched = temporal; label = loss family / hidden size)', fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, .93])
    save_fig(fig, OUT, 'perf_best_config', max_px=1540)


def main():
    c = load()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_performance(c); fig_best(c)
    n = c[c.arm == 'new']
    print("median over 6 mice, NEW (IO-HMM) targets, lambda_H=0 "
          "— KL skill | projection skill  (<1 beats null)")
    for fam, fl in FAM:
        for hid in ['h8', 'lin']:
            for arch in ['spat', 'temp']:
                s = n[(n.loss_family == fam) & (n.hidden == hid) & (n.arch == arch)]
                o = c[(c.arm == 'old') & (c.loss_family == fam) &
                      (c.hidden == hid) & (c.arch == arch)]
                if not len(s):
                    continue
                print(f"  {fam:8s} {hid:4s} {arch:5s}  new {s.kl_skill.median():5.2f} | "
                      f"{s.proj_skill.median():5.2f}   old {o.kl_skill.median():5.2f} | "
                      f"{o.proj_skill.median():5.2f}   beats-null(KL) {int((s.kl_skill<1).sum())}/6")


if __name__ == '__main__':
    main()
