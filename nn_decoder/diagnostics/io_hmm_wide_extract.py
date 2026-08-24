# -*- coding: utf-8 -*-
"""Score the io_hmm_v3 width sweep (175 cells x 6 mice x 2 decoders) into ONE CSV.

The v3 grid: loss in {pca (evar projection), pcaflat (flat projection), kl, js,
ce} x arch in {lin, rr8, h4, h8, h16, h32, h64} x lambda_H in {0, 1e-4, 3e-4,
1e-3, 3e-3}. ``rr8`` is Linear(n,8)->Linear(8,72) with NO nonlinearity (a rank-8
bottleneck); ``hN`` is one tanh hidden layer of width N; ``lin`` is a single
Linear(n,72). Targets are the IO-HMM marginal on the 72-bin circular [0,180)
support; weight_decay = 0, dropout = 0, patience 20, REP 5, seed 0.

EVERY metric here is the settled scorecard metric, computed by the scorecard's
own functions (``diagnostics/io_hmm_vs_export_scorecard.py``: ``score_cell``,
``build_arm_curves``, ``sharpening_metrics``, ``arm_reference_target``) — this
file only enumerates the wider grid and reshapes the output. Per cell x mouse x
decoder:

  kl_skill    mean KL(tgt||dec) / mean KL(tgt||LOO predict-mean), held-out test
              trials; < 1 beats the strictest null
  proj_skill  the same ratio under the projection loss with the evar PINNED from
              one reference cell per mouse (``REF_CELL``); ``Dist.pcs`` is
              asserted identical across all 175 cells
  s_hat       equivalent sharpening (``sharpening_calibration``), calibration
              curve built ONCE per mouse from the targets (asserted identical
              across cells); reported with ``s_hat_agreement`` (> 0.10 = the
              decoder reshapes, s_hat under-describes it) and ``s_hat_clamped``
              (1 = the inversion hit a ladder end, the number is a bound)
  overfit_gap (val_fit - train_fit) at best_epoch / that mouse's LOO predict-mean
              loss under the cell's OWN training loss and OWN evar weighting
  best_epoch, early_stopped   the restored epoch (0-based, = argmin val fit) and
              the epoch training halted (best + patience, or the 200 cap)
  w_in_norm   Frobenius norm of the restored model's first weight matrix
              (``layers.0.weight``) — the input layer for every arch

lambda_H acts on the TEMPORAL decoder only, so a (loss, arch, mouse) group's
five lambda cells are exact spatial replicates; that is asserted here (max
deviation of spatial kl_skill across the five, must be 0).

Verification printed on every run:
  (1) s_hat(target, target) == 1.000 per mouse
  (2) Dist.pcs identical across all 175 cells per mouse (max |drift|)
  (3) spatial kl_skill identical across the 5 lambda cells of every group
  (4) row count == 175 * 6 * 2 = 2100

Output: figures/io_hmm_wide/cells.csv (one row per cell x mouse x decoder) and a
compact per-(loss, arch) median table for lambda = 0 on stdout.

Usage (from nn_decoder/):
    python diagnostics/io_hmm_wide_extract.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
import io_hmm_vs_export_scorecard as sc            # noqa: E402  (the settled metrics)

ROOT = _HERE.parent / 'results' / 'io_hmm_v3'
OUT_DIR = _HERE.parent / 'figures' / 'io_hmm_wide'
CSV_NAME = 'cells.csv'

REF_CELL = 'js_h8_lh0'          # evar pin + calibration-curve reference, per mouse
MICE = tuple(range(6))

LOSSES = ('pca', 'pcaflat', 'kl', 'js', 'ce')
ARCHS = ('lin', 'rr8', 'h4', 'h8', 'h16', 'h32', 'h64')
LAMS = ('lh0', 'lh1e-4', 'lh3e-4', 'lh1e-3', 'lh3e-3')
N_CELLS = len(LOSSES) * len(ARCHS) * len(LAMS)          # 175

HIDDEN = {'lin': 0, 'rr8': 8, 'h4': 4, 'h8': 8, 'h16': 16, 'h32': 32, 'h64': 64}
NONLINEAR = {a: a.startswith('h') for a in ARCHS}
LAM_VALUE = {'lh0': 0.0, 'lh1e-4': 1e-4, 'lh3e-4': 3e-4, 'lh1e-3': 1e-3, 'lh3e-3': 3e-3}
# the cell's OWN training loss (overfitting-gap denominator units); extends the
# scorecard's table with the CE family that v3 added
OWN_METRIC = dict(sc.OWN_METRIC, ce='CE')
# what the checkpoint must say the decoder WAS, per slug token (slug <-> training guard)
EXPECT_LOSS = {'pca': 'PCA', 'pcaflat': 'PCA', 'kl': 'KL', 'js': 'JS', 'ce': 'CE'}
EXPECT_ACT = {a: ('identity' if a == 'rr8' else 'tanh') for a in ARCHS}
EXPECT_HS = {a: ([] if a == 'lin' else [HIDDEN[a]]) for a in ARCHS}

DECODERS = ('spat', 'temp')
COLUMNS = ['cell', 'loss_family', 'arch', 'hidden', 'nonlinear', 'lambda_H',
           'mouse', 'decoder', 'kl_skill', 'proj_skill', 's_hat',
           's_hat_agreement', 's_hat_clamped', 'overfit_gap', 'best_epoch',
           'early_stopped', 'w_in_norm']


def parse_cell(name):
    """('pcaflat', 'rr8', 'lh3e-4') or None — the v3 grid, not the scorecard's."""
    parts = name.split('_')
    if len(parts) != 3:
        return None
    loss, arch, lam = parts
    if loss in LOSSES and arch in ARCHS and lam in LAMS:
        return loss, arch, lam
    return None


def _ckpt_extras(ckpt_path, arch, p):
    """w_in_norm from the RESTORED state_dict, plus the slug<->training guard."""
    c = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    a = c[arch]
    sd = a['state_dict']
    w_in = float(torch.linalg.norm(sd['layers.0.weight'].double()))
    loss, archtok, lam = p
    mp = a['model_params']
    hs = list(mp.get('hidden_sizes') or [])
    act = str(mp.get('activation_function'))
    lf = str(a.get('loss_func'))
    lamv = float(a.get('entropy_lambda', np.nan))
    problems = []
    if hs != EXPECT_HS[archtok]:
        problems.append(f'hidden_sizes {hs} != {EXPECT_HS[archtok]}')
    if act != EXPECT_ACT[archtok]:
        problems.append(f'activation {act!r} != {EXPECT_ACT[archtok]!r}')
    if lf != EXPECT_LOSS[loss]:
        problems.append(f'loss_func {lf!r} != {EXPECT_LOSS[loss]!r}')
    # lambda_H is a temporal-only knob: the spatial checkpoint may record 0 or the
    # cell's value depending on the runner, so only the temporal one is checked
    if arch == 'temp' and not np.isclose(lamv, LAM_VALUE[lam], rtol=1e-6, atol=1e-12):
        problems.append(f'entropy_lambda {lamv} != {LAM_VALUE[lam]}')
    # the best_epoch weight norm recorded in history must equal the restored
    # state_dict's — proves the state_dict IS the restored (best) model
    h = a['history']
    be = h.get('best_epoch')
    wn = h.get('weight_norms')
    if be is not None and wn and 0 <= int(be) < len(wn):
        if not np.isclose(float(wn[int(be)][0]), w_in, rtol=1e-5, atol=1e-6):
            problems.append(f'state_dict ||W_in|| {w_in:.6f} != history weight_norms'
                            f'[best_epoch][0] {float(wn[int(be)][0]):.6f}')
    return w_in, problems


def _pin_from_ref(cell_dir, mouse):
    m = loadmat(str(cell_dir / 'stratified_balanced.mat'), squeeze_me=True,
                struct_as_record=False)
    r = getattr(m['results'], f'mouse_{mouse}')
    pcs = np.asarray(r.Dist.pcs, float)
    evar = np.asarray(r.Dist.explained_var, float)
    if evar.max() <= 2.0 / evar.size:
        raise SystemExit(f'ABORT: reference cell {cell_dir.parent.name} carries a '
                         'FLAT evar — pin from a non-flat cell.')
    return pcs, evar


def score_mouse(dirs, mouse, verbose=True):
    """All 175 cells x 2 decoders for one mouse -> (rows, verification dict)."""
    t0 = time.time()
    pcs, evar = _pin_from_ref(dirs[REF_CELL], mouse)
    ref_tgt, tgt_src, tgt_nb, ref_cell = sc.arm_reference_target(
        {REF_CELL: dirs[REF_CELL]}, mouse)
    geom = sc.geometry(tgt_src, tgt_nb, n_cols=ref_tgt.shape[1])
    curves = sc.build_arm_curves(ref_tgt)
    ident = sc.sharpening_metrics(ref_tgt, ref_tgt, curves)
    if abs(ident['s_hat'] - 1.0) > 1e-9 or abs(ident['s_hat_entropy'] - 1.0) > 1e-9:
        raise SystemExit(f'ABORT: mouse {mouse} calibration broken — '
                         f's_hat(target,target) = {ident["s_hat"]:.6f}')

    rows, pcs_drift, tgt_drift, guard_problems = [], 0.0, 0.0, []
    for cell in sorted(dirs):
        p = parse_cell(cell)
        d = dirs[cell]
        s = sc.score_cell(d, mouse, pcs, evar, curves, ref_tgt,
                          own_metric=OWN_METRIC[p[0]])
        if s is None:
            raise SystemExit(f'ABORT: {cell} has no mouse_{mouse} in its shard.')
        if not np.isfinite(s['_pcs_drift']):
            raise SystemExit(f'ABORT: {cell} mouse {mouse}: Dist.pcs shape differs '
                             'from the pinned basis.')
        pcs_drift = max(pcs_drift, s['_pcs_drift'])
        tgt_drift = max(tgt_drift, s['_tgt_drift'])
        ck = d / 'checkpoints' / f'mouse_{mouse}_stratified_balanced.pt'
        loss, arch, lam = p
        for dec in DECODERS:
            rec = s.get(dec)
            if rec is None:
                raise SystemExit(f'ABORT: {cell} mouse {mouse} has no Dist.{dec}.')
            w_in, problems = _ckpt_extras(ck, dec, p)
            guard_problems += [f'{cell} m{mouse} {dec}: {q}' for q in problems]
            rows.append({
                'cell': cell, 'loss_family': loss, 'arch': arch,
                'hidden': HIDDEN[arch], 'nonlinear': NONLINEAR[arch],
                'lambda_H': LAM_VALUE[lam], 'mouse': mouse, 'decoder': dec,
                'kl_skill': rec['kl_skill'], 'proj_skill': rec['proj_skill'],
                's_hat': rec['s_hat'], 's_hat_agreement': rec['s_hat_agreement'],
                's_hat_clamped': int(rec['s_hat_clamped']),
                'overfit_gap': rec['overfit_gap'],
                'best_epoch': rec['best_epoch'],
                'early_stopped': rec['early_stopped_epoch'],
                'w_in_norm': w_in,
            })
    if verbose:
        print(f'  mouse {mouse}: {len(rows)} rows in {time.time() - t0:5.1f} s | '
              f'support {geom.label} | evar pinned from {REF_CELL} '
              f'(lead {evar.max():.3f}) | {ref_tgt.shape[0]} test trials')
    return rows, {'ident': ident['s_hat'], 'ident_entropy': ident['s_hat_entropy'],
                  'pcs_drift': pcs_drift, 'tgt_drift': tgt_drift,
                  'guard_problems': guard_problems}


def ce_vs_kl_identity(dirs, mice):
    """CE = KL + H(target), a constant in the decoder's parameters, so with one
    seed the ce_* and kl_* cells should be the SAME trained decoder. Measured
    rather than assumed: max |decoded_ce - decoded_kl| over every (arch, lambda,
    mouse, decoder), and how many of those blocks are exactly zero."""
    worst, n_zero, n_tot, worst_at = 0.0, 0, 0, None
    for arch in ARCHS:
        for lam in LAMS:
            fa = dirs[f'kl_{arch}_{lam}'] / 'stratified_balanced.mat'
            fb = dirs[f'ce_{arch}_{lam}'] / 'stratified_balanced.mat'
            ma = loadmat(str(fa), squeeze_me=True, struct_as_record=False)['results']
            mb = loadmat(str(fb), squeeze_me=True, struct_as_record=False)['results']
            for m in mice:
                ra, rb = getattr(ma, f'mouse_{m}'), getattr(mb, f'mouse_{m}')
                for dec in DECODERS:
                    d = float(np.nanmax(np.abs(
                        np.asarray(getattr(ra.Dist, dec).decoded, float)
                        - np.asarray(getattr(rb.Dist, dec).decoded, float))))
                    n_tot += 1
                    n_zero += int(d == 0.0)
                    if d > worst:
                        worst, worst_at = d, (arch, lam, m, dec)
    return worst, n_zero, n_tot, worst_at


def verify(df, per_mouse, ce_kl=None):
    """The four verification lines (+ the CE-vs-KL identity); raises on failure."""
    lines, ok = [], True
    # (1) identity calibration
    idents = {m: v['ident'] for m, v in per_mouse.items()}
    worst = max(abs(v - 1.0) for v in idents.values())
    ok1 = worst <= 1e-9
    lines.append(f"(1) s_hat(target,target) per mouse: "
                 + ' '.join(f'm{m}={v:.3f}' for m, v in idents.items())
                 + f" | max |s_hat-1| = {worst:.1e} -> {'PASS' if ok1 else 'FAIL'}")
    # (2) pcs identical across cells
    drift = {m: v['pcs_drift'] for m, v in per_mouse.items()}
    ok2 = max(drift.values()) <= 1e-12
    lines.append(f"(2) Dist.pcs max |cell - {REF_CELL}| across {N_CELLS} cells per mouse: "
                 + ' '.join(f'm{m}={v:.1e}' for m, v in drift.items())
                 + f" -> {'PASS (bit-identical)' if ok2 else 'FAIL'}")
    tdrift = {m: v['tgt_drift'] for m, v in per_mouse.items()}
    lines.append(f"    targets max |cell - {REF_CELL}| per mouse: "
                 + ' '.join(f'm{m}={v:.1e}' for m, v in tdrift.items())
                 + (' (calibration curve built once per mouse is licensed)'
                    if max(tdrift.values()) == 0 else ' -> FAIL'))
    ok2 = ok2 and max(tdrift.values()) == 0
    # (3) spatial replicates across lambda
    sp = df[df.decoder == 'spat']
    g = sp.groupby(['loss_family', 'arch', 'mouse'])
    n_per = g['kl_skill'].size()
    dev = (g['kl_skill'].max() - g['kl_skill'].min())
    dev_all = (sp.groupby(['loss_family', 'arch', 'mouse'])
               [['kl_skill', 'proj_skill', 's_hat', 'w_in_norm']]
               .agg(lambda v: v.max() - v.min()))
    ok3 = (n_per == len(LAMS)).all() and float(dev.max()) == 0.0
    worst_grp = dev.idxmax()
    lines.append(f"(3) spatial kl_skill across the {len(LAMS)} lambda cells of each "
                 f"(loss,arch,mouse) group [{len(dev)} groups, {int(n_per.min())}-"
                 f"{int(n_per.max())} cells each]: max deviation = {float(dev.max()):.2e} "
                 f"(group {worst_grp}); also max dev proj_skill "
                 f"{float(dev_all['proj_skill'].max()):.1e}, s_hat "
                 f"{float(dev_all['s_hat'].max()):.1e}, w_in_norm "
                 f"{float(dev_all['w_in_norm'].max()):.1e} -> "
                 f"{'PASS (exact replicates)' if ok3 else 'FAIL'}")
    # (4) row count
    n_expect = N_CELLS * len(MICE) * len(DECODERS)
    ok4 = len(df) == n_expect and not df.duplicated(['cell', 'mouse', 'decoder']).any()
    lines.append(f"(4) rows = {len(df)} (expected {N_CELLS}*{len(MICE)}*{len(DECODERS)} "
                 f"= {n_expect}); cells {df.cell.nunique()}, mice {df.mouse.nunique()}, "
                 f"decoders {df.decoder.nunique()} -> {'PASS' if ok4 else 'FAIL'}")
    # finite-ness (not a numbered check, but a NaN-silent metric is a bug)
    nan = df[['kl_skill', 'proj_skill', 's_hat', 'overfit_gap', 'best_epoch',
              'early_stopped', 'w_in_norm']].isna().sum()
    lines.append('    NaN counts: ' + ', '.join(f'{k}={int(v)}' for k, v in nan.items()))
    flags = df.groupby('decoder')[['s_hat_clamped']].sum()['s_hat_clamped']
    agree = df.groupby('decoder')['s_hat_agreement'].apply(
        lambda v: int((v > sc.AGREE_FLAG).sum()))
    lines.append('    s_hat flags: clamped ' + ', '.join(f'{k}={int(v)}' for k, v in flags.items())
                 + f' | agreement>{sc.AGREE_FLAG:.2f} (reshapes) '
                 + ', '.join(f'{k}={int(v)}' for k, v in agree.items())
                 + f'  (of {len(df) // 2} rows per decoder)')
    if ce_kl is not None:
        worst, n_zero, n_tot, at = ce_kl
        lines.append(f"(5) CE vs KL cells are the same decoder (CE = KL + H(target), "
                     f"same seed): {n_zero}/{n_tot} (arch,lambda,mouse,decoder) blocks "
                     f"bit-identical; max |decoded_ce - decoded_kl| = {worst:.1e} at "
                     f"{at} -> prior (d) ce~kl holds BY CONSTRUCTION, not as evidence")
    probs = [q for v in per_mouse.values() for q in v['guard_problems']]
    lines.append(f"    slug<->checkpoint guard (hidden_sizes, activation, loss_func, "
                 f"temporal entropy_lambda, restored ||W_in|| == history): "
                 f"{len(probs)} problems" + ('' if not probs else '\n      '
                                               + '\n      '.join(probs[:20])))
    ok = ok1 and ok2 and ok3 and ok4 and not probs
    return lines, ok


def median_table(df, lam=0.0):
    """Per-(loss, arch) medians over mice at one lambda, spat and temp side by side."""
    sub = df[np.isclose(df.lambda_H, lam)]
    keys = ['kl_skill', 'proj_skill', 's_hat', 'overfit_gap']
    med = (sub.groupby(['loss_family', 'arch', 'decoder'])[keys].median()
           .unstack('decoder'))
    med = med.reindex(pd.MultiIndex.from_product([LOSSES, ARCHS],
                                                 names=['loss_family', 'arch']))
    # flags per (loss, arch, decoder): how many of the 6 mice are clamped / reshape
    fl = (sub.assign(reshape=sub.s_hat_agreement > sc.AGREE_FLAG)
          .groupby(['loss_family', 'arch', 'decoder'])[['s_hat_clamped', 'reshape']]
          .sum().unstack('decoder'))
    hdr = (f"{'loss':8s} {'arch':5s} | {'spat kl':>7s} {'proj':>6s} {'s_hat':>6s} "
           f"{'clmp/rsh':>8s} {'ovfit':>6s} | {'temp kl':>7s} {'proj':>6s} "
           f"{'s_hat':>6s} {'clmp/rsh':>8s} {'ovfit':>6s}")
    out = [f'median over {sub.mouse.nunique()} mice, lambda_H = {lam:g}; '
           'kl/proj = skill vs LOO predict-mean (<1 beats null); s_hat 1 = calibrated; '
           'clmp/rsh = mice clamped / mice with agreement>0.10; ovfit = gap / null loss',
           hdr, '-' * len(hdr)]
    for loss in LOSSES:
        for arch in ARCHS:
            r = med.loc[(loss, arch)]
            f = fl.loc[(loss, arch)]
            cells = []
            for dec in DECODERS:
                cells.append(
                    f"{r[('kl_skill', dec)]:7.3f} {r[('proj_skill', dec)]:6.3f} "
                    f"{r[('s_hat', dec)]:6.2f} "
                    f"{int(f[('s_hat_clamped', dec)]):>4d}/{int(f[('reshape', dec)]):<3d} "
                    f"{r[('overfit_gap', dec)]:6.3f}")
            out.append(f'{loss:8s} {arch:5s} | ' + ' | '.join(cells))
        out.append('')
    return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--out-dir', type=Path, default=OUT_DIR)
    ap.add_argument('--mice', type=int, nargs='*', default=list(MICE))
    ap.add_argument('--csv-name', default=CSV_NAME,
                    help='output CSV filename (e.g. cells_exportref.csv for the '
                         'export-target arm)')
    ap.add_argument('--arm', default=None,
                    help="if given, an 'arm' column with this value is appended "
                         "(e.g. 'old' for the exportref arm)")
    args = ap.parse_args(argv)

    dirs = sc._cell_dirs(args.root)
    bad = sorted(c for c in dirs if parse_cell(c) is None)
    if bad:
        raise SystemExit(f'ABORT: unparseable cell slugs under {args.root}: {bad}')
    expected = {f'{l}_{a}_{m}' for l in LOSSES for a in ARCHS for m in LAMS}
    missing = sorted(expected - set(dirs))
    extra = sorted(set(dirs) - expected)
    print(f'{len(dirs)} finished cells under {args.root} '
          f'(expected {N_CELLS}; missing {len(missing)}, extra {len(extra)})')
    if missing or extra:
        raise SystemExit(f'ABORT: grid incomplete. missing={missing} extra={extra}')
    if REF_CELL not in dirs:
        raise SystemExit(f'ABORT: reference cell {REF_CELL} not found.')

    rows, per_mouse = [], {}
    for m in args.mice:
        r, v = score_mouse(dirs, m)
        rows += r
        per_mouse[m] = v
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.sort_values(['loss_family', 'arch', 'lambda_H', 'mouse', 'decoder'],
                        key=lambda s: (s.map({k: i for i, k in enumerate(LOSSES)})
                                       if s.name == 'loss_family' else
                                       s.map({k: i for i, k in enumerate(ARCHS)})
                                       if s.name == 'arch' else s)).reset_index(drop=True)

    if args.arm is not None:
        df['arm'] = args.arm
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv = args.out_dir / args.csv_name
    df.to_csv(csv, index=False, float_format='%.10g')
    print(f'wrote {csv}  ({len(df)} rows x {len(df.columns)} cols)')

    print('\nVERIFICATION')
    lines, ok = verify(df, per_mouse, ce_vs_kl_identity(dirs, args.mice))
    print('\n'.join(lines))
    print('\nMEDIAN TABLE (lambda_H = 0)')
    print(median_table(df, 0.0))
    if not ok:
        raise SystemExit('VERIFICATION FAILED — see lines above.')
    return df


if __name__ == '__main__':
    main()
