# -*- coding: utf-8 -*-
"""Scan every (cell, loss) in hpsweep_v2 + prodfix_v1 and cache the three selection metrics.

For each cell x loss x architecture we record, over the 6 mice:
  * decoded peakiness (mean max-prob) and the IO target peakiness on the same trials
  * KL and projection normalised loss vs the LOO predict-mean null (<1 beats chance)
  * overfitting = final val fit-loss / final train fit-loss, from the saved checkpoints

Writes one long CSV (one row per cell x loss x arch) so the ranking in
`hp_best_select.py` is a cheap re-read rather than an 8 GB re-scan.

Usage:  python diagnostics/hp_best_scan.py --out <csv>
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import hpsweep_spec as S  # noqa: E402
from performance_vs_hparams import _norm_by_mouse  # noqa: E402

ARCHS = ['spat', 'temp']


def _peaky(res, arch, key='decoded'):
    out = []
    for m in sorted(res):
        try:
            d = res[m]['Dist'][arch][key]
        except Exception:
            continue
        out.append(float(np.asarray(d, float).max(1).mean()))
    return np.array(out, float)


def _overfit(slug_dir, arch):
    rs = []
    for pt in sorted((slug_dir / 'checkpoints').glob('mouse_*_stratified_balanced.pt')):
        try:
            ck = torch.load(str(pt), map_location='cpu', weights_only=False)
        except Exception:
            continue
        h = (ck.get(arch) or {}).get('history') or {}
        t, v = h.get('train_fit_loss'), h.get('val_fit_loss')
        if t and v and t[-1] > 0:
            rs.append(v[-1] / t[-1])
    return np.array(rs, float)


def _ms(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if v.size == 0:
        return np.nan, np.nan, 0
    return v.mean(), (v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0), v.size


def _cells():
    """[(run, cell, loss, slug, hp-dict)] for every available cell x loss."""
    out = []
    man = HERE.parent / 'results' / 'hpsweep_v2' / 'MANIFEST.csv'
    with open(man) as fh:
        for row in csv.DictReader(fh):
            slug = S.LOSS_SLUG[row['loss']]
            d = HERE.parent / 'results' / 'hpsweep_v2' / row['cell'] / slug
            if not (d / 'stratified_balanced.mat').is_file():
                continue
            hp = dict(entropy_lambda=row['entropy_lambda'], dropout=row['dropout'],
                      activation=row['activation'], hidden_width=row['hidden_width'],
                      patience=row['patience'], val_fraction=row['val_fraction'],
                      weight_decay=row['weight_decay'], shape_lambda=row['shape_lambda'])
            out.append(('hpsweep_v2', row['cell'], row['loss'], slug, hp))
    prod = HERE.parent / 'results' / 'prodfix_v1'
    import yaml
    for cd in sorted(p for p in prod.iterdir() if p.is_dir()):
        subs = [p for p in cd.iterdir() if p.is_dir()]
        if len(subs) != 1 or not (subs[0] / 'stratified_balanced.mat').is_file():
            continue
        sd = subs[0]
        cfg = yaml.safe_load(open(sd / 'config.yaml'))
        hp = dict(entropy_lambda=cfg.get('entropy_lambda'), dropout=cfg.get('dropout'),
                  activation=cfg.get('activation_function'),
                  hidden_width=(cfg.get('hidden_sizes') or [0])[0] if cfg.get('hidden_sizes') else 0,
                  patience=cfg.get('patience'), val_fraction=cfg.get('val_fraction'),
                  weight_decay=cfg.get('weight_decay'), shape_lambda=cfg.get('shape_lambda'),
                  smooth_lambda=cfg.get('smooth_lambda'))
        out.append(('prodfix_v1', cd.name, cfg.get('loss_func'), sd.name, hp))
    return out


FIELDS = ['run', 'cell', 'loss', 'slug', 'arch', 'n_mice',
          'peaky', 'peaky_sem', 'tgt_peaky', 'nl_KL', 'nl_KL_sem',
          'nl_PCA', 'nl_PCA_sem', 'overfit', 'overfit_sem',
          'entropy_lambda', 'dropout', 'activation', 'hidden_width',
          'patience', 'val_fraction', 'weight_decay', 'shape_lambda', 'smooth_lambda']


def main(out_csv):
    cells = _cells()
    print(f'{len(cells)} cell x loss combinations found', flush=True)
    rows = []
    t0 = time.time()
    for i, (run, cell, loss, slug, hp) in enumerate(cells):
        sd = HERE.parent / 'results' / run / cell / slug
        try:
            res = sio.loadmat(str(sd / 'stratified_balanced.mat'),
                              simplify_cells=True)['results']
        except Exception as e:
            print(f'  skip {run}/{cell}/{slug}: {e}', flush=True)
            continue
        for arch in ARCHS:
            per = _norm_by_mouse(res, arch)
            pk = _peaky(res, arch, 'decoded')
            tp = _peaky(res, arch, 'target')
            if pk.size == 0:
                continue
            pkm, pks, n = _ms(pk)
            klm, kls, _ = _ms(per[('KL', 'pm')])
            pcm, pcs_, _ = _ms(per[('PCA', 'pm')])
            ofm, ofs, _ = _ms(_overfit(sd, arch))
            rows.append({'run': run, 'cell': cell, 'loss': loss, 'slug': slug, 'arch': arch,
                         'n_mice': n, 'peaky': pkm, 'peaky_sem': pks,
                         'tgt_peaky': float(np.nanmean(tp)) if tp.size else np.nan,
                         'nl_KL': klm, 'nl_KL_sem': kls, 'nl_PCA': pcm, 'nl_PCA_sem': pcs_,
                         'overfit': ofm, 'overfit_sem': ofs,
                         **{k: hp.get(k) for k in FIELDS[15:]}})
        if (i + 1) % 10 == 0:
            print(f'  {i+1}/{len(cells)}  ({time.time()-t0:.0f}s)', flush=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})
    print(f'wrote {len(rows)} rows -> {out_csv}  ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default='/tmp/hp_best_scan.csv')
    a = ap.parse_args()
    main(a.out)
