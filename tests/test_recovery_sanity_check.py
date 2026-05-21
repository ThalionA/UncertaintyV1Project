# -*- coding: utf-8 -*-
"""Integration test for ``recovery_sanity_check`` against a realistic
checkpoint bundle.

The .pt files written by ``training.run._pop_and_save_checkpoints`` hold
the FULL Checkpoints bundle — ``{'spat': <ckpt>, 'temp': <ckpt>}`` — not
a single per-arch checkpoint. An earlier version of
``recovery_sanity_check.main`` assumed the latter and crashed with
``KeyError: 'loss_func'`` on real output. This test pins the bundle
layout so that regression can't recur.

It builds a bundle exactly the way the production pipeline does
(``_extract_checkpoint`` per arch, saved together under one .pt), writes
a matching .mat with the new ``fit_loss`` field, and runs the full
``main()`` driver end-to-end, asserting:
  * Both arches (spat, temp) produce a summary row.
  * Identity fit-loss and |Δ pred_probs| land at the fp32 floor.
  * The CSV and all three SVG plots are written.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NN_DECODER = os.path.join(REPO_ROOT, 'nn_decoder')
if NN_DECODER not in sys.path:
    sys.path.insert(0, NN_DECODER)

torch = pytest.importorskip('torch')
scipy_io = pytest.importorskip('scipy.io')

from nn_classifier import (  # noqa: E402
    SimpleFlexibleNNClassifier,
    get_model_probabilities,
)
from run_experiment import _extract_checkpoint  # noqa: E402
import recovery_sanity_check as rsc  # noqa: E402


N_NEURONS, N_CATS, T_BINS, N_TRIALS = 8, 5, 4, 6


def _build_bundle():
    """Build a {'spat': ckpt, 'temp': ckpt} bundle the way
    run_animal_decoder's Checkpoints field is shaped."""
    model_params = {'input_size': N_NEURONS, 'hidden_sizes': [12],
                     'output_size': N_CATS, 'activation_function': 'tanh'}
    bundle = {}
    for arch, model_type, seed in [('spat', 'ppc', 1), ('temp', 'sampling', 2)]:
        torch.manual_seed(seed)
        model = SimpleFlexibleNNClassifier(
            input_size=N_NEURONS, hidden_sizes=[12],
            output_size=N_CATS, activation='tanh',
        )
        g = torch.Generator().manual_seed(seed + 100)
        batches = [torch.randn(T_BINS, N_NEURONS, generator=g)
                   for _ in range(N_TRIALS)]
        with torch.no_grad():
            pps = [get_model_probabilities(model, b, model_type) for b in batches]
        bundle[arch] = _extract_checkpoint(
            model=model, batches=batches, pred_probs_list=pps,
            model_params=model_params, model_type=model_type,
            pcs=None, explained_var=None,
            loss_func='MSE', entropy_lambda=0.003,
        )
    return bundle


def _write_run_tree(tmp_path):
    """Lay out a results/<run>/<slug>/ tree with a bundle .pt under
    checkpoints/ and a matching .mat carrying the new fit_loss field."""
    run_dir = tmp_path / 'verif_run'
    slug_dir = run_dir / 'd_MSE_half_100ms'
    (slug_dir / 'checkpoints').mkdir(parents=True)

    bundle = _build_bundle()
    torch.save(bundle, str(slug_dir / 'checkpoints' /
                            'mouse_0_stratified_balanced.pt'))

    # Matching .mat: new-format fit_loss dict keyed by arch.
    scipy_io.savemat(str(slug_dir / 'stratified_balanced.mat'), {
        'results': {'mouse_0': {
            'fit_loss': {'spat': [0.05, 0.06, 0.07],
                          'temp': [0.04, 0.05, 0.06]},
        }},
        'config': {'custom_loss_func': 'MSE'},
    })
    return run_dir


def test_main_processes_both_arches_from_bundle(tmp_path):
    """The headline regression test: a bundle .pt must yield one row
    per arch, not a KeyError."""
    run_dir = _write_run_tree(tmp_path)
    out_dir = tmp_path / 'figs'
    rsc.main(run_dir=str(run_dir), out_dir=str(out_dir))

    csv_path = out_dir / 'sanity_check_summary.csv'
    assert csv_path.exists()
    import csv as _csv
    with open(csv_path) as f:
        rows = list(_csv.DictReader(f))
    arches = sorted(r['arch'] for r in rows)
    assert arches == ['spat', 'temp'], (
        f"expected one row per arch, got {arches}"
    )


def test_main_identity_loss_is_fp32_floor(tmp_path):
    """Round-trip on a freshly-built checkpoint must give identity
    residual and |Δ pred_probs| at the fp32 floor. (For these MSE cells
    the residual equals the raw fit-loss, since the MSE baseline is 0.)"""
    run_dir = _write_run_tree(tmp_path)
    out_dir = tmp_path / 'figs'
    rsc.main(run_dir=str(run_dir), out_dir=str(out_dir))

    import csv as _csv
    with open(out_dir / 'sanity_check_summary.csv') as f:
        rows = list(_csv.DictReader(f))
    for r in rows:
        assert abs(float(r['identity_residual_max'])) < 1e-4, r
        assert float(r['max_pred_diff']) < 1e-4, r


def test_main_writes_all_plots(tmp_path):
    """All three SVGs plus the CSV must be emitted."""
    run_dir = _write_run_tree(tmp_path)
    out_dir = tmp_path / 'figs'
    rsc.main(run_dir=str(run_dir), out_dir=str(out_dir))

    for name in ('sanity_check_summary.csv',
                  'identity_fit_loss_strip.svg',
                  'compare_identity_recovery_realdata.svg',
                  'pred_probs_diff_hist.svg'):
        assert (out_dir / name).exists(), f"missing {name}"


def test_main_reads_real_data_fit_loss_from_mat(tmp_path):
    """The real-data column must be populated from the .mat's fit_loss
    field (mean of the per-trial array)."""
    run_dir = _write_run_tree(tmp_path)
    out_dir = tmp_path / 'figs'
    rsc.main(run_dir=str(run_dir), out_dir=str(out_dir))

    import csv as _csv
    with open(out_dir / 'sanity_check_summary.csv') as f:
        rows = {r['arch']: r for r in _csv.DictReader(f)}
    # spat fit_loss array was [0.05, 0.06, 0.07] -> mean 0.06.
    assert abs(float(rows['spat']['real_data_fit_loss']) - 0.06) < 1e-9
    # temp fit_loss array was [0.04, 0.05, 0.06] -> mean 0.05.
    assert abs(float(rows['temp']['real_data_fit_loss']) - 0.05) < 1e-9


def test_main_tolerates_missing_recovery_cache(tmp_path):
    """If no recovery cache exists for the target, recovery_fit_loss is
    blank rather than crashing — step 4 must run before step 3 has
    happened."""
    run_dir = _write_run_tree(tmp_path)
    out_dir = tmp_path / 'figs'
    # d -> 'decision'; if a real recovery_cache_fixed_decision.npy
    # happens to exist in the repo it'll be read, otherwise blank.
    # Either way main() must not raise.
    rsc.main(run_dir=str(run_dir), out_dir=str(out_dir))
    assert (out_dir / 'sanity_check_summary.csv').exists()


# ----------------------------------------------------------------------
# Cross-entropy: identity residual must be ~0 even though the raw
# identity fit-loss equals H(pred) > 0.
# ----------------------------------------------------------------------

def _build_ce_bundle():
    """Bundle of CE-trained checkpoints. CE(p,p) = H(p) ≠ 0, so the raw
    identity fit-loss is large even on a perfect round-trip — the
    residual (fit − baseline) is the metric that must stay at 0."""
    model_params = {'input_size': N_NEURONS, 'hidden_sizes': [12],
                     'output_size': N_CATS, 'activation_function': 'tanh'}
    bundle = {}
    for arch, model_type, seed in [('spat', 'ppc', 7), ('temp', 'sampling', 8)]:
        torch.manual_seed(seed)
        model = SimpleFlexibleNNClassifier(
            input_size=N_NEURONS, hidden_sizes=[12],
            output_size=N_CATS, activation='tanh',
        )
        g = torch.Generator().manual_seed(seed + 100)
        batches = [torch.randn(T_BINS, N_NEURONS, generator=g)
                   for _ in range(N_TRIALS)]
        with torch.no_grad():
            pps = [get_model_probabilities(model, b, model_type) for b in batches]
        bundle[arch] = _extract_checkpoint(
            model=model, batches=batches, pred_probs_list=pps,
            model_params=model_params, model_type=model_type,
            pcs=None, explained_var=None,
            loss_func='CE', entropy_lambda=0.003,
        )
    return bundle


def test_ce_identity_residual_is_zero_despite_nonzero_fit_loss(tmp_path):
    """The regression test for the CE false-FAIL bug. For a CE cell on a
    bit-perfect round-trip:
      * raw identity_fit_loss is large (≈ H(pred), several nats);
      * identity_baseline equals it (CE(p,p) = H(p));
      * identity_residual — their difference — is ~0.
    Verdict must be driven by the residual, not the raw fit-loss."""
    run_dir = tmp_path / 'ce_run'
    slug_dir = run_dir / 'stim_cat_CE_half_100ms'
    (slug_dir / 'checkpoints').mkdir(parents=True)
    torch.save(_build_ce_bundle(),
               str(slug_dir / 'checkpoints' / 'mouse_0_stratified_balanced.pt'))
    scipy_io.savemat(str(slug_dir / 'stratified_balanced.mat'), {
        'results': {'mouse_0': {'fit_loss': {'spat': [1.5], 'temp': [1.6]}}},
        'config': {'custom_loss_func': 'CE'},
    })
    out_dir = tmp_path / 'figs'
    rsc.main(run_dir=str(run_dir), out_dir=str(out_dir))

    import csv as _csv
    with open(out_dir / 'sanity_check_summary.csv') as f:
        rows = list(_csv.DictReader(f))
    assert rows, "no rows produced"
    for r in rows:
        # Round-trip drift must be at the fp32 floor.
        assert abs(float(r['identity_residual_max'])) < 1e-4, r
        assert float(r['max_pred_diff']) < 1e-4, r
        # Raw CE identity fit-loss is H(pred): strictly positive, and
        # essentially equal to the baseline (CE(p,p) = H(p)).
        fit = float(r['identity_fit_loss_mean'])
        base = float(r['identity_baseline_mean'])
        assert fit > 0.1, f"expected CE fit-loss ≈ H(pred) > 0, got {fit}"
        assert abs(fit - base) < 1e-4, (
            f"CE(p,p) should equal its own baseline: fit={fit} base={base}"
        )
