# -*- coding: utf-8 -*-
"""Standalone runner for the unconstrained ("free") spatiotemporal decoder.

This is the **information-ceiling** companion to the production PPC/SBC
pipeline. Where ``run_experiment.run_animal_decoder`` trains the two
*constrained* arms (spatial-first PPC, temporal-sample SBC), this runner
trains decoders that may read the whole ``(T × n_neurons)`` trial slab
"any way they want" and emit a single per-trial posterior. Three flavours
(see ``free_decoder.py``):

  * ``mlp`` — flatten → MLP (no spatial/temporal prior)
  * ``gru`` — recurrent over time
  * ``tcn`` — temporal convolution

It deliberately reuses ``run_experiment.prepare_trial_tensors`` so the
free arms see byte-identical data, splits, z-scoring and PCA bases to the
production decoders — the only thing that changes is *how the trial is
read*. It does **not** touch ``run_animal_decoder`` or the production
output tree; you opt into the ceiling by running this script.

Targets supported: ``Q`` (perceptual posterior), ``L`` (marginalised
likelihood), ``d`` (decision posterior) — exactly the three the question
asked for. Each arm is trained on the real target and on a matched
trial-shuffled control, so the free losses normalise to the same kind of
shuffled baseline the production decoders use.

Losses: the distributional targets (Q, L) are fit under **PCA (basis fit
on all training trials, never condition means), KL, and JS** — one model
per (architecture × loss); the decision posterior (d) uses MSE. The free
arm carries **no entropy / sharpness penalty** (it emits one distribution
per trial, so there is nothing per-bin to regularise).

Output tree (mirrors ``training/run.py`` conventions):

    <results_root>/<run_name>/free_<slug>/
        config.yaml
        <split>.mat        # one per split, keyed results['mouse_<id>']

Each animal's results dict holds, per ``free_<arch>`` (and its
``_shf`` control): per-trial ``decoded`` distributions, the matched
``target``, ``full_decoded`` over all trials, and per-trial ``fit_loss``.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.io as sio
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from free_decoder import VALID_FREE_ARCHS
from nn_classifier import (
    train_and_select_best_model,
    get_model_probabilities,
    _batched_fit_loss,
)
from run_experiment import prepare_trial_tensors, default_device
from training.config import Config, default_config_for_target


# Free arms get more capacity than the constrained MLPs by default: the
# flatten-MLP's input is T*n_neurons wide, and the whole point of the
# ceiling is to let the decoder use as much of the trial as it can. These
# are deliberately un-tuned starting points — sweep per target if the
# ceiling matters quantitatively.
DEFAULT_FREE_HIDDEN = [128, 64]

# Loss functions for the free ceiling on the distributional targets (Q, L).
# PCA is fit on ALL training trials (never condition means) — see the
# forced pca_basis below — and KL/JS are the two divergences we compare it
# against. The decision posterior (d, 2-bin) has no PCA basis, so it uses
# MSE; default_losses_for_target picks the right set.
DEFAULT_DISTRIBUTIONAL_LOSSES = ('PCA', 'KL', 'JS')
DEFAULT_DECISION_LOSSES = ('MSE',)


def default_losses_for_target(target_type):
    """Loss set appropriate to a target. Q/L are 91-bin distributions
    (PCA-all-trials / KL / JS); d is the 2-bin decision posterior (MSE,
    since PCA/divergences on two bins are degenerate)."""
    return (DEFAULT_DECISION_LOSSES if target_type == 'd'
            else DEFAULT_DISTRIBUTIONAL_LOSSES)


def _eval_free_arm(model, loader, *, loss_func, pcs, explained_variance):
    """Forward every trial in ``loader`` through a free model.

    ``loader`` yields one trial per batch as ``(T, n_neurons)`` activity
    and ``(T, n_cats)`` target (the prepare-stage convention). Returns
    ``(decoded, targets, fit_losses)`` as numpy arrays:

      decoded     : (n_trials, n_cats)  per-trial predicted posterior
      targets     : (n_trials, n_cats)  per-trial target (mean over T)
      fit_losses  : (n_trials,)         per-trial divergence (no penalty)
    """
    decoded, targets, fit_losses = [], [], []
    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            pred = get_model_probabilities(model, batch_inputs, 'free')  # (1, n_cats)
            target_mean = batch_targets.mean(dim=0, keepdim=True)        # (1, n_cats)
            fit = _batched_fit_loss(pred, target_mean, loss_func,
                                    pcs, explained_variance)             # (1,)
            decoded.append(pred.cpu().numpy())
            targets.append(target_mean.cpu().numpy())
            fit_losses.append(float(fit.item()))
    return (np.vstack(decoded), np.vstack(targets),
            np.asarray(fit_losses, dtype=float))


def run_free_animal(config, mouse_id, *, archs=VALID_FREE_ARCHS,
                    losses=None, free_hidden_sizes=None, neuron_subset=None,
                    preloaded=None):
    """Train the requested free arms (+ shuffle controls) for one animal,
    one model per (architecture × loss).

    Parameters
    ----------
    config : dict
        Legacy config dict (``Config.to_legacy_dict``). Supplies the
        target, window, bin size, split, and optimiser schedule — shared
        verbatim with the production pipeline. Its ``custom_loss_func`` is
        ignored in favour of ``losses`` (the free ceiling sweeps a fixed
        set). The PCA basis is **forced to 'all_trials'** here regardless
        of what the config carried.
    mouse_id : int
        Animal index.
    archs : sequence of {'mlp', 'gru', 'tcn'}
        Which free architectures to train.
    losses : sequence of {'PCA', 'KL', 'JS', 'MSE', ...} or None
        Loss functions to fit each arch under. ``None`` picks the
        target-appropriate default (PCA-all-trials / KL / JS for the
        distributional Q/L targets, MSE for the 2-bin decision posterior).
        PCA always uses the all-trials basis. There is **no entropy /
        sharpness penalty** on the free arm — it emits a single
        distribution per trial, so there is nothing per-bin to regularise.
    free_hidden_sizes : list[int] or None
        Hidden widths for the free arms. Defaults to
        :data:`DEFAULT_FREE_HIDDEN`. (The GRU/TCN use the first entry.)
    neuron_subset, preloaded :
        Forwarded to :func:`prepare_trial_tensors`.

    Returns
    -------
    dict
        ``{'fit_loss': {...}, 'Dist': {...}, 'trials': ..., 'trial_cats': ...}``
        with one ``free_<arch>_<loss>`` and one ``free_<arch>_<loss>_shf``
        key per (arch, loss) pair.
    """
    for a in archs:
        if a not in VALID_FREE_ARCHS:
            raise ValueError(f"Unknown free arch {a!r}; valid {VALID_FREE_ARCHS}")
    if losses is None:
        losses = default_losses_for_target(config.get('target_type', 'Q'))
    if free_hidden_sizes is None:
        free_hidden_sizes = list(DEFAULT_FREE_HIDDEN)

    # The free ceiling fits its PCA loss on ALL training trials, never on
    # condition means — pin it here so the basis is independent of whatever
    # the incoming config defaulted to.
    config = dict(config)
    config['pca_basis'] = 'all_trials'

    prepared = prepare_trial_tensors(config, mouse_id,
                                     neuron_subset=neuron_subset,
                                     preloaded=preloaded)

    # PCA loss is undefined on a 2-bin target (no orientation PCA basis is
    # built when N_cats == 2). Fail loud rather than silently fall through
    # to cross-entropy the way the batched loss would.
    if prepared.pcs is None and 'PCA' in losses:
        raise ValueError(
            f"PCA loss requested for a {prepared.N_cats}-bin target, but a "
            f"PCA basis is only built for distributional (>2-bin) targets. "
            f"Use MSE for the decision posterior.")

    model_params = {
        'input_size': prepared.input_size,        # neuron count (per-step width)
        'T': prepared.T,                          # bins/trial (flatten-MLP / TCN sizing)
        'hidden_sizes': free_hidden_sizes,
        'output_size': prepared.output_size,
        'activation_function': config.get('activation_function', 'tanh'),
        # free_arch is set per-arch in the loop below.
    }
    base_training_params = {
        'num_epochs': config['num_epochs'],
        'learning_rate': config['learning_rate'],
        'weight_decay': config.get('weight_decay', 1e-4),
        'minibatch_size': config['minibatch_size'],
        'momentum': config.get('momentum', 0.9),
        'device': default_device,
        # 'free' has NO sharpness penalty (single output per trial), but
        # train_and_select_best_model reads this key — pass 0.0 so the
        # penalty term is identically zero.
        'entropy_lambda': 0.0,
        'pcs': prepared.pcs,
        'explained_variance': prepared.explained_variance,
        'optimizer_type': config.get('optimizer_type', 'adam'),
    }
    REP = config['REP']

    Distr = {}
    fit_loss = {}
    for loss_func in losses:
        training_params = dict(base_training_params, loss_func=loss_func)
        for arch in archs:
            real_key = f'free_{arch}_{loss_func}'
            shf_key = f'{real_key}_shf'

            mp = dict(model_params, free_arch=arch)

            print(f"      [free:{arch}/{loss_func}] training real...")
            best_real, _ = train_and_select_best_model(
                REP, 'free', prepared.train_loader, mp, training_params, verbose=False)
            print(f"      [free:{arch}/{loss_func}] training shuffled...")
            best_shf, _ = train_and_select_best_model(
                REP, 'free', prepared.train_loader_shuffle, mp, training_params, verbose=False)

            for key, model in ((real_key, best_real), (shf_key, best_shf)):
                dec, targ, fl = _eval_free_arm(
                    model, prepared.test_loader, loss_func=loss_func,
                    pcs=prepared.pcs, explained_variance=prepared.explained_variance)
                full_dec, _, _ = _eval_free_arm(
                    model, prepared.full_loader, loss_func=loss_func,
                    pcs=prepared.pcs, explained_variance=prepared.explained_variance)
                Distr[key] = {'decoded': dec, 'target': targ, 'full_decoded': full_dec}
                fit_loss[key] = fl

    Distr['pcs'] = (prepared.pcs.cpu().numpy()
                    if prepared.pcs is not None else [])
    Distr['explained_var'] = (prepared.explained_variance.cpu().numpy()
                              if prepared.explained_variance is not None else [])

    return {
        'fit_loss': fit_loss,
        'Dist': Distr,
        'trials': prepared.trials_out,
        'trial_cats': prepared.trial_cats_out,
    }


def run_free_config(
    config: Config,
    mouse_ids: Iterable[int] = range(6),
    splits: Iterable[str] = ('stratified_balanced',
                             'generalize_contrast',
                             'generalize_dispersion'),
    archs: Sequence[str] = VALID_FREE_ARCHS,
    losses: Sequence[str] = None,
    free_hidden_sizes=None,
    results_root: str = 'results',
    on_error: str = 'continue',
) -> Path:
    """Run a Config's free arms across (mice × splits), saving per-split
    ``.mat`` files plus a ``config.yaml`` provenance dump under a
    ``free_<slug>`` directory.

    Mirrors ``training.run.run_config`` but trains only the free arms (one
    per architecture × loss), and never writes to the production decoder
    tree. ``losses`` defaults to the target-appropriate set
    (PCA-all-trials / KL / JS for Q/L, MSE for d).
    """
    if on_error not in ('continue', 'raise'):
        raise ValueError(f"on_error must be 'continue' or 'raise', got {on_error!r}")

    if losses is None:
        losses = default_losses_for_target(config.target_type)

    legacy_base = config.to_legacy_dict()

    # free_<slug> sibling of the production slug so the ceiling can coexist.
    out_dir = config.output_dir(results_root)
    out_dir = out_dir.parent / f'free_{out_dir.name}'
    out_dir.mkdir(parents=True, exist_ok=True)
    config.save_yaml(out_dir / 'config.yaml')

    for split in splits:
        cfg = dict(legacy_base)
        cfg['split_type'] = split

        print()
        print('=' * 60)
        print(f"FREE DECODER: {config.target_type} "
              f"| {config.time_window} {config.bin_size_ms}ms "
              f"| archs={tuple(archs)} | losses={tuple(losses)} "
              f"| split = {split}")
        print('=' * 60)

        all_mice_results = {}
        run_failed = False
        for mid in mouse_ids:
            print(f"  --> Processing Mouse {mid}...")
            try:
                res = run_free_animal(cfg, mid, archs=archs, losses=losses,
                                      free_hidden_sizes=free_hidden_sizes)
                all_mice_results[f"mouse_{mid}"] = res
            except Exception as exc:
                print(f"  [!] Failed for Mouse {mid}: {exc}")
                traceback.print_exc()
                if on_error == 'raise':
                    raise
                run_failed = True

        if all_mice_results:
            save_path = out_dir / f'{split}.mat'
            sio.savemat(str(save_path),
                        {'results': all_mice_results, 'config': cfg,
                         'archs': list(archs), 'losses': list(losses)})
            tag = ' (partial)' if run_failed else ''
            print(f"Saved {save_path.name}{tag} ({len(all_mice_results)} animals)")
        else:
            print(f"No animals succeeded for split={split}; not writing.")

    return out_dir


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', default='Q', choices=['Q', 'L', 'd'],
                   help="Decoding target: Q (perceptual posterior), "
                        "L (likelihood), d (decision posterior).")
    p.add_argument('--bin-size-ms', type=int, default=100)
    p.add_argument('--archs', nargs='+', default=list(VALID_FREE_ARCHS),
                   choices=list(VALID_FREE_ARCHS))
    p.add_argument('--losses', nargs='+', default=None,
                   choices=['PCA', 'KL', 'JS', 'MSE', 'CE', 'Wasserstein'],
                   help="Loss functions to fit each arch under. Default: "
                        "PCA(all-trials) KL JS for Q/L, MSE for d. PCA always "
                        "uses the all-trials basis; the free arm never carries "
                        "an entropy penalty.")
    p.add_argument('--mice', nargs='+', type=int, default=list(range(6)))
    p.add_argument('--splits', nargs='+', default=['stratified_balanced'])
    p.add_argument('--run-name', default='free_reference')
    p.add_argument('--results-root', default='results')
    p.add_argument('--hidden', nargs='+', type=int, default=None,
                   help="Hidden sizes for the free arms (default 128 64).")
    args = p.parse_args(argv)

    config = default_config_for_target(
        args.target, bin_size_ms=args.bin_size_ms, run_name=args.run_name)
    run_free_config(
        config, mouse_ids=args.mice, splits=args.splits, archs=args.archs,
        losses=args.losses, free_hidden_sizes=args.hidden,
        results_root=args.results_root)


if __name__ == '__main__':
    main()
