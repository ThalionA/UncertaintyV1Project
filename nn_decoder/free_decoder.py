# -*- coding: utf-8 -*-
"""Unconstrained ("free") spatiotemporal decoders.

The two production decoders (``ppc`` and ``sampling`` in
``nn_classifier``) are *deliberately constrained* readouts of the same
``(trial, T, n_neurons)`` activity slab — that constraint **is** the
PPC-vs-SBC hypothesis:

  * **PPC** averages over time *first* (``mean_t r`` → MLP → softmax), so
    it can only see the time-collapsed population vector.
  * **SBC** decodes each bin independently with shared weights, then
    averages the per-bin *output distributions*, so it can only see one
    bin at a time.

Neither is allowed to read the trial "any way it wants": both throw away
cross-time / cross-neuron joint structure. This module adds a third,
*unconstrained* family that maps the full ``(T × n_neurons)`` slab to a
single per-trial posterior with no spatial-first / temporal-first bias
baked in. Scientifically it is an **information ceiling**: if neither
PPC nor SBC reaches the free decoder, both constrained hypotheses are
discarding decodable structure; if they match it, the constraint is free.

Three flavours, all sharing the same I/O contract
``forward(x: (B, T, n_neurons)) -> logits (B, output_size)`` so they drop
straight into the ``model_type='free'`` branch of
``nn_classifier._batched_predict`` (softmax applied there, exactly like
the other arms):

  * ``mlp``  — flatten ``(T × n_neurons)`` → MLP. Zero inductive bias
    toward space or time; the most literal "read it any way it wants".
  * ``gru``  — a GRU over the time axis, decode from the last hidden
    state. Keeps the per-step input width at ``n_neurons`` and scales to
    longer ``T``, but assumes temporal ordering matters.
  * ``tcn``  — 1-D convolution across time over the ``n_neurons``
    channels → global average pool → linear head. Captures local
    spatiotemporal motifs with far fewer parameters than the flatten-MLP;
    mild locality bias.

Unlike SBC, none of these carries the instantaneous-entropy (sharpness)
penalty — there is a single output distribution per trial, so there is
nothing per-bin to regularise. The PCA / MSE / CE fit-loss machinery in
``nn_classifier`` applies unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init


VALID_FREE_ARCHS = ('mlp', 'gru', 'tcn')


def _activation_module(name: str) -> nn.Module:
    """Map an activation name to a module, defaulting to ReLU to match
    ``SimpleFlexibleNNClassifier``'s fallback behaviour."""
    return {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
    }.get(name.lower(), nn.ReLU())


class FreeMLP(nn.Module):
    """Flatten the whole ``(T × n_neurons)`` slab into one vector and run
    it through an MLP. The decoder sees every (neuron, time-bin) entry
    jointly with no architectural prior — the purest "unconstrained
    reader" and the natural information ceiling.

    Parameters
    ----------
    n_neurons, T : int
        Population size and number of time bins per trial. The flattened
        input width is ``n_neurons * T``.
    hidden_sizes : list[int]
        Hidden-layer widths. Mirrors ``SimpleFlexibleNNClassifier`` so the
        capacity knob is the same one used for PPC/SBC.
    output_size : int
        Number of output bins (91 for Q/L, 2 for the decision posterior).
    activation : str
        ``'relu' | 'tanh' | 'sigmoid'`` (default matches the production
        ``tanh``).
    """

    def __init__(self, n_neurons, T, hidden_sizes, output_size,
                 activation='tanh'):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        self.n_neurons = int(n_neurons)
        self.T = int(T)
        input_size = self.n_neurons * self.T
        self.activation = _activation_module(activation)

        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        self.layers.extend(
            nn.Linear(h1, h2) for h1, h2 in zip(hidden_sizes[:-1], hidden_sizes[1:]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

        for layer in self.layers:
            init.xavier_uniform_(layer.weight)
            init.constant_(layer.bias, 0)

    def forward(self, x):
        # x: (B, T, n_neurons) -> (B, T*n_neurons)
        x = x.reshape(x.shape[0], -1)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class FreeGRU(nn.Module):
    """Recurrent reader: a GRU consumes the bins in order and the final
    hidden state is projected to the output logits. Retains the temporal
    ordering the flatten-MLP discards, at the cost of a temporal-ordering
    inductive bias.

    Parameters
    ----------
    n_neurons : int
        Per-step input width (the population is the GRU's feature vector
        at each bin).
    hidden_size : int
        GRU hidden width. Taken from ``hidden_sizes[0]`` by the factory.
    output_size : int
        Number of output bins.
    num_layers : int
        Stacked GRU layers (default 1).
    """

    def __init__(self, n_neurons, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.n_neurons = int(n_neurons)
        self.gru = nn.GRU(
            input_size=self.n_neurons,
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
        )
        self.head = nn.Linear(int(hidden_size), output_size)
        init.xavier_uniform_(self.head.weight)
        init.constant_(self.head.bias, 0)

    def forward(self, x):
        # x: (B, T, n_neurons). GRU is batch_first, so this is its native
        # layout. Decode from the last layer's final-step hidden state.
        _, h_n = self.gru(x)
        last_hidden = h_n[-1]            # (B, hidden_size)
        return self.head(last_hidden)


class FreeTCN(nn.Module):
    """Temporal convolutional reader: 1-D convolutions slide across the
    time axis over the ``n_neurons`` input channels, then a global average
    pool over time feeds a linear head. Captures local spatiotemporal
    motifs with far fewer parameters than the flatten-MLP; the only prior
    is locality + translation-in-time.

    Parameters
    ----------
    n_neurons : int
        Input channel count (one channel per neuron).
    T : int
        Number of time bins (kernel size is clamped to ``<= T``).
    channels : int
        Convolutional feature width. Taken from ``hidden_sizes[0]``.
    output_size : int
        Number of output bins.
    kernel_size : int
        Temporal receptive field of each conv (default 3, clamped to T).
    activation : str
        Activation between conv layers.
    """

    def __init__(self, n_neurons, T, channels, output_size,
                 kernel_size=3, activation='tanh'):
        super().__init__()
        self.n_neurons = int(n_neurons)
        self.T = int(T)
        # A kernel wider than the sequence is meaningless; clamp it (and
        # keep it >= 1). With 'same' padding the temporal length is
        # preserved regardless, but clamping avoids a degenerate conv.
        k = max(1, min(int(kernel_size), self.T))
        pad = k // 2
        self.conv1 = nn.Conv1d(self.n_neurons, int(channels), k, padding=pad)
        self.conv2 = nn.Conv1d(int(channels), int(channels), k, padding=pad)
        self.activation = _activation_module(activation)
        self.head = nn.Linear(int(channels), output_size)

        for conv in (self.conv1, self.conv2):
            init.xavier_uniform_(conv.weight)
            init.constant_(conv.bias, 0)
        init.xavier_uniform_(self.head.weight)
        init.constant_(self.head.bias, 0)

    def forward(self, x):
        # x: (B, T, n_neurons) -> Conv1d wants (B, C=n_neurons, L=T)
        x = x.transpose(1, 2)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = x.mean(dim=-1)               # global average pool over time
        return self.head(x)


def build_free_model(arch, *, n_neurons, T, output_size,
                     hidden_sizes=(32,), activation='tanh',
                     num_layers=1, kernel_size=3):
    """Construct a free spatiotemporal decoder by name.

    Parameters
    ----------
    arch : {'mlp', 'gru', 'tcn'}
        Which unconstrained reader to build.
    n_neurons, T, output_size : int
        Population size, bins per trial, output-bin count.
    hidden_sizes : sequence[int]
        Hidden widths. The GRU and TCN use ``hidden_sizes[0]`` as their
        single width; the MLP uses the full list.
    activation : str
        Activation for the MLP / TCN (the GRU uses its built-in gates).
    num_layers : int
        GRU depth (ignored by mlp/tcn).
    kernel_size : int
        TCN temporal kernel (ignored by mlp/gru).

    Returns
    -------
    torch.nn.Module
        A module with the shared ``(B, T, n_neurons) -> (B, output_size)``
        forward contract.
    """
    if isinstance(hidden_sizes, int):
        hidden_sizes = [hidden_sizes]
    hidden_sizes = list(hidden_sizes)

    if arch == 'mlp':
        return FreeMLP(n_neurons, T, hidden_sizes, output_size, activation)
    if arch == 'gru':
        return FreeGRU(n_neurons, hidden_sizes[0], output_size,
                       num_layers=num_layers)
    if arch == 'tcn':
        return FreeTCN(n_neurons, T, hidden_sizes[0], output_size,
                       kernel_size=kernel_size, activation=activation)
    raise ValueError(
        f"Unknown free arch {arch!r}; valid: {VALID_FREE_ARCHS}")
