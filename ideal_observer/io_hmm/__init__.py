# -*- coding: utf-8 -*-
"""IO-HMM: Hidden Markov Model with state-indexed Ideal Observer emissions.

A Stage-2 replacement for the v2 ideal-observer fit. Each hidden state has a
fixed prior over orientation theta and a (partially) fixed Stage-2
psychometric on g(m) = log P(Go | m) / P(NoGo | m). Stage-1 sensory and
velocity-emission parameters are passed in as frozen inputs. The IO-HMM
fits transition probabilities and the free per-state psychometric entries
via EM.

Modules
-------
- ``io_core``   : pure-numpy IO inference (von Mises, posteriors, g(m)).
- ``states``    : ``PsychSpec`` / ``IOState`` and the v0 four-state factory.
- ``emissions`` : per-trial state-conditioned log-emission matrix (+ cached
                  ``precompute_state_terms`` / ``p_go_from_terms`` helpers).
- ``hmm``       : categorical HMM core (forward / backward / xi / Viterbi).
- ``fit``       : EM (Baum-Welch) driver -- ``fit(...)`` with random restarts
                  fits ``pi``, ``A`` and the free per-state psychometric
                  entries; closed-form transition M-step + scipy psychometric
                  M-step. ``viterbi_paths(...)`` decodes latent states.
- ``simulate``  : generative simulation of sequences (used for recovery).

See ``states.py`` for the v0 four-state spec (Perfect / Thirsty / Disengaged
/ Naive). Note: the psychometric *slope* on g(m) is only weakly identified
from choices alone (saturation / logistic separation); transitions and
bias / constant-rate params recover reliably. Robust slope estimation is the
motivation for the planned v0.5 velocity emissions.
"""
