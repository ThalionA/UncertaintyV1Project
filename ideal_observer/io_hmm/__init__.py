# -*- coding: utf-8 -*-
"""IO-HMM: Hidden Markov Model with state-indexed Ideal Observer emissions.

A Stage-2 replacement for the v2 ideal-observer fit. Each hidden state has a
prior over orientation theta and a (partially) fixed Stage-2 psychometric on
g(m) = log P(Go | m) / P(NoGo | m), and optionally a confidence-velocity
emission. Stage-1 sensory params are passed in as frozen inputs. The IO-HMM
fits transition probabilities and the free per-state emission params via EM.

Modules
-------
- ``io_core``   : pure-numpy IO inference (von Mises, posteriors, g(m), and the
                  utility-weighted decision variable DV(m)).
- ``states``    : ``PsychSpec`` / ``VelocitySpec`` / ``IOState`` and the v0
                  four-state factory.
- ``emissions`` : per-trial state-conditioned log-emission matrix. The joint
                  choice(+velocity) emission marginalises over the latent
                  measurement m (``joint_log_emission_state``); cached
                  ``precompute_state_terms`` returns (g(m), DV(m), p(m|s)).
- ``hmm``       : categorical HMM core (forward / backward / xi / Viterbi).
- ``fit``       : EM (Baum-Welch) driver -- ``fit(...)`` with random restarts
                  fits ``pi``, ``A`` and the free per-state emission params;
                  closed-form transition M-step + numerical per-state emission
                  M-step. ``viterbi_paths(...)`` decodes latent states.
- ``simulate``  : generative simulation of sequences (used for recovery).

Confidence-velocity channel
---------------------------
``fit(..., use_velocity=True)`` adds the paper's velocity emission (Eqs. 9-10):
each state's pre-reward-zone velocity is a graded readout of the decision
variable, ``v ~ N(beta_vel * DV(m) + alpha_vel, sigma_vel)``, marginalised over
the latent measurement m *jointly with choice* (so velocity updates choice via
Bayes on m -- never as a direct predictor of choice). ``beta_vel`` is the
per-state confidence gain (``beta_vel = 0`` => a stimulus-independent engagement
marker). Because velocity reads DV(m) -- which depends on perception but not on
the choice psychometric -- it is what dissociates a state's *perception* from
its *action*.

Free perception (Phase 2)
-------------------------
A state may carry a ``PerceptionSpec`` with free perceptual params: a
sensory-precision multiplier ``lambda_`` (``kappa_eff = lambda_ * kappa(c, d)``,
reshaping g(m)/DV(m)/p(m|s) together) and, when ``parameterized_prior`` is set, a
bimodal prior with free concentration ``prior_strength`` and asymmetry
``prior_weight`` (which enter only the inference posterior, moving g(m)/DV(m)).
EM fits them via the joint per-state M-step (the IO terms are recomputed as the
perceptual params vary, since they are no longer frozen). Two states with
identical choice behaviour but different perception are separated and the
perceptual params are recovered -- the perception-vs-action dissociation --
precisely because velocity reads DV(m) independently of the choice psychometric.
Caveat: ``prior_strength`` and ``lambda_`` are both posterior-sharpness knobs and
only weakly separable, so fitting both free per state is fragile; ``prior_weight``
(a perceptual category bias, distinct from the decisional bias ``alpha``) is well
identified.

See ``states.py`` for the v0 four-state spec (Perfect / Thirsty / Disengaged
/ Naive). Note: from choices alone the psychometric *slope* on g(m) is only
weakly identified (saturation / logistic separation) and perceptual vs
decisional changes are confounded; transitions and bias / constant-rate params
recover reliably.
"""
