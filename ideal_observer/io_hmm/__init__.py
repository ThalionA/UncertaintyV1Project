# -*- coding: utf-8 -*-
"""IO-HMM: Hidden Markov Model with state-indexed Ideal Observer emissions.

A Stage-2 replacement for the v2 ideal-observer fit. Each hidden state has a
fixed prior over orientation theta and a (partially) fixed Stage-2
psychometric on g(m) = log P(Go | m) / P(NoGo | m). Stage-1 sensory and
velocity-emission parameters are passed in as frozen inputs. The IO-HMM
fits transition probabilities and the free per-state psychometric entries
via EM.

See ``states.py`` for the v0 four-state spec (Perfect / Thirsty / Disengaged
/ Naive).
"""
