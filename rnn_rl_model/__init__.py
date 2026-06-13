"""RNN + RL test of the Similarity Framework.

A sibling of ``si_network_model``: the same grating -> recurrent-V1 ->
cosine-similarity-to-archetypes -> Go/NoGo pipeline, but the V1->action mapping
is learned by an **actor-critic** reinforcement learner (policy gradient with a
value baseline) instead of three-factor Hebbian plasticity, and the recurrent
V1 itself can optionally be trained end-to-end before being frozen.

The scientific question: does an RL agent whose policy reads a similarity index
between the V1 population and two learned archetypes still converge to the
template direction ``Delta mu`` (not the covariance-whitened optimum), and reach
near-Bayesian decision efficiency -- and does task-training the recurrence
change that?
"""

from __future__ import annotations
