# PCA fit-loss → peaky posteriors (pointer)

Date: 2026-05-29

Investigation of why the PCA-weighted fit-loss produces very peaky
(over-confident) posteriors while KL and JS stay calibrated.

**Root cause (short):** the PCA loss is a *variance-weighted L2 distance*
(`pca_loss.pca_distance`; torch twin `nn_classifier.custom_loss_all_H:192`), not a
divergence. It has no anti-collapse term, and its `evar` weighting puts
near-zero weight on the trailing PCs that encode bump *width*. In the temporal
(sampling) decoder — trial posterior = mean of per-bin posteriors
(`custom_loss_all_H:146`) with a per-bin entropy penalty
(`custom_loss_all_H:149`) — PCA tolerates sharp, gappy averages while forward KL
explodes on them (JS in between).

**Full report + figures + reproducible demo:**
- Report: `../figures/loss_smoothness_demo/LOSS_SMOOTHNESS_REPORT.md`
- Script: `../diagnostics/loss_smoothness_demo.py`
- Test: `tests/test_loss_smoothness_demo.py`
