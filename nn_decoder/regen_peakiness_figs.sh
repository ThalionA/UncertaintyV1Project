#!/usr/bin/env bash
# Regenerate the full PCA-peakiness figure suite (restyled). Run from nn_decoder/.
set -e
cd "$(dirname "$0")"
echo "===== 1/8 entropy trajectory (temp + compare: figs 1,2) ====="
python diagnostics/decode_entropy_trajectory.py --mouse all \
  --compare-runs brier_ctrl_flatevar \
  --compare-labels "PCA (evar-weighted)" "PCA (flat-evar)"
echo "===== 2/8 entropy trajectory maxprob (spatial: fig 9) ====="
python diagnostics/decode_entropy_trajectory.py --mouse all --metric maxprob
echo "===== 3/8 trajectory + landscape (figs 3,4) ====="
python diagnostics/posterior_trajectory_landscape.py
echo "===== 4/8 flat-evar example posteriors (figs 5,10) ====="
python diagnostics/flat_evar_example_posteriors.py --arch temp
python diagnostics/flat_evar_example_posteriors.py --arch spat
echo "===== 5/8 toy model (figs 6,7,8) ====="
python diagnostics/toy_peakiness_model.py
echo "===== 6/8 toy shortcut dynamics (figs 11,12) ====="
python diagnostics/toy_shortcut_dynamics.py
echo "===== 7/8 toy width-matched (figs 13,14,15) ====="
python diagnostics/toy_width_matched.py
echo "===== 8/9 real-data variants + weights (figs 16-26) ====="
python compare_loss_variants.py
python weight_evolution_variants.py --arch spat
echo "===== 9/9 PC basis = location vs shape demonstration (fig 2a/2b) ====="
python diagnostics/pc_location_vs_shape.py
echo "===== peakiness distributions + per-trial scatter (Mate: bulk vs tail) ====="
python diagnostics/peakiness_distributions.py
python diagnostics/peakiness_scatter_spat_temp.py
echo "===== DONE ====="
