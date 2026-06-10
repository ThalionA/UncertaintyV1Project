#!/usr/bin/env bash
# Sync the (pruned) peakiness figure suite into the ResearchVault note attachments.
# PNG only (the vault embeds PNG; SVG vector copies stay in the repo figures/ dirs).
# Run from nn_decoder/. Numbering has gaps (4,12,17,19,24,25 were pruned) — the
# figN is only a filename, never shown as a caption, so gaps are harmless.
set -euo pipefail
cd "$(dirname "$0")"
ET=figures/loss_sweep_plots/loss_comparison_v1/entropy_trajectory
TL=figures/loss_sweep_plots/loss_comparison_v1/trajectory_landscape
FE=figures/loss_sweep_plots/brier_ctrl_flatevar/example_posteriors
TOY=figures/toy_peakiness
LV=figures/loss_variants
LVW=figures/loss_variants/weights
WM=figures/loss_variants/within_mouse
SC=figures/schematic
DG=figures/pc_geometry
V="/Users/theoamvr/Documents/ResearchVault/attachments/PCA-Peakiness-Mechanism"

MAP=(
  "$SC/setup_schematic.png|peakiness_setup_schematic.png"
  "$ET/entropy_compare_temp.png|peakiness_fig1_entropy_compare.png"
  "$ET/entropy_vs_epoch_temp.png|peakiness_fig2_entropy_alllosses.png"
  "$DG/pc_location_vs_shape_toy.png|peakiness_fig2a_pcbasis_toy.png"
  "$DG/pc_location_vs_shape_real.png|peakiness_fig2b_pcbasis_real.png"
  "$TL/loss_vs_width_Q_m0.png|peakiness_fig3_loss_landscape.png"
  "$FE/examples_temp_m0.png|peakiness_fig5_example_posteriors_temp.png"
  "$FE/examples_spat_m0.png|peakiness_fig5_example_posteriors_spat.png"
  "$TOY/toy_examples.png|peakiness_fig6_toy_examples.png"
  "$TOY/toy_spectrum.png|peakiness_fig7_toy_spectrum.png"
  "$TOY/toy_sweeps.png|peakiness_fig8_toy_sweeps.png"
  "$ET/entropy_vs_epoch_spat_maxprob.png|peakiness_fig9_spatial_maxprob.png"
  "figures/peakiness_scatter/bulk_vs_tail.png|peakiness_fig9a_bulk_vs_tail.png"
  "figures/peakiness_scatter/scatter_maxP.png|peakiness_fig9b_scatter_spat_temp.png"
  "$TOY/shortcut_phaseportrait.png|peakiness_fig11_shortcut_phase.png"
  "$TOY/widthmatched_phase.png|peakiness_fig13_widthmatched_phase.png"
  "$TOY/widthmatched_examples.png|peakiness_fig14_widthmatched_examples.png"
  "$TOY/why_peakier.png|peakiness_fig15_why_overfitting.png"
  "$LV/peakiness.png|peakiness_fig16_realdata_peakiness.png"
  "$LV/spat_vs_temp_KL.png|peakiness_fig18_realdata_KLmetric.png"
  "$LV/kl_vs_pca_across_lambda.png|peakiness_fig22_realdata_klvspca.png"
  "$LV/lambda_sweep.png|peakiness_lambda_sweep.png"
  "$LV/perbin_temporal.png|peakiness_fig23_realdata_perbin.png"
  "figures/spat_temp/per_mouse_variants.png|peakiness_per_mouse_variants.png"
  "figures/spat_temp/per_mouse_variants_pca_norms.png|peakiness_per_mouse_variants_pca_norms.png"
  "figures/spat_temp/per_mouse_losses.png|peakiness_per_mouse_losses.png"
  "$LVW/logit_profiles_spat.png|peakiness_fig26_logit_profiles.png"
  "figures/peakiness_scatter/subspace_error_realdata.png|peakiness_subspace_real.png"
  "figures/peakiness_scatter/uncertainty_scaling_realdata.png|peakiness_uncertainty_scaling.png"
  "figures/spat_temp/spat_temp_losses_KL.png|peakiness_spat_temp_losses.png"
  "figures/loss_sweep_plots/hidden_ablation/overfit_vs_width/capacity_summary_spat.png|peakiness_capacity_ablation.png"
  "figures/loss_sweep_plots/hidden_ablation/overfit_vs_width/spat_temp_skill_vs_width_by_loss.png|peakiness_spat_temp_width.png"
)
for pair in "${MAP[@]}"; do
  src="${pair%%|*}"; dst="${pair##*|}"
  if [[ ! -f "$src" ]]; then echo "MISSING SOURCE: $src"; exit 1; fi
  cp "$src" "$V/$dst"
done
echo "Synced ${#MAP[@]} figures to $V"

# Remove the pruned / superseded figures from the vault
PRUNE=(
  peakiness_fig20_within1b_evar.png
  peakiness_fig20_within1b_flat.png
  peakiness_fig20_within1b_shape.png
  peakiness_fig4_morph.png
  peakiness_fig12_shortcut_decomp.png
  peakiness_fig17_realdata_PCAmetric.png
  peakiness_fig19_realdata_examples.png
  peakiness_fig24_weights_evolution.png
  peakiness_fig25_weights_vs_peakiness.png
  peakiness_fig20_realdata_within_KL.png
  peakiness_fig5_example_posteriors.png
  peakiness_fig10_spatial_examples.png
  peakiness_fig21_realdata_examplesgrid.png
)
for f in "${PRUNE[@]}"; do
  [[ -f "$V/$f" ]] && { rm "$V/$f"; echo "removed $f"; } || true
done
echo "done"
