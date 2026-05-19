import re

with open('top_mode_analysis.py', 'r') as f:
    content = f.read()

# 1. Turn off annot
content = content.replace("annot=True, fmt=\".2f\"", "annot=False")

# 2. Return data from plot_cross_condition_similarity
# We need to insert collection logic
content = content.replace(
    "for idx, (ax, feat_name) in enumerate(zip(axes, feature_names)):",
    "feature_mats = []\n    for idx, (ax, feat_name) in enumerate(zip(axes, feature_names)):"
)

content = content.replace(
    "sns.heatmap(sim_mat, xticklabels=labels, yticklabels=labels, \n                    cmap='viridis', vmin=-1, vmax=1, annot=False, ax=ax, cbar=(idx==2))",
    "feature_mats.append((feat_name, labels, sim_mat))\n        sns.heatmap(sim_mat, xticklabels=labels, yticklabels=labels, \n                    cmap='viridis', vmin=-1, vmax=1, annot=False, ax=ax, cbar=(idx==2))"
)

content = content.replace(
    "plt.close(fig)",
    "plt.close(fig)\n    return feature_mats"
)

# Wait, `plt.close(fig)` appears multiple times in the file.
# I will use multi_replace_file_content instead.
