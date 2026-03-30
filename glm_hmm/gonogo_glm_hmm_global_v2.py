# -*- coding: utf-8 -*-
"""
GLM-HMM Analysis for Go/No-Go Task (VR)
Optimized for:
- Handling missing 'trial_in_session' column (Auto-generated)
- Using explicit 'current_reward' from CSV
- Euclidean State Matching
- Engagement Sorting
- Sticky State Initialization
"""

import numpy as np
import pandas as pd
import ssm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment

# -------------------------------
# Hyperparameters
# -------------------------------
L2_REG = 1.5
NUM_ITERS_GLOBAL = 2000
NUM_ITERS_INDIV = 10
SIGMA_INIT = 0.5
TRANSITION_NOISE = 0.1
CANDIDATE_K = [2, 3, 4]
N_INITS = 20

# -------------------------------
# Helper Functions
# -------------------------------

def calculate_bic(model, X_data, y_data, K, input_dim):
    """Calculate BIC for model selection."""
    log_likelihood = model.log_likelihood(y_data, inputs=X_data)
    n_params = K * input_dim + K * (K - 1)
    n_samples = len(y_data)
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    return bic

def sort_global_model_by_engagement(model):
    """
    Sorts model states so that State 0 has the highest absolute 
    weight on the first predictor (Current Stimulus).
    """
    # Assumes index 0 is 'current_stim_signed'
    stim_weights = [abs(model.observations.params[k][0][0]) for k in range(model.K)]
    perm = np.argsort(stim_weights)[::-1]
    model.permute(perm)
    print(f"Global model sorted by engagement magnitude: {stim_weights} -> Permutation: {perm}")
    return model

def match_states_euclidean(subject_model, reference_model):
    """Aligns subject_model states to reference_model using Euclidean distance."""
    K = subject_model.K
    subj_weights = np.array([subject_model.observations.params[k][0] for k in range(K)])
    ref_weights = np.array([reference_model.observations.params[k][0] for k in range(K)])
    
    cost_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost_matrix[i, j] = np.linalg.norm(subj_weights[i] - ref_weights[j])
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    perm = np.zeros(K, dtype=int)
    for r, c in zip(row_ind, col_ind):
        perm[c] = r
        
    subject_model.permute(perm)
    return subject_model

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
print("Loading Data...")
data = pd.read_csv('GLMHmm_predictors_vr.csv')
data['bias'] = 1 

# --- FIX: Generate trial_in_session ---
# The CSV is ordered sequentially by MATLAB. We group by session (which is unique globally in your script)
# and assign a counter.
print("Generating trial indices...")
data['trial_in_session'] = data.groupby('session').cumcount() + 1

# Flip mapping
flip_mapping = {
    'Cb15': False, 'Cb17': False, 'Cb21': True, 
    'Cb22': True, 'Cb24': True, 'Cb25': False
}

data_list = []
for animal_id, group in data.groupby('animal'):
    group = group.copy()
    group["current_stim_signed"] = (group["current_stim_orientation"] - 45) / 45
    group["prev_stim_signed"] = (group["prev_stim_orientation"] - 45) / 45
    
    if flip_mapping.get(animal_id, False):
        group["current_stim_signed"] *= -1
        group["prev_stim_signed"] *= -1
    
    data_list.append(group)

data_transformed = pd.concat(data_list)

# Ensure correct sort order for the rest of the pipeline
data_transformed = data_transformed.sort_values(['animal', 'session', 'trial_in_session'])

# -------------------------------
# Model Setup
# -------------------------------
# Predictors
# predictor_cols = ['current_stim_signed', 'prev_stim_signed', 'prev_choice', 'prev_reward', 'bias']
target_col = 'current_choice'

data_transformed[target_col] = data_transformed[target_col].apply(lambda x: 0 if x == 1 else 1)
data_transformed["current_stim_signed"] *= -1
data_transformed["prev_stim_signed"] *= -1
data_transformed["prev_choice"] = data_transformed["prev_choice"].apply(lambda x: 1 if x == 1 else -1)

predictor_cols = ['current_stim_signed', 'prev_stim_signed', 'prev_choice', 'prev_reward', 'bias']

X_global = data_transformed[predictor_cols].values
y_global = data_transformed[target_col].values.reshape(-1, 1)

animal_data = {}
for animal_id, group in data_transformed.groupby('animal'):
    X_a = group[predictor_cols].values
    y_a = group[target_col].values.reshape(-1, 1)
    animal_data[animal_id] = (X_a, y_a)

obs_dim = 1
input_dim = len(predictor_cols)
num_categories = 2

# -------------------------------
# Stage 1: Global GLM (1 State)
# -------------------------------
print("\n--- Stage 1: Global GLM Fit ---")
glm_model = ssm.HMM(1, obs_dim, input_dim, 
                    observations="input_driven_obs", 
                    observation_kwargs=dict(C=num_categories),
                    transitions="standard")
glm_model.fit(y_global, inputs=X_global, method="em", num_iters=NUM_ITERS_GLOBAL, 
              observations_mstep_kwargs=dict(l2_penalty=L2_REG))
global_glm_weights = glm_model.observations.params[0][0].copy()

# -------------------------------
# Stage 2: Global Model Selection (BIC)
# -------------------------------
print("\n--- Stage 2: Global GLM-HMM Selection ---")
global_bics = {}
global_models = {}

for K in CANDIDATE_K:
    best_ll = -np.inf
    best_model = None
    
    for i in range(N_INITS):
        model = ssm.HMM(K, obs_dim, input_dim, 
                        observations="input_driven_obs", 
                        observation_kwargs=dict(C=num_categories),
                        transitions="standard")
        
        for k in range(K):
            model.observations.params[k][0] = global_glm_weights + \
                np.random.normal(scale=SIGMA_INIT, size=global_glm_weights.shape)
        
        init_trans = np.eye(K) + TRANSITION_NOISE * np.random.rand(K, K)
        init_trans /= init_trans.sum(axis=1, keepdims=True)
        model.transitions.log_Ps = np.log(init_trans)
        
        model.fit(y_global, inputs=X_global, method="em", num_iters=NUM_ITERS_GLOBAL, 
                  observations_mstep_kwargs=dict(l2_penalty=L2_REG), verbose=0)
        
        ll = model.log_likelihood(y_global, inputs=X_global)
        if ll > best_ll:
            best_ll = ll
            best_model = model
            
    bic = calculate_bic(best_model, X_global, y_global, K, input_dim)
    global_bics[K] = bic
    global_models[K] = best_model
    print(f"  K={K}: BIC={bic:.2f}")

best_K = min(global_bics, key=global_bics.get)
print(f"\nSelected K={best_K} based on Global BIC.")

# --- Sort Global Model ---
final_global_model = sort_global_model_by_engagement(global_models[best_K])

# -------------------------------
# Stage 3: Individual Fits & Alignment
# -------------------------------
print(f"\n--- Stage 3: Individual Fits (K={best_K}) ---")
animal_models = {}

for animal_id, (X_a, y_a) in animal_data.items():
    print(f"Fitting {animal_id}...")
    
    model = ssm.HMM(best_K, obs_dim, input_dim, 
                    observations="input_driven_obs", 
                    observation_kwargs=dict(C=num_categories),
                    transitions="standard")
    
    model.transitions.log_Ps = final_global_model.transitions.log_Ps.copy()
    for k in range(best_K):
        model.observations.params[k][0] = final_global_model.observations.params[k][0].copy()
        
    model.fit(y_a, inputs=X_a, method="em", num_iters=NUM_ITERS_INDIV, 
              observations_mstep_kwargs=dict(l2_penalty=2), verbose=0)
    
    model = match_states_euclidean(model, final_global_model)
    animal_models[animal_id] = model

#%%
# Plotting 
# -------------------------------
print("\nGenerating Plots (Aligned)...")

# 1. GLM Weights
fig, axs = plt.subplots(best_K, 1, figsize=(10, 4*best_K), sharex=True)
if best_K == 1: axs = [axs]
colors = plt.cm.tab10(np.linspace(0, 1, len(animal_data)))

for k in range(best_K):
    ax = axs[k]
    # Global Reference: Take the full weight vector [0]
    ref_w = final_global_model.observations.params[k][0] 
    ax.plot(range(input_dim), ref_w, 'k--', linewidth=3, alpha=0.3, label='Global Ref')
    
    for i, (animal, model) in enumerate(animal_models.items()):
        # Individual weights: Take the full weight vector [0]
        w = model.observations.params[k][0]
        ax.plot(range(input_dim), w, marker='o', color=colors[i], label=animal)
        
    ax.set_xticks(range(input_dim))
    ax.set_xticklabels(predictor_cols, rotation=45)
    ax.axhline(0, color='gray', linestyle=':')
    ax.set_title(f"State {k+1} (0=Engaged)")
    if k == 0: ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

plt.tight_layout()
plt.show()

# 2. Transition Matrices
fig, axs = plt.subplots(1, len(animal_data), figsize=(3*len(animal_data), 3))
if len(animal_data) == 1: axs = [axs]

for i, (animal, model) in enumerate(animal_models.items()):
    trans_mat = np.exp(model.transitions.log_Ps)
    sns.heatmap(trans_mat, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=axs[i], vmin=0, vmax=1)
    axs[i].set_title(f"{animal}")
    axs[i].set_xlabel("To State")
    axs[i].set_ylabel("From State" if i==0 else "")
plt.suptitle("Transition Matrices (Matched States)", y=1.05)
plt.tight_layout()
plt.show()

#%% 3. Export Aligned Probabilities
print("Exporting aligned state probabilities...")
export_df = data_transformed.copy()

for animal_id, model in animal_models.items():
    X_a, y_a = animal_data[animal_id]
    posterior_probs = model.expected_states(y_a, input=X_a)[0]
    
    # Use both session and trial index to ensure safe assignment
    mask = export_df['animal'] == animal_id
    idx = export_df[mask].index
    
    for k in range(best_K):
        export_df.loc[idx, f'prob_state_{k+1}'] = posterior_probs[:, k]

export_cols = ['animal', 'session', 'trial_in_session', 'current_choice', 'prev_reward'] + [f'prob_state_{k+1}' for k in range(best_K)]
export_df[export_cols].to_csv('GLM_HMM_states_aligned.csv', index=False)

# Append this to your python script
weight_data = []
predictors = predictor_cols  # ['current_stim', 'prev_stim', 'prev_choice', 'prev_reward', 'bias']

for animal_id, model in animal_models.items():
    for k in range(best_K):
        # Get weights for this state (D,)
        w = model.observations.params[k][0]
        for p_idx, p_name in enumerate(predictors):
            weight_data.append({
                'animal': animal_id,
                'state': k + 1,
                'predictor': p_name,
                'weight': w[p_idx]
            })

pd.DataFrame(weight_data).to_csv('GLM_HMM_weights.csv', index=False)
print("Weights exported to GLM_HMM_weights.csv")

print("Analysis Complete.")