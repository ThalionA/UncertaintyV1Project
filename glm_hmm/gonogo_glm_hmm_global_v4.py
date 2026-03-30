# -*- coding: utf-8 -*-
"""
GLM-HMM Analysis for Go/No-Go Task (VR)
Optimized for:
- 3-Stage Hierarchical Fitting (Global -> Animal -> Session) OR 2-Stage (Global -> Animal)
- Interactive manual selection of latent states (K)
- Stimulus scaled by Contrast and Precision (Dispersion) for BOTH Current and Previous trials
- Handling missing 'trial_in_session' column
- Euclidean State Matching to a Global Template
"""

import numpy as np
import pandas as pd
import ssm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment

# -------------------------------
# Hyperparameters & Toggles
# -------------------------------
DO_SESSION_FITS = True  # <--- TOGGLE: False = Fit concatenated Animal trials; True = Fit per Session

L2_REG = 1.5
NUM_ITERS_GLOBAL = 500
NUM_ITERS_ANIMAL = 10   # Stage 2: Intermediate iterations for Animal
NUM_ITERS_SESSION = 10   # Stage 3: Low iterations for Session
SIGMA_INIT = 0.5
TRANSITION_NOISE = 0.2
CANDIDATE_K = [2, 3, 4]
N_INITS = 10

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
    stim_weights = [abs(model.observations.params[k][0][0]) for k in range(model.K)]
    perm = np.argsort(stim_weights)[::-1]
    model.permute(perm)
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

print("Generating trial indices...")
data['trial_in_session'] = data.groupby(['animal', 'session']).cumcount() + 1

# --- Contrast and Dispersion Scaling (Current & Previous) ---
data['current_contrast'] = np.where(np.isclose(data['current_contrast'], 0.99), 1.0, data['current_contrast'])
data['prev_contrast'] = np.where(np.isclose(data['prev_contrast'], 0.99), 1.0, data['prev_contrast'])
data['current_disp_factor'] = 5.0 / data['current_dispersion']
data['prev_disp_factor'] = 5.0 / data['prev_dispersion']

flip_mapping = {
    'Cb15': False, 'Cb17': False, 'Cb21': True, 
    'Cb22': True, 'Cb24': True, 'Cb25': False
}

data_list = []
for animal_id, group in data.groupby('animal'):
    group = group.copy()
    norm_stim = (group["current_stim_orientation"] - 45) / 45
    norm_prev_stim = (group["prev_stim_orientation"] - 45) / 45
    
    group["current_stim_signed"] = norm_stim * group['current_contrast'] * group['current_disp_factor']
    group["prev_stim_signed"] = norm_prev_stim * group['prev_contrast'] * group['prev_disp_factor']
    
    if flip_mapping.get(animal_id, False):
        group["current_stim_signed"] *= -1
        group["prev_stim_signed"] *= -1
    
    data_list.append(group)

data_transformed = pd.concat(data_list)
data_transformed = data_transformed.sort_values(['animal', 'session', 'trial_in_session'])

# -------------------------------
# Model Setup
# -------------------------------
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
    animal_data[animal_id] = (group[predictor_cols].values, group[target_col].values.reshape(-1, 1))

if DO_SESSION_FITS:
    session_data = {}
    for (animal_id, session_id), group in data_transformed.groupby(['animal', 'session']):
        session_data[(animal_id, session_id)] = (group[predictor_cols].values, group[target_col].values.reshape(-1, 1))

obs_dim = 1
input_dim = len(predictor_cols)
num_categories = 2

# -------------------------------
# Stage 1: Global GLM-HMM Selection & Fit
# -------------------------------
print("\n--- Stage 1: Global GLM Baseline ---")
glm_model = ssm.HMM(1, obs_dim, input_dim, observations="input_driven_obs", observation_kwargs=dict(C=num_categories), transitions="standard")
glm_model.fit(y_global, inputs=X_global, method="em", num_iters=NUM_ITERS_GLOBAL, observations_mstep_kwargs=dict(l2_penalty=L2_REG))
global_glm_weights = glm_model.observations.params[0][0].copy()

print("\n--- Stage 1: Global Multi-State Search ---")
global_bics, global_lls, global_models = {}, {}, {}
all_k_weights = [] # <-- ADDED: List to store ALL weights for export

for K in CANDIDATE_K:
    best_ll, best_model = -np.inf, None
    for i in range(N_INITS):
        model = ssm.HMM(K, obs_dim, input_dim, observations="input_driven_obs", observation_kwargs=dict(C=num_categories), transitions="standard")
        for k in range(K):
            model.observations.params[k][0] = global_glm_weights + np.random.normal(scale=SIGMA_INIT, size=global_glm_weights.shape)
        
        init_trans = np.eye(K) + TRANSITION_NOISE * np.random.rand(K, K)
        init_trans /= init_trans.sum(axis=1, keepdims=True)
        model.transitions.log_Ps = np.log(init_trans)
        
        model.fit(y_global, inputs=X_global, method="em", num_iters=NUM_ITERS_GLOBAL, observations_mstep_kwargs=dict(l2_penalty=L2_REG), verbose=0)
        
        ll = model.log_likelihood(y_global, inputs=X_global)
        if ll > best_ll:
            best_ll, best_model = ll, model
            
    bic = calculate_bic(best_model, X_global, y_global, K, input_dim)
    global_bics[K] = bic
    global_lls[K] = best_ll # <-- ADDED: Store LL for plotting
    
    # Sort immediately for fair comparison plotting
    best_model = sort_global_model_by_engagement(best_model)
    global_models[K] = best_model
    
    # <-- ADDED: Store weights for export before moving on to next K
    for k_state in range(K):
        w = best_model.observations.params[k_state][0]
        for p_idx, p_name in enumerate(predictor_cols):
            all_k_weights.append({
                'K_model': K,
                'state': k_state + 1,
                'predictor': p_name,
                'weight': w[p_idx]
            })
            
    print(f"  K={K}: Log-Likelihood={best_ll:.2f}, BIC={bic:.2f}")

best_K_bic = min(global_bics, key=global_bics.get)

# <-- ADDED: Save all K global weights to CSV
pd.DataFrame(all_k_weights).to_csv('GLM_HMM_global_weights_all_K.csv', index=False)
print("\nSaved global weights for ALL evaluated models to 'GLM_HMM_global_weights_all_K.csv'")

# ==========================================================
# Comparison Plotting for Model Selection
# ==========================================================
print("\nGenerating Model Comparison Plots...")

# Plot 1: BIC & Log-Likelihood Curve (MODIFIED to show both)
fig, ax1 = plt.subplots(figsize=(6, 4))
ax2 = ax1.twinx()
ax1.plot(list(global_bics.keys()), list(global_bics.values()), 'bo-', label='BIC')
ax2.plot(list(global_lls.keys()), list(global_lls.values()), 'rs-', label='Log-Likelihood')

ax1.set_xlabel("Number of States (K)")
ax1.set_ylabel("BIC Score (Lower is better)", color='b')
ax2.set_ylabel("Log-Likelihood (Higher is better)", color='r')
ax1.set_xticks(CANDIDATE_K)
plt.title("Model Selection: BIC vs Log-Likelihood")
plt.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# Plot 2: Global Weights Comparison Grid
fig, axs = plt.subplots(max(CANDIDATE_K), len(CANDIDATE_K), figsize=(4*len(CANDIDATE_K), 3*max(CANDIDATE_K)), sharex=True, sharey=True)
for ax in axs.flat: ax.set_visible(False)  # Hide all first

for col, K in enumerate(CANDIDATE_K):
    model = global_models[K]
    for k in range(K):
        ax = axs[k, col]
        ax.set_visible(True)
        w = model.observations.params[k][0]
        ax.plot(range(input_dim), w, marker='o', color='k')
        ax.axhline(0, color='gray', linestyle=':')
        ax.set_xticks(range(input_dim))
        ax.set_xticklabels(predictor_cols, rotation=45)
        ax.set_title(f"K={K} | State {k+1}" + (" (Engaged)" if k==0 else ""))

plt.suptitle("Global Weights Comparison Across K", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

# Plot 3: Transition Matrices Comparison
fig, axs = plt.subplots(1, len(CANDIDATE_K), figsize=(4*len(CANDIDATE_K), 3))
if len(CANDIDATE_K) == 1: axs = [axs]
for col, K in enumerate(CANDIDATE_K):
    trans_mat = np.exp(global_models[K].transitions.log_Ps)
    sns.heatmap(trans_mat, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=axs[col], vmin=0, vmax=1)
    axs[col].set_title(f"K={K} Transition Matrix")
plt.tight_layout()
plt.show()

# -------------------------------
# Interactive Manual K Selection
# -------------------------------
user_input = input(f"\nEnter the number of states (K) you wish to proceed with {CANDIDATE_K}.\nPress [Enter] to use the optimal BIC suggestion (K={best_K_bic}): ")
try:
    chosen_K = int(user_input.strip())
    if chosen_K not in CANDIDATE_K:
        print(f"Warning: {chosen_K} was not in candidates. Defaulting to K={best_K_bic}")
        chosen_K = best_K_bic
except ValueError:
    chosen_K = best_K_bic

print(f"\n---> Proceeding with K = {chosen_K} <---")
final_global_model = global_models[chosen_K]

#%%
# -------------------------------
# Stage 2: Animal-Level Fits
# -------------------------------
print(f"\n--- Stage 2: Animal-Level Fits (K={chosen_K}) ---")
animal_models = {}

for animal_id, (X_a, y_a) in animal_data.items():
    model_a = ssm.HMM(chosen_K, obs_dim, input_dim, observations="input_driven_obs", observation_kwargs=dict(C=num_categories), transitions="standard")
    model_a.transitions.log_Ps = final_global_model.transitions.log_Ps.copy()
    for k in range(chosen_K):
        model_a.observations.params[k][0] = final_global_model.observations.params[k][0].copy()
        
    model_a.fit(y_a, inputs=X_a, method="em", num_iters=NUM_ITERS_ANIMAL, observations_mstep_kwargs=dict(l2_penalty=1.5), verbose=0)
    model_a = match_states_euclidean(model_a, final_global_model)
    animal_models[animal_id] = model_a

# -------------------------------
# Stage 3: Session-Level Fits (Optional based on DO_SESSION_FITS)
# -------------------------------
if DO_SESSION_FITS:
    print(f"\n--- Stage 3: Session-Level Fits (K={chosen_K}) ---")
    session_models = {}
    for (animal_id, session_id), (X_s, y_s) in session_data.items():
        model_s = ssm.HMM(chosen_K, obs_dim, input_dim, observations="input_driven_obs", observation_kwargs=dict(C=num_categories), transitions="standard")
        parent_model = animal_models[animal_id]
        model_s.transitions.log_Ps = parent_model.transitions.log_Ps.copy()
        for k in range(chosen_K):
            model_s.observations.params[k][0] = parent_model.observations.params[k][0].copy()
            
        model_s.fit(y_s, inputs=X_s, method="em", num_iters=NUM_ITERS_SESSION, observations_mstep_kwargs=dict(l2_penalty=1.5), verbose=0)
        model_s = match_states_euclidean(model_s, final_global_model)
        session_models[(animal_id, session_id)] = model_s

#%%
# Plotting Final Animal Models
# -------------------------------
print("\nGenerating Plots for Final Animal Models...")

# 1. GLM Weights (Comparing Global vs Animal-level)
fig, axs = plt.subplots(chosen_K, 1, figsize=(10, 4*chosen_K), sharex=True)
if chosen_K == 1: axs = [axs]
colors = plt.cm.tab10(np.linspace(0, 1, len(animal_data)))

for k in range(chosen_K):
    ax = axs[k]
    ref_w = final_global_model.observations.params[k][0] 
    ax.plot(range(input_dim), ref_w, 'k--', linewidth=3, alpha=0.3, label='Global Ref')
    
    for i, (animal, model) in enumerate(animal_models.items()):
        w = model.observations.params[k][0]
        ax.plot(range(input_dim), w, marker='o', color=colors[i], label=animal)
        
    ax.set_xticks(range(input_dim))
    ax.set_xticklabels(predictor_cols, rotation=45)
    ax.axhline(0, color='gray', linestyle=':')
    ax.set_title(f"State {k+1}" + (" (Engaged)" if k==0 else ""))
    if k == 0: 
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))

plt.tight_layout()
plt.show()

# 2. Transition Matrices (Animal-level)
fig, axs = plt.subplots(1, len(animal_models), figsize=(3*len(animal_models), 3))
if len(animal_models) == 1: axs = [axs] 
elif isinstance(axs, np.ndarray) and axs.ndim == 2: axs = axs.flatten()

for i, (animal, model) in enumerate(animal_models.items()):
    trans_mat = np.exp(model.transitions.log_Ps)
    sns.heatmap(trans_mat, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=axs[i], vmin=0, vmax=1)
    axs[i].set_title(f"{animal}")
    axs[i].set_xlabel("To State")
    axs[i].set_ylabel("From State" if i==0 else "")
    
plt.suptitle("Transition Matrices (Stage 2: Animal Level)", y=1.05)
plt.tight_layout()
plt.show()

#%%
# Export Aligned Probabilities & Weights
# -------------------------------
print(f"\nExporting aligned state probabilities and weights (DO_SESSION_FITS = {DO_SESSION_FITS})...")
export_df = data_transformed.copy()

for k in range(chosen_K):
    export_df[f'prob_state_{k+1}'] = np.nan

weight_data = []

# Export based on the toggle configuration
if DO_SESSION_FITS:
    for (animal_id, session_id), model in session_models.items():
        X_s, y_s = session_data[(animal_id, session_id)]
        posterior_probs = model.expected_states(y_s, input=X_s)[0]
        
        mask = (export_df['animal'] == animal_id) & (export_df['session'] == session_id)
        idx = export_df[mask].index
        
        for k in range(chosen_K):
            export_df.loc[idx, f'prob_state_{k+1}'] = posterior_probs[:, k]
            w = model.observations.params[k][0]
            for p_idx, p_name in enumerate(predictor_cols):
                weight_data.append({'animal': animal_id, 'session': session_id, 'state': k+1, 'predictor': p_name, 'weight': w[p_idx]})
else:
    for animal_id, model in animal_models.items():
        X_a, y_a = animal_data[animal_id]
        posterior_probs = model.expected_states(y_a, input=X_a)[0]
        
        mask = (export_df['animal'] == animal_id)
        idx = export_df[mask].index
        
        for k in range(chosen_K):
            export_df.loc[idx, f'prob_state_{k+1}'] = posterior_probs[:, k]
            w = model.observations.params[k][0]
            for p_idx, p_name in enumerate(predictor_cols):
                weight_data.append({'animal': animal_id, 'session': 'ALL', 'state': k+1, 'predictor': p_name, 'weight': w[p_idx]})

# Save to CSV
filename_suffix = "_hierarchical.csv" if DO_SESSION_FITS else "_animal_level.csv"
export_cols = ['animal', 'session', 'trial_in_session', 'current_choice', 'prev_reward'] + [f'prob_state_{k+1}' for k in range(chosen_K)]

export_df[export_cols].to_csv(f'GLM_HMM_states_aligned{filename_suffix}', index=False)
pd.DataFrame(weight_data).to_csv(f'GLM_HMM_weights{filename_suffix}', index=False)

print(f"Analysis Complete. Outputs saved to GLM_HMM_...{filename_suffix}")