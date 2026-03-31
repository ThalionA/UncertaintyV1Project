%% Ideal Observer Model Fitting Framework (Hierarchical with BADS)
%
% This script implements a two-stage hierarchical fitting process.
% Stage 1: Fits the model to data pooled from all animals (group-level).
% Stage 2: Uses group parameters as priors to fit each individual animal.
%
% CHANGES:
% - Isotropic Precision: rho_k and phi_pref are fixed to 0.
% - Metacognitive Fit: Fits Licks and Velocity. Choice is PREDICTED.
% - Split Confidence: Separate parameters for Licks vs Velocity.
% - Marginalized Posterior: Inversion now computes the fully marginalized Q(theta) 
%   instead of relying on the MAP approximation, improving theoretical accuracy.
%
% --- Clean up workspace
clear; close all; clc;
% Set seed for reproducibility
rng("twister");
warning off;
% Add BADS path
addpath(genpath('/Users/theoamvr/Desktop/Experiments/bads-master'));

%% --- Part 1: Data Loading for All Animals ---
fprintf('--- Part 1: Loading and Preprocessing Data for All Animals ---\n');

sesnames1 = {'20250605_Cb15', '20250613_Cb15', '20250620_Cb15', '20250624_Cb15', '20250709_Cb15'};
sesnames2 = {'20250606_Cb17', '20250613_Cb17', '20250620_Cb17', '20250624_Cb17', '20250701_Cb17'};
sesnames3 = {'20250904_Cb21', '20250910_Cb21', '20250911_Cb21', '20250912_Cb21', '20250918_Cb21'};
sesnames4 = {'20251024_Cb22', '20251027_Cb22', '20251028_Cb22', '20251030_Cb22', '20251105_Cb22'};
sesnames5 = {'20250918_Cb24', '20250919_Cb24', '20251020_Cb24', '20251021_Cb24'};
sesnames6 = {'20250903_Cb25', '20250904_Cb25', '20250910_Cb25', '20250911_Cb25', '20250916_Cb25'};

% For animals 1..N, specify 'horizontal' (go <45) or 'vertical' (go >45)
go_side_by_animal = { ...
    'horizontal', ... % animal 1
    'horizontal', ... % animal 2
    'vertical', ... % animal 3
    'vertical', ... % animal 4
    'vertical', ... % animal 5
    'horizontal' ... % animal 6
    };

all_sesnames = {sesnames1, sesnames2, sesnames3, sesnames4, sesnames5, sesnames6}; 
all_data = {};
all_animal_ids = [];

for i_animal = 1:numel(all_sesnames)
    sessions_to_load = all_sesnames{i_animal};
    fprintf('\n--- Loading data for Animal %d ---\n', i_animal);
    try
        % --- Loop Through Sessions and Pool Trial-by-Trial Data ---
        all_orientations = {}; all_contrasts = {}; all_dispersions = {};
        all_choices = {}; all_preRZLicks = {}; all_preRZVel = {};
        trial_keys_session = {};   
        trial_keys_sidx    = [];   
        trial_keys_tidx    = [];   
        
        for i_ses = 1:numel(sessions_to_load)
            sesname = sessions_to_load{i_ses};
            fprintf(' Loading session: %s\n', sesname);
            loaded_data = load(['vr_' sessions_to_load{i_ses} '_light.mat'], 'vr');
            vr = loaded_data.vr;
            if isfield(vr.cfg, 'sessionType') && strcmpi(vr.cfg.sessionType, 'basic'), continue; end
            
            n_trials_session = numel(vr.trialLog);
            all_orientations{end+1} = vr.cfg.trialOrientations(1:n_trials_session)';
            
            trialContrasts = vr.cfg.trialContrasts(1:n_trials_session)';
            trialContrasts(trialContrasts > 0.9) = 1;
            all_contrasts{end+1} = round(trialContrasts, 2);
            
            all_dispersions{end+1} = vr.cfg.trialDispersions(1:n_trials_session)';
            
            rzLicks = arrayfun(@(t) sum(vr.trialLog{t}.lick(vr.trialLog{t}.position(2,:) >= vr.cfg.rewardZoneStart & vr.trialLog{t}.position(2,:) <= vr.cfg.rewardZoneEnd)), 1:n_trials_session)';
            all_choices{end+1} = (rzLicks > 0);
            
            all_preRZLicks{end+1} = arrayfun(@(t) sum(vr.trialLog{t}.lick(vr.trialLog{t}.position(2,:) >= vr.cfg.rewardZoneStart-10 & vr.trialLog{t}.position(2,:) < vr.cfg.rewardZoneStart)), 1:n_trials_session)';
            
            preVel = nan(n_trials_session, 1);
            for t = 1:n_trials_session
                inPreRZ = vr.trialLog{t}.position(2,:) >= vr.cfg.rewardZoneStart-10 & vr.trialLog{t}.position(2,:) < vr.cfg.rewardZoneStart;
                if any(inPreRZ), preVel(t) = mean(-vr.trialLog{t}.velocity(2, inPreRZ)*0.2537, 'omitnan'); end
            end
            all_preRZVel{end+1} = preVel;
            
            trial_keys_session{end+1} = repmat({sesname}, n_trials_session, 1);
            trial_keys_sidx  = [trial_keys_sidx;  repmat(i_ses, n_trials_session, 1)];
            trial_keys_tidx  = [trial_keys_tidx;  (1:n_trials_session)'];
        end
        data.orientation = horzcat(all_orientations{:})';
        data.contrast = horzcat(all_contrasts{:})';
        data.dispersion = horzcat(all_dispersions{:})';
        data.choices = vertcat(all_choices{:});
        
        preRZLicks = vertcat(all_preRZLicks{:});
        preRZVel = vertcat(all_preRZVel{:});
        data.n_trials = length(data.orientation);
        
        % Handle missing
        preRZLicks(isnan(preRZLicks)) = mean(preRZLicks, 'omitmissing');
        preRZVel(isnan(preRZVel)) = mean(preRZVel, 'omitmissing');
        
        % robust z-scores
        zLicks = zscore_safe(preRZLicks);
        zVel   = zscore_safe(preRZVel);
        
        % --- SEPARATE CONFIDENCE STREAMS ---
        data.conf_licks = zLicks; 
        data.conf_vel   = zVel;
        % -----------------------------------
        
        data.trial_keys.session_name    = vertcat(trial_keys_session{:});
        data.trial_keys.session_idx     = trial_keys_sidx;
        data.trial_keys.trial_in_session= trial_keys_tidx;
        
        % align by declared go side
        data = align_data_to_go_by_side(data, go_side_by_animal{i_animal});
        all_data{end+1}    = data;
        all_animal_ids(end+1) = i_animal;
        fprintf(' Data for Animal %d loaded successfully (%d trials).\n', i_animal, data.n_trials);
    catch ME
        fprintf('\n***\nCOULD NOT LOAD REAL DATA FOR ANIMAL %d. SKIPPING.\nError: %s\n***\n\n', i_animal, ME.message);
    end
end

% --- Pool data from all loaded animals for the group-level fit ---
data_pooled = pool_data_structs(all_data);
fprintf('\n--- All data loaded and pooled. Total animals: %d. Total trials: %d ---\n', numel(all_data), data_pooled.n_trials);

%% --- Part 2: Model Configuration ---
fprintf('\n--- Part 2: Configuring Model for Fitting ---\n');

% --- FIT MODE: 'conf_only' ---
% We ignore choice likelihoods and only fit licks + velocity
fit_mode = 'conf_only'; 

% --- PARAMETERS ---
% 1. SENSORY (Isotropic: kappa_amp only)
sensory_params = {'kappa_amp', 'c_power', 'd_power'};
% 2. DECISION (Beta)
% We are NOT fitting this. It will be fixed to 1.0.
choice_params = {}; 
% 3. CONFIDENCE (Licks & Velocity)
% Separate parameters for the two independent streams
conf_params = { ...
    'lick_slope', 'lick_intercept', 'lick_std', ...
    'vel_slope',  'vel_intercept',  'vel_std'};
model_spec.fit_params = [sensory_params, choice_params, conf_params];

% --- INITIAL GUESSES ---
% [kappa_amp, c_power, d_power, lick_s, lick_i, lick_std, vel_s, vel_i, vel_std]
initial_guesses = [ ...
    10, 1.0, 1.0, ...    % Sensory
    2.0, 0.0, 0.5, ...   % Licks: Positive Slope (High DV -> High Licks)
   -2.0, 0.0, 0.5];      % Velocity: Negative Slope (High DV -> Low Velocity)

% --- FIXED PARAMETERS ---
model_spec.fixed_params.s_range_deg = 0:1:90;
model_spec.fixed_params.m_range_deg = 0:1:180;
model_spec.fixed_params.prior_type = 'Bimodal';
model_spec.fixed_params.prior_strength = 3;
model_spec.fixed_params.omega = 1;
model_spec.fixed_params.kappa_min = 1.0;

% Force Isotropic (Remove oblique effect)
model_spec.fixed_params.rho_k = 0;
model_spec.fixed_params.phi_pref = 0;

% Fix Beta (Standardizes the DV scale)
if strcmp(fit_mode, 'conf_only')
    model_spec.fixed_params.decision_beta = 1.0;
end

% --- Define FIXED Utility ---
fixed_utility = struct('R_hit', 1, 'R_miss', 0, 'R_cr', 0.1, 'R_fa', -0.2);
fprintf(' Model configured for CONFIDENCE-ONLY fitting (Isotropic Precision).\n');

if ~exist('IOResults','var')
    IOResults = struct();
    IOResults.meta.timestamp   = datestr(now);
    IOResults.meta.model_spec  = model_spec;
    IOResults.meta.fit_mode    = fit_mode;
    IOResults.meta.fixed_utility = fixed_utility; 
    IOResults.group.params     = [];   
    IOResults.group.avg_test_nll = [];
    IOResults.animals          = {};   
end

%% --- Part 3: Group-level and Individual Fit ---
% --- Stage 1: Group-Level Fit ---
fprintf('\n--- Part 3a: Running Stage 1 Group-Level Fit ---\n');
tic;
group_fit_output = fit_model_crossval(initial_guesses, data_pooled, model_spec, fixed_utility, fit_mode);
toc;
fprintf(' Group fit complete. Average Test NLL: %.3f\n', group_fit_output.avg_test_nll);
group_level_params = group_fit_output.params;

IOResults.group.params       = group_level_params;
IOResults.group.avg_test_nll = group_fit_output.avg_test_nll;

% --- Stage 2: Individual-Level Fit ---
fprintf('\n--- Part 3b: Running Stage 2 Individual-Level Fits ---\n');
individual_results = {};

for i_animal = 1:numel(all_data)
    animal_id = all_animal_ids(i_animal);
    fprintf('\n--- Fitting animal %d ---\n', animal_id);
    current_animal_data = all_data{i_animal};
    
    % Use group parameters as the starting point for individual fits
    tic;
    individual_fit_output = fit_model_crossval(group_level_params, current_animal_data, model_spec, fixed_utility, fit_mode);
    toc;
    fprintf(' Animal %d done - average test NLL: %.3f\n', animal_id, individual_fit_output.avg_test_nll);
    
    % --- Assemble final fitted parameters and utility ---
    res.fit_params_vec = individual_fit_output.params;
    res.spec = model_spec;
    res.test_nll = individual_fit_output.avg_test_nll;
    
    % Unpack fitted params
    fit_params_struct = model_spec.fixed_params;
    for i_param = 1:length(individual_fit_output.params)
        param_name = model_spec.fit_params{i_param};
        fit_params_struct.(param_name) = individual_fit_output.params(i_param);
    end
    
    res.full_fit_params = fit_params_struct;
    res.final_utility = fixed_utility;
    individual_results{i_animal} = res;
end

%% --- Part 4: Inversion and Saving ---
for i_animal = 1:numel(all_data)
    animal_id = all_animal_ids(i_animal);
    fprintf('\n\n--- Running Inversion & Saving for Animal %d ---\n', animal_id);
    data = all_data{i_animal};
    fit_result = individual_results{i_animal};
    
    analysis_params = fit_result.full_fit_params;
    analysis_utility = fit_result.final_utility;
    analysis_spec = fit_result.spec;
    
    % --- Part 5: Deeper Analyses (Inversion) ---
    fprintf(' Inverting model for trial-by-trial uncertainty...\n');
    inferred_uncertainties = invert_model_for_single_trial_uncertainty(data, analysis_params, analysis_utility, analysis_spec);
    
    % ---------- SAVE THIS ANIMAL ----------
    ani = struct();
    if isfield(data,'animal_tag')
        ani.tag = data.animal_tag;
    else
        ani.tag = sprintf('Animal_%d', animal_id);
    end
    ani.go_side = go_side_by_animal{i_animal}; 
    
    % Raw inputs
    ani.data = struct( ...
        'orientation', data.orientation(:), ...
        'contrast',    data.contrast(:), ...
        'dispersion',  data.dispersion(:), ...
        'choices',     data.choices(:), ...
        'conf_licks',  data.conf_licks(:), ... % Saved separately
        'conf_vel',    data.conf_vel(:), ...   % Saved separately
        'n_trials',    data.n_trials, ...
        'trial_keys',  data.trial_keys ...
        );
        
    % Fitted params
    ani.fit.params_vec   = individual_results{i_animal}.fit_params_vec;
    ani.fit.full_params  = individual_results{i_animal}.full_fit_params;
    ani.fit.utility      = individual_results{i_animal}.final_utility; 
    ani.fit.avg_test_nll = individual_results{i_animal}.test_nll;
    
    % Trialwise model predictions (Using FIXED beta=1 to predict choice)
    condMat_all = round([data.orientation, data.contrast, data.dispersion], 3);
    [G_all, ~, G_idx_all] = unique(condMat_all, 'rows');
    preds = calculate_model_predictions(G_all, ani.fit.full_params, ani.fit.utility, data);
    
    ani.pred.p_respond  = preds.p_respond(G_idx_all); % Soft predicted choice
    ani.pred.choice_hat = preds.binary_choice(G_idx_all); % Step predicted choice
    ani.pred.licks      = preds.lick_prediction(G_idx_all); % Predicted Licks
    ani.pred.vel        = preds.vel_prediction(G_idx_all); % Predicted Vel
    
    % Trialwise inferred uncertainties (Now mapped to marginalized values by default!)
    ani.inferred.perceptual     = inferred_uncertainties.perc_unc_marginal(:);
    ani.inferred.decision       = inferred_uncertainties.dec_unc_marginal(:);
    ani.inferred.eu_go          = inferred_uncertainties.eu_go_marginal(:);
    ani.inferred.eu_nogo        = inferred_uncertainties.eu_nogo_marginal(:);
    ani.inferred.post_s_marginal= inferred_uncertainties.post_s_marginal; % Full marginalized distribution
    
    % Store the MAP versions for comparison/backup
    ani.inferred.perceptual_map = inferred_uncertainties.perceptual_map(:);
    ani.inferred.decision_map   = inferred_uncertainties.decision_map(:);
    ani.inferred.post_s_given_map = inferred_uncertainties.post_s_given_map;
    
    ani.inferred.m_posteriors   = inferred_uncertainties.m_posteriors;
    ani.inferred.L_s_given_map    = inferred_uncertainties.L_s_given_map;
    
    % Convenience table
    ani.trial_table = table( ...
        ani.data.orientation, ani.data.contrast, ani.data.dispersion, ...
        ani.data.choices, ani.data.conf_licks, ani.data.conf_vel, ...
        ani.pred.p_respond, ani.pred.choice_hat, ani.pred.licks, ani.pred.vel, ...
        ani.inferred.perceptual, ani.inferred.decision, ...
        ani.inferred.perceptual_map, ani.inferred.decision_map, ...
        ani.inferred.eu_go, ani.inferred.eu_nogo, ...
        string(ani.data.trial_keys.session_name), ...
        ani.data.trial_keys.trial_in_session, ...
        'VariableNames', { ...
        'orientation','contrast','dispersion', ...
        'choice', 'licks_z','vel_z', ...
        'p_respond_model', 'choice_pred_binary','licks_model','vel_model', ...
        'unc_perceptual','unc_decision', ...
        'unc_perceptual_map','unc_decision_map', ...
        'eu_go','eu_nogo', ...
        'session_name','trial_in_session' } ...
        );
        
    IOResults.animals{end+1} = ani;
    fprintf(' Animal %d data saved to IOResults.\n', animal_id);
end

%% --- Part 5: Save Final Results ---
fprintf('\n--- Saving final IOResults structure ---\n');
save('IOResults.mat', 'IOResults', '-v7.3');
fprintf('All fitting, inversion, and saving complete.\n');

%% --- Core Fitting & Analysis Functions ---

function data_pooled = pool_data_structs(all_data_cells)
% pool data from all animals
% Note: pooling separate conf fields
fields_to_cat = {'orientation', 'contrast', 'dispersion', 'choices', 'conf_licks', 'conf_vel'};
for i = 1:length(fields_to_cat)
    fn = fields_to_cat{i};
    pooled_field_data = cellfun(@(d) d.(fn), all_data_cells, 'UniformOutput', false);
    data_pooled.(fn) = vertcat(pooled_field_data{:});
end
data_pooled.n_trials = length(data_pooled.orientation);
end

function fit_output = fit_model_crossval(initial_guesses, data, model_spec, utility, fit_mode)
[lb, ub, plb, pub] = get_bads_bounds(model_spec.fit_params, initial_guesses);
bads_options = bads('defaults');
bads_options.Display = 'off';
k_folds = 5; n_obs = data.n_trials;
cv_indices = cvpartition(n_obs, 'KFold', k_folds);

recovered_params_kfold = zeros(k_folds, length(initial_guesses));
test_nll_kfold = zeros(k_folds, 1);

for k = 1:cv_indices.NumTestSets
    train_idx = cv_indices.training(k);
    test_idx = cv_indices.test(k);
    
    jitter = (rand(size(initial_guesses)) - 0.5) .* (pub - plb) * 0.1;
    p_initial_jittered = initial_guesses + jitter;
    p_initial_jittered(p_initial_jittered>ub) = ub(p_initial_jittered>ub);
    p_initial_jittered(p_initial_jittered<lb) = lb(p_initial_jittered<lb);
    
    obj_fun_fold = @(p) calculate_NLL(p, data, model_spec, utility, train_idx, fit_mode);
    p_k_fit = bads(obj_fun_fold, p_initial_jittered, lb, ub, plb, pub, [], bads_options);
    recovered_params_kfold(k, :) = p_k_fit;
    test_nll_kfold(k) = calculate_NLL(p_k_fit, data, model_spec, utility, test_idx, fit_mode);
end

[~, best_fold_idx] = min(test_nll_kfold);
best_p_from_cv = recovered_params_kfold(best_fold_idx, :);

obj_fun_all_data = @(p) calculate_NLL(p, data, model_spec, utility, 1:n_obs, fit_mode);
best_p_final_fit = bads(obj_fun_all_data, best_p_from_cv, lb, ub, plb, pub, [], bads_options);

fit_output.params = best_p_final_fit;
fit_output.avg_test_nll = mean(test_nll_kfold);
end

function [lb, ub, plb, pub] = get_bads_bounds(param_names, initial_guesses)
n_params = length(param_names);
lb = zeros(1, n_params); ub = zeros(1, n_params);
for i = 1:n_params
    name = param_names{i};
    if strcmp(name, 'kappa_amp'), lb(i)=0; ub(i)=50;
    elseif contains(name, 'power'), lb(i)=0; ub(i)=5;
    elseif strcmp(name, 'decision_beta'), lb(i)=0.1; ub(i)=10;
        
    % New Confidence Params
    elseif contains(name, '_slope'), lb(i)=-20; ub(i)=20; % Allow neg or pos slope
    elseif contains(name, '_intercept'), lb(i)=-5; ub(i)=5;
    elseif contains(name, '_std'), lb(i)=0.01; ub(i)=5;
    else, lb(i)=-inf; ub(i)=inf;
    end
end
plb = max(lb, initial_guesses - abs(initial_guesses)*0.5 - 0.1);
pub = min(ub, initial_guesses + abs(initial_guesses)*0.5 + 0.1);
end

function NLL = calculate_NLL(p_natural, data, model_spec, utility, data_idx, fit_mode)
    % Unpack parameters
    params = model_spec.fixed_params;
    for i = 1:length(p_natural)
        params.(model_spec.fit_params{i}) = p_natural(i);
    end
    
    % Extract subset of data
    fold.s = data.orientation(data_idx);
    fold.c = data.contrast(data_idx);
    fold.d = data.dispersion(data_idx);
    fold.ch = data.choices(data_idx);
    fold.licks = data.conf_licks(data_idx); % New field
    fold.vel = data.conf_vel(data_idx);     % New field
    
    if isempty(fold.s), NLL = 0; return; end
    
    % Group unique conditions
    condMat = [fold.s, fold.c, fold.d];
    [G, ~, G_idx] = unique(condMat, 'rows');
    n_conds = size(G, 1);
    
    % Grids and Utility
    s_rad = deg2rad(params.s_range_deg);
    m_rad = deg2rad(params.m_range_deg);
    prior = get_prior(s_rad, params, data); 
    util  = get_utility_vectors(params.s_range_deg, utility);
    
    total_NLL = 0;
    for j = 1:n_conds
        % 1. Get Params for this condition
        s_val = G(j,1); c_val = G(j,2); d_val = G(j,3);
        k_gen = get_kappa_for_trial(s_val, c_val, d_val, params);
        
        % 2. Inference (Posterior over s given m)
        Lik_inf = pdfVonMises(m_rad', s_rad, k_gen);
        Post_s_m = (Lik_inf .* prior) ./ (sum(Lik_inf .* prior, 2) + eps);
        
        % 3. Calculate Decision Variable (DV) for every possible m
        eu_go = Post_s_m * util.respond';
        eu_nogo = Post_s_m * util.no_respond';
        dv = eu_go - eu_nogo; 
        
        % 4. Map DV to Predictions
        % A. Choice Probability (Sigmoid)
        p_go_m = 1 ./ (1 + exp(-params.decision_beta * dv));
        % B. Lick Prediction (Linear from DV)
        mu_licks = params.lick_slope * dv + params.lick_intercept;
        % C. Velocity Prediction (Linear from DV)
        mu_vel = params.vel_slope * dv + params.vel_intercept;
        
        % 5. Generative Probability of m given true stimulus s
        p_m_s = pdfVonMises(m_rad, deg2rad(s_val), k_gen);
        p_m_s = p_m_s ./ (sum(p_m_s) + eps);
        
        % 6. Calculate Likelihood of Data
        mask = (G_idx == j);
        obs_ch = fold.ch(mask);
        obs_licks = fold.licks(mask);
        obs_vel = fold.vel(mask);
        n_curr = sum(mask);
        
        % Expand predictions
        P_Go_Mat   = repmat(p_go_m', n_curr, 1);
        Mu_Licks_Mat = repmat(mu_licks', n_curr, 1);
        Mu_Vel_Mat   = repmat(mu_vel', n_curr, 1);
        P_m_Mat      = repmat(p_m_s, n_curr, 1);
        
        % --- LIKELIHOOD COMPONENTS ---
        if strcmp(fit_mode, 'conf_only')
            L_choice = 1; % Ignore choice data
        else
            L_choice = (P_Go_Mat .* obs_ch) + ((1 - P_Go_Mat) .* (1 - obs_ch));
        end
        
        L_licks = normpdf(repmat(obs_licks,1,numel(m_rad)), Mu_Licks_Mat, params.lick_std);
        L_vel = normpdf(repmat(obs_vel,1,numel(m_rad)), Mu_Vel_Mat, params.vel_std);
        
        % Integrate over m
        L_joint_m = L_choice .* L_licks .* L_vel;
        L_trial = sum(L_joint_m .* P_m_Mat, 2);
        
        total_NLL = total_NLL - sum(log(L_trial + eps));
    end
    NLL = total_NLL;
    if isnan(NLL) || isinf(NLL), NLL = 1e10; end
end

function model_preds = calculate_model_predictions(trial_conditions_mat, params, utility, current_data)
    s_range_rad = deg2rad(params.s_range_deg); 
    m_range_rad = deg2rad(params.m_range_deg);
    
    prior = get_prior(s_range_rad, params, current_data);
    utility_vec = get_utility_vectors(params.s_range_deg, utility);
    
    n_conds = size(trial_conditions_mat, 1);
    
    % Initialize outputs
    p_respond_all_conds = zeros(n_conds, 1); % Probabilistic (Soft)
    binary_choice_all_conds = zeros(n_conds, 1); % Deterministic (Hard)
    licks_pred_all = zeros(n_conds, 1);
    vel_pred_all   = zeros(n_conds, 1);
    
    for i_cond = 1:n_conds
        s_i = trial_conditions_mat(i_cond, 1); 
        c_i = trial_conditions_mat(i_cond, 2); 
        d_i = trial_conditions_mat(i_cond, 3);
        
        % 1. Generative Distribution p(m|s)
        kappa_gen_i = get_kappa_for_trial(s_i, c_i, d_i, params);
        Pm_given_s_i = pdfVonMises(m_range_rad, deg2rad(s_i), kappa_gen_i);
        Pm_given_s_i = Pm_given_s_i ./ (sum(Pm_given_s_i) + eps); 
        
        % 2. Inference for every m
        likelihood_mat_i = pdfVonMises(m_range_rad', s_range_rad, kappa_gen_i);
        posterior_i = (likelihood_mat_i .* prior) ./ (sum(likelihood_mat_i .* prior, 2) + eps);
        
        % 3. Expected Utility & Decision Variable
        EU_respond = posterior_i * utility_vec.respond';
        EU_no_respond = posterior_i * utility_vec.no_respond';
        dv = EU_respond - EU_no_respond;
        
        % 4. PREDICTIONS
        
        % A. Licks/Vel (Fitted)
        licks_vs_m = params.lick_slope * dv + params.lick_intercept;
        vel_vs_m   = params.vel_slope  * dv + params.vel_intercept;
        
        % B. "Soft" Probability (using fixed beta=1, mostly for gradient continuity)
        p_soft_vs_m = 1 ./ (1 + exp(-params.decision_beta * dv));
        
        % C. "Hard" Binary Choice (Pure Prediction)
        % If DV > 0, Choice = 1. Else 0.
        choice_binary_vs_m = double(dv > 0);
        
        % 5. Marginalize over m to get expected behavior for this stimulus
        p_respond_all_conds(i_cond) = sum(p_soft_vs_m' .* Pm_given_s_i);
        binary_choice_all_conds(i_cond) = sum(choice_binary_vs_m' .* Pm_given_s_i);
        
        licks_pred_all(i_cond)      = sum(licks_vs_m' .* Pm_given_s_i);
        vel_pred_all(i_cond)        = sum(vel_vs_m' .* Pm_given_s_i);
    end
    
    model_preds.p_respond = p_respond_all_conds;         % Soft (Beta=1)
    model_preds.binary_choice = binary_choice_all_conds; % Hard (Prediction)
    model_preds.lick_prediction = licks_pred_all;
    model_preds.vel_prediction  = vel_pred_all;
end

function inferred_unc = invert_model_for_single_trial_uncertainty(data, params, utility, base_spec)
% REFACTORED: Now correctly marginalizes across the latent variable m to
% generate Q(theta), yielding theoretically rigorous perceptual and decision 
% uncertainty trial-by-trial. Includes underflow protection and uses linear
% standard deviation for perceptual uncertainty.

m_range_rad = deg2rad(base_spec.fixed_params.m_range_deg);
s_range_deg = base_spec.fixed_params.s_range_deg;
s_range_rad = deg2rad(s_range_deg);
n_m = numel(m_range_rad);
n_s = numel(s_range_rad);

prior_ps = get_prior(s_range_rad, params, data);
utility_vec = get_utility_vectors(s_range_deg, utility);

cond_mat = [data.orientation, data.contrast, data.dispersion];
[G_unique_conds, ~, trial_to_cond_idx] = unique(cond_mat, 'rows');
n_unique_conds = size(G_unique_conds, 1);

maps_perc_unc    = cell(n_unique_conds, 1);
maps_dec_unc     = cell(n_unique_conds, 1);
maps_lik_behavior = cell(n_unique_conds, 1); 
maps_eu_go = cell(n_unique_conds, 1);
maps_eu_nogo = cell(n_unique_conds, 1);
maps_p_s_given_m = cell(n_unique_conds, 1);
maps_L_s_given_m = cell(n_unique_conds, 1); % New: Likelihood map

generative_kappas = zeros(n_unique_conds, 1);

% --- Pre-calculate Category Indices ---
is_go_stim = s_range_deg < 45;
is_nogo_stim = s_range_deg > 45;
is_boundary = s_range_deg == 45;

for j = 1:n_unique_conds
    s_j = G_unique_conds(j, 1); c_j = G_unique_conds(j, 2); d_j = G_unique_conds(j, 3);
    kappa_gen = get_kappa_for_trial(s_j, c_j, d_j, params);
    generative_kappas(j) = kappa_gen;
    
    lik_mat = pdfVonMises(m_range_rad', s_range_rad, kappa_gen);
    maps_L_s_given_m{j} = lik_mat; % Store likelihood (p(m|s)) for output
    
    post_s_m = (lik_mat .* prior_ps);
    post_s_m = post_s_m ./ (sum(post_s_m, 2) + eps);
    maps_p_s_given_m{j} = post_s_m;
    
    % Perceptual Uncertainty (Linear Standard Deviation):
    mean_s_map = sum(post_s_m .* s_range_deg, 2);
    var_s_map = sum(post_s_m .* (s_range_deg - mean_s_map).^2, 2);
    maps_perc_unc{j} = sqrt(var_s_map);
    
    % Expected Utility
    eu_go = post_s_m * utility_vec.respond';
    eu_nogo = post_s_m * utility_vec.no_respond';
    maps_eu_go{j} = eu_go;
    maps_eu_nogo{j} = eu_nogo;
    
    % --- Decision Uncertainty (Entropy of Category Posterior) - For MAP calc
    p_stim_go = sum(post_s_m(:, is_go_stim), 2) + 0.5 * sum(post_s_m(:, is_boundary), 2);
    p_stim_nogo = 1 - p_stim_go;
    
    cat_entropy = -(p_stim_go .* log2(p_stim_go + eps) + ...
                    p_stim_nogo .* log2(p_stim_nogo + eps));
                
    maps_dec_unc{j} = cat_entropy;
    
    % Linear mappings for behavioral predictions
    dv = eu_go - eu_nogo; 
    maps_lik_behavior{j}.pred_lick = params.lick_slope * dv + params.lick_intercept;
    maps_lik_behavior{j}.pred_vel  = params.vel_slope  * dv + params.vel_intercept;
end

n_trials = data.n_trials;

% Pre-allocate outputs
inferred_unc.perc_unc_marginal = nan(n_trials, 1);
inferred_unc.dec_unc_marginal  = nan(n_trials, 1);
inferred_unc.eu_go_marginal    = nan(n_trials, 1);
inferred_unc.eu_nogo_marginal  = nan(n_trials, 1);
inferred_unc.post_s_marginal   = nan(n_trials, n_s);

% Map-specific fields for backwards compat / comparison
inferred_unc.m_posteriors = nan(n_trials, n_m);
inferred_unc.perceptual_map = nan(n_trials, 1);
inferred_unc.decision_map = nan(n_trials, 1);
inferred_unc.L_s_given_map = nan(n_trials, n_s);
inferred_unc.post_s_given_map = nan(n_trials, n_s);

for i = 1:n_trials
    cond_idx = trial_to_cond_idx(i);
    s_true = data.orientation(i);
    
    % Generative Prior p(m|s)
    kappa_gen = generative_kappas(cond_idx);
    prior_m = pdfVonMises(m_range_rad, deg2rad(s_true), kappa_gen);
    prior_m = prior_m / (sum(prior_m) + eps);
    
    % Likelihood of Confidence (Licks + Vel)
    beh_maps = maps_lik_behavior{cond_idx};
    L_l = normpdf(data.conf_licks(i), beh_maps.pred_lick, params.lick_std);
    L_v = normpdf(data.conf_vel(i),   beh_maps.pred_vel,  params.vel_std);
    
    post_m_unnorm = L_l' .* L_v' .* prior_m;
    
    % --- FIX: Underflow Protection ---
    sum_post = sum(post_m_unnorm);
    if sum_post == 0 || isnan(sum_post)
        post_m = prior_m; % Fallback to prior if kinematics are astronomically unlikely
    else
        post_m = post_m_unnorm / sum_post;
    end
    
    inferred_unc.m_posteriors(i, :) = post_m;
    
    % =========================================================================
    % STEP 3: MARGINALISED Q(theta)
    % =========================================================================
    post_s_m_i = maps_p_s_given_m{cond_idx}; % [n_m x n_s]
    Q_marg = post_m * post_s_m_i;            % [1 x n_m] * [n_m x n_s] = [1 x n_s]
    Q_marg = Q_marg ./ (sum(Q_marg) + eps);
    
    inferred_unc.post_s_marginal(i, :) = Q_marg;
    
    % --- FIX: Linear Standard Deviation ---
    mean_s_marg = sum(Q_marg .* s_range_deg);
    var_s_marg = sum(Q_marg .* (s_range_deg - mean_s_marg).^2);
    inferred_unc.perc_unc_marginal(i) = sqrt(var_s_marg);
    
    % Decision uncertainty - binary entropy of P(Go|Q)
    p_go_marg  = sum(Q_marg(is_go_stim)) + 0.5 * sum(Q_marg(is_boundary));
    p_go_marg  = max(eps, min(1-eps, p_go_marg));
    p_ngo_marg = 1 - p_go_marg;
    inferred_unc.dec_unc_marginal(i) = -(p_go_marg * log2(p_go_marg) + p_ngo_marg * log2(p_ngo_marg));
    
    % Expected Utilities
    inferred_unc.eu_go_marginal(i)   = Q_marg * utility_vec.respond';
    inferred_unc.eu_nogo_marginal(i) = Q_marg * utility_vec.no_respond';

    % =========================================================================
    % STEP 4: Store MAP parameters for reference
    % =========================================================================
    [~, map_idx] = max(post_m);
    
    inferred_unc.perceptual_map(i) = maps_perc_unc{cond_idx}(map_idx);
    inferred_unc.decision_map(i)   = maps_dec_unc{cond_idx}(map_idx);
    
    % Store full distributions at MAP
    inferred_unc.L_s_given_map(i,:) = maps_L_s_given_m{cond_idx}(map_idx,:);
    inferred_unc.post_s_given_map(i,:) = maps_p_s_given_m{cond_idx}(map_idx,:);
end
end

%% --- Utility Functions ---

function kappa = get_kappa_for_trial(orientation, contrast, dispersion, params)
% Isotropic precision: depends only on contrast/dispersion, not orientation
kappa_0 = params.kappa_min + params.kappa_amp; 
kappa_contrast = (contrast.^params.c_power);
kappa_disp = exp(-params.d_power .* dispersion);
kappa = kappa_0 .* kappa_contrast .* kappa_disp;
end

function prior = get_prior(s_range_rad, params, data)
switch params.prior_type
    case 'Flat', prior = ones(1, length(s_range_rad));
    case 'Bimodal'
        c1 = pdfVonMises(s_range_rad, deg2rad(0), params.prior_strength);
        c2 = pdfVonMises(s_range_rad, deg2rad(90), params.prior_strength);
        prior = c1 + c2;
    case 'Unimodal', prior = pdfVonMises(s_range_rad, deg2rad(params.prior_center_deg), params.prior_strength);
    case 'Empirical'
        s_range_deg = rad2deg(s_range_rad);
        counts = histcounts(data.orientation, [s_range_deg s_range_deg(end)+1]);
        prior = smooth(counts, 3)';
end
prior = prior / (sum(prior) + eps);
end

function utility_vec = get_utility_vectors(s_range, utility)
n_s_bins = length(s_range);
utility_vec.respond = zeros(1, n_s_bins); utility_vec.no_respond = zeros(1, n_s_bins);
C1 = s_range < 45; C_boundary = s_range == 45; C2 = s_range > 45;

utility_vec.respond(C1) = utility.R_hit;
utility_vec.respond(C_boundary) = 0.5 * utility.R_hit + 0.5*utility.R_fa;
utility_vec.respond(C2) = utility.R_fa;

utility_vec.no_respond(C1) = utility.R_miss;
utility_vec.no_respond(C2) = utility.R_cr;
end

function p = pdfVonMises(x, mu, kappa)
p = exp(kappa .* cos(bsxfun(@minus, 2*x, 2*mu))) ./ (2 * pi * besseli(0, kappa));
end

function xz = zscore_safe(x)
mu = mean(x,'omitnan'); sd = std(x,[],'omitnan');
if ~isfinite(sd) || sd==0, xz = zeros(size(x)); else, xz = (x - mu)./sd; end
end

function data_out = align_data_to_go_by_side(data_in, go_side)
if ~isfield(data_in,'orientation')
    error('align_data_to_go_by_side: data has no field "orientation".');
end
side = lower(strtrim(go_side));
switch side
    case {'horizontal','h','<45','left'}    
        flip_needed = false;
    case {'vertical','v','>45','right'}
        flip_needed = true;
    otherwise
        error('align_data_to_go_by_side: unknown go side "%s". Use "horizontal" or "vertical".', go_side);
end
data_out = data_in;
if flip_needed
    data_out.orientation = 90 - data_in.orientation;
end
if any(data_out.orientation < 0 | data_out.orientation > 90)
    warning('align_data_to_go_by_side: orientations outside [0,90] after flip; clamping.');
    data_out.orientation = max(0, min(90, data_out.orientation));
end
end