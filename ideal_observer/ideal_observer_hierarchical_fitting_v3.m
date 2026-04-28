%% Ideal Observer Model Fitting Framework (Hierarchical with BADS)
%
% FITS: 
% - Stage 1: Group-level (pooled animals)
% - Stage 2: Individual-level (using group params as priors)
%
% FLEXIBLE CONFIGURATION:
% - Defines spatial window for kinematic extraction.
% - Independently toggles Choice, Licks, and Velocity for fitting.
% - Toggles Distribution assumptions (Normal vs Poisson) and automatically
%   adjusts parameters, link functions, and target data (Raw vs Z-scored).
%
% --- Clean up workspace
% clear; close all; clc;
rng("twister");
warning off;

% Add BADS path
addpath(genpath('/Users/theoamvr/Desktop/Experiments/bads-master'));

%% --- Part 0: USER CONFIGURATION ---
fprintf('--- Configuring User Settings ---\n');

% 1. SPATIAL WINDOW (in VR units relative to Reward Zone Start)
% Center = -5 and width = 10 extracts from [RZ_start - 10] to [RZ_start]
config.window_center_from_rz = -85;  
config.window_width          = 10;  

% 2. SOURCES OF INFORMATION FOR FITTING (Toggle true/false)
config.fit_choice = false;  % If false, beta is fixed to 1.0
config.fit_licks  = true;
config.fit_vel    = true;

if ~config.fit_choice && ~config.fit_licks && ~config.fit_vel
    error('You must set at least one fitting source (Choice, Licks, or Vel) to true.');
end

% 3. DISTRIBUTION ASSUMPTIONS
% Licks: 'Normal' (uses Z-scored data) OR 'Poisson' (uses Raw Integer Counts)
config.lick_dist = 'Normal'; 

% Velocity: 'Normal' (uses Z-scored data). 
config.vel_dist  = 'Normal';  

%% --- Part 1: Data Loading for All Animals ---
fprintf('\n--- Part 1: Loading and Preprocessing Data for All Animals ---\n');

sesnames1 = {'20250605_Cb15', '20250613_Cb15', '20250620_Cb15', '20250624_Cb15', '20250709_Cb15'};
sesnames2 = {'20250606_Cb17', '20250613_Cb17', '20250620_Cb17', '20250624_Cb17', '20250701_Cb17'};
sesnames3 = {'20250904_Cb21', '20250910_Cb21', '20250911_Cb21', '20250912_Cb21', '20250918_Cb21'};
sesnames4 = {'20251024_Cb22', '20251027_Cb22', '20251028_Cb22', '20251030_Cb22', '20251105_Cb22'};
sesnames5 = {'20250918_Cb24', '20250919_Cb24', '20251020_Cb24', '20251021_Cb24'};
sesnames6 = {'20250903_Cb25', '20250904_Cb25', '20250910_Cb25', '20250911_Cb25', '20250916_Cb25'};

go_side_by_animal = {'horizontal', 'horizontal', 'vertical', 'vertical', 'vertical', 'horizontal'};
all_sesnames = {sesnames1, sesnames2, sesnames3, sesnames4, sesnames5, sesnames6}; 

all_data = {};
all_animal_ids = [];

for i_animal = 1:numel(all_sesnames)
    sessions_to_load = all_sesnames{i_animal};
    fprintf('\n--- Loading data for Animal %d ---\n', i_animal);
    
    try
        all_orientations = {}; all_contrasts = {}; all_dispersions = {};
        all_choices = {}; all_preRZLicks = {}; all_preRZLickRates = {}; all_preRZVel = {};
        trial_keys_session = {}; trial_keys_sidx = []; trial_keys_tidx = [];   
        
        for i_ses = 1:numel(sessions_to_load)
            sesname = sessions_to_load{i_ses};
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
            all_choices{end+1} = double(rzLicks > 0);
            
            % --- FLEXIBLE SPATIAL WINDOW EXTRACTION ---
            preVel = nan(n_trials_session, 1);
            preLickCounts = nan(n_trials_session, 1);
            preLickRates = nan(n_trials_session, 1);
            
            win_start = vr.cfg.rewardZoneStart + config.window_center_from_rz - (config.window_width/2);
            win_end   = vr.cfg.rewardZoneStart + config.window_center_from_rz + (config.window_width/2);
            
            for t = 1:n_trials_session
                pos_log = vr.trialLog{t}.position(2,:);
                time_log = vr.trialLog{t}.time; 
                
                in_win = pos_log >= win_start & pos_log < win_end;
                
                if any(in_win)
                    preLickCounts(t) = sum(vr.trialLog{t}.lick(in_win));
                    preVel(t) = mean(-vr.trialLog{t}.velocity(2, in_win)*0.2537, 'omitnan'); 
                    
                    idx_entry = find(in_win, 1, 'first');
                    idx_exit  = find(in_win, 1, 'last');
                    time_in_win = time_log(idx_exit) - time_log(idx_entry);
                    
                    if time_in_win > 0.05 
                        preLickRates(t) = preLickCounts(t) / time_in_win;
                    else
                        preLickRates(t) = 0; 
                    end
                else
                    preLickCounts(t) = 0; 
                    preLickRates(t) = 0;
                end
            end
            
            if all(isnan(preVel))
                error('Spatial window extraction failed. All velocity values are NaN for session %s. Verify config.window_center_from_rz.', sesname);
            end

            all_preRZLicks{end+1} = preLickCounts;
            all_preRZLickRates{end+1} = preLickRates;
            all_preRZVel{end+1}   = preVel;
            
            trial_keys_session{end+1} = repmat({sesname}, n_trials_session, 1);
            trial_keys_sidx  = [trial_keys_sidx;  repmat(i_ses, n_trials_session, 1)];
            trial_keys_tidx  = [trial_keys_tidx;  (1:n_trials_session)'];
        end
        
        data.orientation = horzcat(all_orientations{:})';
        data.contrast = horzcat(all_contrasts{:})';
        data.dispersion = horzcat(all_dispersions{:})';
        data.choices = vertcat(all_choices{:});
        
        preRZLicks = vertcat(all_preRZLicks{:});
        preRZLickRates = vertcat(all_preRZLickRates{:});
        preRZVel = vertcat(all_preRZVel{:});
        data.n_trials = length(data.orientation);
        
        % Clean missing
        preRZLicks(isnan(preRZLicks)) = mean(preRZLicks, 'omitnan');
        preRZLickRates(isnan(preRZLickRates)) = mean(preRZLickRates, 'omitnan');
        preRZVel(isnan(preRZVel)) = mean(preRZVel, 'omitnan');
        
        % Store RAW values 
        data.raw_licks = round(preRZLicks); 
        data.raw_lick_rate = preRZLickRates;
        data.raw_vel   = preRZVel;
        
        % Store Z-SCORED values 
        data.z_licks = zscore_safe(preRZLicks); 
        data.z_vel   = zscore_safe(preRZVel);
        
        data.trial_keys.session_name    = vertcat(trial_keys_session{:});
        data.trial_keys.session_idx     = trial_keys_sidx;
        data.trial_keys.trial_in_session= trial_keys_tidx;
        
        data = align_data_to_go_by_side(data, go_side_by_animal{i_animal});
        all_data{end+1}    = data;
        all_animal_ids(end+1) = i_animal;
        
    catch ME
        fprintf('\n***\nCOULD NOT LOAD REAL DATA FOR ANIMAL %d. SKIPPING.\nError: %s\n***\n\n', i_animal, ME.message);
    end
end

data_pooled = pool_data_structs(all_data);
fprintf('\n--- All data loaded and pooled. Total trials: %d ---\n', data_pooled.n_trials);

%% --- Part 1b: Empirical Diagnostic Plots ---
figure('Name', 'Empirical Kinematic Distributions', 'Color', 'w', 'Position', [100, 100, 1200, 600]);

subplot(2,3,1); histogram(data_pooled.raw_licks, 30, 'FaceColor', [0.2 0.6 0.8]);
title('Raw Lick Counts (Poisson Target)'); xlabel('Count'); ylabel('Trials');

subplot(2,3,2); histogram(data_pooled.raw_lick_rate, 30, 'FaceColor', [0.3 0.7 0.5]);
title('Raw Lick Rates (Hz)'); xlabel('Licks/sec'); ylabel('Trials');

subplot(2,3,3); 
histogram(data_pooled.z_licks, 30, 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.7]); hold on;
x_grid = linspace(-4, 4, 100); plot(x_grid, normpdf(x_grid, 0, 1), 'r-', 'LineWidth', 2);
title('Z-Scored Licks vs Normal'); xlabel('Z-Score'); 

subplot(2,3,4); qqplot(data_pooled.raw_licks); title('Q-Q Plot: Raw Licks vs Normal');
subplot(2,3,5); qqplot(data_pooled.raw_vel); title('Q-Q Plot: Raw Velocity vs Normal');

subplot(2,3,6); 
histogram(data_pooled.z_vel, 30, 'Normalization', 'pdf', 'FaceColor', [0.8 0.4 0.4]); hold on;
plot(x_grid, normpdf(x_grid, 0, 1), 'k-', 'LineWidth', 2);
title('Z-Scored Velocity vs Normal'); xlabel('Z-Score'); 
drawnow;

%% --- Part 2: Model Configuration (Dynamic) ---
fprintf('\n--- Part 2: Configuring Model for Fitting ---\n');

% 1. SENSORY
sensory_params = {'kappa_amp', 'c_power', 'd_power'};
initial_guesses = [10, 1.0, 1.0];

% 2. DECISION
choice_params = {};
if config.fit_choice
    choice_params = {'decision_beta'};
    initial_guesses = [initial_guesses, 1.0];
end

% 3. CONFIDENCE (Licks & Velocity)
conf_params = {};
if config.fit_licks
    if strcmp(config.lick_dist, 'Normal')
        conf_params = [conf_params, {'lick_slope', 'lick_intercept', 'lick_std'}];
        initial_guesses = [initial_guesses, 2.0, 0.0, 0.5];
    elseif strcmp(config.lick_dist, 'Poisson')
        conf_params = [conf_params, {'lick_slope', 'lick_intercept'}];
        initial_guesses = [initial_guesses, 0.5, 1.0]; 
    end
end

if config.fit_vel
    conf_params = [conf_params, {'vel_slope', 'vel_intercept', 'vel_std'}];
    initial_guesses = [initial_guesses, -2.0, 0.0, 0.5];
end

model_spec.fit_params = [sensory_params, choice_params, conf_params];
model_spec.config = config; 

% --- FIXED PARAMETERS ---
model_spec.fixed_params.s_range_deg = 0:1:90;
model_spec.fixed_params.m_range_deg = 0:1:180;
model_spec.fixed_params.prior_type = 'Bimodal';
model_spec.fixed_params.prior_strength = 3;
model_spec.fixed_params.omega = 1;
model_spec.fixed_params.kappa_min = 1.0;
model_spec.fixed_params.rho_k = 0;
model_spec.fixed_params.phi_pref = 0;

if ~config.fit_choice
    model_spec.fixed_params.decision_beta = 1.0;
end

fixed_utility = struct('R_hit', 1, 'R_miss', 0, 'R_cr', 0.1, 'R_fa', -0.2);

if ~exist('IOResults','var')
    IOResults = struct();
    IOResults.meta.timestamp   = datestr(now);
    IOResults.meta.model_spec  = model_spec;
    IOResults.meta.fixed_utility = fixed_utility; 
    IOResults.group.params     = [];   
    IOResults.group.avg_test_nll = [];
    IOResults.animals          = {};   
end

%% --- Part 3: Group-level and Individual Fit ---
fprintf('\n--- Part 3a: Running Stage 1 Group-Level Fit ---\n');
tic;
group_fit_output = fit_model_crossval(initial_guesses, data_pooled, model_spec, fixed_utility);
toc;
fprintf(' Group fit complete. Average Test NLL: %.3f\n', group_fit_output.avg_test_nll);

group_level_params = group_fit_output.params;
IOResults.group.params       = group_level_params;
IOResults.group.avg_test_nll = group_fit_output.avg_test_nll;

fprintf('\n--- Part 3b: Running Stage 2 Individual-Level Fits ---\n');
individual_results = {};

for i_animal = 1:numel(all_data)
    animal_id = all_animal_ids(i_animal);
    fprintf('\n--- Fitting animal %d ---\n', animal_id);
    
    tic;
    individual_fit_output = fit_model_crossval(group_level_params, all_data{i_animal}, model_spec, fixed_utility);
    toc;
    fprintf(' Animal %d done - average test NLL: %.3f\n', animal_id, individual_fit_output.avg_test_nll);
    
    res.fit_params_vec = individual_fit_output.params;
    res.spec = model_spec;
    res.test_nll = individual_fit_output.avg_test_nll;
    
    fit_params_struct = model_spec.fixed_params;
    for i_param = 1:length(individual_fit_output.params)
        fit_params_struct.(model_spec.fit_params{i_param}) = individual_fit_output.params(i_param);
    end
    
    res.full_fit_params = fit_params_struct;
    res.final_utility = fixed_utility;
    individual_results{i_animal} = res;
end

%% --- Part 4: Inversion and Saving ---
for i_animal = 1:numel(all_data)
    animal_id = all_animal_ids(i_animal);
    fprintf('\n--- Running Inversion & Saving for Animal %d ---\n', animal_id);
    
    data = all_data{i_animal};
    fit_result = individual_results{i_animal};
    
    inferred_uncertainties = invert_model_for_single_trial_uncertainty(data, fit_result.full_fit_params, fit_result.final_utility, fit_result.spec);
    
    ani = struct();
    ani.tag = sprintf('Animal_%d', animal_id);
    ani.go_side = go_side_by_animal{i_animal}; 
    
    ani.data = struct('orientation', data.orientation(:), 'contrast', data.contrast(:), ...
        'dispersion', data.dispersion(:), 'choices', data.choices(:), ...
        'raw_licks', data.raw_licks(:), 'raw_vel', data.raw_vel(:), ...
        'z_licks', data.z_licks(:), 'z_vel', data.z_vel(:), ...
        'n_trials', data.n_trials, 'trial_keys', data.trial_keys);
        
    ani.fit.params_vec   = fit_result.fit_params_vec;
    ani.fit.full_params  = fit_result.full_fit_params;
    ani.fit.utility      = fit_result.final_utility; 
    ani.fit.avg_test_nll = fit_result.test_nll;
    
    condMat_all = round([data.orientation, data.contrast, data.dispersion], 3);
    [G_all, ~, G_idx_all] = unique(condMat_all, 'rows');
    preds = calculate_model_predictions(G_all, ani.fit.full_params, ani.fit.utility, data, fit_result.spec.config);
    
    ani.pred.p_respond  = preds.p_respond(G_idx_all); 
    ani.pred.choice_hat = preds.binary_choice(G_idx_all); 
    ani.pred.licks      = preds.lick_prediction(G_idx_all); 
    ani.pred.vel        = preds.vel_prediction(G_idx_all); 
    
    ani.inferred.perceptual     = inferred_uncertainties.perc_unc_marginal(:);
    ani.inferred.decision       = inferred_uncertainties.dec_unc_marginal(:);
    ani.inferred.eu_go          = inferred_uncertainties.eu_go_marginal(:);
    ani.inferred.eu_nogo        = inferred_uncertainties.eu_nogo_marginal(:);
    ani.inferred.post_s_marginal= inferred_uncertainties.post_s_marginal; 
    ani.inferred.likelihood_marginal = inferred_uncertainties.lik_s_marginal;
    
    ani.inferred.perceptual_map   = inferred_uncertainties.perceptual_map(:);
    ani.inferred.decision_map     = inferred_uncertainties.decision_map(:);
    ani.inferred.post_s_given_map = inferred_uncertainties.post_s_given_map;
    ani.inferred.m_posteriors     = inferred_uncertainties.m_posteriors;
    ani.inferred.L_s_given_map    = inferred_uncertainties.L_s_given_map;
    
    ani.trial_table = table( ...
        ani.data.orientation, ani.data.contrast, ani.data.dispersion, ...
        ani.data.choices, ani.data.raw_licks, ani.data.z_licks, ani.data.z_vel, ...
        ani.pred.p_respond, ani.pred.choice_hat, ani.pred.licks, ani.pred.vel, ...
        ani.inferred.perceptual, ani.inferred.decision, ...
        ani.inferred.perceptual_map, ani.inferred.decision_map, ...
        string(ani.data.trial_keys.session_name), ani.data.trial_keys.trial_in_session, ...
        'VariableNames', {'orientation','contrast','dispersion', 'choice', ...
        'raw_licks', 'licks_z','vel_z', ...
        'p_respond_model', 'choice_pred_binary','licks_model','vel_model', ...
        'unc_perceptual','unc_decision', 'unc_perceptual_map','unc_decision_map', ...
        'session_name','trial_in_session' });
        
    IOResults.animals{end+1} = ani;
end

%% --- Part 5: Save Final Results ---
fprintf('\n--- Saving final IOResults structure ---\n');
save('IOResults.mat', 'IOResults', '-v7.3');
fprintf('All fitting, inversion, and saving complete.\n');

%% --- Core Fitting & Analysis Functions ---

function data_pooled = pool_data_structs(all_data_cells)
fields_to_cat = {'orientation', 'contrast', 'dispersion', 'choices', ...
                 'raw_licks', 'raw_lick_rate', 'raw_vel', 'z_licks', 'z_vel'};
for i = 1:length(fields_to_cat)
    fn = fields_to_cat{i};
    pooled_field_data = cellfun(@(d) d.(fn), all_data_cells, 'UniformOutput', false);
    data_pooled.(fn) = vertcat(pooled_field_data{:});
end
data_pooled.n_trials = length(data_pooled.orientation);
end

function fit_output = fit_model_crossval(initial_guesses, data, model_spec, utility)
[lb, ub, plb, pub] = get_bads_bounds(model_spec.fit_params, initial_guesses, model_spec.config);
bads_options = bads('defaults'); bads_options.Display = 'off';

k_folds = 5; n_obs = data.n_trials;
cv_indices = cvpartition(n_obs, 'KFold', k_folds);

recovered_params_kfold = zeros(k_folds, length(initial_guesses));
test_nll_kfold = zeros(k_folds, 1);

for k = 1:cv_indices.NumTestSets
    train_idx = cv_indices.training(k);
    test_idx = cv_indices.test(k);
    
    jitter = (rand(size(initial_guesses)) - 0.5) .* (pub - plb) * 0.1;
    p_initial_jittered = max(lb, min(ub, initial_guesses + jitter));
    
    obj_fun_fold = @(p) calculate_NLL(p, data, model_spec, utility, train_idx);
    p_k_fit = bads(obj_fun_fold, p_initial_jittered, lb, ub, plb, pub, [], bads_options);
    
    recovered_params_kfold(k, :) = p_k_fit;
    test_nll_kfold(k) = calculate_NLL(p_k_fit, data, model_spec, utility, test_idx);
end

[~, best_fold_idx] = min(test_nll_kfold);
obj_fun_all_data = @(p) calculate_NLL(p, data, model_spec, utility, 1:n_obs);

fit_output.params = bads(obj_fun_all_data, recovered_params_kfold(best_fold_idx, :), lb, ub, plb, pub, [], bads_options);
fit_output.avg_test_nll = mean(test_nll_kfold);
end

function [lb, ub, plb, pub] = get_bads_bounds(param_names, initial_guesses, config)
n_params = length(param_names);
lb = zeros(1, n_params); ub = zeros(1, n_params);

for i = 1:n_params
    name = param_names{i};
    if strcmp(name, 'kappa_amp'), lb(i)=0; ub(i)=50;
    elseif contains(name, 'power'), lb(i)=0; ub(i)=5;
    elseif strcmp(name, 'decision_beta'), lb(i)=0.1; ub(i)=10;
    elseif contains(name, '_slope')
        % Tighter bounds for Poisson link to prevent exp() explosions
        if contains(name, 'lick') && isfield(config, 'lick_dist') && strcmp(config.lick_dist, 'Poisson')
            lb(i)=-5; ub(i)=5;
        else
            lb(i)=-20; ub(i)=20;
        end
    elseif contains(name, '_intercept'), lb(i)=-5; ub(i)=5;
    elseif contains(name, '_std'), lb(i)=0.01; ub(i)=5;
    else, lb(i)=-inf; ub(i)=inf;
    end
end

plb = max(lb, initial_guesses - abs(initial_guesses)*0.5 - 0.1);
pub = min(ub, initial_guesses + abs(initial_guesses)*0.5 + 0.1);
end

function NLL = calculate_NLL(p_natural, data, model_spec, utility, data_idx)
    params = model_spec.fixed_params;
    config = model_spec.config;
    for i = 1:length(p_natural), params.(model_spec.fit_params{i}) = p_natural(i); end
    
    fold.s = data.orientation(data_idx); fold.c = data.contrast(data_idx);
    fold.d = data.dispersion(data_idx);  fold.ch = data.choices(data_idx);
    
    fold.raw_licks = data.raw_licks(data_idx); fold.z_licks = data.z_licks(data_idx);
    fold.raw_vel   = data.raw_vel(data_idx);   fold.z_vel   = data.z_vel(data_idx);
    
    if isempty(fold.s), NLL = 0; return; end
    condMat = [fold.s, fold.c, fold.d];
    [G, ~, G_idx] = unique(condMat, 'rows');
    n_conds = size(G, 1);
    
    s_rad = deg2rad(params.s_range_deg); m_rad = deg2rad(params.m_range_deg);
    prior = get_prior(s_rad, params, data); 
    util  = get_utility_vectors(params.s_range_deg, utility);
    
    total_NLL = 0;
    for j = 1:n_conds
        s_val = G(j,1); c_val = G(j,2); d_val = G(j,3);
        k_gen = get_kappa_for_trial(s_val, c_val, d_val, params);
        
        Lik_inf = pdfVonMises(m_rad', s_rad, k_gen);
        Post_s_m = (Lik_inf .* prior) ./ (sum(Lik_inf .* prior, 2) + eps);
        dv = (Post_s_m * util.respond') - (Post_s_m * util.no_respond');
        
        p_m_s = pdfVonMises(m_rad, deg2rad(s_val), k_gen);
        p_m_s = p_m_s ./ (sum(p_m_s) + eps);
        
        mask = (G_idx == j); n_curr = sum(mask);
        L_joint_m = ones(n_curr, numel(m_rad));
        
        if config.fit_choice
            p_go_m = 1 ./ (1 + exp(-params.decision_beta * dv));
            P_Go_Mat = repmat(p_go_m', n_curr, 1);
            obs_ch = fold.ch(mask);
            L_choice = (P_Go_Mat .* obs_ch) + ((1 - P_Go_Mat) .* (1 - obs_ch));
            L_joint_m = L_joint_m .* L_choice;
        end
        
        if config.fit_licks
            if strcmp(config.lick_dist, 'Normal')
                mu_licks = params.lick_slope * dv + params.lick_intercept;
                L_licks = normpdf(repmat(fold.z_licks(mask),1,numel(m_rad)), repmat(mu_licks', n_curr, 1), params.lick_std);
            elseif strcmp(config.lick_dist, 'Poisson')
                lambda_licks = exp(params.lick_slope * dv + params.lick_intercept);
                L_licks = poisspdf(repmat(fold.raw_licks(mask),1,numel(m_rad)), repmat(lambda_licks', n_curr, 1));
            end
            L_joint_m = L_joint_m .* L_licks;
        end
        
        if config.fit_vel
            if strcmp(config.vel_dist, 'Normal')
                mu_vel = params.vel_slope * dv + params.vel_intercept;
                L_vel = normpdf(repmat(fold.z_vel(mask),1,numel(m_rad)), repmat(mu_vel', n_curr, 1), params.vel_std);
            end
            L_joint_m = L_joint_m .* L_vel;
        end
        
        L_trial = sum(L_joint_m .* repmat(p_m_s, n_curr, 1), 2);
        total_NLL = total_NLL - sum(log(L_trial + eps));
    end
    NLL = total_NLL; if isnan(NLL) || isinf(NLL), NLL = 1e10; end
end

function model_preds = calculate_model_predictions(trial_conditions_mat, params, utility, current_data, config)
    s_range_rad = deg2rad(params.s_range_deg); m_range_rad = deg2rad(params.m_range_deg);
    prior = get_prior(s_range_rad, params, current_data);
    utility_vec = get_utility_vectors(params.s_range_deg, utility);
    n_conds = size(trial_conditions_mat, 1);
    
    p_respond_all_conds = zeros(n_conds, 1); binary_choice_all_conds = zeros(n_conds, 1);
    licks_pred_all = zeros(n_conds, 1); vel_pred_all = zeros(n_conds, 1);
    
    for i_cond = 1:n_conds
        s_i = trial_conditions_mat(i_cond, 1); c_i = trial_conditions_mat(i_cond, 2); d_i = trial_conditions_mat(i_cond, 3);
        kappa_gen_i = get_kappa_for_trial(s_i, c_i, d_i, params);
        Pm_given_s_i = pdfVonMises(m_range_rad, deg2rad(s_i), kappa_gen_i);
        Pm_given_s_i = Pm_given_s_i ./ (sum(Pm_given_s_i) + eps); 
        
        posterior_i = (pdfVonMises(m_range_rad', s_range_rad, kappa_gen_i) .* prior);
        posterior_i = posterior_i ./ (sum(posterior_i, 2) + eps);
        dv = (posterior_i * utility_vec.respond') - (posterior_i * utility_vec.no_respond');
        
        p_soft_vs_m = 1 ./ (1 + exp(-params.decision_beta * dv));
        p_respond_all_conds(i_cond) = sum(p_soft_vs_m' .* Pm_given_s_i);
        binary_choice_all_conds(i_cond) = sum(double(dv > 0)' .* Pm_given_s_i);
        
        if config.fit_licks
            if strcmp(config.lick_dist, 'Normal')
                exp_licks = params.lick_slope * dv + params.lick_intercept;
            elseif strcmp(config.lick_dist, 'Poisson')
                exp_licks = exp(params.lick_slope * dv + params.lick_intercept);
            end
            licks_pred_all(i_cond) = sum(exp_licks' .* Pm_given_s_i); 
        end
        if config.fit_vel
            if strcmp(config.vel_dist, 'Normal')
                exp_vel = params.vel_slope * dv + params.vel_intercept;
                vel_pred_all(i_cond) = sum(exp_vel' .* Pm_given_s_i); 
            end
        end
    end
    
    model_preds.p_respond = p_respond_all_conds; 
    model_preds.binary_choice = binary_choice_all_conds; 
    model_preds.lick_prediction = licks_pred_all;
    model_preds.vel_prediction  = vel_pred_all;
end

function inferred_unc = invert_model_for_single_trial_uncertainty(data, params, utility, base_spec)
    config = base_spec.config;
    m_range_rad = deg2rad(base_spec.fixed_params.m_range_deg);
    s_range_deg = base_spec.fixed_params.s_range_deg;
    s_range_rad = deg2rad(s_range_deg);
    n_m = numel(m_range_rad); n_s = numel(s_range_rad);
    prior_ps = get_prior(s_range_rad, params, data);
    utility_vec = get_utility_vectors(s_range_deg, utility);
    
    cond_mat = [data.orientation, data.contrast, data.dispersion];
    [G_unique_conds, ~, trial_to_cond_idx] = unique(cond_mat, 'rows');
    n_unique_conds = size(G_unique_conds, 1);
    
    maps_perc_unc = cell(n_unique_conds, 1); maps_dec_unc = cell(n_unique_conds, 1);
    maps_lik_behavior = cell(n_unique_conds, 1); maps_p_s_given_m = cell(n_unique_conds, 1);
    maps_L_s_given_m = cell(n_unique_conds, 1);
    generative_kappas = zeros(n_unique_conds, 1);
    
    is_go_stim = s_range_deg < 45; is_nogo_stim = s_range_deg > 45; is_boundary = s_range_deg == 45;
    
    for j = 1:n_unique_conds
        s_j = G_unique_conds(j, 1); c_j = G_unique_conds(j, 2); d_j = G_unique_conds(j, 3);
        kappa_gen = get_kappa_for_trial(s_j, c_j, d_j, params);
        generative_kappas(j) = kappa_gen;
        
        lik_mat = pdfVonMises(m_range_rad', s_range_rad, kappa_gen);
        maps_L_s_given_m{j} = lik_mat;
        
        post_s_m = (lik_mat .* prior_ps);
        post_s_m = post_s_m ./ (sum(post_s_m, 2) + eps);
        maps_p_s_given_m{j} = post_s_m;
        
        mean_s_map = sum(post_s_m .* s_range_deg, 2);
        maps_perc_unc{j} = sqrt(sum(post_s_m .* (s_range_deg - mean_s_map).^2, 2));
        
        p_stim_go = sum(post_s_m(:, is_go_stim), 2) + 0.5 * sum(post_s_m(:, is_boundary), 2);
        maps_dec_unc{j} = -(p_stim_go .* log2(p_stim_go + eps) + (1-p_stim_go) .* log2((1-p_stim_go) + eps));
        
        dv = (post_s_m * utility_vec.respond') - (post_s_m * utility_vec.no_respond'); 
        
        if config.fit_choice
            maps_lik_behavior{j}.p_go = 1 ./ (1 + exp(-params.decision_beta * dv)); 
        end
        if config.fit_licks
            if strcmp(config.lick_dist, 'Normal')
                maps_lik_behavior{j}.pred_lick = params.lick_slope * dv + params.lick_intercept;
            elseif strcmp(config.lick_dist, 'Poisson')
                maps_lik_behavior{j}.pred_lick = exp(params.lick_slope * dv + params.lick_intercept);
            end
        end
        if config.fit_vel
            if strcmp(config.vel_dist, 'Normal')
                maps_lik_behavior{j}.pred_vel  = params.vel_slope * dv + params.vel_intercept; 
            end
        end
    end
    
    inferred_unc.perc_unc_marginal = nan(data.n_trials, 1); inferred_unc.dec_unc_marginal  = nan(data.n_trials, 1);
    inferred_unc.eu_go_marginal    = nan(data.n_trials, 1); inferred_unc.eu_nogo_marginal  = nan(data.n_trials, 1);
    inferred_unc.post_s_marginal   = nan(data.n_trials, n_s); inferred_unc.m_posteriors = nan(data.n_trials, n_m);
    
    inferred_unc.perceptual_map = nan(data.n_trials, 1); inferred_unc.decision_map = nan(data.n_trials, 1);
    inferred_unc.post_s_given_map = nan(data.n_trials, n_s); inferred_unc.L_s_given_map = nan(data.n_trials, n_s);

    inferred_unc.lik_s_marginal    = nan(data.n_trials, n_s); % [N x 91] Likelihood
    
    for i = 1:data.n_trials
        cond_idx = trial_to_cond_idx(i);
        prior_m = pdfVonMises(m_range_rad, deg2rad(data.orientation(i)), generative_kappas(cond_idx));
        prior_m = prior_m / (sum(prior_m) + eps);
        
        beh_maps = maps_lik_behavior{cond_idx};
        L_joint = ones(1, n_m);
        
        if config.fit_choice
            ch = data.choices(i);
            L_joint = L_joint .* ((beh_maps.p_go .* ch) + ((1 - beh_maps.p_go) .* (1 - ch)))';
        end
        if config.fit_licks
            if strcmp(config.lick_dist, 'Normal')
                L_joint = L_joint .* normpdf(data.z_licks(i), beh_maps.pred_lick, params.lick_std)';
            elseif strcmp(config.lick_dist, 'Poisson')
                L_joint = L_joint .* poisspdf(data.raw_licks(i), beh_maps.pred_lick)';
            end
        end
        if config.fit_vel
            if strcmp(config.vel_dist, 'Normal')
                L_joint = L_joint .* normpdf(data.z_vel(i), beh_maps.pred_vel, params.vel_std)';
            end
        end
        
        post_m_unnorm = L_joint .* prior_m;
        sum_post = sum(post_m_unnorm);
        
        if sum_post == 0 || isnan(sum_post)
            post_m = prior_m; 
        else
            post_m = post_m_unnorm / sum_post; 
        end
        
        inferred_unc.m_posteriors(i, :) = post_m;
        
        Q_marg = post_m * maps_p_s_given_m{cond_idx};
        Q_marg = Q_marg ./ (sum(Q_marg) + eps);
        inferred_unc.post_s_marginal(i, :) = Q_marg;

        % --- NEW: Marginalized Likelihood ---
        lik_s_m_i = maps_L_s_given_m{cond_idx};  % [n_m x n_s] pure likelihoods
        L_marg = post_m * lik_s_m_i;             % Integrate over m weighted by P(m|behavior)
        L_marg = L_marg ./ (sum(L_marg) + eps);  % Normalize over s_grid to form a proper density

        inferred_unc.lik_s_marginal(i, :) = L_marg;
        
        inferred_unc.perc_unc_marginal(i) = sqrt(sum(Q_marg .* (s_range_deg - sum(Q_marg .* s_range_deg)).^2));
        p_go_marg  = max(eps, min(1-eps, sum(Q_marg(is_go_stim)) + 0.5 * sum(Q_marg(is_boundary))));
        inferred_unc.dec_unc_marginal(i) = -(p_go_marg * log2(p_go_marg) + (1-p_go_marg) * log2((1-p_go_marg)));
        
        inferred_unc.eu_go_marginal(i)   = Q_marg * utility_vec.respond';
        inferred_unc.eu_nogo_marginal(i) = Q_marg * utility_vec.no_respond';
        
        [~, map_idx] = max(post_m);
        inferred_unc.perceptual_map(i) = maps_perc_unc{cond_idx}(map_idx);
        inferred_unc.decision_map(i)   = maps_dec_unc{cond_idx}(map_idx);
        inferred_unc.L_s_given_map(i,:) = maps_L_s_given_m{cond_idx}(map_idx,:);
        inferred_unc.post_s_given_map(i,:) = maps_p_s_given_m{cond_idx}(map_idx,:);
    end
end

function kappa = get_kappa_for_trial(orientation, contrast, dispersion, params)
kappa = (params.kappa_min + params.kappa_amp) .* (contrast.^params.c_power) .* exp(-params.d_power .* dispersion);
end

function prior = get_prior(s_range_rad, params, data)
switch params.prior_type
    case 'Flat', prior = ones(1, length(s_range_rad));
    case 'Bimodal'
        prior = pdfVonMises(s_range_rad, 0, params.prior_strength) + pdfVonMises(s_range_rad, pi/2, params.prior_strength);
end
prior = prior / (sum(prior) + eps);
end

function utility_vec = get_utility_vectors(s_range, utility)
n = length(s_range); utility_vec.respond = zeros(1, n); utility_vec.no_respond = zeros(1, n);
utility_vec.respond(s_range < 45) = utility.R_hit; utility_vec.respond(s_range > 45) = utility.R_fa; utility_vec.respond(s_range == 45) = 0.5*utility.R_hit + 0.5*utility.R_fa;
utility_vec.no_respond(s_range < 45) = utility.R_miss; utility_vec.no_respond(s_range > 45) = utility.R_cr;
end

function p = pdfVonMises(x, mu, kappa), p = exp(kappa .* cos(bsxfun(@minus, 2*x, 2*mu))) ./ (2 * pi * besseli(0, kappa)); end

function xz = zscore_safe(x), sd = std(x,[],'omitnan'); if ~isfinite(sd) || sd==0, xz = zeros(size(x)); else, xz = (x - mean(x,'omitnan'))./sd; end; end

function data_out = align_data_to_go_by_side(data_in, go_side)
data_out = data_in;
if any(strcmp(lower(strtrim(go_side)), {'vertical','v','>45','right'})), data_out.orientation = max(0, min(90, 90 - data_in.orientation)); end
end