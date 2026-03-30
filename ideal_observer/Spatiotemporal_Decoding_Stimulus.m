%% Spatiotemporal Hierarchical Fitting & Model Comparison (Stimulus Category)
%
% Re-runs the full hierarchical fitting pipeline independently for every 
% spatial window.
%
% Models evaluated on identical 5-fold cross-validation splits predicting 
% STIMULUS CATEGORY (Go / No-Go) via AUC and Binomial NLL.
%
% 1. Hierarchical IO (Full): Licks + Vel
% 2. Hierarchical IO (Reduced): Vel Only
% 3. GLM Kinematic: Vel + Licks
% 4. GLM Kinematic: Vel Only
% 5. GLM Kinematic: Licks Only

clear; close all; clc;
rng("twister");
warning off;

fprintf('--- Starting Unified Spatiotemporal Stimulus Decoding Pipeline ---\n');

% Add BADS path if necessary
% addpath(genpath('/Users/theoamvr/Desktop/Experiments/bads-master'));

%% --- Part 1: Configuration ---
sesnames1 = {'20250605_Cb15', '20250613_Cb15', '20250620_Cb15', '20250624_Cb15', '20250709_Cb15'};
sesnames2 = {'20250606_Cb17', '20250613_Cb17', '20250620_Cb17', '20250624_Cb17', '20250701_Cb17'};
sesnames3 = {'20250904_Cb21', '20250910_Cb21', '20250911_Cb21', '20250912_Cb21', '20250918_Cb21'};
sesnames4 = {'20251024_Cb22', '20251027_Cb22', '20251028_Cb22', '20251030_Cb22', '20251105_Cb22'};
sesnames5 = {'20250918_Cb24', '20250919_Cb24', '20251020_Cb24', '20251021_Cb24'};
sesnames6 = {'20250903_Cb25', '20250904_Cb25', '20250910_Cb25', '20250911_Cb25', '20250916_Cb25'};

all_sesnames = {sesnames1, sesnames2, sesnames3, sesnames4, sesnames5, sesnames6}; 

% Specify 'horizontal' (go <45) or 'vertical' (go >45) per animal
go_side_by_animal = {'horizontal', 'horizontal', 'vertical', 'vertical', 'vertical', 'horizontal'};

W = 10; % Window width
S = 10; % Step size
corridor_start = 0; 
k_folds = 5;
n_jitter_starts = 5; % 1 group-seed + 2 jittered seeds

% IO Model Specifications
fixed_utility = struct('R_hit', 1, 'R_miss', 0, 'R_cr', 0.1, 'R_fa', -0.2);
base_params = struct('s_range_deg', 0:1:90, 'm_range_deg', 0:1:180, ...
    'prior_type', 'Bimodal', 'prior_strength', 3, 'kappa_min', 1.0, ...
    'rho_k', 0, 'phi_pref', 0, 'decision_beta', 1.0);

spec_full.fit_params = {'kappa_amp', 'c_power', 'd_power', 'lick_slope', 'lick_intercept', 'lick_std', 'vel_slope', 'vel_intercept', 'vel_std'};
spec_full.fixed_params = base_params;

spec_vel.fit_params = {'kappa_amp', 'c_power', 'd_power', 'vel_slope', 'vel_intercept', 'vel_std'};
spec_vel.fixed_params = base_params;

%% --- Part 2: Data Extraction & Windowing ---
fprintf('Extracting raw data, kinematics, and trial history...\n');
all_animal_data = cell(numel(all_sesnames), 1);
spatial_centers = [];

for i_animal = 1:numel(all_sesnames)
    sessions = all_sesnames{i_animal};
    animal_struct = struct('licks', [], 'vel', [], 'choices', [], 'stim', [], 'history', []);
    rz_ref = NaN;
    
    go_side = lower(strtrim(go_side_by_animal{i_animal}));
    flip_needed = any(strcmp(go_side, {'vertical','v','>45','right'}));
    
    for i_ses = 1:numel(sessions)
        try
            loaded_data = load(['vr_' sessions{i_ses} '_light.mat'], 'vr');
            vr = loaded_data.vr;
            if isfield(vr.cfg, 'sessionType') && strcmpi(vr.cfg.sessionType, 'basic'), continue; end
            
            n_trials = numel(vr.trialLog);
            rz_start = vr.cfg.rewardZoneStart;
            if isnan(rz_ref), rz_ref = rz_start; end
            
            win_starts = corridor_start : S : (rz_start - W);
            win_ends = win_starts + W;
            n_win = length(win_starts);
            if isempty(spatial_centers), spatial_centers = win_starts + (W/2); end
            
            ses_licks = nan(n_trials, n_win); ses_vel = nan(n_trials, n_win);
            ses_choices = nan(n_trials, 1); ses_stim = nan(n_trials, 3);
            ses_hist = nan(n_trials, 2); 
            
            for t = 1:n_trials
                pos = vr.trialLog{t}.position(2,:);
                lick = vr.trialLog{t}.lick;
                vel = -vr.trialLog{t}.velocity(2,:) * 0.2537;
                
                ses_choices(t) = sum(lick(pos >= rz_start & pos <= vr.cfg.rewardZoneEnd)) > 0;
                
                % Handle stimulus alignment (Maps everything so Go < 45)
                ori = vr.cfg.trialOrientations(t);
                if flip_needed, ori = max(0, min(90, 90 - ori)); end
                ses_stim(t,:) = [ori, vr.cfg.trialContrasts(t), vr.cfg.trialDispersions(t)];
                
                % Extract Trial History (t-1)
                if t == 1
                    ses_hist(t,:) = [NaN, NaN];
                else
                    prev_choice = ses_choices(t-1);
                    prev_ori = ses_stim(t-1, 1);
                    if prev_ori < 45
                        prev_rewarded = prev_choice == 1; % Hit
                    else
                        prev_rewarded = prev_choice == 0; % CR
                    end
                    ses_hist(t,:) = [prev_choice, double(prev_rewarded)];
                end
                
                for w = 1:n_win
                    in_win = pos >= win_starts(w) & pos < win_ends(w);
                    if any(in_win)
                        ses_licks(t,w) = sum(lick(in_win));
                        ses_vel(t,w) = mean(vel(in_win), 'omitnan');
                    else
                        ses_licks(t,w) = 0;
                    end
                end
            end
            animal_struct.licks = [animal_struct.licks; ses_licks];
            animal_struct.vel = [animal_struct.vel; ses_vel];
            animal_struct.choices = [animal_struct.choices; ses_choices];
            animal_struct.stim = [animal_struct.stim; ses_stim];
            animal_struct.history = [animal_struct.history; ses_hist];
        catch
        end
    end
    
    % Z-score within animal per window
    for w = 1:n_win
        animal_struct.licks(:,w) = zscore_safe(animal_struct.licks(:,w));
        animal_struct.vel(:,w) = zscore_safe(animal_struct.vel(:,w));
    end
    all_animal_data{i_animal} = animal_struct;
end

%% --- Part 3: Window-by-Window Fitting Pipeline ---
n_windows = length(spatial_centers);
n_animals = numel(all_animal_data);
WindowResults = struct('center', num2cell(spatial_centers));

bads_opts = bads('defaults');
bads_opts.Display = 'off';

for w = 1:n_windows
    fprintf('\n========================================\n');
    fprintf('Processing Spatial Window %d / %d (Center: %.1f)\n', w, n_windows, spatial_centers(w));
    fprintf('========================================\n');
    
    % 1. Pool data across animals for the Group-Level IO fits
    pool_full = struct('s',[], 'c',[], 'd',[], 'ch',[], 'licks',[], 'vel',[], 'history', []);
    for i_animal = 1:n_animals
        d = all_animal_data{i_animal};
        valid = ~isnan(d.licks(:,w)) & ~isnan(d.vel(:,w)) & ~any(isnan(d.stim),2) & ~isnan(d.choices) & ~any(isnan(d.history),2);
        pool_full.s = [pool_full.s; d.stim(valid,1)];
        pool_full.c = [pool_full.c; d.stim(valid,2)];
        pool_full.d = [pool_full.d; d.stim(valid,3)];
        pool_full.ch = [pool_full.ch; d.choices(valid)];
        pool_full.licks = [pool_full.licks; d.licks(valid,w)];
        pool_full.vel = [pool_full.vel; d.vel(valid,w)];
    end
    
    % 2. Group-Level BADS fits (Still fitting to kinematics!)
    fprintf(' Running Group-Level Fits...\n');
    [lb_f, ub_f, plb_f, pub_f, p0_f] = get_bounds(spec_full.fit_params);
    obj_group_full = @(p) calc_IO_NLL(p, pool_full, spec_full, fixed_utility, 'both');
    group_p_full = bads(obj_group_full, p0_f, lb_f, ub_f, plb_f, pub_f, [], bads_opts);
    
    [lb_v, ub_v, plb_v, pub_v, p0_v] = get_bounds(spec_vel.fit_params);
    obj_group_vel = @(p) calc_IO_NLL(p, pool_full, spec_vel, fixed_utility, 'vel_only');
    group_p_vel = bads(obj_group_vel, p0_v, lb_v, ub_v, plb_v, pub_v, [], bads_opts);
    
    % 3. Individual-Level Cross-Validation
    fprintf(' Running Individual CV Fits...\n');
    animal_res = cell(n_animals, 1);
    
    parfor i_animal = 1:n_animals
        d = all_animal_data{i_animal};
        valid = ~isnan(d.licks(:,w)) & ~isnan(d.vel(:,w)) & ~any(isnan(d.stim),2) & ~isnan(d.choices) & ~any(isnan(d.history),2);
        
        if sum(valid) < 30 || length(unique(d.stim(valid,1) < 45)) < 2
            animal_res{i_animal} = struct('status', 'insufficient_data');
            continue;
        end
        
        dat_ind = struct('s', d.stim(valid,1), 'c', d.stim(valid,2), 'd', d.stim(valid,3), ...
                         'ch', d.choices(valid), 'licks', d.licks(valid,w), 'vel', d.vel(valid,w), ...
                         'history', d.history(valid,:));
        
        cv = cvpartition(length(dat_ind.ch), 'KFold', k_folds);
        
        % Updated Results Struct (5 Models)
        res = struct('IO_Full_NLL', 0, 'IO_Vel_NLL', 0, ...
            'GLM_Kinematic_LV_NLL', 0, 'GLM_Kinematic_V_NLL', 0, 'GLM_Kinematic_L_NLL', 0);
        
        preds_IO_Full = nan(length(dat_ind.ch), 1); 
        preds_IO_Vel  = nan(length(dat_ind.ch), 1);
        preds_LV      = nan(length(dat_ind.ch), 1); 
        preds_V       = nan(length(dat_ind.ch), 1);
        preds_L       = nan(length(dat_ind.ch), 1); 
        
        for k = 1:cv.NumTestSets
            tr = cv.training(k); te = cv.test(k);
            
            dat_tr = subset_data(dat_ind, tr);
            dat_te = subset_data(dat_ind, te);
            
            % Target for GLMs: Is it a Go Stimulus? (< 45 deg)
            y_tr_stim = dat_tr.s < 45;
            y_te_stim = dat_te.s < 45;
            
            % --- Fit IO Full (Multi-Start to kinematics) ---
            obj_tr_full = @(p) calc_IO_NLL(p, dat_tr, spec_full, fixed_utility, 'both');
            p_fit_full = fit_with_multi_start(obj_tr_full, group_p_full, lb_f, ub_f, plb_f, pub_f, bads_opts, n_jitter_starts);
            % INFER STIMULUS PROBABILITY
            preds_IO_Full(te) = get_IO_stim_preds(p_fit_full, dat_te, spec_full, fixed_utility, 'both');
            
            % --- Fit IO Vel-Only (Multi-Start to kinematics) ---
            obj_tr_vel = @(p) calc_IO_NLL(p, dat_tr, spec_vel, fixed_utility, 'vel_only');
            p_fit_vel = fit_with_multi_start(obj_tr_vel, group_p_vel, lb_v, ub_v, plb_v, pub_v, bads_opts, n_jitter_starts);
            % INFER STIMULUS PROBABILITY
            preds_IO_Vel(te) = get_IO_stim_preds(p_fit_vel, dat_te, spec_vel, fixed_utility, 'vel_only');
            
            % --- Fit GLMs to STIMULUS CATEGORY ---
            X_LV_tr = [dat_tr.vel, dat_tr.licks]; X_LV_te = [dat_te.vel, dat_te.licks];
            X_V_tr = dat_tr.vel; X_V_te = dat_te.vel;
            X_L_tr = dat_tr.licks; X_L_te = dat_te.licks;
            
            mdl_LV = fitglm(X_LV_tr, y_tr_stim, 'Distribution', 'binomial', 'LikelihoodPenalty', 'jeffreys-prior');
            preds_LV(te) = predict(mdl_LV, X_LV_te);
            
            mdl_V = fitglm(X_V_tr, y_tr_stim, 'Distribution', 'binomial', 'LikelihoodPenalty', 'jeffreys-prior');
            preds_V(te) = predict(mdl_V, X_V_te);
            
            mdl_L = fitglm(X_L_tr, y_tr_stim, 'Distribution', 'binomial', 'LikelihoodPenalty', 'jeffreys-prior');
            preds_L(te) = predict(mdl_L, X_L_te);
            
            % --- Binomial NLL Accumulation (Against Stimulus) ---
            calc_glm_nll = @(p, y) -sum(y .* log(p + eps) + (1 - y) .* log(1 - p + eps));
            
            res.IO_Full_NLL          = res.IO_Full_NLL          + calc_glm_nll(preds_IO_Full(te), y_te_stim);
            res.IO_Vel_NLL           = res.IO_Vel_NLL           + calc_glm_nll(preds_IO_Vel(te), y_te_stim);
            res.GLM_Kinematic_LV_NLL = res.GLM_Kinematic_LV_NLL + calc_glm_nll(preds_LV(te), y_te_stim);
            res.GLM_Kinematic_V_NLL  = res.GLM_Kinematic_V_NLL  + calc_glm_nll(preds_V(te), y_te_stim);
            res.GLM_Kinematic_L_NLL  = res.GLM_Kinematic_L_NLL  + calc_glm_nll(preds_L(te), y_te_stim);
        end
        
        y_all_stim = dat_ind.s < 45;
        safe_auc = @(y, p) get_auc(y, p);
        
        res.IO_Full_AUC          = safe_auc(y_all_stim, preds_IO_Full);
        res.IO_Vel_AUC           = safe_auc(y_all_stim, preds_IO_Vel);
        res.GLM_Kinematic_LV_AUC = safe_auc(y_all_stim, preds_LV);
        res.GLM_Kinematic_V_AUC  = safe_auc(y_all_stim, preds_V);
        res.GLM_Kinematic_L_AUC  = safe_auc(y_all_stim, preds_L);
        
        animal_res{i_animal} = res;
    end
    
    WindowResults(w).animal_data = animal_res;
end

%% --- Part 4: Save Results ---
% Save Stimulus Results
save('Spatiotemporal_Decoding_Stimulus.mat', 'WindowResults', 'spatial_centers', 'n_windows', 'n_animals', '-v7.3');
fprintf('Analysis complete and saved to Spatiotemporal_Decoding_Stimulus.mat\n');

%% --- Part 5: Plotting Model Comparison Across Space ---
fprintf('Generating Unified Spatiotemporal Model Comparison plots...\n');
figure('Name', 'Stimulus Decoding: AUC across Spatial Windows', 'Color', 'w', 'Position', [100, 100, 1050, 700]);
hold on;

plot_centers = [WindowResults.center];
metrics = {'IO_Full_AUC', 'IO_Vel_AUC', 'GLM_Kinematic_LV_AUC', 'GLM_Kinematic_V_AUC', 'GLM_Kinematic_L_AUC'};
       
res_matrix = nan(n_animals, n_windows, length(metrics));

% Extract metrics
for w = 1:n_windows
    for a = 1:n_animals
        res = WindowResults(w).animal_data{a};
        if isfield(res, 'status') && strcmp(res.status, 'insufficient_data')
            continue;
        end
        for m = 1:length(metrics)
            if isfield(res, metrics{m}), res_matrix(a, w, m) = res.(metrics{m}); end
        end
    end
end

avg_res = squeeze(mean(res_matrix, 1, 'omitnan'));

% Color Palette
col_io_full = [0, 0.4470, 0.7410];       
col_io_red  = [0.3010, 0.7450, 0.9330];  
col_kin     = [0.4660, 0.6740, 0.1880]; 

% 1. Kinematic GLMs
plot(plot_centers, avg_res(:,5), ':', 'Color', col_kin, 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'DisplayName', 'GLM Kinematic (Licks)');
plot(plot_centers, avg_res(:,4), '--', 'Color', col_kin, 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'DisplayName', 'GLM Kinematic (Vel)');
plot(plot_centers, avg_res(:,3), '-', 'Color', col_kin, 'LineWidth', 2.5, 'Marker', 's', 'MarkerSize', 7, 'MarkerFaceColor', col_kin, 'DisplayName', 'GLM Kinematic (Vel+Licks)');

% 2. Ideal Observers 
plot(plot_centers, avg_res(:,2), '-', 'Color', col_io_red, 'LineWidth', 2.5, 'Marker', 'o', 'MarkerSize', 7, 'MarkerFaceColor', 'w', 'DisplayName', 'IO Reduced (Vel)');
plot(plot_centers, avg_res(:,1), '-', 'Color', col_io_full, 'LineWidth', 3, 'Marker', 'o', 'MarkerSize', 8, 'MarkerFaceColor', col_io_full, 'DisplayName', 'IO Full (Licks+Vel)');

% Formatting & Aesthetics
xlabel('Spatial Window Center (VR units)', 'FontWeight', 'bold');
ylabel('Average Cross-Validated AUC', 'FontWeight', 'bold');
title('Decoding True Stimulus Category over Space', 'FontSize', 14, 'FontWeight', 'bold');

lgd = legend('Location', 'eastoutside', 'NumColumns', 1);
title(lgd, 'Evaluated Models');
lgd.Box = 'off';

set(gca, 'FontSize', 12, 'TickDir', 'out', 'Box', 'off', 'LineWidth', 1.5);
xlim([min(plot_centers)-5, max(plot_centers)+5]);
ylim([0.45, 1.0]); 
yline(0.5, 'k-', 'Chance', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left', 'HandleVisibility', 'off');
grid on; ax = gca; ax.GridAlpha = 0.15;

%% --- Core Functions ---
function best_p = fit_with_multi_start(obj_fun, p_base, lb, ub, plb, pub, opts, n_starts)
    best_nll = inf; best_p = p_base;
    for i = 1:n_starts
        if i == 1, p_start = p_base;
        else, p_start = max(lb, min(ub, p_base + (rand(size(p_base))-0.5).*(pub-plb)*0.1)); end
        try
            [p_fit, nll_fit] = bads(obj_fun, p_start, lb, ub, plb, pub, [], opts);
            if nll_fit < best_nll, best_nll = nll_fit; best_p = p_fit; end
        catch, end
    end
end

function [lb, ub, plb, pub, p0] = get_bounds(param_names)
    n = length(param_names); lb = zeros(1,n); ub = zeros(1,n); p0 = zeros(1,n);
    for i = 1:n
        name = param_names{i};
        if strcmp(name, 'kappa_amp'), lb(i)=0; ub(i)=50; p0(i)=10;
        elseif contains(name, 'power'), lb(i)=0; ub(i)=5; p0(i)=1;
        elseif contains(name, 'lick_slope'), lb(i)=0; ub(i)=10; p0(i)=2; 
        elseif contains(name, 'vel_slope'), lb(i)=-10; ub(i)=0; p0(i)=-2; 
        elseif contains(name, '_intercept'), lb(i)=-5; ub(i)=5; p0(i)=0;
        elseif contains(name, '_std'), lb(i)=0.01; ub(i)=5; p0(i)=0.5;
        end
    end
    plb = max(lb, p0 - abs(p0)*0.5 - 0.1); pub = min(ub, p0 + abs(p0)*0.5 + 0.1);
end

function NLL = calc_IO_NLL(p_vec, data, spec, utility, fit_mode)
    % Keeps fitting to kinematics perfectly intact
    params = spec.fixed_params;
    for i = 1:length(p_vec), params.(spec.fit_params{i}) = p_vec(i); end
    s_grid = deg2rad(params.s_range_deg); m_grid = deg2rad(params.m_range_deg);
    prior = get_prior(s_grid, params);
    util = get_utility_vectors(params.s_range_deg, utility);
    conds = [data.s, data.c, data.d];
    [u_conds, ~, idx] = unique(conds, 'rows');
    total_nll = 0;
    for j = 1:size(u_conds, 1)
        mask = (idx == j); if ~any(mask), continue; end
        s = u_conds(j,1); c = u_conds(j,2); d = u_conds(j,3);
        k_gen = (params.kappa_min + params.kappa_amp) * (c^params.c_power) * exp(-params.d_power * d);
        lik = pdfVonMises(m_grid', s_grid, k_gen);
        post = (lik .* prior) ./ (sum(lik .* prior, 2) + eps);
        dv = (post * util.respond') - (post * util.no_respond');
        p_m_s = pdfVonMises(m_grid, deg2rad(s), k_gen);
        p_m_s = p_m_s ./ (sum(p_m_s) + eps);
        n_t = sum(mask);
        L_v = normpdf(repmat(data.vel(mask), 1, length(m_grid)), repmat((params.vel_slope*dv + params.vel_intercept)', n_t, 1), params.vel_std);
        if strcmp(fit_mode, 'both')
            L_l = normpdf(repmat(data.licks(mask), 1, length(m_grid)), repmat((params.lick_slope*dv + params.lick_intercept)', n_t, 1), params.lick_std);
            L_trial = sum(L_l .* L_v .* repmat(p_m_s, n_t, 1), 2);
        else 
            L_trial = sum(L_v .* repmat(p_m_s, n_t, 1), 2);
        end
        total_nll = total_nll - sum(log(L_trial + eps));
    end
    NLL = total_nll; if isnan(NLL) || isinf(NLL), NLL = 1e10; end
end

function p_stim_inferred = get_IO_stim_preds(p_vec, data, spec, utility, fit_mode)
    % INVERTS the IO model to infer STIMULUS CATEGORY probability
    % FAIR VERSION: Blind to true_s. Infers 'm' strictly from kinematics.
    
    params = spec.fixed_params;
    for i = 1:length(p_vec), params.(spec.fit_params{i}) = p_vec(i); end
    
    s_grid = deg2rad(params.s_range_deg); 
    m_grid = deg2rad(params.m_range_deg);
    prior = get_prior(s_grid, params);
    util = get_utility_vectors(params.s_range_deg, utility);
    
    n_trials = length(data.ch);
    p_stim_inferred = nan(n_trials, 1);
    
    for j = 1:n_trials
        c = data.c(j); d = data.d(j);
        obs_lick = data.licks(j); obs_vel = data.vel(j);
        
        % Precision is still dictated by contrast/dispersion
        k_gen = (params.kappa_min + params.kappa_amp) * (c^params.c_power) * exp(-params.d_power * d);
        
        % 1. Calculate MARGINAL prior over m: P(m) = \int p(m|s)p(s)ds
        % (This replaces p(m|s_true), removing the unfair advantage)
        p_m_marginal = zeros(1, length(m_grid));
        for i_s = 1:length(s_grid)
            p_m_s_temp = pdfVonMises(m_grid, s_grid(i_s), k_gen);
            p_m_marginal = p_m_marginal + p_m_s_temp * prior(i_s);
        end
        p_m_marginal = p_m_marginal ./ (sum(p_m_marginal) + eps);
        
        % 2. Internal inference: p(s|m) 
        lik_inf = pdfVonMises(m_grid', s_grid, k_gen);
        post_s_given_m = (lik_inf .* prior) ./ (sum(lik_inf .* prior, 2) + eps);
        
        % 3. Calculate the perceptual probability that stimulus < 45 for every 'm'
        p_stim_go_m = sum(post_s_given_m(:, params.s_range_deg < 45), 2) + ...
                      0.5 * sum(post_s_given_m(:, params.s_range_deg == 45), 2);
                      
        % Calculate DV strictly for behavior mapping likelihood
        dv = (post_s_given_m * util.respond') - (post_s_given_m * util.no_respond');
        
        % 4. Invert model using ONLY observed kinematics and marginal prior
        L_joint = ones(1, length(m_grid));
        if isfield(params, 'vel_slope')
            L_v = normpdf(obs_vel, params.vel_slope * dv + params.vel_intercept, params.vel_std);
            L_joint = L_joint .* L_v';
        end
        if strcmp(fit_mode, 'both') && isfield(params, 'lick_slope')
            L_l = normpdf(obs_lick, params.lick_slope * dv + params.lick_intercept, params.lick_std);
            L_joint = L_joint .* L_l';
        end
        
        post_m_unnorm = L_joint .* p_m_marginal;
        post_m = post_m_unnorm ./ (sum(post_m_unnorm) + eps);
        
        % 5. Marginalize Perceptual P(Go) over the inverted posterior of 'm'
        p_stim_inferred(j) = sum(p_stim_go_m .* post_m');
    end
end

function dat_sub = subset_data(dat, idx)
    dat_sub.s = dat.s(idx); dat_sub.c = dat.c(idx); dat_sub.d = dat.d(idx);
    dat_sub.ch = dat.ch(idx); dat_sub.licks = dat.licks(idx); dat_sub.vel = dat.vel(idx);
    dat_sub.history = dat.history(idx, :); 
end
function prior = get_prior(s_range, params)
    c1 = pdfVonMises(s_range, deg2rad(0), params.prior_strength);
    c2 = pdfVonMises(s_range, deg2rad(90), params.prior_strength);
    prior = (c1+c2) ./ (sum(c1+c2)+eps);
end
function util = get_utility_vectors(s_range, utility)
    n = length(s_range); util.respond = zeros(1,n); util.no_respond = zeros(1,n);
    C1 = s_range < 45; C2 = s_range > 45; Cb = s_range==45;
    util.respond(C1)=utility.R_hit; util.respond(C2)=utility.R_fa; util.respond(Cb)=0.5*utility.R_hit+0.5*utility.R_fa;
    util.no_respond(C1)=utility.R_miss; util.no_respond(C2)=utility.R_cr;
end
function p = pdfVonMises(x, mu, k)
    p = exp(k.*cos(bsxfun(@minus,2*x,2*mu)))./(2*pi*besseli(0,k));
end
function xz = zscore_safe(x)
    mu = mean(x,'omitnan'); sd = std(x,[],'omitnan');
    if ~isfinite(sd) || sd==0, xz = zeros(size(x)); else, xz = (x - mu)./sd; end
end
function auc = get_auc(y, preds)
    if length(unique(y)) == 2
        [~,~,~,auc] = perfcurve(y, preds, 1);
    else
        auc = NaN;
    end
end