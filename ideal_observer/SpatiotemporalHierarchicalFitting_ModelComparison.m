%% Spatiotemporal Hierarchical Fitting & Model Comparison
%
% Re-runs the full hierarchical fitting pipeline independently for every 
% spatial window.
%
% Models evaluated on identical 5-fold cross-validation splits predicting 
% binary choice (Go/No-Go) via AUC and Binomial NLL.
%
% 1. Hierarchical IO (Full): Licks + Vel
% 2. Hierarchical IO (Reduced): Vel Only
% 3. GLM Full Integrated: Stimulus + Kinematics + History (Absolute Ceiling)
% 4. GLM Sensorimotor: Stimulus + Kinematics
% 5. GLM Kinematic: Vel + Licks
% 6. GLM Kinematic: Vel Only
% 7. GLM Kinematic: Licks Only
% 8. GLM Sensory: Stimulus
% 9. GLM History: Prev Choice + Prev Reward

clear; close all; clc;
rng("twister");
warning off;

fprintf('--- Starting Unified Spatiotemporal Model Comparison Pipeline ---\n');

% Add BADS path if necessary
addpath(genpath('/Users/theoamvr/Desktop/Experiments/bads-master'));

%% --- Part 1: Configuration ---
sesnames1 = {'20250605_Cb15', '20250613_Cb15', '20250620_Cb15', '20250624_Cb15', '20250709_Cb15'};
sesnames2 = {'20250606_Cb17', '20250613_Cb17', '20250620_Cb17', '20250624_Cb17', '20250701_Cb17'};
sesnames3 = {'20250904_Cb21', '20250910_Cb21', '20250911_Cb21', '20250912_Cb21', '20250918_Cb21'};
sesnames4 = {'20251024_Cb22', '20251027_Cb22', '20251028_Cb22', '20251030_Cb22', '20251105_Cb22'};
sesnames5 = {'20250918_Cb24', '20250919_Cb24', '20251020_Cb24', '20251021_Cb24'};
sesnames6 = {'20250903_Cb25', '20250904_Cb25', '20250910_Cb25', '20250911_Cb25', '20250916_Cb25'};
all_sesnames = {sesnames1, sesnames2, sesnames3, sesnames4, sesnames5, sesnames6}; 

% Specify 'horizontal' (go <45) or 'vertical' (go >45) per animal
go_side_by_animal = { ...
    'horizontal', ... % animal 1
    'horizontal', ... % animal 2
    'vertical', ...   % animal 3
    'vertical', ...   % animal 4
    'vertical', ...   % animal 5
    'horizontal' ...  % animal 6
};

W = 10; % Window width
S = 10;  % Step size
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
            ses_hist = nan(n_trials, 2); % [Prev_Choice, Prev_Rewarded]
            
            for t = 1:n_trials
                pos = vr.trialLog{t}.position(2,:);
                lick = vr.trialLog{t}.lick;
                vel = -vr.trialLog{t}.velocity(2,:) * 0.2537;
                
                ses_choices(t) = sum(lick(pos >= rz_start & pos <= vr.cfg.rewardZoneEnd)) > 0;
                
                % Handle stimulus alignment
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
        % Valid now requires history to be non-NaN
        valid = ~isnan(d.licks(:,w)) & ~isnan(d.vel(:,w)) & ~any(isnan(d.stim),2) & ~isnan(d.choices) & ~any(isnan(d.history),2);
        pool_full.s = [pool_full.s; d.stim(valid,1)];
        pool_full.c = [pool_full.c; d.stim(valid,2)];
        pool_full.d = [pool_full.d; d.stim(valid,3)];
        pool_full.ch = [pool_full.ch; d.choices(valid)];
        pool_full.licks = [pool_full.licks; d.licks(valid,w)];
        pool_full.vel = [pool_full.vel; d.vel(valid,w)];
    end
    
    % 2. Group-Level BADS fits
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
        
        if sum(valid) < 30 || length(unique(d.choices(valid))) < 2
            animal_res{i_animal} = struct('status', 'insufficient_data');
            continue;
        end
        
        dat_ind = struct('s', d.stim(valid,1), 'c', d.stim(valid,2), 'd', d.stim(valid,3), ...
                         'ch', d.choices(valid), 'licks', d.licks(valid,w), 'vel', d.vel(valid,w), ...
                         'history', d.history(valid,:));
        
        cv = cvpartition(length(dat_ind.ch), 'KFold', k_folds);
        
        % Updated Results Struct with new names
        res = struct('IO_Full_NLL', 0, 'IO_Vel_NLL', 0, ...
            'IO_Full_Choice_NLL', 0, 'IO_Vel_Choice_NLL', 0, ...
            'GLM_Full_NLL', 0, 'GLM_Sensorimotor_NLL', 0, ...
            'GLM_Kinematic_LV_NLL', 0, 'GLM_Kinematic_V_NLL', 0, ...
            'GLM_Kinematic_L_NLL', 0, 'GLM_Sensory_NLL', 0, 'GLM_History_NLL', 0);
        
        preds_IO_Full = nan(length(dat_ind.ch), 1); 
        preds_IO_Vel  = nan(length(dat_ind.ch), 1);
        
        preds_Full_Int  = nan(length(dat_ind.ch), 1); 
        preds_SensMotor = nan(length(dat_ind.ch), 1); 
        preds_LV        = nan(length(dat_ind.ch), 1); 
        preds_V         = nan(length(dat_ind.ch), 1);
        preds_L         = nan(length(dat_ind.ch), 1); 
        preds_Sensory   = nan(length(dat_ind.ch), 1);
        preds_Hist      = nan(length(dat_ind.ch), 1);
        
        for k = 1:cv.NumTestSets
            tr = cv.training(k); te = cv.test(k);
            
            dat_tr = subset_data(dat_ind, tr);
            dat_te = subset_data(dat_ind, te);
            
            % --- Fit IO Full (Multi-Start) ---
            obj_tr_full = @(p) calc_IO_NLL(p, dat_tr, spec_full, fixed_utility, 'both');
            p_fit_full = fit_with_multi_start(obj_tr_full, group_p_full, lb_f, ub_f, plb_f, pub_f, bads_opts, n_jitter_starts);
            res.IO_Full_NLL = res.IO_Full_NLL + calc_IO_NLL(p_fit_full, dat_te, spec_full, fixed_utility, 'both');
            preds_IO_Full(te) = get_IO_choice_preds(p_fit_full, dat_te, spec_full, fixed_utility, 'both');
            
            % --- Fit IO Vel-Only (Multi-Start) ---
            obj_tr_vel = @(p) calc_IO_NLL(p, dat_tr, spec_vel, fixed_utility, 'vel_only');
            p_fit_vel = fit_with_multi_start(obj_tr_vel, group_p_vel, lb_v, ub_v, plb_v, pub_v, bads_opts, n_jitter_starts);
            res.IO_Vel_NLL = res.IO_Vel_NLL + calc_IO_NLL(p_fit_vel, dat_te, spec_vel, fixed_utility, 'vel_only');
            preds_IO_Vel(te) = get_IO_choice_preds(p_fit_vel, dat_te, spec_vel, fixed_utility, 'vel_only');
            
            % --- Prepare Predictor Matrices for GLMs ---
            % 1. Full Integrated: Stimulus + Kinematics + History
            X_Full_tr = [dat_tr.s, dat_tr.c, dat_tr.d, dat_tr.vel, dat_tr.licks, dat_tr.history];
            X_Full_te = [dat_te.s, dat_te.c, dat_te.d, dat_te.vel, dat_te.licks, dat_te.history];
            
            % 2. Sensorimotor: Stimulus + Kinematics
            X_SensMotor_tr = [dat_tr.s, dat_tr.c, dat_tr.d, dat_tr.vel, dat_tr.licks];
            X_SensMotor_te = [dat_te.s, dat_te.c, dat_te.d, dat_te.vel, dat_te.licks];
            
            % 3. Kinematic variants
            X_LV_tr = [dat_tr.vel, dat_tr.licks]; X_LV_te = [dat_te.vel, dat_te.licks];
            X_V_tr = dat_tr.vel; X_V_te = dat_te.vel;
            X_L_tr = dat_tr.licks; X_L_te = dat_te.licks;
            
            % 4. Sensory: Stimulus only
            X_S_tr = [dat_tr.s, dat_tr.c, dat_tr.d]; X_S_te = [dat_te.s, dat_te.c, dat_te.d];
            
            % 5. History only
            X_Hist_tr = dat_tr.history; X_Hist_te = dat_te.history;
            
            % --- Fit GLMs with Penalized Likelihood ---
            mdl_Full = fitglm(X_Full_tr, dat_tr.ch, 'Distribution', 'binomial', 'LikelihoodPenalty', 'jeffreys-prior');
            preds_Full_Int(te) = predict(mdl_Full, X_Full_te);
            
            mdl_SensMotor = fitglm(X_SensMotor_tr, dat_tr.ch, 'Distribution', 'binomial', 'LikelihoodPenalty', 'jeffreys-prior');
            preds_SensMotor(te) = predict(mdl_SensMotor, X_SensMotor_te);
            
            mdl_LV = fitglm(X_LV_tr, dat_tr.ch, 'Distribution', 'binomial', 'LikelihoodPenalty', 'jeffreys-prior');
            preds_LV(te) = predict(mdl_LV, X_LV_te);
            
            mdl_V = fitglm(X_V_tr, dat_tr.ch, 'Distribution', 'binomial', 'LikelihoodPenalty', 'jeffreys-prior');
            preds_V(te) = predict(mdl_V, X_V_te);
            
            mdl_L = fitglm(X_L_tr, dat_tr.ch, 'Distribution', 'binomial', 'LikelihoodPenalty', 'jeffreys-prior');
            preds_L(te) = predict(mdl_L, X_L_te);
            
            mdl_Sensory = fitglm(X_S_tr, dat_tr.ch, 'Distribution', 'binomial', 'LikelihoodPenalty', 'jeffreys-prior');
            preds_Sensory(te) = predict(mdl_Sensory, X_S_te);
            
            mdl_Hist = fitglm(X_Hist_tr, dat_tr.ch, 'Distribution', 'binomial', 'LikelihoodPenalty', 'jeffreys-prior');
            preds_Hist(te) = predict(mdl_Hist, X_Hist_te);
            
            % --- Binomial NLL Accumulation ---
            calc_glm_nll = @(p, y) -sum(y .* log(p + eps) + (1 - y) .* log(1 - p + eps));
            
            res.IO_Full_Choice_NLL   = res.IO_Full_Choice_NLL   + calc_glm_nll(preds_IO_Full(te), dat_te.ch);
            res.IO_Vel_Choice_NLL    = res.IO_Vel_Choice_NLL    + calc_glm_nll(preds_IO_Vel(te), dat_te.ch);
            res.GLM_Full_NLL         = res.GLM_Full_NLL         + calc_glm_nll(preds_Full_Int(te), dat_te.ch);
            res.GLM_Sensorimotor_NLL = res.GLM_Sensorimotor_NLL + calc_glm_nll(preds_SensMotor(te), dat_te.ch);
            res.GLM_Kinematic_LV_NLL = res.GLM_Kinematic_LV_NLL + calc_glm_nll(preds_LV(te), dat_te.ch);
            res.GLM_Kinematic_V_NLL  = res.GLM_Kinematic_V_NLL  + calc_glm_nll(preds_V(te), dat_te.ch);
            res.GLM_Kinematic_L_NLL  = res.GLM_Kinematic_L_NLL  + calc_glm_nll(preds_L(te), dat_te.ch);
            res.GLM_Sensory_NLL      = res.GLM_Sensory_NLL      + calc_glm_nll(preds_Sensory(te), dat_te.ch);
            res.GLM_History_NLL      = res.GLM_History_NLL      + calc_glm_nll(preds_Hist(te), dat_te.ch);
        end
        
        safe_auc = @(y, p) get_auc(y, p);
        res.IO_Full_AUC          = safe_auc(dat_ind.ch, preds_IO_Full);
        res.IO_Vel_AUC           = safe_auc(dat_ind.ch, preds_IO_Vel);
        res.GLM_Full_AUC         = safe_auc(dat_ind.ch, preds_Full_Int);
        res.GLM_Sensorimotor_AUC = safe_auc(dat_ind.ch, preds_SensMotor);
        res.GLM_Kinematic_LV_AUC = safe_auc(dat_ind.ch, preds_LV);
        res.GLM_Kinematic_V_AUC  = safe_auc(dat_ind.ch, preds_V);
        res.GLM_Kinematic_L_AUC  = safe_auc(dat_ind.ch, preds_L);
        res.GLM_Sensory_AUC      = safe_auc(dat_ind.ch, preds_Sensory);
        res.GLM_History_AUC      = safe_auc(dat_ind.ch, preds_Hist);
        
        animal_res{i_animal} = res;
    end
    
    WindowResults(w).animal_data = animal_res;
end

%% --- Part 4: Save Results ---
save('Spatiotemporal_Model_Comparison_Unified.mat', 'WindowResults', '-v7.3');
fprintf('Analysis complete and saved to Spatiotemporal_Model_Comparison_Unified.mat\n');

save('Spatiotemporal_Decoding_Choice.mat', 'WindowResults', 'spatial_centers', 'n_windows', 'n_animals', '-v7.3');
fprintf('Analysis complete and saved to Spatiotemporal_Decoding_Choice.mat\n');

%% --- Part 5: Plotting Model Comparison Across Space ---
fprintf('Generating Unified Spatiotemporal Model Comparison plots...\n');
figure('Name', 'Unified Model Comparison: AUC across Spatial Windows', 'Color', 'w', 'Position', [100, 100, 1050, 700]);
hold on;

plot_centers = [WindowResults.center];

% Updated metrics list matching our new nomenclature
metrics = {'IO_Full_AUC', 'IO_Vel_AUC', 'GLM_Full_AUC', 'GLM_Sensorimotor_AUC', ...
           'GLM_Kinematic_LV_AUC', 'GLM_Kinematic_V_AUC', 'GLM_Kinematic_L_AUC', ...
           'GLM_Sensory_AUC', 'GLM_History_AUC'};
       
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

% Averages across animals
avg_res = squeeze(mean(res_matrix, 1, 'omitnan'));

% --- Define Color Palette ---
% Family 1: Ideal Observers (Blues)
col_io_full = [0, 0.4470, 0.7410];       % Deep Blue
col_io_red  = [0.3010, 0.7450, 0.9330];  % Light Blue

% Family 2: Complex GLMs (Reds/Purples)
col_glm_full = [0.8500, 0.3250, 0.0980]; % Orange/Red
col_glm_sm   = [0.4940, 0.1840, 0.5560]; % Purple

% Family 3: Kinematic GLMs (Greens)
col_kin      = [0.4660, 0.6740, 0.1880]; % Base Green

% Family 4: Baselines (Greys)
col_sensory  = [0.2, 0.2, 0.2];          % Dark Grey
col_history  = [0.6, 0.6, 0.6];          % Light Grey

% --- Plotting ---
% 1. Baselines (plotted first so they sit in the background)
plot(plot_centers, avg_res(:,9), ':', 'Color', col_history, 'LineWidth', 2.5, 'DisplayName', 'GLM History');
plot(plot_centers, avg_res(:,8), '--', 'Color', col_sensory, 'LineWidth', 2.5, 'DisplayName', 'GLM Sensory');

% 2. Kinematic GLMs (Greens - varying markers/lines)
plot(plot_centers, avg_res(:,7), ':', 'Color', col_kin, 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'DisplayName', 'GLM Kinematic (Licks)');
plot(plot_centers, avg_res(:,6), '--', 'Color', col_kin, 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'DisplayName', 'GLM Kinematic (Vel)');
plot(plot_centers, avg_res(:,5), '-', 'Color', col_kin, 'LineWidth', 2.5, 'Marker', 's', 'MarkerSize', 7, 'MarkerFaceColor', col_kin, 'DisplayName', 'GLM Kinematic (Vel+Licks)');

% 3. Complex GLMs (Reds/Purples)
plot(plot_centers, avg_res(:,4), '-', 'Color', col_glm_sm, 'LineWidth', 2.5, 'Marker', '^', 'MarkerSize', 7, 'MarkerFaceColor', col_glm_sm, 'DisplayName', 'GLM Sensorimotor');
plot(plot_centers, avg_res(:,3), '-', 'Color', col_glm_full, 'LineWidth', 3.5, 'Marker', 'd', 'MarkerSize', 8, 'MarkerFaceColor', col_glm_full, 'DisplayName', 'GLM Full Integrated (Ceiling)');

% 4. Ideal Observers (Blues - plotted last so they are on top)
plot(plot_centers, avg_res(:,2), '-', 'Color', col_io_red, 'LineWidth', 2.5, 'Marker', 'o', 'MarkerSize', 7, 'MarkerFaceColor', 'w', 'DisplayName', 'IO Reduced (Vel)');
plot(plot_centers, avg_res(:,1), '-', 'Color', col_io_full, 'LineWidth', 3, 'Marker', 'o', 'MarkerSize', 8, 'MarkerFaceColor', col_io_full, 'DisplayName', 'IO Full (Licks+Vel)');

% --- Formatting & Aesthetics ---
xlabel('Spatial Window Center (VR units)', 'FontWeight', 'bold');
ylabel('Average Cross-Validated AUC', 'FontWeight', 'bold');
title('Predictive Power of Models over Space', 'FontSize', 14, 'FontWeight', 'bold');

% Improve legend
lgd = legend('Location', 'eastoutside', 'NumColumns', 1);
title(lgd, 'Evaluated Models');
lgd.Box = 'off';

% Axes aesthetics
set(gca, 'FontSize', 12, 'TickDir', 'out', 'Box', 'off', 'LineWidth', 1.5);
xlim([min(plot_centers)-5, max(plot_centers)+5]);
ylim([0.45, 1.0]); % Assuming AUC ranges from chance (0.5) to perfect (1)
yline(0.5, 'k-', 'Chance', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left', 'HandleVisibility', 'off');

% Soften the grid
grid on;
ax = gca;
ax.GridAlpha = 0.15;

%% --- Core Functions ---

function best_p = fit_with_multi_start(obj_fun, p_base, lb, ub, plb, pub, opts, n_starts)
    best_nll = inf;
    best_p = p_base;
    for i = 1:n_starts
        if i == 1
            p_start = p_base; % Seed with group level
        else
            % Jitter within plausible bounds
            jitter = (rand(size(p_base)) - 0.5) .* (pub - plb) * 0.1;
            p_start = p_base + jitter;
            p_start = max(lb, min(ub, p_start)); % Enforce hard bounds
        end
        
        try
            [p_fit, nll_fit] = bads(obj_fun, p_start, lb, ub, plb, pub, [], opts);
            if nll_fit < best_nll
                best_nll = nll_fit;
                best_p = p_fit;
            end
        catch
        end
    end
end

function [lb, ub, plb, pub, p0] = get_bounds(param_names)
    n = length(param_names); lb = zeros(1,n); ub = zeros(1,n); p0 = zeros(1,n);
    for i = 1:n
        name = param_names{i};
        if strcmp(name, 'kappa_amp'), lb(i)=0; ub(i)=50; p0(i)=10;
        elseif contains(name, 'power'), lb(i)=0; ub(i)=5; p0(i)=1;
        elseif contains(name, 'lick_slope'), lb(i)=0; ub(i)=10; p0(i)=2; % BOUNDED strictly positive
        elseif contains(name, 'vel_slope'), lb(i)=-10; ub(i)=0; p0(i)=-2; % BOUNDED strictly negative
        elseif contains(name, '_intercept'), lb(i)=-5; ub(i)=5; p0(i)=0;
        elseif contains(name, '_std'), lb(i)=0.01; ub(i)=5; p0(i)=0.5;
        end
    end
    plb = max(lb, p0 - abs(p0)*0.5 - 0.1);
    pub = min(ub, p0 + abs(p0)*0.5 + 0.1);
end

function NLL = calc_IO_NLL(p_vec, data, spec, utility, fit_mode)
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

function p_go_inferred = get_IO_choice_preds(p_vec, data, spec, utility, fit_mode)
    % Inverts the IO model to infer trial-by-trial choice probability
    % by conditioning the latent variable 'm' on the observed motor behavior.
    
    params = spec.fixed_params;
    for i = 1:length(p_vec), params.(spec.fit_params{i}) = p_vec(i); end
    
    s_grid = deg2rad(params.s_range_deg); 
    m_grid = deg2rad(params.m_range_deg);
    prior = get_prior(s_grid, params);
    util = get_utility_vectors(params.s_range_deg, utility);
    
    n_trials = length(data.ch);
    p_go_inferred = nan(n_trials, 1);
    
    for j = 1:n_trials
        s = data.s(j);
        c = data.c(j);
        d = data.d(j);
        obs_lick = data.licks(j);
        obs_vel = data.vel(j);
        
        % 1. Generative precision for this trial's stimulus
        k_gen = (params.kappa_min + params.kappa_amp) * (c^params.c_power) * exp(-params.d_power * d);
        
        % 2. Prior for inference: Generative p(m|s_true)
        p_m_given_s = pdfVonMises(m_grid, deg2rad(s), k_gen);
        p_m_given_s = p_m_given_s ./ (sum(p_m_given_s) + eps);
        
        % 3. Internal inference: p(s|m) to compute the Decision Variable
        lik_inf = pdfVonMises(m_grid', s_grid, k_gen);
        post_s_given_m = (lik_inf .* prior) ./ (sum(lik_inf .* prior, 2) + eps);
        dv = (post_s_given_m * util.respond') - (post_s_given_m * util.no_respond');
        
        % Softmax P(Go) predicted for every possible 'm'
        p_go_m = 1 ./ (1 + exp(-params.decision_beta * dv));
        
        % 4. Invert model using observed behavior to find Posterior p(m | s_true, behavior)
        L_joint = ones(1, length(m_grid));
        
        % Velocity Likelihood
        if isfield(params, 'vel_slope')
            mu_vel = params.vel_slope * dv + params.vel_intercept;
            L_v = normpdf(obs_vel, mu_vel, params.vel_std);
            L_joint = L_joint .* L_v';
        end
        
        % Lick Likelihood
        if strcmp(fit_mode, 'both') && isfield(params, 'lick_slope')
            mu_licks = params.lick_slope * dv + params.lick_intercept;
            L_l = normpdf(obs_lick, mu_licks, params.lick_std);
            L_joint = L_joint .* L_l';
        end
        
        post_m_unnorm = L_joint .* p_m_given_s;
        post_m = post_m_unnorm ./ (sum(post_m_unnorm) + eps);
        
        % 5. Marginalize P(Go) over the inverted posterior of 'm'
        p_go_inferred(j) = sum(p_go_m .* post_m');
    end
end

function dat_sub = subset_data(dat, idx)
    dat_sub.s = dat.s(idx); dat_sub.c = dat.c(idx); dat_sub.d = dat.d(idx);
    dat_sub.ch = dat.ch(idx); dat_sub.licks = dat.licks(idx); dat_sub.vel = dat.vel(idx);
    dat_sub.history = dat.history(idx, :); % Added history extraction
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