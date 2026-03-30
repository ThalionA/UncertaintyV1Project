%% Ideal Observer Parameter Recovery (Isotropic + Split Confidence)
%
% UPDATES:
% 1. Injects empirically fitted parameters from 'IOResults.mat' into the
%    synthetic animal pool to test recovery in the biological regime.
% 2. Calculates trial-by-trial Kullback-Leibler (KL) divergence between 
%    the true and fitted perceptual posteriors p(s|m).
% -------------------------------------------------------------------------
clear; close all; clc;
rng('twister'); 
warning off;

% Add BADS path
addpath(genpath('/Users/theoamvr/Desktop/Experiments/bads-master'));

%% --- Part 1: Simulation Setup & Empirical Parameter Injection ---
% 1. Load Empirical Parameters from IOResults
if exist('IOResults.mat', 'file')
    load('IOResults.mat', 'IOResults');
    n_emp = length(IOResults.animals);
    emp_params = zeros(n_emp, 9);
    for i = 1:n_emp
        emp_params(i,:) = IOResults.animals{i}.fit.params_vec;
    end
    fprintf('Loaded %d empirical parameter sets from IOResults.\n', n_emp);
else
    warning('IOResults.mat not found. Proceeding with random synthetic animals only.');
    n_emp = 0;
    emp_params = [];
end

n_random = 50; % Number of purely random synthetic animals
n_animals = n_random + n_emp; 
n_trials_per_animal = 2000; 

% Track which animals are random vs. empirically seeded
is_empirical = [false(n_random, 1); true(n_emp, 1)];

param_names = {'kappa_amp', 'c_power', 'd_power', ...
               'lick_slope', 'lick_intercept', 'lick_std', ...
               'vel_slope',  'vel_intercept',  'vel_std'};

% Bounds 
lb = [ 0,   0,   0,   -10, -5, 0.01, -10, -5, 0.01];
ub = [50,   4,   4,    10,  5, 5.00,  10,  5, 5.00];

% Fixed Model Specification
fixed_utility = struct('R_hit', 1, 'R_miss', 0, 'R_cr', 0.1, 'R_fa', -0.2);
model_spec.fit_params = param_names;
model_spec.fixed_params.s_range_deg = 0:1:90;
model_spec.fixed_params.m_range_deg = 0:1:180;
model_spec.fixed_params.prior_type = 'Bimodal';
model_spec.fixed_params.prior_strength = 3;
model_spec.fixed_params.kappa_min = 1.0; 
model_spec.fixed_params.rho_k = 0;
model_spec.fixed_params.phi_pref = 0;
model_spec.fixed_params.decision_beta = 1.0; 

fprintf('--- Simulating %d total animals (N=%d trials) ---\n', n_animals, n_trials_per_animal);

synthetic_data = {};
true_parameters = zeros(n_animals, length(param_names));

% --- Generate Synthetic Data ---
for i = 1:n_animals
    if is_empirical(i)
        p_true = emp_params(i - n_random, :);
    else
        % Random generation with biological constraints
        p_true = lb + (ub - lb) .* rand(1, length(lb));
        p_true(4) = rand*3 + 1;    % positive lick slope
        p_true(7) = -(rand*3 + 1); % negative vel slope
    end
    true_parameters(i,:) = p_true;
    
    % Generate Stimuli 
    oris = randsample([0, 15, 30, 40, 45, 50, 60, 75, 90], n_trials_per_animal, true);
    cons = randsample([0.01, 0.25, 0.5, 1.0], n_trials_per_animal, true);
    disps = randsample([0, 30, 45, 90], n_trials_per_animal, true);
    
    data_template.orientation = oris(:);
    data_template.contrast = cons(:);
    data_template.dispersion = disps(:);
    data_template.n_trials = n_trials_per_animal;
    
    % Simulate
    sim_data = simulate_dataset(data_template, p_true, fixed_utility, model_spec);
    sim_data.animal_tag = sprintf('Sim_%d', i);
    synthetic_data{i} = sim_data;
end

data_pooled = pool_data(synthetic_data);

%% --- Part 2: Group Level Fit ---
fprintf('\n--- Stage 1: Group Level Fit ---\n');
n_starts = 3; 
bads_opts = bads('defaults'); 
bads_opts.Display = 'off'; 
bads_opts.UncertaintyHandling = 0;

best_nll = inf;
p_group_hat = [];

for istart = 1:n_starts
    p0 = lb + (ub - lb) .* rand(size(lb));
    obj_fun = @(p) calculate_NLL_split(p, data_pooled, model_spec, fixed_utility);
    try
        [p_curr, nll_curr] = bads(obj_fun, p0, lb, ub, lb, ub, [], bads_opts);
        if nll_curr < best_nll
            p_group_hat = p_curr;
            best_nll = nll_curr;
        end
        fprintf('.');
    catch
        fprintf('x');
    end
end
fprintf('\n Group Fit Complete.\n');

%% --- Part 3: Individual Fits ---
fprintf('\n--- Stage 2: Individual Fits ---\n');
fitted_parameters = zeros(size(true_parameters));
fit_results = cell(n_animals, 1);

parfor i = 1:n_animals
    data = synthetic_data{i};
    obj_fun = @(p) calculate_NLL_split(p, data, model_spec, fixed_utility);
    [best_p, ~] = bads(obj_fun, p_group_hat, lb, ub, lb, ub, [], bads_opts);
    fitted_parameters(i,:) = best_p;
    
    res = struct();
    res.full_fit_params = model_spec.fixed_params;
    for k = 1:length(param_names), res.full_fit_params.(param_names{k}) = best_p(k); end
    fit_results{i} = res;
end

%% --- Part 4: KL Divergence & Inversion Validation ---
fprintf('\n--- Running Model Inversion & KL Divergence ---\n');
all_trial_data = table();

for i = 1:n_animals
    data = synthetic_data{i};
    
    % 1. True Inversion
    true_params = model_spec.fixed_params;
    for k=1:length(param_names), true_params.(param_names{k}) = true_parameters(i,k); end
    unc_true = invert_model_split(data, true_params, fixed_utility, model_spec);
    
    % 2. Fitted Inversion
    fit_params = fit_results{i}.full_fit_params;
    unc_fit = invert_model_split(data, fit_params, fixed_utility, model_spec);
    
    % 3. Calculate Trial-by-Trial KL Divergence: D_KL(P_true || P_fit)
    % P_true and P_fit are the perceptual posteriors p(s|m)
    P = unc_true.post_s_given_map;
    Q = unc_fit.post_s_given_map;
    
    % Prevent log(0)
    eps_val = 1e-10;
    P = (P + eps_val) ./ sum(P + eps_val, 2);
    Q = (Q + eps_val) ./ sum(Q + eps_val, 2);
    
    trial_KL = sum(P .* log(P ./ Q), 2);
    
    trial_dat = table(unc_true.perceptual, unc_fit.perceptual, ...
                      unc_true.decision, unc_fit.decision, trial_KL, ...
                      data.orientation, data.contrast, data.dispersion, ...
                      repmat(is_empirical(i), data.n_trials, 1), ...
                      'VariableNames', {'True_P', 'Fit_P', 'True_D', 'Fit_D', ...
                      'KL_Div', 'Orientation', 'Contrast', 'Dispersion', 'Is_Empirical'});
                  
    all_trial_data = [all_trial_data; trial_dat];
end

%% --- Part 5: Visualization ---
fprintf('\n--- Plotting Results ---\n');

% --- FIGURE 1: Parameter Recovery ---
figure('Color', 'w', 'Position', [50, 50, 1400, 900], 'Name', 'Parameter Recovery');
t = tiledlayout(3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle('Parameter Recovery (Split Confidence)');

for k = 1:length(param_names)
    nexttile; hold on;
    x = true_parameters(:, k);
    y = fitted_parameters(:, k);
    
    % Plot Random synthetic animals
    scatter(x(~is_empirical), y(~is_empirical), 40, 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.7 0.7 0.7], 'DisplayName', 'Random Seed');
    
    % Plot Empirically-yoked synthetic animals
    if any(is_empirical)
        scatter(x(is_empirical), y(is_empirical), 70, 'p', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'DisplayName', 'Empirical Seed');
    end
    
    min_v = min([x; y]); max_v = max([x; y]); range_v = max_v - min_v;
    plot([min_v-range_v*0.1 max_v+range_v*0.1], [min_v-range_v*0.1 max_v+range_v*0.1], 'k--');
    
    r_val = corr(x,y);
    title(sprintf('%s (r=%.2f)', param_names{k}, r_val), 'Interpreter', 'none');
    if k > 6, xlabel('True Parameter'); end
    if mod(k,3)==1, ylabel('Recovered Parameter'); end
    if k==1 && any(is_empirical), legend('Location','best'); end
    axis square; box on; grid on; 
    xlim([min_v-range_v*0.1 max_v+range_v*0.1]); ylim([min_v-range_v*0.1 max_v+range_v*0.1]);
end

% --- FIGURE 2: KL Divergence & Posterior Validation ---
figure('Color', 'w', 'Position', [100, 100, 1500, 400], 'Name', 'Posterior Validation');

% 1. Global Uncertainty Scatters
subplot(1,4,1); hold on;
scatter(all_trial_data.True_P, all_trial_data.Fit_P, 10, all_trial_data.KL_Div, 'filled', 'MarkerFaceAlpha', 0.2);
colormap(gca, 'parula'); cb=colorbar; cb.Label.String = 'KL Divergence';
plot([0 100], [0 100], 'k--'); 
r_p = corr(all_trial_data.True_P, all_trial_data.Fit_P);
title(sprintf('Perceptual Uncertainty\n(r = %.2f)', r_p));
xlabel('True Model Unc'); ylabel('Fitted Model Unc'); axis square; grid on;
xlim([0 max(all_trial_data.True_P)]); ylim([0 max(all_trial_data.True_P)]);

% 2. KL Divergence by Orientation
subplot(1,4,2); hold on;
[mu_kl, g_kl, sem_kl] = grpstats(all_trial_data.KL_Div, all_trial_data.Orientation, {'mean','gname','sem'});
errorbar(str2double(g_kl), mu_kl, sem_kl, 'k-o', 'LineWidth', 2, 'MarkerFaceColor','k');
title('KL Divergence by Orientation');
xlabel('Orientation (deg)'); ylabel('Mean D_{KL} (nats)');
grid on; box on; xlim([-5 95]);

% 3. KL Divergence by Contrast
subplot(1,4,3); hold on;
[mu_kl, g_kl, sem_kl] = grpstats(all_trial_data.KL_Div, all_trial_data.Contrast, {'mean','gname','sem'});
errorbar(str2double(g_kl), mu_kl, sem_kl, 'b-o', 'LineWidth', 2, 'MarkerFaceColor','b');
title('KL Divergence by Contrast');
xlabel('Contrast'); ylabel('Mean D_{KL} (nats)');
grid on; box on; xlim([-0.05 1.05]);

% 4. KL Divergence by Dispersion
subplot(1,4,4); hold on;
[mu_kl, g_kl, sem_kl] = grpstats(all_trial_data.KL_Div, all_trial_data.Dispersion, {'mean','gname','sem'});
errorbar(str2double(g_kl), mu_kl, sem_kl, 'r-o', 'LineWidth', 2, 'MarkerFaceColor','r');
title('KL Divergence by Dispersion');
xlabel('Dispersion (deg)'); ylabel('Mean D_{KL} (nats)');
grid on; box on; xlim([-5 max(all_trial_data.Dispersion)+5]);

%% --- Core Functions ---
% (simulate_dataset, pool_data, calculate_NLL_split remain functionally identical)

function data_pooled = pool_data(synthetic_data)
    data_pooled.orientation = []; data_pooled.contrast = []; data_pooled.dispersion = [];
    data_pooled.choices = []; data_pooled.conf_licks = []; data_pooled.conf_vel = [];
    for i = 1:numel(synthetic_data)
        d = synthetic_data{i};
        data_pooled.orientation = [data_pooled.orientation; d.orientation];
        data_pooled.contrast = [data_pooled.contrast; d.contrast];
        data_pooled.dispersion = [data_pooled.dispersion; d.dispersion];
        data_pooled.choices = [data_pooled.choices; d.choices];
        data_pooled.conf_licks = [data_pooled.conf_licks; d.conf_licks];
        data_pooled.conf_vel = [data_pooled.conf_vel; d.conf_vel];
    end
    data_pooled.n_trials = length(data_pooled.orientation);
end

function sim_data = simulate_dataset(template, params_vec, utility, model_spec)
    params = model_spec.fixed_params;
    for i=1:length(model_spec.fit_params), params.(model_spec.fit_params{i}) = params_vec(i); end
    sim_data = template;
    s_range = deg2rad(model_spec.fixed_params.s_range_deg);
    m_range = deg2rad(model_spec.fixed_params.m_range_deg);
    util = get_utility_vectors(model_spec.fixed_params.s_range_deg, utility);
    prior = get_prior(s_range, params);
    
    n = template.n_trials;
    choices_out = zeros(n, 1); licks_out = zeros(n, 1); vel_out = zeros(n, 1);
    conds = [template.orientation, template.contrast, template.dispersion];
    [u_conds, ~, idx] = unique(conds, 'rows');
    m_maps = cell(size(u_conds, 1), 1);
    
    for j = 1:size(u_conds, 1)
        s = u_conds(j,1); c = u_conds(j,2); d = u_conds(j,3);
        k_gen = (params.kappa_min + params.kappa_amp) * (c^params.c_power) * exp(-params.d_power * d);
        p_m_s = pdfVonMises(m_range, deg2rad(s), k_gen);
        m_maps{j}.p_m_s = p_m_s ./ (sum(p_m_s) + eps);
        
        lik = pdfVonMises(m_range', s_range, k_gen);
        post = (lik .* prior) ./ (sum(lik .* prior, 2) + eps);
        dv = (post * util.respond') - (post * util.no_respond');
        
        m_maps{j}.p_go_m = 1 ./ (1 + exp(-params.decision_beta * dv));
        m_maps{j}.licks_m = params.lick_slope * dv + params.lick_intercept;
        m_maps{j}.vel_m   = params.vel_slope  * dv + params.vel_intercept;
    end
    
    for i = 1:n
        maps_i = m_maps{idx(i)};
        m_idx = find(mnrnd(1, maps_i.p_m_s), 1);
        choices_out(i) = rand() < maps_i.p_go_m(m_idx);
        licks_out(i)   = maps_i.licks_m(m_idx) + randn() * params.lick_std;
        vel_out(i)     = maps_i.vel_m(m_idx)   + randn() * params.vel_std;
    end
    sim_data.choices = choices_out; sim_data.conf_licks = licks_out; sim_data.conf_vel = vel_out;
end

function NLL = calculate_NLL_split(p_vec, data, spec, utility)
    params = spec.fixed_params;
    for i = 1:length(p_vec), params.(spec.fit_params{i}) = p_vec(i); end
    s_grid = deg2rad(params.s_range_deg); m_grid = deg2rad(params.m_range_deg);
    prior = get_prior(s_grid, params);
    util = get_utility_vectors(params.s_range_deg, utility);
    conds = [data.orientation, data.contrast, data.dispersion];
    [u_conds, ~, idx] = unique(conds, 'rows');
    total_nll = 0;
    
    for j = 1:size(u_conds, 1)
        mask = (idx == j); if ~any(mask), continue; end
        s = u_conds(j,1); c = u_conds(j,2); d = u_conds(j,3);
        
        k_gen = (params.kappa_min + params.kappa_amp) * (c^params.c_power) * exp(-params.d_power * d);
        lik = pdfVonMises(m_grid', s_grid, k_gen);
        post = (lik .* prior) ./ (sum(lik .* prior, 2) + eps);
        dv = (post * util.respond') - (post * util.no_respond');
        
        mu_licks = (params.lick_slope * dv + params.lick_intercept)';
        mu_vel   = (params.vel_slope  * dv + params.vel_intercept)';
        p_m_s = pdfVonMises(m_grid, deg2rad(s), k_gen);
        p_m_s = p_m_s ./ (sum(p_m_s) + eps);
        
        n_t = sum(mask);
        L_l = normpdf(repmat(data.conf_licks(mask), 1, length(m_grid)), repmat(mu_licks, n_t, 1), params.lick_std);
        L_v = normpdf(repmat(data.conf_vel(mask),   1, length(m_grid)), repmat(mu_vel,   n_t, 1), params.vel_std);
        
        L_trial = sum(L_l .* L_v .* repmat(p_m_s, n_t, 1), 2);
        total_nll = total_nll - sum(log(L_trial + eps));
    end
    NLL = total_nll; if isnan(NLL) || isinf(NLL), NLL = 1e10; end
end

function inferred_unc = invert_model_split(data, params, utility, base_spec)
    % MODIFIED to return the full stimulus posterior at the MAP measurement
    m_rad = deg2rad(base_spec.fixed_params.m_range_deg);
    s_deg = base_spec.fixed_params.s_range_deg; s_rad = deg2rad(s_deg);
    prior_ps = get_prior(s_rad, params);
    utility_vec = get_utility_vectors(s_deg, utility);
    cond_mat = [data.orientation, data.contrast, data.dispersion];
    [G_unique, ~, t_idx] = unique(cond_mat, 'rows');
    
    maps = cell(size(G_unique,1), 1);
    gen_kappas = zeros(size(G_unique,1), 1);
    
    for j = 1:size(G_unique,1)
        s = G_unique(j,1); c = G_unique(j,2); d = G_unique(j,3);
        k_gen = (params.kappa_min + params.kappa_amp) * (c^params.c_power) * exp(-params.d_power * d);
        gen_kappas(j) = k_gen;
        
        lik = pdfVonMises(m_rad', s_rad, k_gen);
        post_s_m = (lik .* prior_ps) ./ (sum(lik .* prior_ps, 2) + eps);
        maps{j}.post_s_m = post_s_m; % Save full posterior grid
        
        R = abs(sum(post_s_m .* exp(1i*s_rad), 2));
        maps{j}.perc_unc = rad2deg(sqrt(-2 * log(R + eps)));
        
        idx_go = s_deg < 45; idx_bnd = s_deg==45;
        p_go = sum(post_s_m(:, idx_go), 2) + 0.5*sum(post_s_m(:, idx_bnd), 2);
        p_no = 1 - p_go; p_go = max(eps, min(1-eps, p_go)); p_no = max(eps, min(1-eps, p_no));
        maps{j}.dec_unc = -(p_go.*log2(p_go) + p_no.*log2(p_no));
        
        dv = (post_s_m * utility_vec.respond') - (post_s_m * utility_vec.no_respond');
        maps{j}.pred_lick = params.lick_slope * dv + params.lick_intercept;
        maps{j}.pred_vel  = params.vel_slope  * dv + params.vel_intercept;
    end
    
    out_perc = nan(data.n_trials, 1);
    out_dec = nan(data.n_trials, 1);
    out_post_s = nan(data.n_trials, length(s_deg));
    
    for i = 1:data.n_trials
        idx = t_idx(i); beh_map = maps{idx};
        prior_m = pdfVonMises(m_rad, deg2rad(data.orientation(i)), gen_kappas(idx));
        prior_m = prior_m / (sum(prior_m)+eps);
        
        L_l = normpdf(data.conf_licks(i), beh_map.pred_lick, params.lick_std);
        L_v = normpdf(data.conf_vel(i),   beh_map.pred_vel,  params.vel_std);
        
        post_m = L_l' .* L_v' .* prior_m; 
        [~, map_idx] = max(post_m);
        
        out_perc(i) = beh_map.perc_unc(map_idx);
        out_dec(i)  = beh_map.dec_unc(map_idx);
        out_post_s(i, :) = beh_map.post_s_m(map_idx, :);
    end
    inferred_unc.perceptual = out_perc;
    inferred_unc.decision = out_dec;
    inferred_unc.post_s_given_map = out_post_s;
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