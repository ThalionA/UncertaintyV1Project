function parameter_recovery_v2(prior_type, prior_strength, output_path, io_results_path, varargin)
%% Parameter Recovery (V2-aligned)
%
% Mirrors ideal_observer_hierarchical_fitting_v2.m end-to-end:
%   - Stage 1 conf_only: fits {kappa_amp, c_power, d_power} + velocity
%     emission {vel_slope, vel_intercept, vel_std}. Choice excluded.
%   - Stage 2: fits {alpha_r, beta_r, gamma_r, delta_r} mapping log
%     posterior odds g(m) -> P(Go) with lapses, conditioned on velocity
%     via Bayes update on m.
%   - Shared 5-fold cvpartition across Stage 1 and Stage 2 (matches v2
%     after the 2026-05-04 refactor).
%   - Marginalised inversion -> Q(theta) and L(theta).
%
% Synthetic generation (per trial):
%   1) sample m ~ vonMises(2*s, kappa_gen);
%   2) vel ~ Normal(vel_slope*DV(m) + vel_intercept, vel_std);
%   3) choice ~ Bernoulli(gamma_r + (1-gamma_r-delta_r) * sigm(alpha_r*g(m)+beta_r)).
%   DV(m) and g(m) are computed under the SAME prior used for fitting
%   ("match generative + fit"). This is a generative-from-the-true-θ
%   process; the fitter recovers θ via the v2 marginalised likelihood.
%
% Args
%   prior_type        : 'Flat' | 'Bimodal' | 'Unimodal' | 'Empirical'
%   prior_strength    : scalar (kappa for the prior; ignored for 'Flat')
%   output_path       : where to save RecoveryResults_<tag>.mat
%   io_results_path   : path to matching IOResults_<tag>.mat for empirical-yoked
%                       seeding. Pass '' to skip yoking and use random only.
%   varargin (name/value):
%     'n_random'      : default 20 (random synthetic animals)
%     'n_yoked'       : default 6  (max yoked animals; capped by IOResults)
%     'n_trials'      : default 2000
%     'fit_licks'     : default false (matches v2 default)
%     'rng_seed'      : default 'twister' (call rng with this)
%
% NOTE: Helper functions at the bottom are duplicated from v2 for
% self-containment. Keep them in sync if v2 is updated.

% ---------------- arg defaults ----------------
p = inputParser;
p.addParameter('n_random',  20, @(x) isnumeric(x) && isscalar(x) && x>=0);
p.addParameter('n_yoked',    6, @(x) isnumeric(x) && isscalar(x) && x>=0);
p.addParameter('n_trials', 2000, @(x) isnumeric(x) && isscalar(x) && x>=100);
p.addParameter('fit_licks', false, @islogical);
p.addParameter('rng_seed', 'twister');
p.parse(varargin{:});
opts = p.Results;

if nargin < 4 || isempty(io_results_path), io_results_path = ''; end
if nargin < 3 || isempty(output_path)
    error('parameter_recovery_v2:NoOutputPath', 'output_path is required.');
end
if nargin < 2 || isempty(prior_strength), prior_strength = NaN; end
if nargin < 1 || isempty(prior_type), error('prior_type required'); end

rng(opts.rng_seed);
warning off;

% Add BADS path
addpath(genpath('/Users/theoamvr/Desktop/Experiments/bads-master'));

fprintf('=== parameter_recovery_v2 ===\n');
fprintf(' Prior: type=%s, strength=%g\n', prior_type, prior_strength);
fprintf(' Random animals: %d | Yoked animals: up to %d | Trials/animal: %d\n', ...
    opts.n_random, opts.n_yoked, opts.n_trials);
fprintf(' fit_licks=%d | output=%s\n', opts.fit_licks, output_path);

%% ---------------- Model spec (matches v2) ----------------
config = struct('window_start', 30, 'window_width', 10, ...
                'fit_licks', opts.fit_licks, 'fit_choice_psych', true);

sensory_params  = {'kappa_amp','c_power','d_power'};
vel_conf_params  = {'vel_slope','vel_intercept','vel_std'};
lick_conf_params = {'lick_slope','lick_intercept','lick_std'};
if config.fit_licks
    conf_params = [lick_conf_params, vel_conf_params];
    initial_guesses = [10, 1.0, 1.0, 2.0, 0.0, 0.5, -2.0, 0.0, 0.5];
else
    conf_params = vel_conf_params;
    initial_guesses = [10, 1.0, 1.0, -2.0, 0.0, 0.5];
end
model_spec.fit_params = [sensory_params, conf_params];
model_spec.config = config;
model_spec.fixed_params.s_range_deg = 0:1:90;
model_spec.fixed_params.m_range_deg = 0:1:180;
model_spec.fixed_params.prior_type  = prior_type;
model_spec.fixed_params.prior_strength = prior_strength;
model_spec.fixed_params.omega = 1;
model_spec.fixed_params.kappa_min = 1.0;
model_spec.fixed_params.rho_k = 0;
model_spec.fixed_params.phi_pref = 0;
model_spec.fixed_params.decision_beta = 1.0;

fixed_utility = struct('R_hit', 1, 'R_miss', 0, 'R_cr', 0.1, 'R_fa', -0.2);
fit_mode = 'conf_only';

% Stage 2 ground-truth / fit setup (matches v2)
choice_initial = [1.0, 0.0, 0.02, 0.02];
choice_lb  = [0,    -10, 0,    0   ];
choice_ub  = [50,    10, 0.45, 0.45];
choice_plb = [0.1, -3,  1e-3, 1e-3];
choice_pub = [10,   3,  0.2,  0.2 ];

%% ---------------- Bio-plausible bounds for random ground truth ----------------
% Stage 1 (vel-only). Tighter than the BADS optimizer bounds so synthetic
% animals stay in the regime real fits live in.
gen_bounds_stage1 = struct( ...
    'kappa_amp',     [3,    25  ], ...
    'c_power',       [0.5,  3.0 ], ...
    'd_power',       [0.005,0.05], ...
    'vel_slope',     [-3,  -0.3 ], ...
    'vel_intercept', [-1,   1   ], ...
    'vel_std',       [0.3,  1.2 ]);
if config.fit_licks
    gen_bounds_stage1.lick_slope     = [0.3, 3];
    gen_bounds_stage1.lick_intercept = [-1, 1];
    gen_bounds_stage1.lick_std       = [0.3, 1.2];
end
% Stage 2 ground truth bounds
gen_bounds_stage2 = struct( ...
    'alpha_r', [0.5, 5  ], ...
    'beta_r',  [-2,  2  ], ...
    'gamma_r', [0.005, 0.1], ...
    'delta_r', [0.005, 0.1]);

stage1_names = model_spec.fit_params;       % length 6 or 9
stage2_names = {'alpha_r','beta_r','gamma_r','delta_r'};

n_p1 = numel(stage1_names);
n_p2 = numel(stage2_names);

%% ---------------- Load IOResults for yoking ----------------
n_yoked_avail = 0;
yoked_stage1 = []; yoked_stage2 = [];
if ~isempty(io_results_path)
    if exist(io_results_path, 'file')
        L = load(io_results_path, 'IOResults');
        IOR = L.IOResults;
        n_yoked_avail = min(opts.n_yoked, numel(IOR.animals));
        yoked_stage1 = nan(n_yoked_avail, n_p1);
        yoked_stage2 = nan(n_yoked_avail, n_p2);
        for k = 1:n_yoked_avail
            ani = IOR.animals{k};
            % Stage 1
            yoked_stage1(k,:) = ani.fit.params_vec(1:n_p1);
            % Stage 2 (choice psychometric)
            if isfield(ani.fit, 'choice_vec') && numel(ani.fit.choice_vec) == 4
                yoked_stage2(k,:) = ani.fit.choice_vec(:)';
            else
                error('IOResults animal %d missing 4-element choice_vec', k);
            end
        end
        fprintf(' Loaded %d yoked param sets from %s\n', n_yoked_avail, io_results_path);
    else
        warning('parameter_recovery_v2:NoIOResults', ...
            'IOResults file not found: %s. Falling back to random-only.', io_results_path);
    end
end

n_random = opts.n_random;
n_animals = n_random + n_yoked_avail;
is_yoked = [false(n_random,1); true(n_yoked_avail,1)];

%% ---------------- Generate synthetic data ----------------
fprintf('\n--- Generating %d synthetic animals (%d trials each) ---\n', ...
    n_animals, opts.n_trials);

true_params_stage1 = nan(n_animals, n_p1);
true_params_stage2 = nan(n_animals, n_p2);
synth_data = cell(n_animals, 1);

for i = 1:n_animals
    if is_yoked(i)
        i_yoke = i - n_random;
        p1 = yoked_stage1(i_yoke,:);
        p2 = yoked_stage2(i_yoke,:);
    else
        p1 = sample_from_bounds(gen_bounds_stage1, stage1_names);
        p2 = sample_from_bounds(gen_bounds_stage2, stage2_names);
    end
    true_params_stage1(i,:) = p1;
    true_params_stage2(i,:) = p2;

    % Build a per-trial template
    oris  = randsample([0, 15, 30, 40, 45, 50, 60, 75, 90], opts.n_trials, true);
    cons  = randsample([0.01, 0.25, 0.5, 1.0],              opts.n_trials, true);
    disps = randsample([0, 30, 45, 90],                     opts.n_trials, true);

    template.orientation = oris(:);
    template.contrast    = cons(:);
    template.dispersion  = disps(:);
    template.n_trials    = opts.n_trials;

    synth_data{i} = simulate_v2_dataset(template, p1, p2, model_spec, fixed_utility);
    synth_data{i}.animal_tag = sprintf('Sim_%d', i);
end
fprintf(' Synthetic generation complete.\n');

%% ---------------- Stage 1 fit ----------------
fprintf('\n--- Stage 1 (conf_only) per-animal fits ---\n');
bads_options = bads('defaults'); bads_options.Display = 'off';

% Build per-animal cvpartition (shared with Stage 2)
all_cvpartitions = cell(n_animals,1);
for i = 1:n_animals
    all_cvpartitions{i} = cvpartition(synth_data{i}.n_trials, 'KFold', 5);
end

fitted_params_stage1 = nan(n_animals, n_p1);
stage1_test_nll = nan(n_animals, 1);
stage1_cv_indices = cell(n_animals, 1);

for i = 1:n_animals
    fprintf(' Animal %d/%d (yoked=%d)... ', i, n_animals, is_yoked(i));
    tic;
    out = fit_model_crossval(initial_guesses, synth_data{i}, model_spec, ...
        fixed_utility, fit_mode, all_cvpartitions{i});
    fitted_params_stage1(i,:) = out.params;
    stage1_test_nll(i) = out.avg_test_nll;
    stage1_cv_indices{i} = out.cv_indices;
    fprintf('done in %.1fs (NLL=%.2f)\n', toc, out.avg_test_nll);
end

%% ---------------- Stage 2 fit (choice psychometric) ----------------
fprintf('\n--- Stage 2 (choice psychometric) per-animal fits ---\n');
fitted_params_stage2 = nan(n_animals, n_p2);
stage2_cv_nll = nan(n_animals, 1);
stage2_alldata_nll = nan(n_animals, 1);

for i = 1:n_animals
    fprintf(' Animal %d/%d... ', i, n_animals);
    tic;
    % Build a "full_params" struct from Stage 1 fit
    full_params_i = model_spec.fixed_params;
    for ip = 1:n_p1, full_params_i.(stage1_names{ip}) = fitted_params_stage1(i,ip); end

    precomp = precompute_choice_inputs(synth_data{i}, full_params_i, fixed_utility);

    cv_s2 = stage1_cv_indices{i};
    k_folds_s2 = cv_s2.NumTestSets;
    fold_choice_params = nan(k_folds_s2, 4);
    fold_test_nll      = nan(k_folds_s2, 1);

    for kk = 1:k_folds_s2
        tr_idx = find(cv_s2.training(kk));
        te_idx = find(cv_s2.test(kk));
        p0_k = choice_initial + (rand(size(choice_initial))-0.5).*(choice_pub-choice_plb)*0.1;
        p0_k = min(max(p0_k, choice_lb), choice_ub);
        obj_fun_k = @(pp) calculate_choice_NLL(pp, synth_data{i}, full_params_i, precomp, tr_idx);
        theta_k = bads(obj_fun_k, p0_k, choice_lb, choice_ub, choice_plb, choice_pub, [], bads_options);
        fold_choice_params(kk,:) = theta_k;
        fold_test_nll(kk) = calculate_choice_NLL(theta_k, synth_data{i}, full_params_i, precomp, te_idx);
    end

    [~, best_k] = min(fold_test_nll);
    p_init_all = fold_choice_params(best_k,:);
    obj_fun_all = @(pp) calculate_choice_NLL(pp, synth_data{i}, full_params_i, precomp);
    choice_fit  = bads(obj_fun_all, p_init_all, choice_lb, choice_ub, choice_plb, choice_pub, [], bads_options);
    choice_nll  = calculate_choice_NLL(choice_fit, synth_data{i}, full_params_i, precomp);

    fitted_params_stage2(i,:) = choice_fit;
    stage2_cv_nll(i)      = mean(fold_test_nll);
    stage2_alldata_nll(i) = choice_nll;
    fprintf('done in %.1fs (cv NLL=%.2f, alldata NLL=%.2f)\n', toc, mean(fold_test_nll), choice_nll);
end

%% ---------------- Inversion + KL(Q_true || Q_fit), KL(L_true || L_fit) ----------------
fprintf('\n--- Inversion (true vs fitted) + KL ---\n');

per_animal = cell(n_animals,1);
for i = 1:n_animals
    fprintf(' Animal %d/%d... ', i, n_animals);

    % True params struct
    true_params_struct = model_spec.fixed_params;
    for ip = 1:n_p1, true_params_struct.(stage1_names{ip}) = true_params_stage1(i,ip); end

    % Fit params struct
    fit_params_struct = model_spec.fixed_params;
    for ip = 1:n_p1, fit_params_struct.(stage1_names{ip}) = fitted_params_stage1(i,ip); end

    inv_true = invert_model_for_single_trial_uncertainty(synth_data{i}, true_params_struct, fixed_utility, model_spec);
    inv_fit  = invert_model_for_single_trial_uncertainty(synth_data{i}, fit_params_struct,  fixed_utility, model_spec);

    % Per-trial KL on the marginalised posterior Q(theta) and likelihood L(theta)
    eps_v = 1e-12;
    Pq = inv_true.post_s_marginal + eps_v; Pq = Pq ./ sum(Pq,2);
    Qq = inv_fit.post_s_marginal  + eps_v; Qq = Qq ./ sum(Qq,2);
    KL_Q = sum(Pq .* log(Pq ./ Qq), 2);

    Pl = inv_true.L_s_marginal + eps_v; Pl = Pl ./ sum(Pl,2);
    Ql = inv_fit.L_s_marginal  + eps_v; Ql = Ql ./ sum(Ql,2);
    KL_L = sum(Pl .* log(Pl ./ Ql), 2);

    per_animal{i} = struct( ...
        'is_yoked', is_yoked(i), ...
        'data',     synth_data{i}, ...
        'true_stage1', true_params_stage1(i,:), ...
        'true_stage2', true_params_stage2(i,:), ...
        'fit_stage1',  fitted_params_stage1(i,:), ...
        'fit_stage2',  fitted_params_stage2(i,:), ...
        'stage1_test_nll', stage1_test_nll(i), ...
        'stage2_cv_nll',   stage2_cv_nll(i), ...
        'stage2_alldata_nll', stage2_alldata_nll(i), ...
        'KL_Q', KL_Q, ...
        'KL_L', KL_L, ...
        'unc_true_perc', inv_true.perc_unc_marginal, ...
        'unc_fit_perc',  inv_fit.perc_unc_marginal, ...
        'unc_true_dec',  inv_true.dec_unc_marginal, ...
        'unc_fit_dec',   inv_fit.dec_unc_marginal);

    fprintf('mean KL(Q)=%.3f, KL(L)=%.3f\n', mean(KL_Q), mean(KL_L));
end

%% ---------------- Save ----------------
% Resolve to absolute path and ensure the directory exists. mkdir's silent
% failure modes (relative paths from a wrong cwd, missing intermediates)
% have eaten an entire run before — fail loudly here instead.
if ~java.io.File(output_path).isAbsolute()
    output_path = fullfile(pwd, output_path);
end
[out_dir, out_name, out_ext] = fileparts(output_path); %#ok<ASGLU>
if ~exist(out_dir, 'dir')
    [ok_mk, msg_mk] = mkdir(out_dir);
    assert(ok_mk, 'parameter_recovery_v2: failed to create output dir %s: %s', out_dir, msg_mk);
end
fprintf(' Resolved save path: %s\n', output_path);

% --- Last-resort safety net: if save fails for ANY reason, dump the
%     RecoveryResults to a tempdir before raising, so we don't lose
%     a multi-hour BADS run.
backup_path = fullfile(tempdir, sprintf('RecoveryResults_backup_%s.mat', datestr(now,'yyyymmdd_HHMMSS')));

RecoveryResults = struct();
RecoveryResults.meta.timestamp     = datestr(now);
RecoveryResults.meta.prior_type    = prior_type;
RecoveryResults.meta.prior_strength= prior_strength;
RecoveryResults.meta.n_random      = n_random;
RecoveryResults.meta.n_yoked       = n_yoked_avail;
RecoveryResults.meta.n_trials      = opts.n_trials;
RecoveryResults.meta.fit_licks     = config.fit_licks;
RecoveryResults.meta.io_results_path = io_results_path;
RecoveryResults.meta.stage1_names  = stage1_names;
RecoveryResults.meta.stage2_names  = stage2_names;
RecoveryResults.meta.gen_bounds_stage1 = gen_bounds_stage1;
RecoveryResults.meta.gen_bounds_stage2 = gen_bounds_stage2;

RecoveryResults.is_yoked            = is_yoked;
RecoveryResults.true_params_stage1  = true_params_stage1;
RecoveryResults.true_params_stage2  = true_params_stage2;
RecoveryResults.fitted_params_stage1= fitted_params_stage1;
RecoveryResults.fitted_params_stage2= fitted_params_stage2;
RecoveryResults.stage1_test_nll     = stage1_test_nll;
RecoveryResults.stage2_cv_nll       = stage2_cv_nll;
RecoveryResults.stage2_alldata_nll  = stage2_alldata_nll;
RecoveryResults.per_animal          = per_animal;

fprintf('\nSaving to %s\n', output_path);
try
    save(output_path, 'RecoveryResults', '-v7.3');
    fprintf('Done.\n');
catch ME_save
    % If anything goes wrong, dump the data to a backup so the BADS work
    % isn't lost, then re-throw.
    warning('parameter_recovery_v2:SaveFailed', ...
        'Save to %s failed: %s\n  Dumping backup to %s', ...
        output_path, ME_save.message, backup_path);
    save(backup_path, 'RecoveryResults', '-v7.3');
    rethrow(ME_save);
end

end  % function parameter_recovery_v2

%% =====================================================================
%% Helpers (DUPLICATED from ideal_observer_hierarchical_fitting_v2.m).
%% Keep in sync. See top of file for note.
%% =====================================================================

function p_vec = sample_from_bounds(bounds, names)
% Uniform sample within per-param bounds.
p_vec = nan(1, numel(names));
for k = 1:numel(names)
    b = bounds.(names{k});
    p_vec(k) = b(1) + (b(2)-b(1)) * rand;
end
end

function sim_data = simulate_v2_dataset(template, p1, p2, model_spec, utility)
% Generate one animal's data under the v2 generative process:
%   m ~ vonMises(2*s, kappa_gen)
%   vel ~ Normal(vel_slope*DV(m) + vel_intercept, vel_std)
%   choice ~ Bernoulli(gamma + (1-gamma-delta) * sigm(alpha*g(m) + beta))
% where DV(m), g(m) use the same prior the fitter will use.

stage1_names = model_spec.fit_params;
params = model_spec.fixed_params;
for k = 1:numel(stage1_names), params.(stage1_names{k}) = p1(k); end

s_rad = deg2rad(params.s_range_deg);
m_rad = deg2rad(params.m_range_deg);
prior = get_prior(s_rad, params, struct('orientation', template.orientation));
util  = get_utility_vectors(params.s_range_deg, utility);

is_go_stim  = params.s_range_deg < 45;
is_boundary = params.s_range_deg == 45;

cond_mat = [template.orientation, template.contrast, template.dispersion];
[u_conds, ~, c_idx] = unique(cond_mat, 'rows');
n_c = size(u_conds,1);

cond_pm   = zeros(n_c, numel(m_rad));   % p(m | s,c,d)
cond_dv   = zeros(n_c, numel(m_rad));   % DV(m)
cond_gm   = zeros(n_c, numel(m_rad));   % g(m)

for j = 1:n_c
    s_j = u_conds(j,1); c_j = u_conds(j,2); d_j = u_conds(j,3);
    k_gen = get_kappa_for_trial(s_j, c_j, d_j, params);

    p_m_s = pdfVonMises(m_rad, deg2rad(s_j), k_gen);
    cond_pm(j,:) = p_m_s ./ (sum(p_m_s) + eps);

    Lik = pdfVonMises(m_rad', s_rad, k_gen);
    Post = (Lik .* prior) ./ (sum(Lik .* prior, 2) + eps);

    eu_go   = Post * util.respond';
    eu_nogo = Post * util.no_respond';
    cond_dv(j,:) = (eu_go - eu_nogo)';

    p_go = sum(Post(:, is_go_stim), 2) + 0.5 * sum(Post(:, is_boundary), 2);
    p_go = max(eps, min(1-eps, p_go));
    cond_gm(j,:) = log(p_go ./ (1 - p_go))';
end

% Stage 2 params
alpha_r = p2(1); beta_r = p2(2); gamma_r = p2(3); delta_r = p2(4);

n_t = template.n_trials;
choices_out = zeros(n_t,1);
vel_out     = zeros(n_t,1);

for t = 1:n_t
    j = c_idx(t);
    % Sample one m from p(m|s)
    m_idx = sample_categorical(cond_pm(j,:));

    % Velocity emission
    mu_vel_t = params.vel_slope * cond_dv(j,m_idx) + params.vel_intercept;
    vel_out(t) = mu_vel_t + randn() * params.vel_std;

    % Choice via lapsed psychometric on g(m_idx)
    g_t = cond_gm(j,m_idx);
    sigm = 1 / (1 + exp(-(alpha_r * g_t + beta_r)));
    p_go_t = gamma_r + (1 - gamma_r - delta_r) * sigm;
    p_go_t = max(eps, min(1-eps, p_go_t));
    choices_out(t) = rand() < p_go_t;
end

sim_data = template;
sim_data.choices    = choices_out;
% IMPORTANT: do NOT z-score synthetic velocity. Real data is z-scored
% upstream because its raw units are arbitrary; the fit's velocity-emission
% parameters are then interpreted on the z-scored scale. For synthetic data
% we know the true (slope, intercept, std) on the generative scale —
% z-scoring per-animal divides each animal's params by its own std(vel),
% which depends on (kappa_amp, c_power, d_power) via DV variance. That
% rescaling is animal-specific and destroys cross-animal recoverability.
% The gen_bounds at the top of this file produce vel values on a
% z-score-like scale already.
sim_data.conf_vel   = vel_out;
sim_data.conf_licks = zeros(n_t,1);  % unused under fit_licks=false
% Wrap cell-array fields in extra braces so struct() doesn't create a
% struct array — MATLAB-ism.
trial_keys.session_name     = repmat({'sim'}, n_t, 1);
trial_keys.session_idx      = ones(n_t, 1);
trial_keys.trial_in_session = (1:n_t)';
sim_data.trial_keys = trial_keys;
end

function k = sample_categorical(p)
% Sample one index from a categorical distribution given by p (row).
p = p(:)' / (sum(p) + eps);
c = cumsum(p);
k = find(rand <= c, 1, 'first');
if isempty(k), k = numel(p); end
end

%% ---- The functions below are direct copies of v2's local helpers ----
%% (with v2's precise behaviour preserved).

function fit_output = fit_model_crossval(initial_guesses, data, model_spec, utility, fit_mode, cv_indices)
[lb, ub, plb, pub] = get_bads_bounds(model_spec.fit_params, initial_guesses);
bads_options = bads('defaults'); bads_options.Display = 'off';
k_folds = 5; n_obs = data.n_trials;
if nargin < 6 || isempty(cv_indices)
    cv_indices = cvpartition(n_obs, 'KFold', k_folds);
else
    assert(cv_indices.NumObservations == n_obs);
    k_folds = cv_indices.NumTestSets;
end

recovered_params_kfold = zeros(k_folds, length(initial_guesses));
test_nll_kfold = zeros(k_folds, 1);

for k = 1:cv_indices.NumTestSets
    train_idx = cv_indices.training(k);
    test_idx  = cv_indices.test(k);

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
fit_output.cv_indices = cv_indices;
end

function [lb, ub, plb, pub] = get_bads_bounds(param_names, initial_guesses)
n_params = length(param_names);
lb = zeros(1, n_params); ub = zeros(1, n_params);
for i = 1:n_params
    name = param_names{i};
    if strcmp(name, 'kappa_amp'), lb(i)=0; ub(i)=50;
    elseif contains(name, 'power'), lb(i)=0; ub(i)=5;
    elseif strcmp(name, 'decision_beta'), lb(i)=0.1; ub(i)=10;
    elseif contains(name, '_slope'), lb(i)=-20; ub(i)=20;
    elseif contains(name, '_intercept'), lb(i)=-5; ub(i)=5;
    elseif contains(name, '_std'), lb(i)=0.01; ub(i)=5;
    else, lb(i)=-inf; ub(i)=inf;
    end
end
plb = max(lb, initial_guesses - abs(initial_guesses)*0.5 - 0.1);
pub = min(ub, initial_guesses + abs(initial_guesses)*0.5 + 0.1);
end

function NLL = calculate_NLL(p_natural, data, model_spec, utility, data_idx, fit_mode)
params = model_spec.fixed_params;
for i = 1:length(p_natural)
    params.(model_spec.fit_params{i}) = p_natural(i);
end
fit_licks = isfield(model_spec, 'config') && isfield(model_spec.config, 'fit_licks') && model_spec.config.fit_licks;

fold.s = data.orientation(data_idx);
fold.c = data.contrast(data_idx);
fold.d = data.dispersion(data_idx);
fold.ch = data.choices(data_idx);
fold.licks = data.conf_licks(data_idx);
fold.vel = data.conf_vel(data_idx);

if isempty(fold.s), NLL = 0; return; end

condMat = [fold.s, fold.c, fold.d];
[G, ~, G_idx] = unique(condMat, 'rows');
n_conds = size(G, 1);

s_rad = deg2rad(params.s_range_deg);
m_rad = deg2rad(params.m_range_deg);
prior = get_prior(s_rad, params, data);
util  = get_utility_vectors(params.s_range_deg, utility);

total_NLL = 0;
for j = 1:n_conds
    s_val = G(j,1); c_val = G(j,2); d_val = G(j,3);
    k_gen = get_kappa_for_trial(s_val, c_val, d_val, params);

    Lik_inf = pdfVonMises(m_rad', s_rad, k_gen);
    Post_s_m = (Lik_inf .* prior) ./ (sum(Lik_inf .* prior, 2) + eps);

    eu_go = Post_s_m * util.respond';
    eu_nogo = Post_s_m * util.no_respond';
    dv = eu_go - eu_nogo;

    p_go_m = 1 ./ (1 + exp(-params.decision_beta * dv));
    mu_vel = params.vel_slope * dv + params.vel_intercept;
    if fit_licks
        mu_licks = params.lick_slope * dv + params.lick_intercept;
    end

    p_m_s = pdfVonMises(m_rad, deg2rad(s_val), k_gen);
    p_m_s = p_m_s ./ (sum(p_m_s) + eps);

    mask = (G_idx == j);
    obs_ch = fold.ch(mask);
    obs_vel = fold.vel(mask);
    n_curr = sum(mask);

    P_Go_Mat   = repmat(p_go_m', n_curr, 1);
    Mu_Vel_Mat = repmat(mu_vel', n_curr, 1);
    P_m_Mat    = repmat(p_m_s, n_curr, 1);

    if strcmp(fit_mode, 'conf_only')
        L_choice = 1;
    else
        L_choice = (P_Go_Mat .* obs_ch) + ((1 - P_Go_Mat) .* (1 - obs_ch));
    end

    L_vel = normpdf(repmat(obs_vel,1,numel(m_rad)), Mu_Vel_Mat, params.vel_std);

    if fit_licks
        obs_licks = fold.licks(mask);
        Mu_Licks_Mat = repmat(mu_licks', n_curr, 1);
        L_licks = normpdf(repmat(obs_licks,1,numel(m_rad)), Mu_Licks_Mat, params.lick_std);
    else
        L_licks = 1;
    end

    L_joint_m = L_choice .* L_licks .* L_vel;
    L_trial = sum(L_joint_m .* P_m_Mat, 2);

    total_NLL = total_NLL - sum(log(L_trial + eps));
end
NLL = total_NLL;
if isnan(NLL) || isinf(NLL), NLL = 1e10; end
end

function precomp = precompute_choice_inputs(data, params, utility)
m_rad = deg2rad(params.m_range_deg);
s_rad = deg2rad(params.s_range_deg);
s_deg = params.s_range_deg;
n_m = numel(m_rad);

prior_ps    = get_prior(s_rad, params, data);
utility_vec = get_utility_vectors(s_deg, utility);

is_go_stim  = s_deg < 45;
is_boundary = s_deg == 45;

cond_mat = [data.orientation, data.contrast, data.dispersion];
[G_unique, ~, G_idx] = unique(cond_mat, 'rows');
n_conds = size(G_unique, 1);

p_m_s_all = zeros(n_conds, n_m);
dv_all    = zeros(n_conds, n_m);
g_all     = zeros(n_conds, n_m);

for j = 1:n_conds
    s_j = G_unique(j,1); c_j = G_unique(j,2); d_j = G_unique(j,3);
    kappa_gen = get_kappa_for_trial(s_j, c_j, d_j, params);

    p_m_s = pdfVonMises(m_rad, deg2rad(s_j), kappa_gen);
    p_m_s_all(j,:) = p_m_s ./ (sum(p_m_s) + eps);

    Lik_inf = pdfVonMises(m_rad', s_rad, kappa_gen);
    Post_s_m = (Lik_inf .* prior_ps) ./ (sum(Lik_inf .* prior_ps, 2) + eps);

    eu_go   = Post_s_m * utility_vec.respond';
    eu_nogo = Post_s_m * utility_vec.no_respond';
    dv_all(j,:) = (eu_go - eu_nogo)';

    p_go = sum(Post_s_m(:, is_go_stim), 2) + 0.5 * sum(Post_s_m(:, is_boundary), 2);
    p_go = max(eps, min(1-eps, p_go));
    g_all(j,:) = log(p_go ./ (1 - p_go))';
end

precomp.p_m_s_all = p_m_s_all;
precomp.dv_all    = dv_all;
precomp.g_all     = g_all;
precomp.G_idx     = G_idx;
end

function nll = calculate_choice_NLL(p_choice, data, params, precomp, data_idx)
if nargin < 5 || isempty(data_idx)
    data_idx = 1:data.n_trials;
end
if islogical(data_idx), data_idx = find(data_idx); end

alpha_r = p_choice(1); beta_r = p_choice(2);
gamma_r = p_choice(3); delta_r = p_choice(4);

if gamma_r < 0 || delta_r < 0 || (gamma_r + delta_r) >= 1
    nll = 1e10; return;
end

vel_slope    = params.vel_slope;
vel_intercept= params.vel_intercept;
vel_std      = params.vel_std;

G_idx    = precomp.G_idx;
g_all    = precomp.g_all;
dv_all   = precomp.dv_all;
p_m_s_all= precomp.p_m_s_all;

sigm = 1 ./ (1 + exp(-(alpha_r * g_all + beta_r)));
p_go_given_m = gamma_r + (1 - gamma_r - delta_r) .* sigm;

nll = 0;
for ii = 1:numel(data_idx)
    i   = data_idx(ii);
    j   = G_idx(i);
    vel = data.conf_vel(i);
    ch  = data.choices(i);

    mu_vel = vel_slope * dv_all(j,:) + vel_intercept;
    L_vel  = normpdf(vel, mu_vel, vel_std);

    post_m_unnorm = L_vel .* p_m_s_all(j,:);
    Z = sum(post_m_unnorm);
    if ~isfinite(Z) || Z <= 0
        post_m = p_m_s_all(j,:);
    else
        post_m = post_m_unnorm / Z;
    end

    p_go = sum(p_go_given_m(j,:) .* post_m);
    p_go = max(eps, min(1-eps, p_go));

    if ch
        nll = nll - log(p_go);
    else
        nll = nll - log(1 - p_go);
    end
end

if ~isfinite(nll), nll = 1e10; end
end

function inferred_unc = invert_model_for_single_trial_uncertainty(data, params, utility, base_spec)
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
maps_L_s_given_m = cell(n_unique_conds, 1);

generative_kappas = zeros(n_unique_conds, 1);

is_go_stim = s_range_deg < 45;
is_boundary = s_range_deg == 45;

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
    var_s_map = sum(post_s_m .* (s_range_deg - mean_s_map).^2, 2);
    maps_perc_unc{j} = sqrt(var_s_map);

    eu_go = post_s_m * utility_vec.respond';
    eu_nogo = post_s_m * utility_vec.no_respond';
    maps_eu_go{j} = eu_go;
    maps_eu_nogo{j} = eu_nogo;

    p_stim_go = sum(post_s_m(:, is_go_stim), 2) + 0.5 * sum(post_s_m(:, is_boundary), 2);
    p_stim_nogo = 1 - p_stim_go;

    cat_entropy = -(p_stim_go .* log2(p_stim_go + eps) + p_stim_nogo .* log2(p_stim_nogo + eps));
    maps_dec_unc{j} = cat_entropy;

    dv = eu_go - eu_nogo;
    if isfield(params, 'lick_slope')
        maps_lik_behavior{j}.pred_lick = params.lick_slope * dv + params.lick_intercept;
    else
        maps_lik_behavior{j}.pred_lick = [];
    end
    maps_lik_behavior{j}.pred_vel  = params.vel_slope  * dv + params.vel_intercept;
end

n_trials = data.n_trials;

inferred_unc.perc_unc_marginal = nan(n_trials, 1);
inferred_unc.dec_unc_marginal  = nan(n_trials, 1);
inferred_unc.eu_go_marginal    = nan(n_trials, 1);
inferred_unc.eu_nogo_marginal  = nan(n_trials, 1);
inferred_unc.post_s_marginal   = nan(n_trials, n_s);
inferred_unc.L_s_marginal      = nan(n_trials, n_s);

inferred_unc.m_posteriors = nan(n_trials, n_m);
inferred_unc.perceptual_map = nan(n_trials, 1);
inferred_unc.decision_map = nan(n_trials, 1);
inferred_unc.L_s_given_map = nan(n_trials, n_s);
inferred_unc.post_s_given_map = nan(n_trials, n_s);

for i = 1:n_trials
    cond_idx = trial_to_cond_idx(i);
    s_true = data.orientation(i);

    kappa_gen = generative_kappas(cond_idx);
    prior_m = pdfVonMises(m_range_rad, deg2rad(s_true), kappa_gen);
    prior_m = prior_m / (sum(prior_m) + eps);

    beh_maps = maps_lik_behavior{cond_idx};
    L_v = normpdf(data.conf_vel(i), beh_maps.pred_vel, params.vel_std);
    if isfield(params, 'lick_slope') && ~isempty(beh_maps.pred_lick)
        L_l = normpdf(data.conf_licks(i), beh_maps.pred_lick, params.lick_std);
        post_m_unnorm = L_l' .* L_v' .* prior_m;
    else
        post_m_unnorm = L_v' .* prior_m;
    end

    sum_post = sum(post_m_unnorm);
    if sum_post == 0 || isnan(sum_post)
        post_m = prior_m;
    else
        post_m = post_m_unnorm / sum_post;
    end

    inferred_unc.m_posteriors(i, :) = post_m;

    post_s_m_i = maps_p_s_given_m{cond_idx};
    Q_marg = post_m * post_s_m_i;
    Q_marg = Q_marg ./ (sum(Q_marg) + eps);
    inferred_unc.post_s_marginal(i, :) = Q_marg;

    L_s_m_i = maps_L_s_given_m{cond_idx};
    L_marg = post_m * L_s_m_i;
    L_marg = L_marg ./ (sum(L_marg) + eps);
    inferred_unc.L_s_marginal(i, :) = L_marg;

    mean_s_marg = sum(Q_marg .* s_range_deg);
    var_s_marg = sum(Q_marg .* (s_range_deg - mean_s_marg).^2);
    inferred_unc.perc_unc_marginal(i) = sqrt(var_s_marg);

    p_go_marg  = sum(Q_marg(is_go_stim)) + 0.5 * sum(Q_marg(is_boundary));
    p_go_marg  = max(eps, min(1-eps, p_go_marg));
    p_ngo_marg = 1 - p_go_marg;
    inferred_unc.dec_unc_marginal(i) = -(p_go_marg * log2(p_go_marg) + p_ngo_marg * log2(p_ngo_marg));

    inferred_unc.eu_go_marginal(i)   = Q_marg * utility_vec.respond';
    inferred_unc.eu_nogo_marginal(i) = Q_marg * utility_vec.no_respond';

    [~, map_idx] = max(post_m);
    inferred_unc.perceptual_map(i) = maps_perc_unc{cond_idx}(map_idx);
    inferred_unc.decision_map(i)   = maps_dec_unc{cond_idx}(map_idx);
    inferred_unc.L_s_given_map(i,:) = maps_L_s_given_m{cond_idx}(map_idx,:);
    inferred_unc.post_s_given_map(i,:) = maps_p_s_given_m{cond_idx}(map_idx,:);
end
end

function kappa = get_kappa_for_trial(orientation, contrast, dispersion, params) %#ok<INUSL>
kappa_0 = params.kappa_min + params.kappa_amp;
kappa_contrast = (contrast.^params.c_power);
kappa_disp = exp(-params.d_power .* dispersion);
kappa = kappa_0 .* kappa_contrast .* kappa_disp;
end

function prior = get_prior(s_range_rad, params, data)
switch params.prior_type
    case 'Flat'
        prior = ones(1, length(s_range_rad));
    case 'Bimodal'
        c1 = pdfVonMises(s_range_rad, deg2rad(0),  params.prior_strength);
        c2 = pdfVonMises(s_range_rad, deg2rad(90), params.prior_strength);
        prior = c1 + c2;
    case 'Unimodal'
        prior = pdfVonMises(s_range_rad, deg2rad(params.prior_center_deg), params.prior_strength);
    case 'Empirical'
        s_range_deg = rad2deg(s_range_rad);
        counts = histcounts(data.orientation, [s_range_deg s_range_deg(end)+1]);
        prior = smooth(counts, 3)';
    otherwise
        error('Unknown prior_type: %s', params.prior_type);
end
prior = prior / (sum(prior) + eps);
end

function utility_vec = get_utility_vectors(s_range, utility)
n_s_bins = length(s_range);
utility_vec.respond = zeros(1, n_s_bins);
utility_vec.no_respond = zeros(1, n_s_bins);
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
