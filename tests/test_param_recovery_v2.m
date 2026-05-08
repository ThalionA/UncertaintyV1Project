function test_param_recovery_v2(varargin)
%% test_param_recovery_v2  Integration tests for parameter_recovery_v2.
%
% Usage:
%   test_param_recovery_v2                       % default: 'smoke' mode
%   test_param_recovery_v2('mode','full')        % strict TDD criterion
%   test_param_recovery_v2('skip_recovery',true) % only the prior-shape unit test
%
% Modes:
%   'smoke'  4 random animals, 1000 trials, r_threshold=0.50, bias 30% of range.
%            Roughly 5–10 minutes of BADS. Catches gross structural bugs.
%   'full'   12 animals, 2000 trials, r_threshold=0.85, bias 20% of range.
%            Roughly 30–60 minutes of BADS. Locks in production-grade recovery.

p = inputParser;
p.addParameter('mode', 'smoke', @ischar);
p.addParameter('skip_recovery', false, @islogical);
p.parse(varargin{:});
mode = p.Results.mode;
skip_recovery = p.Results.skip_recovery;

% Make sure parameter_recovery_v2 is on the path
this_dir = fileparts(mfilename('fullpath'));
io_dir   = fullfile(fileparts(this_dir), 'ideal_observer');
addpath(io_dir);

n_fail = 0;

% Test 1: prior shape (unit test, no fitting)
fprintf('\n================================================================\n');
fprintf(' Test 1: prior shape — Flat vs Bimodal{k=5}\n');
fprintf('================================================================\n');

s_deg = 0:1:90;
s_rad = deg2rad(s_deg);

% Flat
prior_flat = ones(1, length(s_rad));
prior_flat = prior_flat / sum(prior_flat);

% Bimodal{k=5} — uses orientation-doubled von Mises (same convention as v2)
k_gen = 5;
c1 = exp(k_gen .* cos(2*s_rad - 2*deg2rad(0)))  / (2*pi*besseli(0, k_gen));
c2 = exp(k_gen .* cos(2*s_rad - 2*deg2rad(90))) / (2*pi*besseli(0, k_gen));
prior_bk5 = (c1 + c2) / sum(c1 + c2);

ratio_flat = prior_flat(s_deg==0) / prior_flat(s_deg==45);
ratio_bk5  = prior_bk5(s_deg==0)  / prior_bk5(s_deg==45);

[ok_flat, n_fail] = check_(abs(ratio_flat - 1.0) < 0.01, ...
    sprintf('Flat prior uniform (P[0]/P[45] = %.3f)', ratio_flat), n_fail);
[ok_bk5,  n_fail] = check_(ratio_bk5 > 5.0, ...
    sprintf('Bimodal{k=5} sharply peaked at 0/90 (P[0]/P[45] = %.2f)', ratio_bk5), n_fail);

% Test 2: parameter recovery integration (Bimodal{k=3})
if skip_recovery
    fprintf('\nSkipping Test 2 (skip_recovery=true)\n');
    finalise(n_fail);
    return;
end

switch mode
    case 'smoke'
        n_random = 4;  n_trials = 1000;
        r_threshold = 0.50; bias_rel_threshold = 0.30;
    case 'full'
        n_random = 12; n_trials = 2000;
        r_threshold = 0.85; bias_rel_threshold = 0.20;
    otherwise
        error('Unknown mode: %s', mode);
end

fprintf('\n================================================================\n');
fprintf(' Test 2: parameter recovery (mode=%s)\n', mode);
fprintf('   n_random=%d, n_trials=%d, r_threshold=%.2f, bias_threshold=%.2f\n', ...
    n_random, n_trials, r_threshold, bias_rel_threshold);
fprintf('================================================================\n');

out_dir = tempname;
mkdir(out_dir);
out_path = fullfile(out_dir, 'recov_test.mat');

parameter_recovery_v2('Bimodal', 3, out_path, '', ...
    'n_random', n_random, 'n_yoked', 0, 'n_trials', n_trials, ...
    'rng_seed', 42);

L = load(out_path, 'RecoveryResults');
R = L.RecoveryResults;

fprintf('\n--- Stage 1 recovery ---\n');
stage1_names = R.meta.stage1_names;
n_stage1_pass = 0;
for k = 1:numel(stage1_names)
    x = R.true_params_stage1(:,k);
    y = R.fitted_params_stage1(:,k);
    r = corr(x, y);
    bias = mean(y - x);
    range_x = max(x) - min(x);
    rel_bias = bias / max(abs(range_x), eps);
    pass = (r >= r_threshold) && (abs(rel_bias) < bias_rel_threshold);
    n_stage1_pass = n_stage1_pass + pass;
    fprintf('  %-15s r=%+.2f  bias=%+0.3f  rel_bias=%+0.2f  [%s]\n', ...
        stage1_names{k}, r, bias, rel_bias, pass_label(pass));
end

% Pass criterion: at least 2/3 of Stage 1 params clear the bar
n_required = ceil(0.66 * numel(stage1_names));
[ok_stage1, n_fail] = check_(n_stage1_pass >= n_required, ...
    sprintf('Stage 1 params recovered: %d/%d (need >= %d)', n_stage1_pass, numel(stage1_names), n_required), n_fail);

fprintf('\n--- Stage 2 recovery (diagnostic only) ---\n');
% Stage 2 diagnostic — recovery is harder for the lapse params; report but
% don't fail on it at the smoke threshold.
stage2_names = R.meta.stage2_names;
for k = 1:numel(stage2_names)
    x = R.true_params_stage2(:,k);
    y = R.fitted_params_stage2(:,k);
    r = corr(x, y);
    bias = mean(y - x);
    fprintf('  %-15s r=%+.2f  bias=%+0.3f\n', stage2_names{k}, r, bias);
end

% In 'full' mode, also require alpha_r recovery (the slope is the most
% identifiable Stage 2 parameter)
if strcmp(mode, 'full')
    x_a = R.true_params_stage2(:,1);
    y_a = R.fitted_params_stage2(:,1);
    r_alpha = corr(x_a, y_a);
    [ok_alpha, n_fail] = check_(r_alpha >= 0.6, ...
        sprintf('alpha_r recovery r=%+.2f (need >= 0.60)', r_alpha), n_fail);
end

% Cleanup
try, rmdir(out_dir, 's'); end %#ok<TRYNC>

finalise(n_fail);
end

function lbl = pass_label(pass)
if pass, lbl = 'PASS'; else, lbl = 'FAIL'; end
end

function [ok, n_fail] = check_(cond, msg, n_fail)
ok = logical(cond);
if ok
    fprintf('  [PASS] %s\n', msg);
else
    fprintf('  [FAIL] %s\n', msg);
    n_fail = n_fail + 1;
end
end

function finalise(n_fail)
fprintf('\n================================================================\n');
if n_fail == 0
    fprintf(' ALL TESTS PASSED\n');
else
    fprintf(' %d test(s) FAILED\n', n_fail);
end
fprintf('================================================================\n');
if n_fail > 0
    error('test_param_recovery_v2: %d failures', n_fail);
end
end
