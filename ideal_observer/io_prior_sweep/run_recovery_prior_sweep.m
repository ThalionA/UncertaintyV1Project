%% run_recovery_prior_sweep.m
%
% Runs parameter_recovery_v2 four times, each with a different fixed prior
% matched to the same prior used at fit time ("match generative + fit").
% For each prior, empirically-yoked animals are seeded from the matching
% IOResults_<tag>.mat (so run_real_prior_sweep.m must finish first).
%
% Saves to ideal_observer/io_prior_sweep/RecoveryResults_<tag>.mat.

clear; close all; clc;

this_dir = fileparts(mfilename('fullpath'));
io_dir   = fileparts(this_dir);
addpath(io_dir);  % so MATLAB can find parameter_recovery_v2

prior_configs = { ...
    struct('type','Flat',    'strength', NaN, 'tag', 'Flat'), ...
    struct('type','Bimodal', 'strength', 1,   'tag', 'Bimodal_k1'), ...
    struct('type','Bimodal', 'strength', 3,   'tag', 'Bimodal_k3'), ...
    struct('type','Bimodal', 'strength', 5,   'tag', 'Bimodal_k5')};

n_random = 20;
n_yoked  = 6;
n_trials = 2000;

for i_p = 1:numel(prior_configs)
    pc = prior_configs{i_p};
    fprintf('\n================================================================\n');
    fprintf(' [%d/%d] Recovery | prior=%s, strength=%g\n', ...
        i_p, numel(prior_configs), pc.type, pc.strength);
    fprintf('================================================================\n');

    iores_path = fullfile(this_dir, sprintf('IOResults_%s.mat', pc.tag));
    if ~exist(iores_path, 'file')
        warning('run_recovery_prior_sweep:NoIOResults', ...
            'IOResults file not found: %s. Recovery will fall back to random-only.', iores_path);
        iores_path = '';
    end

    out_path = fullfile(this_dir, sprintf('RecoveryResults_%s.mat', pc.tag));
    parameter_recovery_v2(pc.type, pc.strength, out_path, iores_path, ...
        'n_random', n_random, 'n_yoked', n_yoked, 'n_trials', n_trials, ...
        'fit_licks', false);
end

fprintf('\nAll %d recovery prior runs complete.\n', numel(prior_configs));
