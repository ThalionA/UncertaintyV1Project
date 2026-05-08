%% run_real_prior_sweep.m
%
% Runs ideal_observer_hierarchical_fitting_v2.m four times, each with a
% different fixed prior. Saves results under
% ideal_observer/io_prior_sweep/IOResults_<tag>.mat.
%
% Drives v2 via these caller-workspace overrides:
%   - prior_type_override
%   - prior_strength_override
%   - iores_filename
% v2 reads them in its FIXED PARAMETERS / save sections.

clear; close all; clc;

% Run from the repo root (or the ideal_observer/ directory).
this_dir = fileparts(mfilename('fullpath'));
io_dir   = fileparts(this_dir);  % ideal_observer/

prior_configs = { ...
    struct('type','Flat',    'strength', NaN, 'tag', 'Flat'), ...
    struct('type','Bimodal', 'strength', 1,   'tag', 'Bimodal_k1'), ...
    struct('type','Bimodal', 'strength', 3,   'tag', 'Bimodal_k3'), ...
    struct('type','Bimodal', 'strength', 5,   'tag', 'Bimodal_k5')};

for i_p = 1:numel(prior_configs)
    pc = prior_configs{i_p};
    fprintf('\n================================================================\n');
    fprintf(' [%d/%d] Real-data v2 fit | prior=%s, strength=%g\n', ...
        i_p, numel(prior_configs), pc.type, pc.strength);
    fprintf('================================================================\n');

    % v2 reads these from caller workspace
    prior_type_override     = pc.type;
    prior_strength_override = pc.strength;
    iores_filename = fullfile(this_dir, sprintf('IOResults_%s.mat', pc.tag));

    % v2 reuses an existing IOResults var if present — clear it so each prior
    % starts fresh
    clear IOResults;

    % Run v2 in this workspace
    run(fullfile(io_dir, 'ideal_observer_hierarchical_fitting_v2.m'));
end

fprintf('\nAll %d real-data prior fits complete.\n', numel(prior_configs));
