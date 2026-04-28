%% Refresh Ideal-Observer Outputs in VR_Decoder_Data_Export.mat
%
% PURPOSE:
%   The neural side of the pipeline (NeuralStore, neural metrics) is expensive
%   and never changes once computed. The ideal-observer fit, however, is
%   re-run frequently (e.g., when changing the spatial window in
%   ideal_observer_hierarchical_fitting_v2.m). This script swaps in a fresh
%   IOResults.mat without re-running the neural-side pipeline.
%
% WHAT IT DOES:
%   1. Loads the existing VR_Decoder_Data_Export.mat (NeuralStore preserved).
%   2. Loads a fresh IOResults.mat.
%   3. Drops old IO-derived columns from TrialTbl_Struct.
%   4. Re-runs stack_IO_trial_tables and decision-posterior derivation
%      (identical logic to VR_multi_animal_analysis.m, Parts joining + Part 8).
%   5. Saves the export back, replacing IO and the IO-derived columns.
%
% USAGE:
%   - Edit the paths in the Configuration block to match your machine.
%   - Run.
%
% NOTE:
%   The IO-related column list MUST stay in sync with VR_multi_animal_analysis.m.
%   If you add/remove IO columns there, mirror the change in `ioCols` and
%   `derived_cols` below.

clear; close all; clc;
warning off;

%% --- Configuration ---------------------------------------------------------
% Resolve the UncertaintyV1 project root from this script's location:
%   <project>/preprocessing/refresh_IO_in_export.m  -> <project>
this_dir    = fileparts('/Users/theoamvr/Desktop/Experiments/UncertaintyV1/preprocessing/refresh_IO_in_export.m');
project_dir = fileparts(this_dir);
data_dir    = fullfile(project_dir, 'data');

export_path = fullfile(data_dir, 'VR_Decoder_Data_Export.mat');
io_path     = fullfile(data_dir, 'IOResults.mat');
make_backup = true;   % copy export to *_pre_refresh.bak.mat before overwrite

% IO columns joined into TrialTbl_all in VR_multi_animal_analysis.m
ioCols = {'unc_perceptual', 'unc_decision', ...
          'post_s_given_map', 'post_s_marginal', ...
          'L_s_given_map', 'L_s_marginal'};

% Columns derived in Part 8 of VR_multi_animal_analysis.m
derived_cols = {'decision_posterior_map', ...
                'decision_posterior_marginal', ...
                'decision_posterior'};

%% --- Load existing export & fresh IO --------------------------------------
fprintf('Loading existing export: %s\n', export_path);
E = load(export_path, 'NeuralStore', 'TrialTbl_Struct', 'IO');

fprintf('Loading fresh IOResults : %s\n', io_path);
S = load(io_path, 'IOResults');
IO_new = S.IOResults;

NeuralStore  = E.NeuralStore;                       % preserved untouched
TrialTbl_all = struct2table(E.TrialTbl_Struct);
fprintf('Existing TrialTbl: %d trials, %d columns.\n', ...
    height(TrialTbl_all), width(TrialTbl_all));

%% --- Build fresh IO table --------------------------------------------------
T_new = stack_IO_trial_tables(IO_new);
fprintf('Fresh IO table: %d rows.\n', height(T_new));

%% --- Drop old IO-derived columns ------------------------------------------
to_drop = intersect([ioCols, derived_cols], TrialTbl_all.Properties.VariableNames);
if ~isempty(to_drop)
    TrialTbl_all = removevars(TrialTbl_all, to_drop);
    fprintf('Dropped %d stale IO column(s): %s\n', numel(to_drop), strjoin(to_drop, ', '));
end

%% --- Attach fresh IO columns (preserve trial alignment) -------------------
if height(TrialTbl_all) == height(T_new)
    TrialTbl_all = [TrialTbl_all, T_new(:, ioCols)];
    fprintf('Row counts match - direct horizontal concat used.\n');
else
    warning(['Row mismatch: existing TrialTbl (%d) vs fresh IO table (%d). ', ...
             'Falling back to innerjoin on (animal, session, trial_in_session).'], ...
            height(TrialTbl_all), height(T_new));

    % Build a numeric "session" index per animal in the fresh IO table that
    % mirrors the per-animal numbering used in VR_multi_animal_analysis.m
    % (1..N in original session order). This lets us join against the
    % existing TrialTbl_all which already has 'animal','session','trial'.
    T_new.animal = string(T_new.animal);
    T_new.session = zeros(height(T_new), 1);
    animals_unique = unique(T_new.animal, 'stable');
    for ia = 1:numel(animals_unique)
        m = T_new.animal == animals_unique(ia);
        [~, ~, sIdx] = unique(T_new.session_name(m), 'stable');
        T_new.session(m) = sIdx;
    end
    T_new.trial = double(T_new.trial_in_session);

    % Match types of join keys
    if ~iscell(TrialTbl_all.animal)
        TrialTbl_all.animal = cellstr(string(TrialTbl_all.animal));
    end
    T_new.animal = cellstr(T_new.animal);

    keyVars = {'animal', 'session', 'trial'};
    TrialTbl_all = innerjoin(TrialTbl_all, T_new(:, [keyVars, ioCols]), ...
        'Keys', keyVars);
    fprintf('Innerjoin complete: %d trials retained.\n', height(TrialTbl_all));
end

%% --- Recompute derived decision-posterior columns -------------------------
% (Mirrors Part 8 of VR_multi_animal_analysis.m exactly.)
post_s_map = TrialTbl_all.post_s_given_map;
p_lt = sum(post_s_map(:, 1:45), 2, 'omitnan');
p_gt = sum(post_s_map(:, 47:91), 2, 'omitnan');
p_eq = post_s_map(:, 46);
TrialTbl_all.decision_posterior_map = [p_lt + p_eq/2, p_gt + p_eq/2];

post_s_marg = TrialTbl_all.post_s_marginal;
p_lt = sum(post_s_marg(:, 1:45), 2, 'omitnan');
p_gt = sum(post_s_marg(:, 47:91), 2, 'omitnan');
p_eq = post_s_marg(:, 46);
TrialTbl_all.decision_posterior_marginal = [p_lt + p_eq/2, p_gt + p_eq/2];

% Default decision_posterior follows the marginalized version
TrialTbl_all.decision_posterior = TrialTbl_all.decision_posterior_marginal;

%% --- Save ------------------------------------------------------------------
TrialTbl_Struct = table2struct(TrialTbl_all, 'ToScalar', true);
IO = IO_new;

if make_backup && exist(export_path, 'file')
    [p, n, ~] = fileparts(export_path);
    bak = fullfile(p, [n '_pre_refresh.bak.mat']);
    copyfile(export_path, bak);
    fprintf('Backup written: %s\n', bak);
end

save(export_path, 'NeuralStore', 'TrialTbl_Struct', 'IO', '-v7');
fprintf('Refreshed export saved: %s\n', export_path);
fprintf('NeuralStore preserved unchanged. IO columns refreshed.\n');

%% Helper: identical to stack_IO_trial_tables in VR_multi_animal_analysis.m
function T = stack_IO_trial_tables(IO)
    canon = {'orientation','contrast','dispersion', ...
             'choice','conf_empirical', ...
             'p_respond_model','conf_model', ...
             'unc_perceptual','unc_decision', ...
             'session_name','trial_in_session','animal'};
    canonTypes = {'double','double','double', ...
                  'double','double', ...
                  'double','double', ...
                  'double','double', ...
                  'string','double','string'};

    T = table();
    for k = 1:numel(IO.animals)
        [~, session_starts] = unique(IO.animals{k}.trial_table.session_name);
        last_session_trials = [session_starts(2:end) - 1; size(IO.animals{k}.trial_table, 1)];
        trials_to_keep = true(size(IO.animals{k}.trial_table, 1), 1);
        trials_to_keep(last_session_trials) = false;

        tmp = IO.animals{k}.trial_table(trials_to_keep, :);

        if isfield(IO.animals{k}, 'inferred') && isfield(IO.animals{k}.inferred, 'post_s_given_map')
            full_post = IO.animals{k}.inferred.post_s_given_map;
            tmp.post_s_given_map = full_post(trials_to_keep, :);
        else
            tmp.post_s_given_map = nan(height(tmp), 91);
        end

        if isfield(IO.animals{k}, 'inferred') && isfield(IO.animals{k}.inferred, 'post_s_marginal')
            full_post = IO.animals{k}.inferred.post_s_marginal;
            tmp.post_s_marginal = full_post(trials_to_keep, :);
        else
            tmp.post_s_marginal = nan(height(tmp), 91);
        end

        if isfield(IO.animals{k}, 'inferred') && isfield(IO.animals{k}.inferred, 'L_s_given_map')
            full_lik = IO.animals{k}.inferred.L_s_given_map;
            tmp.L_s_given_map = full_lik(trials_to_keep, :);
        else
            tmp.L_s_given_map = nan(height(tmp), 91);
        end

        if isfield(IO.animals{k}, 'inferred') && isfield(IO.animals{k}.inferred, 'L_s_marginal')
            full_lik_marg = IO.animals{k}.inferred.L_s_marginal;
            tmp.L_s_marginal = full_lik_marg(trials_to_keep, :);
        else
            tmp.L_s_marginal = nan(height(tmp), 91);
        end

        if ~ismember('session_name', tmp.Properties.VariableNames)
            tmp.session_name = string(IO.animals{k}.data.trial_keys.session_name(trials_to_keep));
            tmp.trial_in_session = double(IO.animals{k}.data.trial_keys.trial_in_session(trials_to_keep));
        else
            tmp.session_name = string(tmp.session_name);
        end

        tmp.animal = repmat(string(IO.animals{k}.tag), height(tmp), 1);

        for i = 1:numel(canon)
            vn = canon{i};
            if ~ismember(vn, tmp.Properties.VariableNames)
                if strcmp(canonTypes{i}, 'string'), tmp.(vn) = strings(height(tmp),1);
                else, tmp.(vn) = nan(height(tmp),1); end
            end
        end

        T = [T; tmp]; %#ok<AGROW>
    end
end
