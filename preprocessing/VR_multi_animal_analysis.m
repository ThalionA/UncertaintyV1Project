%% Multi-animal VR Analysis & Ideal Observer Modeling Pipeline (Integrated)
%
% What this does:
% 1) Loads per-trial unified data across animals/sessions (neural + behavior).
% 2) Matches ROIs within each animal (if multiple sessions; else uses all ROIs).
% 3) Computes per-trial neural metrics:
%       - Mean activity (inferred spikes) during grating and corridor
%       - Generalized variance: logdet(Cov + lambda*I)
%       - Normalized GV: GV / ||centroid||
% 4) Computes behavioral metrics:
%       - Go choice (hit/FA) proportion
%       - Pre-RZ velocity and pre-RZ licks
%       - Performance (correct vs incorrect)
% 5) Plots (per-animal and group):
%       - Psychometric curves (+ 4-parameter sigmoid fits) for goChoice,
%         preRZ_velocity, preRZ_licks, binned by orientation (optionally
%         by contrast/dispersion).
%       - Behavior–performance correlations.
%       - Neural mean activity vs stimulus; GV; normalized GV.
% 6) Fits a hierarchical ideal observer model with BADS (group then individual),
%    falls back to fmincon if BADS not available.
% 7) Inverts the model to infer trial-by-trial perceptual/decision uncertainty,
%    appends to the main table, and plots neural metrics vs inferred uncertainty.
%
% Required: createUnifiedSessionData.m,
% Optional (if multi-session per animal): multiSessionROImatching_cached.m
%

%% --- Cleanup & Setup ---
clear; close all; clc;
warning off;
% set(0, 'DefaultFigureColor', 'w');
% rng(0); % reproducibility

%% Part 1: Configuration

% --- Paths ---
working_dir  = 'C:\Users\theox\Desktop\Experiments\Working';
addpath(genpath(working_dir));
% addpath(genpath('C:\path\to\bads-master')); % optional; else falls back

% --- Animals & Sessions ---
% You can put single sessions here (works fine) or expand each cell array
% to many sessions per animal; the code handles both.
Animals = struct( ...
    'tag',    {'Cb15','Cb17','Cb21', 'Cb22', 'Cb24','Cb25'}, ...
    'sesList',{{'20250605_Cb15', '20250613_Cb15', '20250620_Cb15', '20250624_Cb15', '20250709_Cb15'},...
               {'20250606_Cb17', '20250613_Cb17', '20250620_Cb17', '20250624_Cb17', '20250701_Cb17'},...
               {'20250904_Cb21', '20250910_Cb21', '20250911_Cb21', '20250912_Cb21', '20250918_Cb21'},...
               {'20251024_Cb22', '20251027_Cb22', '20251028_Cb22', '20251030_Cb22', '20251105_Cb22'},...
               {'20250918_Cb24', '20250919_Cb24', '20251020_Cb24', '20251021_Cb24'},...
               {'20250903_Cb25', '20250904_Cb25', '20250910_Cb25', '20250911_Cb25', '20250916_Cb25'}} ...
);

% --- General Options ---
saveCSV              = false;       % write final integrated table
defaultRecID         = 'A_0_1';     % default rec ID for createUnifiedSessionData
altRecMap            = containers.Map({'20250904_Cb21', '20250910_Cb21'}, {'A_1_1', 'A_1_1'}); % special cases

% --- Neural Analysis Options ---
timeWindowG          = [0 2];       % seconds into grating
avgCorridorAll       = true;        % true: average across all corridor bins
posWindowC           = [];          % used only if avgCorridorAll=false, e.g., [0 120]
minNeuronSpkThr      = 0.1;         % neuron keep-threshold (max spike prob)
contrastMergeTol     = 0.02;        % merge near-identical contrasts
gvLambda             = 1e-0;        % regularizer for GV

% --- Init ---
nAnimals        = numel(Animals);
TrialTbl_all    = table();
AnimalSummaries = struct([]);
all_data_for_model = cell(1, nAnimals);    % for modeling
semFun = @(x) sem_omit(x);

% --- NEW stores to keep everything ---
PerAnimalRaw = struct([]);     % raw UD per animal (for pupil/anything later)
NeuralStore  = struct([]);     % tensors per animal (kept neurons only)


%% Part 2: Per-Animal Load, ROI Matching, Metrics, & Model Data
fprintf('\n=== Part 2: Data Loading & Metrics ===\n');
for a = 1:nAnimals
    tag = Animals(a).tag; ses = Animals(a).sesList(:)';
    fprintf('\n--- %s | %d session(s) ---\n', tag, numel(ses));
    
    % Load unified sessions
    allUD = cell(numel(ses),1);
    for s = 1:numel(ses)
        sesname = ses{s};
        recID = defaultRecID;
        if isKey(altRecMap, sesname), recID = altRecMap(sesname); end
        fprintf('  Loading %s (rec=%s)... ', sesname, recID);
        allUD{s} = createUnifiedSessionData(sesname, working_dir, recID, [], 5, 0.05);
        fprintf('ok\n');
    end
    
    % ROI matching within animal (or use all if single session)
    if isscalar(ses)
        nNeurons1 = size(allUD{1}(1).grating.neural_spikes, 1);
        commonIDs = {1:nNeurons1};
        fprintf('  [%s] Single session: using all %d ROIs.\n', tag, nNeurons1);
    else
        if exist('multiSessionROImatching_cached','file')==2
            commonIDs = multiSessionROImatching_cached(ses, working_dir);
        else
            warning('multiSessionROImatching_cached not found; using intersection by index length.');
            minN = inf;
            for s = 1:numel(ses)
                minN = min(minN, size(allUD{s}(1).grating.neural_spikes,1));
            end
            commonIDs = repmat({1:minN}, 1, numel(ses));
        end
    end
    
    % Subset to matched ROIs and tag meta
    for s = 1:numel(ses)
        ids = commonIDs{s};
        for t = 1:numel(allUD{s})
            % Spikes
            allUD{s}(t).grating.neural_spikes = allUD{s}(t).grating.neural_spikes(ids,:);
            if isfield(allUD{s}(t).corridor,'neural_spikes') && ~isempty(allUD{s}(t).corridor.neural_spikes)
                allUD{s}(t).corridor.neural_spikes = allUD{s}(t).corridor.neural_spikes(ids,:);
            end
            % dF/F
            allUD{s}(t).grating.neural_dF = allUD{s}(t).grating.neural_dF(ids,:);
            if isfield(allUD{s}(t).corridor,'neural_dF') && ~isempty(allUD{s}(t).corridor.neural_dF)
                allUD{s}(t).corridor.neural_dF = allUD{s}(t).corridor.neural_dF(ids,:);
            end
            
            allUD{s}(t).session = s;
            allUD{s}(t).animal  = tag;
        end
    end
    UD = [allUD{:}];
    
    % store raw concatenated trials per animal for later use (e.g., pupil)
    PerAnimalRaw(a).tag = tag;
    PerAnimalRaw(a).UD  = UD;                 % keeps original trial structs intact
    
    if isempty(UD), warning('%s: no trials after load.', tag); continue; end
    
    % Axes
    xG = UD(1).grating.timeBins;
    if isfield(UD(1).corridor,'binCenters')
        xC = UD(1).corridor.binCenters;
    else
        xC = [];
    end
    
    % Spike tensors
    [Gspk, Cspk, nN0] = build_spike_tensors(UD);  % also returns pre-filter neuron count
    
    % Build dF/F Tensors manually directly from UD
    nTrials = numel(UD);
    nBinsG  = length(xG);
    GdF = nan(nTrials, nN0, nBinsG);
    if ~isempty(xC)
        nBinsC = length(xC);
        CdF = nan(nTrials, nN0, nBinsC);
    else
        CdF = [];
    end
    
    for t = 1:nTrials
        if ~isempty(UD(t).grating.neural_dF)
            GdF(t, :, :) = UD(t).grating.neural_dF;
        end
        if ~isempty(xC) && ~isempty(UD(t).corridor.neural_dF)
            CdF(t, :, :) = UD(t).corridor.neural_dF;
        end
    end
    
    % Filter neurons
    keep = filter_neurons(Gspk, Cspk, minNeuronSpkThr);
    fprintf('  kept %d/%d neurons\n', numel(keep), nN0);
    
    Gspk = Gspk(:, keep, :);
    GdF  = GdF(:, keep, :);
    if ~isempty(Cspk)
        Cspk = Cspk(:, keep, :); 
        CdF  = CdF(:, keep, :);
    end
    
    % keep full tensors (kept neurons) for later re-use/plotting/decoding
    NeuralStore(a).tag   = tag;
    NeuralStore(a).xG    = xG;
    NeuralStore(a).xC    = xC;
    NeuralStore(a).Gspk  = Gspk;   % [nTrials x nKeptNeurons x tBins]
    NeuralStore(a).Cspk  = Cspk;   % [] or [nTrials x nKeptNeurons x sBins]
    NeuralStore(a).GdF   = GdF;    % <-- Storing dF Grating
    NeuralStore(a).CdF   = CdF;    % <-- Storing dF Corridor
    NeuralStore(a).keep  = keep;   % kept neuron indices relative to original
    NeuralStore(a).nKept = numel(keep);
    NeuralStore(a).contrast   = [UD.contrast]';
    NeuralStore(a).dispersion = [UD.dispersion]';
    NeuralStore(a).stimulus   = [UD.stimulus]';
    NeuralStore(a).outcome    = ensure_outcome(UD);
    
    % Labels & Behaviour
    stim = [UD.stimulus];
    con  = normalize_contrast([UD.contrast]);  % 0.99 -> 1.00
    if isfield(UD,'dispersion'), dps = [UD.dispersion]; else, dps = zeros(1,numel(UD)); end
    outcome = ensure_outcome(UD);                 % 1=Hit,2=Miss,3=FA,4=CR
    perf    = ismember(outcome, [1,4]);           % correct (Hit or CR)
    goChoice= ismember(outcome, [1,3]);           % responded (Hit or FA)
    preVel     = field_or_default(UD,'preRZ_velocity',   NaN);  % cm/s (mean in window)
    preLck     = field_or_default(UD,'preRZ_licks',      NaN);  
    preDur     = field_or_default(UD,'preRZ_duration_s', NaN);  
    preLckRate = field_or_default(UD,'preRZ_lick_rate',  NaN);  
    
    % --- Fallback: compute lick rate if field missing but counts+duration exist
    if all(isnan(preLckRate)) && ~all(isnan(preLck)) && ~all(isnan(preDur))
        preLckRate = preLck ./ preDur;
        preLckRate(~isfinite(preLckRate)) = NaN; % guard against 0/0 or Inf
    end
    
    % Grating/corridor masks (robust)
    tmaskG = xG >= timeWindowG(1) & xG <= timeWindowG(2);
    if isempty(xC)
        smaskC = [];
    elseif avgCorridorAll || isempty(posWindowC)
        smaskC = true(size(xC));
    else
        smaskC = xC >= posWindowC(1) & xC <= posWindowC(2);
        if ~any(smaskC), smaskC = true(size(xC)); end
    end
    
    % Neural metrics (Spikes)
    meanAct_gr = squeeze(mean(mean(Gspk(:,:,tmaskG),3,'omitnan'),2,'omitnan'));      % [nTrials x 1]
    Sgr = compute_gv_and_trace(Gspk(:,:,tmaskG), ...
            'regularizer', 1, ...
            'clip_det_threshold', 100, ...
            'variant', 'snippet', ...
            'return_normGV', true);
    logGV_gr    = Sgr.gv_log;
    normGV_gr   = Sgr.normGV;
    
    % Neural metrics (dF/F) - Average Trial df/f
    meandF_gr = squeeze(mean(mean(GdF(:,:,tmaskG),3,'omitnan'),2,'omitnan'));      
    
    if ~isempty(smaskC)
        meanAct_co = squeeze(mean(mean(Cspk(:,:,smaskC),3,'omitnan'),2,'omitnan'));
        [logGV_co, normGV_co] = per_trial_logdetGV_and_norm(Cspk(:,:,smaskC), gvLambda);
        meandF_co = squeeze(mean(mean(CdF(:,:,smaskC),3,'omitnan'),2,'omitnan'));
    else
        meanAct_co = nan(numel(UD),1);
        logGV_co   = nan(numel(UD),1);
        normGV_co  = nan(numel(UD),1);
        meandF_co  = nan(numel(UD),1);
    end
    
    % Confidence proxy for the model: use lick RATE (z-scored) vs velocity
    zLickRate = nan_zscore(preLckRate);
    zVel      = nan_zscore(preVel);
    confidence_proxy = zLickRate - zVel;
    
    % Assemble per-animal trial table
    TrialTbl = table( ...
        repmat({tag}, numel(UD),1), [UD.session]', (1:numel(UD))', ...
        stim(:), con(:), dps(:), outcome(:), goChoice(:), perf(:), ...
        meanAct_gr(:), meanAct_co(:), meandF_gr(:), meandF_co(:), ...  % <-- Added dF
        logGV_gr(:), logGV_co(:), normGV_gr(:), normGV_co(:), ...
        preVel(:), preLck(:), preLckRate(:), confidence_proxy(:), ...
        'VariableNames',{'animal','session','trial', ...
        'stimulus','contrast','dispersion','outcome','goChoice','performance', ...
        'meanAct_gr','meanAct_co','meandF_gr','meandF_co', ... % <-- Added dF
        'logGV_gr','logGV_co','normGV_gr','normGV_co', ...
        'preRZ_velocity','preRZ_licks','preRZ_lick_rate','confidence'});
    TrialTbl_all = [TrialTbl_all; TrialTbl]; %#ok<AGROW>
    
    % Prepare model data for this animal
    data_for_model.orientation = TrialTbl.stimulus;
    data_for_model.contrast    = TrialTbl.contrast;
    data_for_model.dispersion  = TrialTbl.dispersion;
    data_for_model.choices     = TrialTbl.goChoice;   % "go" choice
    data_for_model.confidence  = TrialTbl.confidence;
    data_for_model.n_trials    = height(TrialTbl);
    all_data_for_model{a}      = data_for_model;
    
    % Condition means (for pooling later)
    [CM, lv] = condition_means(TrialTbl, semFun);
    AnimalSummaries(a).tag      = tag;
    AnimalSummaries(a).levels   = lv;
    AnimalSummaries(a).CondMean = CM;
end
if isempty(TrialTbl_all), error('No usable trials across animals.'); end

% per-animal go-alignment for behaviour
TrialTbl_all.theta_deg      = TrialTbl_all.stimulus; % true orientation (for neural)
TrialTbl_all.go_is_vertical = false(height(TrialTbl_all),1);
TrialTbl_all.theta_from_go  = nan(height(TrialTbl_all),1); % signed distance to go ([-90,90])
TrialTbl_all.abs_from_go    = nan(height(TrialTbl_all),1); % |distance| (>=0)
uniqueAnimals = unique(TrialTbl_all.animal);
goMap = struct(); % keep mapping for record
for ia = 1:numel(uniqueAnimals)
    tag = uniqueAnimals{ia};
    % pull the same UD we used for this animal (from PerAnimalRaw)
    idxAnimal = strcmp(TrialTbl_all.animal, tag);
    ixPA = find(strcmp({PerAnimalRaw.tag}, tag), 1, 'first');
    UD_a = PerAnimalRaw(ixPA).UD;
    [isVert, goRef] = detect_go_mapping(UD_a);
    goMap(ia).tag    = tag; %#ok<AGROW>
    goMap(ia).isVert = isVert;
    goMap(ia).goRef  = goRef;
    TrialTbl_all.go_is_vertical(idxAnimal) = isVert;
    theta = TrialTbl_all.theta_deg(idxAnimal);
    d     = wrap_signed_deg(theta - goRef);   % signed distance to animal's go
    TrialTbl_all.theta_from_go(idxAnimal) = d;
    TrialTbl_all.abs_from_go(idxAnimal)   = abs(d);
end

% --- NEW: use |Δ from go| for all modeling (unify mappings across animals)
for a = 1:nAnimals
    tagA = Animals(a).tag;
    maskA = strcmp(TrialTbl_all.animal, tagA);
    if ~any(maskA) || isempty(all_data_for_model{a}), continue; end
    % overwrite model orientation with absolute delta-from-go in [0,90]
    all_data_for_model{a}.orientation = abs(TrialTbl_all.theta_from_go(maskA));
    % (optional) keep for reference/debug:
    all_data_for_model{a}.orientation_true = TrialTbl_all.theta_deg(maskA);
    all_data_for_model{a}.delta_from_go    = abs(TrialTbl_all.theta_from_go(maskA));
end

%% Join with ideal observer analysis
S  = load(fullfile(working_dir,'IOResults.mat'),'IOResults');
IO = S.IOResults;
T  = stack_IO_trial_tables(IO);  

% Ensure posteriors AND likelihoods are included in the columns to join!
ioCols = {'unc_perceptual', 'unc_decision', 'post_s_given_map', 'post_s_marginal', 'L_s_given_map', 'L_s_marginal'}; 

if height(TrialTbl_all) ~= height(T)
    warning('Row mismatch: Neural table (%d) vs IO table (%d). Using innerjoin for safety.', ...
        height(TrialTbl_all), height(T));
    TrialTbl_all = innerjoin(TrialTbl_all, T(:, [ {'animal','session','trial'}, ioCols ]));
else
    TrialTbl_all = [TrialTbl_all, T(:, ioCols)];
end

%% --- Part 8: Data Export for Python PyTorch Decoder (Refined) ---
fprintf('--- Part 8: Preparing Export for Python Decoder ---\n');

% 1. Calculate Decision Posterior from the MAP inference (Original)
post_s_map = TrialTbl_all.post_s_given_map;
p_less_45_map    = sum(post_s_map(:, 1:45), 2, 'omitnan');
p_greater_45_map = sum(post_s_map(:, 47:91), 2, 'omitnan');
p_exactly_45_map = post_s_map(:, 46);
TrialTbl_all.decision_posterior_map = [p_less_45_map + p_exactly_45_map/2, p_greater_45_map + p_exactly_45_map/2];

% 2. Calculate Decision Posterior from the MARGINALISED inference (New target)
post_s_marg = TrialTbl_all.post_s_marginal;
p_less_45_marg    = sum(post_s_marg(:, 1:45), 2, 'omitnan');
p_greater_45_marg = sum(post_s_marg(:, 47:91), 2, 'omitnan');
p_exactly_45_marg = post_s_marg(:, 46);
TrialTbl_all.decision_posterior_marginal = [p_less_45_marg + p_exactly_45_marg/2, p_greater_45_marg + p_exactly_45_marg/2];

% Overwrite decision_posterior to marginalized to feed Python smoothly
TrialTbl_all.decision_posterior = TrialTbl_all.decision_posterior_marginal; 

% Convert to scalar struct for scipy.io compatibility
TrialTbl_Struct = table2struct(TrialTbl_all, 'ToScalar', true);

% Save
export_filename = fullfile('C:\Users\theox\Desktop\Experiments\2afcanalysis', 'VR_Decoder_Data_Export.mat');
save(export_filename, 'NeuralStore', 'TrialTbl_Struct', 'IO', '-v7');
fprintf('Clean Export Complete!\n');

%% Helpers

function s = sem_omit(x, dim)
% SEM with NaN omission
if nargin<2, dim = []; end
if isempty(dim), dim = find(size(x)~=1,1); if isempty(dim), dim = 1; end, end
n = sum(~isnan(x), dim);
s = std(x, 0, dim, 'omitnan') ./ max(1, sqrt(n));
end

function S = compute_gv_and_trace(tensorObs, varargin)
% Compute per-trial covariance metrics for a tensor:
%   tensorObs : [nTrials x nNeurons x nObs]  (rows=trials, cols=neurons, 3rd dim=time/obs)
%
% For each trial, with R = [nObs x nNeurons] trajectory:
%   C      = cov(R)
%   detReg = det(C + reg*I)
%   trC    = trace(C)
%
% Variants:
%   'variant' == 'snippet' → gv_log = log(det(C+I)) / sqrt(nNeurons)     (matches your code)
%   'variant' == 'theo'    → gv_log = log(det(C+I) / 2)                  (your earlier spec)
%
% Options:
%   'regularizer'        : scalar added to the diagonal (default 1)
%   'clip_det_threshold' : NaN-out trials whose raw det(C+I) > threshold (default Inf)
%   'return_normGV'      : also return normGV = exp(gv_log)/||mean(R)|| (default true)
%
% Returns struct S with fields (nTrials x 1):
%   .gv_log            : log generalized variance (as per chosen variant)
%   .overall_var       : trace(C)
%   .det_raw           : det(C + reg*I)  (useful for debugging/clipping)
%   .normGV            : optional normalization (if return_normGV==true)

    ip = inputParser;
    addParameter(ip, 'regularizer', 1);
    addParameter(ip, 'clip_det_threshold', Inf);
    addParameter(ip, 'variant', 'snippet');     % 'snippet' or 'theo'
    addParameter(ip, 'return_normGV', true);
    parse(ip, varargin{:});
    Opt = ip.Results;

    nT = size(tensorObs,1);
    S.gv_log      = nan(nT,1);
    S.overall_var = nan(nT,1);
    S.det_raw     = nan(nT,1);
    if Opt.return_normGV, S.normGV = nan(nT,1); end

    for tr = 1:nT
        % R: [nObs x nNeurons]
        R = squeeze(tensorObs(tr,:,:)).';
        if isempty(R) || size(R,1) < 2 || all(isnan(R(:))), continue; end

        % drop NaN rows
        good = all(~isnan(R), 2);
        R = R(good,:);
        if size(R,1) < 2, continue; end

        C = cov(R);                       % [nNeurons x nNeurons]
        nN = size(C,1);
        Cre = C + Opt.regularizer*eye(nN);

        % raw det for clipping/debug
        detCre = det(Cre);
        S.det_raw(tr) = detCre;

        if detCre > Opt.clip_det_threshold
            % leave S.gv_log(tr) as NaN and still report overall variance
            S.overall_var(tr) = trace(C);
            continue;
        end

        % stable log(det(Cre)) if needed
        logdetCre = NaN;
        if isfinite(detCre) && detCre > 0
            logdetCre = log(detCre);
        else
            % numerical fallbacks
            [Rchol,p] = chol(Cre);
            if p==0
                logdetCre = 2*sum(log(diag(Rchol)));
            else
                s = svd(Cre);
                s = s(s>eps);
                logdetCre = sum(log(s));
            end
        end

        switch lower(Opt.variant)
            case 'snippet'
                S.gv_log(tr) = logdetCre / sqrt(nN);      % your loop behavior
            case 'theo'
                S.gv_log(tr) = logdetCre - log(2);        % log(det(C+I)/2)
            otherwise
                error('Unknown variant: %s', Opt.variant);
        end

        S.overall_var(tr) = trace(C);

        if Opt.return_normGV
            mu = mean(R,1);
            S.normGV(tr) = exp(S.gv_log(tr)) / (norm(mu) + eps);
        end
    end
end



function [goIsVertical, goRefDeg] = detect_go_mapping(UD)
% 0° = horizontal, 90° = vertical
% Returns:
%   goIsVertical : true if GO is vertical (90°), false if horizontal (0°)
%   goRefDeg     : 90 or 0 accordingly

    % fold orientations to [0,180)
    theta = mod([UD.stimulus], 180);
    theta = theta(:);
    theta = theta(~isnan(theta));
    if isempty(theta), goIsVertical = true; goRefDeg = 90; return; end
    tol = 1e-9;
    dist = @(x,mu) abs(mod(x - mu + 90,180) - 90);  % circular distance to mu∈{0,90}

    % --- 1) explicit metadata, if present
    if isfield(UD,'go_is_vertical') || isfield(UD,'goIsVertical') || isfield(UD,'goRefDeg')
        if isfield(UD,'goRefDeg')
            g = mod(round(UD(1).goRefDeg), 180);
            if abs(g-0) <= 45, goIsVertical = false; goRefDeg = 0;
            else,               goIsVertical = true;  goRefDeg = 90; end
            return;
        end
        vals = [];
        if isfield(UD,'go_is_vertical'), vals = [vals, [UD.go_is_vertical]]; end %#ok<AGROW>
        if isfield(UD,'goIsVertical'),   vals = [vals, [UD.goIsVertical]];   end %#ok<AGROW>
        vals = vals(~isnan(vals));
        if ~isempty(vals)
            goIsVertical = mode(logical(vals));
            goRefDeg = 90*goIsVertical;
            return;
        end
    end

    % --- 2) use outcomes (derive from hit/miss/FA/CR if needed)
    oc = ensure_outcome(UD);                       % 1=Hit,2=Miss,3=FA,4=CR
    if any(~isnan(oc))
        goStim = ismember(oc, [1 2]);              % Go = Hit or Miss
        if any(goStim)
            og = mod([UD(goStim).stimulus], 180);
            if mean(dist(og,0), 'omitnan') <= mean(dist(og,90), 'omitnan')
                goIsVertical = false; goRefDeg = 0;
            else
                goIsVertical = true;  goRefDeg = 90;
            end
            return;
        end
    end

    % --- 3) fallback: trial-wise goChoice if it actually exists in UD
    if isfield(UD, 'goChoice')
        oU = unique(mod([UD.stimulus],180));
        pGo_by_o = arrayfun(@(o) mean([UD(mod([UD.stimulus],180)==o).goChoice], 'omitnan'), oU);
        if nansum(pGo_by_o) > 0
            dH = nansum(pGo_by_o(:) .* dist(oU(:),0));
            dV = nansum(pGo_by_o(:) .* dist(oU(:),90));
            if dH <= dV, goIsVertical = false; goRefDeg = 0;
            else,        goIsVertical = true;  goRefDeg = 90; end
            return;
        end
    end

end


function d = wrap_signed_deg(x)
% Signed circular difference in degrees, wrapped to [-90, 90].
% Works for vectors and handles edge cases (e.g., exactly 90 or -90).
d = mod(x + 90, 180) - 90;
% ensure exact 90 maps consistently (optional)
d(abs(d-90) < 1e-9)  =  90;
d(abs(d+90) < 1e-9)  = -90;
end

function outcome = ensure_outcome(UD)
if ~isfield(UD,'outcome')
    outcome = nan(numel(UD),1);
    for i = 1:numel(UD)
        if isfield(UD(i),'hit') && UD(i).hit, outcome(i)=1;
        elseif isfield(UD(i),'miss') && UD(i).miss, outcome(i)=2;
        elseif isfield(UD(i),'FA')   && UD(i).FA,   outcome(i)=3;
        elseif isfield(UD(i),'CR')   && UD(i).CR,   outcome(i)=4;
        end
    end
else
    outcome = [UD.outcome];
end
end

function v = field_or_default(UD, f, d)
if isfield(UD,f), v = [UD.(f)]; else, v = repmat(d,1,numel(UD)); end
end

function [logGV, normGV] = per_trial_logdetGV_and_norm(tensorObs, lambda)
% tensorObs: [nTrials x nNeurons x nObs]
nT = size(tensorObs,1); nN = size(tensorObs,2);
logGV  = nan(nT,1);
normGV = nan(nT,1);
for tr = 1:nT
    X = squeeze(tensorObs(tr,:,:)).';     % [nObs x nNeurons]
    if isempty(X) || all(isnan(X(:))), continue; end
    good = all(~isnan(X),2);
    X = X(good,:);
    if size(X,1) < 2, continue; end
    mu = mean(X,1);
    Xc = X - mu;
    C  = (Xc.'*Xc) / max(1,(size(X,1)-1));
    C  = C + lambda*eye(nN);
    [R,p] = chol(C);
    if p==0
        logGV(tr) = 2*sum(log(diag(R)));
    else
        s = svd(C); s = s(s>eps);
        logGV(tr) = sum(log(s));
    end
    normGV(tr) = exp(logGV(tr)) / (norm(mu)+eps);
end
end

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

function [CM, lv] = condition_means(T, semFun)
lv.orients     = unique(T.stimulus);
lv.contrasts   = unique(T.contrast);
lv.dispersions = unique(T.dispersion);
nO=numel(lv.orients); nC=numel(lv.contrasts); nD=numel(lv.dispersions);
Mnames = {'meanAct_gr','meanAct_co','logGV_gr','logGV_co','normGV_gr','normGV_co'};
for k=1:numel(Mnames)
    CM.(Mnames{k}).mean = nan(nO,nC,nD);
    CM.(Mnames{k}).sem  = nan(nO,nC,nD);
    CM.(Mnames{k}).n    = zeros(nO,nC,nD);
end
for io=1:nO
    for ic=1:nC
        for id=1:nD
            idx = T.stimulus==lv.orients(io) & T.contrast==lv.contrasts(ic) & T.dispersion==lv.dispersions(id);
            if ~any(idx), continue; end
            for k=1:numel(Mnames)
                x = T{idx, Mnames{k}};
                CM.(Mnames{k}).mean(io,ic,id) = mean(x,'omitnan');
                CM.(Mnames{k}).sem (io,ic,id) = semFun(x);
                CM.(Mnames{k}).n   (io,ic,id) = sum(~isnan(x));
            end
        end
    end
end
end

function [Gspk, Cspk, nN0] = build_spike_tensors(UD)
nT = numel(UD);
nN = size(UD(1).grating.neural_spikes,1);
tB = size(UD(1).grating.neural_spikes,2);
if isfield(UD(1).corridor,'neural_spikes') && ~isempty(UD(1).corridor.neural_spikes)
    sB = size(UD(1).corridor.neural_spikes,2);
else
    sB = 0;
end
Gspk = nan(nT, nN, tB);
if sB>0, Cspk = nan(nT, nN, sB); else, Cspk = []; end
for tr=1:nT
    Gspk(tr,:,:) = UD(tr).grating.neural_spikes;
    if sB>0 && isfield(UD(tr).corridor,'neural_spikes') && ~isempty(UD(tr).corridor.neural_spikes)
        Cspk(tr,:,:) = UD(tr).corridor.neural_spikes;
    end
end
nN0 = nN;
end

function keep = filter_neurons(Gspk, Cspk, thr)
gmax = squeeze(max(max(Gspk,[],3),[],1));
if ~isempty(Cspk)
    cmax = squeeze(max(max(Cspk,[],3),[],1));
    keep = find(max([gmax(:), cmax(:)],[],2) > thr);
else
    keep = find(gmax(:) > thr);
end
end

function cn = normalize_contrast(c)
tol = 0.02;
c2 = round(c,3);
c2(abs(c2-1)<=tol) = 1;
cn = c2;
end