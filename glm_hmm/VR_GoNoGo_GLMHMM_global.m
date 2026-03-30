warning off
addpath 'C:\Users\theox\Desktop\Experiments\Working\VR_files_2025'

sesnames1 = {'20250605_Cb15', '20250613_Cb15', '20250620_Cb15', '20250624_Cb15', '20250709_Cb15'};
sesnames2 = {'20250606_Cb17', '20250613_Cb17', '20250620_Cb17', '20250624_Cb17', '20250701_Cb17'};
sesnames3 = {'20250904_Cb21', '20250910_Cb21', '20250911_Cb21', '20250912_Cb21', '20250918_Cb21'};
sesnames4 = {'20251024_Cb22', '20251027_Cb22', '20251028_Cb22', '20251030_Cb22', '20251105_Cb22'};
sesnames5 = {'20250918_Cb24', '20250919_Cb24', '20251020_Cb24', '20251021_Cb24'};
sesnames6 = {'20250903_Cb25', '20250904_Cb25', '20250910_Cb25', '20250911_Cb25', '20250916_Cb25'};

all_sesnames = {sesnames1, sesnames2, sesnames3, sesnames4, sesnames5, sesnames6};

% Compute total number of sessions across all animals.
total_sessions = sum(cellfun(@numel, all_sesnames));

% Preallocate cell arrays.
allVrSessions = cell(1, total_sessions);
allSesnamesConc = cell(1, total_sessions);

% Create a waitbar.
hWait = waitbar(0, 'Loading VR sessions...');

idx = 1;
for ianimal = 1:length(all_sesnames)
    currentSesnames = all_sesnames{ianimal};
    for iSession = 1:numel(currentSesnames)
        sesname = currentSesnames{iSession};
        load(['vr_', sesname, '_light.mat']);  % Loads variable 'vr'
        allVrSessions{idx} = vr;
        allSesnamesConc{idx} = sesname;
        idx = idx + 1;
        waitbar(idx / total_sessions, hWait, sprintf('Loading session %d of %d', idx, total_sessions));
    end
end

close(hWait);

% Now pass the concatenated sessions and session names to your extractor.
extractGLMHmmPredictorsVR(allVrSessions, allSesnamesConc, 'GLMHmm_predictors_vr.csv');

%%

% extractGLMHmmPredictorsVR(allVrSessions, all_sesnames)


function extractGLMHmmPredictorsVR(vrCell, sesnames, outFile)
% extractGLMHmmPredictorsVR  Extract predictors and outputs from VR structures.
%
%   extractGLMHmmPredictorsVR(vrCell, sesnames, outFile)
%
% Inputs:
%   vrCell  - A cell array of VR structures (one per session). Each VR structure
%             should contain:
%               • vr.trialLog: a cell array where each element has a field 'stimulus'.
%               • vr.cfg.rewardEligibility: a vector (length nTrials) indicating
%                 whether each trial was eligible for reward.
%               • Each trial in vr.trialLog should have the fields 'rewardGiven'
%                 and 'falseAlarm' indicating if a "go" response was made.
%   sesnames - A cell array of session name strings. Each session name is assumed
%              to have at least 13 characters, where characters 10 to 13 correspond
%              to the animal ID.
%   outFile - (Optional) Name of the output CSV file. Default is 'GLMHmm_predictors_vr.csv'
%
% For each trial (starting with trial 2 so that previous-trial info exists),
% the following variables are extracted:
%
%   - current_stim_orientation: from vr.trialLog{t}.stimulus (current trial)
%   - current_reward: from vr.cfg.rewardEligibility(t) (current trial's reward eligibility)
%   - prev_stim_orientation: from vr.trialLog{t-1}.stimulus (previous trial)
%   - prev_choice: 1 if (rewardGiven or falseAlarm) is true for previous trial, 0 otherwise
%   - current_choice: 1 if (rewardGiven or falseAlarm) is true for current trial, 0 otherwise
%   - session: session index (within this animal)
%   - animal: extracted from the session name (characters 10 to 13)
%
% Example usage:
%   sesnames = {'20250220_Cb14', '20250221_Cb14', '20250222_Cb14', ...};
%   extractGLMHmmPredictorsVR(vrSessions, sesnames, 'GLMHmm_predictors_vr.csv');

if nargin < 3 || isempty(outFile)
    outFile = 'GLMHmm_predictors_vr.csv';
end

allData = table();

nSessions = length(vrCell);
hWait = waitbar(0, 'Processing sessions...');

for s = 1:length(vrCell)

    % Update waitbar with current session number.
    waitbar(s/nSessions, hWait, sprintf('Processing session %d of %d', s, nSessions));

    vr = vrCell{s};
    nTrials = length(vr.trialLog);
    if nTrials < 2
        continue;
    end
    nData = nTrials - 2;  % Skip first and last trial 

    currentStimOrientation = nan(nData,1);
    currentContrast = nan(nData,1);
    currentDispersion = nan(nData,1);
    currentReward = nan(nData,1);
    prevStimOrientation = nan(nData,1);
    prevContrast = nan(nData,1);
    prevDispersion = nan(nData,1);
    prevChoice = nan(nData,1);
    currentChoice = nan(nData,1);
    prevReward = nan(nData,1);

    for t = 2:nTrials-1
        idx = t - 1;

        % Current trial's stimulus
        currentStimOrientation(idx) = vr.trialLog{t}.stimulus;
        currentContrast(idx) = vr.trialLog{t}.contrast;
        currentDispersion(idx) = vr.trialLog{t}.dispersion;
        % Current trial's reward eligibility
        if isfield(vr.cfg, 'rewardEligibility') && numel(vr.cfg.rewardEligibility) >= t
            currentReward(idx) = double(vr.cfg.rewardEligibility(t));
        else
            currentReward(idx) = NaN;
        end

        % Previous trial's stimulus
        prevStimOrientation(idx) = vr.trialLog{t-1}.stimulus;
        prevContrast(idx) = vr.trialLog{t-1}.contrast;
        prevDispersion(idx) = vr.trialLog{t-1}.dispersion;

        % Previous trial's choice: 1 if rewardGiven or falseAlarm is true
        if isfield(vr.trialLog{t-1}, 'rewardGiven') && isfield(vr.trialLog{t-1}, 'falseAlarm')
            if vr.trialLog{t-1}.rewardGiven || vr.trialLog{t-1}.falseAlarm
                prevChoice(idx) = 1;
            else
                prevChoice(idx) = 0;
            end
        else
            prevChoice(idx) = NaN;
        end

        prevReward(idx) = vr.trialLog{t-1}.rewardGiven;

        % Current trial's choice: 1 if rewardGiven or falseAlarm is true
        if isfield(vr.trialLog{t}, 'rewardGiven') && isfield(vr.trialLog{t}, 'falseAlarm')
            if vr.trialLog{t}.rewardGiven || vr.trialLog{t}.falseAlarm
                currentChoice(idx) = 1;
            else
                currentChoice(idx) = 0;
            end
        else
            currentChoice(idx) = NaN;
        end
    end

    % Create table for this session.
    T = table(currentStimOrientation, currentContrast, currentDispersion, currentReward,...
        prevStimOrientation, prevContrast, prevDispersion,...
        prevChoice, prevReward, currentChoice, 'VariableNames', ...
        {'current_stim_orientation', 'current_contrast', 'current_dispersion', 'current_reward',...
        'prev_stim_orientation', 'prev_contrast', 'prev_dispersion',...
        'prev_choice', 'prev_reward', 'current_choice'});
    T.session = repmat(s, height(T), 1);

    % Extract animal ID from the session name (characters 10 to 13).
    animal_id = sesnames{s}(10:13);
    T.animal = repmat({animal_id}, height(T), 1);

    allData = [allData; T];  %#ok<AGROW>
end

close(hWait);  % Close the waitbar when done.

writetable(allData, outFile);
fprintf('Predictor table saved to %s\n', outFile);
end