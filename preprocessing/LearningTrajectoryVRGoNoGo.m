%% Settings
% List the animal IDs you want to process.
animalIDs = {'Cb4', 'Cb11', 'Cb14', 'Cb15', 'Cb17', 'Cb21', 'Cb22', 'Cb23', 'Cb24', 'Cb25'};  % modify as needed
% Folder containing the session files.
baseDir = 'C:\Users\theox\Desktop\Experiments\Working\VR_files_2025';  % modify as needed
% Define the cutoff date (sessions before this will be skipped).
cutoffDateStr = '20250210';
cutoffDate = datenum(cutoffDateStr, 'yyyymmdd');
% Initialize a structure to store each animal's sessions.
learningTrajectories = struct();
warning off

%% Loop over each animal and their sessions
for a = 1:length(animalIDs)
    animalID = animalIDs{a};
    
    % Look for files matching the pattern: 'vr_YYYYMMDD_AnimalID.mat'
    filePattern = fullfile(baseDir, sprintf('vr_*_%s.mat', animalID));
    files = dir(filePattern);
    
    % Preallocate arrays for session data for two groups:
    % Group 'half' for rewardedTrialProportion == 0.5
    % Group 'high' for rewardedTrialProportion > 0.5
    half_dates = [];
    half_dprime = [];
    half_nTrials = [];
    half_hitRate = [];
    half_missRate = [];
    half_faRate = [];
    half_crRate = [];
    
    high_dates = [];
    high_dprime = [];
    high_nTrials = [];
    high_hitRate = [];
    high_missRate = [];
    high_faRate = [];
    high_crRate = [];
    
    % Initialize a progress bar for this animal's files.
    numFiles = length(files);
    hWait = waitbar(0, sprintf('Processing sessions for %s...', animalID));
    
    for f = 1:numFiles
        fileName = files(f).name;
        filePath = fullfile(baseDir, fileName);
        
        % Update progress bar.
        waitbar(f/numFiles, hWait, sprintf('Processing file %d of %d for %s', f, numFiles, animalID));
        
        % Extract date from the file name.
        tokens = regexp(fileName, 'vr_(\d{8})_.*\.mat', 'tokens');
        if isempty(tokens)
            warning('File %s does not match expected naming convention', fileName);
            continue;
        end
        dateStr = tokens{1}{1};  
        sessionDate = datenum(dateStr, 'yyyymmdd');
        
        % Skip sessions before the cutoff date.
        if sessionDate < cutoffDate
            fprintf('Skipping session %s because it is before cutoff date.\n', fileName);
            continue;
        end
        
        % Load the session file.
        loadedData = load(filePath);
        vr = loadedData.vr;
        
        % Determine if this session is Basic Learning:
        % For Basic Learning, trialStimuli should have only 2 unique values.
        if isfield(vr.cfg, 'trialStimuli')
            uniqueOrients = unique(vr.cfg.trialStimuli);
            if length(uniqueOrients) > 2
                fprintf('Non-Basic Learning session %s found (trialStimuli > 2). Skipping subsequent sessions for %s.\n', fileName, animalID);
                break;  % Exit loop for this animal.
            end
        end
        
        % Also skip sessions that have a requiredLicksVector with any element > 1.
        if isfield(vr.cfg, 'requiredLicksVector') && any(vr.cfg.requiredLicksVector > 1)
            fprintf('Non-Basic Learning session %s found (requiredLicksVector > 1). Skipping subsequent sessions for %s.\n', fileName, animalID);
            break;
        end
        
        % Check if rewardedTrialProportion exists.
        if ~isfield(vr.cfg, 'rewardedTrialProportion')
            fprintf('Skipping session %s because rewardedTrialProportion is missing\n', fileName);
            continue;
        end
        
        prop = vr.cfg.rewardedTrialProportion;
        % Only process sessions where prop is 0.5 or >0.5.
        if prop == 0.5 || prop > 0.5
            % Compute dprime for the session.
            dprime = computeDprime(vr);
            % Compute effective number of trials (exclude default trials)
            nTrials = length(vr.trialLog) - vr.cfg.numDefaultTrials;
            
            % Compute performance rates.
            [hitRate, missRate, faRate, crRate] = computeRates(vr);
            
            % Store based on rewardedTrialProportion.
            if prop == 0.5
                half_dates(end+1,1) = sessionDate;
                half_dprime(end+1,1) = dprime;
                half_nTrials(end+1,1) = nTrials;
                half_hitRate(end+1,1) = hitRate;
                half_missRate(end+1,1) = missRate;
                half_faRate(end+1,1) = faRate;
                half_crRate(end+1,1) = crRate;
            elseif prop > 0.5
                high_dates(end+1,1) = sessionDate;
                high_dprime(end+1,1) = dprime;
                high_nTrials(end+1,1) = nTrials;
                high_hitRate(end+1,1) = hitRate;
                high_missRate(end+1,1) = missRate;
                high_faRate(end+1,1) = faRate;
                high_crRate(end+1,1) = crRate;
            end
        else
            fprintf('Skipping session %s with rewardedTrialProportion < 0.5\n', fileName);
        end
    end
    delete(hWait);
    
    % Save results for this animal.
    learningTrajectories.(animalID).half.dates = half_dates;
    learningTrajectories.(animalID).half.dprime = half_dprime;
    learningTrajectories.(animalID).half.nTrials = half_nTrials;
    learningTrajectories.(animalID).half.hitRate = half_hitRate;
    learningTrajectories.(animalID).half.missRate = half_missRate;
    learningTrajectories.(animalID).half.faRate = half_faRate;
    learningTrajectories.(animalID).half.crRate = half_crRate;
    
    learningTrajectories.(animalID).high.dates = high_dates;
    learningTrajectories.(animalID).high.dprime = high_dprime;
    learningTrajectories.(animalID).high.nTrials = high_nTrials;
    learningTrajectories.(animalID).high.hitRate = high_hitRate;
    learningTrajectories.(animalID).high.missRate = high_missRate;
    learningTrajectories.(animalID).high.faRate = faRate;
    learningTrajectories.(animalID).high.crRate = crRate;
end

%% Plotting d' vs. Cumulative Trials (High vs. Half Reward Probability)
% This new section plots d' for both conditions, showing individual animals
% and the group average with SEM for each condition.

figure('Color', 'w');

% --- Define plotting parameters ---
indivColor = [0.8 0.8 0.8]; % Faint gray for individual animals
meanColorHigh = [0.8500 0.3250 0.0980]; % Orange for high reward mean
meanColorHalf = [0 0.4470 0.7410];      % Blue for half reward mean

% --- Subplot 1: Sessions with rewarded trial proportion > 0.5 ---
subplot(1,2,1);
hold on;
title('d'' (Reward Prob > 0.5)');
xlabel('Cumulative Trials Completed');
ylabel('d''');

% Find max trials for common x-axis
maxTrialsHigh = 0;
for a = 1:length(animalIDs)
    animalID = animalIDs{a};
    if isfield(learningTrajectories.(animalID), 'high') && ~isempty(learningTrajectories.(animalID).high.nTrials)
        maxTrialsHigh = max(maxTrialsHigh, max(cumsum(learningTrajectories.(animalID).high.nTrials)));
    end
end
common_x_high = linspace(0, maxTrialsHigh, 200);
interpolated_high = nan(length(common_x_high), length(animalIDs));

% Plot individual lines and collect data for averaging
for a = 1:length(animalIDs)
    animalID = animalIDs{a};
    if isfield(learningTrajectories.(animalID), 'high') && ~isempty(learningTrajectories.(animalID).high.dprime)
        x = cumsum(learningTrajectories.(animalID).high.nTrials);
        y = learningTrajectories.(animalID).high.dprime;
        plot(x, y, '-', 'Color', indivColor, 'LineWidth', 1);
        [x_unique, ia] = unique(x);
        if length(x_unique) >= 2
            interpolated_high(:, a) = interp1(x_unique, y(ia), common_x_high);
        end
    end
end

% Calculate and plot mean + SEM
mean_y_high = mean(interpolated_high, 2, 'omitnan');
n_animals_high = sum(~isnan(interpolated_high), 2);
sem_y_high = std(interpolated_high, 0, 2, 'omitnan') ./ sqrt(n_animals_high);
sem_y_high(n_animals_high < 2) = 0;
valid_idx_high = ~isnan(mean_y_high);
h1 = shadedErrorBar(common_x_high(valid_idx_high), mean_y_high(valid_idx_high), sem_y_high(valid_idx_high), ...
    'lineprops', {'-', 'Color', meanColorHigh, 'LineWidth', 2});

set(gca, 'TickDir', 'out', 'LineWidth', 1); box off; xlim([0, maxTrialsHigh]);
legend([h1.mainLine], {'Group Mean'}, 'Location', 'best');


% --- Subplot 2: Sessions with rewarded trial proportion = 0.5 ---
subplot(1,2,2);
hold on;
title('d'' (Reward Prob = 0.5)');
xlabel('Cumulative Trials Completed');
ylabel('d''');

% Find max trials for common x-axis
maxTrialsHalf = 0;
for a = 1:length(animalIDs)
    animalID = animalIDs{a};
    if isfield(learningTrajectories.(animalID), 'half') && ~isempty(learningTrajectories.(animalID).half.nTrials)
        maxTrialsHalf = max(maxTrialsHalf, max(cumsum(learningTrajectories.(animalID).half.nTrials)));
    end
end
common_x_half = linspace(0, maxTrialsHalf, 200);
interpolated_half = nan(length(common_x_half), length(animalIDs));

% Plot individual lines and collect data for averaging
for a = 1:length(animalIDs)
    animalID = animalIDs{a};
    if isfield(learningTrajectories.(animalID), 'half') && ~isempty(learningTrajectories.(animalID).half.dprime)
        x = cumsum(learningTrajectories.(animalID).half.nTrials);
        y = learningTrajectories.(animalID).half.dprime;
        plot(x, y, '-', 'Color', indivColor, 'LineWidth', 1);
        [x_unique, ia] = unique(x);
        if length(x_unique) >= 2
            interpolated_half(:, a) = interp1(x_unique, y(ia), common_x_half);
        end
    end
end

% Calculate and plot mean + SEM
mean_y_half = mean(interpolated_half, 2, 'omitnan');
n_animals_half = sum(~isnan(interpolated_half), 2);
sem_y_half = std(interpolated_half, 0, 2, 'omitnan') ./ sqrt(n_animals_half);
sem_y_half(n_animals_half < 2) = 0;
valid_idx_half = ~isnan(mean_y_half);
h2 = shadedErrorBar(common_x_half(valid_idx_half), mean_y_half(valid_idx_half), sem_y_half(valid_idx_half), ...
    'lineprops', {'-', 'Color', meanColorHalf, 'LineWidth', 2});

set(gca, 'TickDir', 'out', 'LineWidth', 1); box off; xlim([0, maxTrialsHalf]);
legend([h2.mainLine], {'Group Mean'}, 'Location', 'best');

%% Plotting Performance Rates and d' vs. Cumulative Trials (0.5 Reward Prob Only)
% This section generates a multi-panel figure suitable for publication.
% Each panel shows individual animal learning trajectories in faint gray,
% with the group average and SEM shown as a thick black line with shading.

figure('Color', 'w'); % Use a white background
t = tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle(t, 'Learning Trajectories (Reward Prob = 0.5)', 'FontWeight', 'bold');

% --- Data to plot ---
metrics = {'hitRate', 'missRate', 'faRate', 'crRate', 'dprime'};
yLabels = {'Hit Rate', 'Miss Rate', 'FA Rate', 'CR Rate', 'd'''};
tileIndices = {1, 2, 3, 4, [5, 6]}; % d' plot spans the bottom row

% Define colors
meanColor = 'k'; % Black for the average

% --- Loop through each metric and create the plots ---
for i = 1:length(metrics)
    metricName = metrics{i};
    
    % Go to the correct tile in the layout
    if strcmp(metricName, 'dprime')
        nexttile(tileIndices{i}(1), [1, 2]); % Span 1 row, 2 columns
    else
        nexttile(tileIndices{i});
    end
    hold on;
    
    % Store data for interpolation
    interpolated_y = nan(length(common_x_half), length(animalIDs));
    
    % Plot individual animal trajectories and collect data for interpolation
    for a = 1:length(animalIDs)
        animalID = animalIDs{a};
        if isfield(learningTrajectories.(animalID), 'half') && ~isempty(learningTrajectories.(animalID).half.(metricName))
            x = cumsum(learningTrajectories.(animalID).half.nTrials);
            y = learningTrajectories.(animalID).half.(metricName);
            
            % Plot individual line
            if ~isempty(x)
                plot(x, y, '-', 'Color', indivColor, 'LineWidth', 1);
            end
            
            % Interpolate data for averaging
            [x_unique, ia, ~] = unique(x);
            y_unique = y(ia);
            if length(x_unique) >= 2
                interpolated_y(:, a) = interp1(x_unique, y_unique, common_x_half, 'linear');
            end
        end
    end
    
    % Calculate mean and SEM
    mean_y = mean(interpolated_y, 2, 'omitnan');
    n_animals_per_bin = sum(~isnan(interpolated_y), 2);
    sem_y = std(interpolated_y, 0, 2, 'omitnan') ./ sqrt(n_animals_per_bin);
    sem_y(n_animals_per_bin < 2) = NaN; % Don't show SEM for fewer than 2 data points

    % Plot average trajectory with shaded error bar
    valid_idx = ~isnan(mean_y);
    if any(valid_idx)
        shadedErrorBar(common_x_half(valid_idx), mean_y(valid_idx), sem_y(valid_idx), ...
                       'lineprops', {'-', 'Color', meanColor, 'LineWidth', 2.5}, ...
                       'patchSaturation', 0.15);
    end
    
    % --- Final plot formatting ---
    box off;
    set(gca, 'TickDir', 'out', 'LineWidth', 1.5, 'FontSize', 12);
    xlabel('Cumulative Trials');
    ylabel(yLabels{i});
    title(yLabels{i});
    
    if contains(metricName, 'Rate')
        ylim([0, 1]);
        yline(0.5, '--', 'Color', [0.5 0.5 0.5]); % Chance line
    end
    
    xlim([0, maxTrialsHalf]);
    grid on;
    hold off;
end


%% Helper Functions
function dprime = computeDprime(vr)
    % Process only non-default trials.
    trialIndices = (vr.cfg.numDefaultTrials+1):length(vr.trialLog);
    
    % Determine go trials on an entire session basis.
    if isfield(vr.cfg, 'rewardEligibility') && ~isempty(trialIndices) && (numel(vr.cfg.rewardEligibility) >= max(trialIndices))
        % For newer sessions, use the rewardEligibility field directly.
        goFlag = vr.cfg.rewardEligibility(trialIndices);
    else
        % For older sessions, use the stimulus field.
        stimuli = cellfun(@(x) x.stimulus, vr.trialLog(trialIndices));
        if any(stimuli > 2)  % assume orientations in degrees
            if vr.cfg.rewardedOrientation == 1
                goFlag = (stimuli < 45);
            elseif vr.cfg.rewardedOrientation == 2
                goFlag = (stimuli > 45);
            else
                error('Invalid rewardedOrientation in cfg.');
            end
        else
            % If stimuli are coded as 1 or 2.
            goFlag = (stimuli == vr.cfg.rewardedOrientation);
        end
    end
    % Now, get go and no-go trial indices.
    goTrials = trialIndices(goFlag);
    noGoTrials = trialIndices(~goFlag);
    
    % For each set, extract performance info.
    if isempty(goTrials)
        hits = 0;
        misses = 0;
    else
        rewardGivenGo = cellfun(@(x) x.rewardGiven, vr.trialLog(goTrials));
        hits = sum(rewardGivenGo);
        misses = numel(goTrials) - sum(rewardGivenGo);
    end
    if isempty(noGoTrials)
        fa = 0;
        cr = 0;
    else
        % For no-go trials, a false alarm is recorded if the trial has a field 'falseAlarm' that is true.
        fa = sum(cellfun(@(x) isfield(x, 'falseAlarm') && x.falseAlarm, vr.trialLog(noGoTrials)));
        cr = numel(noGoTrials) - fa;
    end
    % If the rewarded trial proportion is exactly 0.5,
    % compute d' directly without bootstrapping.
    if vr.cfg.rewardedTrialProportion == 0.5
        if (hits + misses) == 0
            hitRate = 0;
        else
            hitRate = hits / (hits + misses);
        end
        if (fa + cr) == 0
            faRate = 0;
        else
            faRate = fa / (fa + cr);
        end
        if hitRate == 1
            hitRate = 1 - 0.5/(hits + misses);
        elseif hitRate == 0
            hitRate = 0.5/(hits + misses);
        end
        if faRate == 0
            faRate = 0.5/(fa + cr);
        elseif faRate == 1
            faRate = 1 - 0.5/(fa + cr);
        end
        dprime = norminv(hitRate) - norminv(faRate);
    else
        % For sessions with rewardedTrialProportion > 0.5, perform bootstrapping.
        N_go = numel(goTrials);
        N_noGo = numel(noGoTrials);
        N_min = min(N_go, N_noGo);
        nBoot = 50;  % adjust as needed
        dprimeVals = nan(nBoot,1);
        
        for i = 1:nBoot
            if N_go > 0
                sampled_go = randsample(1:N_go, N_min);
                sampled_hits = sum(rewardGivenGo(sampled_go));
                sampled_misses = N_min - sampled_hits;
            else
                sampled_hits = 0;
                sampled_misses = 0;
            end
            
            if N_noGo > 0
                sampled_noGo = randsample(1:N_noGo, N_min);
                sampled_fa = sum(cellfun(@(x) isfield(x, 'falseAlarm') && x.falseAlarm, vr.trialLog(noGoTrials(sampled_noGo))));
                sampled_cr = N_min - sampled_fa;
            else
                sampled_fa = 0;
                sampled_cr = 0;
            end
            
            if (sampled_hits + sampled_misses) == 0
                hitRate = 0;
            else
                hitRate = sampled_hits / (sampled_hits + sampled_misses);
            end
            if (sampled_fa + sampled_cr) == 0
                faRate = 0;
            else
                faRate = sampled_fa / (sampled_fa + sampled_cr);
            end
            
            if hitRate == 1
                hitRate = 1 - 0.5/(sampled_hits + sampled_misses);
            elseif hitRate == 0
                hitRate = 0.5/(sampled_hits + sampled_misses);
            end
            if faRate == 0
                faRate = 0.5/(sampled_fa + sampled_cr);
            elseif faRate == 1
                faRate = 1 - 0.5/(sampled_fa + sampled_cr);
            end
            
            dprimeVals(i) = norminv(hitRate) - norminv(faRate);
        end
        
        dprime = mean(dprimeVals);
    end
end
function [hitRate, missRate, faRate, crRate] = computeRates(vr)
    % Process only non-default trials.
    trialIndices = (vr.cfg.numDefaultTrials+1):length(vr.trialLog);
    
    % Determine go trials.
    if isfield(vr.cfg, 'rewardEligibility') && ~isempty(trialIndices) && (numel(vr.cfg.rewardEligibility) >= max(trialIndices))
        goFlag = vr.cfg.rewardEligibility(trialIndices);
    else
        stimuli = cellfun(@(x) x.stimulus, vr.trialLog(trialIndices));
        if any(stimuli > 2)
            if vr.cfg.rewardedOrientation == 1
                goFlag = (stimuli < 45);
            elseif vr.cfg.rewardedOrientation == 2
                goFlag = (stimuli > 45);
            else
                error('Invalid rewardedOrientation in cfg.');
            end
        else
            goFlag = (stimuli == vr.cfg.rewardedOrientation);
        end
    end
    
    goTrials = trialIndices(goFlag);
    noGoTrials = trialIndices(~goFlag);
    
    if isempty(goTrials)
        hits = 0; misses = 0; n_go = 0;
    else
        rewardGivenGo = cellfun(@(x) x.rewardGiven, vr.trialLog(goTrials));
        hits = sum(rewardGivenGo);
        misses = numel(goTrials) - sum(rewardGivenGo);
        n_go = numel(goTrials);
    end
    if isempty(noGoTrials)
        fa = 0; cr = 0; n_noGo = 0;
    else
        fa = sum(cellfun(@(x) isfield(x, 'falseAlarm') && x.falseAlarm, vr.trialLog(noGoTrials)));
        cr = numel(noGoTrials) - fa;
        n_noGo = numel(noGoTrials);
    end
    
    if n_go > 0
        hitRate = hits/n_go;
        missRate = misses/n_go;
    else
        hitRate = NaN; missRate = NaN;
    end
    if n_noGo > 0
        faRate = fa/n_noGo;
        crRate = cr/n_noGo;
    else
        faRate = NaN; crRate = NaN;
    end
end
function hits = computeHits(vr)
    hits = 0;
    for t = (vr.cfg.numDefaultTrials+1):length(vr.trialLog)
        trial = vr.trialLog{t};
        if isGoTrial(trial, vr.cfg) && trial.rewardGiven
            hits = hits + 1;
        end
    end
end
function misses = computeMisses(vr)
    misses = 0;
    for t = (vr.cfg.numDefaultTrials+1):length(vr.trialLog)
        trial = vr.trialLog{t};
        if isGoTrial(trial, vr.cfg) && ~trial.rewardGiven
            misses = misses + 1;
        end
    end
end
function fa = computeFalseAlarms(vr)
    fa = 0;
    for t = (vr.cfg.numDefaultTrials+1):length(vr.trialLog)
        trial = vr.trialLog{t};
        if ~isGoTrial(trial, vr.cfg) && isfield(trial, 'falseAlarm') && trial.falseAlarm
            fa = fa + 1;
        end
    end
end
function cr = computeCorrectRejections(vr)
    cr = 0;
    for t = (vr.cfg.numDefaultTrials+1):length(vr.trialLog)
        trial = vr.trialLog{t};
        if ~isGoTrial(trial, vr.cfg) && (~isfield(trial, 'falseAlarm') || ~trial.falseAlarm)
            cr = cr + 1;
        end
    end
end
function go = isGoTrial(trial, cfg)
    s = trial.stimulus;
    % If stimulus is numeric and greater than 2, assume it is an orientation (in degrees).
    if isnumeric(s) && s > 2
        if cfg.rewardedOrientation == 1
            go = (s < 45);
        elseif cfg.rewardedOrientation == 2
            go = (s > 45);
        else
            error('Invalid rewardedOrientation in cfg.');
        end
    else
        % Otherwise, assume a binary code (e.g. 1 or 2).
        go = (s == cfg.rewardedOrientation);
    end
end