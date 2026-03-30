function code = NewExperimentTheo25_d
% NewExperimentTheo25   Code for the ViRMEn experiment.
%   This version implements:
%     1. Oriented gratings with varying dispersions are now shown through ViRMEn.
%        - World 1: corridor.
%        - World 2: dark world (timeout).
%        - Worlds 3-38: grating worlds, where each of 9 orientations is paired with 4 dispersion levels.
%          The worlds are arranged as:
%              Worlds 3-11: 5deg dispersion,
%              Worlds 12-20: 30deg dispersion,
%              Worlds 21-29: 45deg dispersion,
%              Worlds 30-38: 90deg dispersion.
%     2. Session types determine which grating orientations are available:
%           Basic Learning: [0, 90] (5deg dispersion only).
%           Orientation Learning 1: [0, 15, 75, 90]
%           Orientation Learning 2: [0, 15, 30, 60, 75, 90]
%           Orientation Learning 3: [0, 15, 30, 40, 45, 50, 60, 75, 90]
%           Complete: (placeholder, not implemented)
%     3. The decision boundary remains at 45°:
%           rewarded orientation = 1 means stimulus < 45 are rewarded,
%           rewarded orientation = 2 means stimulus > 45.
%     4. For Orientation Learning sessions, the initial "easy" trials use only 5deg dispersion,
%        while the remaining trials use a geometric probability distribution over dispersions,
%        with a scalar factor controlling the weighting.
%
% Begin header code - DO NOT EDIT
code.initialization = @initializationCodeFun;
code.runtime        = @runtimeCodeFun;
code.termination    = @terminationCodeFun;
% End header code - DO NOT EDIT

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION code: executes before the ViRMEn engine starts.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function vr = initializationCodeFun(vr)
        %% Session Parameter Input
        prompt = {'Animal ID', 'Session length (minutes)', 'Number of trials', ...
            'Max consecutive errors', 'Licks required for reward (ones/exponential)', ...
            'Valve open duration (s)', 'Number of default rewarded trials', ...
            'Rewarded orientation (1 = horizontal, 2 = vertical)', ...
            'Rewarded trial proportion (0-1)', 'Timeout duration (s)', ...
            'Grating phase duration (s)', 'Movement gain', 'Exponential Factor (higher = more licks)', ...
            'Dispersion scalar factor (0-1, lower biases towards 5deg dispersion)'};
        dlgtitle = 'Session Parameters';
        dims = [1 50];
        definput = {'Cb01', '15', '200', '10', 'exponential', '0.5', '10', '1', '0.5', '4', '2', '1', '1.33', '0.35'};
        answer = inputdlg(prompt, dlgtitle, dims, definput);
        if isempty(answer)
            error('User cancelled session parameter input.');
        end

        % Assign session parameters.
        cfg.animalID              = answer{1};
        cfg.maxSessionLength      = str2double(answer{2});
        cfg.numTrials             = str2double(answer{3});
        cfg.maxConsecutiveErrors  = str2double(answer{4});
        rewardLickType = lower(strtrim(answer{5}));

        cfg.valveOpenDuration     = str2double(answer{6});
        cfg.numDefaultTrials      = str2double(answer{7});
        cfg.rewardedOrientation   = str2double(answer{8});
        cfg.rewardedTrialProportion = str2double(answer{9});
        cfg.timeoutDuration       = str2double(answer{10});
        cfg.gratingDuration       = str2double(answer{11});
        cfg.movement_gain         = str2double(answer{12});
        cfg.exp_factor            = str2double(answer{13});
        cfg.dispersionScalar        = str2double(answer{14});

        switch rewardLickType
            case 'ones'
                cfg.requiredLicksVector = ones(1, cfg.numTrials);
            case 'exponential'
                cfg.requiredLicksVector = geornd(1/cfg.exp_factor, 1, cfg.numTrials) + 1;
            otherwise
                error('Invalid input for Licks required for reward. Use "ones" or "exponential".');
        end

        %% World & Grating Setup
        % Define world indices:
        %   World 1: corridor.
        %   World 2: dark world (timeout).
        %   Worlds 3-38: grating worlds.
        % Each of 9 orientations is paired with 4 dispersion levels:
        %   Worlds 3-11: 5deg disperion,
        %   Worlds 12-20: 30deg dispersion,
        %   Worlds 21-29: 45deg dispersion,
        %   Worlds 30-38: 90deg dispersion.
        cfg.corridorWorld = 1;
        cfg.darkWorld = 2;
        % Full set of grating orientations corresponding to the 9 orientations.
        fullGratingOrientations = [0, 15, 30, 40, 45, 50, 60, 75, 90];
        cfg.fullGratingOrientations = fullGratingOrientations;
        % Define dispersion levels.
        dispersionLevels = [5, 30, 45, 90];
        cfg.dispersionLevels = dispersionLevels;

        %% Session Type Selection and Allowed Orientations
        % --- Session Type Selection and Allowed Dispersions ---
        sessionTypes = {'Basic Learning', 'Orientation Learning 1', ...
            'Orientation Learning 2', 'Orientation Learning 3', ...
            'Dispersion Learning 1', 'Dispersion Learning 2', 'Complete'};
        [ind, ok] = listdlg('PromptString', 'Select session type:', ...
            'SelectionMode', 'single', 'ListString', sessionTypes);
        if ok
            sessionType = sessionTypes{ind};
            fprintf('Selected session type: %s\n', sessionType);
        else
            error('No session type selected.');
        end

        switch sessionType
            case 'Basic Learning'
                allowedOrientations = [0, 90];
                allowedDispersionLevels = 5;  % 5deg dispersion only
            case 'Orientation Learning 1'
                allowedOrientations = [0, 15, 75, 90];
                allowedDispersionLevels = 5;
            case 'Orientation Learning 2'
                allowedOrientations = [0, 15, 30, 60, 75, 90];
                allowedDispersionLevels = 5;
            case 'Orientation Learning 3'
                allowedOrientations = [0, 15, 30, 40, 45, 50, 60, 75, 90];
                allowedDispersionLevels = 5;
            case {'Dispersion Learning 1', 'Dispersion Learning 2'}
                allowedOrientations = [0, 15, 30, 40, 45, 50, 60, 75, 90];
                if strcmp(sessionType, 'Dispersion Learning 1')
                    allowedDispersionLevels = [5, 30, 45];   % Exclude 90deg dispersion
                else % 'Dispersion Learning 2'
                    allowedDispersionLevels = [5, 30, 45, 90];  % Include all dispersions
                end
            case 'Complete'
                error('Complete session not implemented yet.');
            otherwise
                error('Unknown session type.');
        end
        cfg.allowedOrientations = allowedOrientations;
        cfg.allowedDispersionLevels = allowedDispersionLevels;

        % Map allowed orientations to their corresponding grating world indices for 5deg dispersion.
        allowedWorldIndices = zeros(size(allowedOrientations));
        for ii = 1:length(allowedOrientations)
            idx = find(fullGratingOrientations == allowedOrientations(ii));
            if isempty(idx)
                error('Allowed orientation %d not found in full list.', allowedOrientations(ii));
            end
            % For 5deg dispersion, world index = idx + 2.
            allowedWorldIndices(ii) = idx + 2;
        end
        cfg.allowedWorldIndices = allowedWorldIndices;

        %% Create Trial Stimuli Based on Allowed Orientations and Dispersions
        % For Orientation Learning sessions, initial "easy" trials use only 5deg dispersion.
        % For remaining trials, dispersions are chosen from a geometric probability distribution.
        maxRun = 7;
        maxAttempts = 2e6;

        if cfg.rewardedTrialProportion > 0.5
            maxRun = cfg.numTrials;
        end

        if strcmp(sessionType, 'Basic Learning')
            % In Basic Learning all stimuli come from [0,90] at 5deg
            % dispersion
            numRewardedTrials = round(cfg.numTrials * cfg.rewardedTrialProportion);
            numNonRewardedTrials = cfg.numTrials - numRewardedTrials;
            if cfg.rewardedOrientation == 1
                rewardedAllowed = allowedOrientations(allowedOrientations < 45);
                nonRewardedAllowed = allowedOrientations(allowedOrientations > 45);
            elseif cfg.rewardedOrientation == 2
                rewardedAllowed = allowedOrientations(allowedOrientations > 45);
                nonRewardedAllowed = allowedOrientations(allowedOrientations < 45);
            else
                error('Invalid input for rewarded orientation.');
            end
            rewardedBlock = zeros(numRewardedTrials, 1);
            for i = 1:numRewardedTrials
                rewardedBlock(i) = rewardedAllowed(randi(numel(rewardedAllowed)));
            end
            nonRewardedBlock = zeros(numNonRewardedTrials, 1);
            for i = 1:numNonRewardedTrials
                nonRewardedBlock(i) = nonRewardedAllowed(randi(numel(nonRewardedAllowed)));
            end
            allTrials = [rewardedBlock; nonRewardedBlock];

            valid = false;
            for attempt = 1:maxAttempts
                perm = randperm(cfg.numTrials);
                candidate = allTrials(perm);
                if cfg.rewardedOrientation == 1
                    candidateSides = candidate < 45;
                else
                    candidateSides = candidate > 45;
                end
                runLength = 1;
                validCandidate = true;
                for j = 2:length(candidateSides)
                    if candidateSides(j) == candidateSides(j-1)
                        runLength = runLength + 1;
                        if runLength > maxRun
                            validCandidate = false;
                            break;
                        end
                    else
                        runLength = 1;
                    end
                end
                if validCandidate
                    trialStimuli = candidate;
                    valid = true;
                    break;
                end
            end
            if ~valid
                error('Could not generate a valid pseudorandom sequence for Basic Learning with max run constraint.');
            end
            cfg.trialStimuli = trialStimuli;
            % For Basic Learning, all trials are at 5deg dispersion.
            cfg.trialDispersions = 5*ones(cfg.numTrials, 1);
        else
            % For Orientation Learning sessions, prompt for number of initial "easy" trials.
            answer2 = inputdlg({'Number of initial easy trials (0 or 90 degrees only)'}, 'Initial Easy Trials', [1 30], {'50'});
            if isempty(answer2)
                error('User cancelled easy trial parameter input.');
            end
            cfg.numEasyTrials = str2double(answer2{1});

            numRewardedTrials = round(cfg.numTrials * cfg.rewardedTrialProportion);
            nRemaining = cfg.numTrials - cfg.numEasyTrials;

            % Create the easy block from the basic set [0, 90] and compute its reward flag.
            easySet = [0, 90];
            easyBlock = zeros(cfg.numEasyTrials, 1);
            easyFlag = false(cfg.numEasyTrials, 1);
            for i = 1:cfg.numEasyTrials
                easyBlock(i) = easySet(randi(2));
                if cfg.rewardedOrientation == 1
                    easyFlag(i) = (easyBlock(i) < 45);
                elseif cfg.rewardedOrientation == 2
                    easyFlag(i) = (easyBlock(i) > 45);
                end
            end
            rewardedEasy = sum(easyFlag);

            remainingRewarded = numRewardedTrials - rewardedEasy;
            remainingNonRewarded = nRemaining - remainingRewarded;
            if remainingRewarded < 0 || remainingNonRewarded < 0
                error('Number of easy trials exceeds the total rewarded or non-rewarded trial count.');
            end

            if any(allowedOrientations == 45)
                allowedNo45 = allowedOrientations(allowedOrientations ~= 45);
                has45 = true;
            else
                allowedNo45 = allowedOrientations;
                has45 = false;
            end

            if cfg.rewardedOrientation == 1
                candidatesRewarded = allowedNo45(allowedNo45 < 45);
                candidatesNonRewarded = allowedNo45(allowedNo45 > 45);
            elseif cfg.rewardedOrientation == 2
                candidatesRewarded = allowedNo45(allowedNo45 > 45);
                candidatesNonRewarded = allowedNo45(allowedNo45 < 45);
            end
            if has45
                candidatesRewarded = [candidatesRewarded, 45];
                % candidatesNonRewarded = [candidatesNonRewarded, 45];
            end

            rewardedBlock = zeros(remainingRewarded, 1);
            rewardedFlag = true(remainingRewarded, 1);
            rewardedDispersions = zeros(remainingRewarded, 1);
            nonRewardedBlock = zeros(remainingNonRewarded, 1);
            nonRewardedFlag = false(remainingNonRewarded, 1);
            nonRewardedDispersions = zeros(remainingNonRewarded, 1);
            
            % Define weights for dispersion selection using geometric distribution.
            weights = (cfg.dispersionScalar).^(0:(length(cfg.allowedDispersionLevels)-1));
            weights = weights / sum(weights);

            for i = 1:remainingRewarded
                rewardedBlock(i) = candidatesRewarded(randi(numel(candidatesRewarded)));
                r = rand;
                cumP = cumsum(weights);
                dispersionIdx = find(r <= cumP, 1, 'first');
                rewardedDispersions(i) = cfg.allowedDispersionLevels(dispersionIdx);

            end
            for i = 1:remainingNonRewarded
                nonRewardedBlock(i) = candidatesNonRewarded(randi(numel(candidatesNonRewarded)));
                r = rand;
                cumP = cumsum(weights);
                dispersionIdx = find(r <= cumP, 1, 'first');
                nonRewardedDispersions(i) = cfg.allowedDispersionLevels(dispersionIdx);
            end

            remainingTrialsStim = [rewardedBlock; nonRewardedBlock];
            remainingTrialsDispersions = [rewardedDispersions; nonRewardedDispersions];

            valid = false;
            for attempt = 1:maxAttempts
                perm = randperm(nRemaining);
                candidateStim = remainingTrialsStim(perm);
                candidateFlag = [true(remainingRewarded,1); false(remainingNonRewarded,1)];
                candidateFlag = candidateFlag(perm);
                fullCandidateFlag = [easyFlag; candidateFlag];
                runLength = 1;
                validCandidate = true;
                for j = 2:length(fullCandidateFlag)
                    if fullCandidateFlag(j) == fullCandidateFlag(j-1)
                        runLength = runLength + 1;
                        if runLength > maxRun
                            validCandidate = false;
                            break;
                        end
                    else
                        runLength = 1;
                    end
                end
                if validCandidate
                    remainingTrialsStim = candidateStim;
                    remainingTrialsDispersions = remainingTrialsDispersions(perm);
                    valid = true;
                    break;
                end
            end
            if ~valid
                error('Could not generate a valid pseudorandom sequence for Orientation Learning with max run constraint.');
            end

            trialStimuli = [easyBlock; remainingTrialsStim];
            trialDispersions = [5*ones(cfg.numEasyTrials,1); remainingTrialsDispersions];
            trialRewardAssignment = [easyFlag; candidateFlag];
            cfg.trialStimuli = trialStimuli;
            cfg.trialDispersions = trialDispersions;
            cfg.trialRewardAssignment = trialRewardAssignment;
        end

        %% Precompute Reward Eligibility Vector
        rewardEligibility = false(1, cfg.numTrials);
        for trial = 1:cfg.numTrials
            stimOrient = cfg.trialStimuli(trial);
            if stimOrient == 45
                rewardEligibility(trial) = rand < 0.5;
            else
                if cfg.rewardedOrientation == 1
                    rewardEligibility(trial) = (stimOrient < 45);
                elseif cfg.rewardedOrientation == 2
                    rewardEligibility(trial) = (stimOrient > 45);
                end
            end
        end
        cfg.rewardEligibility = rewardEligibility;

        %% Other Experimental Parameters
        cfg.punishmentEnabled   = true;
        cfg.rewardZoneStart     = 100;
        cfg.rewardZoneEnd       = 140;
        cfg.minTrialEndPosition = 200;
        cfg.maxTrialEndPosition = 250;

        cfg.inputPort      = 'COM3';
        cfg.outputPort     = 'COM4';
        cfg.inputBaudRate  = 115200;
        cfg.outputBaudRate = 115200;

        cfg.encoderCountsPerRevolution = 2500;
        cfg.wheelDiameter              = 24.2;
        cfg.wheelCircumference         = pi * cfg.wheelDiameter;
        cfg.distancePerCount           = cfg.wheelCircumference / cfg.encoderCountsPerRevolution;

        vr.cfg = cfg;

        %% Initialize Hardware Connections
        vr.hardware.inputMC = serialport(cfg.inputPort, cfg.inputBaudRate);
        flush(vr.hardware.inputMC);
        vr.hardware.outputMC = serialport(cfg.outputPort, cfg.outputBaudRate);
        flush(vr.hardware.outputMC);
        pause(3);
        disp('Hardware initialized using serialport.');

        %% Initialize Runtime Variables
        vr.sessionTimer         = tic;
        vr.previousTime         = 0;
        vr.iteration            = 0;
        vr.inRewardZone         = false;
        vr.lickCount            = 0;
        vr.zoneLickCount        = 0;
        vr.trialLicks           = 0;
        vr.abortTrial           = false;
        vr.valveIsOpen          = false;
        vr.valveOpenTime        = 0;
        vr.consecutiveErrors    = 0;
        vr.inTimeout            = false;
        vr.timeoutStartTime     = 0;

        vr.rewardGivenVector    = false(1, cfg.numTrials);
        vr.trialLog = {};
        vr.currentTrialLog = struct('absStartTime', [], 'time', [], 'position', [], 'velocity', [], ...
            'displacement', [], 'lick', [], 'world', [], 'valveState', [], 'trialPhase', []);

        vr.currentTrial = 1;
        vr.numTrialsCompleted = 0;
        vr.trialPhase = 'grating';
        vr.gratingStartTime = [];
        vr.trialEndPosition = cfg.minTrialEndPosition + (cfg.maxTrialEndPosition - cfg.minTrialEndPosition)*rand;
        vr.falseAlarm = false;
        vr.currentWorld = cfg.darkWorld;
        vr.position = vr.worlds{vr.currentWorld}.startLocation;
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RUNTIME code: executes on every iteration of the ViRMEn engine.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function vr = runtimeCodeFun(vr)
        vr.iteration = vr.iteration + 1;
        vr.timeElapsed = toc(vr.sessionTimer);
        vr.dt = vr.timeElapsed - vr.previousTime;
        vr.previousTime = vr.timeElapsed;

        if vr.valveIsOpen && (toc(vr.sessionTimer) - vr.valveOpenTime >= vr.cfg.valveOpenDuration)
            write(vr.hardware.outputMC, 't', "char");
            vr.valveIsOpen = false;
        end

        frameLick = 0;
        displacement = 0;
        if vr.hardware.inputMC.NumBytesAvailable > 0
            A = read(vr.hardware.inputMC, vr.hardware.inputMC.NumBytesAvailable, "uint8");
            lickIdx = (A == 49) | (A == 50);
            frameLick = double(any(lickIdx));
            encoderValues = A(~lickIdx);
            diffCount = sum(encoderValues == 1) - sum(encoderValues == 0);
            displacement = diffCount * vr.cfg.distancePerCount * vr.cfg.movement_gain;
        end

        switch vr.trialPhase
            case 'grating'
                if isempty(vr.gratingStartTime)
                    vr.gratingStartTime = vr.timeElapsed;
                    vr.currentTrialLog.absStartTime = datetime('now');
                    % Determine stimulus orientation and dispersion for this trial.
                    currentStim = vr.cfg.trialStimuli(vr.currentTrial);
                    currentDispersion = vr.cfg.trialDispersions(vr.currentTrial);
                    idx = find(vr.cfg.fullGratingOrientations == currentStim, 1);
                    if isempty(idx)
                        error('Current stimulus orientation %d not found in full grating list.', currentStim);
                    end
                    dispersionIdx = find(vr.cfg.dispersionLevels == currentDispersion, 1);
                    if isempty(dispersionIdx)
                        error('Current dispersion %.3f not found in dispersion levels.', currentDispersion);
                    end
                    % Map to world index: world index = 2 + (dispersionIndex-1)*9 + idx.
                    gratingWorld = 2 + (dispersionIdx - 1)*9 + idx;
                    vr.currentWorld = gratingWorld;
                end
                if vr.timeElapsed - vr.gratingStartTime >= vr.cfg.gratingDuration
                    vr.gratingStartTime = [];
                    vr.currentWorld = vr.cfg.corridorWorld;
                    vr.position = vr.worlds{vr.cfg.corridorWorld}.startLocation;
                    vr.trialPhase = 'corridor';
                end

            case 'corridor'
                isStimulusRewarded = vr.cfg.rewardEligibility(vr.currentTrial);
                if vr.inTimeout
                    if vr.timeElapsed - vr.timeoutStartTime >= vr.cfg.timeoutDuration
                        vr.inTimeout = false;
                        vr.trialPhase = 'grating';
                        vr.gratingStartTime = [];
                        vr.trialEndPosition = vr.cfg.minTrialEndPosition + (vr.cfg.maxTrialEndPosition - vr.cfg.minTrialEndPosition)*rand;
                        vr.currentWorld = vr.cfg.darkWorld;
                        vr.position = vr.worlds{vr.currentWorld}.startLocation;
                        vr.dp(:) = 0;
                        vr.velocity = [0, 0, 0, 0];
                        vr.inRewardZone = false;
                        vr.falseAlarm = false;
                        vr.trialLicks = 0;
                    else
                        vr.dp = [0, 0, 0, 0];
                    end
                    return;
                end

                vr.dp = [0, -displacement, 0, 0];
                if vr.dt > 0
                    vr.velocity = [0, displacement/vr.dt, 0, 0];
                else
                    vr.velocity = [0, 0, 0, 0];
                end

                yPos = vr.position(2);
                vr.inRewardZone = (yPos >= vr.cfg.rewardZoneStart && yPos <= vr.cfg.rewardZoneEnd);
                if vr.inRewardZone
                    if ~vr.rewardGivenVector(vr.currentTrial) && (vr.currentTrial <= vr.cfg.numDefaultTrials) && isStimulusRewarded
                        write(vr.hardware.outputMC, 'r', "char");
                        vr.valveIsOpen = true;
                        vr.valveOpenTime = vr.timeElapsed;
                        vr.rewardGivenVector(vr.currentTrial) = true;
                    end
                    if (vr.currentTrial > vr.cfg.numDefaultTrials) && isStimulusRewarded
                        vr.trialLicks = vr.trialLicks + frameLick;
                        if ~vr.rewardGivenVector(vr.currentTrial) && (vr.trialLicks >= vr.cfg.requiredLicksVector(vr.currentTrial))
                            write(vr.hardware.outputMC, 'r', "char");
                            vr.valveIsOpen = true;
                            vr.valveOpenTime = vr.timeElapsed;
                            vr.rewardGivenVector(vr.currentTrial) = true;
                        end
                    end
                    if (~isStimulusRewarded) && (frameLick == 1) && (vr.currentTrial > vr.cfg.numDefaultTrials) && ~vr.falseAlarm
                        vr.falseAlarm = true;
                    end
                else
                    vr.trialLicks = 0;
                end

                if yPos >= vr.trialEndPosition
                    vr.currentTrialLog.stimulus = vr.cfg.trialStimuli(vr.currentTrial);
                    vr.currentTrialLog.dispersion = vr.cfg.trialDispersions(vr.currentTrial);
                    vr.currentTrialLog.isDefault = (vr.currentTrial <= vr.cfg.numDefaultTrials);
                    vr.currentTrialLog.rewardGiven = vr.rewardGivenVector(vr.currentTrial);
                    printSummary = @() fprintf(['Completed %d trials in %.1f min. Active trial performance: ' ...
                        'Hits: %d, Misses: %d, False Alarms: %d, Correct Rejections: %d\n'], ...
                        vr.numTrialsCompleted, vr.timeElapsed/60, ...
                        computeHits(vr), computeMisses(vr), computeFalseAlarms(vr), computeCorrectRejections(vr));
                    if vr.falseAlarm
                        vr.currentTrialLog.falseAlarm = true;
                        vr.trialLog{vr.currentTrial} = vr.currentTrialLog;
                        vr.numTrialsCompleted = vr.numTrialsCompleted + 1;
                        vr.currentTrial = vr.currentTrial + 1;
                        vr.trialLicks = 0;
                        vr.inTimeout = true;
                        vr.timeoutStartTime = vr.timeElapsed;
                        vr.currentWorld = vr.cfg.darkWorld;
                        printSummary();
                        vr.currentTrialLog = struct('absStartTime', [], 'time', [], 'position', [], 'velocity', [], ...
                            'displacement', [], 'lick', [], 'world', [], 'valveState', [], 'trialPhase', []);
                        vr.consecutiveErrors = vr.consecutiveErrors + 1;
                        return;
                    else
                        vr.currentTrialLog.falseAlarm = false;
                        vr.trialLog{vr.currentTrial} = vr.currentTrialLog;
                        vr.numTrialsCompleted = vr.numTrialsCompleted + 1;
                        if vr.cfg.rewardEligibility(vr.currentTrial)
                            if vr.rewardGivenVector(vr.currentTrial)
                                vr.consecutiveErrors = 0;
                            else
                                vr.consecutiveErrors = vr.consecutiveErrors + 1;
                            end
                        else
                            vr.consecutiveErrors = 0;
                        end
                        printSummary();
                    end

                    if vr.consecutiveErrors >= vr.cfg.maxConsecutiveErrors
                        disp('Maximum consecutive errors reached. Ending experiment.');
                        vr.experimentEnded = true;
                        return;
                    end

                    vr.currentTrial = vr.currentTrial + 1;
                    vr.trialLicks = 0;
                    if vr.currentTrial > vr.cfg.numTrials
                        vr.experimentEnded = true;
                        return;
                    end

                    vr.trialPhase = 'grating';
                    vr.gratingStartTime = [];
                    vr.trialEndPosition = vr.cfg.minTrialEndPosition + (vr.cfg.maxTrialEndPosition - vr.cfg.minTrialEndPosition)*rand;
                    vr.currentWorld = vr.cfg.darkWorld;
                    vr.position = vr.worlds{vr.currentWorld}.startLocation;
                    vr.dp(:) = 0;
                    vr.velocity = [0, 0, 0, 0];
                    vr.inRewardZone = false;
                    vr.falseAlarm = false;
                    vr.currentTrialLog = struct('absStartTime', [], 'time', [], 'position', [], 'velocity', [], ...
                        'displacement', [], 'lick', [], 'world', [], 'valveState', [], 'trialPhase', []);
                end

            otherwise
        end

        vr.currentTrialLog.time(end+1) = vr.timeElapsed;
        vr.currentTrialLog.position(:, end+1) = vr.position;
        vr.currentTrialLog.velocity(:, end+1) = vr.velocity;
        vr.currentTrialLog.displacement(end+1) = displacement;
        vr.currentTrialLog.lick(end+1) = frameLick;
        vr.currentTrialLog.world(end+1) = vr.currentWorld;
        vr.currentTrialLog.valveState(end+1) = vr.valveIsOpen;
        vr.currentTrialLog.trialPhase{end+1} = vr.trialPhase;

        if vr.timeElapsed > vr.cfg.maxSessionLength * 60
            disp('Session time exceeded. Ending experiment.');
            vr.experimentEnded = true;
        end
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TERMINATION code: executes after the ViRMEn engine stops.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function vr = terminationCodeFun(vr)
        write(vr.hardware.outputMC, 't', "char");

        if isvalid(vr.hardware.inputMC)
            delete(vr.hardware.inputMC);
        end
        if isvalid(vr.hardware.outputMC)
            delete(vr.hardware.outputMC);
        end
        disp('Hardware connections terminated.');

        dateStr = char(datetime("now", "Format", 'yyyyMMdd'));
        baseFilename = sprintf('vr_%s_%s', dateStr, vr.cfg.animalID);
        fname = [baseFilename, '.mat'];
        n = 1;
        while exist(fname, 'file')
            fname = sprintf('%s_%d.mat', baseFilename, n);
            n = n + 1;
        end
        save(fname, 'vr');
        disp(['Experiment completed and data saved as ' fname '.']);

        % plotSessionFigures(vr)

        hits = computeHits(vr);
        misses = computeMisses(vr);
        falseAlarms = computeFalseAlarms(vr);
        correctRejections = computeCorrectRejections(vr);

        n_signal = hits + misses;
        n_noise = falseAlarms + correctRejections;
        if n_signal == 0
            hitRate = 0;
        else
            hitRate = hits / n_signal;
        end
        if n_noise == 0
            faRate = 0;
        else
            faRate = falseAlarms / n_noise;
        end

        if hitRate == 1
            hitRate = 1 - 0.5/n_signal;
        elseif hitRate == 0
            hitRate = 0.5/n_signal;
        end

        if faRate == 0
            faRate = 0.5/n_noise;
        elseif faRate == 1
            faRate = 1 - 0.5/n_noise;
        end

        dprime = norminv(hitRate) - norminv(faRate);
        fprintf('Behavioral d'' = %.2f\n', dprime);

    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions for computing performance statistics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function hits = computeHits(vr)
        hits = 0;
        for t = (vr.cfg.numDefaultTrials+1):length(vr.trialLog)
            if vr.cfg.rewardEligibility(t)
                if vr.trialLog{t}.rewardGiven
                    hits = hits + 1;
                end
            end
        end
    end

    function misses = computeMisses(vr)
        misses = 0;
        for t = (vr.cfg.numDefaultTrials+1):length(vr.trialLog)
            if vr.cfg.rewardEligibility(t)
                if ~vr.trialLog{t}.rewardGiven
                    misses = misses + 1;
                end
            end
        end
    end

    function fa = computeFalseAlarms(vr)
        fa = 0;
        for t = (vr.cfg.numDefaultTrials+1):length(vr.trialLog)
            if ~vr.cfg.rewardEligibility(t)
                if isfield(vr.trialLog{t}, 'falseAlarm') && vr.trialLog{t}.falseAlarm
                    fa = fa + 1;
                end
            end
        end
    end

    function cr = computeCorrectRejections(vr)
        cr = 0;
        for t = (vr.cfg.numDefaultTrials+1):length(vr.trialLog)
            if ~vr.cfg.rewardEligibility(t)
                if ~isfield(vr.trialLog{t}, 'falseAlarm') || ~vr.trialLog{t}.falseAlarm
                    cr = cr + 1;
                end
            end
        end
    end

end
