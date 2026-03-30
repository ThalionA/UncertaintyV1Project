function code = NewExperimentTheo25_full1_v2
% NewExperimentTheo25_full   Code for the ViRMEn experiment with two session types:
%   - Basic: only “easy” stimuli (0° or 90°, contrast=0.99, dispersion=5°).
%   - Full: a small block of “easy” stimuli first, then all 9 orientations × 8 (contrast, dispersion) pairs,
%           sampled with a geometric bias toward low dispersion / high contrast, respecting max‐run.
%
% Begin header code – DO NOT EDIT
code.initialization = @initializationCodeFun;
code.runtime        = @runtimeCodeFun;
code.termination    = @terminationCodeFun;
% End header code – DO NOT EDIT

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
            'Grating phase duration (s)', 'Movement gain', ...
            'Exponential Factor (higher = more licks)', ...
            'Dispersion scalar factor (0-1, lower biases towards 5deg dispersion)'}; 
        dlgtitle = 'Session Parameters';
        dims = [1 50];
        definput = {'Cb01','20','250','10','exponential','0.5','10','1','0.5','4','2','1','1.33','0.5'};
        answer = inputdlg(prompt, dlgtitle, dims, definput);
        if isempty(answer)
            error('User cancelled session parameter input.');
        end

        % Assign session parameters
        cfg.animalID               = answer{1};
        cfg.maxSessionLength       = str2double(answer{2});
        cfg.numTrials              = str2double(answer{3});
        cfg.maxConsecutiveErrors   = str2double(answer{4});
        rewardLickType             = lower(strtrim(answer{5}));
        cfg.valveOpenDuration      = str2double(answer{6});
        cfg.numDefaultTrials       = str2double(answer{7});
        cfg.rewardedOrientation    = str2double(answer{8});
        cfg.rewardedTrialProportion= str2double(answer{9});
        cfg.timeoutDuration        = str2double(answer{10});
        cfg.gratingDuration        = str2double(answer{11});
        cfg.movement_gain          = str2double(answer{12});
        cfg.exp_factor             = str2double(answer{13});
        cfg.dispersionScalar       = str2double(answer{14});

        switch rewardLickType
            case 'ones'
                cfg.requiredLicksVector = ones(1, cfg.numTrials);
            case 'exponential'
                cfg.requiredLicksVector = geornd(1/cfg.exp_factor, 1, cfg.numTrials) + 1;
            otherwise
                error('Invalid input for Licks required for reward. Use "ones" or "exponential".');
        end

        %% World & Grating/Stimulus Setup
        cfg.corridorWorld = 1;
        cfg.darkWorld     = 2;

        % Full set of 9 grating orientations
        fullGratingOrientations = [0, 15, 30, 40, 45, 50, 60, 75, 90];
        cfg.fullGratingOrientations = fullGratingOrientations;

        % The 8 (contrast, dispersion) pairs
        pairs = [ ...
            1.00    5 ;
            1.00   45 ;
            0.5    30 ;
            0.5    90 ;
            0.25   5 ;
            0.25  45 ;
            0.01  30 ;
            0.01  90  ];
        cfg.fullPairs = pairs;   % 8×2 array: [contrast, dispersion]

        %% Session Type Selection
        sessionTypes = {'Basic', 'Full'};
        [ind, ok] = listdlg('PromptString','Select session type:', ...
            'SelectionMode','single','ListString',sessionTypes);
        if ok
            sessionType = sessionTypes{ind};
            fprintf('Selected session type: %s\n', sessionType);
        else
            error('No session type selected.');
        end
        cfg.sessionType = sessionType;

        %% Precompute sorting of all pairs by (dispersion ↑, contrast ↓)
        allPairs = cfg.fullPairs;  % 8×2
        sortMat = [allPairs(:,2), -allPairs(:,1)];  % [dispersion, -contrast]
        [~, sortOrderAll] = sortrows(sortMat, [1,2]);
        cfg.sortedPairIdxAll = sortOrderAll;  % permutation of 1:8

        % Highest‐contrast available at dispersion=5
        idx5 = find(allPairs(:,2) == 5);
        maxContrast5 = max(allPairs(idx5,1));  % = 0.99
        cfg.maxContrast5 = maxContrast5;

        %% Create Trial Stimuli Based on sessionType
        maxRun      = 7;
        maxAttempts = 2e6;
        if cfg.rewardedTrialProportion > 0.5
            maxRun = cfg.numTrials;
        end

        % Preallocate trial arrays
        trialOrientations = zeros(cfg.numTrials, 1);
        trialDispersions  = zeros(cfg.numTrials, 1);
        trialContrasts    = zeros(cfg.numTrials, 1);
        trialRewardFlag   = false(cfg.numTrials, 1);

        switch sessionType
            case 'Basic'
                % ------------------------------------------------------------
                % BASIC session:
                %   - Only orientations {0°, 90°}, dispersion=5°, contrast=0.99
                %   - Enforce maxRun on rewarded vs nonrewarded side.
                % ------------------------------------------------------------
                numRewardedTrials    = round(cfg.numTrials * cfg.rewardedTrialProportion);
                numNonRewardedTrials = cfg.numTrials - numRewardedTrials;

                if cfg.rewardedOrientation == 1
                    rewardedAllowed   = 0;   % (0° < 45°)
                    nonRewardedAllowed= 90;  % (90° > 45°)
                else
                    rewardedAllowed   = 90;  % (90° > 45°)
                    nonRewardedAllowed= 0;   % (0° < 45°)
                end

                rewardedBlockOri    = rewardedAllowed(randi(numel(rewardedAllowed),  numRewardedTrials, 1));
                nonRewardedBlockOri = nonRewardedAllowed(randi(numel(nonRewardedAllowed), numNonRewardedTrials, 1));
                allOri = [rewardedBlockOri; nonRewardedBlockOri];

                % Permute under maxRun constraint
                valid = false;
                for attempt = 1:maxAttempts
                    permIdx = randperm(cfg.numTrials);
                    candOri = allOri(permIdx);
                    if cfg.rewardedOrientation == 1
                        candSide = candOri < 45;
                    else
                        candSide = candOri > 45;
                    end
                    runLength = 1;
                    validCandidate = true;
                    for j = 2:cfg.numTrials
                        if candSide(j) == candSide(j-1)
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
                        trialOrientations = candOri;
                        valid = true;
                        break;
                    end
                end
                if ~valid
                    error('Could not generate a valid sequence for Basic session.');
                end

                % All Basic‐session trials use (dispersion=5°, contrast=0.99)
                trialDispersions(:) = 5;
                trialContrasts(:)   = cfg.maxContrast5;

                % Compute the true reward flag for each orientation
                for t = 1:cfg.numTrials
                    ori = trialOrientations(t);
                    if ori == 45
                        trialRewardFlag(t) = rand < 0.5; 
                    else
                        if cfg.rewardedOrientation == 1
                            trialRewardFlag(t) = (ori < 45);
                        else
                            trialRewardFlag(t) = (ori > 45);
                        end
                    end
                end

            case 'Full'
                % ------------------------------------------------------------
                % FULL session WITH an early “easy” block.
                % Steps:
                %   1) Ask number of easy trials (0° or 90°, 0.99 contrast, 5° dispersion).
                %   2) Generate those N_easy “easy” trials.
                %   3) Generate the remaining trials from all 9×8 grid (geometric bias).
                %   4) Enforce maxRun across [easyBlock; remainder].
                % ------------------------------------------------------------

                % 1) Prompt for number of easy trials
                answerEasy = inputdlg({'Number of initial easy trials (0 or 90° only):'}, ...
                                'Full Session: Easy Trials', [1 30], {'20'});
                if isempty(answerEasy)
                    error('User cancelled easy trial input.');
                end
                N_easy = str2double(answerEasy{1});
                if isnan(N_easy) || N_easy < 0 || N_easy > cfg.numTrials
                    error('Invalid number of easy trials.');
                end

                % 2) Build the “easy” block
                easyOri  = zeros(N_easy,1);
                easyDisp = 5 * ones(N_easy,1);
                easyCont = cfg.maxContrast5 * ones(N_easy,1);
                easyFlag = false(N_easy,1);
                for i = 1:N_easy
                    if rand < 0.5
                        easyOri(i) = 0;
                    else
                        easyOri(i) = 90;
                    end
                    if cfg.rewardedOrientation == 1
                        easyFlag(i) = (easyOri(i) < 45);
                    else
                        easyFlag(i) = (easyOri(i) > 45);
                    end
                end
                rewardedEasy   = sum(easyFlag);
                remainingTrials= cfg.numTrials - N_easy;
                numRewardedTotal   = round(cfg.numTrials * cfg.rewardedTrialProportion);
                remRewardedCount   = numRewardedTotal - rewardedEasy;
                remNonRewardedCount= remainingTrials - remRewardedCount;
                if remRewardedCount < 0 || remNonRewardedCount < 0
                    error('Too many easy trials relative to rewarded proportion.');
                end

                % 3) Build the “remainder” block from all 9 orientations × 8 pairs
                allowedOrientations = cfg.fullGratingOrientations;  % all 9
                allowedNo45 = allowedOrientations(allowedOrientations ~= 45);
                if cfg.rewardedOrientation == 1
                    candR_ori = allowedNo45(allowedNo45 < 45);
                    candNR_ori = allowedNo45(allowedNo45 > 45);
                else
                    candR_ori = allowedNo45(allowedNo45 > 45);
                    candNR_ori = allowedNo45(allowedNo45 < 45);
                end
                candR_ori = [candR_ori, 45];  % include 45° as rewarded candidate

                % Preallocate the remainder arrays
                remOri      = zeros(remainingTrials,1);
                remDisp     = zeros(remainingTrials,1);
                remCont     = zeros(remainingTrials,1);
                remFlag     = false(remainingTrials,1);

                % Geometric weights over 8 pairs
                kPairs = size(cfg.fullPairs,1);  % 8
                rawW   = (cfg.dispersionScalar).^(0:(kPairs-1));
                wts    = rawW / sum(rawW);

                % Fill “would‐be rewarded” portion
                for i = 1:remRewardedCount
                    remOri(i)  = candR_ori(randi(numel(candR_ori)));
                    remFlag(i) = true;
                    r = rand; cumP = cumsum(wts);
                    pickIdx = find(r <= cumP, 1, 'first');
                    pairIdx = cfg.sortedPairIdxAll(pickIdx);
                    remDisp(i) = cfg.fullPairs(pairIdx, 2);
                    remCont(i) = cfg.fullPairs(pairIdx, 1);
                end
                % Fill “would‐be nonrewarded” portion
                for i = 1:remNonRewardedCount
                    idx = remRewardedCount + i;
                    remOri(idx)  = candNR_ori(randi(numel(candNR_ori)));
                    remFlag(idx) = false;
                    r = rand; cumP = cumsum(wts);
                    pickIdx = find(r <= cumP, 1, 'first');
                    pairIdx = cfg.sortedPairIdxAll(pickIdx);
                    remDisp(idx) = cfg.fullPairs(pairIdx, 2);
                    remCont(idx) = cfg.fullPairs(pairIdx, 1);
                end

                % 4) Enforce maxRun across [easyFlag; remFlag] with boundary check
                combinedFlag = [easyFlag; remFlag];
                valid = false;
                for attempt = 1:maxAttempts
                    permIdx = randperm(remainingTrials);
                    cFlag = remFlag(permIdx);

                    % Check boundary run length
                    runLen = 1;
                    ok = true;
                    if N_easy > 0
                        if combinedFlag(N_easy) == cFlag(1)
                            runLen = 2;
                        end
                        if runLen > maxRun
                            ok = false;
                        end
                    end

                    if ok
                        runLen = max(runLen, 1);
                        for j = 2:remainingTrials
                            if cFlag(j) == cFlag(j-1)
                                runLen = runLen + 1;
                                if runLen > maxRun
                                    ok = false;
                                    break;
                                end
                            else
                                runLen = 1;
                            end
                        end
                    end

                    if ok
                        trialOrientations = [easyOri;          remOri(permIdx)];
                        trialDispersions  = [easyDisp;         remDisp(permIdx)];
                        trialContrasts    = [easyCont;         remCont(permIdx)];
                        trialRewardFlag   = [easyFlag;         cFlag];
                        valid = true;
                        break;
                    end
                end
                if ~valid
                    error('Could not generate a valid “Full + easy” sequence under maxRun.');
                end

            otherwise
                error('Unknown session type.');
        end

        % Store into cfg
        cfg.trialOrientations     = trialOrientations;
        cfg.trialDispersions      = trialDispersions;
        cfg.trialContrasts        = trialContrasts;
        cfg.trialRewardAssignment = trialRewardFlag;

        %% Precompute Reward Eligibility Vector
        rewardEligibility = false(1, cfg.numTrials);
        for t = 1:cfg.numTrials
            ori = cfg.trialOrientations(t);
            if ori == 45
                rewardEligibility(t) = rand < 0.5;
            else
                if cfg.rewardedOrientation == 1
                    rewardEligibility(t) = (ori < 45);
                else
                    rewardEligibility(t) = (ori > 45);
                end
            end
        end
        cfg.rewardEligibility = rewardEligibility;

        %% Other Experimental Parameters
        cfg.punishmentEnabled   = true;
        cfg.rewardZoneStart     = 100;
        cfg.rewardZoneEnd       = 140;
        cfg.minTrialEndPosition = 300;
        cfg.maxTrialEndPosition = 350;

        cfg.inputPort      = 'COM3';
        cfg.outputPort     = 'COM4';
        cfg.inputBaudRate  = 115200;
        cfg.outputBaudRate = 115200;

        cfg.encoderCountsPerRevolution = 2500;
        cfg.wheelDiameter              = 24.2;
        cfg.wheelCircumference         = pi * cfg.wheelDiameter;
        cfg.distancePerCount           = cfg.wheelCircumference / cfg.encoderCountsPerRevolution;
        

        %% Initialize Hardware Connections
        vr.hardware.inputMC  = serialport(cfg.inputPort, cfg.inputBaudRate);
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
        vr.trialLog             = {};
        vr.currentTrialLog      = struct('absStartTime', [], 'time', [], 'position', [], ...
            'velocity', [], 'displacement', [], 'lick', [], 'world', [], ...
            'valveState', [], 'trialPhase', []);

        vr.currentTrial         = 1;
        vr.numTrialsCompleted   = 0;
        vr.trialPhase           = 'grating';
        vr.gratingStartTime     = [];
        vr.trialEndPosition     = cfg.minTrialEndPosition + ...
            (cfg.maxTrialEndPosition - cfg.minTrialEndPosition)*rand;
        vr.falseAlarm           = false;
        vr.currentWorld         = cfg.darkWorld;
        vr.position             = vr.worlds{vr.currentWorld}.startLocation;

        vr.cfg = cfg;
    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RUNTIME code: executes on every iteration.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function vr = runtimeCodeFun(vr)
        vr.iteration   = vr.iteration + 1;
        vr.timeElapsed = toc(vr.sessionTimer);
        vr.dt          = vr.timeElapsed - vr.previousTime;
        vr.previousTime= vr.timeElapsed;

        if vr.valveIsOpen && (toc(vr.sessionTimer) - vr.valveOpenTime >= vr.cfg.valveOpenDuration)
            write(vr.hardware.outputMC, 't', "char");
            vr.valveIsOpen = false;
        end

        frameLick    = 0;
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

                    % Determine stimulus orientation, dispersion, contrast
                    currentOri      = vr.cfg.trialOrientations(vr.currentTrial);
                    currentDisp     = vr.cfg.trialDispersions(vr.currentTrial);
                    currentContrast = vr.cfg.trialContrasts(vr.currentTrial);

                    % Find orientation index
                    oriIdx = find(vr.cfg.fullGratingOrientations == currentOri, 1);
                    if isempty(oriIdx)
                        error('Current orientation %d not found in full list.', currentOri);
                    end
                    % Find pair index (contrast + dispersion)
                    allPairs = vr.cfg.fullPairs;  % 8×2
                    pairIdx = find(allPairs(:,1)==currentContrast & ...
                                   allPairs(:,2)==currentDisp, 1);
                    if isempty(pairIdx)
                        error('Pair (%.2f, %d) not found in fullPairs.', ...
                              currentContrast, currentDisp);
                    end
                    % Map to world index: world = 3 + (pairIdx−1)*9 + oriIdx
                    gratingWorld = 3 + (pairIdx-1)*9 + oriIdx;
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
                        vr.trialEndPosition = vr.cfg.minTrialEndPosition + ...
                            (vr.cfg.maxTrialEndPosition - vr.cfg.minTrialEndPosition)*rand;
                        vr.currentWorld = vr.cfg.darkWorld;
                        vr.position = vr.worlds{vr.currentWorld}.startLocation;
                        vr.dp(:) = 0;
                        vr.velocity = [0,0,0,0];
                        vr.inRewardZone = false;
                        vr.falseAlarm = false;
                        vr.trialLicks = 0;
                    else
                        vr.dp = [0,0,0,0];
                    end
                    return;
                end

                vr.dp = [0, -displacement, 0, 0];
                if vr.dt > 0
                    vr.velocity = [0, displacement/vr.dt, 0, 0];
                else
                    vr.velocity = [0,0,0,0];
                end

                yPos = vr.position(2);
                vr.inRewardZone = (yPos >= vr.cfg.rewardZoneStart && yPos <= vr.cfg.rewardZoneEnd);
                if vr.inRewardZone
                    if ~vr.rewardGivenVector(vr.currentTrial) && ...
                       (vr.currentTrial <= vr.cfg.numDefaultTrials) && isStimulusRewarded
                        write(vr.hardware.outputMC, 'r', "char");
                        vr.valveIsOpen = true;
                        vr.valveOpenTime = vr.timeElapsed;
                        vr.rewardGivenVector(vr.currentTrial) = true;
                    end
                    if (vr.currentTrial > vr.cfg.numDefaultTrials) && isStimulusRewarded
                        vr.trialLicks = vr.trialLicks + frameLick;
                        if ~vr.rewardGivenVector(vr.currentTrial) && ...
                           (vr.trialLicks >= vr.cfg.requiredLicksVector(vr.currentTrial))
                            write(vr.hardware.outputMC, 'r', "char");
                            vr.valveIsOpen = true;
                            vr.valveOpenTime = vr.timeElapsed;
                            vr.rewardGivenVector(vr.currentTrial) = true;
                        end
                    end
                    if (~isStimulusRewarded) && (frameLick == 1) && ...
                       (vr.currentTrial > vr.cfg.numDefaultTrials) && ~vr.falseAlarm
                        vr.falseAlarm = true;
                    end
                else
                    vr.trialLicks = 0;
                end

                if yPos >= vr.trialEndPosition
                    vr.currentTrialLog.stimulus    = vr.cfg.trialOrientations(vr.currentTrial);
                    vr.currentTrialLog.dispersion  = vr.cfg.trialDispersions(vr.currentTrial);
                    vr.currentTrialLog.contrast    = vr.cfg.trialContrasts(vr.currentTrial);
                    vr.currentTrialLog.isDefault   = (vr.currentTrial <= vr.cfg.numDefaultTrials);
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
                        vr.currentTrialLog = struct('absStartTime', [], 'time', [], 'position', [], ...
                            'velocity', [], 'displacement', [], 'lick', [], 'world', [], ...
                            'valveState', [], 'trialPhase', []);
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
                    vr.trialEndPosition = vr.cfg.minTrialEndPosition + ...
                        (vr.cfg.maxTrialEndPosition - vr.cfg.minTrialEndPosition)*rand;
                    vr.currentWorld = vr.cfg.darkWorld;
                    vr.position = vr.worlds{vr.currentWorld}.startLocation;
                    vr.dp(:) = 0;
                    vr.velocity = [0,0,0,0];
                    vr.inRewardZone = false;
                    vr.falseAlarm = false;
                    vr.currentTrialLog = struct('absStartTime', [], 'time', [], 'position', [], ...
                        'velocity', [], 'displacement', [], 'lick', [], 'world', [], ...
                        'valveState', [], 'trialPhase', []);
                end

            otherwise
                % No other phases
        end

        % Log continuous time/position/etc.
        vr.currentTrialLog.time(end+1)        = vr.timeElapsed;
        vr.currentTrialLog.position(:, end+1) = vr.position;
        vr.currentTrialLog.velocity(:, end+1) = vr.velocity;
        vr.currentTrialLog.displacement(end+1)= displacement;
        vr.currentTrialLog.lick(end+1)        = frameLick;
        vr.currentTrialLog.world(end+1)       = vr.currentWorld;
        vr.currentTrialLog.valveState(end+1)  = vr.valveIsOpen;
        vr.currentTrialLog.trialPhase{end+1}  = vr.trialPhase;

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
        save(fname, 'vr', '-v7.3');
        disp(['Experiment completed and data saved as ' fname '.']);

        hits = computeHits(vr);
        misses = computeMisses(vr);
        falseAlarms = computeFalseAlarms(vr);
        correctRejections = computeCorrectRejections(vr);

        n_signal = hits + misses;
        n_noise  = falseAlarms + correctRejections;
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
