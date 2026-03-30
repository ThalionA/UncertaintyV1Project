function code = NewExperimentTheo25_full1_priorblocks
% NewExperimentTheo25_full   Code for the ViRMEn experiment.
%
% MODIFIED for alternating block-based prior manipulation:
%   - Adds a 'grey screen' phase for user-defined duration before each stimulus.
%   - Block 1 (user-defined N): 'Easy' trials (0/90) with 50/50 Go/NoGo prob.
%   - Post-easy: Alternating blocks of 80/20 and 20/80 Go/NoGo priors.
%   - Block lengths are random (user-defined min/max).
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
        % MODIFIED: Added block and grey screen parameters to dialog
        prompt = {'Animal ID', 'Session length (minutes)', 'Number of trials', ...
            'Max consecutive errors', 'Licks required for reward (ones/exponential)', ...
            'Valve open duration (s)', 'Number of default rewarded trials', ...
            'Rewarded orientation (1 = horizontal, 2 = vertical)', ...
            'Number of initial easy trials', ...
            'Minimum block length', ...
            'Maximum block length', ...
            'Grey screen duration (s)', ... % <-- NEW
            'Timeout duration (s)', 'Grating phase duration (s)', 'Movement gain', ...
            'Exponential Factor (higher = more licks)', ...
            'Dispersion scalar factor (0-1, lower biases towards 5deg dispersion)'}; 
        dlgtitle = 'Session Parameters';
        dims = [1 50];
        % MODIFIED: Added defaults for new parameters
        definput = {'Cb01','20','250','10','exponential','0.5','10','1', ...
                    '50', '25', '75', ...
                    '2', ... % <-- NEW default
                    '4','2','1','1.33','0.5'};
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
        
        % Block structure parameters
        cfg.block1_N_easy          = str2double(answer{9});
        cfg.minBlockLen            = str2double(answer{10});
        cfg.maxBlockLen            = str2double(answer{11});
        
        % --- NEW: Grey screen duration ---
        cfg.greyDuration           = str2double(answer{12});
        
        % --- Shifted indices ---
        cfg.timeoutDuration        = str2double(answer{13});
        cfg.gratingDuration        = str2double(answer{14});
        cfg.movement_gain          = str2double(answer{15});
        cfg.exp_factor             = str2double(answer{16});
        cfg.dispersionScalar       = str2double(answer{17});
        
        if cfg.numTrials <= cfg.block1_N_easy
            error('Total number of trials (%d) must be greater than easy block size (%d).', ...
                  cfg.numTrials, cfg.block1_N_easy);
        end

        % --- NEW: Find the Grey World index by name ---
        cfg.greyWorldIdx = 0;
        cfg.greyWorldIdx = numel(vr.worlds);
        if cfg.greyWorldIdx == 0
            error('Could not find the "a0_c0.00_d90" grey world. Did you run the "addGreyWorld.m" script on your .mat file?');
        end
        % fprintf('Found grey world ("%s") at index %d\n', vr.worlds{cfg.greyWorldIdx}.name, cfg.greyWorldIdx);
        
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
        % The 8 (contrast, dispersion) pairs (ORIGINAL SET)
        pairs = [ ...
            1    5 ;
            1   45 ;
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
        idx5 = allPairs(:,2) == 5;
        maxContrast5 = max(allPairs(idx5,1));  % = 1
        cfg.maxContrast5 = maxContrast5;

        %% Create Trial Stimuli Based on sessionType
        
        % Preallocate trial arrays
        trialOrientations = zeros(cfg.numTrials, 1);
        trialDispersions  = zeros(cfg.numTrials, 1);
        trialContrasts    = zeros(cfg.numTrials, 1);
        trialBlockID      = cell(cfg.numTrials, 1); % <-- NEW variable

        switch sessionType
            case 'Basic'
                % ------------------------------------------------------------
                % BASIC session: (Unchanged, uses 50/50 split for cfg.numTrials)
                % ------------------------------------------------------------
                numRewardedTrials    = round(cfg.numTrials * 0.5); 
                numNonRewardedTrials = cfg.numTrials - numRewardedTrials;
                maxRun = 7;
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
                maxAttempts = 2e6;
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
                % All Basic‐session trials use (dispersion=5°, contrast=1)
                trialDispersions(:) = 5;
                trialContrasts(:)   = cfg.maxContrast5;
                trialBlockID(:)     = {'basic'};

            case 'Full'
                % ------------------------------------------------------------
                % FULL session - NEW ALTERNATING BLOCK LOGIC
                % Block 1: N_easy trials, 50/50 Go/NoGo
                % Then: Alternate 80/20 and 20/80 blocks (min/max length)
                % ------------------------------------------------------------
                
                N_easy = cfg.block1_N_easy;
                
                % --- Define Go/NoGo orientations ---
                allowedOrientations = cfg.fullGratingOrientations;  % all 9
                allowedNo45 = allowedOrientations(allowedOrientations ~= 45);
                
                if cfg.rewardedOrientation == 1 % Horizontal = Go
                    goEasyOri    = 0;
                    nogoEasyOri  = 90;
                    candGo_ori   = [allowedNo45(allowedNo45 < 45), 45]; % 0,15,30,40, 45
                    candNoGo_ori = allowedNo45(allowedNo45 > 45);     % 50,60,75,90
                else % Vertical = Go
                    goEasyOri    = 90;
                    nogoEasyOri  = 0;
                    candGo_ori   = [allowedNo45(allowedNo45 > 45), 45]; % 50,60,75,90, 45
                    candNoGo_ori = allowedNo45(allowedNo45 < 45);     % 0,15,30,40
                end
                
                % --- Block 1: Easy (N_easy trials, 50/50) ---
                N_easy_go   = round(N_easy * 0.5);
                N_easy_nogo = N_easy - N_easy_go;
                
                b1_ori = [repmat(goEasyOri, N_easy_go, 1); ...
                          repmat(nogoEasyOri, N_easy_nogo, 1)];
                b1_disp= repmat(5, N_easy, 1);
                b1_cont= repmat(cfg.maxContrast5, N_easy, 1);
                
                perm1 = randperm(N_easy);
                b1_ori  = b1_ori(perm1);
                b1_disp = b1_disp(perm1);
                b1_cont = b1_cont(perm1);
                
                % Assign Block 1 to main arrays
                trialOrientations(1:N_easy) = b1_ori;
                trialDispersions(1:N_easy)  = b1_disp;
                trialContrasts(1:N_easy)    = b1_cont;
                trialBlockID(1:N_easy)      = {'easy'};
                
                fprintf('Block 1: %d easy trials (50/50) generated.\n', N_easy);
                
                % --- Helper function for generating full-set trials ---
                kPairs = size(cfg.fullPairs,1);  % 8
                rawW   = (cfg.dispersionScalar).^(0:(kPairs-1));
                wts    = rawW / sum(rawW);
                
                build_full_block = @(N_go, N_nogo) ...
                build_full_block_helper(N_go, N_nogo, candGo_ori, candNoGo_ori, ...
                                         wts, cfg.sortedPairIdxAll, cfg.fullPairs);
                
                % --- Generate Alternating Blocks ---
                currentTrialIdx = N_easy + 1;
                
                % Randomly pick first block type
                if rand < 0.5
                    currentBlockType = '80_20';
                    fprintf('First full block: 80/20\n');
                else
                    currentBlockType = '20_80';
                    fprintf('First full block: 20/80\n');
                end
                
                % Loop until all trials are filled
                while currentTrialIdx <= cfg.numTrials
                    % 1. Determine block length
                    blockLen = randi([cfg.minBlockLen, cfg.maxBlockLen]);
                    remaining = cfg.numTrials - currentTrialIdx + 1;
                    blockLen = min(blockLen, remaining); % Don't overshoot
                    
                    % 2. Determine Go/NoGo counts
                    if strcmp(currentBlockType, '80_20')
                        N_go = round(blockLen * 0.8);
                        blockName = '80_20';
                    else % '20_80'
                        N_go = round(blockLen * 0.2);
                        blockName = '20_80';
                    end
                    N_nogo = blockLen - N_go;
                    
                    % 3. Generate the block
                    [b_ori, b_disp, b_cont] = build_full_block(N_go, N_nogo);
                    fprintf('  Generating block: %s (N=%d trials, %d Go, %d NoGo)\n', ...
                            blockName, blockLen, N_go, N_nogo);
                    
                    % 4. Assign to main arrays
                    endIdx = currentTrialIdx + blockLen - 1;
                    trialOrientations(currentTrialIdx:endIdx) = b_ori;
                    trialDispersions(currentTrialIdx:endIdx)  = b_disp;
                    trialContrasts(currentTrialIdx:endIdx)    = b_cont;
                    trialBlockID(currentTrialIdx:endIdx)      = {blockName};
                    
                    % 5. Update pointer
                    currentTrialIdx = endIdx + 1;
                    
                    % 6. Flip block type for next iteration
                    if strcmp(currentBlockType, '80_20')
                        currentBlockType = '20_80';
                    else
                        currentBlockType = '80_20';
                    end
                end
                
            otherwise
                error('Unknown session type.');
        end
        % Store into cfg
        cfg.trialOrientations     = trialOrientations;
        cfg.trialDispersions      = trialDispersions;
        cfg.trialContrasts        = trialContrasts;
        cfg.trialBlockID          = trialBlockID;
        
        %% Precompute Reward Eligibility Vector (UNCHANGED)
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
        
        %% Other Experimental Parameters (UNCHANGED)
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
        
        %% Initialize Hardware Connections (UNCHANGED)
        vr.hardware.inputMC  = serialport(cfg.inputPort, cfg.inputBaudRate);
        flush(vr.hardware.inputMC);
        vr.hardware.outputMC = serialport(cfg.outputPort, cfg.outputBaudRate);
        flush(vr.hardware.outputMC);
        pause(3);
        disp('Hardware initialized using serialport.');
        
        %% Initialize Runtime Variables (MODIFIED)
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
        vr.trialPhase           = 'greyscreen'; % <-- MODIFIED: Start with grey screen
        vr.gratingStartTime     = [];
        vr.greyStartTime        = [];           % <-- NEW
        vr.trialEndPosition     = cfg.minTrialEndPosition + ...
            (cfg.maxTrialEndPosition - cfg.minTrialEndPosition)*rand;
        vr.falseAlarm           = false;
        vr.currentWorld         = cfg.darkWorld; % Will switch to grey world on first frame
        vr.position             = vr.worlds{vr.currentWorld}.startLocation;
        vr.cfg = cfg;
    end

% --- Helper function for building trial blocks ---
    function [ori, disp, cont] = build_full_block_helper(N_go, N_nogo, ...
            candGo_ori, candNoGo_ori, wts, sortedPairIdxAll, fullPairs)
        
        N_total = N_go + N_nogo;
        
        % Preallocate
        blockOri  = zeros(N_total,1);
        blockDisp = zeros(N_total,1);
        blockCont = zeros(N_total,1);
        
        cumP = cumsum(wts);
        
        % Fill "Go" portion
        for i = 1:N_go
            blockOri(i) = candGo_ori(randi(numel(candGo_ori)));
            r = rand;
            pickIdx = find(r <= cumP, 1, 'first');
            pairIdx = sortedPairIdxAll(pickIdx);
            blockDisp(i) = fullPairs(pairIdx, 2);
            blockCont(i) = fullPairs(pairIdx, 1);
        end
        
        % Fill "No-Go" portion
        for i = 1:N_nogo
            idx = N_go + i;
            blockOri(idx) = candNoGo_ori(randi(numel(candNoGo_ori)));
            r = rand;
            pickIdx = find(r <= cumP, 1, 'first');
            pairIdx = sortedPairIdxAll(pickIdx);
            blockDisp(idx) = fullPairs(pairIdx, 2);
            blockCont(idx) = fullPairs(pairIdx, 1);
        end
        
        % Shuffle this block
        perm = randperm(N_total);
        ori  = blockOri(perm);
        disp = blockDisp(perm);
        cont = blockCont(perm);
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

        % --- MODIFIED: New state machine logic ---
        switch vr.trialPhase
            case 'greyscreen' % <-- NEW PHASE
                if isempty(vr.greyStartTime)
                    vr.greyStartTime = vr.timeElapsed;
                    vr.currentTrialLog.absStartTime = datetime('now');
                    vr.currentWorld = vr.cfg.greyWorldIdx; % Show the grey world
                end
                
                % Check if grey screen duration is over
                if vr.timeElapsed - vr.greyStartTime >= vr.cfg.greyDuration
                    vr.greyStartTime = []; % Reset timer
                    vr.trialPhase = 'grating'; % Move to grating phase
                end

            case 'grating'
                if isempty(vr.gratingStartTime)
                    vr.gratingStartTime = vr.timeElapsed;
                    % --- This logic now runs *after* the grey screen ---
                    currentOri      = vr.cfg.trialOrientations(vr.currentTrial);
                    currentDisp     = vr.cfg.trialDispersions(vr.currentTrial);
                    currentContrast = vr.cfg.trialContrasts(vr.currentTrial);

                    oriIdx = find(vr.cfg.fullGratingOrientations == currentOri, 1);
                    if isempty(oriIdx)
                        error('Current orientation %d not found in full list.', currentOri);
                    end
                    allPairs = vr.cfg.fullPairs;
                    pairIdx = find(allPairs(:,1)==currentContrast & ...
                                   allPairs(:,2)==currentDisp, 1);
                    if isempty(pairIdx)
                        error('Pair (%.2f, %d) not found in fullPairs.', ...
                              currentContrast, currentDisp);
                    end
                    gratingWorld = 3 + (pairIdx-1)*9 + oriIdx;
                    vr.currentWorld = gratingWorld; % Now show the actual stimulus
                end
                
                % Check if grating duration is over
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
                        vr.trialPhase = 'greyscreen'; % <-- MODIFIED: Go to grey, not grating
                        vr.greyStartTime = [];        % <-- NEW: Reset grey timer
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
                    % Log trial data
                    vr.currentTrialLog.stimulus    = vr.cfg.trialOrientations(vr.currentTrial);
                    vr.currentTrialLog.dispersion  = vr.cfg.trialDispersions(vr.currentTrial);
                    vr.currentTrialLog.contrast    = vr.cfg.trialContrasts(vr.currentTrial);
                    vr.currentTrialLog.blockID     = vr.cfg.trialBlockID{vr.currentTrial};
                    vr.currentTrialLog.isDefault   = (vr.currentTrial <= vr.cfg.numDefaultTrials);
                    vr.currentTrialLog.rewardGiven = vr.rewardGivenVector(vr.currentTrial);
                    
                    printSummary = @() fprintf(['Completed %d trials in %.1f min. (Block: %s) Active trial performance: ' ...
                        'Hits: %d, Misses: %d, False Alarms: %d, Correct Rejections: %d\n'], ...
                        vr.numTrialsCompleted, vr.timeElapsed/60, ...
                        vr.cfg.trialBlockID{vr.currentTrial}, ...
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
                    
                    vr.trialPhase = 'greyscreen'; % <-- MODIFIED: Go to grey, not grating
                    vr.gratingStartTime = [];
                    vr.greyStartTime = [];      % <-- NEW: Reset grey timer
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
% TERMINATION code: executes after the ViRMEn engine stops. (UNCHANGED)
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
% Helper functions for computing performance statistics (UNCHANGED)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function hits = computeHits(vr)
        hits = 0;
        for t = (vr.cfg.numDefaultTrials+1):length(vr.trialLog)
            if ~isempty(vr.trialLog{t}) && vr.cfg.rewardEligibility(t)
                if vr.trialLog{t}.rewardGiven
                    hits = hits + 1;
                end
            end
        end
    end
    function misses = computeMisses(vr)
        misses = 0;
        for t = (vr.cfg.numDefaultTrials+1):length(vr.trialLog)
            if ~isempty(vr.trialLog{t}) && vr.cfg.rewardEligibility(t)
                if ~vr.trialLog{t}.rewardGiven
                    misses = misses + 1;
                end
            end
        end
    end
    function fa = computeFalseAlarms(vr)
        fa = 0;
        for t = (vr.cfg.numDefaultTrials+1):length(vr.trialLog)
            if ~isempty(vr.trialLog{t}) && ~vr.cfg.rewardEligibility(t)
                if isfield(vr.trialLog{t}, 'falseAlarm') && vr.trialLog{t}.falseAlarm
                    fa = fa + 1;
                end
            end
        end
    end
    function cr = computeCorrectRejections(vr)
        cr = 0;
        for t = (vr.cfg.numDefaultTrials+1):length(vr.trialLog)
            if ~isempty(vr.trialLog{t}) && ~vr.cfg.rewardEligibility(t)
                if ~isfield(vr.trialLog{t}, 'falseAlarm') || ~vr.trialLog{t}.falseAlarm
                    cr = cr + 1;
                end
            end
        end
    end
end