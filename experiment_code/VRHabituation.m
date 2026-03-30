function code = VRHabituation
% VRHabituation   Simplified ViRMEn experiment for corridor habituation.
%   Five corridor worlds differing by reward‐zone location. Animal must traverse
%   the corridor and receive reward in the zone; progression through worlds is
%   determined by a sliding window of the last 5 trials at each corridor length:
%     – If average pace over those 5 is ≥ 5 trials/min → promote one level
%     – If < 2 trials/min → demote one level
%   Includes starting‐level selection, maximum‐trials option, and plots world level per trial.
%   Active trials now dispense reward on the first lick *inside* the reward zone.
%   Tracks active‐trial hits/misses; if 3 misses in a row, next trial is forced default.

code.initialization = @initializationCodeFun;
code.runtime        = @runtimeCodeFun;
code.termination    = @terminationCodeFun;

    function vr = initializationCodeFun(vr)
        %%=== 1) SESSION PARAMETER INPUT ===%%
        prompt = {'Animal ID', 'Session length (minutes)', ...
            'Number of default trials', 'Maximum number of trials', ...
            'Starting corridor level (1–5)', 'Valve open duration (s)'};
        dlgtitle = 'Session Parameters';
        dims = [1 50];
        definput = {'Cb01','15','20','200','1','0.5'};
        answer = inputdlg(prompt, dlgtitle, dims, definput);
        if isempty(answer)
            error('User cancelled session parameter input.');
        end

        cfg.animalID          = answer{1};
        cfg.maxSessionLength  = str2double(answer{2});  % in minutes
        cfg.numDefaultTrials  = str2double(answer{3});
        cfg.maxTrials         = str2double(answer{4});
        startLevel            = str2double(answer{5});
        cfg.valveOpenDuration = str2double(answer{6});  % in seconds

        % Validate starting level
        if isnan(startLevel) || startLevel < 1 || startLevel > 5
            error('Starting corridor level must be an integer between 1 and 5.');
        end

        % Define five corridor worlds with reward‐zone centers [60,70,80,100,120]
        rewardCenters   = [80, 100, 120, 140, 160];
        zoneHalfWidth   = 20;
        cfg.numWorlds   = numel(rewardCenters);
        cfg.zoneStarts  = rewardCenters - zoneHalfWidth;    % e.g. [40,50,60,80,100]
        cfg.zoneEnds    = rewardCenters + zoneHalfWidth;    % e.g. [80,90,100,120,140]
        cfg.resetPoints = cfg.zoneEnds + 0.1;               % “reset” a little past the zone

        %%=== 2) HARDWARE SETUP ===%%
        cfg.inputPort       = 'COM3';
        cfg.outputPort      = 'COM4';
        cfg.inputBaudRate   = 115200;
        cfg.outputBaudRate  = 115200;
        cfg.encoderCountsPerRevolution = 2500;
        cfg.wheelDiameter   = 24.2;
        cfg.wheelCircumference = pi * cfg.wheelDiameter;
        cfg.distancePerCount   = cfg.wheelCircumference / cfg.encoderCountsPerRevolution;
        cfg.movement_gain      = 1;  % adjust if needed

        vr.hardware.inputMC  = serialport(cfg.inputPort, cfg.inputBaudRate);
        flush(vr.hardware.inputMC);
        vr.hardware.outputMC = serialport(cfg.outputPort, cfg.outputBaudRate);
        flush(vr.hardware.outputMC);
        pause(3);
        disp('Hardware initialized using serialport.');

        %%=== 3) INITIALIZE RUNTIME VARIABLES ===%%
        vr.sessionTimer        = tic;
        vr.previousTime        = 0;
        vr.iteration           = 0;

        % Start in the user‐selected corridor level:
        vr.currentWorldIndex   = startLevel;
        vr.position            = vr.worlds{vr.currentWorldIndex}.startLocation;

        vr.timeElapsed         = toc(vr.sessionTimer);
        vr.totalTrials         = 0;       % how many trials have fully completed so far
        vr.rewardGiven         = false;   % have we already dispensed reward this trial?
        vr.valveIsOpen         = false;
        vr.valveOpenTime       = 0;

        % We keep track of “which world each completed trial was in” + when it ended:
        vr.trialWorldHistory   = nan(1, cfg.maxTrials);
        vr.trialEndTimes       = nan(1, cfg.maxTrials);

        % Preallocate “isDefault” vector: first numDefaultTrials are true, rest false
        vr.isDefault = false(1, cfg.maxTrials);
        vr.isDefault(1:cfg.numDefaultTrials) = true;

        % Active‐trial performance tracking:
        vr.activeHits              = 0;
        vr.activeMisses            = 0;
        vr.consecutiveActiveMisses = 0;

        vr.cfg = cfg;
    end

    function vr = runtimeCodeFun(vr)
        vr.iteration    = vr.iteration + 1;
        vr.timeElapsed  = toc(vr.sessionTimer);
        vr.dt           = vr.timeElapsed - vr.previousTime;
        vr.previousTime = vr.timeElapsed;

        %% 1) CLOSE VALVE IF NECESSARY
        if vr.valveIsOpen && (vr.timeElapsed - vr.valveOpenTime >= vr.cfg.valveOpenDuration)
            write(vr.hardware.outputMC, 't', "char");
            vr.valveIsOpen = false;
        end

        %% 2) READ HAPTICS/LICKS + UPDATE POSITION
        frameLick    = 0;
        displacement = 0;
        if vr.hardware.inputMC.NumBytesAvailable > 0
            A = read(vr.hardware.inputMC, vr.hardware.inputMC.NumBytesAvailable, "uint8");
            lickIdx       = (A == 49) | (A == 50);
            frameLick     = double(any(lickIdx));
            encoderValues = A(~lickIdx);
            diffCount     = sum(encoderValues == 1) - sum(encoderValues == 0);
            displacement  = diffCount * vr.cfg.distancePerCount * vr.cfg.movement_gain;
        end

        vr.dp = [0, -displacement, 0, 0];
        if vr.dt > 0
            vr.velocity = [0, -displacement/vr.dt, 0, 0];
        else
            vr.velocity = [0, 0, 0, 0];
        end

        %% 3) DETERMINE ZONE BOUNDS FOR CURRENT WORLD
        idx        = vr.currentWorldIndex;
        zoneStart  = vr.cfg.zoneStarts(idx);
        zoneEnd    = vr.cfg.zoneEnds(idx);
        resetPoint = vr.cfg.resetPoints(idx);

        yPos   = vr.position(2);
        inZone = (yPos >= zoneStart) && (yPos <= zoneEnd);

        %% 4) FIGURE OUT CURRENT TRIAL INDEX & “DEFAULT” STATUS
        % Next trial index is totalTrials+1 (unless totalTrials==maxTrials, but then we end soon)
        currentTrialIdx = vr.totalTrials + 1;
        if currentTrialIdx > vr.cfg.maxTrials
            % No more trials; just end
            vr.experimentEnded = true;
            return;
        end
        isDefaultTrial = vr.isDefault(currentTrialIdx);

        %% 5) REWARD LOGIC
        if ~vr.rewardGiven
            if isDefaultTrial
                % Default trial: reward immediately upon entering zone
                if inZone
                    write(vr.hardware.outputMC, 'r', "char");
                    vr.valveIsOpen   = true;
                    vr.valveOpenTime = vr.timeElapsed;
                    vr.rewardGiven   = true;
                end
            else
                % Active trial: a single lick inside the zone → immediate reward
                if inZone && frameLick
                    write(vr.hardware.outputMC, 'r', "char");
                    vr.valveIsOpen   = true;
                    vr.valveOpenTime = vr.timeElapsed;
                    vr.rewardGiven   = true;
                end
            end
        end

        %% 6) CHECK FOR TRIAL COMPLETION (CROSSING RESET POINT)
        if yPos >= resetPoint
            fprintf('Completed trial %d in %.2f minutes\n', currentTrialIdx, vr.timeElapsed/60);
            oldIndex = vr.currentWorldIndex;
            vr.totalTrials = vr.totalTrials + 1;

            % Log which world this trial was in, and the exact end‐time:
            if currentTrialIdx <= vr.cfg.maxTrials
                vr.trialWorldHistory(currentTrialIdx) = oldIndex;
                vr.trialEndTimes(currentTrialIdx)     = vr.timeElapsed;
            end

            % === Active‐trial performance update ===
            if ~isDefaultTrial
                if vr.rewardGiven
                    % Hit
                    vr.activeHits = vr.activeHits + 1;
                    vr.consecutiveActiveMisses = 0;
                else
                    % Miss
                    vr.activeMisses = vr.activeMisses + 1;
                    vr.consecutiveActiveMisses = vr.consecutiveActiveMisses + 1;
                    if vr.consecutiveActiveMisses >= 3
                        nextIdx = currentTrialIdx + 1;
                        if nextIdx <= vr.cfg.maxTrials
                            vr.isDefault(nextIdx) = true;
                            fprintf('Three active misses in a row → trial %d forced default.\n', nextIdx);
                        end
                        vr.consecutiveActiveMisses = 0;  % reset counter after enforcing default
                    end
                end

                % Print current active performance
                totalActive = vr.activeHits + vr.activeMisses;
                pctHit = 100 * vr.activeHits / max(1, totalActive);
                fprintf('Active trials so far: %d hits, %d misses (%.1f%%)\n', ...
                    vr.activeHits, vr.activeMisses, pctHit);
            end

            % === Sliding‐window promotion/demotion logic ===
            idxsAtWorld = find(vr.trialWorldHistory(1:vr.totalTrials) == oldIndex);
            if numel(idxsAtWorld) >= 5
                lastFiveIdxs = idxsAtWorld(end-4:end);
                timesFive    = vr.trialEndTimes(lastFiveIdxs);
                timeSpan     = timesFive(end) - timesFive(1);  % seconds for those 5
                rate         = 5 / (timeSpan / 60);            % convert to trials/min

                if rate >= 5 && vr.currentWorldIndex < vr.cfg.numWorlds
                    vr.currentWorldIndex = vr.currentWorldIndex + 1;
                elseif rate < 2 && vr.currentWorldIndex > 1
                    vr.currentWorldIndex = vr.currentWorldIndex - 1;
                end
            end

            % Move to (possibly updated) world
            vr.currentWorld = vr.currentWorldIndex;
            vr.position    = vr.worlds{vr.currentWorldIndex}.startLocation;
            vr.rewardGiven = false;   % reset reward flag for next trial

            % Stop if we hit maxTrials
            if vr.totalTrials >= vr.cfg.maxTrials
                disp('Maximum number of trials reached. Ending experiment.');
                vr.experimentEnded = true;
            end

            return;  % done with this frame
        end

        %% 7) END SESSION IF TIME EXCEEDED
        if vr.timeElapsed >= vr.cfg.maxSessionLength * 60
            disp('Session time exceeded. Ending experiment.');
            vr.experimentEnded = true;
        end
    end

    function vr = terminationCodeFun(vr)
        % Close valve (just in case)
        write(vr.hardware.outputMC, 't', "char");

        % Clean up hardware
        if isvalid(vr.hardware.inputMC)
            delete(vr.hardware.inputMC);
        end
        if isvalid(vr.hardware.outputMC)
            delete(vr.hardware.outputMC);
        end
        disp('Hardware connections terminated.');

        % Save data
        dateStr      = char(datetime("now", "Format", 'yyyyMMdd'));
        baseFilename = sprintf('vr_%s_%s', dateStr, vr.cfg.animalID);
        fname        = [baseFilename, '.mat'];
        n = 1;
        while exist(fname, 'file')
            fname = sprintf('%s_%d.mat', baseFilename, n);
            n = n + 1;
        end
        save(fname, 'vr', '-v7.3');
        disp(['Experiment completed and data saved as ' fname '.']);

        % Print summary
        fprintf('Total trials completed: %d\n', vr.totalTrials);
        fprintf('Final corridor world index: %d\n', vr.currentWorldIndex);
        fprintf('Overall active performance: %d hits, %d misses (%.1f%%)\n', ...
            vr.activeHits, vr.activeMisses, ...
            100 * vr.activeHits / max(1, vr.activeHits + vr.activeMisses));

        % Plot world level per trial, only if ≥1 trial completed
        if vr.totalTrials > 0
            nTrialsToPlot = min(vr.totalTrials, vr.cfg.maxTrials);
            worldLevels   = vr.trialWorldHistory(1:nTrialsToPlot);
            figure;
            plot(1:nTrialsToPlot, worldLevels, '-o', 'LineWidth', 1.5);
            xlabel('Trial Number');
            ylabel('Corridor Length Level (1 = shortest, 5 = longest)');
            title('World Level per Trial');
            ylim([0.8, vr.cfg.numWorlds + 0.2]);
            yticks(1:vr.cfg.numWorlds);
            grid on;
        else
            disp('No trials completed; skipping plot.');
        end
    end
end
