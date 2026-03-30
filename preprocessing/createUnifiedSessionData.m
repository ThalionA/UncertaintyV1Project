function unifiedData = createUnifiedSessionData(sesname, workingDir, recID, img_freq, corridorBinWidth, gratingTimeBinWidth, varargin)
% CREATEUNIFIEDSESSIONDATA Combines neural, VR behavioral, and pupil data for a given session.
%
%
% Optional Arguments (Name-Value Pairs):
%   'force_pupil_reprocess' - logical (default: false). Set to true to force
%                             re-processing of the pupil video even if a
%                             saved resampled file exists.
%

%% --- Default Parameters, Constants & Input Parsing ---
if nargin < 4 || isempty(img_freq); img_freq = 32.6825; fprintf('Using default imaging frequency: %.4f Hz\n', img_freq); end
if nargin < 5 || isempty(corridorBinWidth); corridorBinWidth = 5; fprintf('Using default corridor bin width: %d\n', corridorBinWidth); end
if nargin < 6 || isempty(gratingTimeBinWidth); gratingTimeBinWidth = 0.05; fprintf('Using default grating time bin width: %.3f s\n', gratingTimeBinWidth); end
vrScaling = 0.2537;

% Parse optional arguments
p = inputParser;
addParameter(p, 'force_pupil_reprocess', false, @islogical);
parse(p, varargin{:});
% force_pupil_reprocess = p.Results.force_pupil_reprocess;
force_pupil_reprocess = false;
if force_pupil_reprocess
    fprintf('** Force Pupil Reprocess flag is TRUE. Video will be processed.**\n');
end

%% --- Load Core Data (Neural, VR, Metadata) ---
fprintf('Loading data for session: %s, RecID: %s\n', sesname, recID);
metadataDir = fullfile(workingDir, sesname, 'Metadata');
vrdataDir = fullfile(workingDir, 'VR_files_2025');
caActivityDir = fullfile(workingDir, sesname, 'CaActivity');
pupilBaseDir = fullfile(workingDir, sesname, 'PupilData'); % Base directory for pupil output

% File Paths
pixelintFile = fullfile(metadataDir, 'pixelint.mat');
imagingTimestampsFile = fullfile(metadataDir, 'ImagingTimeStamps', ['ImagingTimestamps_', recID, '.mat']);
vrFile = fullfile(vrdataDir, ['vr_', sesname, '_light.mat']);
dFFile = fullfile(caActivityDir, [recID, 'deltaF_fissa.mat']);
spikesFile = fullfile(caActivityDir, [recID, 'deltaF_fissa_spkprob.mat']);
pupilVideoFile = fullfile(workingDir, 'VR_files_2025', 'VR_videos', [sesname, recID, '.avi']); % Path to pupil video
pupilSaveFile = fullfile(pupilBaseDir, [sesname, recID, '_pupil_resampled_aligned.mat']); % File to save/load resampled pupil

% Load Data & Basic Checks (same as before)
if ~exist(pixelintFile, 'file'); error('Pixel intensity file not found: %s', pixelintFile); end
load(pixelintFile);
if ~exist(imagingTimestampsFile, 'file'); error('Imaging timestamps file not found: %s', imagingTimestampsFile); end
load(imagingTimestampsFile);
ImagingTimestamps = imaging_timestamps_ms / 1000;
nImagingFrames = length(ImagingTimestamps);
if nImagingFrames < 2; error('Insufficient imaging timestamps loaded.'); end
fprintf('Loaded %d imaging timestamps (%.3f s duration).\n', nImagingFrames, ImagingTimestamps(end)-ImagingTimestamps(1));
if ~exist(vrFile, 'file'); error('VR session file not found: %s', vrFile); end
load(vrFile); vrSession = vr;
if ~exist(dFFile, 'file'); error('dF file not found: %s', dFFile); end
load(dFFile);
if ~exist(spikesFile, 'file'); error('Spikes file not found: %s', spikesFile); end
load(spikesFile);
if size(deltaF,2) ~= nImagingFrames || size(spike_prob,2) ~= nImagingFrames; error('Mismatch between neural data frames and imaging timestamps.'); end
nNeurons = size(deltaF, 1);
fprintf('Loaded neural data for %d neurons.\n', nNeurons);

%% --- Extract Trial Boundaries from Pixel Intensity ---
fprintf('Finding trial boundaries using pixel intensity...\n');
try
    if strcmp(sesname, '20250709_Cb17')
        fprintf('examine pixelint')
    end
    [trialStarts, ~, trialEnds] = vr_startframefinder_refined(detrend(pixelint), vrSession, 0.3);
    trialStarts = max(1, trialStarts);
    trialEnds = min(nImagingFrames, trialEnds);
    valid_trial_mask = trialStarts < trialEnds;
    trialStarts = trialStarts(valid_trial_mask);
    trialEnds = trialEnds(valid_trial_mask);
    numTrials = length(trialStarts);
    if numTrials == 0; error('No valid trials found using pixel intensity signal.'); end
    fprintf('Found %d trials based on pixelint.\n', numTrials);
catch ME_pixelint
    error('Error finding trial boundaries: %s', ME_pixelint.message);
end

%% --- Process Pupil Data: Load Resampled OR Process Video ---
pupil_radius_resampled = []; % Initialize empty
pupil_processing_required = true; % Assume we need to process unless loaded

% --- Attempt to Load Previously Processed Data ---
if ~force_pupil_reprocess && exist(pupilSaveFile, 'file')
    fprintf('Found existing resampled pupil file: %s\n', pupilSaveFile);
    try
        loaded_data = load(pupilSaveFile);
        % Verification Step
        if isfield(loaded_data, 'pupil_radius_resampled') && isfield(loaded_data, 'ImagingTimestamps') && ...
           length(loaded_data.pupil_radius_resampled) == nImagingFrames && ...
           isequal(loaded_data.ImagingTimestamps, ImagingTimestamps)
            fprintf('Successfully loaded resampled pupil data. Length and Timestamps match. Skipping video processing.\n');
            pupil_radius_resampled = loaded_data.pupil_radius_resampled;
            pupil_processing_required = false;
        else
            warning('Loaded pupil file validation failed (length/timestamp mismatch or missing field). Forcing re-processing.');
            clear loaded_data;
        end
    catch ME_load
        warning('Could not load or verify existing pupil file (%s): %s. Forcing re-processing.', pupilSaveFile, ME_load.message);
        clear loaded_data;
    end
end

% --- Process Video IF Required ---
if pupil_processing_required
    fprintf('Processing pupil video: %s\n', pupilVideoFile);
    if ~exist(pupilVideoFile, "file")
        warning('Pupil video file not found: %s. Pupil data will be unavailable.', pupilVideoFile);
    else
        try
            % --- Stage 1: Read Video and Extract Raw Pupil Radius & Brightness ---
            obj = VideoReader(pupilVideoFile);
            pupil_frames_reported = obj.NumFrames;
            pupil_freq_reported = obj.FrameRate;
            fprintf('Pupil video info: %d frames reported, %.4f Hz reported rate.\n', pupil_frames_reported, pupil_freq_reported);
            fprintf('Reading video, detecting raw pupil radius and frame brightness...\n');

            % Preallocate outputs based on reported frame count
            pixelint_vid_full = nan(1, pupil_frames_reported); % Brightness needed
            pupil_radius_raw_full = nan(1, pupil_frames_reported);
            pupil_center_raw_full = nan(pupil_frames_reported, 2); % Keep center detection as part of algorithm
            actual_frames_read = 0;
            scf = 4; % Downscaling factor
            hWaitProc = waitbar(0, 'Processing pupil frames (Radius & Brightness)...');

            obj.CurrentTime = 0; % Ensure start from beginning
            while hasFrame(obj) % Check hasFrame first
                current_frame_index = actual_frames_read + 1;
                if current_frame_index > pupil_frames_reported
                     warning('Reading more frames than initially reported (%d). Stopping read.', pupil_frames_reported);
                     break;
                end
                try
                    frame = readFrame(obj);
                    frameGray = rgb2gray(frame);

                    % 1. Calculate and store brightness
                    pixelint_vid_full(current_frame_index) = mean(frameGray(:));

                    % 2. Perform pupil detection (same logic as original)
                    I = imresize(frameGray, 1/scf);
                    Ifilt = medfilt2(I);
                    Ia = imadjust(Ifilt);
                    BW_out = imbinarize(Ia, 'adaptive', 'ForegroundPolarity', 'bright', 'Sensitivity', 0.7);
                    BW_out = imclearborder(BW_out);
                    BW_out = bwpropfilt(BW_out, 'Perimeter', [20* (1/scf), Inf]);
                    BW_out = bwpropfilt(BW_out, 'Eccentricity', [0, 0.8]);
                    BW_out = imfill(BW_out, 'holes');
                    BW_out = bwpropfilt(BW_out, 'Solidity', [0.85, Inf]);
                    BW_out = bwpropfilt(BW_out, 'ConvexArea', [80* (1/scf)^2, Inf]);
                    BW_out = bwpropfilt(BW_out, 'ConvexArea', 1);

                    if sum(BW_out(:)) > 20 * (1/scf)^2
                        stats = regionprops('table', BW_out, 'Centroid', 'MajorAxisLength', 'MinorAxisLength');
                        if ~isempty(stats)
                            pupil_radius_raw_full(current_frame_index) = sqrt((stats.MajorAxisLength(1)/2) * (stats.MinorAxisLength(1)/2)) * scf;
                            pupil_center_raw_full(current_frame_index, :) = stats.Centroid(1,:) * scf;
                        else
                            pupil_radius_raw_full(current_frame_index) = NaN; pupil_center_raw_full(current_frame_index, :) = [NaN, NaN];
                        end
                    else
                        pupil_radius_raw_full(current_frame_index) = NaN; pupil_center_raw_full(current_frame_index, :) = [NaN, NaN];
                    end

                    actual_frames_read = current_frame_index;

                catch ME_frame
                    warning('Error processing pupil frame %d: %s. Skipping frame.', current_frame_index, ME_frame.message);
                    pixelint_vid_full(current_frame_index) = NaN;
                    pupil_radius_raw_full(current_frame_index) = NaN;
                    pupil_center_raw_full(current_frame_index, :) = [NaN, NaN];
                    if obj.CurrentTime + 1/obj.FrameRate < obj.Duration
                        try obj.CurrentTime = obj.CurrentTime + 1/obj.FrameRate; catch; end
                    else
                        break;
                    end
                end
                if mod(actual_frames_read, 100) == 0
                    waitbar(actual_frames_read / pupil_frames_reported, hWaitProc, sprintf('Processing frame %d of ~%d', actual_frames_read, pupil_frames_reported));
                end
            end
            close(hWaitProc);
            fprintf('Finished processing %d pupil frames.\n', actual_frames_read);

            % Trim arrays to actual frames read
            if actual_frames_read == 0; error('Could not read or process any frames from the pupil video.'); end
            pupil_frames = actual_frames_read;
            pixelint_vid = pixelint_vid_full(1:pupil_frames);
            pupil_radius_raw = pupil_radius_raw_full(1:pupil_frames);

            % --- Stage 2: Find Imaging Start/End Pupil Frames using Brightness ---
            fprintf('Estimating imaging start/end frames in pupil video using brightness...\n');
            median_int = median(pixelint_vid, 'omitnan');
            numBaselineFrames = min(pupil_frames, max(100, round(2 * pupil_freq_reported)));
            valid_baseline_pixels = pixelint_vid(1:numBaselineFrames);
            valid_baseline_pixels = valid_baseline_pixels(~isnan(valid_baseline_pixels));
            if isempty(valid_baseline_pixels); error('No valid brightness values found in the initial pupil video frames.'); end

            baseline_median = median(valid_baseline_pixels);
            baseline_mad = mad(valid_baseline_pixels, 1);
            if baseline_mad == 0; baseline_mad = std(valid_baseline_pixels, 'omitnan')/0.6745; end
            if isnan(baseline_mad) || baseline_mad == 0; baseline_mad = 1; end

            brightness_threshold = max(baseline_median + 3*baseline_mad, baseline_median + 0.5*(median_int - baseline_median));
            fprintf('  Brightness: Baseline Median=%.2f, Overall Median=%.2f, Threshold=%.2f\n', baseline_median, median_int, brightness_threshold);

            img_start_pupil = find(pixelint_vid > brightness_threshold, 1, 'first');
            if isempty(img_start_pupil)
                diff_int = diff(pixelint_vid);
                [~, img_start_pupil] = max(diff_int);
                img_start_pupil = img_start_pupil + 1;
                 if pixelint_vid(img_start_pupil) <= brightness_threshold
                     error('Could not reliably determine imaging start pupil frame using brightness threshold (%.2f) or difference. Check video/LEDs.', brightness_threshold);
                 else
                      warning('Brightness threshold failed, using max difference method for start frame.');
                 end
            end
            fprintf('Estimated imaging start at pupil frame: %d\n', img_start_pupil);

            indices_after_start = img_start_pupil:pupil_frames;
            last_bright_frame_relative = find(pixelint_vid(indices_after_start) > brightness_threshold, 1, 'last');
            if isempty(last_bright_frame_relative)
                img_end_pupil = img_start_pupil;
                warning('Could not find frames above threshold AFTER img_start_pupil. Setting img_end_pupil = img_start_pupil.');
            else
                img_end_pupil = indices_after_start(last_bright_frame_relative);
            end
            fprintf('Estimated imaging end at pupil frame: %d\n', img_end_pupil);

            if img_end_pupil <= img_start_pupil
                 error('Estimated pupil end frame (%d) is not after start frame (%d). Check brightness detection.', img_end_pupil, img_start_pupil);
            end

            % --- Stage 3: Calculate Effective Pupil Frame Rate ---
            true_imaging_duration = ImagingTimestamps(end) - ImagingTimestamps(1);
            num_pupil_frames_during_imaging = img_end_pupil - img_start_pupil + 1;

            if num_pupil_frames_during_imaging <= 1 || true_imaging_duration <= 0
                error('Invalid imaging duration (%.4f s) or pupil frame range (%d frames) detected. Check start/end detection.', true_imaging_duration, num_pupil_frames_during_imaging);
            end
            effective_pupil_freq = num_pupil_frames_during_imaging / true_imaging_duration;
            fprintf('--> True imaging duration: %.4f s\n', true_imaging_duration);
            fprintf('--> Pupil frames during imaging: %d (Indices %d to %d)\n', num_pupil_frames_during_imaging, img_start_pupil, img_end_pupil);
            fprintf('--> Effective pupil frame rate: %.4f Hz (Reported: %.4f Hz)\n', effective_pupil_freq, pupil_freq_reported);

            % --- Stage 4: Extract, Clean, and Resample Pupil Data During Imaging ---
            fprintf('Extracting, cleaning, and resampling pupil data during imaging period...\n');
            pupil_indices_during_imaging = img_start_pupil:img_end_pupil;
            pupil_radius_raw_imaging = pupil_radius_raw(pupil_indices_during_imaging);

            medPupil_imaging = median(pupil_radius_raw_imaging, 'omitnan');
            madPupil_imaging = mad(pupil_radius_raw_imaging);
            if madPupil_imaging == 0; madPupil_imaging = std(pupil_radius_raw_imaging(~isnan(pupil_radius_raw_imaging)), 'omitnan') / 0.6745; end
            if isnan(madPupil_imaging) || madPupil_imaging == 0; madPupil_imaging = 1; end
            artifact_indices_imaging = abs(pupil_radius_raw_imaging - medPupil_imaging) > 5 * madPupil_imaging;

            pupil_radius_cleaned_imaging = pupil_radius_raw_imaging;
            pupil_radius_cleaned_imaging(artifact_indices_imaging) = NaN;
            fprintf('Marked %d points as outliers within imaging period (%.2f%% of period).\n', sum(artifact_indices_imaging), 100*sum(artifact_indices_imaging)/num_pupil_frames_during_imaging);

            pupil_time_imaging_rel = (0:(num_pupil_frames_during_imaging-1))' / effective_pupil_freq;

            valid_clean_samples_imaging = ~isnan(pupil_radius_cleaned_imaging);
            if sum(valid_clean_samples_imaging) < 5
                error('Insufficient valid pupil radius samples (%d) during imaging period for interpolation.', sum(valid_clean_samples_imaging));
            end
            pupil_radius_interp_imaging = interp1(pupil_time_imaging_rel(valid_clean_samples_imaging), ...
                                                  pupil_radius_cleaned_imaging(valid_clean_samples_imaging), ...
                                                  pupil_time_imaging_rel, ...
                                                  'linear');
            pupil_radius_interp_imaging = fillmissing(pupil_radius_interp_imaging, 'nearest', 'EndValues', 'nearest');

            % Target times are the imaging timestamps, relative to the FIRST imaging timestamp
            target_time_imaging_rel = ImagingTimestamps - ImagingTimestamps(1);

            % Perform Resampling
            pupil_radius_resampled_temp = interp1(pupil_time_imaging_rel, ...
                                                  pupil_radius_interp_imaging, ...
                                                  target_time_imaging_rel, ...
                                                  'linear', ...
                                                  'extrap');

            nan_count_resampled = sum(isnan(pupil_radius_resampled_temp));
            if nan_count_resampled > 0
                warning('%d NaNs present in final resampled pupil trace (%.2f%%). Filling with nearest.', ...
                        nan_count_resampled, 100*nan_count_resampled/nImagingFrames);
                pupil_radius_resampled_temp = fillmissing(pupil_radius_resampled_temp, 'nearest', 'EndValues', 'nearest');
            end

            if length(pupil_radius_resampled_temp) ~= nImagingFrames
                 error('Final resampled pupil trace length (%d) does not match nImagingFrames (%d).', length(pupil_radius_resampled_temp), nImagingFrames);
            end

            pupil_radius_resampled = pupil_radius_resampled_temp;
            fprintf('Resampling complete. Final pupil trace length: %d\n', length(pupil_radius_resampled));

            % Save
            fprintf('Saving resampled pupil data to: %s\n', pupilSaveFile);
            if ~exist(pupilBaseDir, 'dir'); mkdir(pupilBaseDir); fprintf('Created directory: %s\n', pupilBaseDir); end
            try
                save(pupilSaveFile, 'pupil_radius_resampled', 'ImagingTimestamps', 'effective_pupil_freq');
            catch ME_save
                warning('Could not save resampled pupil data to %s: %s', pupilSaveFile, ME_save.message);
            end

        catch ME_pupil
            warning('An error occurred during pupil processing: %s\nPupil data will be unavailable for this session.', "%s", ME_pupil.message);
            getReport(ME_pupil)
            pupil_radius_resampled = []; % Ensure it's empty on error
        end
    end
end

% --- Final check if pupil data is available ---
if isempty(pupil_radius_resampled)
    fprintf('NOTE: Pupil data processing failed or was skipped. Pupil fields in output will be empty.\n');
else
    fprintf('Pupil data successfully processed or loaded.\n');
end

% --- Pupil Normalization ---
pupil_radius_normalized = []; % Initialize an empty variable for normalized data

if ~isempty(pupil_radius_resampled) && length(ImagingTimestamps) > 1
    fprintf('Normalizing pupil data using baseline (20-30s post-imaging start)...\n');

    baseline_start_time = ImagingTimestamps(1) + 20;
    baseline_end_time   = ImagingTimestamps(1) + 30;

    baseline_indices = find(ImagingTimestamps >= baseline_start_time & ImagingTimestamps <= baseline_end_time);

    if ~isempty(baseline_indices)
        baseline_pupil_values = pupil_radius_resampled(baseline_indices);
        baseline_mean = mean(baseline_pupil_values, 'omitnan');

        if ~isnan(baseline_mean) && baseline_mean ~= 0
            fprintf('  Calculated baseline mean pupil radius: %.4f\n', baseline_mean);
            pupil_radius_normalized = ((pupil_radius_resampled - baseline_mean) ./ baseline_mean) * 100;
            fprintf('  Applied percent change normalization.\n');
        else
            warning('Baseline mean calculation resulted in NaN or zero. Cannot normalize. Keeping original data.');
            pupil_radius_normalized = pupil_radius_resampled;
        end
    else
        warning('No imaging timestamps found within the 20-30s baseline window. Cannot normalize. Keeping original data.');
        pupil_radius_normalized = pupil_radius_resampled;
    end
else
    fprintf('Pupil data not available, skipping normalization.\n');
    % pupil_radius_normalized remains empty
end


%% --- Process Each Trial ---
% --- Preallocate Unified Data Structure ---
unifiedData = struct();

% --- Fixed Time Bins for Grating Epoch ---
fixedBinEdges = 0:gratingTimeBinWidth:2; % fixed 2-s window
nTimeBins = length(fixedBinEdges) - 1;
gratingTimeBinCenters = fixedBinEdges(1:end-1) + gratingTimeBinWidth/2;

fprintf('Processing %d trials...\n', numTrials);
for itrial = 1:numTrials
    % It uses 'pupil_radius_normalized' which is either loaded or calculated above.

    trialNum = itrial;
    fprintf('  Trial %d/%d: ', trialNum, numTrials);

    % --- Get Trial Indices and Timestamps ---
    idx_start = trialStarts(trialNum); idx_end = trialEnds(trialNum);
    if idx_start >= idx_end; fprintf('Skipping trial %d (start >= end index).\n', trialNum); continue; end
    imagingIdx = idx_start:idx_end;
    trial_imagingT_abs = ImagingTimestamps(imagingIdx);
    trial_imagingT_rel = trial_imagingT_abs - trial_imagingT_abs(1);
    nTrialFrames = length(imagingIdx);

    % --- Extract Trial Data (Neural, Pupil) ---
    trial_dF = deltaF(:, imagingIdx); trial_spikes = spike_prob(:, imagingIdx);
    if ~isempty(pupil_radius_normalized)
        trial_pupil = pupil_radius_normalized(imagingIdx);
    else
        trial_pupil = [];
    end

    % --- VR Trial Log ---
    if trialNum > length(vrSession.trialLog); warning('Trial number %d > vrSession.trialLog length. Skipping.', trialNum); continue; end
    trialLog = vrSession.trialLog{trialNum};
    if isempty(trialLog) || ~isfield(trialLog, 'time') || isempty(trialLog.time); fprintf('Skipping trial %d (invalid VR log).\n', trialNum); continue; end
    vrTrialStartTime = trialLog.time(1);

    % --- Store Trial-Level Info ---
    unifiedData(trialNum).trialNumber = trialNum;
    unifiedData(trialNum).imagingStartFrame = idx_start; unifiedData(trialNum).imagingEndFrame = idx_end;
    unifiedData(trialNum).stimulus = trialLog.stimulus; unifiedData(trialNum).contrast = trialLog.contrast; unifiedData(trialNum).dispersion = trialLog.dispersion;
    unifiedData(trialNum).isDefault = trialLog.isDefault; unifiedData(trialNum).falseAlarm = trialLog.falseAlarm;
    unifiedData(trialNum).rewardGiven = trialLog.rewardGiven;

    % Determine trial outcome (Hit, Miss, FA, CR)
    if isfield(vrSession.cfg, 'rewardEligibility')
        goTrial = vrSession.cfg.rewardEligibility(trialNum);
    elseif isfield(vrSession.cfg, 'rewardedOrientation')
        goTrial = (trialLog.stimulus == vrSession.cfg.rewardedOrientation);
    else
        warning('Cannot determine trial type for trial %d.', trialNum);
        goTrial = NaN;
    end

    hit = false; miss = false; FA = false; CR = false;

    if goTrial == 1
        if trialLog.rewardGiven
            hit = true;
        else
            miss = true;
        end
    elseif goTrial == 0
        if trialLog.falseAlarm
            FA = true;
        else
            CR = true;
        end
    end

    unifiedData(trialNum).hit = hit; unifiedData(trialNum).miss = miss; unifiedData(trialNum).FA = FA; unifiedData(trialNum).CR = CR;

    % Initialize structs
    unifiedData(trialNum).corridor = struct(); unifiedData(trialNum).grating = struct();
    unifiedData(trialNum).pupil = struct('corridor', [], 'grating', []);

    % --- Corridor Epoch Processing ---
    fprintf('Corridor... ');
    corridor_vr_idx = find(strcmp(trialLog.trialPhase, 'corridor'));
    binned_dF_corr = [];
    binned_spikes_corr = [];
    binned_pupil_corr = [];
    binned_vrPos_beh = [];
    binned_vrVel_beh = [];
    binned_vrLick_beh = [];
    corridorBinCenters = [];

    if ~isempty(corridor_vr_idx) && length(corridor_vr_idx) > 1
        vr_corridor_pos_log = trialLog.position(2, corridor_vr_idx);
        vr_corridor_vel_log = -trialLog.velocity(2, corridor_vr_idx) * vrScaling;
        vr_corridor_lick_log = trialLog.lick(corridor_vr_idx);
        vr_corridor_time_log_abs = trialLog.time(corridor_vr_idx);
        vr_corridor_time_log_rel = vr_corridor_time_log_abs - vrTrialStartTime;

        pos_min = 0;
        pos_max = vrSession.cfg.maxTrialEndPosition;
        binEdgesCorr = pos_min:corridorBinWidth:pos_max;
        if binEdgesCorr(end) < pos_max
            binEdgesCorr = [binEdgesCorr, pos_max];
        end
        corridorBinCenters = binEdgesCorr(1:end-1) + diff(binEdgesCorr)/2;
        nCorridorBins = length(corridorBinCenters);

        [~, ~, binIdx_beh] = histcounts(vr_corridor_pos_log, binEdgesCorr);
        binned_vrPos_beh = nan(1, nCorridorBins);
        binned_vrVel_beh = nan(1, nCorridorBins);
        binned_vrLick_beh = zeros(1, nCorridorBins);
        for b = 1:nCorridorBins
            idxs_in_bin = (binIdx_beh == b);
            if any(idxs_in_bin)
                binned_vrPos_beh(b) = mean(vr_corridor_pos_log(idxs_in_bin), 'omitnan');
                binned_vrVel_beh(b) = mean(vr_corridor_vel_log(idxs_in_bin), 'omitnan');
                binned_vrLick_beh(b) = sum(vr_corridor_lick_log(idxs_in_bin));
            end
        end

        corridor_start_time_rel = vr_corridor_time_log_rel(1);
        corridor_end_time_rel = vr_corridor_time_log_rel(end);
        corridor_img_mask = (trial_imagingT_rel >= corridor_start_time_rel) & (trial_imagingT_rel <= corridor_end_time_rel);
        corridor_img_indices_in_trial = find(corridor_img_mask);
        nCorridorFrames = length(corridor_img_indices_in_trial);
        if nCorridorFrames > 0
            corridor_imagingT_rel = trial_imagingT_rel(corridor_img_indices_in_trial);
            [vr_corridor_time_log_rel_unique, ia] = unique(vr_corridor_time_log_rel);
            vr_corridor_pos_log_unique = vr_corridor_pos_log(ia);
            if length(vr_corridor_time_log_rel_unique) > 1
                assigned_pos_neural = interp1(vr_corridor_time_log_rel_unique, vr_corridor_pos_log_unique, corridor_imagingT_rel, 'linear');
            else
                assigned_pos_neural = repmat(vr_corridor_pos_log_unique, size(corridor_imagingT_rel));
            end
            [~, ~, binIdx_neural] = histcounts(assigned_pos_neural, binEdgesCorr);
            binned_dF_corr = nan(nNeurons, nCorridorBins);
            binned_spikes_corr = nan(nNeurons, nCorridorBins);
            if ~isempty(trial_pupil)
                binned_pupil_corr = nan(1, nCorridorBins);
            else
                binned_pupil_corr = [];
            end
            for b = 1:nCorridorBins
                frameIdx_in_bin = corridor_img_indices_in_trial(binIdx_neural == b);
                if ~isempty(frameIdx_in_bin)
                    binned_dF_corr(:, b) = mean(trial_dF(:, frameIdx_in_bin), 2, 'omitnan');
                    binned_spikes_corr(:, b) = mean(trial_spikes(:, frameIdx_in_bin), 2, 'omitnan');
                    if ~isempty(trial_pupil)
                        binned_pupil_corr(b) = mean(trial_pupil(frameIdx_in_bin), 'omitnan');
                    end
                end
            end
        end
    end
    unifiedData(trialNum).corridor.imagingIndices = imagingIdx;
    unifiedData(trialNum).corridor.imagingTimestamps = trial_imagingT_rel;
    unifiedData(trialNum).corridor.neural_dF = binned_dF_corr;
    unifiedData(trialNum).corridor.neural_spikes = binned_spikes_corr;
    unifiedData(trialNum).corridor.vr_position = binned_vrPos_beh;
    unifiedData(trialNum).corridor.vr_velocity = binned_vrVel_beh;
    unifiedData(trialNum).corridor.vr_lick = binned_vrLick_beh;
    unifiedData(trialNum).corridor.binCenters = corridorBinCenters;
    unifiedData(trialNum).pupil.corridor = binned_pupil_corr;

    % --- Grating Epoch Processing ---
    fprintf('Grating... ');
    grating_vr_idx = find(strcmp(trialLog.trialPhase, 'grating'));
    binned_dF_grat = [];
    binned_spikes_grat = [];
    binned_pupil_grat = [];
    binned_vrVel_grat = [];
    binned_vrLick_grat = [];
    if ~isempty(grating_vr_idx)
        grating_start_time_abs = trialLog.time(grating_vr_idx(1));
        grating_end_time_abs = grating_start_time_abs + 2.0;
        grating_start_time_rel = grating_start_time_abs - vrTrialStartTime;
        grating_end_time_rel = 2;
        grating_img_mask = (trial_imagingT_rel >= grating_start_time_rel) & (trial_imagingT_rel < grating_end_time_rel);
        grating_img_indices_in_trial = find(grating_img_mask);
        nGratingFrames = length(grating_img_indices_in_trial);
        if nGratingFrames > 0
            grating_imagingT_rel_epoch = trial_imagingT_rel(grating_img_indices_in_trial);
            [~, ~, binIdx_time] = histcounts(grating_imagingT_rel_epoch, fixedBinEdges);
            binned_dF_grat = nan(nNeurons, nTimeBins);
            binned_spikes_grat = nan(nNeurons, nTimeBins);
            if ~isempty(trial_pupil)
                binned_pupil_grat = nan(1, nTimeBins);
            else
                binned_pupil_grat = [];
            end
            for b = 1:nTimeBins
                frameIdx_in_bin = grating_img_indices_in_trial(binIdx_time == b);
                if ~isempty(frameIdx_in_bin)
                    binned_dF_grat(:, b) = mean(trial_dF(:, frameIdx_in_bin), 2, 'omitnan');
                    binned_spikes_grat(:, b) = mean(trial_spikes(:, frameIdx_in_bin), 2, 'omitnan');
                    if ~isempty(trial_pupil)
                        binned_pupil_grat(b) = mean(trial_pupil(frameIdx_in_bin), 'omitnan');
                    end
                end
            end
        end
        grating_vr_mask = (trialLog.time >= grating_start_time_abs) & (trialLog.time < grating_end_time_abs);
        if any(grating_vr_mask)
            vr_grating_time_abs = trialLog.time(grating_vr_mask);
            vr_grating_time_rel_epoch = vr_grating_time_abs - grating_start_time_abs;
            % --- MODIFIED VELOCITY CALCULATION USING DISPLACEMENT & DT ---
            if isfield(trialLog, 'displacement') && isfield(trialLog, 'time')
                
                % Compute time delta (dt) across the whole trial array to maintain length mapping
                dt_full = zeros(size(trialLog.time));
                if length(trialLog.time) > 1
                    dt_full(2:end) = diff(trialLog.time);
                    dt_full(1) = dt_full(2); % Estimate for first frame
                else
                    dt_full(1) = eps;
                end
                
                % Mask for grating period only
                vr_grating_disp = trialLog.displacement(grating_vr_mask);
                vr_grating_dt = dt_full(grating_vr_mask);
                vr_grating_dt(vr_grating_dt <= 0) = eps; % Catch zero division exceptions
                
                % Re-calculate actual mouse velocity mapping, applying original scaling
                vr_grating_vel_log = -(vr_grating_disp ./ vr_grating_dt) * vrScaling;
                
                [~, ~, binIdx_beh_gr_vel] = histcounts(vr_grating_time_rel_epoch, fixedBinEdges);
                binned_vrVel_grat = nan(1, nTimeBins);
                for b = 1:nTimeBins
                    idxs_in_bin = (binIdx_beh_gr_vel == b);
                    if any(idxs_in_bin)
                        binned_vrVel_grat(b) = mean(vr_grating_vel_log(idxs_in_bin), 'omitnan');
                    end
                end
                
            elseif isfield(trialLog, 'velocity') % Backwards compatibility fallback if displacement field doesn't exist
                vr_grating_vel_log = -trialLog.velocity(2, grating_vr_mask) * vrScaling;
                [~, ~, binIdx_beh_gr_vel] = histcounts(vr_grating_time_rel_epoch, fixedBinEdges);
                binned_vrVel_grat = nan(1, nTimeBins);
                for b = 1:nTimeBins
                    idxs_in_bin = (binIdx_beh_gr_vel == b);
                    if any(idxs_in_bin)
                        binned_vrVel_grat(b) = mean(vr_grating_vel_log(idxs_in_bin), 'omitnan');
                    end
                end
            end
            if isfield(trialLog, 'lick')
                vr_grating_lick_log = trialLog.lick(grating_vr_mask);
                [~, ~, binIdx_beh_gr_lick] = histcounts(vr_grating_time_rel_epoch, fixedBinEdges);
                binned_vrLick_grat = zeros(1, nTimeBins);
                for b = 1:nTimeBins
                    idxs_in_bin = (binIdx_beh_gr_lick == b);
                    if any(idxs_in_bin)
                        binned_vrLick_grat(b) = sum(vr_grating_lick_log(idxs_in_bin));
                    end
                end
            end
        end
    end
    unifiedData(trialNum).grating.imagingIndices = imagingIdx;
    unifiedData(trialNum).grating.imagingTimestamps = trial_imagingT_rel;
    unifiedData(trialNum).grating.neural_dF = binned_dF_grat;
    unifiedData(trialNum).grating.neural_spikes = binned_spikes_grat;
    unifiedData(trialNum).grating.timeBins = gratingTimeBinCenters;
    unifiedData(trialNum).grating.vr_velocity = binned_vrVel_grat;
    unifiedData(trialNum).grating.vr_lick = binned_vrLick_grat;
    unifiedData(trialNum).pupil.grating = binned_pupil_grat;

    % --- Compute Per-Trial Metrics ---
    fprintf('Metrics... ');

    % Grating velocity first/second half (unchanged)
    if ~isempty(binned_vrVel_grat)
        halfBin = floor(nTimeBins/2);
        if nTimeBins >= 2
            first_half_indices = 1:min(halfBin, nTimeBins);
            second_half_indices = (halfBin + 1):nTimeBins;
            if isempty(second_half_indices) && nTimeBins == 1; second_half_indices = 1; end
            unifiedData(trialNum).grating_vel_first  = mean(binned_vrVel_grat(first_half_indices),  'omitmissing');
            unifiedData(trialNum).grating_vel_second = mean(binned_vrVel_grat(second_half_indices), 'omitmissing');
        else
            unifiedData(trialNum).grating_vel_first  = mean(binned_vrVel_grat, 'omitmissing');
            unifiedData(trialNum).grating_vel_second = NaN;
        end
    else
        unifiedData(trialNum).grating_vel_first  = NaN;
        unifiedData(trialNum).grating_vel_second = NaN;
    end

    unifiedData(trialNum).total_trial_time = trial_imagingT_rel(end);

    % Corridor-derived metrics (now including LAST 10 cm pre-RZ window)
    if ~isempty(corridor_vr_idx) && length(corridor_vr_idx) > 1
        corridor_times_rel = vr_corridor_time_log_rel;
        corridor_pos       = vr_corridor_pos_log;
        corridor_licks     = vr_corridor_lick_log;
        rz_start           = vrSession.cfg.rewardZoneStart;
        rz_end             = vrSession.cfg.rewardZoneEnd;

        % --- Reward Zone dwell time (existing logic) ---
        inRZ_mask = (corridor_pos >= rz_start) & (corridor_pos <= rz_end);
        if any(inRZ_mask)
            first_entry_idx       = find(inRZ_mask, 1, 'first');
            last_entry_idx        = find(inRZ_mask, 1, 'last');
            unifiedData(trialNum).RZ_dwell_time = corridor_times_rel(last_entry_idx) - corridor_times_rel(first_entry_idx);
        else
            unifiedData(trialNum).RZ_dwell_time = 0;
        end

        
        %  LAST 10 cm PRE-RZ
        preWindow_cm = 10; % window size
        window_start = max(0, rz_start - preWindow_cm);
        preRZ10_mask = (corridor_pos >= window_start) & (corridor_pos < rz_start);

        % Duration spent in last 10cm before RZ (sum of diffs over samples in window)
        preRZ_duration_s = NaN;
        if any(preRZ10_mask)
            t_pre = corridor_times_rel(preRZ10_mask);
            preRZ_duration_s = 0;
            if numel(t_pre) >= 2
                preRZ_duration_s = sum(diff(t_pre));
            end
        end

        % Licks in last 10cm before RZ 
        preRZ_licks_last10cm = NaN;
        if any(preRZ10_mask)
            preRZ_licks_last10cm = sum(corridor_licks(preRZ10_mask));
        end

        % Lick rate (licks per second) in that last 10cm window
        preRZ_lick_rate_last10cm = NaN;
        if ~isnan(preRZ_licks_last10cm) && preRZ_duration_s > 0
            preRZ_lick_rate_last10cm = preRZ_licks_last10cm / preRZ_duration_s;
        end

        % Mean velocity within last 10cm before RZ
        preRZ_velocity_last10cm = NaN;
        if any(preRZ10_mask)
            preRZ_velocity_last10cm = mean(vr_corridor_vel_log(preRZ10_mask), 'omitnan');
        end

        % Store
        unifiedData(trialNum).preRZ_window_cm     = preWindow_cm;
        unifiedData(trialNum).preRZ_duration_s    = preRZ_duration_s;
        unifiedData(trialNum).preRZ_licks         = preRZ_licks_last10cm;       % <-- last 10 cm sum, as requested
        unifiedData(trialNum).preRZ_lick_rate     = preRZ_lick_rate_last10cm;   % licks/s over last 10 cm
        unifiedData(trialNum).preRZ_velocity      = preRZ_velocity_last10cm;    % mean vel over last 10 cm

        % Reward-zone licks
        unifiedData(trialNum).RZ_licks = sum(corridor_licks(inRZ_mask));

        % First-lick timing metrics
        firstLick_idx_corr = find(corridor_licks > 0, 1, 'first');
        if ~isempty(firstLick_idx_corr)
            unifiedData(trialNum).time_to_first_lick_corridor = corridor_times_rel(firstLick_idx_corr) - corridor_times_rel(1);
            unifiedData(trialNum).distance_to_first_lick_corridor = corridor_pos(firstLick_idx_corr) - corridor_pos(1);
        else
            unifiedData(trialNum).time_to_first_lick_corridor = NaN;
            unifiedData(trialNum).distance_to_first_lick_corridor = NaN;
        end

        if any(inRZ_mask)
            first_entry_time_rz = corridor_times_rel(find(inRZ_mask, 1, 'first'));
            first_entry_pos_rz  = corridor_pos(find(inRZ_mask, 1, 'first'));
            licks_in_rz_mask    = inRZ_mask & (corridor_licks > 0);
            firstLick_idx_rz    = find(licks_in_rz_mask, 1, 'first');
            if ~isempty(firstLick_idx_rz)
                unifiedData(trialNum).time_to_first_lick_RZ = corridor_times_rel(firstLick_idx_rz) - first_entry_time_rz;
                unifiedData(trialNum).distance_to_first_lick_RZ = corridor_pos(firstLick_idx_rz) - first_entry_pos_rz;
            else
                unifiedData(trialNum).time_to_first_lick_RZ = NaN;
                unifiedData(trialNum).distance_to_first_lick_RZ = NaN;
            end
        else
            unifiedData(trialNum).time_to_first_lick_RZ = NaN;
            unifiedData(trialNum).distance_to_first_lick_RZ = NaN;
        end

    else
        % No valid corridor indices for this trial
        unifiedData(trialNum).RZ_dwell_time              = NaN;
        unifiedData(trialNum).preRZ_window_cm            = 10;
        unifiedData(trialNum).preRZ_duration_s           = NaN;
        unifiedData(trialNum).preRZ_licks                = NaN;
        unifiedData(trialNum).preRZ_lick_rate            = NaN;
        unifiedData(trialNum).preRZ_velocity             = NaN;
        unifiedData(trialNum).RZ_licks                   = NaN;
        unifiedData(trialNum).time_to_first_lick_corridor = NaN;
        unifiedData(trialNum).distance_to_first_lick_corridor = NaN;
        unifiedData(trialNum).time_to_first_lick_RZ      = NaN;
        unifiedData(trialNum).distance_to_first_lick_RZ  = NaN;
    end

    unifiedData(trialNum).vrTrial = trialLog;
    fprintf('Done.\n');

end % End of trial loop

fprintf('Finished processing all trials. Unified data structure created.\n');

end % End of function createUnifiedSessionData