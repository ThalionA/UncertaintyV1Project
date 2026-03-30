%% Diagnostic: Spatial Window Duration across Animals
clear; close all; clc;

% --- 1. Configuration ---
sesnames1 = {'20250605_Cb15', '20250613_Cb15', '20250620_Cb15', '20250624_Cb15', '20250709_Cb15'};
sesnames2 = {'20250606_Cb17', '20250613_Cb17', '20250620_Cb17', '20250624_Cb17', '20250701_Cb17'};
sesnames3 = {'20250904_Cb21', '20250910_Cb21', '20250911_Cb21', '20250912_Cb21', '20250918_Cb21'};
sesnames4 = {'20251024_Cb22', '20251027_Cb22', '20251028_Cb22', '20251030_Cb22', '20251105_Cb22'};
sesnames5 = {'20250918_Cb24', '20250919_Cb24', '20251020_Cb24', '20251021_Cb24'};
sesnames6 = {'20250903_Cb25', '20250904_Cb25', '20250910_Cb25', '20250911_Cb25', '20250916_Cb25'};

all_sesnames = {sesnames1, sesnames2, sesnames3, sesnames4, sesnames5, sesnames6}; 
animal_tags = {'Cb15', 'Cb17', 'Cb21', 'Cb22', 'Cb24', 'Cb25'};
n_animals = numel(all_sesnames);

W = 10; % Window width
corridor_start = 0; 

% Preallocate arrays for flat data table
all_durations = [];
all_centers = [];
all_animals = [];

fprintf('Extracting spatial bin durations...\n');

%% --- 2. Data Extraction ---
for i_animal = 1:n_animals
    sessions = all_sesnames{i_animal};
    fprintf('  Processing %s...\n', animal_tags{i_animal});
    
    for i_ses = 1:numel(sessions)
        try
            loaded_data = load(['vr_' sessions{i_ses} '_light.mat'], 'vr');
            vr = loaded_data.vr;
            
            % Skip basic sessions if any
            if isfield(vr.cfg, 'sessionType') && strcmpi(vr.cfg.sessionType, 'basic'), continue; end
            
            n_trials = numel(vr.trialLog);
            rz_start = vr.cfg.rewardZoneStart;
            
            % Define bin edges and centers
            bin_edges = corridor_start : W : rz_start;
            bin_centers = bin_edges(1:end-1) + (W/2);
            
            for t = 1:n_trials
                pos = vr.trialLog{t}.position(2,:);
                time = vr.trialLog{t}.time;
                
                % Ensure time is strictly monotonically increasing
                [time_unique, unq_idx] = unique(time);
                pos_unique = pos(unq_idx);
                
                % Only process if the animal actually traversed the corridor
                if max(pos_unique) >= rz_start && min(pos_unique) <= corridor_start + W
                    
                    cross_times = nan(1, length(bin_edges));
                    for b = 1:length(bin_edges)
                        cross_idx = find(pos_unique >= bin_edges(b), 1, 'first');
                        if ~isempty(cross_idx)
                            cross_times(b) = time_unique(cross_idx);
                        end
                    end
                    
                    % Calculate duration in each bin
                    bin_durations = diff(cross_times);
                    
                    % Store valid durations (exclude wild outliers > 10s)
                    valid_bins = ~isnan(bin_durations) & bin_durations > 0 & bin_durations < 10; 
                    
                    all_durations = [all_durations; bin_durations(valid_bins)']; %#ok<*AGROW>
                    all_centers   = [all_centers; bin_centers(valid_bins)'];
                    all_animals   = [all_animals; repmat(i_animal, sum(valid_bins), 1)];
                end
            end
        catch ME
            warning('Failed on %s: %s', sessions{i_ses}, ME.message);
        end
    end
end

%% --- 3. Plotting ---
fprintf('Plotting distributions...\n');

fig = figure('Name', 'Spatial Bin Durations', 'Color', 'w', 'Position', [100, 100, 1200, 500]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
unique_centers = unique(all_centers);
n_win = length(unique_centers);

% ==========================================
% PANEL A: Pooled Trials
% ==========================================
ax1 = nexttile; hold on;

% Pure Boxchart, no lines, no legend
b1 = boxchart(all_centers, all_durations, 'BoxFaceColor', [0.4 0.4 0.4], ...
    'MarkerStyle', '.', 'MarkerColor', [0.7 0.7 0.7], 'LineWidth', 1.5);
b1.JitterOutliers = 'on';

rz_line_pos_A = max(unique_centers) + W/2;
xline(rz_line_pos_A, 'k--', 'Reward Zone Start', 'LineWidth', 2, 'LabelVerticalAlignment', 'bottom');

xlabel('Spatial Window Center (VR units)', 'FontWeight', 'bold');
ylabel('Dwell Time in Bin (Seconds)', 'FontWeight', 'bold');
title('A. Pooled Trials', 'FontSize', 14, 'FontWeight', 'bold');

xticks(unique_centers);
ylim([0, prctile(all_durations, 98)]); % Cap Y-axis to 98th percentile
grid on; gca.GridAlpha = 0.2;

% ==========================================
% PANEL B: Animal Level (n = 6)
% ==========================================
ax2 = nexttile; hold on;

% Precompute animal means per spatial window: [n_animals x n_windows]
animal_means_mat = nan(n_animals, n_win);
for a = 1:n_animals
    for c_idx = 1:n_win
        c = unique_centers(c_idx);
        mask = (all_animals == a) & (all_centers == c);
        if any(mask)
            animal_means_mat(a, c_idx) = mean(all_durations(mask), 'omitnan');
        end
    end
end

% Call your custom plotting function
my_errorbar_plot(animal_means_mat);

% Your function resets the x-axis to 1:N. We need to relabel it to match VR units.
xticks(1:n_win);
xticklabels(string(unique_centers));

% Scale the reward zone line to the new 1:N coordinate system
xline(n_win + 0.5, 'k--', 'Reward Zone Start', 'LineWidth', 2, 'LabelVerticalAlignment', 'bottom');

xlabel('Spatial Window Center (VR units)', 'FontWeight', 'bold');
ylabel('Mean Dwell Time (Seconds)', 'FontWeight', 'bold');
title('B. Animal Level (n = 6)', 'FontSize', 14, 'FontWeight', 'bold');

ylim([0, max(animal_means_mat(:)) * 1.1]);
grid on; gca.GridAlpha = 0.2;

linkaxes([ax1, ax2], 'y')

fprintf('Done.\n');

