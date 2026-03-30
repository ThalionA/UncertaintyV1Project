%% Diagnostic: Trial-by-Trial Spatial Dwell Time Heatmaps
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
color_cap_seconds = 2.0; % Cap heatmap colors at 2 seconds to make outliers POP

fprintf('Extracting trial-by-trial matrices...\n');

%% --- 2. Data Extraction ---
animal_matrices = cell(n_animals, 1);
session_boundaries = cell(n_animals, 1);
outlier_records = struct('animal', {}, 'session', {}, 'trial', {}, 'bin_center', {}, 'duration', {});
outlier_idx = 1;

for i_animal = 1:n_animals
    sessions = all_sesnames{i_animal};
    animal_mat = [];
    ses_bounds = [];
    cumulative_trials = 0;
    
    for i_ses = 1:numel(sessions)
        try
            loaded_data = load(['vr_' sessions{i_ses} '_light.mat'], 'vr');
            vr = loaded_data.vr;
            
            if isfield(vr.cfg, 'sessionType') && strcmpi(vr.cfg.sessionType, 'basic'), continue; end
            
            n_trials = numel(vr.trialLog);
            rz_start = vr.cfg.rewardZoneStart;
            
            bin_edges = corridor_start : W : rz_start;
            bin_centers = bin_edges(1:end-1) + (W/2);
            n_win = length(bin_centers);
            
            ses_mat = nan(n_trials, n_win);
            
            for t = 1:n_trials
                pos = vr.trialLog{t}.position(2,:);
                time = vr.trialLog{t}.time;
                
                [time_unique, unq_idx] = unique(time);
                pos_unique = pos(unq_idx);
                
                if max(pos_unique) >= corridor_start + W
                    cross_times = nan(1, length(bin_edges));
                    for b = 1:length(bin_edges)
                        cross_idx = find(pos_unique >= bin_edges(b), 1, 'first');
                        if ~isempty(cross_idx), cross_times(b) = time_unique(cross_idx); end
                    end
                    
                    bin_durations = diff(cross_times);
                    
                    % Flag extreme outliers (> 5 seconds in a 10-unit bin)
                    ext_idx = find(bin_durations > 5);
                    for ex = 1:length(ext_idx)
                        outlier_records(outlier_idx).animal = animal_tags{i_animal};
                        outlier_records(outlier_idx).session = sessions{i_ses};
                        outlier_records(outlier_idx).trial = t;
                        outlier_records(outlier_idx).bin_center = bin_centers(ext_idx(ex));
                        outlier_records(outlier_idx).duration = bin_durations(ext_idx(ex));
                        outlier_idx = outlier_idx + 1;
                    end
                    
                    ses_mat(t, :) = bin_durations;
                end
            end
            
            animal_mat = [animal_mat; ses_mat]; %#ok<*AGROW>
            cumulative_trials = cumulative_trials + n_trials;
            ses_bounds = [ses_bounds; cumulative_trials];
            
        catch ME
            warning('Failed on %s: %s', sessions{i_ses}, ME.message);
        end
    end
    animal_matrices{i_animal} = animal_mat;
    session_boundaries{i_animal} = ses_bounds;
end

%% --- 3. Plotting Heatmaps ---
fprintf('\nPlotting Trial-by-Bin Heatmaps...\n');

fig = figure('Name', 'Trial-by-Bin Dwell Times', 'Color', 'w', 'Position', [100, 100, 1600, 800]);
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

for a = 1:n_animals
    nexttile; hold on;
    
    mat = animal_matrices{a};
    
    % Plot Heatmap. Use AlphaData so NaNs (incomplete trials) show up as grey background
    im = imagesc(mat);
    im.AlphaData = ~isnan(mat);
    set(gca, 'Color', [0.8 0.8 0.8]); % Grey background for NaNs
    
    colormap(hot);
    clim([0, Inf]); % Cap the color scale so outliers saturate it as bright white/yellow
    
    % Draw horizontal lines for session boundaries
    bounds = session_boundaries{a};
    for b = 1:length(bounds)-1
        yline(bounds(b) + 0.5, 'w-', 'LineWidth', 1.5, 'Alpha', 0.5);
    end
    
    % Formatting
    title(animal_tags{a}, 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Cumulative Trial Number', 'FontWeight', 'bold');
    if a > 3
        xlabel('Spatial Bin Index', 'FontWeight', 'bold');
    end
    
    axis tight;
    ax = gca;
    ax.YDir = 'reverse'; % Standard for imagesc
    ax.TickDir = 'out';
    ax.Box = 'off';
end

% Add a single global colorbar
cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'Dwell Time (Seconds)';
cb.Label.FontWeight = 'bold';
cb.Label.FontSize = 12;

fprintf('Done.\n');