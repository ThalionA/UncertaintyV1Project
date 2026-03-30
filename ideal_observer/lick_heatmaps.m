%% Diagnostic: Trial-by-Trial Lick Heatmaps
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
color_cap_licks = 5; % Cap color scale at 5 licks to make behavior POP

fprintf('Extracting trial-by-trial Lick matrices...\n');

%% --- 2. Data Extraction ---
animal_matrices = cell(n_animals, 1);
session_boundaries = cell(n_animals, 1);

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
                lick = vr.trialLog{t}.lick;
                
                if max(pos) >= corridor_start + W
                    bin_licks = nan(1, n_win);
                    for b = 1:n_win
                        in_win = (pos >= bin_edges(b)) & (pos < bin_edges(b+1));
                        if any(in_win)
                            bin_licks(b) = sum(lick(in_win));
                        else
                            bin_licks(b) = 0;
                        end
                    end
                    ses_mat(t, :) = bin_licks;
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
fprintf('Plotting Trial-by-Bin Lick Heatmaps...\n');

fig = figure('Name', 'Trial-by-Bin Licks', 'Color', 'w', 'Position', [100, 100, 1600, 800]);
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

for a = 1:n_animals
    nexttile; hold on;
    
    mat = animal_matrices{a};
    
    % Plot Heatmap. Use AlphaData so NaNs (incomplete trials) show up as grey background
    im = imagesc(mat);
    im.AlphaData = ~isnan(mat);
    set(gca, 'Color', [0.8 0.8 0.8]); % Grey background for NaNs
    
    % Use a colormap where 0 is white/blue and high licks are dark red
    % colormap(flipud(hot)) is usually great for sparse lick data
    cmap = [1 1 1; parula(256)]; % 0 is strictly white
    colormap(gca, cmap);
    clim([0, Inf]); 
    
    % Draw horizontal lines for session boundaries
    bounds = session_boundaries{a};
    for b = 1:length(bounds)-1
        yline(bounds(b) + 0.5, 'k-', 'LineWidth', 1.5, 'Alpha', 0.5);
    end
    
    % Draw vertical line for the problematic Bin 35
    bin_35_idx = find(bin_centers == 35);
    if ~isempty(bin_35_idx)
        xline(bin_35_idx, 'r--', 'Bin 35', 'LineWidth', 2, 'LabelVerticalAlignment', 'bottom');
    end
    
    % Formatting
    title(animal_tags{a}, 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Cumulative Trial Number', 'FontWeight', 'bold');
    if a > 3
        xlabel('Spatial Bin Index', 'FontWeight', 'bold');
    end
    
    axis tight;
    ax = gca;
    ax.YDir = 'reverse'; 
    ax.TickDir = 'out';
    ax.Box = 'off';
end

% Add a global colorbar
cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'Licks per Bin';
cb.Label.FontWeight = 'bold';
cb.Label.FontSize = 12;

fprintf('Done.\n');