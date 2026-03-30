%% Plot Combined Spatiotemporal Decoding (Choice & Stimulus)
% clear; close all; clc;

% --- 1. Load Data ---
fprintf('Loading saved results...\n');
ChoiceData = load('Spatiotemporal_Model_Comparison_Unified.mat');
StimData   = load('Spatiotemporal_Decoding_Stimulus.mat');

% Extract basic dimensions directly from the saved structs
plot_centers = [ChoiceData.WindowResults.center]; 
n_windows = length(plot_centers);
n_animals = length(ChoiceData.WindowResults(1).animal_data);

% --- 2. Extract Choice Metrics ---
metrics_choice = {'IO_Full_AUC', 'IO_Vel_AUC', 'GLM_Full_AUC', 'GLM_Sensorimotor_AUC', ...
           'GLM_Kinematic_LV_AUC', 'GLM_Kinematic_V_AUC', 'GLM_Kinematic_L_AUC', ...
           'GLM_Sensory_AUC', 'GLM_History_AUC'};
res_choice = nan(n_animals, n_windows, length(metrics_choice));

for w = 1:n_windows
    for a = 1:n_animals
        if length(ChoiceData.WindowResults(w).animal_data) >= a
            res = ChoiceData.WindowResults(w).animal_data{a};
            if isfield(res, 'status') && strcmp(res.status, 'insufficient_data'), continue; end
            for m = 1:length(metrics_choice)
                if isfield(res, metrics_choice{m}), res_choice(a, w, m) = res.(metrics_choice{m}); end
            end
        end
    end
end
avg_choice = squeeze(mean(res_choice, 1, 'omitnan'));

% --- 3. Extract Stimulus Metrics ---
metrics_stim = {'IO_Full_AUC', 'IO_Vel_AUC', 'GLM_Kinematic_LV_AUC', 'GLM_Kinematic_V_AUC', 'GLM_Kinematic_L_AUC'};
res_stim = nan(n_animals, n_windows, length(metrics_stim));

for w = 1:n_windows
    for a = 1:n_animals
        if length(StimData.WindowResults(w).animal_data) >= a
            res = StimData.WindowResults(w).animal_data{a};
            if isfield(res, 'status') && strcmp(res.status, 'insufficient_data'), continue; end
            for m = 1:length(metrics_stim)
                if isfield(res, metrics_stim{m}), res_stim(a, w, m) = res.(metrics_stim{m}); end
            end
        end
    end
end
avg_stim = squeeze(mean(res_stim, 1, 'omitnan'));

% --- 4. Define Universal Color Palette ---
col_io_full  = [0, 0.4470, 0.7410];       % Deep Blue
col_io_red   = [0.3010, 0.7450, 0.9330];  % Light Blue
col_glm_full = [0.8500, 0.3250, 0.0980];  % Orange/Red
col_glm_sm   = [0.4940, 0.1840, 0.5560];  % Purple
col_kin      = [0.4660, 0.6740, 0.1880];  % Base Green
col_sensory  = [0.2, 0.2, 0.2];           % Dark Grey
col_history  = [0.6, 0.6, 0.6];           % Light Grey

%% --- 5. Generate Figure ---
fig = figure('Name', 'Joint Spatiotemporal Decoding', 'Color', 'w', 'Position', [100, 100, 1400, 600]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% ==========================================
% PANEL A: Choice Decoding
% ==========================================
nexttile; hold on;

% Baselines (Greys)
plot(plot_centers, avg_choice(:,9), ':', 'Color', col_history, 'LineWidth', 2.5, 'DisplayName', 'GLM History');
plot(plot_centers, avg_choice(:,8), '--', 'Color', col_sensory, 'LineWidth', 2.5, 'DisplayName', 'GLM Sensory');
% Kinematic GLMs (Greens)
plot(plot_centers, avg_choice(:,7), ':', 'Color', col_kin, 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', 'GLM Kinematic (Licks)');
plot(plot_centers, avg_choice(:,6), '--', 'Color', col_kin, 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', 'GLM Kinematic (Vel)');
plot(plot_centers, avg_choice(:,5), '-', 'Color', col_kin, 'LineWidth', 2.5, 'Marker', 's', 'MarkerSize', 6, 'MarkerFaceColor', col_kin, 'DisplayName', 'GLM Kinematic (Vel+Licks)');
% Complex GLMs (Purples/Reds)
plot(plot_centers, avg_choice(:,4), '-', 'Color', col_glm_sm, 'LineWidth', 2.5, 'Marker', '^', 'MarkerSize', 6, 'MarkerFaceColor', col_glm_sm, 'DisplayName', 'GLM Sensorimotor');
plot(plot_centers, avg_choice(:,3), '-', 'Color', col_glm_full, 'LineWidth', 3, 'Marker', 'd', 'MarkerSize', 7, 'MarkerFaceColor', col_glm_full, 'DisplayName', 'GLM Full Integrated');
% IO Models (Blues)
plot(plot_centers, avg_choice(:,2), '-', 'Color', col_io_red, 'LineWidth', 2.5, 'Marker', 'o', 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'DisplayName', 'IO Reduced (Vel)');
plot(plot_centers, avg_choice(:,1), '-', 'Color', col_io_full, 'LineWidth', 3, 'Marker', 'o', 'MarkerSize', 7, 'MarkerFaceColor', col_io_full, 'DisplayName', 'IO Full (Licks+Vel)');

% Aesthetics Panel A
xlabel('Spatial Window Center (VR units)', 'FontWeight', 'bold');
ylabel('Cross-Validated AUC', 'FontWeight', 'bold');
title('A. Decoding Trial Choice', 'FontSize', 14, 'FontWeight', 'bold');
% set(gca, 'FontSize', 12, 'TickDir', 'out', 'Box', 'off', 'LineWidth', 1.5);
xlim([min(plot_centers)-5, max(plot_centers)+5]); ylim([0.45, 1.0]); 
yline(0.5, 'k-', 'Chance', 'LineWidth', 1.5, 'HandleVisibility', 'off');
grid on; gca.GridAlpha = 0.15;
lgd1 = legend('Location', 'eastoutside', 'NumColumns', 1, 'FontSize', 9);

% ==========================================
% PANEL B: Stimulus Decoding
% ==========================================
nexttile; hold on;

% Kinematic GLMs (Greens)
plot(plot_centers, avg_stim(:,5), ':', 'Color', col_kin, 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', 'GLM Kinematic (Licks)');
plot(plot_centers, avg_stim(:,4), '--', 'Color', col_kin, 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', 'GLM Kinematic (Vel)');
plot(plot_centers, avg_stim(:,3), '-', 'Color', col_kin, 'LineWidth', 2.5, 'Marker', 's', 'MarkerSize', 6, 'MarkerFaceColor', col_kin, 'DisplayName', 'GLM Kinematic (Vel+Licks)');
% IO Models (Blues)
plot(plot_centers, avg_stim(:,2), '-', 'Color', col_io_red, 'LineWidth', 2.5, 'Marker', 'o', 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'DisplayName', 'IO Reduced (Vel)');
plot(plot_centers, avg_stim(:,1), '-', 'Color', col_io_full, 'LineWidth', 3, 'Marker', 'o', 'MarkerSize', 7, 'MarkerFaceColor', col_io_full, 'DisplayName', 'IO Full (Licks+Vel)');

% Aesthetics Panel B
xlabel('Spatial Window Center (VR units)', 'FontWeight', 'bold');
title('B. Decoding True Stimulus Category', 'FontSize', 14, 'FontWeight', 'bold');
% set(gca, 'FontSize', 12, 'TickDir', 'out', 'Box', 'off', 'LineWidth', 1.5);
xlim([min(plot_centers)-5, max(plot_centers)+5]); ylim([0.45, 1.0]); 
yline(0.5, 'k-', 'Chance', 'LineWidth', 1.5, 'HandleVisibility', 'off');
grid on; gca.GridAlpha = 0.15;
lgd2 = legend('Location', 'eastoutside', 'NumColumns', 1, 'FontSize', 10);

fprintf('Joint spatiotemporal decoding figure generated successfully.\n');

%% Scatter Plot: Choice vs. Stimulus Decoding (Full IO)


% Preallocate arrays
choice_io_full = nan(n_animals, n_windows);
stim_io_full   = nan(n_animals, n_windows);

% Extract IO_Full_AUC for both tasks
for w = 1:n_windows
    for a = 1:n_animals
        % Extract Choice
        if length(ChoiceData.WindowResults(w).animal_data) >= a
            res_c = ChoiceData.WindowResults(w).animal_data{a};
            if isfield(res_c, 'IO_Full_AUC')
                choice_io_full(a, w) = res_c.IO_Full_AUC; 
            end
        end
        
        % Extract Stimulus
        if length(StimData.WindowResults(w).animal_data) >= a
            res_s = StimData.WindowResults(w).animal_data{a};
            if isfield(res_s, 'IO_Full_AUC')
                stim_io_full(a, w) = res_s.IO_Full_AUC; 
            end
        end
    end
end

figure('Name', 'IO Full: Choice vs Stimulus', 'Color', 'w', 'Position', [200, 200, 700, 600]);
hold on;

% Unity Line (y = x)
% Points above this line mean the model decodes the objective stimulus better than the animal's choice.
% Points below mean it decodes the animal's choice better than the true stimulus.
plot([0.4 1.05], [0.4 1.05], 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');

% Create color array mapped to spatial window centers
C_array = repmat(plot_centers, n_animals, 1);

% Scatter plot
scatter(choice_io_full(:), stim_io_full(:), 75, C_array(:), 'filled', ...
    'MarkerFaceAlpha', 0.8, 'MarkerEdgeColor', [0.3 0.3 0.3], 'LineWidth', 0.5);

% Formatting
xlabel('Choice Decoding AUC', 'FontWeight', 'bold');
ylabel('Stimulus Decoding AUC', 'FontWeight', 'bold');
title('Full Ideal Observer: Choice vs. Stimulus Predictivity', 'FontSize', 14, 'FontWeight', 'bold');

% Aesthetics
axis square; % Keeps the visual scaling 1:1 so the unity line is perfectly 45 degrees
xlim([0.45 1.0]); 
ylim([0.45 1.0]);
grid on; 
gca.GridAlpha = 0.15;

% Add Colorbar to indicate spatial progress
colormap(parula);
cb = colorbar;
cb.Label.String = 'Spatial Window Center (VR units)';
cb.Label.FontWeight = 'bold';
cb.Ticks = plot_centers(1:2:end); % Show every other spatial label to keep it uncluttered

fprintf('Scatter plot generated.\n');

%% 

figure
imagesc(choice_io_full)

figure
imagesc(stim_io_full)