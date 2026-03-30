%% --- (Corrected) Behavioral Confidence Plots ---
% This block corrects binning issues and consolidates the five confidence
% plots into a single, "prettified" figure, plus a 6th bonus plot.

fprintf('\n--- Plotting Behavioral Confidence Dashboard ---\n');

% --- Setup ---
if ~exist('TrialTbl_all', 'var')
    error('TrialTbl_all is not in the workspace. Please run the main analysis first.');
end

T_all = TrialTbl_all; % Working copy
animal_names = unique(T_all.animal);
nAnimals = numel(animal_names);
n_quantiles = 5; % Bins for plots D & E

% Create a clean, rounded contrast variable for grouping
T_all.contrast_rounded = round(T_all.contrast, 3);
T_all.contrast_rounded(abs(T_all.contrast_rounded - 1.0) < 0.02) = 1.0;

% Create figure
figure('Name', 'Behavioral Confidence Dashboard', 'Color', 'w', 'Position', [100 100 1600 900]);
tl = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle('Behavioural Confidence Analysis');

% --- Plot A: Confidence vs. Orientation ---
ax1 = nexttile; hold on; box on; grid on;
title('A. Confidence vs. Orientation');
xlabel('|\Delta from Go| (deg)');
ylabel('Confidence');

% Storage for per-animal curves
all_x_ori = {};
all_y_ori = {};

for i = 1:nAnimals
    ani_mask = strcmp(T_all.animal, animal_names{i});
    T_ani = T_all(ani_mask, :);
    
    [mu, x_str] = grpstats(T_ani.confidence, T_ani.abs_from_go, {'mean', 'gname'});
    x_num = str2double(x_str);
    
    % Plot individual animal (light gray)
    plot(x_num, mu, 'o-', 'Color', [0.8 0.8 0.8], 'LineWidth', 0.5);
    
    % Store for averaging (need to interpolate)
    all_x_ori{i} = x_num;
    all_y_ori{i} = mu;
end

% Plot group mean (bold)
[mu_group, sem_group, x_group_str] = grpstats(T_all.confidence, T_all.abs_from_go, {'mean', 'sem', 'gname'});
x_group_num = str2double(x_group_str);
shadedErrorBar(x_group_num, mu_group, sem_group, 'lineprops', {'-o', 'Color', 'k', 'MarkerFaceColor', 'k'});
xticks(sort(x_group_num));
xlim([min(x_group_num)-1, max(x_group_num)+1]);


% --- Plot B: |Confidence| vs. Contrast ---
ax2 = nexttile; hold on; box on; grid on;
title('B. |Confidence| vs. Contrast');
xlabel('Contrast');
ylabel('|Confidence|');

T_all.abs_confidence = abs(T_all.confidence); % Use absolute confidence

for i = 1:nAnimals
    ani_mask = strcmp(T_all.animal, animal_names{i});
    T_ani = T_all(ani_mask, :);
    
    [mu, x_str] = grpstats(T_ani.abs_confidence, T_ani.contrast_rounded, {'mean', 'gname'});
    x_num = str2double(x_str);
    
    plot(x_num, mu, 'o-', 'Color', [0.8 0.8 0.8], 'LineWidth', 0.5);
end

% Plot group mean (bold)
[mu_group_c, sem_group_c, x_group_str_c] = grpstats(T_all.abs_confidence, T_all.contrast_rounded, {'mean', 'sem', 'gname'});
x_group_num_c = str2double(x_group_str_c);
shadedErrorBar(x_group_num_c, mu_group_c, sem_group_c, 'lineprops', {'-o', 'Color', [0.3 0.7 0.3], 'MarkerFaceColor', [0.3 0.7 0.3]});
xticks(sort(x_group_num_c));
xticklabels(num2str(x_group_num_c));
xlim([min(x_group_num_c)-0.05, max(x_group_num_c)+0.05]);


% --- Plot C: |Confidence| vs. Dispersion ---
ax3 = nexttile; hold on; box on; grid on;
title('C. |Confidence| vs. Dispersion');
xlabel('Dispersion (deg)');
ylabel('|Confidence|');

for i = 1:nAnimals
    ani_mask = strcmp(T_all.animal, animal_names{i});
    T_ani = T_all(ani_mask, :);
    
    [mu, x_str] = grpstats(T_ani.abs_confidence, T_ani.dispersion, {'mean', 'gname'});
    x_num = str2double(x_str);
    
    plot(x_num, mu, 'o-', 'Color', [0.8 0.8 0.8], 'LineWidth', 0.5);
end

% Plot group mean (bold)
[mu_group_d, sem_group_d, x_group_str_d] = grpstats(T_all.abs_confidence, T_all.dispersion, {'mean', 'sem', 'gname'});
x_group_num_d = str2double(x_group_str_d);
shadedErrorBar(x_group_num_d, mu_group_d, sem_group_d, 'lineprops', {'-o', 'Color', [0.7 0.3 0.3], 'MarkerFaceColor', [0.7 0.3 0.3]});
xticks(sort(x_group_num_d));
xlim([min(x_group_num_d)-1, max(x_group_num_d)+1]);


% --- Quantile Binning Logic (for D & E) ---
% Get global quantile edges from all data
valid_conf = T_all.confidence(~isnan(T_all.confidence));
q_edges = quantile(valid_conf, linspace(0, 1, n_quantiles + 1));
q_edges = uniquetol(q_edges, 1e-9); % Ensure unique edges
q_edges(end) = q_edges(end) + eps; % Include max value
n_bins_actual = numel(q_edges) - 1;
% Calculate bin centers based on the mean of the data *within* each bin (more accurate)
[~, ~, bin_idx_global] = histcounts(valid_conf, q_edges);
q_bin_centers = accumarray(bin_idx_global(bin_idx_global>0), valid_conf(bin_idx_global>0), [n_bins_actual 1], @mean, NaN)';

% Pre-allocate per-animal storage
prop_go_by_conf = nan(nAnimals, n_bins_actual);
accuracy_by_conf = nan(nAnimals, n_bins_actual);

for i = 1:nAnimals
    ani_mask = strcmp(T_all.animal, animal_names{i});
    
    % Get animal's data
    ani_conf = T_all.confidence(ani_mask);
    ani_choice = T_all.goChoice(ani_mask);
    ani_perf = T_all.performance(ani_mask);
    
    % Get bin indices based on global edges
    [~, ~, bin_idx] = histcounts(ani_conf, q_edges);
    
    % Calculate means per bin
    valid_bins = bin_idx > 0;
    if any(valid_bins)
        prop_go_by_conf(i, :) = accumarray(bin_idx(valid_bins), ani_choice(valid_bins), [n_bins_actual 1], @(x) mean(x, 'omitnan'), NaN)';
        accuracy_by_conf(i, :) = accumarray(bin_idx(valid_bins), ani_perf(valid_bins), [n_bins_actual 1], @(x) mean(x, 'omitnan'), NaN)';
    end
end

% --- Plot D: P(Go) vs. Confidence Quantiles ---
ax4 = nexttile; hold on; box on; grid on;
title('D. Choice vs. Confidence');
xlabel('Confidence (Binned by Global Quantile)');
ylabel('P(Go Choice)');

mu_go = mean(prop_go_by_conf, 1, 'omitnan');
sem_go = std(prop_go_by_conf, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(prop_go_by_conf), 1));
shadedErrorBar(q_bin_centers, mu_go, sem_go, 'lineprops', {'-o', 'Color', [0.2 0.2 0.8], 'MarkerFaceColor', [0.2 0.2 0.8]});
ylim([0 1]);
% xlim([min(q_edges) max(q_edges)]);


% --- Plot E: Accuracy vs. Confidence Quantiles ---
ax5 = nexttile; hold on; box on; grid on;
title('E. Accuracy vs. Confidence');
xlabel('Confidence (Binned by Global Quantile)');
ylabel('Accuracy (P(Correct))');

mu_acc = mean(accuracy_by_conf, 1, 'omitnan');
sem_acc = std(accuracy_by_conf, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(accuracy_by_conf), 1));
shadedErrorBar(q_bin_centers, mu_acc, sem_acc, 'lineprops', {'-o', 'Color', [0.8 0.2 0.8], 'MarkerFaceColor', [0.8 0.2 0.8]});
yline(0.5, 'k:', 'Chance');
ylim([0 1]);
% xlim([min(q_edges) max(q_edges)]);

% --- Plot F: Histogram of Confidence ---
ax6 = nexttile; hold on; box on; grid on;
title('F. Distribution of Confidence Proxy');
xlabel('Confidence');
ylabel('Probability Density');
histogram(T_all.confidence, 100, 'Normalization', 'pdf', 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', 'none');
% Add vlines for the quantile edges
xline(q_edges, 'r--', 'LineWidth', 0.5, 'Alpha', 0.7);
xlim([min(q_edges) max(q_edges)]);

