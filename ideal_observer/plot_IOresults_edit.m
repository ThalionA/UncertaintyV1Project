%% Plotting Utilities for Ideal Observer Results (Metacognitive / Split Confidence)
%
%  1. Loads 'IOResults.mat'.
%  2. Plots per-animal fits for LICKS and VELOCITY.
%  3. Validates Choice Prediction.
%  4. Reconstructs Decision Posteriors from Stimulus Posteriors.
%  5. Visualizes Inferred Uncertainty.
%
%  ---------------------------------------------------------------------
% clear; close all; clc;

%% --- Part 0: Load and Setup ---
fprintf('--- Loading IOResults.mat ---\n');
if ~exist('IOResults.mat', 'file')
    error('Could not find IOResults.mat. Please run the fitting script first.');
end
load('IOResults.mat', 'IOResults');

if ~isfield(IOResults, 'animals') || isempty(IOResults.animals)
    error('IOResults.mat does not contain valid animal data.');
end

all_animals = IOResults.animals;
n_animals = numel(all_animals);

% Stack all trial tables into one big table for group stats
T_all = table();
for i_animal = 1:n_animals
    if isfield(all_animals{i_animal}, 'trial_table')
        T_all = [T_all; all_animals{i_animal}.trial_table];
    end
end

% Definitions
UNC_FIELD_P = 'unc_perceptual';
UNC_FIELD_D = 'unc_decision';

% --- Robust Field Name Resolution ---
% Different IO fitting versions use different names for the same fields.
% v2: conf_licks, conf_vel  |  v3: z_licks, z_vel  |  trial_table: licks_z, vel_z
% This helper resolves them robustly.
LICK_CANDIDATES = {'conf_licks', 'z_licks', 'licks_z'};
VEL_CANDIDATES  = {'conf_vel',   'z_vel',   'vel_z'};

function val = resolve_field(s, candidates)
    % Given a struct (or table), return the first matching field from candidates.
    if istable(s)
        available = s.Properties.VariableNames;
    elseif isstruct(s)
        available = fieldnames(s);
    else
        error('resolve_field: input must be struct or table');
    end
    for i = 1:numel(candidates)
        if ismember(candidates{i}, available)
            if istable(s)
                val = s.(candidates{i});
            else
                val = s.(candidates{i});
            end
            return;
        end
    end
    warning('resolve_field: none of [%s] found. Returning NaN.', strjoin(candidates, ', '));
    if istable(s)
        val = nan(height(s), 1);
    else
        val = NaN;
    end
end

%% --- Part 1: Per-Animal Fit & Prediction Visualization ---
fprintf('\n--- Part 1: Plotting Per-Animal Fits & Predictions ---\n');

figure('Color','w','Name','Per-Animal Fits','Position',[50 50 1400 300*n_animals]);
t_per = tiledlayout(n_animals, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

for i_animal = 1:n_animals
    ani = all_animals{i_animal};
    data = ani.data;
    preds = ani.pred;

    % --- 1. Choice Prediction (Validation) ---
    nexttile; hold on;
    % Empirical Choice
    [p_ch, p_ch_sem, g] = grpstats(data.choices, data.orientation, {'mean', 'sem', 'gname'});
    % Model Choice
    [p_pred, g_mod] = grpstats(preds.choice_hat, data.orientation, {'mean', 'gname'});

    errorbar(str2double(g), p_ch, p_ch_sem, 'ko', 'MarkerFaceColor','k', 'DisplayName','Data');
    plot(str2double(g_mod), p_pred, 'r--', 'LineWidth', 2, 'DisplayName','Pred (Lick/Vel)');

    ylabel('P(Go)'); xlim([0 90]); ylim([-0.05 1.05]);
    title(sprintf('%s: Choice', ani.tag), 'Interpreter','none');
    if i_animal==1, legend('Location','best'); end
    grid on; box on;

    % --- 2. Licks Fit ---
    nexttile; hold on;
    y_dat = resolve_field(ani.data, LICK_CANDIDATES);
    y_mod = ani.pred.licks;

    [mu_dat, sem_dat, g] = grpstats(y_dat, data.orientation, {'mean', 'sem', 'gname'});
    [mu_mod, g_mod]      = grpstats(y_mod, data.orientation, {'mean', 'gname'});

    errorbar(str2double(g), mu_dat, sem_dat, 'ko', 'MarkerFaceColor','k');
    plot(str2double(g_mod), mu_mod, 'g-o', 'LineWidth', 2, 'MarkerFaceColor','g');
    ylabel('Licks (z)'); xlim([0 90]);
    title('Licks (Fit)'); grid on; box on;

    % --- 3. Velocity Fit ---
    nexttile; hold on;
    y_dat = resolve_field(ani.data, VEL_CANDIDATES);
    y_mod = ani.pred.vel;

    [mu_dat, sem_dat, g] = grpstats(y_dat, data.orientation, {'mean', 'sem', 'gname'});
    [mu_mod, g_mod]      = grpstats(y_mod, data.orientation, {'mean', 'gname'});

    errorbar(str2double(g), mu_dat, sem_dat, 'ko', 'MarkerFaceColor','k');
    plot(str2double(g_mod), mu_mod, 'm-o', 'LineWidth', 2, 'MarkerFaceColor','m');
    ylabel('Vel (z)'); xlim([0 90]);
    title('Velocity (Fit)'); grid on; box on;

    % --- 4. Composite Confidence (Data vs Model) ---
    nexttile; hold on;
    % Data Proxy: Licks - Vel
    conf_data = resolve_field(ani.data, LICK_CANDIDATES) - resolve_field(ani.data, VEL_CANDIDATES);
    % Model Proxy: Licks_pred - Vel_pred
    conf_model = ani.pred.licks - ani.pred.vel;

    [mu_dat, sem_dat, g] = grpstats(conf_data, data.orientation, {'mean', 'sem', 'gname'});
    [mu_mod, g_mod]      = grpstats(conf_model, data.orientation, {'mean', 'gname'});

    errorbar(str2double(g), mu_dat, sem_dat, 'ko', 'MarkerFaceColor','k');
    plot(str2double(g_mod), mu_mod, 'b-o', 'LineWidth', 2, 'MarkerFaceColor','b');

    yline(0, 'k:');
    ylabel('Conf (Lick-Vel)'); xlim([0 90]);
    title('Composite Confidence'); grid on; box on;
end

%% --- Part 1b: Stimulus Distributions (Discrete Bins) ---
fprintf('\n--- Part 1b: Plotting Stimulus Histograms per Animal ---\n');

% 1. Determine Global Discrete Values across all animals
%    We do this so every subplot shares the exact same x-axis bins.
all_ori = []; all_cnt = []; all_dsp = [];
for i = 1:n_animals
    all_ori = [all_ori; all_animals{i}.data.orientation(:)];
    all_cnt = [all_cnt; all_animals{i}.data.contrast(:)];
    all_dsp = [all_dsp; all_animals{i}.data.dispersion(:)];
end

% Rounding handles floating point noise (e.g. 0.99999 vs 1.0)
u_ori = unique(round(all_ori, 1));
u_cnt = unique(round(all_cnt, 2));
u_dsp = unique(round(all_dsp, 1));

figure('Color','w','Name','Stimulus Statistics','Position',[50 50 1200 250*n_animals]);
t_stats = tiledlayout(n_animals, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle('Stimulus Distributions (Counts)');

for i_animal = 1:n_animals
    ani = all_animals{i_animal};
    d = ani.data;
    
    % --- 1. Orientation ---
    nexttile(t_stats);
    % Count occurrences of the global unique values
    counts = sum(round(d.orientation(:),1) == u_ori', 1);
    bar(u_ori, counts, 'FaceColor', 'k', 'FaceAlpha', 0.6, 'EdgeColor', 'none');
    xlim([min(u_ori)-5, max(u_ori)+5]);
    if i_animal == 1, title('Orientation'); end
    ylabel(sprintf('%s', ani.tag), 'Interpreter', 'none', 'FontWeight', 'bold');
    grid on; box off;
    
    % --- 2. Contrast ---
    nexttile(t_stats);
    counts = sum(round(d.contrast(:),2) == u_cnt', 1);
    bar(u_cnt, counts, 'FaceColor', 'b', 'FaceAlpha', 0.6, 'EdgeColor', 'none');
    xlim([-0.05 1.05]);
    if i_animal == 1, title('Contrast'); end
    grid on; box off;
    
    % --- 3. Dispersion ---
    nexttile(t_stats);
    counts = sum(round(d.dispersion(:),1) == u_dsp', 1);
    bar(u_dsp, counts, 'FaceColor', 'r', 'FaceAlpha', 0.6, 'EdgeColor', 'none');
    if ~isempty(u_dsp)
        xlim([min(u_dsp)-2, max(u_dsp)+2]);
    end
    if i_animal == 1, title('Dispersion'); end
    grid on; box off;
    
    % Bottom Labels
    if i_animal == n_animals
        xlabel(t_stats.Children(3), 'Orientation (deg)');
        xlabel(t_stats.Children(2), 'Contrast');
        xlabel(t_stats.Children(1), 'Dispersion (deg)');
    end
end

fprintf('Stimulus histograms generated using discrete bins.\n');

%% --- Part 2: Group Summaries ---
fprintf('\n--- Part 2: Plotting Group Summaries ---\n');
figure('Color','w','Name','Group Summary','Position',[100 100 1200 400]);
sgtitle('Group Level Results');

% 1. Master X-Axis
unique_orients = unique(T_all.orientation);
n_orients = length(unique_orients);

% 2. Initialize
mat_choice_dat  = nan(n_animals, n_orients);
mat_choice_mod  = nan(n_animals, n_orients);
mat_licks_mod   = nan(n_animals, n_orients);
mat_vel_mod     = nan(n_animals, n_orients);

% 3. Fill directly
for i = 1:n_animals
    ani = all_animals{i};
    % Assumes standard sorted orientations for all animals
    mat_choice_dat(i,:) = grpstats(ani.data.choices, ani.data.orientation, {'mean'});
    mat_choice_mod(i,:) = grpstats(ani.pred.choice_hat, ani.data.orientation, {'mean'});
    mat_licks_mod(i,:)  = grpstats(ani.pred.licks, ani.data.orientation, {'mean'});
    mat_vel_mod(i,:)    = grpstats(ani.pred.vel, ani.data.orientation, {'mean'});
end

% 4. Stats
avg_choice_dat = mean(mat_choice_dat, 1); sem_choice_dat = sem(mat_choice_dat);
avg_choice_mod = mean(mat_choice_mod, 1); sem_choice_mod = sem(mat_choice_mod);
avg_licks      = mean(mat_licks_mod, 1);  sem_licks      = sem(mat_licks_mod);
avg_vel        = mean(mat_vel_mod, 1);    sem_vel        = sem(mat_vel_mod);

% 5. Plot
subplot(1,3,1); hold on;
plot(unique_orients, mat_choice_mod', '-', 'Color', [1 0 0 0.2], 'LineWidth', 1);
errorbar(unique_orients, avg_choice_dat, sem_choice_dat, 'k-o', 'LineWidth', 2, 'DisplayName', 'Avg Data');
errorbar(unique_orients, avg_choice_mod, sem_choice_mod, 'r--', 'LineWidth', 3, 'DisplayName', 'Avg Prediction');
title('Choice Prediction'); xlabel('Orientation'); ylabel('P(Go)');
ylim([0 1]); xlim([0 90]); grid on; legend('Location','best');

subplot(1,3,2); hold on;
plot(unique_orients, mat_licks_mod', '-', 'Color', [0 1 0 0.3], 'LineWidth', 1);
errorbar(unique_orients, avg_licks, sem_licks, 'g-o', 'LineWidth', 3, 'MarkerFaceColor','g');
title('Licks Fit'); xlabel('Orientation'); ylabel('Licks (z)');
xlim([0 90]); grid on;

subplot(1,3,3); hold on;
plot(unique_orients, mat_vel_mod', '-', 'Color', [1 0 1 0.3], 'LineWidth', 1);
errorbar(unique_orients, avg_vel, sem_vel, 'm-o', 'LineWidth', 3, 'MarkerFaceColor','m');
title('Velocity Fit'); xlabel('Orientation'); ylabel('Velocity (z)');
xlim([0 90]); grid on;

%% --- Part 3: Inferred Uncertainty Validation ---
fprintf('\n--- Part 3: Plotting Inferred Uncertainty (Grouped) ---\n');
unc_p = T_all.(UNC_FIELD_P);
unc_d = T_all.(UNC_FIELD_D);

% Normalize (0-1) for comparison
unc_p_norm = (unc_p - min(unc_p,[],'omitnan')) / (max(unc_p,[],'omitnan') - min(unc_p,[],'omitnan'));
unc_d_norm = (unc_d - min(unc_d,[],'omitnan')) / (max(unc_d,[],'omitnan') - min(unc_d,[],'omitnan'));

figure('Color','w','Name','Uncertainty Validation','Position',[100 100 1600 450]);
sgtitle('Inferred Uncertainty Validation');

% --- Subplot 1: vs. Performance ---
subplot(1,4,1); hold on;
% Assumes alignment: <45 is Go, >45 is NoGo
is_correct = (T_all.orientation < 45 & T_all.choice == 1) | ...
    (T_all.orientation > 45 & T_all.choice == 0);

is_correct_char = cell(length(is_correct), 1);
is_correct_char(is_correct) = {'Correct'};
is_correct_char(~is_correct) = {'Incorrect'};
[mean_p, sem_p, g_p] = grpstats(unc_p_norm, is_correct_char, {'mean','sem','gname'});
[mean_d, sem_d, g_d] = grpstats(unc_d_norm, is_correct_char, {'mean','sem','gname'});

if numel(g_p) > 0
    errorbar(1:numel(g_p), mean_p, sem_p, 'bo-', 'LineWidth', 2, 'MarkerFaceColor', 'b', 'DisplayName', 'Perceptual');
    errorbar(1:numel(g_d), mean_d, sem_d, 'ro-', 'LineWidth', 2, 'MarkerFaceColor', 'r', 'DisplayName', 'Decision');
    title('vs. Performance'); ylabel('Normalized Value');
    xticks(1:numel(g_p)); xticklabels(g_p);
    legend('Location', 'best'); grid on; box on; xlim([0.5 numel(g_p)+0.5]);
end

% --- Subplots 2-4: vs. other variables ---
subplot(1,4,2); plot_uncertainty_vs_var(T_all.orientation, unc_p_norm, unc_d_norm, 'Orientation (deg)');
subplot(1,4,3); plot_uncertainty_vs_var(T_all.contrast, unc_p_norm, unc_d_norm, 'Contrast');
subplot(1,4,4); plot_uncertainty_vs_var(T_all.dispersion, unc_p_norm, unc_d_norm, 'Dispersion (deg)');

%% --- Part 4: Parameter Summary ---
fprintf('\n--- Part 4: Fitted Parameters ---\n');
figure('Color','w','Name','Parameters','Position',[100 100 1200 500]);
if isfield(IOResults.meta.model_spec, 'fit_params')
    p_names = IOResults.meta.model_spec.fit_params;
    all_p = zeros(n_animals, length(p_names));
    for i = 1:n_animals
        all_p(i,:) = all_animals{i}.fit.params_vec;
    end

    subplot(1,2,1);
    imagesc(zscore(all_p));
    colorbar; title('Z-Scored Parameters per Animal');
    xlabel('Parameter'); ylabel('Animal ID');
    xticks(1:length(p_names)); xticklabels(p_names); xtickangle(45);

    subplot(1,2,2);
    boxplot(all_p, 'Labels', p_names);
    title('Parameter Ranges');
    xtickangle(45); grid on;
end

%% --- Part 5: Uncertainty Relationships (Scatters & Heatmaps) ---
fprintf('\n--- Part 5: Plotting Uncertainty Relationships (2D) ---\n');

figure('Color','w','Name','Uncertainty Relationships','Position',[100 100 1000 800]);
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle('Uncertainty Relationships');

% Extract Data
x_vals = T_all.(UNC_FIELD_P); % Perceptual
y_vals = T_all.(UNC_FIELD_D); % Decision
good_idx = ~isnan(x_vals) & ~isnan(y_vals);

% --- 1. Scatter: Perceptual vs Decision (Color = Contrast) ---
nexttile;
scatter(x_vals(good_idx), y_vals(good_idx), 25, T_all.contrast(good_idx), ...
    'filled', 'MarkerFaceAlpha', 0.4, 'MarkerEdgeColor', 'k');
xlabel('Perceptual Uncertainty'); ylabel('Decision Uncertainty');
title('Coloured by Contrast');
colormap(gca, 'parula'); 
cb1 = colorbar; cb1.Label.String = 'Contrast';
grid on; box on;

% --- 2. Scatter: Perceptual vs Decision (Color = Dispersion) ---
nexttile;
scatter(x_vals(good_idx), y_vals(good_idx), 25, T_all.dispersion(good_idx), ...
    'filled', 'MarkerFaceAlpha', 0.4, 'MarkerEdgeColor', 'k');
xlabel('Perceptual Uncertainty'); ylabel('Decision Uncertainty');
title('Coloured by Dispersion');
colormap(gca, 'autumn'); % Use different map to distinguish
cb2 = colorbar; cb2.Label.String = 'Dispersion (deg)';
grid on; box on;

% --- 3. Heatmap: Mean Perceptual Uncertainty ---
nexttile;
[M_p, x_c, y_c] = bin_heatmap(T_all.contrast, T_all.dispersion, T_all.(UNC_FIELD_P));
imagesc(1:numel(x_c), 1:numel(y_c), M_p');
set(gca, 'YDir', 'normal');
xlabel('Contrast'); ylabel('Dispersion (deg)');
xticks(1:numel(x_c)); xticklabels(x_c);
yticks(1:numel(y_c)); yticklabels(y_c);
title('Mean Perceptual Uncertainty');
colormap(gca, 'parula'); % Reset map for this tile
colorbar;
axis square;

% --- 4. Heatmap: Mean Decision Uncertainty ---
nexttile;
[M_d, x_c, y_c] = bin_heatmap(T_all.contrast, T_all.dispersion, T_all.(UNC_FIELD_D));
imagesc(1:numel(x_c), 1:numel(y_c), M_d');
set(gca, 'YDir', 'normal');
xlabel('Contrast'); ylabel('Dispersion (deg)');
xticks(1:numel(x_c)); xticklabels(x_c);
yticks(1:numel(y_c)); yticklabels(y_c);
title('Mean Decision Uncertainty');
colormap(gca, 'parula'); % Reset map for this tile
colorbar;
axis square;

%% --- Part 6: Visualize Inferred Perceptual Posteriors ---
fprintf('\n--- Part 6: Plotting Inferred Stimulus Posteriors ---\n');
figure('Color','w','Name','Perceptual Posteriors','Position',[100 100 1200 600]);
sgtitle('Average Inferred Stimulus (s) Posterior by Choice');

if isfield(IOResults.meta.model_spec.fixed_params, 's_range_deg')
    s_range = IOResults.meta.model_spec.fixed_params.s_range_deg;
else
    error('Could not find s_range_deg in IOResults.meta');
end

t_post = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
for i_animal = 1:n_animals
    ani = all_animals{i_animal};

    % FIX: Use post_s_given_map (Trials x Stimulus Grid)
    if ~isfield(ani.inferred, 'post_s_given_map')
        warning('Animal %d missing post_s_given_map', i_animal); continue;
    end
    % posteriors = ani.inferred.post_s_given_map;
    posteriors = ani.inferred.post_s_marginal;
    choices = ani.data.choices;

    % Group by choice
    go_posteriors = posteriors(choices == 1, :);
    nogo_posteriors = posteriors(choices == 0, :);

    avg_post_go = mean(go_posteriors, 1, 'omitnan');
    sem_post_go = sem(go_posteriors);
    avg_post_nogo = mean(nogo_posteriors, 1, 'omitnan');
    sem_post_nogo = sem(nogo_posteriors);

    nexttile(t_post); hold on;
    shadedErrorBar_simple(s_range, avg_post_go, sem_post_go, 'b');
    shadedErrorBar_simple(s_range, avg_post_nogo, sem_post_nogo, 'r');

    xline(45, 'k--', 'DisplayName', 'Boundary (45\circ)');

    title(sprintf('%s', ani.tag), 'Interpreter', 'none');
    grid on; xlim([min(s_range), max(s_range)]);

    if i_animal == 1
        % Dummy lines for legend
        p1 = plot(nan,nan,'b','LineWidth',2);
        p2 = plot(nan,nan,'r','LineWidth',2);
        legend([p1 p2], {'Chose Go', 'Chose NoGo'}, 'Location', 'best');
    end
end
xlabel(t_post, 'Orientation (deg)')
ylabel(t_post, 'Posterior P(s|m)');

%% --- Part 7: Visualize Inferred Decision Posteriors ---
fprintf('\n--- Part 7: Plotting Inferred Decision Posteriors ---\n');
figure('Color','w','Name','Decision Posteriors','Position',[100 100 1200 600]);
sgtitle('Average Inferred Decision Posterior by Choice');
t_dec = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
cats = {'P(Go)', 'P(NoGo)'};

for i_animal = 1:n_animals
    ani = all_animals{i_animal};

    % RECONSTRUCTION: The fit script does not save decision_pdf.
    % We must calculate P(Go|m) from P(s|m).
    % P(Go|m) = Sum(P(s|m)) where s < 45.

    % post_s = ani.inferred.post_s_given_map;
    post_s = ani.inferred.post_s_marginal;
    s_axis = IOResults.meta.model_spec.fixed_params.s_range_deg;

    is_go_s = s_axis < 45;
    is_boundary = s_axis == 45;

    % Sum probabilities for Go Stimuli
    p_go = sum(post_s(:, is_go_s), 2);
    % Add half mass if on boundary
    if any(is_boundary)
        p_go = p_go + 0.5 * sum(post_s(:, is_boundary), 2);
    end

    decision_pdf = [p_go, 1 - p_go]; % [nTrials x 2]
    choices = ani.data.choices;

    % Group by Choice
    go_posteriors = decision_pdf(choices == 1, :);
    nogo_posteriors = decision_pdf(choices == 0, :);

    % Means and SEMs
    mu_chose_go = mean(go_posteriors, 1, 'omitnan');
    sem_chose_go = sem(go_posteriors);

    mu_chose_nogo = mean(nogo_posteriors, 1, 'omitnan');
    sem_chose_nogo = sem(nogo_posteriors);

    % Data for Bar Plot
    % Group 1: Chose Go (Blue) -> [P(Go), P(NoGo)]
    % Group 2: Chose NoGo (Red) -> [P(Go), P(NoGo)]
    y_data = [mu_chose_go; mu_chose_nogo]'; % Transpose to [2 categories x 2 choice groups]
    err_data = [sem_chose_go; sem_chose_nogo]';

    nexttile(t_dec); hold on;
    b = bar(y_data, 'grouped');

    b(1).FaceColor = 'b'; b(1).DisplayName = 'Chose Go';
    b(2).FaceColor = 'r'; b(2).DisplayName = 'Chose No-Go';
    b(1).EdgeColor = 'none'; b(2).EdgeColor = 'none';
    b(1).FaceAlpha = 0.7; b(2).FaceAlpha = 0.7;

    % Error Bars calculation
    ngroups = 2; nbars = 2;
    groupwidth = min(0.8, nbars/(nbars + 1.5));
    for i = 1:nbars
        x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
        errorbar(x, y_data(:,i), err_data(:,i), 'k.', 'LineWidth', 1.5);
    end

    title(sprintf('%s', ani.tag), 'Interpreter', 'none');
    ylim([0 1.05]); xticks([1 2]); xticklabels(cats);
    if i_animal == 1, legend({'choose GO', 'choose NOGO'}, 'Location', 'best'); end
    grid on; box off;
end
ylabel(t_dec, 'Posterior Probability');

%% --- Part 8: Metacognitive ---
fprintf('\n--- Part 8: Metacognitive Analysis (Sensitivity & Correlation) ---\n');

figure('Color','w','Name','Metacognition','Position',[100 100 1400 500]);
% sgtitle('Metacognitive Validation');

% 1. Define Confidence Metric (Magnitude of Decision)
% Raw Composite tracks the Signed DV (High = Go, Low = NoGo)
raw_composite = resolve_field(T_all, LICK_CANDIDATES) - resolve_field(T_all, VEL_CANDIDATES);

% We take ABS() to get "Confidence" (Distance from decision boundary)
% High Abs = Certain (either Certain Go or Certain NoGo)
% Low Abs  = Uncertain (near boundary)
T_all.conf_abs = abs(raw_composite);

% --- Subplot 1: Metacognitive Sensitivity ("X-Pattern") ---
subplot(1,3,1); hold on;

% Define Difficulty exactly (No Binning)
T_all.difficulty = abs(T_all.orientation - 45);

% Define Correct/Incorrect
is_corr = (T_all.orientation < 45 & T_all.choice == 1) | ...
    (T_all.orientation > 45 & T_all.choice == 0);

% Calculate Stats grouped by exact Difficulty values
[mu_c, g_c, sem_c] = grpstats(T_all.conf_abs(is_corr), T_all.difficulty(is_corr), {'mean','gname','sem'});
[mu_e, g_e, sem_e] = grpstats(T_all.conf_abs(~is_corr), T_all.difficulty(~is_corr), {'mean','gname','sem'});

% Convert group names back to numbers
x_c = str2double(g_c);
x_e = str2double(g_e);

% Plot Correct (Expect increasing slope)
if ~isempty(x_c)
    errorbar(x_c, mu_c, sem_c, 'b-o', 'LineWidth', 2, 'DisplayName', 'Correct');
end

% Plot Error (Expect flat or decreasing slope)
if ~isempty(x_e)
    errorbar(x_e, mu_e, sem_e, 'r-o', 'LineWidth', 2, 'DisplayName', 'Error');
end

xlabel('Difficulty (|Orientation - 45|)');
ylabel('Confidence Magnitude |Licks - Vel|');
% title('Metacognitive Sensitivity');
legend('Location','best'); grid on; box on;
xlim([-1 max(T_all.difficulty)+1]); % Pad slightly

% --- Subplot 2: Psychometric Slopes Split by Confidence ---
subplot(1,3,2); hold on;

% Median Split based on the ABSOLUTE confidence
med_conf = median(T_all.conf_abs, 'omitnan');
is_high_conf = T_all.conf_abs > med_conf;

% Group by exact Orientation (No Binning)
[p_high, g_high, sem_high] = grpstats(T_all.choice(is_high_conf), T_all.orientation(is_high_conf), {'mean','gname','sem'});
[p_low, g_low, sem_low]    = grpstats(T_all.choice(~is_high_conf), T_all.orientation(~is_high_conf), {'mean','gname','sem'});

x_high = str2double(g_high);
x_low  = str2double(g_low);

% Plot High Confidence (Expect Steeper Slope)
errorbar(x_high, p_high, sem_high, 'k-o', 'LineWidth', 2, 'MarkerFaceColor','k', 'DisplayName', 'High Conf');

% Plot Low Confidence (Expect Shallower Slope)
errorbar(x_low, p_low, sem_low, 'k--', 'LineWidth', 1, 'Color', [0.6 0.6 0.6], 'DisplayName', 'Low Conf');

xlabel('Orientation (deg)'); ylabel('P(Go)');
title('Psychometric Split');
legend('Location','best'); grid on; box on;
xlim([min(T_all.orientation)-2, max(T_all.orientation)+2]); ylim([0 1]);

% --- Subplot 3: Channel Correlation ---
subplot(1,3,3)

% Plot all points
licks_z_vals = resolve_field(T_all, LICK_CANDIDATES);
vel_z_vals = resolve_field(T_all, VEL_CANDIDATES);
scatter(licks_z_vals, vel_z_vals, 50, 'filled', 'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'w');
lsline
% Correlation
[Rho, Pval] = corr(licks_z_vals, vel_z_vals, 'rows','complete');
title(sprintf('R = %.2f, p= %.3f', Rho, Pval));
xlabel('Licks (z)'); ylabel('Velocity (z)');
grid on; box on;
ylim([min(vel_z_vals), max(vel_z_vals)])

fprintf('Metacognitive plots generated using Absolute Confidence metric and Exact Orientations.\n');

%% --- Part 9: Average Posteriors by Contrast, Dispersion, and Stimulus Class ---
fprintf('\n--- Part 9: Plotting Posteriors by Stimulus Quality and Class ---\n');

s_range = IOResults.meta.model_spec.fixed_params.s_range_deg;
posteriors_all = [];
c_all   = []; 
d_all   = [];
ori_all = [];

for i_animal = 1:n_animals
    ani = all_animals{i_animal};
    if isfield(ani.inferred, 'post_s_given_map')
        % post = ani.inferred.post_s_given_map;
        post = ani.inferred.post_s_marginal;
        posteriors_all = [posteriors_all; post];
        c_all   = [c_all; ani.data.contrast(:)];
        d_all   = [d_all; ani.data.dispersion(:)];
        ori_all = [ori_all; ani.data.orientation(:)];
    end
end

if isempty(posteriors_all)
    warning('post_s_given_map not found. Skipping Part 9.');
else
    figure('Color','w','Name','Posteriors by Quality & Stim Class','Position',[100 100 1400 800]);
    
    u_c = unique(round(c_all, 2));
    u_d = unique(round(d_all, 1));
    
    % Define Stimulus Classes
    is_go   = ori_all < 45;
    is_nogo = ori_all > 45;
    
    classes = {'Go Stimuli (<45\circ)', 'NoGo Stimuli (>45\circ)'};
    masks   = {is_go, is_nogo};
    
    t_post = tiledlayout(2, 2, 'TileSpacing', 'compact');
    
    for cls_idx = 1:2
        current_mask = masks{cls_idx};
        
        % Subplot: Contrast
        nexttile(cls_idx); hold on;
        colors_c = parula(length(u_c) + 1);
        for i = 1:length(u_c)
            idx = current_mask & (round(c_all, 2) == u_c(i));
            if any(idx)
                avg_post = mean(posteriors_all(idx, :), 1, 'omitnan');
                plot(s_range, avg_post, 'LineWidth', 2, 'Color', colors_c(i,:), ...
                    'DisplayName', sprintf('C = %.2f', u_c(i)));
            end
        end
        xline(45, 'k--');
        title(sprintf('By Contrast: %s', classes{cls_idx}));
        xlabel('Orientation (deg)'); ylabel('P(s|m)');
        if cls_idx == 1, legend('Location', 'best'); end
        grid on;
        
        % Subplot: Dispersion
        nexttile(cls_idx + 2); hold on;
        colors_d = autumn(length(u_d) + 1);
        for i = 1:length(u_d)
            idx = current_mask & (round(d_all, 1) == u_d(i));
            if any(idx)
                avg_post = mean(posteriors_all(idx, :), 1, 'omitnan');
                plot(s_range, avg_post, 'LineWidth', 2, 'Color', colors_d(i,:), ...
                    'DisplayName', sprintf('D = %.1f', u_d(i)));
            end
        end
        xline(45, 'k--');
        title(sprintf('By Dispersion: %s', classes{cls_idx}));
        xlabel('Orientation (deg)'); ylabel('P(s|m)');
        if cls_idx == 1, legend('Location', 'best'); end
        grid on;
    end
end

%% --- EXTRA: Compare MAP vs Marginalized Posteriors ---
fprintf('\n--- Plotting MAP vs Marginalized Comparison ---\n');

% We will use the first animal as a representative example
ani_idx = 6; 
ani = all_animals{ani_idx};
s_range = IOResults.meta.model_spec.fixed_params.s_range_deg;

% Extract Uncertainty arrays
unc_marg = ani.inferred.perceptual;       % Fully Marginalized
unc_map  = ani.inferred.perceptual_map;   % MAP Approximation

figure('Color','w','Name','MAP vs Marginalized Comparison','Position',[100 100 1300 400]);
sgtitle(sprintf('Why Marginalization Matters (%s)', ani.tag), 'Interpreter', 'none', 'FontWeight', 'bold');

% --- Subplot 1: Scatter Plot showing Systematic Bias ---
subplot(1,3,1); hold on;
scatter(unc_marg, unc_map, 20, 'k', 'filled', 'MarkerFaceAlpha', 0.2);

% Draw identity line (y = x)
max_val = max([unc_marg; unc_map], [], 'omitnan');
plot([0 max_val], [0 max_val], 'r--', 'LineWidth', 2, 'DisplayName', 'Identity (y=x)');

xlabel('Marginalized Uncertainty (True Bayesian)');
ylabel('MAP Uncertainty (Approximation)');
title('Systematic Underestimation by MAP');
legend('Location', 'northwest');
grid on; axis square;
% You should see almost all points fall BELOW the red line, 
% proving MAP artificially truncates uncertainty.

% --- Find representative trials ---
% We define "Confidence" magnitude based on the behavioral outputs
conf_metric = abs(resolve_field(ani.data, LICK_CANDIDATES) - resolve_field(ani.data, VEL_CANDIDATES));

[~, high_conf_idx] = max(conf_metric); % Trial where behavior strongly pinned down 'm'
[~, low_conf_idx]  = min(conf_metric); % Trial where behavior was highly ambiguous

% --- Subplot 2: High Confidence Trial ---
subplot(1,3,2); hold on;
plot(s_range, ani.inferred.post_s_marginal(high_conf_idx,:), 'b-', 'LineWidth', 2.5, 'DisplayName', 'Marginalized');
plot(s_range, ani.inferred.post_s_given_map(high_conf_idx,:), 'r--', 'LineWidth', 2, 'DisplayName', 'MAP');
title(sprintf('Trial %d: High Confidence Behavior', high_conf_idx));
subtitle('When behavior is strong, MAP is a decent approximation');
xlabel('Orientation (deg)'); ylabel('P(s | behavior)');
legend('Location', 'best'); grid on;
xlim([min(s_range), max(s_range)]);

% --- Subplot 3: Low Confidence Trial ---
subplot(1,3,3); hold on;
plot(s_range, ani.inferred.post_s_marginal(low_conf_idx,:), 'b-', 'LineWidth', 2.5, 'DisplayName', 'Marginalized');
plot(s_range, ani.inferred.post_s_given_map(low_conf_idx,:), 'r--', 'LineWidth', 2, 'DisplayName', 'MAP');
title(sprintf('Trial %d: Ambiguous Behavior', low_conf_idx));
subtitle('When behavior is weak, MAP severely underestimates uncertainty');
xlabel('Orientation (deg)'); ylabel('P(s | behavior)');
legend('Location', 'best'); grid on;
xlim([min(s_range), max(s_range)]);



%% --- Part 10: Likelihood vs Posterior Comparison Per Animal ---
fprintf('\n--- Part 10: Likelihood vs Posterior Comparison ---\n');

s_range = IOResults.meta.model_spec.fixed_params.s_range_deg;

for i_animal = 1:n_animals
    ani = all_animals{i_animal};
    
    % Robust field access
    if isfield(ani.inferred, 'L_s_marginal')
        liks = ani.inferred.L_s_marginal;
    elseif isfield(ani.inferred, 'likelihood_marginal')
        liks = ani.inferred.likelihood_marginal;
    else
        fprintf('  Skipping %s: no likelihood field found.\n', ani.tag);
        continue;
    end
    posts = ani.inferred.post_s_marginal;
    oris = ani.data.orientation;
    
    figure('Color','w','Name', sprintf('Likelihood vs Posterior - %s', ani.tag), ...
           'Position', [100 100 1400 900]);
    sgtitle(sprintf('Likelihood vs Posterior: %s', ani.tag), ...
            'FontWeight', 'bold', 'Interpreter', 'none');
    
    % --- Row 1: Population Heatmaps (sorted by orientation) ---
    [sorted_oris, sort_idx] = sort(oris);
    num_trials = length(oris);
    
    subplot(3, 3, 1);
    imagesc(s_range, 1:num_trials, liks(sort_idx, :));
    colormap(gca, 'parula'); colorbar;
    title('Likelihood L(s)');
    ylabel('Trial (sorted by ori)');
    hold on; plot(sorted_oris, 1:num_trials, 'r-', 'LineWidth', 1.5);
    
    subplot(3, 3, 2);
    imagesc(s_range, 1:num_trials, posts(sort_idx, :));
    colormap(gca, 'parula'); colorbar;
    title('Posterior P(s|data)');
    hold on; plot(sorted_oris, 1:num_trials, 'r-', 'LineWidth', 1.5);
    
    subplot(3, 3, 3);
    diff_map = posts(sort_idx, :) - liks(sort_idx, :);
    imagesc(s_range, 1:num_trials, diff_map);
    colormap("parula"); colorbar;
    title('Difference (Posterior - Likelihood)');
    % subtitle('Red = Prior pushes mass here');
    hold on; plot(sorted_oris, 1:num_trials, 'k-', 'LineWidth', 1.5);
    clim([-max(abs(diff_map(:))) max(abs(diff_map(:)))]);
    
    % --- Row 2: Average Distributions by Stimulus Class ---
    is_go = oris < 45;
    is_ambig = abs(oris - 45) <= 10; % Near-boundary
    is_nogo = oris > 45;
    class_names = {'Go (ori < 45)', 'Ambiguous (35-55)', 'NoGo (ori > 55)'};
    class_masks = {is_go & ~is_ambig, is_ambig, is_nogo & ~is_ambig};
    
    for cls = 1:3
        subplot(3, 3, 3 + cls); hold on;
        mask = class_masks{cls};
        if sum(mask) == 0, title(class_names{cls}); continue; end
        
        avg_lik = mean(liks(mask, :), 1, 'omitnan');
        avg_post = mean(posts(mask, :), 1, 'omitnan');
        sem_lik = std(liks(mask, :), 0, 1, 'omitnan') / sqrt(sum(mask));
        sem_post = std(posts(mask, :), 0, 1, 'omitnan') / sqrt(sum(mask));
        
        shadedErrorBar_simple(s_range, avg_lik, sem_lik, [0.3 0.7 0.9]);
        shadedErrorBar_simple(s_range, avg_post, sem_post, [0.9 0.4 0.2]);
        
        xline(45, 'k--');
        title(sprintf('%s (n=%d)', class_names{cls}, sum(mask)));
        xlabel('Orientation (deg)'); ylabel('Probability');
        if cls == 1
            p1 = plot(nan,nan,'-','Color',[0.3 0.7 0.9],'LineWidth',2);
            p2 = plot(nan,nan,'-','Color',[0.9 0.4 0.2],'LineWidth',2);
            legend([p1 p2], {'Likelihood','Posterior'}, 'Location','best');
        end
        grid on;
    end
    
    % --- Row 3: Average by Contrast and Dispersion ---
    u_c = unique(round(ani.data.contrast, 2));
    u_d = unique(round(ani.data.dispersion, 1));
    
    subplot(3, 3, 7); hold on;
    colors_c = parula(length(u_c) + 1);
    for i = 1:length(u_c)
        mask = round(ani.data.contrast, 2) == u_c(i);
        if ~any(mask), continue; end
        avg_diff = mean(posts(mask,:) - liks(mask,:), 1, 'omitnan');
        plot(s_range, avg_diff, 'LineWidth', 2, 'Color', colors_c(i,:), ...
             'DisplayName', sprintf('C=%.2f', u_c(i)));
    end
    yline(0, 'k:'); xline(45, 'k--');
    title('Prior Effect by Contrast');
    xlabel('Orientation (deg)'); ylabel('Post - Lik');
    legend('Location','best','FontSize',7); grid on;
    
    subplot(3, 3, 8); hold on;
    colors_d = autumn(length(u_d) + 1);
    for i = 1:length(u_d)
        mask = round(ani.data.dispersion, 1) == u_d(i);
        if ~any(mask), continue; end
        avg_diff = mean(posts(mask,:) - liks(mask,:), 1, 'omitnan');
        plot(s_range, avg_diff, 'LineWidth', 2, 'Color', colors_d(i,:), ...
             'DisplayName', sprintf('D=%.1f', u_d(i)));
    end
    yline(0, 'k:'); xline(45, 'k--');
    title('Prior Effect by Dispersion');
    xlabel('Orientation (deg)'); ylabel('Post - Lik');
    legend('Location','best','FontSize',7); grid on;
    
    % --- KL Divergence: How much does the prior contribute? ---
    subplot(3, 3, 9); hold on;
    kl_per_trial = sum(posts .* log((posts + 1e-10) ./ (liks + 1e-10)), 2);
    [mu_kl, g_kl, sem_kl] = grpstats(kl_per_trial, round(oris), {'mean','gname','sem'});
    x_kl = str2double(g_kl);
    errorbar(x_kl, mu_kl, sem_kl, 'k-o', 'LineWidth', 2, 'MarkerFaceColor', [0.5 0.2 0.8]);
    xlabel('Orientation (deg)'); ylabel('KL(Post || Lik)');
    title('Prior Influence by Orientation');
    subtitle('Higher KL = Prior matters more');
    grid on; xlim([0 90]);
    
    drawnow;
end

%% --- Helper Functions ---

function plot_uncertainty_vs_var(variable, var1_data, var2_data, var_name)
hold on;
[bin, edges] = discretize(variable, 8); % Reduced bins for cleaner plot

if isempty(bin) || all(isnan(bin)), title(sprintf('%s (No Data)', var_name)); return; end
centers = edges(1:end-1) + diff(edges)/2;

[mean1, g1, sem1] = grpstats(var1_data, bin, {'mean','gname','sem'});
[mean2, g2, sem2] = grpstats(var2_data, bin, {'mean','gname','sem'});
g1 = str2double(g1); g2 = str2double(g2);

if ~isempty(g1) && any(~isnan(mean1))
    errorbar(centers(g1), mean1, sem1, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
end
if ~isempty(g2) && any(~isnan(mean2))
    errorbar(centers(g2), mean2, sem2, 'r-s', 'LineWidth', 2, 'MarkerFaceColor', 'r');
end

title(var_name); xlabel(var_name); ylabel('Normalized Unc.');
grid on; box on; ylim([-0.1 1.1]);
xlim([min(edges) max(edges)]);
end

function [M, x_centers, y_centers] = bin_heatmap(x, y, z)
good = ~isnan(x) & ~isnan(y) & ~isnan(z);
x = x(good); y = y(good); z = z(good);
x_centers = sort(unique(x)); y_centers = sort(unique(y));
n_x = numel(x_centers); n_y = numel(y_centers);
[~, x_idx] = ismember(x, x_centers);
[~, y_idx] = ismember(y, y_centers);
M = accumarray([x_idx, y_idx], z, [n_x, n_y], @(v) mean(v, 'omitnan'), NaN);
end


function shadedErrorBar_simple(x, y, err, color_spec)
% Simple shaded error bar implementation. Accepts char ('b') or RGB vector ([0.3 0.7 0.9]).
upper = y + err;
lower = y - err;
x_poly = [x, fliplr(x)];
y_poly = [upper, fliplr(lower)];
if ischar(color_spec)
    fill(x_poly, y_poly, color_spec, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    plot(x, y, color_spec, 'LineWidth', 1.5);
else
    fill(x_poly, y_poly, color_spec, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    plot(x, y, 'Color', color_spec, 'LineWidth', 1.5);
end
end

function cmap = redblue_custom(n)
% Diverging red-blue colormap for difference maps.
if nargin < 1, n = 256; end
half = floor(n/2);
r = [linspace(0, 1, half)'; ones(n - half, 1)];
g = [linspace(0, 1, half)'; linspace(1, 0, n - half)'];
b = [ones(half, 1); linspace(1, 0, n - half)'];
cmap = [r g b];
end