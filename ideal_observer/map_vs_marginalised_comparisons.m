%% Load the ideal observer results
load('IOResults.mat');

%% Pooled data plots
% Initialize containers for pooled data
pooled_trial_data = table();
pooled_P_map = [];
pooled_Q_marg = [];

% Loop through all animals and concatenate their data
n_animals = length(IOResults.animals);
for a = 1:n_animals
    pooled_trial_data = [pooled_trial_data; IOResults.animals{a}.trial_table];
    pooled_P_map = [pooled_P_map; IOResults.animals{a}.inferred.post_s_given_map];
    pooled_Q_marg = [pooled_Q_marg; IOResults.animals{a}.inferred.post_s_marginal];
end

% -------------------------------------------------------------------------
% 1. Compare Decision Uncertainty
% -------------------------------------------------------------------------
figure('Name', 'Pooled Decision Uncertainty', 'Color', 'w', 'Position', [100, 100, 500, 500]);
% Lowered alpha for dense pooled plotting
scatter(pooled_trial_data.unc_decision_map, pooled_trial_data.unc_decision, 15, 'filled', ...
    'MarkerFaceAlpha', 0.1, 'MarkerFaceColor', [0.2 0.4 0.7]);
hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 2); 
xlabel('MAP Decision Uncertainty (bits)');
ylabel('Marginalised Decision Uncertainty (bits)');
title(sprintf('Decision Uncertainty (Pooled, n=%d trials)', height(pooled_trial_data)));
% axis square;
box off;
axis tight

[R_dec, ~] = corrcoef(pooled_trial_data.unc_decision_map, pooled_trial_data.unc_decision);
fprintf('Pooled Correlation for Decision Uncertainty: %.3f\n', R_dec(1,2));



% -------------------------------------------------------------------------
% 2. Compare Perceptual Uncertainty
% -------------------------------------------------------------------------

% Keep only valid trials where underflow didn't break the calculation
valid_idx = pooled_trial_data.unc_perceptual < 90;

figure('Name', 'Pooled Perceptual Uncertainty', 'Color', 'w', 'Position', [650, 100, 500, 500]);
scatter(pooled_trial_data.unc_perceptual_map(valid_idx), pooled_trial_data.unc_perceptual(valid_idx), 15, 'filled', ...
    'MarkerFaceAlpha', 0.1, 'MarkerFaceColor', [0.7 0.3 0.3]);
hold on;
xlabel('MAP Perceptual Uncertainty (deg)');
ylabel('Marginalised Perceptual Uncertainty (deg)');
title(sprintf('Perceptual Uncertainty (Pooled, n=%d trials)', height(pooled_trial_data)));
axis square;
box off;
axis tight
identity_line

[R_perc, ~] = corrcoef(pooled_trial_data.unc_perceptual_map(valid_idx), pooled_trial_data.unc_perceptual(valid_idx));
fprintf('Pooled Correlation for Perceptual Uncertainty: %.3f\n', R_perc(1,2));

% -------------------------------------------------------------------------
% 3. Calculate Jensen-Shannon (JS) Divergence
% -------------------------------------------------------------------------
M = 0.5 * (pooled_P_map + pooled_Q_marg);

kl_P_M = sum(pooled_P_map .* log2((pooled_P_map + eps) ./ (M + eps)), 2);
kl_Q_M = sum(pooled_Q_marg .* log2((pooled_Q_marg + eps) ./ (M + eps)), 2);

js_div = 0.5 * kl_P_M + 0.5 * kl_Q_M;
pooled_trial_data.js_div = js_div;

% -------------------------------------------------------------------------
% 4. Plot JS Divergence by Condition (Orientation, Contrast, Dispersion)
% -------------------------------------------------------------------------
figure('Name', 'JS Divergence by Stimulus Condition (Pooled)', 'Color', 'w', 'Position', [100, 650, 1000, 450]);

% --- Subplot A: Boxplot by Orientation ---
subplot(1, 2, 1);
boxplot(pooled_trial_data.js_div, pooled_trial_data.orientation, 'Colors', 'k', 'Symbol', 'k.');
xlabel('Stimulus Orientation (deg)', 'FontWeight', 'bold');
ylabel('Jensen-Shannon Divergence (bits)', 'FontWeight', 'bold');
title('JS Divergence vs. Orientation (Pooled)');
box off;
set(gca, 'TickDir', 'out', 'FontSize', 12);

% --- Subplot B: 2D Heatmap (Contrast vs Dispersion) ---
subplot(1, 2, 2);
u_c = unique(pooled_trial_data.contrast);
u_d = unique(pooled_trial_data.dispersion);

hm_data = nan(length(u_d), length(u_c));

for i = 1:length(u_c)
    for j = 1:length(u_d)
        idx = (pooled_trial_data.contrast == u_c(i)) & (pooled_trial_data.dispersion == u_d(j));
        if any(idx)
            % Using omitnan in case any extreme trials evaluate to NaN
            hm_data(j, i) = mean(pooled_trial_data.js_div(idx), 'omitnan');
        end
    end
end

h = heatmap(u_c, u_d, hm_data);
h.Title = 'Mean JS Divergence (MAP vs Marginalised)';
h.XLabel = 'Contrast';
h.YLabel = 'Dispersion';
h.Colormap = parula; 
h.MissingDataColor = [0.9 0.9 0.9];

%% Hierarchical Data Plots (Animal-by-Animal Integration)
fprintf('Generating hierarchical plots...\n');

% Identify unique conditions across the whole dataset
u_ori = unique(pooled_trial_data.orientation);
u_c   = unique(pooled_trial_data.contrast);
u_d   = unique(pooled_trial_data.dispersion);

% Pre-allocate arrays to hold the per-animal means
% Dimensions: [Animal x Orientation]
animal_js_ori = nan(n_animals, length(u_ori));
% Dimensions: [Dispersion x Contrast x Animal]
animal_js_hm  = nan(length(u_d), length(u_c), n_animals);

% --- 1. Integrate data WITHIN each animal ---
for a = 1:n_animals
    % Extract current animal's data
    t_data = IOResults.animals{a}.trial_table;
    P_map_a = IOResults.animals{a}.inferred.post_s_given_map;
    Q_marg_a = IOResults.animals{a}.inferred.post_s_marginal;
    
    % Compute JS Divergence for this specific animal
    M_a = 0.5 * (P_map_a + Q_marg_a);
    kl_P_a = sum(P_map_a .* log2((P_map_a + eps) ./ (M_a + eps)), 2);
    kl_Q_a = sum(Q_marg_a .* log2((Q_marg_a + eps) ./ (M_a + eps)), 2);
    t_data.js_div = 0.5 * kl_P_a + 0.5 * kl_Q_a;
    
    % Calculate Mean JS by Orientation for this animal
    for i = 1:length(u_ori)
        idx = t_data.orientation == u_ori(i);
        animal_js_ori(a, i) = mean(t_data.js_div(idx), 'omitnan');
    end
    
    % Calculate Mean JS by Contrast & Dispersion for this animal
    for i = 1:length(u_c)
        for j = 1:length(u_d)
            idx = (t_data.contrast == u_c(i)) & (t_data.dispersion == u_d(j));
            animal_js_hm(j, i, a) = mean(t_data.js_div(idx), 'omitnan');
        end
    end
end

% --- 2. Present data ACROSS animals ---
figure('Name', 'JS Divergence (Hierarchical)', 'Color', 'w', 'Position', [150, 200, 1000, 450]);

% -- Subplot A: Animal Trajectories + Grand Mean (Orientation) --
subplot(1, 2, 1);
hold on;

% Plot individual animal trajectories (light grey)
plot(u_ori, animal_js_ori', '-', 'Color', [0.6 0.6 0.6 0.5], 'LineWidth', 1.5, 'HandleVisibility', 'off');

% Calculate and plot Grand Mean + Standard Error of the Mean (SEM)
grand_mean_ori = mean(animal_js_ori, 1, 'omitnan');
grand_sem_ori  = std(animal_js_ori, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(animal_js_ori), 1));

errorbar(u_ori, grand_mean_ori, grand_sem_ori, 'k-o', 'LineWidth', 2.5, ...
    'MarkerFaceColor', 'w', 'MarkerSize', 6, 'DisplayName', 'Grand Mean ± SEM');

xlabel('Stimulus Orientation (deg)', 'FontWeight', 'bold');
ylabel('Mean Jensen-Shannon Divergence (bits)', 'FontWeight', 'bold');
title('JS Divergence vs. Orientation');
legend('Location', 'best', 'Box', 'off');
box off;
set(gca, 'TickDir', 'out', 'FontSize', 12);

% -- Subplot B: Grand Mean Heatmap (Contrast x Dispersion) --
subplot(1, 2, 2);

% Average the 2D heatmaps across the 3rd dimension (animals)
grand_mean_hm = mean(animal_js_hm, 3, 'omitnan');

h2 = heatmap(u_c, u_d, grand_mean_hm);
h2.Title = 'Grand Mean JS Divergence (MAP vs Marginalised)';
h2.XLabel = 'Contrast';
h2.YLabel = 'Dispersion';
h2.Colormap = parula; 
h2.MissingDataColor = [0.9 0.9 0.9];

% =========================================================================
%% PCA-based Posterior Divergence (Hierarchical)
% =========================================================================
fprintf('Generating PCA-based divergence plots...\n');

% Pre-allocate arrays to hold the per-animal means for PCA divergence
% Dimensions: [Animal x Orientation]
animal_pca_ori = nan(n_animals, length(u_ori));
% Dimensions: [Dispersion x Contrast x Animal]
animal_pca_hm  = nan(length(u_d), length(u_c), n_animals);

for a = 1:n_animals
    % Extract current animal's data
    t_data = IOResults.animals{a}.trial_table;
    P_map_a = IOResults.animals{a}.inferred.post_s_given_map;
    Q_marg_a = IOResults.animals{a}.inferred.post_s_marginal;
    
    % 1. Run PCA on the marginalised posteriors
    % coeff: principal components, latent: eigenvalues (variance explained), mu: mean
    [coeff, ~, latent, ~, ~, mu] = pca(Q_marg_a);
    
    % 2. Project both posteriors into the PC space
    % Center the data using the mean of the marginalised posteriors
    Q_proj = (Q_marg_a - mu) * coeff;
    P_proj = (P_map_a - mu) * coeff;
    
    % 3. Calculate Squared Error per trial per PC
    sq_diff = (Q_proj - P_proj).^2;
    
    % 4. Compute weighted average of SSEs using eigenvalues
    % Normalise the eigenvalues so they sum to 1 (true weighted average)
    weights = latent / sum(latent);
    pca_div = sq_diff * weights; % Matrix mult: [N x PCs] * [PCs x 1] -> [N x 1]
    
    % Store in table for easy logical indexing
    t_data.pca_div = pca_div;
    
    % 5. Aggregate by Orientation
    for i = 1:length(u_ori)
        idx = t_data.orientation == u_ori(i);
        animal_pca_ori(a, i) = mean(t_data.pca_div(idx), 'omitnan');
    end
    
    % 6. Aggregate by Contrast & Dispersion
    for i = 1:length(u_c)
        for j = 1:length(u_d)
            idx = (t_data.contrast == u_c(i)) & (t_data.dispersion == u_d(j));
            animal_pca_hm(j, i, a) = mean(t_data.pca_div(idx), 'omitnan');
        end
    end
end

% --- Plotting ---
figure('Name', 'PCA Divergence (Hierarchical)', 'Color', 'w', 'Position', [200, 250, 1000, 450]);

% -- Subplot A: Animal Trajectories + Grand Mean (Orientation) --
subplot(1, 2, 1);
hold on;

% Plot individual animal trajectories (light grey)
plot(u_ori, animal_pca_ori', '-', 'Color', [0.6 0.6 0.6 0.5], 'LineWidth', 1.5, 'HandleVisibility', 'off');

% Calculate and plot Grand Mean + Standard Error of the Mean (SEM)
grand_mean_pca_ori = mean(animal_pca_ori, 1, 'omitnan');
grand_sem_pca_ori  = std(animal_pca_ori, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(animal_pca_ori), 1));

errorbar(u_ori, grand_mean_pca_ori, grand_sem_pca_ori, 'k-o', 'LineWidth', 2.5, ...
    'MarkerFaceColor', 'w', 'MarkerSize', 6, 'DisplayName', 'Grand Mean ± SEM');

xlabel('Stimulus Orientation (deg)', 'FontWeight', 'bold');
ylabel('PCA-Weighted SSE Divergence', 'FontWeight', 'bold');
title('PCA Divergence vs. Orientation');
legend('Location', 'best', 'Box', 'off');
box off;
set(gca, 'TickDir', 'out', 'FontSize', 12);

% -- Subplot B: Grand Mean Heatmap (Contrast x Dispersion) --
subplot(1, 2, 2);

% Average the 2D heatmaps across the 3rd dimension (animals)
grand_mean_pca_hm = mean(animal_pca_hm, 3, 'omitnan');

h3 = heatmap(u_c, u_d, grand_mean_pca_hm);
h3.Title = 'Grand Mean PCA Divergence (MAP vs Marginalised)';
h3.XLabel = 'Contrast';
h3.YLabel = 'Dispersion';
h3.Colormap = parula; 
h3.MissingDataColor = [0.9 0.9 0.9];