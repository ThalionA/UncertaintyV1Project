%% Plot Mean Pupil Size Grouped by Various Task Metrics
%
% What this script does:
% 1. Loads 'VR_multi_animal_analysis.mat' (must contain TrialTbl_all and PerAnimalRaw).
% 2. Consolidates metrics from TrialTbl_all with a SINGLE scalar value
%    for pupil: the mean(pupil) during the grating epoch.
% 3. Defines bins for continuous metrics (confidence, uncertainty) using quantiles.
% 4. Generates ONE figure with a 3x3 tiled layout, plotting all 8
%    grouping conditions as separate panels.



% --- 2. Consolidate Scalar Pupil & Metrics ---
% This creates a single struct array 'P' where each element is a trial
% containing its scalar metrics AND the mean grating pupil size.

fprintf('Consolidating pupil data and metrics...\n');
P = []; % This will be a struct array

for a = 1:numel(PerAnimalRaw)
    UD = PerAnimalRaw(a).UD;
    tag = PerAnimalRaw(a).tag;
    n_trials_animal = numel(UD);
    
    % Find the corresponding rows in TrialTbl_all
    table_rows_for_animal = find(strcmp(TrialTbl_all.animal, tag));
    
    if numel(table_rows_for_animal) ~= n_trials_animal
        warning('Trial count mismatch for %s (Table: %d, Raw: %d). Skipping.', ...
            tag, numel(table_rows_for_animal), n_trials_animal);
        continue;
    end
    
    T_animal = TrialTbl_all(table_rows_for_animal, :);
    
    for tr = 1:n_trials_animal
        s = struct();
        
        % --- Calculate SCALAR mean pupil size ---
        s.pupil_grating_mean = NaN;
        if isfield(UD(tr),'pupil') && isfield(UD(tr).pupil,'grating') && ~isempty(UD(tr).pupil.grating)
            s.pupil_grating_mean = mean(UD(tr).pupil.grating(:), 'omitnan');
        end
        
        % Grouping Variables from TrialTbl_all
        s.abs_from_go   = T_animal.abs_from_go(tr);
        s.contrast      = T_animal.contrast(tr);
        s.dispersion    = T_animal.dispersion(tr);
        s.goChoice      = T_animal.goChoice(tr);
        s.outcome       = T_animal.outcome(tr);
        s.confidence    = abs(T_animal.confidence(tr));
        s.unc_perceptual= T_animal.unc_perceptual(tr);
        s.unc_decision  = T_animal.unc_decision(tr);
        
        P = [P; s];
    end
end
fprintf('Consolidated %d trials with scalar pupil data.\n', numel(P));


% --- 3. Define Bins for Continuous Variables ---
n_quantiles = 5; % Quartiles
quantile_labels = arrayfun(@(x) sprintf('Q%d', x), 1:n_quantiles, 'Uni', 0);

% Get all data for binning
all_conf = [P.confidence];
all_unc_p = [P.unc_perceptual];
all_unc_d = [P.unc_decision];

% Calculate quantile edges
conf_bins = prctile(all_conf(~isnan(all_conf)), linspace(0, 100, n_quantiles + 1));
unc_p_bins = prctile(all_unc_p(~isnan(all_unc_p)), linspace(0, 100, n_quantiles + 1));
unc_d_bins = prctile(all_unc_d(~isnan(all_unc_d)), linspace(0, 100, n_quantiles + 1));

% Ensure unique edges
conf_bins = unique(conf_bins); unc_p_bins = unique(unc_p_bins); unc_d_bins = unique(unc_d_bins);
conf_bins(end) = conf_bins(end) + eps;
unc_p_bins(end) = unc_p_bins(end) + eps;
unc_d_bins(end) = unc_d_bins(end) + eps;

n_conf_bins = numel(conf_bins) - 1;
n_unc_p_bins = numel(unc_p_bins) - 1;
n_unc_d_bins = numel(unc_d_bins) - 1;

% Discretize continuous variables and add to struct
[~, bin_conf] = histc(all_conf, conf_bins);
[~, bin_unc_p] = histc(all_unc_p, unc_p_bins);
[~, bin_unc_d] = histc(all_unc_d, unc_d_bins);

for i = 1:numel(P)
    P(i).bin_conf = bin_conf(i);
    P(i).bin_unc_p = bin_unc_p(i);
    P(i).bin_unc_d = bin_unc_d(i);
end

% Get bins for categorical variables
unique_ori = unique([P.abs_from_go]); unique_ori = unique_ori(~isnan(unique_ori));
unique_con = unique([P.contrast]); unique_con = unique_con(~isnan(unique_con));
unique_disp = unique([P.dispersion]); unique_disp = unique_disp(~isnan(unique_disp));


% --- 4. Generate All 8 Plot Groups in One Figure ---

fprintf('Generating plots...\n');
figure('Name', 'Pupil Analysis (Mean Grating)', 'Color', 'w', 'Position', [50 50 1400 1000]);
tl = tiledlayout(3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% 1. Grouped by Orientation (|abs_from_go|)
nexttile(tl);
plot_pupil_scalar_grouped(P, 'abs_from_go', unique_ori, ...
    arrayfun(@(x) sprintf('%.0f', x), unique_ori, 'Uni', 0), ...
    'Pupil by Orientation (|Δ from Go|)');

% 2. Grouped by Contrast
nexttile(tl);
plot_pupil_scalar_grouped(P, 'contrast', unique_con, ...
    arrayfun(@(x) sprintf('%.0f%%', x*100), unique_con, 'Uni', 0), ...
    'Pupil by Contrast');

% 3. Grouped by Dispersion
nexttile(tl);
plot_pupil_scalar_grouped(P, 'dispersion', unique_disp, ...
    arrayfun(@(x) sprintf('%.0f', x), unique_disp, 'Uni', 0), ...
    'Pupil by Dispersion');

% 4. Grouped by Go/No-Go Choice
nexttile(tl);
plot_pupil_scalar_grouped(P, 'goChoice', [0, 1], {'No-Go (0)', 'Go (1)'}, ...
    'Pupil by Choice');

% 5. Grouped by Outcome
nexttile(tl);
plot_pupil_scalar_grouped(P, 'outcome', [1, 2, 3, 4], {'Hit', 'Miss', 'FA', 'CR'}, ...
    'Pupil by Outcome');

% 6. Grouped by Confidence Quantiles
nexttile(tl);
plot_pupil_scalar_grouped(P, 'bin_conf', 1:n_conf_bins, ...
    quantile_labels(1:n_conf_bins), ...
    'Pupil by Confidence');

% 7. Grouped by Perceptual Uncertainty Quantiles
nexttile(tl);
plot_pupil_scalar_grouped(P, 'bin_unc_p', 1:n_unc_p_bins, ...
    quantile_labels(1:n_unc_p_bins), ...
    'Pupil by Perceptual Uncertainty');

% 8. Grouped by Decision Uncertainty Quantiles
nexttile(tl);
plot_pupil_scalar_grouped(P, 'bin_unc_d', 1:n_unc_d_bins, ...
    quantile_labels(1:n_unc_d_bins), ...
    'Pupil by Decision Uncertainty');

fprintf('--- Pupil analysis plots complete. ---\n');


%% --- Helper Function ---

function plot_pupil_scalar_grouped(P, group_field, group_bins, group_labels, panel_title)
% Plots mean ± SEM of the scalar 'pupil_grating_mean' for each group.
% This function assumes it is being called after `nexttile` has been called.

    hold on; grid on; box off;
    colors = lines(numel(group_bins));
    
    % --- Determine X-axis type ---
    is_continuous_x = isnumeric(group_bins) && ...
                      any(strcmp(group_field, {'abs_from_go', 'contrast', 'dispersion'}));
                  
    if is_continuous_x
        x_axis = group_bins;
    else
        x_axis = 1:numel(group_bins);
    end

    % --- Calculate Stats ---
    y_means = nan(numel(group_bins), 1);
    y_sems = nan(numel(group_bins), 1);
    
    for g = 1:numel(group_bins)
        % Collect all pupil values for this group
        vals = [];
        for k = 1:numel(P)
            % Use isequaln to handle NaNs in group_field
            if isequaln(P(k).(group_field), group_bins(g))
                vals = [vals; P(k).pupil_grating_mean];
            end
        end
        
        if ~isempty(vals)
            y_means(g) = mean(vals, 'omitnan');
            y_sems(g) = std(vals, 0, 'omitnan') / sqrt(sum(~isnan(vals)));
        end
    end
    
    valid_pts = ~isnan(y_means);
    
    
    % --- Plot ---
    if is_continuous_x
        % Plot as a line plot
        h = errorbar(x_axis(valid_pts), y_means(valid_pts), y_sems(valid_pts), ...
            'o-', 'LineWidth', 1.5, 'MarkerFaceColor', 'w', 'CapSize', 0);
        h.Color = [0.2 0.2 0.2];
        xticks(x_axis);
        xticklabels(group_labels);
        % xlim([min(x_axis)-1, max(x_axis)+1]);
    else
        % Plot as categorical points
        errorbar(x_axis(valid_pts), y_means(valid_pts), y_sems(valid_pts), ...
            'k', 'LineWidth', 1.5, 'LineStyle', 'none', 'CapSize', 0);
        
        % Plot colored markers on top
        for g = find(valid_pts)'
            plot(x_axis(g), y_means(g), 'o', 'MarkerSize', 8, ...
                'MarkerFaceColor', colors(g,:), 'MarkerEdgeColor', 'k');
        end
        xticks(x_axis);
        xticklabels(group_labels);
        xlim([0.5, numel(group_bins) + 0.5]);
    end
    
    title(panel_title);
    ylabel('Mean Pupil (a.u.)');
    xtickangle(30);
end