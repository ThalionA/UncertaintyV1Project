%% --- Section 1: Aligned Population Tuning Curves (Robust to 0-90 Limits) ---
% Requires: NeuralStore
% Methodology:
% 1. Fit tuning curves strictly within 0-90 deg (NaN outside range).
% 2. Align to preferred orientation.
% 3. Average using 'omitnan', so neurons only contribute where they have data.
% 4. Plot the Sample Size (N) to show reliability of the tails.

fprintf('\n=== Aligned Population Tuning Curve Analysis (Robust) ===\n');

% --- Parameters ---
target_contrasts = [0.01, 0.25, 0.50, 1.00]; 
target_dispersions = [5, 30, 45, 90]; 

% Colors
colors_con = parula(length(target_contrasts) + 2); colors_con = colors_con(2:end-1,:);
colors_disp = hot(length(target_dispersions) + 2); colors_disp = colors_disp(1:end-2,:);

% Storage
group_tuning_by_con = cell(length(target_contrasts), 1);
group_tuning_by_disp = cell(length(target_dispersions), 1);

% -------------------------------------------------------------------------
% 1. PROCESS DATA
% -------------------------------------------------------------------------
for i_ani = 1:length(NeuralStore)
    Gspk = NeuralStore(i_ani).Gspk; 
    X = squeeze(mean(Gspk, 3, 'omitnan')); 
    stim = NeuralStore(i_ani).stimulus;
    con = NeuralStore(i_ani).contrast;
    disp = NeuralStore(i_ani).dispersion;
    
    % --- Determine Preference (High Fidelity) ---
    con_rounded = round(con, 2);
    mask_pref = (con_rounded >= 0.9) & (disp <= 5);
    if sum(mask_pref) < 20, mask_pref = (con_rounded >= 0.5); end
    
    % Get raw curves (0-90 only)
    tc_raw_align = get_tuning_curves_strict(X(mask_pref,:), stim(mask_pref)); 
    [~, max_idx] = max(tc_raw_align, [], 2);
    pref_ori_deg = max_idx - 1; 
    
    valid_neurons = max(tc_raw_align, [], 2) > 0.001;
    
    % --- Aggregate by CONTRAST ---
    for c = 1:length(target_contrasts)
        mask_c = (abs(con - target_contrasts(c)) < 0.1); 
        if sum(mask_c) > 5
            tc_raw = get_tuning_curves_strict(X(mask_c,:), stim(mask_c));
            aligned_stack = align_curves_strict(tc_raw(valid_neurons,:), pref_ori_deg(valid_neurons));
            group_tuning_by_con{c} = [group_tuning_by_con{c}; aligned_stack];
        end
    end
    
    % --- Aggregate by DISPERSION ---
    for d = 1:length(target_dispersions)
        mask_d = (abs(disp - target_dispersions(d)) < 5); 
        if sum(mask_d) > 5
            tc_raw = get_tuning_curves_strict(X(mask_d,:), stim(mask_d));
            aligned_stack = align_curves_strict(tc_raw(valid_neurons,:), pref_ori_deg(valid_neurons));
            group_tuning_by_disp{d} = [group_tuning_by_disp{d}; aligned_stack];
        end
    end
end

% -------------------------------------------------------------------------
% 2. PLOTTING
% -------------------------------------------------------------------------
figure('Name', 'Group Aligned Tuning (0-90 Limited)', 'Color', 'w', 'Position', [100 100 1200 800]);
tlg = tiledlayout(2, 2, 'Padding', 'compact');

% --- Plot A: Contrast Modulation ---
nexttile(tlg); hold on;
for c = 1:length(target_contrasts)
    data = group_tuning_by_con{c};
    if ~isempty(data)
        mu = mean(data, 1, 'omitnan');
        % SEM calculation must account for varying N per bin!
        n_per_bin = sum(~isnan(data), 1);
        sem = std(data, 0, 1, 'omitnan') ./ sqrt(n_per_bin);
        
        shadedErrorBar(-90:90, mu, sem, 'lineprops', {'Color', colors_con(c,:), 'LineWidth', 2});
    end
end
title('A. Contrast Modulation');
xlabel('\Delta Orientation (deg)'); ylabel('Norm. Activity');
xline(0, ':k', 'HandleVisibility', 'off'); xlim([-90 90]);

% Legend Fix
h_leg_con = gobjects(length(target_contrasts), 1);
for c=1:length(target_contrasts)
    h_leg_con(c) = plot(nan,nan,'Color',colors_con(c,:),'LineWidth',2,'DisplayName',sprintf('%.0f%%',target_contrasts(c)*100)); 
end
legend(h_leg_con, 'Location', 'best');

% --- Plot B: Sample Size for Contrast ---
nexttile(tlg); hold on;
for c = 1:length(target_contrasts)
    data = group_tuning_by_con{c};
    if ~isempty(data)
        n_per_bin = sum(~isnan(data), 1);
        plot(-90:90, n_per_bin, 'Color', colors_con(c,:), 'LineWidth', 1.5);
    end
end
title('B. Neurons Contributing per Bin');
xlabel('\Delta Orientation'); ylabel('Count (N)');
xline(0, ':k', 'HandleVisibility', 'off'); xlim([-90 90]); grid on;

% --- Plot C: Dispersion Modulation ---
nexttile(tlg); hold on;
for d = 1:length(target_dispersions)
    data = group_tuning_by_disp{d};
    if ~isempty(data)
        mu = mean(data, 1, 'omitnan');
        n_per_bin = sum(~isnan(data), 1);
        sem = std(data, 0, 1, 'omitnan') ./ sqrt(n_per_bin);
        
        shadedErrorBar(-90:90, mu, sem, 'lineprops', {'Color', colors_disp(d,:), 'LineWidth', 2});
    end
end
title('C. Dispersion Modulation');
xlabel('\Delta Orientation (deg)'); ylabel('Norm. Activity');
xline(0, ':k', 'HandleVisibility', 'off'); xlim([-90 90]);

% Legend Fix
h_leg_disp = gobjects(length(target_dispersions), 1);
for d=1:length(target_dispersions)
    h_leg_disp(d) = plot(nan,nan,'Color',colors_disp(d,:),'LineWidth',2,'DisplayName',sprintf('%d\circ',target_dispersions(d))); 
end
legend(h_leg_disp, 'Location', 'best');

% --- Plot D: Sample Size for Dispersion ---
nexttile(tlg); hold on;
for d = 1:length(target_dispersions)
    data = group_tuning_by_disp{d};
    if ~isempty(data)
        n_per_bin = sum(~isnan(data), 1);
        plot(-90:90, n_per_bin, 'Color', colors_disp(d,:), 'LineWidth', 1.5);
    end
end
title('D. Neurons Contributing per Bin');
xlabel('\Delta Orientation'); ylabel('Count (N)');
xline(0, ':k', 'HandleVisibility', 'off'); xlim([-90 90]); grid on;


%% --- Helper Functions ---

function tc_norm = get_tuning_curves_strict(X, stim)
    % Interpolates ONLY within 0-90. Returns NaN for extrapolation.
    n_neurons = size(X, 2);
    oris = unique(stim);
    oris = oris(~isnan(oris));
    [oris, ~] = sort(oris);
    
    means = nan(n_neurons, length(oris));
    for i = 1:length(oris)
        means(:, i) = mean(X(stim == oris(i), :), 1, 'omitnan');
    end
    
    tc = nan(n_neurons, 91);
    for n = 1:n_neurons
        if all(isnan(means(n,:))), continue; end
        % Use NaN for extrapolation!
        tc(n, :) = interp1(oris, means(n, :), 0:90, 'linear', NaN);
    end
    
    max_val = max(tc, [], 2);
    tc_norm = tc ./ (max_val + eps); 
end

function aligned = align_curves_strict(tc_matrix, pref_indices)
    % Align tuning curves. Result will have NaNs where data didn't exist.
    [n_neurons, ~] = size(tc_matrix);
    aligned = nan(n_neurons, 181); % -90 to +90
    
    for n = 1:n_neurons
        pref_idx = pref_indices(n);
        if isnan(pref_idx), continue; end
        
        % Shift so pref_idx lands at index 91 (0 deg)
        shift_amount = 91 - pref_idx;
        input_indices = 1:91;
        target_indices = input_indices + shift_amount;
        
        valid_mask = target_indices >= 1 & target_indices <= 181;
        
        % Only copy data, NaNs in source propagate, empty slots remain NaN
        aligned(n, target_indices(valid_mask)) = tc_matrix(n, valid_mask);
    end
end

%% --- Section 2: Population Fano Factor Analysis ---
% Requires: NeuralStore
% Computes FF in 50ms bins for grating, 5cm bins for corridor

fprintf('\n=== Population Fano Factor Analysis ===\n');

% Parameters
time_bin_width = 0.1; % 100ms
pos_bin_width = 5;   % 10cm

% Storage
ff_grating_all = [];
ff_corridor_all = [];

figure('Name', 'Population Fano Factor', 'Color', 'w', 'Position', [100 100 1000 400]);

for i_ani = 1:length(NeuralStore)
    % 1. Grating (Temporal)
    % Gspk is [Trials x Neurons x Time]
    G = NeuralStore(i_ani).Gspk;
    n_bins = size(G, 3);
    
    % Compute Mean and Var across trials for EACH neuron, EACH bin
    % Then regress Var vs Mean (slope = FF) or simple Ratio
    % Churchland method: Var_pop / Mean_pop matching conditions?
    % Simplified here: Mean of (Var/Mean) across neurons, or Slope fit
    
    var_g = squeeze(var(G, 0, 1, 'omitnan')); % [Neurons x Time]
    mean_g = squeeze(mean(G, 1, 'omitnan'));  % [Neurons x Time]
    
    % Filter low firing neurons to avoid division by zero artifacts
    valid_mask = mean_g > 0.01; 
    
    ff_trace = nan(1, n_bins);
    for t = 1:n_bins
        v = var_g(valid_mask(:,t), t);
        m = mean_g(valid_mask(:,t), t);
        if ~isempty(v)
             % Slope of regression through origin is robust FF
             % b = v \ m; 
             
             pf = polyfit(v, m, 1);
                %         fitfano = fit(raw_centroid_magnitudes(isnoise)', raw_generalised_variance(isnoise)', 'poly1', 'Robust', 'LAR');
        % noise_slopes(iNoise) = p(1);
             ff_trace(t) = pf(1);
        end
    end
    ff_grating_all = [ff_grating_all; ff_trace];
    
    % 2. Corridor (Spatial)
    % Cspk is [Trials x Neurons x PosBins]
    C = NeuralStore(i_ani).Cspk;
    if ~isempty(C)
        n_pos_bins = size(C, 3);
        var_c = squeeze(var(C, 0, 1, 'omitnan'));
        mean_c = squeeze(mean(C, 1, 'omitnan'));
        
        valid_mask_c = mean_c >= 0.01;
        ff_trace_c = nan(1, n_pos_bins);
        for p = 1:n_pos_bins
            v = var_c(valid_mask_c(:,p), p);
            m = mean_c(valid_mask_c(:,p), p);
            if ~isempty(v)
                 % b = v \ m; 
                 pf = polyfit(v, m, 1);
                 % ff_trace_c(p) = b;
                 ff_trace_c(p) = pf(1);
            end
        end
        % Pad if needed
        if size(ff_corridor_all, 2) > 0 && length(ff_trace_c) ~= size(ff_corridor_all, 2)
             % Handle resizing if bin counts differ (omitted for brevity)
        else
             ff_corridor_all = [ff_corridor_all; ff_trace_c];
        end
    end
end

% Plot Grating
subplot(1,2,1); hold on;
t_axis = linspace(0, 2, size(ff_grating_all, 2));
shadedErrorBar(t_axis, mean(ff_grating_all,1,'omitnan'), std(ff_grating_all,0,1,'omitnan')./sqrt(size(ff_grating_all,1)), 'lineprops', {'Color', 'k'});
xlabel('Time (s)'); ylabel('Fano Factor'); title('Grating Epoch');
xline(0, '--'); xline(2, '--');

% Plot Corridor
subplot(1,2,2); hold on;
if ~isempty(ff_corridor_all)
    p_axis = linspace(0, 5*n_pos_bins, size(ff_corridor_all, 2)); % Approx
    shadedErrorBar(p_axis, mean(ff_corridor_all,1,'omitnan'), std(ff_corridor_all,0,1,'omitnan')./sqrt(size(ff_corridor_all,1)), 'lineprops', {'Color', 'b'});
    xlabel('Position (cm)'); ylabel('Fano Factor'); title('Corridor Epoch');
end

%% --- Section 5: Reduced 2D Space Visualization ---
% Requires: NeuralStore

fprintf('\n=== 2D PCA Visualization ===\n');

figure('Name', 'PCA by Stimulus Condition', 'Color', 'w', 'Position', [100 100 1200 500]);

for i_ani = 1:min(3, length(NeuralStore)) % Plot first 3 animals
    % Data
    G = NeuralStore(i_ani).Gspk;
    X = squeeze(mean(G, 3, 'omitnan')); % Time-averaged
    
    stim = NeuralStore(i_ani).stimulus;
    con = NeuralStore(i_ani).contrast;
    disp = NeuralStore(i_ani).dispersion;
    
    % PCA
    [coeff, score] = pca(X);
    
    % Plot 1: By Orientation
    subplot(2, 3, i_ani); hold on;
    scatter(score(:,1), score(:,2), 15, stim, 'filled');
    colormap(gca, 'parula'); colorbar;
    title(sprintf('%s - Ori', NeuralStore(i_ani).tag));
    xlabel('PC1'); ylabel('PC2');
    
    % Plot 2: By Contrast/Dispersion (Composite)
    subplot(2, 3, i_ani+3); hold on;
    % Color by Contrast (Red=High), Size by Dispersion (Small=High Disp/Low Certainty)
    colors = repmat([0 0 0], length(con), 1);
    colors(:, 1) = con; % Red channel = contrast
    
    scatter(score(:,1), score(:,2), 15, colors, 'filled');
    title(sprintf('%s - Contrast (Red)', NeuralStore(i_ani).tag));
    xlabel('PC1'); ylabel('PC2');
end

%% --- Section 6: Coding Dimension Projection & d-prime Analysis (Final v2) ---
% Requires: NeuralStore
% Improvements:
% 1. Defines CD on easy trials (LDA) to find the "optimal" axis.
% 2. Projects all trials.
% 3. Visualizes Go vs NoGo separation for ALL animals.
% 4. Averages across nuisance variables (Contrast averages over Disp, etc).
% 5. Uses graded colors fading to GRAY for low certainty.

fprintf('\n=== Coding Dimension Projection & d-prime Analysis ===\n');

% --- Parameters ---
contrasts_u = [0.01, 0.25, 0.50, 1.00]; 
dispersions_u = [5, 30, 45, 90]; % Removed 15 deg

% Storage for group summary
dprime_contrast = nan(length(NeuralStore), length(contrasts_u));
dprime_dispersion = nan(length(NeuralStore), length(dispersions_u));

% Prepare Figures
n_animals = length(NeuralStore);
n_cols = 4; n_rows = ceil(n_animals * 2 / n_cols); 

fig_cd = figure('Name', 'CD Projections (All Animals)', 'Color', 'w', 'Position', [50 50 1600 900]);
tl = tiledlayout(n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');

for i_ani = 1:n_animals
    % --- Data Prep ---
    tag = NeuralStore(i_ani).tag;
    G = NeuralStore(i_ani).Gspk;
    X = squeeze(mean(G, 3, 'omitnan')); 
    
    stim = NeuralStore(i_ani).stimulus;
    con = NeuralStore(i_ani).contrast; 
    con = round(con, 2); con(con > 0.95) = 1.0;
    disp = NeuralStore(i_ani).dispersion;
    
    % --- Define Coding Dimension (CD) on "Easy" Trials ---
    % (High Contrast, Low Dispersion)
    mask_train = (con >= 0.9) & (disp <= 5) & (stim == 0 | stim == 90);
    
    if sum(mask_train & stim==0) < 5 || sum(mask_train & stim==90) < 5
        continue; 
    end
    
    X_train = X(mask_train, :);
    Y_train = stim(mask_train);
    
    % Diagonal LDA Vector
    mu0 = mean(X_train(Y_train == 0, :), 1);
    mu90 = mean(X_train(Y_train == 90, :), 1);
    dMu = (mu90 - mu0)';
    
    var0 = var(X_train(Y_train == 0, :), 0, 1);
    var90 = var(X_train(Y_train == 90, :), 0, 1);
    pooled_var = 0.5 * (var0 + var90) + eps; 
    
    W = dMu ./ pooled_var'; 
    W = W / norm(W); 
    
    % --- Project All Trials ---
    proj = X * W;
    
    % Robust Axis Limits
    p_lims = prctile(proj, [1, 99]);
    range_p = p_lims(2) - p_lims(1);
    x_axis = linspace(p_lims(1) - 0.1*range_p, p_lims(2) + 0.1*range_p, 100);
    
    % Bandwidth for KDE
    bw = std(proj) / 5; if bw == 0, bw = 0.1; end

    % --- A. Contrast Modulation (Avg across Dispersions) ---
    nexttile(tl); hold on;
    title(sprintf('%s: Contrast', tag));
    
    for c_idx = 1:length(contrasts_u)
        c_val = contrasts_u(c_idx);
        % Mask: Specific Contrast, Any Dispersion
        mask_c = (abs(con - c_val) < 0.05);
        
        p0 = proj(mask_c & stim == 0);
        p90 = proj(mask_c & stim == 90);
        
        if length(p0) > 3 && length(p90) > 3
            % Calculate d'
            dprime_contrast(i_ani, c_idx) = abs(mean(p90) - mean(p0)) / sqrt(0.5*(var(p0) + var(p90)));
            
            % Color Grading: Low C -> Gray, High C -> Pure Color
            certainty = c_idx / length(contrasts_u); % 0.25 to 1.0
            col0  = colorGradient([0 0 1], [0.5 0.5 0.5], certainty); % Blue to Gray
            col90 = colorGradient([1 0 0], [0.5 0.5 0.5], certainty); % Red to Gray
            
            [f0, xi] = ksdensity(p0, x_axis, 'Bandwidth', bw);
            [f90, xi] = ksdensity(p90, x_axis, 'Bandwidth', bw);
            
            plot(xi, f0, 'Color', col0, 'LineWidth', 1.5, 'DisplayName', sprintf('0 (C=%.2f)', c_val));
            plot(xi, f90, 'Color', col90, 'LineWidth', 1.5, 'DisplayName', sprintf('90 (C=%.2f)', c_val));
        end
    end
    xlim([x_axis(1) x_axis(end)]);
    
    % --- B. Dispersion Modulation (Avg across Contrasts) ---
    nexttile(tl); hold on;
    title(sprintf('%s: Dispersion', tag));
    
    for d_idx = 1:length(dispersions_u)
        d_val = dispersions_u(d_idx);
        % Mask: Specific Dispersion, Any Contrast (Robust matching)
        mask_d = (abs(disp - d_val) < 2); 
        
        p0 = proj(mask_d & stim == 0);
        p90 = proj(mask_d & stim == 90);
        
        if length(p0) > 1 && length(p90) > 1
            dprime_dispersion(i_ani, d_idx) = abs(mean(p90) - mean(p0)) / sqrt(0.5*(var(p0) + var(p90)));
            
            % Color Grading: Low D (High Cert) -> Pure, High D (Low Cert) -> Gray
            % Index 1 (Disp 5) -> High Certainty
            % Index End (Disp 90) -> Low Certainty
            certainty = 1 - ((d_idx-1) / (length(dispersions_u)-1)); 
            % Ensure it doesn't go completely to 0 if you want some color remaining
            certainty = 0.2 + 0.8*certainty; 
            
            col0  = colorGradient([0 0 1], [0.5 0.5 0.5], certainty); 
            col90 = colorGradient([1 0 0], [0.5 0.5 0.5], certainty);
            
            [f0, xi] = ksdensity(p0, x_axis, 'Bandwidth', bw);
            [f90, xi] = ksdensity(p90, x_axis, 'Bandwidth', bw);
            
            plot(xi, f0, 'Color', col0, 'LineWidth', 1.5);
            plot(xi, f90, 'Color', col90, 'LineWidth', 1.5);
        end
    end
    xlim([x_axis(1) x_axis(end)]);
end

% 2. Summary Figure: d' vs Conditions
figure('Name', 'Group d-prime on CD', 'Color', 'w', 'Position', [100 100 800 400]);
t_sum = tiledlayout(1, 2, 'Padding', 'compact');

% Contrast Summary
nexttile(t_sum); hold on;
mu = mean(dprime_contrast, 1, 'omitnan');
se = std(dprime_contrast, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(dprime_contrast), 1));
errorbar(1:length(contrasts_u), mu, se, '-bo', 'LineWidth', 2, 'MarkerFaceColor', 'b', 'CapSize', 0);
xticks(1:length(contrasts_u)); xticklabels(string(contrasts_u));
ylabel('d'' (Separability)'); xlabel('Contrast');
title('Effect of Contrast (Avg Disp)');
grid on;

% Dispersion Summary
nexttile(t_sum); hold on;
mu_d = mean(dprime_dispersion, 1, 'omitnan');
se_d = std(dprime_dispersion, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(dprime_dispersion), 1));
errorbar(1:length(dispersions_u), mu_d, se_d, '-ro', 'LineWidth', 2, 'MarkerFaceColor', 'r', 'CapSize', 0);
xticks(1:length(dispersions_u)); xticklabels(string(dispersions_u));
xlabel('Dispersion');
title('Effect of Dispersion (Avg Con)');
grid on;

%% --- Section 7: 2D PCA Cloud Projection (PC1 vs PC2) ---
% Requires: NeuralStore
% Methodology:
% 1. PCA on time-averaged population activity (unsupervised).
% 2. Project ALL trials onto PC1 and PC2.
% 3. Visualize "Clouds" (individual trials) and Centroids.
% 4. Shows how the dominant population geometry changes with uncertainty.

fprintf('\n=== 2D PCA Cloud Projection (PC1 vs PC2) ===\n');

% --- Parameters ---
contrasts_u = [0.01, 0.25, 0.50, 1.00]; 
dispersions_u = [5, 30, 45, 90]; 

% Prepare Figure
n_animals = length(NeuralStore);
n_cols = 4; n_rows = ceil(n_animals * 2 / n_cols);

fig_pca = figure('Name', 'PCA Cloud Projections (PC1 vs PC2)', 'Color', 'w', 'Position', [50 50 1600 900]);
tl = tiledlayout(n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');

for i_ani = 1:n_animals
    % --- Data Prep ---
    tag = NeuralStore(i_ani).tag;
    G = NeuralStore(i_ani).Gspk;
    X = squeeze(mean(G, 3, 'omitnan')); % [Trials x Neurons]
    
    % Z-score neurons (standard PCA practice)
    X_z = zscore(X, 0, 1);
    X_z(isnan(X_z)) = 0;
    
    stim = NeuralStore(i_ani).stimulus;
    con = NeuralStore(i_ani).contrast; 
    con = round(con, 2); con(con > 0.95) = 1.0;
    disp = NeuralStore(i_ani).dispersion;
    
    % --- Run PCA ---
    [~, score] = pca(X_z); 
    PC1 = score(:,1);
    PC2 = score(:,2);
    
    % Axis limits (robust)
    p_lims_1 = prctile(PC1, [1, 99]);
    p_lims_2 = prctile(PC2, [1, 99]);
    
    % --- A. Contrast Modulation (Avg across Dispersions) ---
    nexttile(tl); hold on;
    title(sprintf('%s: Contrast (PCA)', tag));
    
    for c_idx = 1:length(contrasts_u)
        c_val = contrasts_u(c_idx);
        % Mask: Specific Contrast, Any Dispersion
        mask_c = (abs(con - c_val) < 0.05); 
        
        idx0 = find(mask_c & stim==0);
        idx90 = find(mask_c & stim==90);
        
        if ~isempty(idx0) && ~isempty(idx90)
            % Colors: High C = Pure, Low C = Gray
            certainty = c_idx / length(contrasts_u);
            col0 = colorGradient([0 0 1], [0.8 0.8 0.8], certainty); 
            col90 = colorGradient([1 0 0], [0.8 0.8 0.8], certainty);
            
            % Plot CLOUDS
            scatter(PC1(idx0), PC2(idx0), 10, col0, 'filled', ...
                'MarkerFaceAlpha', 0.2, 'MarkerEdgeColor', 'none');
            scatter(PC1(idx90), PC2(idx90), 10, col90, 'filled', ...
                'MarkerFaceAlpha', 0.2, 'MarkerEdgeColor', 'none');
            
            % Plot CENTROIDS
            plot(mean(PC1(idx0)), mean(PC2(idx0)), 'o', ...
                'MarkerSize', 6, 'MarkerFaceColor', col0, 'MarkerEdgeColor', 'k');
            plot(mean(PC1(idx90)), mean(PC2(idx90)), 'o', ...
                'MarkerSize', 6, 'MarkerFaceColor', col90, 'MarkerEdgeColor', 'k');
        end
    end
    xlabel('PC1'); ylabel('PC2'); axis square; grid on;
    xlim([p_lims_1(1) p_lims_1(2)]); ylim([p_lims_2(1) p_lims_2(2)]);
    
    % --- B. Dispersion Modulation (Avg across Contrasts) ---
    nexttile(tl); hold on;
    title(sprintf('%s: Dispersion (PCA)', tag));
    
    for d_idx = 1:length(dispersions_u)
        d_val = dispersions_u(d_idx);
        % Mask: Specific Dispersion, Any Contrast
        mask_d = (abs(disp - d_val) < 2); 
        
        idx0 = find(mask_d & stim==0);
        idx90 = find(mask_d & stim==90);
        
        if ~isempty(idx0) && ~isempty(idx90)
            % Colors: Low Disp = Pure, High Disp = Gray
            certainty = 1 - ((d_idx-1)/(length(dispersions_u)-1));
            certainty = 0.2 + 0.8*certainty;
            
            col0 = colorGradient([0 0 1], [0.8 0.8 0.8], certainty); 
            col90 = colorGradient([1 0 0], [0.8 0.8 0.8], certainty);
            
            scatter(PC1(idx0), PC2(idx0), 10, col0, 'filled', ...
                'MarkerFaceAlpha', 0.2, 'MarkerEdgeColor', 'none');
            scatter(PC1(idx90), PC2(idx90), 10, col90, 'filled', ...
                'MarkerFaceAlpha', 0.2, 'MarkerEdgeColor', 'none');
            
            plot(mean(PC1(idx0)), mean(PC2(idx0)), 's', ...
                'MarkerSize', 6, 'MarkerFaceColor', col0, 'MarkerEdgeColor', 'k');
            plot(mean(PC1(idx90)), mean(PC2(idx90)), 's', ...
                'MarkerSize', 6, 'MarkerFaceColor', col90, 'MarkerEdgeColor', 'k');
        end
    end
    xlabel('PC1'); ylabel('PC2'); axis square; grid on;
    xlim([p_lims_1(1) p_lims_1(2)]); ylim([p_lims_2(1) p_lims_2(2)]);
end

%% --- Helper Function ---
function col = colorGradient(c_high, c_low, certainty)
    col = c_low + (c_high - c_low) * certainty;
end