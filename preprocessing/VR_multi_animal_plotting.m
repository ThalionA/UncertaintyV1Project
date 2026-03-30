%% --- Autocorrelation Analysis & Subpopulation Correlograms (Dual Signal) ---
fprintf('\n=== Running Neural Autocorrelation Analysis (Spikes & dF/F) ===\n');

% 1. Configuration Options
calc_time_window = [1.0, 2.0]; 
grouping_method = 'per_stimulus'; 
grouping_vars = {'stimulus', 'contrast', 'dispersion'}; 
min_trials = 3;
max_lag_s = 1.0; 
signals_to_process = {'Gspk', 'GdF'}; 

% 2. Setup Data Structures
num_animals = numel(NeuralStore);
num_signals = numel(signals_to_process);

% Store ESS: [Animals x Signals] (Animal Means)
ESS_stat = nan(num_animals, num_signals);
ESS_nonstat = nan(num_animals, num_signals);

% Store ESS: All Neurons Pooled for Histograms
all_ESS_stat = cell(1, num_signals);
all_ESS_nonstat = cell(1, num_signals);

n_timebins_window = 0; 
feat_names = {'stimulus', 'contrast', 'dispersion', 'unc_perceptual'};
feat_traces = struct(); 
for s = 1:num_signals
    sig = signals_to_process{s};
    feat_traces.(sig) = struct();
    for f = 1:numel(feat_names)
        feat_traces.(sig).(feat_names{f}) = struct();
    end
end

% 3. Processing Loop across Animals
for a = 1:num_animals
    tag = NeuralStore(a).tag;
    fprintf('  Processing %s...\n', tag);
    
    tbl_idx = strcmp(TrialTbl_all.animal, tag);
    animal_tbl = TrialTbl_all(tbl_idx, :);
    
    % --- Pre-processing: Fix 0.99 contrast to 1.0 ---
    if isfield(NeuralStore(a), 'contrast')
        fix_mask_ns = abs(NeuralStore(a).contrast - 0.99) < 1e-4;
        NeuralStore(a).contrast(fix_mask_ns) = 1.0;
    end
    if ismember('contrast', animal_tbl.Properties.VariableNames)
        fix_mask_tbl = abs(animal_tbl.contrast - 0.99) < 1e-4;
        animal_tbl.contrast(fix_mask_tbl) = 1.0;
    end
    
    xG = NeuralStore(a).xG;
    time_mask = (xG >= calc_time_window(1) & xG <= calc_time_window(2));
    dt = median(diff(xG(time_mask)));
    max_lag_bins = min(round(max_lag_s / dt), sum(time_mask) - 1);
    n_timebins_window = sum(time_mask);
    
    if n_timebins_window < 2
        warning('Time window contains too few bins for %s. Skipping.', tag);
        continue;
    end
    
    % ---------------------------------------------------------
    % Determine BASE Groups for Z-Scoring
    % ---------------------------------------------------------
    nTrials = height(animal_tbl);
    if strcmp(grouping_method, 'all_trials')
        base_groups = ones(nTrials, 1);
        n_base_groups = 1;
    else
        cond_data = [];
        for v = 1:numel(grouping_vars)
            if isfield(NeuralStore(a), grouping_vars{v})
                cond_data = [cond_data, round(NeuralStore(a).(grouping_vars{v}), 3)]; %#ok<AGROW>
            end
        end
        [~, ~, base_groups] = unique(cond_data, 'rows');
        n_base_groups = max(base_groups);
    end
    
    % Loop over the two signal types (Spikes, dF/F)
    for s_idx = 1:num_signals
        sig = signals_to_process{s_idx};
        if ~isfield(NeuralStore(a), sig) || isempty(NeuralStore(a).(sig))
            warning('Signal %s missing for %s.', sig, tag);
            continue;
        end
        
        % [nTimeBins x nTrials x nNeurons]
        raw_tensor = NeuralStore(a).(sig);
        phi_all = permute(raw_tensor(:, :, time_mask), [3, 1, 2]);
        [nTime, ~, nNeurons] = size(phi_all);
        
        Z_phi = nan(size(phi_all));
        phi_centered = nan(size(phi_all));
        valid_trial_mask = false(nTrials, 1);
        
        % Standardize
        for g = 1:n_base_groups
            t_mask = (base_groups == g);
            if sum(t_mask) < min_trials; continue; end
            
            valid_trial_mask(t_mask) = true;
            phi_g = phi_all(:, t_mask, :);
            
            mu_nst = mean(phi_g, 2, 'omitnan');
            sigma_nst = std(phi_g, 0, 2, 'omitnan');
            sigma_nst(sigma_nst == 0) = 1e-6; % Explicit NaN propagation for zero variance
            
            Z_phi(:, t_mask, :) = (phi_g - mu_nst) ./ sigma_nst;
            
            mu_ns = mean(phi_g, [1, 2], 'omitnan');
            phi_centered(:, t_mask, :) = phi_g - mu_ns;
        end
        
        % ---------------------------------------------------------
        % Vectorized ACF & ESS Calculation
        % ---------------------------------------------------------
        num_stat = phi_centered(:, valid_trial_mask, :) .* phi_centered(:, valid_trial_mask, :); 
        sigma2_ns = squeeze(mean(num_stat, [1, 2], 'omitnan'))'; % [1 x nNeurons]
        sigma2_ns(sigma2_ns == 0) = 1e-6; 
        
        rho_s = nan(max_lag_bins+1, nNeurons);
        rho_ns = nan(max_lag_bins+1, nNeurons);
        
        for tau = 0:max_lag_bins
            t_idx = 1:(nTime - tau);
            t_tau_idx = (1 + tau):nTime;
            
            % Stat
            prod_s = phi_centered(t_idx, valid_trial_mask, :) .* phi_centered(t_tau_idx, valid_trial_mask, :);
            rho_s(tau+1, :) = squeeze(mean(prod_s, [1, 2], 'omitnan'))' ./ sigma2_ns;
            
            % Non-Stat
            prod_ns = Z_phi(t_idx, valid_trial_mask, :) .* Z_phi(t_tau_idx, valid_trial_mask, :);
            rho_ns(tau+1, :) = squeeze(mean(prod_ns, [1, 2], 'omitnan'))';
        end
        
        temp_ess_s = nan(nNeurons, 1);
        temp_ess_ns = nan(nNeurons, 1);
        
        noise_floor = 0.05; 
        for n = 1:nNeurons
            % Stationary ESS
            r_s = rho_s(:, n);
            if ~any(isnan(r_s))
                % Truncate when it crosses 0 OR hits the noise floor
                idx = find(r_s <= noise_floor, 1, 'first'); 
                if isempty(idx)
                    idx = max_lag_bins + 1; 
                end
                temp_ess_s(n) = n_timebins_window / (1 + 2 * sum(r_s(2:max(1, idx-1))));
            end
            
            % Non-Stationary ESS
            r_ns = rho_ns(:, n);
            if ~any(isnan(r_ns))
                idx = find(r_ns <= noise_floor, 1, 'first');
                if isempty(idx)
                    idx = max_lag_bins + 1; 
                end
                temp_ess_ns(n) = n_timebins_window / (1 + 2 * sum(r_ns(2:max(1, idx-1))));
            end
        end
        
        ESS_stat(a, s_idx) = mean(temp_ess_s, 'omitnan');
        ESS_nonstat(a, s_idx) = mean(temp_ess_ns, 'omitnan');
        
        % Pool for Histogram
        all_ESS_stat{s_idx} = [all_ESS_stat{s_idx}; temp_ess_s];
        all_ESS_nonstat{s_idx} = [all_ESS_nonstat{s_idx}; temp_ess_ns];
        
        % ---------------------------------------------------------
        % Extract Feature Correlograms
        % ---------------------------------------------------------
        for f = 1:numel(feat_names)
            feat = feat_names{f};
            
            if isfield(NeuralStore(a), feat)
                feat_vector = NeuralStore(a).(feat);
            elseif ismember(feat, animal_tbl.Properties.VariableNames)
                feat_vector = animal_tbl.(feat);
            else
                continue; 
            end
            
            valid_f_mask = valid_trial_mask & ~isnan(feat_vector);
            if sum(valid_f_mask) == 0; continue; end
            
            is_continuous = startsWith(feat, 'unc_');
            if is_continuous
                edges = unique(prctile(feat_vector(valid_f_mask), [0 33.3 66.6 100]));
                if numel(edges) < 2; continue; end 
                edges(1) = edges(1) - eps; edges(end) = edges(end) + eps;
                f_groups = discretize(feat_vector, edges);
                group_labels = arrayfun(@(x) sprintf('Q%d', x), 1:(numel(edges)-1), 'UniformOutput', false);
            else
                f_groups = round(feat_vector, 2);
                group_labels = num2cell(unique(f_groups(valid_f_mask)));
            end
            
            u_groups = unique(f_groups(valid_f_mask));
            
            for g_idx = 1:numel(u_groups)
                g_val = u_groups(g_idx);
                g_mask = (f_groups == g_val) & valid_f_mask;
                
                if sum(g_mask) < 2; continue; end
                
                rho_group = nan(nNeurons, max_lag_bins+1);
                for tau = 0:max_lag_bins
                    t_idx = 1:(nTime - tau);
                    t_tau_idx = (1 + tau):nTime;
                    
                    prod_ns = Z_phi(t_idx, g_mask, :) .* Z_phi(t_tau_idx, g_mask, :);
                    rho_group(:, tau+1) = squeeze(mean(prod_ns, [1, 2], 'omitnan'));
                end
                
                mean_rho_group = mean(rho_group, 1, 'omitnan');
                
                if is_continuous
                    fld = group_labels{g_idx};
                else
                    fld = sprintf('val_%g', g_val);
                    fld = strrep(strrep(fld, '.', '_'), '-', 'neg');
                end
                
                if ~isfield(feat_traces.(sig).(feat), fld)
                    feat_traces.(sig).(feat).(fld) = [];
                end
                feat_traces.(sig).(feat).(fld) = [feat_traces.(sig).(feat).(fld); mean_rho_group];
            end
        end
    end
end

% 4. Plotting
fprintf('Plotting Results...\n');
time_axis = (0:max_lag_bins) * dt;

% --- Figure 1 & 2: Correlograms Split by Features (Spikes & dF/F) ---
for s_idx = 1:num_signals
    sig = signals_to_process{s_idx};
    fig_name = sprintf('Non-Stationary ACF by Features (%s)', sig);
    figure('Name', fig_name, 'Color', 'w', 'Position', [100+(s_idx*50), 100+(s_idx*50), 1400, 600]);
    
    valid_feats = {};
    for f = 1:numel(feat_names)
        if ~isempty(fieldnames(feat_traces.(sig).(feat_names{f})))
            valid_feats{end+1} = feat_names{f}; %#ok<AGROW>
        end
    end
    n_subplots = numel(valid_feats);
    plot_idx = 1;
    
    for f = 1:n_subplots
        feat = valid_feats{f};
        subplot(2, ceil(n_subplots/2), plot_idx); hold on;
        
        flds = fieldnames(feat_traces.(sig).(feat));
        colors = parula(numel(flds));
        
        for i = 1:numel(flds)
            traces = feat_traces.(sig).(feat).(flds{i});
            if isempty(traces); continue; end
            
            mean_trace = mean(traces, 1, 'omitnan');
            sem_trace = std(traces, 0, 1, 'omitnan') ./ sqrt(size(traces,1));
            
            lbl = strrep(strrep(flds{i}, 'val_', ''), '_', '.');
            errorbar(time_axis, mean_trace, sem_trace, 'Color', colors(i,:), ...
                'LineWidth', 2, 'DisplayName', lbl, 'CapSize', 0);
        end
        
        yline(0, 'k--');
        xlabel('Time Lag \tau (s)'); ylabel('Autocorrelation \rho');
        title(strrep(feat, '_', ' '));
        legend('Location', 'best'); box off;
        plot_idx = plot_idx + 1;
    end
    sgtitle(sprintf('%s Non-Stationary Autocorrelograms', sig));
end

% --- Figure 3: Overall Effective Sample Size (ESS) Bar Plot ---
figure('Name', 'ESS Across Signals', 'Color', 'w', 'Position', [200, 200, 600, 450]);
grand_mean = [mean(ESS_stat, 1, 'omitnan'); mean(ESS_nonstat, 1, 'omitnan')];
valid_N = [sum(~isnan(ESS_stat), 1); sum(~isnan(ESS_nonstat), 1)];
sem = [std(ESS_stat, 0, 1, 'omitnan'); std(ESS_nonstat, 0, 1, 'omitnan')] ./ sqrt(valid_N);
b = bar(grand_mean, 'grouped');
b(1).FaceColor = [0 0.4470 0.7410]; % Gspk
b(2).FaceColor = [0.8500 0.3250 0.0980]; % GdF
hold on;
% Add error bars and scatter points
[ngroups, nbars] = size(grand_mean);
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = b(i).XEndPoints;
    errorbar(x(i,:), grand_mean(:,i), sem(:,i), 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 5);
    
    % Overlay individual animals
    for j = 1:ngroups
        if j == 1
            pts = ESS_stat(:, i);
        else
            pts = ESS_nonstat(:, i);
        end
        scatter(x(i,j)*ones(num_animals,1) + (rand(num_animals,1)-0.5)*0.1, pts, ...
            30, 'k', 'filled', 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.6);
    end
end
xticks([1, 2]); xticklabels({'Stationary', 'Non-Stationary'});
ylabel('Effective Sample Size (Frames)');
yline(n_timebins_window, 'k--', 'Theoretical Max (N)', 'LabelHorizontalAlignment', 'left');
legend(signals_to_process, 'Location', 'northeast');
title('Cross-Animal Mean ESS: Spikes vs dF/F');
set(gca, 'FontSize', 12); box off; 

% --- Figure 4: ESS Distributions (Histograms) ---
figure('Name', 'ESS Distributions (All Neurons)', 'Color', 'w', 'Position', [300, 300, 1000, 450]);
colors = struct('Stat', [0 0.4470 0.7410], 'NonStat', [0.8500 0.3250 0.0980]);

for s_idx = 1:num_signals
    sig = signals_to_process{s_idx};
    subplot(1, 2, s_idx); hold on;
    
    % Extract non-NaN values for current signal
    curr_stat = all_ESS_stat{s_idx}(~isnan(all_ESS_stat{s_idx}));
    curr_nonstat = all_ESS_nonstat{s_idx}(~isnan(all_ESS_nonstat{s_idx}));
    
    % Plot distributions
    histogram(curr_stat, 'BinWidth', 1, 'FaceColor', colors.Stat, 'FaceAlpha', 0.6, 'DisplayName', 'Stationary');
    histogram(curr_nonstat, 'BinWidth', 1, 'FaceColor', colors.NonStat, 'FaceAlpha', 0.6, 'DisplayName', 'Non-Stationary');
    
    % Add Mean lines for reference
    xline(mean(curr_stat), '--', sprintf('Mean: %.1f', mean(curr_stat)), ...
        'Color', colors.Stat, 'LineWidth', 2, 'LabelOrientation', 'horizontal', ...
        'LabelVerticalAlignment', 'top', 'HandleVisibility', 'off');
    xline(mean(curr_nonstat), '--', sprintf('Mean: %.1f', mean(curr_nonstat)), ...
        'Color', colors.NonStat, 'LineWidth', 2, 'LabelOrientation', 'horizontal', ...
        'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
        
    % Theoretical Max Reference
    xline(n_timebins_window, 'k-', 'Max', 'LineWidth', 1.5, ...
        'LabelVerticalAlignment', 'middle', 'HandleVisibility', 'off');
    
    xlabel('Effective Sample Size (Frames)'); 
    ylabel('Count (Neurons)');
    title(sprintf('%s ESS Distribution', sig));
    legend('Location', 'northeast'); 
    box off;
    
    % Ensure axes make sense for visual comparison
    xlim([0 n_timebins_window + 2]); 
end

fprintf('\n--- ESS Results (Across Animals) ---\n');
fprintf('Theoretical Maximum ESS: %d frames\n', n_timebins_window);
for s_idx = 1:num_signals
    sig = signals_to_process{s_idx};
    fprintf('>> Signal: %s\n', sig);
    fprintf('   Stat ESS:     %.2f ± %.3f (N=%d)\n', grand_mean(1, s_idx), sem(1, s_idx), valid_N(1, s_idx));
    fprintf('   Non-Stat ESS: %.2f ± %.3f (N=%d)\n', grand_mean(2, s_idx), sem(2, s_idx), valid_N(2, s_idx));
end
fprintf('\n');

%% --- Autocorrelation Analysis (Strict PDF Implementation - Dual Signal) ---
fprintf('\n=== Running Neural Autocorrelation Analysis ===\n');

% 1. Configuration Options
calc_time_window = [1.0, 2.0]; 
grouping_vars = {'stimulus'}; % "s: stimulus value" (Eq 27)
min_trials = 4; % Minimum trials required for autocorrelation (per PDF)
max_lag_s = 1.0; 
signals = {'Gspk', 'GdF'}; % Process both spikes and dF/F

% Pre-extract general animal parameters (Assumes 'a' is defined in workspace)
xG = NeuralStore(a).xG;
time_mask = (xG >= calc_time_window(1) & xG <= calc_time_window(2));
dt = median(diff(xG(time_mask)));
max_lag_bins = min(round(max_lag_s / dt), sum(time_mask) - 1);
n_timebins_window = sum(time_mask);
time_axis = (0:max_lag_bins) * dt;

cond_data = round(NeuralStore(a).(grouping_vars{1}), 3);
[u_stim, ~, base_groups] = unique(cond_data, 'rows');
n_base_groups = max(base_groups);

% Structure to hold results for plotting
Res = struct();

% Processing Loop for Both Signals
for sig_idx = 1:numel(signals)
    sig = signals{sig_idx};
    fprintf('\n--- Processing Signal: %s ---\n', sig);
    
    raw_tensor = NeuralStore(a).(sig);
    phi_all = permute(raw_tensor(:, :, time_mask), [3, 1, 2]);
    [nTime, nTrials, nNeurons] = size(phi_all);
    
    % METHOD 2: Optimized Z-Score Implementation 
    % (We use the optimized one for the final metrics to save time, as it mathematically matches Method 1)
    rho_s_opt  = nan(nNeurons, n_base_groups, max_lag_bins+1);
    rho_ns_opt = nan(nNeurons, n_base_groups, max_lag_bins+1);
    
    for s_idx = 1:n_base_groups
        i_mask = (base_groups == s_idx);
        if sum(i_mask) < min_trials; continue; end
        
        phi_s = phi_all(:, i_mask, :);
        
        % --- Stationary ---
        mu_stat = mean(phi_s, [1, 2], 'omitnan');
        phi_cent = phi_s - mu_stat;
        sigma2_stat = squeeze(mean(phi_cent.^2, [1, 2], 'omitnan')); 
        sigma2_stat(sigma2_stat == 0) = NaN;
        
        % --- Non-Stationary (Z-scoring) ---
        mu_t = mean(phi_s, 2, 'omitnan');
        sig_t = std(phi_s, 0, 2, 'omitnan');
        sig_t(sig_t == 0) = NaN;
        Z_phi = (phi_s - mu_t) ./ sig_t;
        
        % --- Vectorized Lagging ---
        for tau = 0:max_lag_bins
            t_idx = 1:(nTime - tau);
            t_tau_idx = (1 + tau):nTime;
            
            % Eq 32 (Stationary)
            prod_s = phi_cent(t_idx, :, :) .* phi_cent(t_tau_idx, :, :);
            rho_s_opt(:, s_idx, tau+1) = squeeze(mean(prod_s, [1, 2], 'omitnan')) ./ sigma2_stat;
            
            % Eq 35 (Non-Stationary)
            prod_ns = Z_phi(t_idx, :, :) .* Z_phi(t_tau_idx, :, :);
            rho_ns_opt(:, s_idx, tau+1) = squeeze(mean(prod_ns, [1, 2], 'omitnan'));
        end
    end
    
    % ESS Calculation (Simplified: Sum All Timepoints)
    % Fixed bug: mean_rho_s now properly uses rho_s_opt
    mean_rho_s  = squeeze(mean(rho_s_opt, 2, 'omitnan')); 
    mean_rho_ns = squeeze(mean(rho_ns_opt, 2, 'omitnan')); 
    
    ESS_stat = nan(nNeurons, 1);
    ESS_nonstat = nan(nNeurons, 1);
    
    for n = 1:nNeurons
        r_s = mean_rho_s(n, :);
        if ~any(isnan(r_s))
            ESS_stat(n) = n_timebins_window / (1 + 2 * sum(r_s(2:end)));
        end
        
        r_ns = mean_rho_ns(n, :);
        if ~any(isnan(r_ns))
            ESS_nonstat(n) = n_timebins_window / (1 + 2 * sum(r_ns(2:end)));
        end
    end
    
    % Pool and clean data for plotting
    rho_ns_pooled = reshape(rho_ns_opt, [], max_lag_bins + 1);
    rho_s_pooled  = reshape(rho_s_opt, [], max_lag_bins + 1);
    
    valid_rows_ns = ~any(isnan(rho_ns_pooled), 2);
    valid_rows_s = ~any(isnan(rho_s_pooled), 2);
    
    % Save to results structure
    Res.(sig).rho_ns_clean = rho_ns_pooled(valid_rows_ns, :);
    Res.(sig).rho_s_clean = rho_s_pooled(valid_rows_s, :);
    Res.(sig).ESS_stat = ESS_stat;
    Res.(sig).ESS_nonstat = ESS_nonstat;
end

% 4. Comparative Plotting (Gspk vs GdF)
fprintf('\nPlotting Comparative Results...\n');
colors = struct('Stat', [0 0.4470 0.7410], 'NonStat', [0.8500 0.3250 0.0980]);

% --- Figure 1: Mean Autocorrelograms (1x2 Subplots) ---
figure('Name', 'Mean Autocorrelation (Spikes vs dF/F)', 'Color', 'w', 'Position', [100, 100, 1000, 400]);
for s_idx = 1:numel(signals)
    sig = signals{s_idx};
    subplot(1, 2, s_idx); hold on;
    
    % Stationary
    m_s = mean(Res.(sig).rho_s_clean, 1);
    sem_s = std(Res.(sig).rho_s_clean, 0, 1) ./ sqrt(size(Res.(sig).rho_s_clean, 1));
    errorbar(time_axis, m_s, sem_s, 'LineWidth', 2, 'Color', colors.Stat, 'DisplayName', 'Stationary (Eq 32)');
    
    % Non-Stationary
    m_ns = mean(Res.(sig).rho_ns_clean, 1);
    sem_ns = std(Res.(sig).rho_ns_clean, 0, 1) ./ sqrt(size(Res.(sig).rho_ns_clean, 1));
    errorbar(time_axis, m_ns, sem_ns, 'LineWidth', 2, 'Color', colors.NonStat, 'DisplayName', 'Non-Stationary (Eq 35)');
    
    yline(0, 'k--');
    xlabel('Time Lag \tau (s)'); ylabel('Autocorrelation \rho');
    title(sprintf('%s ACF (Pooled)', sig));
    legend('Location', 'northeast'); box off;
    ylim([-0.2 1]); % Standardize Y-axis for visual comparison
end

% --- Figure 2: 2D Histograms (2x2 Subplots) ---
figure('Name', '2D Histograms of ACF', 'Color', 'w', 'Position', [150, 150, 1000, 800]);
plot_idx = 1;
for s_idx = 1:numel(signals)
    sig = signals{s_idx};
    X_lags = repmat(time_axis, size(Res.(sig).rho_s_clean, 1), 1);
    
    % Stationary Heatmap
    subplot(2, 2, plot_idx);
    histogram2(X_lags(:), Res.(sig).rho_s_clean(:), ...
        'XBinEdges', [time_axis - dt/2, time_axis(end) + dt/2], 'YBinLimits', [-1, 1], ...
        'DisplayStyle', 'tile', 'ShowEmptyBins', 'off');
    colormap(parula); yline(0, 'r--', 'LineWidth', 1.5);
    xlabel('Time Lag \tau (s)'); ylabel('\rho_{n,s}');
    title(sprintf('%s - Stationary (Eq 32)', sig)); box off;
    
    % Non-Stationary Heatmap
    subplot(2, 2, plot_idx + 2);
    histogram2(X_lags(:), Res.(sig).rho_ns_clean(:), ...
        'XBinEdges', [time_axis - dt/2, time_axis(end) + dt/2], 'YBinLimits', [-1, 1], ...
        'DisplayStyle', 'tile', 'ShowEmptyBins', 'off');
    colormap(parula); yline(0, 'r--', 'LineWidth', 1.5);
    xlabel('Time Lag \tau (s)'); ylabel('\rho_{n,s}');
    title(sprintf('%s - Non-Stationary (Eq 35)', sig)); box off;
    
    plot_idx = plot_idx + 1;
end

% --- Figure 3: ESS Distributions (1x2 Subplots) ---
figure('Name', 'Effective Sample Size Distributions', 'Color', 'w', 'Position', [200, 200, 1000, 400]);
for s_idx = 1:numel(signals)
    sig = signals{s_idx};
    subplot(1, 2, s_idx); hold on;
    
    % Plot Histograms
    histogram(Res.(sig).ESS_stat, 'BinWidth', 1, 'FaceColor', colors.Stat, 'FaceAlpha', 0.6, 'DisplayName', 'Stationary');
    histogram(Res.(sig).ESS_nonstat, 'BinWidth', 1, 'FaceColor', colors.NonStat, 'FaceAlpha', 0.6, 'DisplayName', 'Non-Stationary');
    
    % Calculate Means
    mu_stat = mean(Res.(sig).ESS_stat, 'omitnan');
    mu_ns   = mean(Res.(sig).ESS_nonstat, 'omitnan');
    
    % Add Mean Lines
    xline(mu_stat, '--', sprintf('Mean: %.1f', mu_stat), ...
        'Color', colors.Stat, 'LineWidth', 2, 'LabelOrientation', 'horizontal', ...
        'LabelVerticalAlignment', 'top', 'HandleVisibility', 'off');
        
    xline(mu_ns, '--', sprintf('Mean: %.1f', mu_ns), ...
        'Color', colors.NonStat, 'LineWidth', 2, 'LabelOrientation', 'horizontal', ...
        'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
    
    % Add Theoretical Max Line
    xline(n_timebins_window, 'k-', 'Max', 'LineWidth', 1.5, ...
        'LabelVerticalAlignment', 'middle', 'HandleVisibility', 'off');
        
    xlabel('Effective Sample Size (Frames)'); ylabel('Count (Neurons)');
    title(sprintf('%s ESS Distribution', sig));
    legend('Location', 'northeast'); box off;
    xlim([0 n_timebins_window + 2]); % Standardize X-axis
end

fprintf('\n--- Processing Complete ---\n');


%% Part 8: Plotting Behavioral Traces (Velocity & Licks)
fprintf('\n=== Part 8: Plotting Behavioral Traces ===\n');

% 1. Determine maximum bins to preallocate matrices safely
maxBinsG = 0; maxBinsC = 0;
for a = 1:nAnimals
    if isempty(PerAnimalRaw(a).UD), continue; end
    maxBinsG = max(maxBinsG, length(PerAnimalRaw(a).UD(1).grating.timeBins));
    maxBinsC = max(maxBinsC, length(PerAnimalRaw(a).UD(1).corridor.binCenters));
end

nTotalTrials = height(TrialTbl_all);
velG_traces  = nan(nTotalTrials, maxBinsG);
lickG_traces = nan(nTotalTrials, maxBinsG);
velC_traces  = nan(nTotalTrials, maxBinsC);
lickC_traces = nan(nTotalTrials, maxBinsC);

% Grab axis labels (assuming uniformity across trials)
timeBinsG = PerAnimalRaw(1).UD(1).grating.timeBins;
posBinsC  = PerAnimalRaw(1).UD(1).corridor.binCenters;

% 2. Extract traces from the nested structs into flat matrices
rowIdx = 1;
for a = 1:nAnimals
    UD = PerAnimalRaw(a).UD;
    for t = 1:numel(UD)
        % Grating Epoch
        nG = length(UD(t).grating.vr_velocity);
        if nG > 0
            velG_traces(rowIdx, 1:nG)  = UD(t).grating.vr_velocity;
            lickG_traces(rowIdx, 1:nG) = UD(t).grating.vr_lick;
        end
        
        % Corridor Epoch
        nC = length(UD(t).corridor.vr_velocity);
        if nC > 0
            velC_traces(rowIdx, 1:nC)  = UD(t).corridor.vr_velocity;
            lickC_traces(rowIdx, 1:nC) = UD(t).corridor.vr_lick;
        end
        
        rowIdx = rowIdx + 1;
    end
end

% --- Scrub the first bin of the Grating epoch due to start-up artifacts ---
velG_traces(:, 1)  = NaN;
lickG_traces(:, 1) = NaN;
% -------------------------------------------------------------------------------

% 3. Compute trial scalars (averaged/summed across bins for each trial)
meanVelG_trial  = mean(velG_traces, 2, 'omitnan');
meanVelC_trial  = mean(velC_traces, 2, 'omitnan');
totalLickG_trial = sum(lickG_traces, 2, 'omitnan'); 
totalLickC_trial = sum(lickC_traces, 2, 'omitnan'); 

% --- NEW: Create signed contrast and dispersion for plotting ---
% Hits (1) and Misses (2) are Go stimuli. FAs (3) and CRs (4) are No-Go.
% Map Go to -1 (left side of plots) and No-Go to +1 (right side of plots)
is_go_stim = ismember(TrialTbl_all.outcome, [1, 2]); 
stim_sign = ones(nTotalTrials, 1);
stim_sign(is_go_stim) = -1; 

TrialTbl_all.signed_contrast   = TrialTbl_all.contrast .* stim_sign;
TrialTbl_all.signed_dispersion = TrialTbl_all.dispersion .* stim_sign;

% 4. Plot grouped by the specified stimulus conditions
groupVars   = {'abs_from_go', 'signed_contrast', 'signed_dispersion'};
groupLabels = {'|Distance from Go| (deg)', 'Signed Contrast (-:Go, +:NoGo)', 'Signed Dispersion (-:Go, +:NoGo)'};

for g = 1:length(groupVars)
    gVar   = groupVars{g};
    gLabel = groupLabels{g};
    
    % Find unique groups, rounding slightly to fix float matching issues
    groups = unique(round(TrialTbl_all.(gVar) * 100) / 100);
    groups(isnan(groups)) = []; 
    nGroups = length(groups);
    
    if nGroups <= 1, continue; end % Skip if the condition didn't vary
    
    % --- Dynamic Blue/Red Colormap ---
    if strcmp(gVar, 'abs_from_go')
        isGoGrp = groups < 45; % Category boundary is 45
    else
        isGoGrp = groups < 0;  % Go is strictly negative for signed vars
    end
    
    nGo = sum(isGoGrp);
    nNoGo = sum(~isGoGrp);
    colors = zeros(nGroups, 3);
    
    % Assign Blue gradient for Go groups
    if nGo == 1
        colors(isGoGrp, :) = [0, 0.4, 1]; % Solid blue
    elseif nGo > 1
        colors(isGoGrp, :) = [linspace(0, 0.4, nGo)', linspace(0.2, 0.8, nGo)', linspace(0.8, 1, nGo)'];
    end
    
    % Assign Red gradient for No-Go groups
    if nNoGo == 1
        colors(~isGoGrp, :) = [1, 0.2, 0.2]; % Solid red
    elseif nNoGo > 1
        colors(~isGoGrp, :) = [linspace(1, 0.6, nNoGo)', linspace(0.6, 0, nNoGo)', linspace(0.4, 0, nNoGo)'];
    end
    % ---------------------------------
    
    figure('Name', sprintf('Behavior by %s', gVar), 'Position', [100, 100, 1400, 800]);
    
    % Arrays for plotting the scalar averages
    sc_velG = nan(1, nGroups);  sc_velG_sem = nan(1, nGroups);
    sc_velC = nan(1, nGroups);  sc_velC_sem = nan(1, nGroups);
    sc_lckG = nan(1, nGroups);  sc_lckG_sem = nan(1, nGroups);
    sc_lckC = nan(1, nGroups);  sc_lckC_sem = nan(1, nGroups);
    
    for i = 1:nGroups
        % Boolean mask for the current condition
        idx = abs(TrialTbl_all.(gVar) - groups(i)) < 1e-4;
        n_idx = sum(idx);
        
        % ---------- TRACES (Mean across trials, keeping bins) ----------
        subplot(2,3,1); hold on;
        plot(timeBinsG, mean(velG_traces(idx,:), 1, 'omitnan'), 'Color', colors(i,:), 'LineWidth', 2);
        
        subplot(2,3,2); hold on;
        plot(posBinsC, mean(velC_traces(idx,:), 1, 'omitnan'), 'Color', colors(i,:), 'LineWidth', 2);
        
        subplot(2,3,4); hold on;
        plot(timeBinsG, mean(lickG_traces(idx,:), 1, 'omitnan'), 'Color', colors(i,:), 'LineWidth', 2);
        
        subplot(2,3,5); hold on;
        plot(posBinsC, mean(lickC_traces(idx,:), 1, 'omitnan'), 'Color', colors(i,:), 'LineWidth', 2);
        
        % ---------- SCALARS (Mean across bins and trials) ----------
        sc_velG(i) = mean(meanVelG_trial(idx), 'omitnan');
        sc_velC(i) = mean(meanVelC_trial(idx), 'omitnan');
        sc_lckG(i) = mean(totalLickG_trial(idx), 'omitnan');
        sc_lckC(i) = mean(totalLickC_trial(idx), 'omitnan');
        
        sc_velG_sem(i) = std(meanVelG_trial(idx), 'omitnan') / sqrt(n_idx);
        sc_velC_sem(i) = std(meanVelC_trial(idx), 'omitnan') / sqrt(n_idx);
        sc_lckG_sem(i) = std(totalLickG_trial(idx), 'omitnan') / sqrt(n_idx);
        sc_lckC_sem(i) = std(totalLickC_trial(idx), 'omitnan') / sqrt(n_idx);
    end
    
    % Add Labels and Aesthetics
    subplot(2,3,1); title('Grating Velocity'); xlabel('Time (s)'); ylabel('Velocity (cm/s)');
    subplot(2,3,2); title('Corridor Velocity'); xlabel('Position (cm)'); ylabel('Velocity (cm/s)');
    subplot(2,3,4); title('Grating Licks'); xlabel('Time (s)'); ylabel('Licks per Bin');
    subplot(2,3,5); title('Corridor Licks'); xlabel('Position (cm)'); ylabel('Licks per Bin');
    
    % Plot the Scalars (Subplot 3 and 6)
    subplot(2,3,3); hold on;
    errorbar(groups, sc_velG, sc_velG_sem, '-k', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    errorbar(groups, sc_velC, sc_velC_sem, '--k', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    
    % Overlay colored markers (HandleVisibility off so they avoid the legend)
    for i = 1:nGroups
        plot(groups(i), sc_velG(i), 'o', 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'k', 'MarkerSize', 8, 'HandleVisibility', 'off');
        plot(groups(i), sc_velC(i), 's', 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'k', 'MarkerSize', 8, 'HandleVisibility', 'off');
    end
    title('Epoch Average Velocity'); xlabel(gLabel); ylabel('Mean Vel (cm/s)');
    
    % Dummy plots for clean legend
    plot(NaN,NaN,'-ko','MarkerFaceColor','none','DisplayName','Grating');
    plot(NaN,NaN,'--ks','MarkerFaceColor','none','DisplayName','Corridor');
    legend('Location', 'best'); grid on;
    
    subplot(2,3,6); hold on;
    errorbar(groups, sc_lckG, sc_lckG_sem, '-k', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    errorbar(groups, sc_lckC, sc_lckC_sem, '--k', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    
    for i = 1:nGroups
        plot(groups(i), sc_lckG(i), 'o', 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'k', 'MarkerSize', 8, 'HandleVisibility', 'off');
        plot(groups(i), sc_lckC(i), 's', 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'k', 'MarkerSize', 8, 'HandleVisibility', 'off');
    end
    title('Epoch Total Licks'); xlabel(gLabel); ylabel('Total Licks');
    
    plot(NaN,NaN,'-ko','MarkerFaceColor','none','DisplayName','Grating');
    plot(NaN,NaN,'--ks','MarkerFaceColor','none','DisplayName','Corridor');
    legend('Location', 'best'); grid on;
end

%% Part 10: Single-Cell Tuning Profiles (Orientation, Contrast, Dispersion)
fprintf('\n=== Analyzing Single-Cell Tuning ===\n');

% Initialize stores for population arrays
pop_tuning_ori  = [];
pop_tuning_con  = [];
pop_tuning_disp = [];
pref_ori_idx    = [];

% Define global bins/levels based on actual data
all_oris = unique(TrialTbl_all.stimulus(~isnan(TrialTbl_all.stimulus)));
all_cons = unique(round(TrialTbl_all.contrast(~isnan(TrialTbl_all.contrast)), 2));
all_disps = unique(round(TrialTbl_all.dispersion(~isnan(TrialTbl_all.dispersion))));

for a = 1:numel(NeuralStore)
    tag = NeuralStore(a).tag;
    Gspk = NeuralStore(a).Gspk; % [nTrials x nNeurons x tBins]
    if isempty(Gspk), continue; end
    
    % Get trial-averaged activity in the grating window per neuron
    tmaskG = NeuralStore(a).xG >= timeWindowG(1) & NeuralStore(a).xG <= timeWindowG(2);
    X = squeeze(mean(Gspk(:,:,tmaskG), 3, 'omitnan')); % [nTrials x nNeurons]
    nNeurons = size(X, 2);
    
    % Trial metadata
    O = NeuralStore(a).stimulus(:);
    C = round(NeuralStore(a).contrast(:), 2);
    D = round(NeuralStore(a).dispersion(:));
    
    % Preallocate per-animal tuning matrices
    tun_o = nan(nNeurons, numel(all_oris));
    tun_c = nan(nNeurons, numel(all_cons));
    tun_d = nan(nNeurons, numel(all_disps));
    
    for n = 1:nNeurons
        rate = X(:, n);
        
        % 1. Orientation Tuning (Average across all C and D)
        for i = 1:numel(all_oris)
            tun_o(n, i) = mean(rate(O == all_oris(i)), 'omitnan');
        end
        
        % 2. Contrast Tuning
        for i = 1:numel(all_cons)
            tun_c(n, i) = mean(rate(C == all_cons(i)), 'omitnan');
        end
        
        % 3. Dispersion Tuning
        for i = 1:numel(all_disps)
            tun_d(n, i) = mean(rate(D == all_disps(i)), 'omitnan');
        end
    end
    
    % Min-max normalize tuning curves for population visualization
    norm_o = (tun_o - min(tun_o, [], 2)) ./ (max(tun_o, [], 2) - min(tun_o, [], 2) + eps);
    norm_c = (tun_c - min(tun_c, [], 2)) ./ (max(tun_c, [], 2) - min(tun_c, [], 2) + eps);
    norm_d = (tun_d - min(tun_d, [], 2)) ./ (max(tun_d, [], 2) - min(tun_d, [], 2) + eps);
    
    % Determine preferred orientation for sorting
    [~, p_idx] = max(norm_o, [], 2);
    
    pop_tuning_ori  = [pop_tuning_ori; norm_o];
    pop_tuning_con  = [pop_tuning_con; norm_c];
    pop_tuning_disp = [pop_tuning_disp; norm_d];
    pref_ori_idx    = [pref_ori_idx; p_idx];
end

% Sort population by preferred orientation
[~, sort_idx] = sort(pref_ori_idx);
sorted_pop_ori = pop_tuning_ori(sort_idx, :);

% Plotting Population Tuning
figure('Name', 'Population Tuning Profiles', 'Color', 'w', 'Position', [150, 150, 1200, 400]);

% 1. Orientation Heatmap
subplot(1,3,1);
imagesc(sorted_pop_ori);
colormap(parula);
set(gca, 'YDir', 'reverse');
xticks(1:numel(all_oris)); xticklabels(all_oris);
xlabel('Orientation (deg)'); ylabel('Neuron (sorted)');
title('Orientation Tuning (Normalized)');

% 2. Contrast Tuning (Mean +/- SEM across neurons)
subplot(1,3,2); hold on; grid on; box off;
m_con = mean(pop_tuning_con, 1, 'omitnan');
s_con = std(pop_tuning_con, 0, 1, 'omitnan') ./ sqrt(size(pop_tuning_con, 1));
errorbar(all_cons, m_con, s_con, 'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'w');
xlabel('Contrast'); ylabel('Normalized Activity');
title('Average Contrast Tuning');

% 3. Dispersion Tuning (Mean +/- SEM across neurons)
subplot(1,3,3); hold on; grid on; box off;
m_disp = mean(pop_tuning_disp, 1, 'omitnan');
s_disp = std(pop_tuning_disp, 0, 1, 'omitnan') ./ sqrt(size(pop_tuning_disp, 1));
errorbar(all_disps, m_disp, s_disp, 'k-s', 'LineWidth', 2, 'MarkerFaceColor', 'w');
xlabel('Dispersion (deg)'); ylabel('Normalized Activity');
title('Average Dispersion Tuning');

%% Part 12: Cross-Session Variance of Preferred Orientation
fprintf('\n=== Computing Cross-Session Orientation Spread ===\n');

% Initialize table to hold the circular standard deviation per neuron
CircVarData = struct('animal', {}, 'neuron_idx', {}, 'n_sessions', {}, 'circ_std_deg', {});
row_idx = 1;

for a = 1:numel(NeuralStore)
    tag = NeuralStore(a).tag;
    Gspk = NeuralStore(a).Gspk;
    if isempty(Gspk), continue; end
    
    % Get trial metadata for this animal
    idx_animal = strcmp(TrialTbl_all.animal, tag);
    T_ani = TrialTbl_all(idx_animal, :);
    sessions = unique(T_ani.session);
    nSessions = numel(sessions);
    
    if nSessions < 2, continue; end
    
    % Mean activity in grating window
    tmaskG = NeuralStore(a).xG >= timeWindowG(1) & NeuralStore(a).xG <= timeWindowG(2);
    X = squeeze(mean(Gspk(:,:,tmaskG), 3, 'omitnan')); % [nTrials x nNeurons]
    nNeurons = size(X, 2);
    
    oris = unique(T_ani.stimulus(~isnan(T_ani.stimulus)));
    angles_rad = deg2rad(oris * 2); % Double angle for 180-deg periodicity
    
    for n = 1:nNeurons
        pref_oris_across_sessions = nan(1, nSessions);
        
        for s_idx = 1:nSessions
            maskS = T_ani.session == sessions(s_idx);
            oriS = T_ani.stimulus(maskS);
            rateS = X(maskS, n);
            
            % Compute tuning curve for this session
            tunS = arrayfun(@(o) mean(rateS(oriS == o), 'omitnan'), oris);
            if any(isnan(tunS)), continue; end
            
            % Vector average to get preferred orientation
            x_comp = sum(tunS .* cos(angles_rad), 'omitnan');
            y_comp = sum(tunS .* sin(angles_rad), 'omitnan');
            
            theta_pref_rad = atan2(y_comp, x_comp) / 2;
            pref_oris_across_sessions(s_idx) = mod(rad2deg(theta_pref_rad) + 180, 180);
        end
        
        % Drop sessions where tuning couldn't be computed
        valid_prefs = pref_oris_across_sessions(~isnan(pref_oris_across_sessions));
        
        if numel(valid_prefs) >= 2
            % Circular standard deviation
            alpha = valid_prefs * (pi / 90);
            R = sqrt(mean(cos(alpha))^2 + mean(sin(alpha))^2);
            
            % Bound R to avoid log(0) or complex numbers from rounding errors
            R = min(max(R, 1e-6), 1); 
            circ_std = (90 / pi) * sqrt(-2 * log(R));
            
            CircVarData(row_idx).animal = tag;
            CircVarData(row_idx).neuron_idx = n;
            CircVarData(row_idx).n_sessions = numel(valid_prefs);
            CircVarData(row_idx).circ_std_deg = circ_std;
            row_idx = row_idx + 1;
        end
    end
end

CircVarTbl = struct2table(CircVarData);

% Plotting the Distributions
fprintf('--- Plotting Cross-Session Orientation Variance ---\n');

figure('Name', 'Cross-Session Drift of Preferred Orientation', 'Color', 'w', 'Position', [150, 150, 1000, 500]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% 1. Cumulative Distribution across all animals
nexttile; hold on; grid on; box off;
unique_animals = unique(CircVarTbl.animal);
colors = lines(numel(unique_animals));

for i = 1:numel(unique_animals)
    ani = unique_animals{i};
    data = CircVarTbl.circ_std_deg(strcmp(CircVarTbl.animal, ani));
    
    [f, x] = ecdf(data);
    plot(x, f, 'Color', colors(i,:), 'LineWidth', 2, 'DisplayName', sprintf('%s (n=%d)', ani, numel(data)));
end

% Add group average CDF
[f_all, x_all] = ecdf(CircVarTbl.circ_std_deg);
plot(x_all, f_all, 'k--', 'LineWidth', 3, 'DisplayName', 'Group Average');

xlabel('Circular Standard Deviation (\sigma_{deg})');
ylabel('Cumulative Fraction of Neurons');
title('CDF of Orientation Drift Across Sessions');
legend('Location', 'southeast', 'Interpreter', 'none');
xlim([0 90]); % Max possible spread

% 2. Boxplot comparing animals
nexttile; hold on; grid on; box off;
boxplot(CircVarTbl.circ_std_deg, CircVarTbl.animal, 'Colors', 'k', 'Symbol', 'k.');
ylabel('Circular Standard Deviation (\sigma_{deg})');
title('Distribution of Drift by Animal');
set(gca, 'TickLabelInterpreter', 'none');
ylim([0 90]);

%% Part 13: Distribution of Exact Preferred Orientations per Animal
fprintf('\n=== Computing Distribution of Exact Preferred Orientations ===\n');

figure('Name', 'Exact Preferred Orientations per Animal', 'Color', 'w', 'Position', [100 100 1200 600]);

nA = numel(NeuralStore);
nCols = 3; 
nRows = ceil(nA / nCols);
t = tiledlayout(nRows, nCols, 'TileSpacing', 'compact', 'Padding', 'compact');

for a = 1:nA
    tag = NeuralStore(a).tag;
    Gspk = NeuralStore(a).Gspk;
    
    if isempty(Gspk)
        nexttile; title(sprintf('%s (no data)', tag), 'Interpreter', 'none'); axis off;
        continue;
    end
    
    % Extract mean activity in the grating window
    tmaskG = NeuralStore(a).xG >= timeWindowG(1) & NeuralStore(a).xG <= timeWindowG(2);
    X = squeeze(mean(Gspk(:,:,tmaskG), 3, 'omitnan')); % [nTrials x nNeurons]
    nNeurons = size(X, 2);
    
    % Match trials to this animal's stimulus orientations
    idx_animal = strcmp(TrialTbl_all.animal, tag);
    oris_trial = TrialTbl_all.stimulus(idx_animal);
    
    % Get the exact unique orientations shown to this animal
    unique_oris = unique(oris_trial(~isnan(oris_trial)));
    
    pref_oris = nan(nNeurons, 1);
    
    for n = 1:nNeurons
        % Compute session-pooled tuning curve for the discrete stimuli
        tun = arrayfun(@(o) mean(X(oris_trial == o, n), 'omitnan'), unique_oris);
        
        % Skip cells with undefined or flat tuning
        if any(isnan(tun)) || all(tun == tun(1))
            continue; 
        end
        
        % Assign preference to the exact stimulus driving the highest mean rate
        [~, max_idx] = max(tun);
        pref_oris(n) = unique_oris(max_idx);
    end
    
    % Clean NaNs
    pref_oris = pref_oris(~isnan(pref_oris));
    
    % Plotting
    nexttile; hold on; grid on; box off;
    
    if isempty(pref_oris)
        title(sprintf('%s (no valid neurons)', tag), 'Interpreter', 'none'); axis off;
        continue;
    end
    
    % Count occurrences of each exact orientation
    counts = arrayfun(@(o) sum(pref_oris == o), unique_oris);
    
    % Plot as categorical bar chart to enforce the discrete stimulus space
    bar(unique_oris, counts, 'FaceColor', [0.3 0.5 0.7], 'EdgeColor', 'k');
    
    % Format axes tightly around the presented stimuli
    xlim([min(unique_oris) - 5, max(unique_oris) + 5]);
    xticks(unique_oris);
    xtickangle(45);
    xlabel('Exact Preferred Orientation (\circ)');
    ylabel('Neuron Count');
    title(sprintf('%s (n = %d)', tag, numel(pref_oris)), 'Interpreter', 'none');
end

sgtitle(t, 'Distribution of Exact Preferred Orientations by Animal', 'Interpreter', 'none');
%% Part 14: Longitudinal Drift as a Function of Preferred Orientation
fprintf('\n=== Computing Drift vs. Initial Preferred Orientation ===\n');

% Initialize storage for discrete shifts
DriftData = struct('animal', {}, 'session_pair', {}, 'neuron_idx', {}, ...
                   'pref_A', {}, 'pref_B', {}, 'abs_drift', {}, 'is_stable', {});
row_idx = 1;

for a = 1:numel(NeuralStore)
    tag = NeuralStore(a).tag;
    Gspk = NeuralStore(a).Gspk;
    if isempty(Gspk), continue; end
    
    idx_animal = strcmp(TrialTbl_all.animal, tag);
    T_ani = TrialTbl_all(idx_animal, :);
    sessions = unique(T_ani.session);
    
    if numel(sessions) < 2, continue; end
    
    % Mean activity in grating window
    tmaskG = NeuralStore(a).xG >= timeWindowG(1) & NeuralStore(a).xG <= timeWindowG(2);
    X = squeeze(mean(Gspk(:,:,tmaskG), 3, 'omitnan')); % [nTrials x nNeurons]
    nNeurons = size(X, 2);
    
    oris_animal = T_ani.stimulus(~isnan(T_ani.stimulus));
    unique_oris = unique(oris_animal);
    
    for s_idx = 1:(numel(sessions) - 1)
        maskA = T_ani.session == sessions(s_idx);
        maskB = T_ani.session == sessions(s_idx + 1);
        
        oriA = T_ani.stimulus(maskA);
        oriB = T_ani.stimulus(maskB);
        
        XA = X(maskA, :);
        XB = X(maskB, :);
        
        for n = 1:nNeurons
            % Tuning for session A
            tunA = arrayfun(@(o) mean(XA(oriA == o, n), 'omitnan'), unique_oris);
            % Tuning for session B
            tunB = arrayfun(@(o) mean(XB(oriB == o, n), 'omitnan'), unique_oris);
            
            % Skip if tuning is undefined or completely flat in either session
            if any(isnan(tunA)) || any(isnan(tunB)) || all(tunA == tunA(1)) || all(tunB == tunB(1))
                continue;
            end
            
            % Extract exact discrete preference
            [~, max_idx_A] = max(tunA);
            [~, max_idx_B] = max(tunB);
            
            pref_A = unique_oris(max_idx_A);
            pref_B = unique_oris(max_idx_B);
            
            abs_drift = abs(pref_B - pref_A);
            is_stable = (abs_drift == 0);
            
            % Store
            DriftData(row_idx).animal = tag;
            DriftData(row_idx).session_pair = sprintf('%d_vs_%d', sessions(s_idx), sessions(s_idx+1));
            DriftData(row_idx).neuron_idx = n;
            DriftData(row_idx).pref_A = pref_A;
            DriftData(row_idx).pref_B = pref_B;
            DriftData(row_idx).abs_drift = abs_drift;
            DriftData(row_idx).is_stable = is_stable;
            
            row_idx = row_idx + 1;
        end
    end
end

DriftTbl = struct2table(DriftData);

% Plotting Drift by Initial Preference (Group Level)
fprintf('--- Plotting Representational Drift vs Orientation ---\n');

if isempty(DriftTbl)
    warning('Not enough valid longitudinal data to compute orientation drift.');
else
    figure('Name', 'Drift vs Initial Preference', 'Color', 'w', 'Position', [150 150 1000 450]);
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % Get global unique orientations across the dataset
    all_unique_oris = unique(DriftTbl.pref_A);
    nOris = numel(all_unique_oris);
    
    m_drift = nan(nOris, 1);
    se_drift = nan(nOris, 1);
    p_stable = nan(nOris, 1);
    n_counts = nan(nOris, 1);
    
    for i = 1:nOris
        o = all_unique_oris(i);
        idx = DriftTbl.pref_A == o;
        
        n_counts(i) = sum(idx);
        if n_counts(i) > 0
            m_drift(i) = mean(DriftTbl.abs_drift(idx));
            se_drift(i) = std(DriftTbl.abs_drift(idx)) / sqrt(n_counts(i));
            p_stable(i) = mean(DriftTbl.is_stable(idx));
        end
    end
    
    % 1. Mean Absolute Drift
    nexttile; hold on; grid on; box off;
    bar(all_unique_oris, m_drift, 'FaceColor', [0.8 0.4 0.4], 'EdgeColor', 'k', 'FaceAlpha', 0.8);
    errorbar(all_unique_oris, m_drift, se_drift, 'k.', 'LineWidth', 1.5, 'CapSize', 0);
    
    xlim([min(all_unique_oris)-5, max(all_unique_oris)+5]);
    xticks(all_unique_oris);
    xtickangle(45);
    xlabel('Initial Preferred Orientation (\theta_A)');
    ylabel('Mean Absolute Drift |\theta_B - \theta_A| (\circ)');
    title('Magnitude of Drift by Preference');
    
    % Add N counts as text
    for i = 1:nOris
        text(all_unique_oris(i), m_drift(i) + se_drift(i) + 1, sprintf('n=%d', n_counts(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8);
    end
    
    % 2. Probability of Strict Stability
    nexttile; hold on; grid on; box off;
    bar(all_unique_oris, p_stable, 'FaceColor', [0.4 0.6 0.8], 'EdgeColor', 'k', 'FaceAlpha', 0.8);
    
    yline(1/nOris, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Chance Level');
    
    xlim([min(all_unique_oris)-5, max(all_unique_oris)+5]);
    xticks(all_unique_oris);
    xtickangle(45);
    ylim([0 1]);
    xlabel('Initial Preferred Orientation (\theta_A)');
    ylabel('Probability of Remaining Stable (\Delta\theta = 0)');
    title('Strict Stability by Preference');
    legend('Location', 'northeast');
end
%% Part 11: High-Dimensional Population Variability Metrics
fprintf('\n=== Computing Population Activity Variability Metrics ===\n');

% Preallocate arrays to append to TrialTbl_all
nT_total = height(TrialTbl_all);
[all_path_length, all_maha_dist, all_total_var, all_fano] = deal(nan(nT_total, 1));

idx_offset = 0;
for a = 1:numel(NeuralStore)
    tag = NeuralStore(a).tag;
    Gspk = NeuralStore(a).Gspk; % [nTrials x nNeurons x tBins]
    if isempty(Gspk), continue; end
    
    nT = size(Gspk, 1);
    nN = size(Gspk, 2);
    
    % Extract neural data in the chosen grating window
    tmaskG = NeuralStore(a).xG >= timeWindowG(1) & NeuralStore(a).xG <= timeWindowG(2);
    X = Gspk(:,:,tmaskG); % [nTrials x nNeurons x nTimeBins]
    nTime = size(X, 3);
    
    % Trial-averaged states
    X_mean = squeeze(mean(X, 3, 'omitnan')); % [nTrials x nNeurons]
    
    % Condition metadata for Mahalanobis
    C = round(NeuralStore(a).contrast(:), 3);
    D = round(NeuralStore(a).dispersion(:));
    O = NeuralStore(a).stimulus(:);
    
    % Precompute condition means and covariances to isolate trial-to-trial noise
    [unique_conds, ~, cond_idx] = unique([O, C, D], 'rows');
    nConds = size(unique_conds, 1);
    cond_mu = nan(nConds, nN);
    cond_cov = cell(nConds, 1);
    
    for c_idx = 1:nConds
        mask = (cond_idx == c_idx);
        % Require at least a few trials to estimate covariance, else use identity fallback
        if sum(mask) > 3
            cond_mu(c_idx, :) = mean(X_mean(mask, :), 1, 'omitnan');
            c_cov = cov(X_mean(mask, :), 'omitrows');
            % Shrinkage/regularization for matrix inversion stability
            cond_cov{c_idx} = c_cov + eye(nN) * 1e-3; 
        else
            cond_mu(c_idx, :) = mean(X_mean(mask, :), 1, 'omitnan');
            cond_cov{c_idx} = eye(nN); 
        end
    end
    
    % Compute per-trial metrics
    path_len = nan(nT, 1);
    maha_dist = nan(nT, 1);
    tot_var = nan(nT, 1);
    fano = nan(nT, 1);
    
    for tr = 1:nT
        trial_traj = squeeze(X(tr, :, :))'; % [nTimeBins x nNeurons]
        
        if all(isnan(trial_traj(:))), continue; end
        
        % 1. Path Length (Sum of L2 norms between consecutive time steps)
        if nTime > 1
            diffs = diff(trial_traj, 1, 1); % [(nTime-1) x nNeurons]
            path_len(tr) = sum(sqrt(sum(diffs.^2, 2))); % Euclidean distance
        end
        
        % 2. Condition-Conditioned Mahalanobis Distance
        c_i = cond_idx(tr);
        if ~any(isnan(X_mean(tr, :))) && ~any(isnan(cond_mu(c_i, :)))
            delta = X_mean(tr, :) - cond_mu(c_i, :);
            maha_dist(tr) = sqrt(delta * pinv(cond_cov{c_i}) * delta');
        end
        
        % 3. Total Variance (Trace of the temporal covariance matrix)
        trial_cov = cov(trial_traj, 'omitrows');
        tot_var(tr) = trace(trial_cov);
        
        % 4. Population Fano Factor (Time variance / Time mean)
        time_var = var(trial_traj, 0, 1, 'omitnan');
        time_mean = mean(trial_traj, 1, 'omitnan');
        fano(tr) = mean(time_var ./ (time_mean + eps), 'omitnan');
    end
    
    % Store in global arrays
    all_path_length(idx_offset + (1:nT)) = path_len;
    all_maha_dist(idx_offset + (1:nT)) = maha_dist;
    all_total_var(idx_offset + (1:nT)) = tot_var;
    all_fano(idx_offset + (1:nT)) = fano;
    
    idx_offset = idx_offset + nT;
end

% Append to the main table
TrialTbl_all.traj_path_length = all_path_length;
TrialTbl_all.traj_maha_dist   = all_maha_dist;
TrialTbl_all.traj_total_var   = all_total_var;
TrialTbl_all.traj_fano        = all_fano;


% Plotting: Population Variability vs. Inferred Uncertainty
fprintf('\n--- Plotting Variability Metrics vs. Uncertainty ---\n');

% Define the metrics and their labels
var_metrics = {'traj_path_length', 'traj_maha_dist', 'traj_total_var', 'traj_fano', 'logGV_gr'};
var_labels  = {'Path Length', 'Mahalanobis Dist (from Cond Mean)', 'Total Variance (Trace)', 'Pop. Fano Factor', 'log(GV)'};
unc_metrics = {'unc_perceptual', 'unc_decision'};
unc_labels  = {'Perceptual Uncertainty', 'Decision Uncertainty'};

nQ = 8; % Quantile bins for smoothing
figure('Name', 'Variability Metrics vs Uncertainty', 'Color', 'w', 'Position', [100, 100, 1400, 800]);
tiledlayout(numel(unc_metrics), numel(var_metrics), 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:numel(unc_metrics)
    x_var = TrialTbl_all.(unc_metrics{i});
    
    for j = 1:numel(var_metrics)
        y_var = TrialTbl_all.(var_metrics{j});
        
        nexttile; hold on; grid on; box off;
        
        % Filter valid data
        valid = isfinite(x_var) & isfinite(y_var);
        if sum(valid) < 50
            title('Insufficient Data'); continue; 
        end
        
        x_clean = x_var(valid);
        y_clean = y_var(valid);
        
        % Compute quantiles
        q_edges = uniquetol(quantile(x_clean, linspace(0, 1, nQ+1)), 1e-9);
        if numel(q_edges) < 3, q_edges = linspace(min(x_clean), max(x_clean), nQ+1); end
        q_edges(end) = q_edges(end) + eps;
        
        [~, ~, bin] = histcounts(x_clean, q_edges);
        
        % Binned statistics
        mX = accumarray(bin, x_clean, [], @mean, NaN);
        mY = accumarray(bin, y_clean, [], @mean, NaN);
        sY = accumarray(bin, y_clean, [], @(v) std(v)/sqrt(max(1, sum(~isnan(v)))), NaN);
        
        % Plot error bars
        errorbar(mX, mY, sY, 'k-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'w');
        
        % Spearman correlation on raw trials
        [r, p] = corr(x_clean, y_clean, 'type', 'Spearman');
        
        xlabel(unc_labels{i});
        ylabel(var_labels{j});
        title(sprintf('\\rho = %.2f (p = %.2g)', r, p));
        axis tight; 
    end
end
%% Plot the empirical prior (Categorical Bar Plots)
figure('Color','w'); 
t = tiledlayout('flow');

% --- 1. Orientations ---
nexttile
hold on;
% Get counts for each unique stimulus value
[counts_ori, unique_ori] = groupcounts(TrialTbl_all.stimulus);
% Convert to probability
probs_ori = counts_ori / sum(counts_ori);
% Plot as a bar chart
bar(probs_ori, 'FaceColor', [0.3 0.3 0.7]);
% Set labels
xticks(1:numel(unique_ori));
xticklabels(unique_ori);
title('Stimulus Orientation');
ylabel('Probability');
xlabel('Orientation (deg)');
grid on; box off;

% --- 2. Contrasts ---
nexttile
hold on;
[counts_con, unique_con] = groupcounts(TrialTbl_all.contrast);
probs_con = counts_con / sum(counts_con);
bar(probs_con, 'FaceColor', [0.3 0.7 0.3]);
xticks(1:numel(unique_con));
xticklabels(unique_con);
title('Stimulus Contrast');
ylabel('Probability');
xlabel('Contrast');
grid on; box off;

% --- 3. Dispersions ---
nexttile
hold on;
[counts_disp, unique_disp] = groupcounts(TrialTbl_all.dispersion);
probs_disp = counts_disp / sum(counts_disp);
bar(probs_disp, 'FaceColor', [0.7 0.3 0.3]);
xticks(1:numel(unique_disp));
xticklabels(unique_disp);
title('Stimulus Dispersion');
ylabel('Probability');
xlabel('Dispersion (deg)');
grid on; box off;

%% New behavioural plots
%
% This section generates psychometric and neurometric curves based on the
% full 'TrialTbl_all' table, without any model fitting.
% It uses the exact 'abs_from_go' values present in the data as the
% x-axis bins for all plots.
%
fprintf('\n=== Starting New Plotting Section (Simpler grpstats approach) ===\n');

% --- Setup: Define helper functions and common variables ---
if ~exist('TrialTbl_all', 'var')
    error('Please run VR_multi_animal_analysis.m first to generate TrialTbl_all.');
end

% Create a working copy and round the key variables for robust grouping
T = TrialTbl_all;
T.abs_from_go_binned = round(T.abs_from_go, 6); % Bin by exact stimulus value
T.contrast_binned = round(T.contrast, 3);
T.dispersion_binned = round(T.dispersion);

% Helper functions for aggregation
mean_fun = @(x) mean(x, 'omitnan');
sem_fun  = @(x) std(x, 0, 'omitnan') / sqrt(max(1, sum(~isnan(x))));

% --- Plot 1: Per-Session Psychometrics ---
fprintf('Plot 1: Per-Session Psychometrics...\n');
fig1 = figure('Name', 'Psychometric: Per Session', 'Color', 'w', 'Position', [100 100 800 600]);
hold on;

% Get all unique sessions (as identified by animal + session number)
[unique_sessions, ~, session_group_idx] = unique(T(:, {'animal', 'session'}), 'rows');
n_sessions = height(unique_sessions);
all_session_curves = {}; % Store curves for averaging

% Get all unique x-bins across the entire dataset for plotting the average
all_x_bins = unique(T.abs_from_go_binned(~isnan(T.abs_from_go_binned)));

% Loop through each session
for i = 1:n_sessions
    % Get data for this session
    T_sess = T(session_group_idx == i, :);
    valid_sess = ~isnan(T_sess.abs_from_go_binned) & ~isnan(T_sess.goChoice);
    if sum(valid_sess) < 10, continue; end % Skip if not enough data
    
    % Use grpstats to get this session's curve
    [means, x_bins_str] = grpstats(T_sess.goChoice(valid_sess), T_sess.abs_from_go_binned(valid_sess), {'mean', 'gname'});
    x_bins_num = str2double(x_bins_str);
    
    % Plot this session's curve
    h_indiv_line = plot(x_bins_num', means', 'Color', [0.8 0.8 0.8], 'LineWidth', 0.5);
    
    % Store this curve (interpolated to common x-axis) for averaging
    all_session_curves{i} = interp1(x_bins_num', means', all_x_bins', 'linear');
end

% Calculate and plot the average
if ~isempty(all_session_curves)
    session_matrix = vertcat(all_session_curves{:}); % n_sessions x n_bins
    avg_all_sessions = mean(session_matrix, 1, 'omitnan');
    h_avg = plot(all_x_bins, avg_all_sessions, 'k-o', 'LineWidth', 2.5, 'MarkerFaceColor', 'k');
    
    legend([h_avg, h_indiv_line], {'Average (all sessions)', 'Individual sessions'}, 'Location', 'best');
end

% Tidy plot
title('Psychometrics: Per Session');
xlabel('|\Delta from Go| (deg)');
ylabel('P(Go)');
grid on; box on;
ylim([0 1]);
xlim([min(all_x_bins)-1, max(all_x_bins)+1]);

% --- Plot 2: Per-Animal Psychometrics ---
fprintf('Plot 2: Per-Animal Psychometrics...\n');
fig2 = figure('Name', 'Psychometric: Per Animal', 'Color', 'w', 'Position', [150 150 800 600]);
hold on;

% Get all unique animals
[unique_animals, ~, animal_group_idx] = unique(T.animal);
n_animals = numel(unique_animals);
all_animal_curves = {}; % Store curves for averaging

% Loop through each animal
for i = 1:n_animals
    T_animal = T(animal_group_idx == i, :);
    valid_animal = ~isnan(T_animal.abs_from_go_binned) & ~isnan(T_animal.goChoice);
    if sum(valid_animal) < 10, continue; end
    
    [means, x_bins_str] = grpstats(T_animal.goChoice(valid_animal), T_animal.abs_from_go_binned(valid_animal), {'mean', 'gname'});
    x_bins_num = str2double(x_bins_str);
    
    h_indiv_ani_line = plot(x_bins_num', means', 'Color', [0.7 0.7 0.7], 'LineWidth', 1.0);
    
    all_animal_curves{i} = interp1(x_bins_num', means', all_x_bins', 'linear');
end

% Calculate and plot the average
if ~isempty(all_animal_curves)
    animal_matrix = vertcat(all_animal_curves{:}); % n_animals x n_bins
    avg_all_animals = mean(animal_matrix, 1, 'omitnan');
    h_avg_ani = plot(all_x_bins, avg_all_animals, 'k-o', 'LineWidth', 2.5, 'MarkerFaceColor', 'k');

    legend([h_avg_ani, h_indiv_ani_line], {'Average (all animals)', 'Individual animals'}, 'Location', 'best');
end

% Tidy plot
title('Psychometrics: Per Animal');
xlabel('|\Delta from Go| (deg)');
ylabel('P(Go)');
grid on; box on;
ylim([0 1]);
xlim([min(all_x_bins)-1, max(all_x_bins)+1]);


% --- Plot 3: Psychometrics by Contrast ---
fprintf('Plot 3: Psychometrics by Contrast...\n');
fig3 = figure('Name', 'Psychometric: By Contrast', 'Color', 'w', 'Position', [200 200 800 600]);
hold on;

valid_mask_con = ~isnan(T.abs_from_go_binned) & ~isnan(T.goChoice) & ~isnan(T.contrast_binned);
T_con_valid = T(valid_mask_con, :);
contrast_levels = unique(T_con_valid.contrast_binned);
n_contrasts = numel(contrast_levels);

colors_con = parula(n_contrasts + 2); % Get a nice colormap
colors_con = colors_con(2:end-1, :); % Trim the ends
h_con = [];
leg_con = {};

% Loop through each contrast level
for i_c = 1:n_contrasts
    % Filter data for this contrast
    T_con_level = T_con_valid(T_con_valid.contrast_binned == contrast_levels(i_c), :);
    if height(T_con_level) < 10, continue; end
    
    % Get the curve for this contrast
    [means, x_bins_str] = grpstats(T_con_level.goChoice, T_con_level.abs_from_go_binned, {'mean', 'gname'});
    x_bins_num = str2double(x_bins_str);
    
    % Plot it
    h = plot(x_bins_num', means', 'o-', 'Color', colors_con(i_c, :), ...
             'LineWidth', 2, 'MarkerFaceColor', 'w');
    h_con(end+1) = h;
    leg_con{end+1} = sprintf('Contrast = %.0f%%', contrast_levels(i_c) * 100);
end

% Tidy plot
title('Psychometrics by Contrast');
xlabel('|\Delta from Go| (deg)');
ylabel('P(Go)');
if ~isempty(h_con)
    legend(h_con, leg_con, 'Location', 'best');
end
grid on; box on;
ylim([0 1]);
xlim([min(all_x_bins)-1, max(all_x_bins)+1]);

% --- Plot 4: Psychometrics by Dispersion ---
fprintf('Plot 4: Psychometrics by Dispersion...\n');
fig4 = figure('Name', 'Psychometric: By Dispersion', 'Color', 'w', 'Position', [250 250 800 600]);
hold on;

valid_mask_disp = ~isnan(T.abs_from_go_binned) & ~isnan(T.goChoice) & ~isnan(T.dispersion_binned);
T_disp_valid = T(valid_mask_disp, :);
disp_levels = unique(T_disp_valid.dispersion_binned);
n_disps = numel(disp_levels);

colors_disp = hot(n_disps + 2); % Use a different colormap
colors_disp = colors_disp(1:end-2, :); % Trim the light end
h_disp = [];
leg_disp = {};

% Loop through each dispersion level
for i_d = 1:n_disps
    T_disp_level = T_disp_valid(T_disp_valid.dispersion_binned == disp_levels(i_d), :);
    if height(T_disp_level) < 10, continue; end
    
    [means, x_bins_str] = grpstats(T_disp_level.goChoice, T_disp_level.abs_from_go_binned, {'mean', 'gname'});
    x_bins_num = str2double(x_bins_str);
    
    h = plot(x_bins_num', means', 's-', 'Color', colors_disp(i_d, :), ...
             'LineWidth', 2, 'MarkerFaceColor', 'w');
    h_disp(end+1) = h;
    leg_disp{end+1} = sprintf('Dispersion = %d deg', disp_levels(i_d));
end

% Tidy plot
title('Psychometrics by Dispersion');
xlabel('|\Delta from Go| (deg)');
ylabel('P(Go)');
if ~isempty(h_disp)
    legend(h_disp, leg_disp, 'Location', 'best');
end
grid on; box on;
ylim([0 1]);
xlim([min(all_x_bins)-1, max(all_x_bins)+1]);

% --- Plot 5: Pre-RZ Lick Rate vs. Stimulus ---
fprintf('Plot 5: Pre-RZ Lick Rate...\n');
fig5 = figure('Name', 'Behavior: Lick Rate', 'Color', 'w', 'Position', [300 300 800 600]);
hold on;

valid_mask_lick = ~isnan(T.abs_from_go_binned) & ~isnan(T.preRZ_lick_rate);
if sum(valid_mask_lick) > 10
    % This is a simple group plot, so one grpstats call is perfect
    [lick_rate_means, lick_rate_sems, x_bins_str] = grpstats(T.preRZ_lick_rate(valid_mask_lick), T.abs_from_go_binned(valid_mask_lick), {'mean', 'sem', 'gname'});
    x_bins_lick = str2double(x_bins_str);

    % Plot
    shadedErrorBar(x_bins_lick, lick_rate_means, lick_rate_sems, ...
        'lineprops', {'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'w'});

    % Tidy plot
    title('Pre-RZ Lick Rate vs. Stimulus');
    xlabel('|\Delta from Go| (deg)');
    ylabel('Pre-RZ Lick Rate (licks/s)');
    grid on; box on;
    xlim([min(x_bins_lick)-1, max(x_bins_lick)+1]);
else
    title('Pre-RZ Lick Rate vs. Stimulus (No sufficient data)');
end

% --- Plot 6: Pre-RZ Velocity vs. Stimulus ---
fprintf('Plot 6: Pre-RZ Velocity...\n');
fig6 = figure('Name', 'Behavior: Velocity', 'Color', 'w', 'Position', [350 350 800 600]);
hold on;

valid_mask_vel = ~isnan(T.abs_from_go_binned) & ~isnan(T.preRZ_velocity);
if sum(valid_mask_vel) > 10
    [vel_means, vel_sems, x_bins_str] = grpstats(T.preRZ_velocity(valid_mask_vel), T.abs_from_go_binned(valid_mask_vel), {'mean', 'sem', 'gname'});
    x_bins_vel = str2double(x_bins_str);

    % Plot
    shadedErrorBar(x_bins_vel, vel_means, vel_sems, ...
        'lineprops', {'r-s', 'LineWidth', 2, 'MarkerFaceColor', 'w'});

    % Tidy plot
    title('Pre-RZ Velocity vs. Stimulus');
    xlabel('|\Delta from Go| (deg)');
    ylabel('Pre-RZ Velocity (cm/s)');
    grid on; box on;
    xlim([min(x_bins_vel)-1, max(x_bins_vel)+1]);
else
    title('Pre-RZ Velocity vs. Stimulus (No sufficient data)');
end

fprintf('--- New plotting section complete. ---\de');
%% Part 3: Group-Level Neural & Behavioural Plots (inc. psychometrics)
fprintf('\n=== Part 3: Group Plots ===\n');

% Merge contrast levels with tolerance
all_orients   = unique(TrialTbl_all.stimulus);
all_contrasts = merge_levels(unique(TrialTbl_all.contrast), contrastMergeTol);
all_disp      = unique(TrialTbl_all.dispersion);
TrialTbl_all.contrast = snap_to_levels(TrialTbl_all.contrast, all_contrasts);

% --- Ensure lick RATE is present (compute from counts/duration if needed) ---
if ~ismember('preRZ_lick_rate', TrialTbl_all.Properties.VariableNames)
    if ismember('preRZ_licks', TrialTbl_all.Properties.VariableNames) && ...
       ismember('preRZ_duration_s', TrialTbl_all.Properties.VariableNames)
        TrialTbl_all.preRZ_lick_rate = TrialTbl_all.preRZ_licks ./ TrialTbl_all.preRZ_duration_s;
        TrialTbl_all.preRZ_lick_rate(~isfinite(TrialTbl_all.preRZ_lick_rate)) = NaN;
        warning('preRZ_lick_rate was missing; computed from preRZ_licks ./ preRZ_duration_s.');
    else
        error('preRZ_lick_rate unavailable and cannot be derived (need preRZ_licks & preRZ_duration_s).');
    end
end

% Pool neural metrics across animals (random-effects via per-animal means)
[GrandMean, GrandSEM] = pool_metrics_across_animals(AnimalSummaries, ...
    {'meanAct_gr','meanAct_co'}, ...
    all_orients, all_contrasts, all_disp);

% Plot neural curves
dispLabels = arrayfun(@(d) sprintf('disp=%.3g', d), all_disp, 'UniformOutput', false);
conColors = lines(numel(all_contrasts));

plot_neural_errorbars_group_z(NeuralStore, timeWindowG, ...
    'target_contrast', 1.0, 'target_dispersion', 5);

% Per-animal (using precomputed per-trial field):
plot_activity_heatmaps_per_animal(TrialTbl_all, Animals, 'meanAct_gr');

% Group, z-scored per animal/per neuron (uses NeuralStore & your time window):
plot_activity_heatmap_group_z(NeuralStore, timeWindowG);

% --- Behaviour-only psychometrics aligned by go (group & per animal) ---
plot_psy_goAligned_perAnimal(TrialTbl_all, Animals, 'goChoice',        'P(Go)');
plot_psy_goAligned_perAnimal(TrialTbl_all, Animals, 'preRZ_velocity',  'Pre-RZ velocity (cm/s)');
plot_psy_goAligned_perAnimal(TrialTbl_all, Animals, 'preRZ_lick_rate', 'Pre-RZ lick rate (licks/s)');

% If your sigmoid helper still assumes a field named 'preRZ_licks', create a local alias:
Tbl_rate = TrialTbl_all;
Tbl_rate.preRZ_licks = TrialTbl_all.preRZ_lick_rate;  % shim for legacy plotting code
plot_preRZ_sigmoid_perAnimal_exact(Tbl_rate, Animals, false);

plot_psy_goAligned_group(TrialTbl_all, 'goChoice',        'P(Go)');
plot_psy_goAligned_group(TrialTbl_all, 'preRZ_velocity',  'Pre-RZ velocity (cm/s)');
plot_psy_goAligned_group(TrialTbl_all, 'preRZ_lick_rate', 'Pre-RZ lick rate (licks/s)');

% --- NEW: pupil plots by orientation, outcome, contrast (grouped across animals) ---
plot_pupil_by_orientation_C1D5_and_avg(PerAnimalRaw);
plot_pupil_heatmap_contrast_dispersion(PerAnimalRaw);

%% A) preRZ behavior (licks/velocity) ↔ neural activity (mean & GV)
% Global quantiles; split by GO / NO-GO side

% --- Side definition per trial ---
isVertGo   = logical(TrialTbl_all.go_is_vertical);  % per-trial flag (copied per animal)
theta      = TrialTbl_all.theta_deg;                % true orientation (deg)
isVertCat  = theta > 45;                            % vertical-category trials
onBoundary = theta == 45;

goSideMask    = ( isVertGo  &  isVertCat) | (~isVertGo  & ~isVertCat);
noGoSideMask  = ~goSideMask;
baseValidMask = ~onBoundary & ~isnan(theta) & ~isnan(isVertGo);

nQ = 10;  % number of quantile bins to use for all panels

figure('Name','preRZ behavior vs neural activity (GO vs NO-GO, global quantiles)','Position',[80 80 1200 520]);
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

% ---- 1) preRZ lick RATE vs mean activity (grating) ----
nexttile; hold on; grid on; box off;
[xc, mY_go, sY_go, mY_ng, sY_ng, r_go, p_go, r_ng, p_ng] = ...
    bin_split_global(TrialTbl_all.preRZ_lick_rate, TrialTbl_all.meanAct_gr, goSideMask, noGoSideMask, baseValidMask, nQ);
h1 = errorbar(xc, mY_go, sY_go, '-o','LineWidth',2,'MarkerFaceColor','w','DisplayName','GO side');
h2 = errorbar(xc, mY_ng, sY_ng, '-s','LineWidth',2,'MarkerFaceColor','w','DisplayName','NO-GO side');
xlabel('preRZ lick rate (licks/s; global quantiles on x)'); ylabel('Mean activity (grating)');
title(sprintf('Lick rate vs MeanAct  (r_{GO}=%.2f,p=%.2g; r_{NO}=%.2f,p=%.2g)', r_go, p_go, r_ng, p_ng));
legend([h1 h2],'Location','best');

% ---- 2) preRZ lick RATE vs logGV (grating) ----
nexttile; hold on; grid on; box off;
[xc, mY_go, sY_go, mY_ng, sY_ng, r_go, p_go, r_ng, p_ng] = ...
    bin_split_global(TrialTbl_all.preRZ_lick_rate, TrialTbl_all.logGV_gr, goSideMask, noGoSideMask, baseValidMask, nQ);
h1 = errorbar(xc, mY_go, sY_go, '-o','LineWidth',2,'MarkerFaceColor','w','DisplayName','GO side');
h2 = errorbar(xc, mY_ng, sY_ng, '-s','LineWidth',2,'MarkerFaceColor','w','DisplayName','NO-GO side');
xlabel('preRZ lick rate (licks/s; global quantiles on x)'); ylabel('log GV (grating)');
title(sprintf('Lick rate vs logGV  (r_{GO}=%.2f,p=%.2g; r_{NO}=%.2f,p=%.2g)', r_go, p_go, r_ng, p_ng));
legend([h1 h2],'Location','best');

% ---- 3) preRZ lick RATE vs normGV (grating) ----
nexttile; hold on; grid on; box off;
[xc, mY_go, sY_go, mY_ng, sY_ng, r_go, p_go, r_ng, p_ng] = ...
    bin_split_global(TrialTbl_all.preRZ_lick_rate, TrialTbl_all.normGV_gr, goSideMask, noGoSideMask, baseValidMask, nQ);
h1 = errorbar(xc, mY_go, sY_go, '-o','LineWidth',2,'MarkerFaceColor','w','DisplayName','GO side');
h2 = errorbar(xc, mY_ng, sY_ng, '-s','LineWidth',2,'MarkerFaceColor','w','DisplayName','NO-GO side');
xlabel('preRZ lick rate (licks/s; global quantiles on x)'); ylabel('norm GV (grating)');
title(sprintf('Lick rate vs normGV  (r_{GO}=%.2f,p=%.2g; r_{NO}=%.2f,p=%.2g)', r_go, p_go, r_ng, p_ng));
legend([h1 h2],'Location','best');

% ---- 4) preRZ velocity vs mean activity (grating) ----
nexttile; hold on; grid on; box off;
[xc, mY_go, sY_go, mY_ng, sY_ng, r_go, p_go, r_ng, p_ng] = ...
    bin_split_global(TrialTbl_all.preRZ_velocity, TrialTbl_all.meanAct_gr, goSideMask, noGoSideMask, baseValidMask, nQ);
h1 = errorbar(xc, mY_go, sY_go, '-o','LineWidth',2,'MarkerFaceColor','w','DisplayName','GO side');
h2 = errorbar(xc, mY_ng, sY_ng, '-s','LineWidth',2,'MarkerFaceColor','w','DisplayName','NO-GO side');
xlabel('preRZ velocity (cm/s, global quantiles on x)'); ylabel('Mean activity (grating)');
title(sprintf('Vel vs MeanAct  (r_{GO}=%.2f,p=%.2g; r_{NO}=%.2f,p=%.2g)', r_go, p_go, r_ng, p_ng));
legend([h1 h2],'Location','best');

% ---- 5) preRZ velocity vs logGV (grating) ----
nexttile; hold on; grid on; box off;
[xc, mY_go, sY_go, mY_ng, sY_ng, r_go, p_go, r_ng, p_ng] = ...
    bin_split_global(TrialTbl_all.preRZ_velocity, TrialTbl_all.logGV_gr, goSideMask, noGoSideMask, baseValidMask, nQ);
h1 = errorbar(xc, mY_go, sY_go, '-o','LineWidth',2,'MarkerFaceColor','w','DisplayName','GO side');
h2 = errorbar(xc, mY_ng, sY_ng, '-s','LineWidth',2,'MarkerFaceColor','w','DisplayName','NO-GO side');
xlabel('preRZ velocity (cm/s, global quantiles on x)'); ylabel('log GV (grating)');
title(sprintf('Vel vs logGV  (r_{GO}=%.2f,p=%.2g; r_{NO}=%.2f,p=%.2g)', r_go, p_go, r_ng, p_ng));
legend([h1 h2],'Location','best');

% ---- 6) preRZ velocity vs normGV (grating) ----
nexttile; hold on; grid on; box off;
[xc, mY_go, sY_go, mY_ng, sY_ng, r_go, p_go, r_ng, p_ng] = ...
    bin_split_global(TrialTbl_all.preRZ_velocity, TrialTbl_all.normGV_gr, goSideMask, noGoSideMask, baseValidMask, nQ);
h1 = errorbar(xc, mY_go, sY_go, '-o','LineWidth',2,'MarkerFaceColor','w','DisplayName','GO side');
h2 = errorbar(xc, mY_ng, sY_ng, '-s','LineWidth',2,'MarkerFaceColor','w','DisplayName','NO-GO side');
xlabel('preRZ velocity (cm/s, global quantiles on x)'); ylabel('norm GV (grating)');
title(sprintf('Vel vs normGV  (r_{GO}=%.2f,p=%.2g; r_{NO}=%.2f,p=%.2g)', r_go, p_go, r_ng, p_ng));
legend([h1 h2],'Location','best');




%% F) Quantile-binned preRZ behavior vs performance — split by GO / NO-GO side
% Side definition:
% - If this animal's GO is "vertical", then stimuli < 45° are GO-side.
% - If this animal's GO is "horizontal", then stimuli > 45° are GO-side.
% - Trials exactly at 45° are excluded (boundary).

isVertGo   = logical(TrialTbl_all.go_is_vertical);   % per-trial boolean (copied per animal)
theta      = TrialTbl_all.theta_deg;                 % true orientation (deg)
isVertCat  = theta > 45;                             % vertical-category trials
onBoundary = theta == 45;

goSideMask    = ( isVertGo  &  isVertCat) | (~isVertGo  & ~isVertCat);
noGoSideMask  = ~goSideMask;
validMask     = ~onBoundary & ~isnan(theta) & ~isnan(isVertGo);

% Helper to plot one metric split by side
plot_split_quantiles = @(x_all, y_all, xlab_str, ttl_str) ...
    ( ...
        plot_split_quantiles_impl(x_all, y_all, goSideMask & validMask, noGoSideMask & validMask, xlab_str, ttl_str) ...
    );

figure('Name','preRZ behavior (by side) vs performance','Position',[130 130 980 420]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

% ---- (1) preRZ lick RATE quantiles vs P(correct), GO vs NO-GO sides ----
nexttile; hold on; grid on; box off;
plot_split_quantiles(TrialTbl_all.preRZ_lick_rate, double(TrialTbl_all.performance), ...
    'preRZ lick rate (licks/s, quantiles on x)', 'Lick rate vs Performance — GO vs NO-GO');

% ---- (2) preRZ velocity quantiles vs P(correct), GO vs NO-GO sides ----
nexttile; hold on; grid on; box off;
plot_split_quantiles(TrialTbl_all.preRZ_velocity, double(TrialTbl_all.performance), ...
    'preRZ velocity (cm/s, quantiles on x)', 'Velocity vs Performance — GO vs NO-GO');

%% 4) 2D heatmaps across *actual* contrast × dispersion (mean uncertainty)
C = TrialTbl_all.contrast(:);
D = TrialTbl_all.dispersion(:);
Uper = TrialTbl_all.unc_perceptual(:);
Udec = TrialTbl_all.unc_decision(:);

% light rounding to avoid floating-point duplicates; adjust decimals if needed
Cr = round(C, 6);
Dr = round(D, 6);

% unique actual axis values (sorted)
cVals = unique(Cr);
dVals = unique(Dr);

% map each trial to exact index on each axis
[~, cIdx] = ismember(Cr, cVals);
[~, dIdx] = ismember(Dr, dVals);

% mean and counts on the exact grid
meanfun = @(v) mean(v,'omitnan');
M_p = accumarray([cIdx dIdx], Uper, [numel(cVals) numel(dVals)], meanfun, NaN);
M_d = accumarray([cIdx dIdx], Udec, [numel(cVals) numel(dVals)], meanfun, NaN);
N   = accumarray([cIdx dIdx], 1,    [numel(cVals) numel(dVals)], @sum, 0);

% (optional) mask cells with very low N
minN = 3;                          % set threshold if you want
M_p(N < minN) = NaN;
M_d(N < minN) = NaN;

% prepare alpha so NaNs render transparent
A_p = ~isnan(M_p');
A_d = ~isnan(M_d');

figure('Color','w','Position',[100 100 980 380]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

% --- Perceptual
nexttile; 
imagesc(cVals, dVals, M_p'); 
hold on; h = imagesc(cVals, dVals, M_p'); set(h, 'AlphaData', A_p);
title('Perceptual uncertainty (mean)');
xlabel('Contrast'); ylabel('Dispersion (deg)');
cb = colorbar; cb.Label.String = 'Uncertainty';


% --- Decision
nexttile; 
imagesc(cVals, dVals, M_d'); 
hold on; h2 = imagesc(cVals, dVals, M_d'); set(h2, 'AlphaData', A_d);
title('Decision uncertainty (mean)');
xlabel('Contrast'); ylabel('Dispersion (deg)');
cb2 = colorbar; cb2.Label.String = 'Uncertainty';

%% Quantile-binned neural means vs uncertainty
varsY = {'meanAct_gr','logGV_gr','normGV_gr'};
Xvars = {'unc_perceptual','unc_decision'};

% how many quantile bins?
nq = 10;   % e.g., sextiles; change to taste (5–9 usually fine)

figure('Name','Binned means (neural) vs quantiles of uncertainty','Color','w','Position',[50 50 1300 700]);
tiledlayout(numel(Xvars), numel(varsY), 'TileSpacing','compact','Padding','compact');

for i = 1:numel(Xvars)
    x_all = TrialTbl_all.(Xvars{i});
    for j = 1:numel(varsY)
        y_all = TrialTbl_all.(varsY{j});
        nexttile; hold on; box off; grid on;

        % keep complete cases
        good = isfinite(x_all) & isfinite(y_all);
        x = x_all(good); 
        y = y_all(good);

        if numel(x) < max(10, 2*nq)
            title(sprintf('%s vs %s (too few pts)', varsY{j}, Xvars{i}), 'Interpreter','none');
            xlabel(strrep(Xvars{i},'_','\_')); ylabel(strrep(varsY{j},'_','\_'));
            continue
        end

        % --- Compute quantile bins on x (uncertainty) ---
        % robust edges; uniquetol handles many ties/flat regions
        qEdges = quantile(x, linspace(0,1,nq+1));
        qEdges = uniquetol(qEdges, 1e-12);
        % ensure at least 3 edges (2 bins); if heavy ties reduce bins as needed
        while numel(qEdges) < 3 && nq > 2
            nq = nq - 1;
            qEdges = quantile(x, linspace(0,1,nq+1));
            qEdges = uniquetol(qEdges, 1e-12);
        end

        [bin,~] = discretize(x, qEdges);
        nb = max(bin);

        % per-bin stats
        mX  = nan(1,nb);           % mean x in bin (for x-position)
        mY  = nan(1,nb);           % mean y in bin
        eY  = nan(1,nb);           % sem (or 95% CI) for y
        Ns  = zeros(1,nb);         % counts

        for b = 1:nb
            idx = (bin==b);
            if ~any(idx), continue; end
            xb   = x(idx); 
            yb   = y(idx);
            Ns(b) = numel(yb);
            mX(b) = mean(xb,'omitnan');
            mY(b) = mean(yb,'omitnan');
            % SEM; swap to 1.96*SEM for ~95% CI if you prefer
            eY(b) = std(yb, 'omitnan') / sqrt(max(1, Ns(b)));
        end

        % drop empty bins
        ok = isfinite(mX) & isfinite(mY) & isfinite(eY) & Ns>0;
        mX = mX(ok); mY = mY(ok); eY = eY(ok); Ns = Ns(ok);

        % plot: binned mean ± SEM with line to guide the eye
        h = errorbar(mX, mY, eY, 'o-', 'LineWidth', 1.8, 'MarkerFaceColor', 'w');
        % add light rug of raw x for context (optional)
        % plot(x, repmat(min(ylim)+0.02*range(ylim), size(x)), '.', 'Color', [0 0 0 0.08], 'MarkerSize', 6);

        % % annotate N on each point
        % for k = 1:numel(Ns)
        %     text(mX(k), mY(k), sprintf('  n=%d', Ns(k)), 'VerticalAlignment','middle', 'FontSize', 9, 'Color',[0.25 0.25 0.25]);
        % end

        % also report rank correlation on full data (monotonic trend)
        r_s = corr(x, y, 'type','Spearman', 'rows','complete');

        title(sprintf('%s vs %s (Spearman ρ=%.2f)', varsY{j}, Xvars{i}, r_s), 'Interpreter','none');
        xlabel(strrep(Xvars{i},'_','\_'));
        ylabel(strrep(varsY{j},'_','\_'));

        % tidy axes
        axis tight
        xlim([min(qEdges) max(qEdges)]);
    end
end

%%
nQ = 10; 
isVertGo  = logical(TrialTbl_all.go_is_vertical);
theta     = TrialTbl_all.theta_deg;
isVertCat = theta > 45; onBoundary = theta == 45;
goSide    = ( isVertGo &  isVertCat) | (~isVertGo & ~isVertCat);
noSide    = ~goSide;
baseMask  = ~onBoundary & isfinite(theta) & isfinite(TrialTbl_all.unc_perceptual);

plot_unc_binned(TrialTbl_all, 'unc_perceptual','logGV_gr', goSide, noSide, baseMask, nQ, 'Perceptual uncertainty','logGV (gr)');
plot_unc_binned(TrialTbl_all, 'unc_decision','logGV_gr',   goSide, noSide, baseMask & isfinite(TrialTbl_all.unc_decision), nQ, 'Decision uncertainty','logGV (gr)');

function plot_unc_binned(TrialTbl_all, xname,yname,goMask,noMask,valid,nQ,xlab,ylab)
    x = TrialTbl_all.(xname); y = TrialTbl_all.(yname);
    good = valid & isfinite(x) & isfinite(y);
    e = quantile(x(good), linspace(0,1,nQ+1)); e(end)=e(end)+eps;
    [~,~,b] = histcounts(x, e);
    xc = accumarray(b(good), x(good), [], @mean, NaN);

    figure('Color','w','Position',[80 80 520 420]); hold on; grid on; box off;
    [m1,s1] = agg(y(goMask & good), b(goMask & good), nQ);
    [m2,s2] = agg(y(noMask & good), b(noMask & good), nQ);
    errorbar(xc, m1, s1, '-o','LineWidth',2,'MarkerFaceColor','w','DisplayName','GO side');
    errorbar(xc, m2, s2, '-s','LineWidth',2,'MarkerFaceColor','w','DisplayName','NO-GO side');
    xlabel(xlab); ylabel(ylab); legend('Location','best'); title(sprintf('%s → %s',xname,yname),'Interpreter','none');
end
function [m,se]=agg(vals,bins,nQ)
    m = nan(nQ,1); se = nan(nQ,1);
    for k=1:nQ
        v = vals(bins==k); v = v(isfinite(v));
        if ~isempty(v), m(k)=mean(v); se(k)=std(v)/sqrt(numel(v)); end
    end
end

%%
pairs = {
    'unc_perceptual','logGV_gr'
    'unc_decision','logGV_gr'
    'unc_perceptual','meanAct_gr'
    'unc_decision','meanAct_gr'
};
U = unique(TrialTbl_all.animal);
res = array2table(nan(numel(U), size(pairs,1)+1), 'VariableNames', ['animal', compose("b_%d",1:size(pairs,1))]);
res.animal = U(:);

for a = 1:numel(U)
    Ta = TrialTbl_all(strcmp(TrialTbl_all.animal,U{a}),:);
    for p = 1:size(pairs,1)
        x = Ta.(pairs{p,1}); y = Ta.(pairs{p,2});
        good = isfinite(x) & isfinite(y);
        if nnz(good) > 10
            b = robustfit(x(good), y(good)); % [intercept; slope]
            res{a,1+p} = b(2);
        end
    end
end

figure('Color','w'); 
for p=1:size(pairs,1)
    subplot(2,2,p); hold on; grid on; box off;
    bar(res{:,1+p}); plot(xlim, [0 0], 'k:'); 
    title(sprintf('Slope: %s → %s', pairs{p,1}, pairs{p,2}), 'Interpreter','none');
    xticks(1:numel(U)); xticklabels(U); xtickangle(30);
end


%% 4) 2D heatmaps across contrast × dispersion (mean uncertainty)
C = TrialTbl_all.contrast; D = TrialTbl_all.dispersion;

% choose sensible binning
cEdges = unique(prctile(C, [0 25 50 75 100]));  % quartiles
dEdges = unique(prctile(D, [0 25 50 75 100]));

% map to bins
[cBin,~] = discretize(C, cEdges);
[dBin,~] = discretize(D, dEdges);

% accumulator for means
fMean = @(v) mean(v,'omitnan');

% perceptual
M_p = accumarray([cBin dBin], TrialTbl_all.unc_perceptual, [numel(cEdges)-1 numel(dEdges)-1], fMean, NaN);
% decision
M_d = accumarray([cBin dBin], TrialTbl_all.unc_decision,   [numel(cEdges)-1 numel(dEdges)-1], fMean, NaN);

figure('Color','w','Position',[100 100 900 380]); 
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile; 
imagesc(M_p'); axis image; colorbar; 
title('Perceptual unc. (mean)'); 
xlabel('Contrast bin'); ylabel('Dispersion bin'); 
set(gca,'YDir','normal','XTick',1:size(M_p,1),'YTick',1:size(M_p,2));

nexttile; 
imagesc(M_d'); axis image; colorbar; 
title('Decision unc. (mean)'); 
xlabel('Contrast bin'); ylabel('Dispersion bin'); 
set(gca,'YDir','normal','XTick',1:size(M_d,1),'YTick',1:size(M_d,2));


%% 5) Choice-conditioned uncertainties (Correct vs Incorrect)
is_correct = TrialTbl_all.performance==1;

g = categorical(is_correct, [false true], {'Incorrect','Correct'});

mp = splitapply(@mean, TrialTbl_all.unc_perceptual, findgroups(g));
sp = splitapply(@(v) std(v,'omitnan')./sqrt(sum(isfinite(v))), TrialTbl_all.unc_perceptual, findgroups(g));

md = splitapply(@mean, TrialTbl_all.unc_decision, findgroups(g));
sd = splitapply(@(v) std(v,'omitnan')./sqrt(sum(isfinite(v))), TrialTbl_all.unc_decision, findgroups(g));

figure('Color','w','Position',[100 100 680 360]); 
hold on;
x = 1:2;
errorbar(x-0.12, mp, sp, 'o-','LineWidth',1.5, 'DisplayName','Perceptual'); 
yyaxis('right')
errorbar(x+0.12, md, sd, 'o-','LineWidth',1.5, 'DisplayName','Decision'); 
xticks(x); xticklabels(categories(g)); ylabel('Uncertainty'); 
title('Uncertainty by accuracy'); grid on; box on; legend('Location','best');

 
%% 3) Curves vs *relative* stimulus orientations (means ± SEM at each unique angle)

s = TrialTbl_all.stimulus(:);
uP = TrialTbl_all.unc_perceptual(:);
uD = TrialTbl_all.unc_decision(:);

% group by exact orientation values (sort to keep x increasing)
[ori_unique, ~, gid] = unique(s);
[ori_unique, sortIdx] = sort(ori_unique);

% helper lambdas for summary stats
mfun  = @(v) mean(v,'omitnan');
semfun= @(v) std(v,'omitnan')./sqrt(sum(isfinite(v)));

% perceptual
mP   = splitapply(mfun,  uP, gid);
seP  = splitapply(semfun,uP, gid);

% decision
mD   = splitapply(mfun,  uD, gid);
seD  = splitapply(semfun,uD, gid);

% sort stats by orientation
mP = mP(sortIdx);  seP = seP(sortIdx);
mD = mD(sortIdx);  seD = seD(sortIdx);

figure('Color','w','Position',[100 100 1000 380]); 
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

% --- Perceptual
nexttile; hold on;
% (optional) raw trials as faint points
scatter(s, uP, 6, 'filled', 'MarkerFaceAlpha',0.08, 'MarkerEdgeAlpha',0.08);
errorbar(ori_unique, mP, seP, '-o','LineWidth',1.3,'MarkerSize',4);
        xlabel('Δ from go (deg)');
        ylabel('Perceptual uncertainty');
title('Perceptual vs orientation'); grid on; box on; xlim([min(s) max(s)]);

% --- Decision
nexttile; hold on;
scatter(s, uD, 6, 'filled', 'MarkerFaceAlpha',0.08, 'MarkerEdgeAlpha',0.08);
errorbar(ori_unique, mD, seD, '-o','LineWidth',1.3,'MarkerSize',4);
        xlabel('Δ from go (deg)');
 ylabel('Decision uncertainty');
title('Decision vs orientation'); grid on; box on; xlim([min(s) max(s)]);

%% Curves vs contrast (means ± SEM at each unique contrast)
c  = round(TrialTbl_all.contrast(:), 3);        % round to avoid fp duplicates
uP = TrialTbl_all.unc_perceptual(:);
uD = TrialTbl_all.unc_decision(:);

[cu, ~, gid] = unique(c);
[cu, ord]    = sort(cu);

mfun   = @(v) mean(v,'omitnan');
semfun = @(v) std(v,'omitnan')./sqrt(sum(isfinite(v)));

mP = splitapply(mfun,   uP, gid);  seP = splitapply(semfun, uP, gid);
mD = splitapply(mfun,   uD, gid);  seD = splitapply(semfun, uD, gid);

mP = mP(ord); seP = seP(ord);
mD = mD(ord); seD = seD(ord);

figure('Color','w','Position',[100 100 1000 380]); 
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

% --- Perceptual vs contrast
nexttile; hold on;
scatter(c, uP, 6, 'filled', 'MarkerFaceAlpha',0.08, 'MarkerEdgeAlpha',0.08);
errorbar(cu, mP, seP, '-o','LineWidth',1.3,'MarkerSize',4);
xlabel('Contrast'); ylabel('Perceptual uncertainty');
title('Perceptual vs contrast'); grid on; box on; xlim([min(c) max(c)]);

% --- Decision vs contrast
nexttile; hold on;
scatter(c, uD, 6, 'filled', 'MarkerFaceAlpha',0.08, 'MarkerEdgeAlpha',0.08);
errorbar(cu, mD, seD, '-o','LineWidth',1.3,'MarkerSize',4);
xlabel('Contrast'); ylabel('Decision uncertainty');
title('Decision vs contrast'); grid on; box on; xlim([min(c) max(c)]);

%% Curves vs dispersion (means ± SEM at each unique dispersion)
d  = round(TrialTbl_all.dispersion(:), 3);      % round to avoid fp duplicates
uP = TrialTbl_all.unc_perceptual(:);
uD = TrialTbl_all.unc_decision(:);

[du, ~, gid] = unique(d);
[du, ord]    = sort(du);

mfun   = @(v) mean(v,'omitnan');
semfun = @(v) std(v,'omitnan')./sqrt(sum(isfinite(v)));

mP = splitapply(mfun,   uP, gid);  seP = splitapply(semfun, uP, gid);
mD = splitapply(mfun,   uD, gid);  seD = splitapply(semfun, uD, gid);

mP = mP(ord); seP = seP(ord);
mD = mD(ord); seD = seD(ord);

figure('Color','w','Position',[100 100 1000 380]); 
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

% --- Perceptual vs dispersion
nexttile; hold on;
scatter(d, uP, 6, 'filled', 'MarkerFaceAlpha',0.08, 'MarkerEdgeAlpha',0.08);
errorbar(du, mP, seP, '-o','LineWidth',1.3,'MarkerSize',4);
xlabel('Dispersion (deg)'); ylabel('Perceptual uncertainty');
title('Perceptual vs dispersion'); grid on; box on; xlim([min(d) max(d)]);

% --- Decision vs dispersion
nexttile; hold on;
scatter(d, uD, 6, 'filled', 'MarkerFaceAlpha',0.08, 'MarkerEdgeAlpha',0.08);
errorbar(du, mD, seD, '-o','LineWidth',1.3,'MarkerSize',4);
xlabel('Dispersion (deg)'); ylabel('Decision uncertainty');
title('Decision vs dispersion'); grid on; box on; xlim([min(d) max(d)]);

%% --- Plot: Disentangling Neural Correlates of Uncertainty (Corrected) ---
fprintf('\n--- Plotting: 2D Neural Heatmaps of Uncertainty ---\n');

% Check for necessary columns
required_cols = {'unc_perceptual', 'unc_decision', 'meanAct_gr', 'logGV_gr'};
if ~all(ismember(required_cols, TrialTbl_all.Properties.VariableNames))
    missing = required_cols(~ismember(required_cols, TrialTbl_all.Properties.VariableNames));
    warning('Cannot create Plot 5: Missing one or more required columns: %s', strjoin(missing, ', '));
    return;
end

figure('Color','w','Position',[100 100 1000 500]); % Position for 1x2 layout
sgtitle('Neural Activity vs. Inferred Uncertainty');

n_bins_2d = 8; % Use 8x8 grid for the heatmap

% --- Get all data ---
x_data = TrialTbl_all.unc_perceptual;
y_data = TrialTbl_all.unc_decision;
z_data_mean = TrialTbl_all.meanAct_gr;
z_data_gv = TrialTbl_all.logGV_gr;

% --- Create ONE valid mask for all data ---
valid_mask = ~isnan(x_data) & ~isnan(y_data) & ~isnan(z_data_mean) & ~isnan(z_data_gv);

if sum(valid_mask) < (n_bins_2d * n_bins_2d) % Check if we have enough data
    warning('Not enough valid (non-NaN) data points to create 2D heatmaps.');
    return;
end

% --- Filter data to only valid points ---
x_clean = x_data(valid_mask);
y_clean = y_data(valid_mask);
z_clean_mean = z_data_mean(valid_mask);
z_clean_gv = z_data_gv(valid_mask);

% --- 1. Define Quantile Bins ---
% Get edges from 0th to 100th percentile
x_edges = quantile(x_clean, linspace(0, 1, n_bins_2d + 1));
y_edges = quantile(y_clean, linspace(0, 1, n_bins_2d + 1));

% Make edges unique (handles cases with low variance where quantiles collapse)
x_edges = uniquetol(x_edges, 1e-9);
y_edges = uniquetol(y_edges, 1e-9);

% Get actual number of bins created
n_bins_x = numel(x_edges) - 1;
n_bins_y = numel(y_edges) - 1;

if n_bins_x <= 0 || n_bins_y <= 0
    warning('Could not create valid bins (data might be constant). Skipping heatmaps.');
    return;
end

% --- 2. Get Bin Indices ---
% Get bin index for each data point *from the clean data*
[~, ~, x_bin] = histcounts(x_clean, x_edges);
[~, ~, y_bin] = histcounts(y_clean, y_edges);

% --- 3. Create Heatmaps ---

% Filter out any 0-indices (from data outside the exact edges, though rare)
bin_mask = x_bin > 0 & y_bin > 0;
if ~any(bin_mask)
    warning('No data points fell inside the 2D bins.');
    return;
end

% --- 3a. Mean Activity vs. Uncertainties ---
nexttile;

% Use only the points that fell in a valid bin
M_mean = accumarray([y_bin(bin_mask), x_bin(bin_mask)], z_clean_mean(bin_mask), ...
                    [n_bins_y, n_bins_x], @(v) mean(v, 'omitnan'), NaN);

% Plot
imagesc(M_mean);
set(gca, 'YDir', 'normal'); % Put Q1 (y_bin=1) at the bottom
title('Mean Activity (Grating)');
xlabel('Perceptual Uncertainty (Quantile Bin)');
ylabel('Decision Uncertainty (Quantile Bin)');

% Create x/y tick labels (e.g., 'Q1', 'Q2'...)
x_tick_labels = arrayfun(@(i) sprintf('Q%d', i), 1:n_bins_x, 'Uni', 0);
y_tick_labels = arrayfun(@(i) sprintf('Q%d', i), 1:n_bins_y, 'Uni', 0);
xticks(1:n_bins_x); xticklabels(x_tick_labels);
yticks(1:n_bins_y); yticklabels(y_tick_labels);

colorbar;
axis square;

% --- 3b. Log GV vs. Uncertainties ---
nexttile;

M_gv = accumarray([y_bin(bin_mask), x_bin(bin_mask)], z_clean_gv(bin_mask), ...
                  [n_bins_y, n_bins_x], @(v) mean(v, 'omitnan'), NaN);

% Plot
imagesc(M_gv);
set(gca, 'YDir', 'normal');
title('log(Generalized Variance) (Grating)');
xlabel('Perceptual Uncertainty (Quantile Bin)');
ylabel('Decision Uncertainty (Quantile Bin)');
xticks(1:n_bins_x); xticklabels(x_tick_labels);
yticks(1:n_bins_y); yticklabels(y_tick_labels);

colorbar;
axis square;

%% --- Integrated Multi-Animal Plotting ---
%
% This script replaces the various plotting sections of the main analysis
% script with three consolidated, publication-style figures.
%
% Assumes the following are in the workspace:
%   - TrialTbl_all: The master table with behavior, neural, and IO metrics.
%   - NeuralStore: The struct array with per-animal spike tensors.
%   - PerAnimalRaw: The struct array with raw UD data (for pupil).
%   - Animals: The initial animal definition struct.
%

fprintf('\n=== Part 4: Integrated Multi-Animal Plotting ===\n');

% --- Setup: Define helper functions and common variables ---
if ~exist('TrialTbl_all', 'var')
    error('Workspace is missing TrialTbl_all. Please run processing first.');
end

% Helper functions for aggregation
mean_fun = @(x) mean(x, 'omitnan');
sem_fun  = @(x) std(x, 0, 'omitnan') / sqrt(max(1, sum(~isnan(x))));
n_animals = numel(Animals);

% Add pupil data to the main table for easier plotting
% (This logic is borrowed from your GLM-HMM integration script)
if ~ismember('pupil_grating', TrialTbl_all.Properties.VariableNames)
    fprintf('Extracting pupil data from PerAnimalRaw...\n');
    TrialTbl_all.pupil_grating = nan(height(TrialTbl_all), 1);
    TrialTbl_all.pupil_z = nan(height(TrialTbl_all), 1);
    unique_animals = unique(TrialTbl_all.animal);
    
    for i = 1:numel(unique_animals)
        ani = unique_animals{i};
        raw_idx = find(strcmp({PerAnimalRaw.tag}, ani));
        if isempty(raw_idx), continue; end
        
        UD = PerAnimalRaw(raw_idx).UD;
        neur_rows = find(strcmp(TrialTbl_all.animal, ani));
        
        % Extract
        pupil_vals = nan(numel(UD), 1);
        for k = 1:numel(UD)
            if isfield(UD(k), 'pupil') && isfield(UD(k).pupil, 'grating') && ~isempty(UD(k).pupil.grating)
                pupil_vals(k) = mean(UD(k).pupil.grating, 'omitnan');
            end
        end
        
        % Assign (assuming row alignment)
        if numel(neur_rows) == numel(pupil_vals)
            TrialTbl_all.pupil_grating(neur_rows) = pupil_vals;
            
            % Z-score within animal
            valid = ~isnan(pupil_vals);
            if any(valid)
                mu = mean(pupil_vals, 'omitnan'); sig = std(pupil_vals, 'omitnan');
                if sig > 0
                    TrialTbl_all.pupil_z(neur_rows) = (pupil_vals - mu) / sig;
                else
                    TrialTbl_all.pupil_z(neur_rows) = 0;
                end
            end
        end
    end
end


% --- Figure 1: Group-Level Behavioral Summary ---

fprintf('Generating Figure 1: Behavior...\n');
fig1 = figure('Name', 'Figure 1: Group Behavioral Summary', 'Color', 'w', 'Position', [100 100 1400 900]);
tl1 = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- 1A: Group Psychometric (Choice) ---
ax = nexttile(tl1); hold(ax, 'on');
T = TrialTbl_all;
T.abs_from_go_binned = round(T.abs_from_go, 6); % Bin by exact stimulus value
all_x_bins = unique(T.abs_from_go_binned(~isnan(T.abs_from_go_binned)));
all_animal_curves = cell(n_animals, 1);
[unique_animals, ~, animal_group_idx] = unique(T.animal);

for i = 1:n_animals
    T_animal = T(animal_group_idx == i, :);
    valid_animal = ~isnan(T_animal.abs_from_go_binned) & ~isnan(T_animal.goChoice);
    if sum(valid_animal) < 10, continue; end
    
    [means, x_bins_str] = grpstats(T_animal.goChoice(valid_animal), T_animal.abs_from_go_binned(valid_animal), {'mean', 'gname'});
    x_bins_num = str2double(x_bins_str);
    
    plot(ax, x_bins_num', means', 'Color', [0.8 0.8 0.8], 'LineWidth', 1.0);
    all_animal_curves{i} = interp1(x_bins_num', means', all_x_bins', 'linear');
end
animal_matrix = vertcat(all_animal_curves{:});
avg_all_animals = mean(animal_matrix, 1, 'omitnan');
sem_all_animals = std(animal_matrix, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(animal_matrix), 1));
h_avg_ani = shadedErrorBar(all_x_bins, avg_all_animals, sem_all_animals, 'lineprops', {'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'k', 'MarkerSize', 4});

title(ax, 'A: Psychometric (Choice)');
xlabel(ax, '|\Delta from Go| (deg)'); ylabel(ax, 'P(Go)');
grid(ax, 'on'); box(ax, 'on');
ylim(ax, [0 1]); xlim(ax, [min(all_x_bins)-1, max(all_x_bins)+1]);
legend(h_avg_ani.mainLine, {'Group Mean \pm SEM'}, 'Location', 'best');

% --- 1B: Choice vs. Contrast ---
ax = nexttile(tl1); hold(ax, 'on');
T_con_valid = T(~isnan(T.abs_from_go_binned) & ~isnan(T.goChoice) & ~isnan(T.contrast), :);
T_con_valid.contrast_binned = round(T_con_valid.contrast, 3);
contrast_levels = unique(T_con_valid.contrast_binned);
n_contrasts = numel(contrast_levels);
colors_con = parula(n_contrasts + 2);
colors_con = colors_con(2:end-1, :);
h_con = []; leg_con = {};

for i_c = 1:n_contrasts
    T_con_level = T_con_valid(T_con_valid.contrast_binned == contrast_levels(i_c), :);
    if height(T_con_level) < 10, continue; end
    
    [means, x_bins_str] = grpstats(T_con_level.goChoice, T_con_level.abs_from_go_binned, {'mean', 'gname'});
    x_bins_num = str2double(x_bins_str);
    
    h = plot(ax, x_bins_num', means', 'o-', 'Color', colors_con(i_c, :), ...
             'LineWidth', 2, 'MarkerFaceColor', 'w');
    h_con(end+1) = h;
    leg_con{end+1} = sprintf('C = %.0f%%', contrast_levels(i_c) * 100);
end
title(ax, 'B: Psychometric by Contrast');
xlabel(ax, '|\Delta from Go| (deg)'); ylabel(ax, 'P(Go)');
if ~isempty(h_con), legend(h_con, leg_con, 'Location', 'best'); end
grid(ax, 'on'); box(ax, 'on');
ylim(ax, [0 1]); xlim(ax, [min(all_x_bins)-1, max(all_x_bins)+1]);

% --- 1C: Choice vs. Dispersion ---
ax = nexttile(tl1); hold(ax, 'on');
T_disp_valid = T(~isnan(T.abs_from_go_binned) & ~isnan(T.goChoice) & ~isnan(T.dispersion), :);
T_disp_valid.dispersion_binned = round(T_disp_valid.dispersion);
disp_levels = unique(T_disp_valid.dispersion_binned);
n_disps = numel(disp_levels);
colors_disp = hot(n_disps + 2);
colors_disp = colors_disp(1:end-2, :);
h_disp = []; leg_disp = {};

for i_d = 1:n_disps
    T_disp_level = T_disp_valid(T_disp_valid.dispersion_binned == disp_levels(i_d), :);
    if height(T_disp_level) < 10, continue; end
    
    [means, x_bins_str] = grpstats(T_disp_level.goChoice, T_disp_level.abs_from_go_binned, {'mean', 'gname'});
    x_bins_num = str2double(x_bins_str);
    
    h = plot(ax, x_bins_num', means', 's-', 'Color', colors_disp(i_d, :), ...
             'LineWidth', 2, 'MarkerFaceColor', 'w');
    h_disp(end+1) = h;
    leg_disp{end+1} = sprintf('D = %d\circ', disp_levels(i_d));
end
title(ax, 'C: Psychometric by Dispersion');
xlabel(ax, '|\Delta from Go| (deg)'); ylabel(ax, 'P(Go)');
if ~isempty(h_disp), legend(h_disp, leg_disp, 'Location', 'best'); end
grid(ax, 'on'); box(ax, 'on');
ylim(ax, [0 1]); xlim(ax, [min(all_x_bins)-1, max(all_x_bins)+1]);

% --- 1D: Lick Rate vs. Stimulus ---
ax = nexttile(tl1); hold(ax, 'on');
valid_mask_lick = ~isnan(T.abs_from_go_binned) & ~isnan(T.preRZ_lick_rate);
if sum(valid_mask_lick) > 10
    [lick_rate_means, lick_rate_sems, x_bins_str] = grpstats(T.preRZ_lick_rate(valid_mask_lick), T.abs_from_go_binned(valid_mask_lick), {'mean', 'sem', 'gname'});
    x_bins_lick = str2double(x_bins_str);
    shadedErrorBar(x_bins_lick, lick_rate_means, lick_rate_sems, ...
        'lineprops', {'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'w'});
    title(ax, 'D: Pre-RZ Lick Rate vs. Stimulus');
    xlabel(ax, '|\Delta from Go| (deg)');
    ylabel(ax, 'Pre-RZ Lick Rate (licks/s)');
    grid(ax, 'on'); box(ax, 'on');
    xlim(ax, [min(x_bins_lick)-1, max(x_bins_lick)+1]);
end

% --- 1E: Velocity vs. Stimulus ---
ax = nexttile(tl1); hold(ax, 'on');
valid_mask_vel = ~isnan(T.abs_from_go_binned) & ~isnan(T.preRZ_velocity);
if sum(valid_mask_vel) > 10
    [vel_means, vel_sems, x_bins_str] = grpstats(T.preRZ_velocity(valid_mask_vel), T.abs_from_go_binned(valid_mask_vel), {'mean', 'sem', 'gname'});
    x_bins_vel = str2double(x_bins_str);
    shadedErrorBar(x_bins_vel, vel_means, vel_sems, ...
        'lineprops', {'r-s', 'LineWidth', 2, 'MarkerFaceColor', 'w'});
    title(ax, 'E: Pre-RZ Velocity vs. Stimulus');
    xlabel(ax, '|\Delta from Go| (deg)');
    ylabel(ax, 'Pre-RZ Velocity (cm/s)');
    grid(ax, 'on'); box(ax, 'on');
    xlim(ax, [min(x_bins_vel)-1, max(x_bins_vel)+1]);
end

% --- 1F: Performance vs. Lick Rate (Go/No-Go Split) ---
ax = nexttile(tl1); hold(ax, 'on');
isVertGo   = logical(T.go_is_vertical);
theta      = T.theta_deg;
isVertCat  = theta > 45;
onBoundary = theta == 45;
goSideMask    = ( isVertGo  &  isVertCat) | (~isVertGo  & ~isVertCat);
noGoSideMask  = ~goSideMask;
validMask     = ~onBoundary & ~isnan(theta) & ~isnan(isVertGo);

plot_split_quantiles_impl(ax, T.preRZ_lick_rate, double(T.performance), ...
    goSideMask & validMask, noGoSideMask & validMask, ...
    'Pre-RZ Lick Rate (Quantiles)', 'F: Lick Rate vs. Performance');
ylabel(ax, 'P(Correct)');


% --- Figure 2: Group-Level Neural Activity (Stimulus & Pupil) ---

fprintf('Generating Figure 2: Neural Activity...\n');
fig2 = figure('Name', 'Figure 2: Group Neural Summary (Stimulus & Pupil)', 'Color', 'w', 'Position', [150 150 1400 900]);
tl2 = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- 2A: Z-Activity vs. Contrast ---
ax = nexttile(tl2); hold(ax, 'on');
% This plot is generated by plot_neural_errorbars_group_z, 
% we replicate the logic for the 'contrast' panel here.
[contrast_cells, cons] = collect_z_activity(NeuralStore, 'contrast', timeWindowG);
my_errorbar_plot(contrast_cells);
xticklabels(ax, composeCon(cons)); xtickangle(ax, 0);
ylabel(ax, 'Z-Activity (Mean \pm SEM)'); title(ax, 'A: Activity vs. Contrast');
yline(ax, 0, 'k:'); box(ax, 'off'); grid(ax, 'on');

% --- 2B: Z-Activity vs. Dispersion ---
ax = nexttile(tl2); hold(ax, 'on');
[dispersion_cells, disps] = collect_z_activity(NeuralStore, 'dispersion', timeWindowG);
my_errorbar_plot(dispersion_cells);
xticklabels(ax, composeDisp(disps));
ylabel(ax, 'Z-Activity'); title(ax, 'B: Activity vs. Dispersion');
yline(ax, 0, 'k:'); box(ax, 'off'); grid(ax, 'on');

% --- 2C: Z-Activity vs. Orientation (C=1, D=5) ---
ax = nexttile(tl2); hold(ax, 'on');
[orient_cells_c1d5, oris] = collect_z_activity(NeuralStore, 'orientation_c1d5', timeWindowG);
my_errorbar_plot(orient_cells_c1d5);
xticklabels(ax, composeOri(oris));
xlabel(ax, 'Orientation (deg)'); ylabel(ax, 'Z-Activity');
title(ax, 'C: Activity vs. Orientation (C=1, D=5)');
yline(ax, 0, 'k:'); box(ax, 'off'); grid(ax, 'on');

% --- 2D: Group Heatmap: Mean Activity (C x D) ---
ax = nexttile(tl2);
plot_group_heatmap_z(ax, NeuralStore, timeWindowG, 'mean_z');
title(ax, 'D: Mean Z-Activity (C x D)');

% --- 2E: Group Heatmap: log(GV) (C x D) ---
ax = nexttile(tl2);
plot_group_heatmap_z(ax, NeuralStore, timeWindowG, 'loggv_z');
title(ax, 'E: log(GV) (Z-scored, C x D)');

% --- 2F: Group Heatmap: Pupil (C x D) ---
ax = nexttile(tl2);
plot_pupil_heatmap_contrast_dispersion(ax, PerAnimalRaw, 'grating');
title(ax, 'F: Pupil Amplitude (Grating, C x D)');


% --- Figure 3: Neural & Pupil Correlates of Uncertainty ---

fprintf('Generating Figure 3: Uncertainty...\n');
fig3 = figure('Name', 'Figure 3: Neural & Pupil Correlates of Uncertainty', 'Color', 'w', 'Position', [200 200 1400 900]);
tl3 = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

nQ = 8; % Number of quantiles

% --- 3A: Mean Activity vs. Perceptual Uncertainty ---
ax = nexttile(tl3); hold(ax, 'on');
plot_quantile_relationship(ax, T, 'unc_perceptual', 'meanAct_gr', nQ);
title(ax, 'A: Mean Activity vs. Perceptual Unc.');

% --- 3B: log(GV) vs. Perceptual Uncertainty ---
ax = nexttile(tl3); hold(ax, 'on');
plot_quantile_relationship(ax, T, 'unc_perceptual', 'logGV_gr', nQ);
title(ax, 'B: log(GV) vs. Perceptual Unc.');

% --- 3C: Pupil vs. Perceptual Uncertainty ---
ax = nexttile(tl3); hold(ax, 'on');
plot_quantile_relationship(ax, T, 'unc_perceptual', 'pupil_z', nQ);
title(ax, 'C: Pupil (Z) vs. Perceptual Unc.');

% --- 3D: Mean Activity vs. Decision Uncertainty ---
ax = nexttile(tl3); hold(ax, 'on');
plot_quantile_relationship(ax, T, 'unc_decision', 'meanAct_gr', nQ);
title(ax, 'D: Mean Activity vs. Decision Unc.');

% --- 3E: log(GV) vs. Decision Uncertainty ---
ax = nexttile(tl3); hold(ax, 'on');
plot_quantile_relationship(ax, T, 'unc_decision', 'logGV_gr', nQ);
title(ax, 'E: log(GV) vs. Decision Unc.');

% --- 3F: Pupil vs. Decision Uncertainty ---
ax = nexttile(tl3); hold(ax, 'on');
plot_quantile_relationship(ax, T, 'unc_decision', 'pupil_z', nQ);
title(ax, 'F: Pupil (Z) vs. Decision Unc.');

fprintf('\n--- All plotting complete. ---\n');


%% --- Local Helper Functions for Integrated Plotting ---

function plot_split_quantiles_impl(ax, x_all, y_all, m_go, m_nogo, xlab_str, ttl_str)
    % (Modified to plot onto a specific axes 'ax')
    
    % GO side
    x = x_all(m_go); y = y_all(m_go);
    [mY_go, sY_go, mX_go] = bin_by_quantile(x, y, 9);
    h1 = errorbar(ax, mX_go, mY_go, sY_go, '-o', 'LineWidth', 2, 'MarkerFaceColor','w', ...
        'DisplayName','GO side');

    % NO-GO side
    x = x_all(m_nogo); y = y_all(m_nogo);
    [mY_ng, sY_ng, mX_ng] = bin_by_quantile(x, y, 9);
    h2 = errorbar(ax, mX_ng, mY_ng, sY_ng, '-s', 'LineWidth', 2, 'MarkerFaceColor','w', ...
        'DisplayName','NO-GO side');

    % Correlations (trial-wise)
    [r_go, p_go]   = corr(x_all(m_go),   y_all(m_go),   'rows','complete','type','Pearson');
    [r_ng, p_ng]   = corr(x_all(m_nogo), y_all(m_nogo), 'rows','complete','type','Pearson');

    xlabel(ax, xlab_str);
    title(ax, sprintf('%s\n(r_{GO}=%.2f, p=%.2g; r_{NO}=%.2f, p=%.2g)', ...
        ttl_str, r_go, p_go, r_ng, p_ng), 'Interpreter', 'none');
    legend(ax, [h1 h2], 'Location','best'); yline(ax, 0.5,'k:');
    grid(ax, 'on'); box(ax, 'on');
end

function [collected_cells, levels] = collect_z_activity(NeuralStore, bin_by, timeWindowG)
    % Collects per-animal mean Z-activity binned by a specified variable
    
    % ----- Collect global levels -----
    allC = []; allD = []; allO = [];
    for a = 1:numel(NeuralStore)
        allC = [allC; NeuralStore(a).contrast(:)];
        allD = [allD; NeuralStore(a).dispersion(:)];
        allO = [allO; NeuralStore(a).stimulus(:)];
    end
    allC = round(allC,3); allC(abs(allC-1)<=0.02) = 1.0;
    cons  = unique(allC);
    disps = unique(allD);
    oris  = unique(allO);
    
    % ----- Build per-animal per-trial z-activity (grating) -----
    perAnimal = struct([]);
    for a = 1:numel(NeuralStore)
        tag  = NeuralStore(a).tag;
        xG   = NeuralStore(a).xG;
        Gspk = NeuralStore(a).Gspk; % [nTrials x nNeurons x tBins]
        if isempty(Gspk), continue; end
        tmask = xG >= timeWindowG(1) & xG <= timeWindowG(2);
        if ~any(tmask), continue; end

        X = squeeze(mean(Gspk(:,:,tmask),3,'omitnan')); % trials × neurons
        mu = mean(X,1,'omitnan');
        sd = std(X,0,1,'omitnan'); sd(sd==0 | isnan(sd)) = 1;
        Z = (X - mu) ./ sd;
        zTrial = mean(Z,2,'omitnan');

        C = NeuralStore(a).contrast(:);
        C = round(C,3); C(abs(C-1)<=0.02) = 1.0;
        D = NeuralStore(a).dispersion(:);
        O = NeuralStore(a).stimulus(:);

        perAnimal(a).tag = tag;
        perAnimal(a).z   = zTrial;
        perAnimal(a).C   = C;
        perAnimal(a).D   = D;
        perAnimal(a).O   = O;
    end
    
    % ----- Binning Logic -----
    switch bin_by
        case 'contrast'
            levels = cons;
            collected_cells = cell(numel(levels),1);
            for i = 1:numel(levels)
                acc = [];
                for a = 1:numel(perAnimal)
                    if ~isfield(perAnimal(a),'z'), acc = [acc; NaN]; continue; end
                    idx = perAnimal(a).C==levels(i);
                    if any(idx), acc = [acc; mean(perAnimal(a).z(idx),'omitnan')];
                    else, acc = [acc; NaN]; end
                end
                collected_cells{i} = acc;
            end
            
        case 'dispersion'
            levels = disps;
            collected_cells = cell(numel(levels),1);
            for i = 1:numel(levels)
                acc = [];
                for a = 1:numel(perAnimal)
                    if ~isfield(perAnimal(a),'z'), acc = [acc; NaN]; continue; end
                    idx = perAnimal(a).D==levels(i);
                    if any(idx), acc = [acc; mean(perAnimal(a).z(idx),'omitnan')];
                    else, acc = [acc; NaN]; end
                end
                collected_cells{i} = acc;
            end

        case 'orientation_c1d5'
            levels = oris;
            targetC = 1.0; targetD = 5;
            collected_cells = cell(numel(levels),1);
            for i = 1:numel(levels)
                acc = [];
                for a = 1:numel(perAnimal)
                    if ~isfield(perAnimal(a),'z'), acc = [acc; NaN]; continue; end
                    idx = (abs(perAnimal(a).C - targetC) < 1e-6) & ...
                          (perAnimal(a).D==targetD) & ...
                          (perAnimal(a).O==levels(i));
                    if any(idx), acc = [acc; mean(perAnimal(a).z(idx),'omitnan')];
                    else, acc = [acc; NaN]; end
                end
                collected_cells{i} = acc;
            end
    end
end


function plot_group_heatmap_z(ax, NeuralStore, timeWindowG, metric)
    % (Modified to plot onto a specific axes 'ax')
    % metric: 'mean_z' or 'loggv_z'
    
    nA = numel(NeuralStore);
    allCons = []; allDisps = [];
    for a = 1:nA
        allCons  = [allCons;  NeuralStore(a).contrast(:)];
        allDisps = [allDisps; NeuralStore(a).dispersion(:)];
    end
    allCons = round(allCons,3); allCons(abs(allCons-1)<=0.02) = 1.0;
    cons  = unique(allCons);
    disps = unique(allDisps);
    nC = numel(cons); nD = numel(disps);
    mats = nan(nC, nD, nA);

    for a = 1:nA
        xG   = NeuralStore(a).xG;
        Gspk = NeuralStore(a).Gspk;
        if isempty(Gspk), continue; end
        tmask = xG >= timeWindowG(1) & xG <= timeWindowG(2);
        if ~any(tmask), continue; end
        
        switch metric
            case 'mean_z'
                X = squeeze(mean(Gspk(:,:,tmask),3,'omitnan')); % trials × neurons
                mu = mean(X,1,'omitnan');
                sd = std(X,0,1,'omitnan'); sd(sd==0 | isnan(sd)) = 1;
                Z = (X - mu) ./ sd;
                trial_metric = mean(Z,2,'omitnan'); % trials × 1
            case 'loggv_z'
                S = compute_gv_and_trace(Gspk(:,:,tmask));
                trial_metric = S.gv_log;
                % z-score logGV within animal
                trial_metric = (trial_metric - mean(trial_metric, 'omitnan')) / std(trial_metric, 0, 'omitnan');
        end

        cA = NeuralStore(a).contrast(:);
        dA = NeuralStore(a).dispersion(:);
        cA = round(cA,3); cA(abs(cA-1)<=0.02) = 1.0;
        M = nan(nC, nD);
        for ic = 1:nC
            for id = 1:nD
                idx = (cA==cons(ic)) & (dA==disps(id));
                if any(idx), M(ic,id) = mean(trial_metric(idx), 'omitnan'); end
            end
        end
        mats(:,:,a) = M;
    end
    
    G = mean(mats, 3, 'omitnan'); % Average across animals
    
    imagesc(ax, 1:nD, 1:nC, G);
    set(ax,'YDir','normal'); colormap(ax, parula);
    set(get(ax,'Children'),'AlphaData',~isnan(G));
    xticks(ax, 1:nD); xticklabels(ax, composeDisp(disps));
    yticks(ax, 1:nC); yticklabels(ax, composeCon(cons));
    xlabel(ax, 'Dispersion'); ylabel(ax, 'Contrast'); colorbar(ax);
end

function plot_pupil_heatmap_contrast_dispersion(ax, PerAnimalRaw, epoch)
    % (Modified to plot onto a specific axes 'ax')
    % epoch: 'grating' or 'corridor'
    
    rows = [];
    for a = 1:numel(PerAnimalRaw)
        UD = PerAnimalRaw(a).UD;
        if isempty(UD), continue; end
        for tr = 1:numel(UD)
            val = NaN;
            if strcmp(epoch, 'grating') && isfield(UD(tr),'pupil') && isfield(UD(tr).pupil,'grating') && ~isempty(UD(tr).pupil.grating)
                val = mean(UD(tr).pupil.grating, 'omitnan');
            elseif strcmp(epoch, 'corridor') && isfield(UD(tr),'pupil') && isfield(UD(tr).pupil,'corridor') && ~isempty(UD(tr).pupil.corridor)
                val = mean(UD(tr).pupil.corridor, 'omitnan');
            end
            if isnan(val), continue; end

            con  = getfield(UD(tr),'contrast');
            disp = getfield(UD(tr),'dispersion');
            con = round(con,3);
            if abs(con-1) <= 0.02, con = 1.0; end
            rows = [rows; struct('con',con,'disp',disp,'val',val)];
        end
    end
    if isempty(rows), title(ax, 'No Pupil Data'); return; end

    cons  = unique([rows.con]);
    disps = unique([rows.disp]);
    M = nan(numel(cons), numel(disps));
    for ic = 1:numel(cons)
        for id = 1:numel(disps)
            mask = [rows.con]==cons(ic) & [rows.disp]==disps(id);
            if any(mask)
                vals = [rows(mask).val];
                M(ic,id) = mean(vals,'omitnan');
            end
        end
    end
    
    imagesc(ax, 1:numel(disps), 1:numel(cons), M);
    set(ax,'YDir','normal'); colormap(ax, parula);
    set(get(ax,'Children'),'AlphaData',~isnan(M));
    xticks(ax, 1:numel(disps)); xticklabels(ax, composeDisp(disps));
    yticks(ax, 1:numel(cons));  yticklabels(ax, composeCon(cons));
    xlabel(ax, 'Dispersion'); ylabel(ax, 'Contrast'); colorbar(ax);
end

function plot_quantile_relationship(ax, T, x_var, y_var, nQ)
    % (Plots a quantile-binned relationship onto 'ax')
    
    x_all = T.(x_var);
    y_all = T.(y_var);
    
    good = isfinite(x_all) & isfinite(y_all);
    if sum(good) < nQ * 5
        title(ax, sprintf('%s vs %s\n(Insufficient Data)', y_var, x_var), 'Interpreter', 'none');
        return;
    end
    
    x = x_all(good); 
    y = y_all(good);

    qEdges = quantile(x, linspace(0,1,nQ+1));
    qEdges = uniquetol(qEdges, 1e-12);
    if numel(qEdges) < 3, qEdges = linspace(min(x), max(x), nQ+1); end
    qEdges(end) = qEdges(end) + eps;
    
    [bin,~] = discretize(x, qEdges);
    nb = max(bin);
    
    mX  = accumarray(bin, x, [nb 1], @mean, NaN);
    mY  = accumarray(bin, y, [nb 1], @mean, NaN);
    sY  = accumarray(bin, y, [nb 1], @(v) std(v,'omitnan')/sqrt(max(1, sum(~isnan(v)))), NaN);
    
    ok = isfinite(mX) & isfinite(mY) & isfinite(sY);
    
    shadedErrorBar(mX(ok), mY(ok), sY(ok), 'lineprops', {'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'w'});
    
    [r_s, p_s] = corr(x, y, 'type','Spearman', 'rows','complete');
    
    title(ax, sprintf('%s vs. %s\n(Spearman \rho=%.2f, p=%.2g)', y_var, x_var, r_s, p_s), 'Interpreter', 'none');
    xlabel(ax, sprintf('%s (Quantile Binned)', x_var), 'Interpreter', 'none');
    ylabel(ax, y_var, 'Interpreter', 'none');
    grid(ax, 'on'); box(ax, 'on');
    axis(ax, 'tight');
end

% (You will also need the other helper functions from your original script,
%  e.g., shadedErrorBar, composeCon, composeDisp, composeOri, bin_by_quantile,
%  sem_omit, etc. I've included the most critical ones here.)

% ============================ Helpers ===================================


function [xfit, yfit, fit, stats] = fit_sigmoid_counts(x, y, varargin)
% Robust sigmoid fit for count-like data (e.g., preRZ_licks).
% Fits a 4PL or Richards (5PL) *mean* curve using WLS + Huber/epsilon-insensitive loss,
% with bounded parameters and multiple inits. Optional log1p-space fitting.
%
% Usage:
%   [xfit,yfit,fit,stats] = fit_sigmoid_counts(x, y, 'useRichards', true, 'useLog1p', true);
%
% Options:
%   'useRichards'  (false)   -> allow 5th shape parameter nu (Richards curve)
%   'useLog1p'     (true)    -> fit log1p(y); back-transform for plotting
%   'xgrid'        ([])      -> default = linspace(min(x),max(x),200)
%   'binIfManyX'   (true)    -> if many unique x with sparse reps, pre-aggregate to means/vars
%   'huberK'       (1.345)   -> Huber threshold
%   'epsVar'       (1e-6)    -> variance floor
%   'nRestarts'    (5)       -> random restarts around initials
%
% Returns:
%   fit.params = [A B C D (nu?)]
%   fit.form   = '4PL' or 'Richards'
%   stats.sse, stats.r2, stats.ok

    ip = inputParser;
    addParameter(ip,'useRichards',false);
    addParameter(ip,'useLog1p',true);
    addParameter(ip,'xgrid',[]);
    addParameter(ip,'binIfManyX',true);
    addParameter(ip,'huberK',1.345);
    addParameter(ip,'epsVar',1e-6);
    addParameter(ip,'nRestarts',5);
    parse(ip,varargin{:});
    Opt = ip.Results;

    x = x(:); y = y(:);
    good = ~isnan(x) & ~isnan(y);
    x = x(good); y = y(good);

    if isempty(x)
        xfit=[]; yfit=[]; fit=struct('form','none'); stats=struct('ok',false); return;
    end
    if isempty(Opt.xgrid)
        xr = [min(x) max(x)]; if xr(1)==xr(2), xr = xr + [-0.5 0.5]; end
        Opt.xgrid = linspace(xr(1), xr(2), 200);
    end

    % --- Optionally transform to stabilize variance ---
    if Opt.useLog1p
        y_work = log1p(y);
        back   = @(z) max(0, exp(z) - 1);
    else
        y_work = y;
        back   = @(z) z;
    end

    % --- Optionally aggregate at identical x to get means & variance (for WLS) ---
    xw = x; yw = y_work;
    if Opt.binIfManyX
        [ux,~,ix] = unique(round(x,6));
        m  = accumarray(ix, yw, [], @(v) mean(v,'omitnan'));
        v  = accumarray(ix, yw, [], @(v) var(v,'omitnan'));
        n  = accumarray(ix, 1);
        xw = ux; yw = m;
        % Poisson-ish variance grows with mean; combine sample var and mean-based prior
        v  = coalesceVar(v, (abs(m)+1).^1.0);     % simple mean-variance proxy
        w  = n ./ max(Opt.epsVar, v);             % inverse-variance weights
    else
        % per-point weights ~ 1/max(eps, var proxy)
        v  = (abs(yw)+1).^1.0;
        w  = 1 ./ max(Opt.epsVar, v);
        % keep identity xw,yw
    end

    % --- Choose model form ---
    use5 = Opt.useRichards;
    if use5
        f = @(p,xx) richards(xx, p);   % p=[A B C D nu]
        pinits = initials(yw, xw, 5);
        lb = [min(yw)-range(yw), max(min(yw)-range(yw), -50), min(xw),  1e-3, 0.25];
        ub = [max(yw)+range(yw), min(max(yw)+range(yw),  50), max(xw), 5*range(xw), 4];
    else
        f = @(p,xx) fourpl(xx, p);     % p=[A B C D]
        pinits = initials(yw, xw, 4);
        lb = [min(yw)-range(yw), max(min(yw)-range(yw), -50), min(xw),  1e-3];
        ub = [max(yw)+range(yw), min(max(yw)+range(yw),  50), max(xw), 5*range(xw)];
    end

    % --- Robust WLS objective with Huber loss ---
    obj = @(p) robust_wls(yw, f(p, xw), w, Opt.huberK);

    % --- Optimize with bounds; multi-start ---
    [pBest, fBest] = deal([] , inf);
    for r = 1:max(1, Opt.nRestarts)
        p0 = pinits;
        % jitter C and D a bit
        j  = [0 0 0.1*range(xw) 0.1*range(xw) 0.2];
        j  = j(1:numel(p0));
        p0 = p0 + (randn(size(p0)).*j);
        p0 = clamp(p0, lb, ub);

        try
            if exist('fminsearchbnd','file')==2
                pr = fminsearchbnd(obj, p0, lb, ub, optimset('Display','off','TolX',1e-8,'MaxIter',5e3));
            else
                % poor man's bounds
                tr  = @(q) lb + (ub-lb)./(1+exp(-q));
                inv = @(p) log((p-lb)./(ub-p+eps));
                pr  = fminsearch(@(q) obj(tr(q)), inv(p0), optimset('Display','off','TolX',1e-8,'MaxIter',5e3));
                pr  = tr(pr);
            end
            fr = obj(pr);
            if fr < fBest
                pBest = pr; fBest = fr;
            end
        catch
        end
    end
    if isempty(pBest)
        % very last resort: linear slope in x
        pBest = pinits; fBest = obj(pBest);
    end

    % --- Outputs ---
    yhat_w   = f(pBest, xw);
    sse      = nansum((yw - yhat_w).^2);
    sst      = nansum((yw - mean(yw,'omitnan')).^2);
    r2       = max(0, 1 - sse/max(eps,sst));

    xfit = Opt.xgrid(:);
    yfit = back(f(pBest, xfit));      % back-transform to original count space
    fit.params = pBest; fit.form = tern(use5,'Richards','4PL');
    stats.sse = sse; stats.r2 = r2; stats.ok = isfinite(r2) & (r2>=0);

    % also provide a convenience function handle to evaluate on arbitrary x
    fit.predict = @(xx) back(f(pBest, xx));
end

% ==== helpers ====
function y = fourpl(x,p)
% y = A + (B-A)/(1+exp(-(x-C)/D))
A=p(1); B=p(2); C=p(3); D=p(4); D = sign(D)*max(1e-6,abs(D));
y = A + (B-A) ./ (1 + exp(-(x-C)./D));
end

function y = richards(x,p)
% y = A + (B-A) / (1 + exp(-(x-C)/D))^nu
A=p(1); B=p(2); C=p(3); D=p(4); nu=p(5);
D = sign(D)*max(1e-6,abs(D)); nu = max(0.1,nu);
y = A + (B-A) ./ (1 + exp(-(x-C)./D)).^nu;
end

function val = robust_wls(y, yhat, w, k)
% Huber loss with weights w (inverse-variance), returns scalar objective
r = y - yhat;
s = mad0(y); if ~isfinite(s) || s<=0, s = max(1e-3, std(y,'omitnan')); end
u = r./max(1e-6, s);
hub = (abs(u) <= k) .* (0.5*(u.^2)) + (abs(u) > k) .* (k*abs(u) - 0.5*k^2);
val = nansum(w(:) .* hub(:));
if ~isfinite(val), val = 1e9; end
end

function s = mad0(y), s = 1.4826*median(abs(y - median(y,'omitnan')),'omitnan'); end
function r = range(v), r = max(v)-min(v); end
function p = clamp(p,lb,ub), p = min(max(p,lb),ub); end
function v = coalesceVar(samp, prior), v = nanmax(samp, prior); end
function t = tern(c,a,b)
if c, t=a;
else, t=b;
end 
end

function p = initials(y,x,dim)
    A0 = prctile(y,10); B0 = prctile(y,90);
    C0 = median(x);     D0 = max(1, range(x)/6);
    if dim==5, p=[A0 B0 C0 D0 1]; else, p=[A0 B0 C0 D0]; end
end

function merged = merge_levels(vals, tol)
vals = sort(vals(:));
if isempty(vals), merged = vals; return; end
clusters = {vals(1)};
for i = 2:numel(vals)
    if abs(vals(i)-clusters{end}(end)) <= tol
        clusters{end}(end+1) = vals(i); %#ok<AGROW>
    else
        clusters{end+1} = vals(i); %#ok<AGROW>
    end
end
merged = zeros(numel(clusters),1);
for k = 1:numel(clusters)
    if any(clusters{k} >= 0.98), merged(k) = 1.0;
    else, merged(k) = mean(clusters{k}); end
end
merged = unique(round(merged,3));
end

function snapped = snap_to_levels(x, levels)
snapped = x;
for i = 1:numel(x)
    [~,j] = min(abs(levels - x(i)));
    snapped(i) = levels(j);
end
end


function [GrandMean, GrandSEM] = pool_metrics_across_animals(AnimalSummaries, MetricNames, all_orients, all_contrasts, all_disp)
nA = numel(AnimalSummaries);
for m = 1:numel(MetricNames)
    GrandMean.(MetricNames{m}) = NaN(numel(all_orients), numel(all_contrasts), numel(all_disp));
    GrandSEM.(MetricNames{m})  = NaN(numel(all_orients), numel(all_contrasts), numel(all_disp));
end

% Assemble [Ori x Con x Disp x Animal]
for m = 1:numel(MetricNames)
    X = nan(numel(all_orients), numel(all_contrasts), numel(all_disp), nA);
    for a = 1:nA
        if ~isfield(AnimalSummaries(a),'CondMean'), continue; end
        CM = AnimalSummaries(a).CondMean;
        lv = AnimalSummaries(a).levels;
        for io = 1:numel(lv.orients)
            oi = find(all_orients==lv.orients(io));
            for ic = 1:numel(lv.contrasts)
                [~,ci] = min(abs(all_contrasts - lv.contrasts(ic))); % snap
                for id = 1:numel(lv.dispersions)
                    di = find(all_disp==lv.dispersions(id));
                    val = CM.(MetricNames{m}).mean(io,ic,id);
                    if ~isnan(val), X(oi,ci,di,a) = val; end
                end
            end
        end
    end
    GrandMean.(MetricNames{m}) = mean(X, 4, 'omitnan');
    nEff = sum(~isnan(X), 4);
    GrandSEM.(MetricNames{m})  = std(X,0,4,'omitnan') ./ sqrt(max(1,nEff));
end
end


function plot_preRZ_sigmoid_perAnimal_exact(T, Animals, useSigned)
% Plots per-animal preRZ_licks vs |Δ from go| (default) or signed Δ,
% with robust sigmoid fit (fit_sigmoid_counts above).
%
% useSigned = false (default) → |Δ from go|
% useSigned = true            → signed Δ from go

    if nargin<3, useSigned = false; end

    figure('Name','preRZ licks (per animal, robust sigmoid)', 'Position',[100 100 1100 550]);
    nA = numel(Animals);
    nCols = 3; nRows = ceil(nA/nCols);
    tl = tiledlayout(nRows, nCols, 'TileSpacing','compact','Padding','compact');

    for a = 1:nA
        nexttile; hold on; grid on; box off;
        tag = Animals(a).tag;
        Ta  = T(strcmp(T.animal, tag), :);
        if isempty(Ta), title(tag); continue; end

        x = round(useSigned .* Ta.theta_from_go + (~useSigned).*Ta.abs_from_go, 6);
        y = Ta.preRZ_licks;

        % group means & SE for points
        [ux,~,ix] = unique(x(~isnan(x)));
        m  = accumarray(ix, y(~isnan(x)), [], @(v) mean(v,'omitnan'));
        se = accumarray(ix, y(~isnan(x)), [], @(v) std(v,'omitnan')/sqrt(nnz(~isnan(v))), []);
        errorbar(ux, m, se, 'ko', 'MarkerFaceColor','k');

        % robust sigmoid fit (counts)
        try
            [xfit, yfit, fit, stats] = fit_sigmoid_counts(x, y, ...
                'useRichards', true, 'useLog1p', true, 'binIfManyX', true);
            plot(xfit, yfit, 'r-', 'LineWidth', 2);
            ttl = sprintf('%s  (%s, R^2=%.2f)', tag, fit.form, stats.r2);
        catch
            ttl = sprintf('%s  (fit failed)', tag);
        end

        xlabel(sprintf('%sΔ from go (deg)', tern(useSigned,'Signed ','')));
        ylabel('Pre-RZ licks (count)');
        title(ttl, 'Interpreter','none');
    end

    sgtitle(tl, 'Pre-RZ licks with robust sigmoid fits (per animal)', 'Interpreter','none');
end


function plot_neural_errorbars_group_z(NeuralStore, timeWindowG, varargin)
% Four panels using my_errorbar_plot:
% (1) Across CONTRASTS (merged 0.99→1), averaging over dispersions & orientations
% (2) Across DISPERSIONS, averaging over contrasts & orientations
% (3) Across ORIENTATIONS, **restricted to** contrast==1 and dispersion==5
% (4) Across ORIENTATIONS, averaged over contrasts & dispersions
%
% Options:
%   'target_contrast'   (default 1.0; also matches 0.99)
%   'target_dispersion' (default 5)
%   'orientations'      (default: unique true orientations present across animals)

    ip = inputParser;
    addParameter(ip,'target_contrast',1.0);
    addParameter(ip,'target_dispersion',5);
    addParameter(ip,'orientations',[]);
    parse(ip,varargin{:});
    targetC = ip.Results.target_contrast;
    targetD = ip.Results.target_dispersion;

    % ----- Collect global levels -----
    allC = []; allD = []; allO = [];
    for a = 1:numel(NeuralStore)
        allC = [allC; NeuralStore(a).contrast(:)];
        allD = [allD; NeuralStore(a).dispersion(:)];
        allO = [allO; NeuralStore(a).stimulus(:)];
    end
    allC = round(allC,3); allC(abs(allC-1)<=0.02) = 1.0;
    cons  = unique(allC);
    disps = unique(allD);
    oris  = unique(allO);
    if isempty(ip.Results.orientations), ORI = oris; else, ORI = ip.Results.orientations(:); end

    % ----- Build per-animal per-trial z-activity (grating) -----
    perAnimal = struct([]);
    for a = 1:numel(NeuralStore)
        tag  = NeuralStore(a).tag;
        xG   = NeuralStore(a).xG;
        Gspk = NeuralStore(a).Gspk; % [nTrials x nNeurons x tBins]
        if isempty(Gspk), continue; end
        tmask = xG >= timeWindowG(1) & xG <= timeWindowG(2);
        if ~any(tmask), continue; end

        X = squeeze(mean(Gspk(:,:,tmask),3,'omitnan')); % trials × neurons
        mu = mean(X,1,'omitnan');
        sd = std(X,0,1,'omitnan'); sd(sd==0 | isnan(sd)) = 1;
        Z = (X - mu) ./ sd;
        zTrial = mean(Z,2,'omitnan');

        C = NeuralStore(a).contrast(:);
        C = round(C,3); C(abs(C-1)<=0.02) = 1.0;
        D = NeuralStore(a).dispersion(:);
        O = NeuralStore(a).stimulus(:);

        perAnimal(a).tag = tag;
        perAnimal(a).z   = zTrial;
        perAnimal(a).C   = C;
        perAnimal(a).D   = D;
        perAnimal(a).O   = O;
    end

    % Helper to build cell vectors per level (each cell = values across animals)
    getAcrossAnimals = @(vals_per_animal) vals_per_animal(~cellfun(@isempty,vals_per_animal));

    % ----- (1) by CONTRAST -----
    contrast_cells = cell(numel(cons),1);
    for ic = 1:numel(cons)
        acc = [];
        for a = 1:numel(perAnimal)
            if ~isfield(perAnimal(a),'z'), continue; end
            idx = perAnimal(a).C==cons(ic);
            if any(idx)
                acc = [acc; mean(perAnimal(a).z(idx),'omitnan')];
            else
                acc = [acc; NaN];
            end
        end
        contrast_cells{ic} = acc; % vector (animals)
    end

    % ----- (2) by DISPERSION -----
    dispersion_cells = cell(numel(disps),1);
    for id = 1:numel(disps)
        acc = [];
        for a = 1:numel(perAnimal)
            if ~isfield(perAnimal(a),'z'), continue; end
            idx = perAnimal(a).D==disps(id);
            if any(idx)
                acc = [acc; mean(perAnimal(a).z(idx),'omitnan')];
            else
                acc = [acc; NaN];
            end
        end
        dispersion_cells{id} = acc;
    end

    % ----- (3) by ORIENTATION at C=1 & D=targetD -----
    orient_cells_c1d = cell(numel(ORI),1);
    for io = 1:numel(ORI)
        acc = [];
        for a = 1:numel(perAnimal)
            if ~isfield(perAnimal(a),'z'), continue; end
            idx = (abs(perAnimal(a).C - targetC) < 1e-6) & (perAnimal(a).D==targetD) & (perAnimal(a).O==ORI(io));
            if any(idx)
                acc = [acc; mean(perAnimal(a).z(idx),'omitnan')];
            else
                acc = [acc; NaN];
            end
        end
        orient_cells_c1d{io} = acc;
    end

    % ----- (4) by ORIENTATION averaged across C & D -----
    orient_cells_all = cell(numel(ORI),1);
    for io = 1:numel(ORI)
        acc = [];
        for a = 1:numel(perAnimal)
            if ~isfield(perAnimal(a),'z'), continue; end
            idx = (perAnimal(a).O==ORI(io));
            if any(idx)
                acc = [acc; mean(perAnimal(a).z(idx),'omitnan')];
            else
                acc = [acc; NaN];
            end
        end
        orient_cells_all{io} = acc;
    end

    % ----- Plot (my_errorbar_plot) -----
    figure('Name','Neural activity (z, per animal → across animals)','Position',[100 100 1300 600]);
    tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

    % Panel 1: by contrast
    nexttile; hold on;
    my_errorbar_plot(contrast_cells);
    xticklabels(composeCon(cons)); xtickangle(0);
    ylabel('Z-activity (mean across neurons)'); title('By contrast (avg over disp & ori)');
    yline(0,'k:'); box off;

    % Panel 2: by dispersion
    nexttile; hold on;
    my_errorbar_plot(dispersion_cells);
    xticklabels(composeDisp(disps));
    ylabel('Z-activity'); title('By dispersion (avg over contrast & ori)');
    yline(0,'k:'); box off;

    % Panel 3: by orientation @ C=1 & D=targetD
    nexttile; hold on;
    my_errorbar_plot(orient_cells_c1d);
    xticklabels(composeOri(ORI));
    xlabel('Orientation (deg)'); ylabel('Z-activity');
    title(sprintf('By orientation (C=1, D=%g)', targetD));
    yline(0,'k:'); box off;

    % Panel 4: by orientation (avg over C & D)
    nexttile; hold on;
    my_errorbar_plot(orient_cells_all);
    xticklabels(composeOri(ORI));
    xlabel('Orientation (deg)'); ylabel('Z-activity');
    title('By orientation (avg over contrast & dispersion)');
    yline(0,'k:'); box off;

    sgtitle(tl, 'Across-animal neural activity (per-animal per-neuron z-scored, grating)', 'Interpreter','none');
end

function plot_activity_heatmap_group_z(NeuralStore, timeWindowG)
% Group-level heatmap after z-scoring within each animal, within each neuron.
% - Uses grating activity from NeuralStore(a).Gspk and NeuralStore(a).xG
% - timeWindowG: [t0 t1] seconds for grating window (same as you used upstream)

    nA = numel(NeuralStore);
    if nA==0, error('NeuralStore is empty.'); end

    % Collect global sets of contrast/dispersion across animals to build a common grid
    allCons = []; allDisps = [];
    for a = 1:nA
        allCons  = [allCons;  NeuralStore(a).contrast(:)];
        allDisps = [allDisps; NeuralStore(a).dispersion(:)];
    end

    % Merge ~1.0 to 1.0 (robust)
    allCons = round(allCons,3); allCons(abs(allCons-1)<=0.02) = 1.0;
    cons  = unique(allCons);
    disps = unique(allDisps);
    nC = numel(cons); nD = numel(disps);

    % Per-animal heatmaps to average
    mats = nan(nC, nD, nA);

    for a = 1:nA
        xG   = NeuralStore(a).xG;            % time vector (s)
        Gspk = NeuralStore(a).Gspk;          % [nTrials x nNeurons x tBins]
        if isempty(Gspk), warning('%s: empty Gspk', NeuralStore(a).tag); continue; end

        % time mask for grating window
        tmask = xG >= timeWindowG(1) & xG <= timeWindowG(2);
        if ~any(tmask), warning('%s: no grating samples in window', NeuralStore(a).tag); continue; end

        nT = size(Gspk,1); nN = size(Gspk,2);

        % Per-trial, per-neuron mean in window -> [nTrials x nNeurons]
        X = squeeze(mean(Gspk(:,:,tmask), 3, 'omitnan'));  % trials × neurons

        % Z-score within each neuron across trials
        mu = mean(X, 1, 'omitnan');
        sd = std(X, 0, 1, 'omitnan');
        sd(sd==0 | isnan(sd)) = 1; % avoid div-by-zero
        Z = (X - mu) ./ sd;        % trials × neurons

        % Collapse neurons -> per-trial z-activity
        zTrial = mean(Z, 2, 'omitnan');      % trials × 1

        % Build this animal's Contrast×Dispersion matrix
        cA = NeuralStore(a).contrast(:);
        dA = NeuralStore(a).dispersion(:);
        cA = round(cA,3); cA(abs(cA-1)<=0.02) = 1.0;

        M = nan(nC, nD);
        for ic = 1:nC
            for id = 1:nD
                idx = (cA==cons(ic)) & (dA==disps(id));
                if any(idx), M(ic,id) = mean(zTrial(idx), 'omitnan'); end
            end
        end
        mats(:,:,a) = M;
    end

    % Average across animals (random effects via simple mean of matrices)
    G = mean(mats, 3, 'omitnan');

    % Plot
    figure('Name','Group heatmap (z-scored per animal/neuron, grating)', 'Position', [100 100 520 460]);
    imagesc(1:nD, 1:nC, G);
    set(gca,'YDir','normal'); colormap(parula);
    set(get(gca,'Children'),'AlphaData',~isnan(G));
    xticks(1:nD); xticklabels(composeDisp(disps));
    yticks(1:nC); yticklabels(composeCon(cons));
    xlabel('Dispersion'); ylabel('Contrast');
    title('Group (z within animal&neuron) — Grating'); colorbar;
end

function out = composeCon(v)
    if max(v) <= 1.0001
        out = arrayfun(@(x) sprintf('%.0f%%', 100*x), v, 'uni', 0);
    else
        out = arrayfun(@(x) sprintf('%.3g', x), v, 'uni', 0);
    end
end

function out = composeDisp(v)
    out = arrayfun(@(x) sprintf('%.3g', x), v, 'uni', 0);
end

function out = composeOri(v),  out = arrayfun(@(x) sprintf('%.3g', x), v, 'uni', 0); end


function [mean_y, sem_y, mean_x] = bin_by_quantile(x, y, nQ)
good = ~isnan(x) & ~isnan(y);
x = x(good); y = y(good);
if isempty(x), mean_y=[]; sem_y=[]; mean_x=[]; return; end
q_edges = quantile(x, linspace(0,1,nQ+1)); q_edges(end) = q_edges(end)+1e-9;
[~,~,bin] = histcounts(x, q_edges);
mean_y = accumarray(bin, y, [], @nanmean);
sem_y  = accumarray(bin, y, [], @(v) nanstd(v)/sqrt(sum(~isnan(v))));
mean_x = accumarray(bin, x, [], @nanmean);
end




function plot_activity_heatmaps_per_animal(T, Animals, fieldName)
% Plot per-animal heatmaps of neural activity averaged across orientations,
% arranged as (contrasts × dispersions).
%
% T: TrialTbl_all with columns: animal, contrast, dispersion, and fieldName (e.g., 'meanAct_gr')
% Animals: struct with .tag
% fieldName: 'meanAct_gr' (or 'meanAct_co', 'logGV_gr', etc.)

    assert(ismember(fieldName, T.Properties.VariableNames), 'Field %s not in table.', fieldName);

    % Ensure 0.99 gets merged to 1.0 (you already normalize upstream, but be safe)
    c = round(T.contrast, 3);
    c(abs(c-1) <= 0.02) = 1.0;
    T.contrast = c;

    cons = unique(T.contrast);
    disps = unique(T.dispersion);
    nC = numel(cons); nD = numel(disps);

    figure('Name', ['Per-animal heatmaps: ' fieldName], 'Position', [100 100 1200 600]);
    nA = numel(Animals);
    nCols = 3; nRows = ceil(nA/nCols);
    tl = tiledlayout(nRows, nCols, 'TileSpacing','compact','Padding','compact');

    for a = 1:nA
        nexttile; hold on;
        tag = Animals(a).tag;
        Ta  = T(strcmp(T.animal, tag), :);
        if isempty(Ta)
            title([tag ' (no data)']); axis off; continue;
        end

        % Fill matrix [nC x nD] with means (omit NaN)
        M = nan(nC, nD);
        for ic = 1:nC
            for id = 1:nD
                idx = Ta.contrast==cons(ic) & Ta.dispersion==disps(id);
                if any(idx)
                    M(ic,id) = mean(Ta{idx, fieldName}, 'omitnan');
                end
            end
        end

        % Heatmap with NaN transparency
        imagesc(1:nD, 1:nC, M);
        set(gca,'YDir','normal');
        colormap(parula);
        h = gca;
        % make NaNs transparent
        set(get(h,'Children'),'AlphaData',~isnan(M));

        xticks(1:nD); xticklabels(composeDisp(disps));
        yticks(1:nC); yticklabels(composeCon(cons));
        xlabel('Dispersion'); ylabel('Contrast');
        title(tag, 'Interpreter','none'); colorbar;
    end

    sgtitle(tl, ['Per-animal heatmaps (avg across orientations): ' fieldName], 'Interpreter','none');
end


function plot_psy_goAligned_perAnimal(T, Animals, metric, ylab)
% Plot psychometrics per animal using the actual |Δ from go| levels (no binning).
% - For binary 'goChoice': fits a logistic GLM on trials: P(go) ~ |Δ|
% - For continuous metrics (e.g., preRZ_velocity/licks): fits a 4-PL to means
%
% Inputs:
%   T        : master trial table with fields:
%              - animal (string/cellstr)
%              - abs_from_go (numeric)
%              - goChoice (binary) and/or another continuous metric
%   Animals  : struct array with .tag (animal IDs)
%   metric   : 'goChoice' or name of a continuous column in T
%   ylab     : y-axis label

    figure('Name',['Psychometric (per animal exact, aligned to go): ' metric], ...
           'Position',[100 100 1100 550]);
    nA = numel(Animals);
    nCols = 3;
    nRows = ceil(nA / nCols);
    tl = tiledlayout(nRows, nCols, 'TileSpacing','compact','Padding','compact');

    for a = 1:nA
        nexttile; hold on; grid on; box off;
        tag = Animals(a).tag;

        % subset this animal
        Ta = T(strcmp(T.animal, tag), :);
        if isempty(Ta)
            title(tag); 
            continue;
        end

        % x and y
        x = round(Ta.abs_from_go, 6);                  % protect against FP noise
        switch metric
            case 'goChoice', y = Ta.goChoice;
            otherwise
                if ~ismember(metric, T.Properties.VariableNames)
                    warning('Metric "%s" not found for %s.', metric, tag);
                    title(tag); 
                    continue;
                end
                y = Ta.(metric);
        end

        % unique x-levels actually present
        ux = unique(x(~isnan(x)));
        if isempty(ux)
            title(sprintf('%s (no data)', tag));
            continue;
        end

        % per-level mean & SE
        m  = arrayfun(@(v) mean(y(x==v),'omitnan'), ux);
        se = arrayfun(@(v) std(y(x==v),'omitnan')/sqrt(max(1,nnz(~isnan(y(x==v))))), ux);

        % points with error bars
        errorbar(ux, m, se, 'ko', 'MarkerFaceColor','k');

        % fits
        isBinary = strcmp(metric,'goChoice');

        if isBinary
            good = ~isnan(y) & ~isnan(x);
            if nnz(good) >= 4 && numel(unique(x(good))) >= 2
                % logistic regression on trials
                b = glmfit(x(good), y(good), 'binomial', 'link', 'logit');
                xfit = linspace(min(ux), max(ux), 200);
                yfit = 1 ./ (1 + exp(-(b(1) + b(2)*xfit)));
                plot(xfit, yfit, 'r-', 'LineWidth', 2);
            end
        else
            % 4-parameter logistic on means (requires your sigmoid4pl helper)
            gix = ~isnan(m);
            if nnz(gix) >= 3 && numel(unique(ux(gix))) >= 2
                p0  = [min(m(gix)) max(m(gix)) median(ux(gix)) 10]; % [A B C D]
                try
                    p   = fminsearch(@(p) nansum((m(gix) - sigmoid4pl(ux(gix), p)).^2), ...
                                     p0, optimset('Display','off'));
                    xfit= linspace(min(ux(gix)), max(ux(gix)), 200);
                    plot(xfit, sigmoid4pl(xfit,p), 'r-', 'LineWidth', 2);
                catch
                    % if fit fails, just show the points
                end
            end
        end

        xlabel('|Δ from go| (deg)'); ylabel(ylab);
        title(tag, 'Interpreter','none');
        xlim([min(ux)-1 max(ux)+1]); % small padding
        ylim auto
    end

    sgtitle(tl, ['Psychometric (per animal exact, aligned to go): ' metric], 'Interpreter','none');
end


function plot_psy_goAligned_group(T, metric, ylab)
% Computes psychometric using actual |Δ from go| levels present in the data (no binning).
x = round(T.abs_from_go, 6);  % protect against tiny FP noise
ux = unique(x(~isnan(x)));
switch metric
    case 'goChoice', y = T.goChoice;
    otherwise,       y = T.(metric);
end

m  = arrayfun(@(v) mean(y(x==v),'omitnan'), ux);
se = arrayfun(@(v) std(y(x==v),'omitnan')/sqrt(nnz(~isnan(y(x==v)))), ux);

figure('Name',['Psychometric (group exact, aligned to go): ' metric],'Position',[100 100 650 500]);
hold on; grid on; box off;
errorbar(ux, m, se, 'ko','MarkerFaceColor','k');

% Fit on TRIALS (preferred): logistic GLM on the raw data
isBinary = strcmp(metric,'goChoice');
if isBinary
    % logistic regression P(go) ~ |Δ|
    X = [ones(size(x)) x];         % intercept + |Δ|
    good = ~isnan(y) & ~isnan(x);
    b = glmfit(x(good), y(good), 'binomial','link','logit'); % intercept + slope
    xfit = linspace(min(ux), max(ux), 200);
    yfit = 1./(1 + exp(-(b(1)+b(2)*xfit)));
    plot(xfit, yfit, 'r-', 'LineWidth', 2);
else
    % 4-PL to means for continuous metrics
    p0 = [min(m) max(m) median(ux) 10];
    p  = fminsearch(@(p) nansum((m - sigmoid4pl(ux,p)).^2), p0, optimset('Display','off'));
    xfit= linspace(min(ux), max(ux), 200);
    plot(xfit, sigmoid4pl(xfit,p), 'r-', 'LineWidth', 2);
end

xlabel('|Δ from go| (deg)'); ylabel(ylab);
title(['Psychometric (group exact, aligned to go): ' metric], 'Interpreter','none');
end

function plot_pupil_group_by_condition(PerAnimalRaw)
% Aggregates pupil.grating/corridor traces across animals by orientation/outcome/contrast.
% Assumes each UD(trial).pupil.grating and UD(trial).pupil.corridor are row vectors
% of same length within an animal (mismatched animals are handled via nan-padding).

% Collect per-trial meta + traces
P = struct('ori',[],'con',[],'out',[],'tG',[],'yG',[],'xC',[],'yC',[]);
allP = [];

for a = 1:numel(PerAnimalRaw)
    UD = PerAnimalRaw(a).UD;
    if isempty(UD), continue; end

    % axes per animal (can differ across animals, so we carry them trial-wise)
    for tr = 1:numel(UD)
        p = struct();
        p.ori = UD(tr).stimulus;
        p.con = getfield(UD(tr), 'contrast'); %#ok<GFLD>
        if isfield(UD(tr),'outcome')
            p.out = UD(tr).outcome;
        else
            p.out = NaN;
        end

        if isfield(UD(tr),'grating') && isfield(UD(tr).grating,'timeBins')
            p.tG = UD(tr).grating.timeBins(:)';  % 1×T
        else
            p.tG = [];
        end
        if isfield(UD(tr),'pupil') && isfield(UD(tr).pupil,'grating') && ~isempty(UD(tr).pupil.grating)
            p.yG = UD(tr).pupil.grating(:)';     % 1×T
        else
            p.yG = [];
        end

        if isfield(UD(tr),'corridor') && isfield(UD(tr).corridor,'binCenters')
            p.xC = UD(tr).corridor.binCenters(:)'; % 1×S
        else
            p.xC = [];
        end
        if isfield(UD(tr),'pupil') && isfield(UD(tr).pupil,'corridor') && ~isempty(UD(tr).pupil.corridor)
            p.yC = UD(tr).pupil.corridor(:)';      % 1×S
        else
            p.yC = [];
        end
        allP = [allP; p]; %#ok<AGROW>
    end
end
if isempty(allP), warning('No pupil data found.'); return; end

% --- By orientation (grating & corridor) ---
oris = unique([allP.ori]);
figure('Name','Pupil vs Orientation'); 
subplot(1,2,1); hold on; grid on; box off; title('Grating'); 
for o = 1:numel(oris)
    Y = []; X = [];
    % collect & resample per trial to a common time vector via simple nanpad (assume equal within animal)
    for k = 1:numel(allP)
        if allP(k).ori ~= oris(o), continue; end
        if ~isempty(allP(k).yG)
            Y = padcat_row(Y, allP(k).yG);
        end
        if isempty(X) && ~isempty(allP(k).tG), X = allP(k).tG; end
    end
    if ~isempty(Y)
        m = mean(Y, 1, 'omitnan'); s = sem_omit(Y,1);
        shaded_error_if_avail(X, m, s, 'LineWidth',2);
    end
end
xlabel('Time (s)'); ylabel('Pupil (a.u.)'); legend(strsplit(num2str(oris)),'Location','best');

subplot(1,2,2); hold on; grid on; box off; title('Corridor'); 
for o = 1:numel(oris)
    Y = []; X = [];
    for k = 1:numel(allP)
        if allP(k).ori ~= oris(o), continue; end
        if ~isempty(allP(k).yC)
            Y = padcat_row(Y, allP(k).yC);
        end
        if isempty(X) && ~isempty(allP(k).xC), X = allP(k).xC; end
    end
    if ~isempty(Y)
        m = mean(Y, 1, 'omitnan'); s = sem_omit(Y,1);
        shaded_error_if_avail(X, m, s, 'LineWidth',2);
    end
end
xlabel('Position (cm)'); ylabel('Pupil (a.u.)'); legend(strsplit(num2str(oris)),'Location','best');

% --- By outcome ---
outs = intersect(unique([allP.out]), 1:4);
labs = {'Hit','Miss','FA','CR'};
figure('Name','Pupil vs Outcome'); 
subplot(1,2,1); hold on; grid on; box off; title('Grating');
for q = 1:numel(outs)
    Y = []; X = [];
    for k = 1:numel(allP)
        if allP(k).out ~= outs(q), continue; end
        if ~isempty(allP(k).yG)
            Y = padcat_row(Y, allP(k).yG);
        end
        if isempty(X) && ~isempty(allP(k).tG), X = allP(k).tG; end
    end
    if ~isempty(Y)
        m = mean(Y,1,'omitnan'); s = sem_omit(Y,1);
        shaded_error_if_avail(X, m, s, 'LineWidth',2);
    end
end
xlabel('Time (s)'); ylabel('Pupil (a.u.)'); legend(labs(outs),'Location','best');

subplot(1,2,2); hold on; grid on; box off; title('Corridor');
for q = 1:numel(outs)
    Y = []; X = [];
    for k = 1:numel(allP)
        if allP(k).out ~= outs(q), continue; end
        if ~isempty(allP(k).yC)
            Y = padcat_row(Y, allP(k).yC);
        end
        if isempty(X) && ~isempty(allP(k).xC), X = allP(k).xC; end
    end
    if ~isempty(Y)
        m = mean(Y,1,'omitnan'); s = sem_omit(Y,1);
        shaded_error_if_avail(X, m, s, 'LineWidth',2);
    end
end
xlabel('Position (cm)'); ylabel('Pupil (a.u.)'); legend(labs(outs),'Location','best');

% --- By contrast (optional; mirrors above) ---
cons = unique([allP.con]);
figure('Name','Pupil vs Contrast'); 
subplot(1,2,1); hold on; grid on; box off; title('Grating');
for c = 1:numel(cons)
    Y = []; X = [];
    for k = 1:numel(allP)
        if allP(k).con ~= cons(c), continue; end
        if ~isempty(allP(k).yG), Y = padcat_row(Y, allP(k).yG); end
        if isempty(X) && ~isempty(allP(k).tG), X = allP(k).tG; end
    end
    if ~isempty(Y)
        m = mean(Y,1,'omitnan'); s = sem_omit(Y,1);
        shaded_error_if_avail(X, m, s, 'LineWidth',2);
    end
end
xlabel('Time (s)'); ylabel('Pupil (a.u.)'); legend(arrayfun(@(v) sprintf('C=%.2f',v), cons,'uni',0),'Location','best');

subplot(1,2,2); hold on; grid on; box off; title('Corridor');
for c = 1:numel(cons)
    Y = []; X = [];
    for k = 1:numel(allP)
        if allP(k).con ~= cons(c), continue; end
        if ~isempty(allP(k).yC), Y = padcat_row(Y, allP(k).yC); end
        if isempty(X) && ~isempty(allP(k).xC), X = allP(k).xC; end
    end
    if ~isempty(Y)
        m = mean(Y,1,'omitnan'); s = sem_omit(Y,1);
        shaded_error_if_avail(X, m, s, 'LineWidth',2);
    end
end
xlabel('Position (cm)'); ylabel('Pupil (a.u.)'); legend(arrayfun(@(v) sprintf('C=%.2f',v), cons,'uni',0),'Location','best');
end

function M = padcat_row(M, row)
% append row (1×T) with NaN-padding to match existing width
row = row(:)'; % ensure row
if isempty(M), M = row; return; end
nC = size(M,2); mC = numel(row);
if mC > nC
    M = [M, nan(size(M,1), mC-nC)]; %#ok<AGROW>
elseif mC < nC
    row = [row, nan(1, nC-mC)];
end
M = [M; row]; %#ok<AGROW>
end

function shaded_error_if_avail(x, m, s, varargin)
if exist('shadedErrorBar','file')==2
    shadedErrorBar(x, m, s, 'lineprops', varargin);
else
    errorbar(x, m, s, varargin{:});
end
end


function s = sigmoid4pl(x, p)
    % y = A + (B-A)/(1+exp(-(x-C)/D))
    A = p(1); B = p(2); C = p(3); D = p(4);
    D = sign(D)*max(1e-6, abs(D));
    s = A + (B-A) ./ (1 + exp(-(x-C)./D));
end

function plot_pupil_by_orientation_C1D5_and_avg(PerAnimalRaw)
% Plots pupil (grating & corridor) across orientations in two ways:
% (A) restricted to contrast=1 & dispersion=5
% (B) averaged across all contrasts & dispersions
%
% Uses per-trial UD(tr).pupil.grating (1×T) and UD(tr).pupil.corridor (1×S)

    % --------- Collect per-trial meta + pupil traces ----------
    Trials = struct('ori',[],'con',[],'disp',[],'tG',[],'yG',[],'xC',[],'yC',[]);
    P = [];
    for a = 1:numel(PerAnimalRaw)
        UD = PerAnimalRaw(a).UD;
        if isempty(UD), continue; end
        for tr = 1:numel(UD)
            q.ori  = getfield(UD(tr),'stimulus'); %#ok<GFLD>
            q.con  = getfield(UD(tr),'contrast'); %#ok<GFLD>
            q.disp = getfield(UD(tr),'dispersion'); %#ok<GFLD>

            % normalize contrast near 1.0 → 1.0 (robust to 0.99 etc.)
            c = round(q.con,3);
            if abs(c-1) <= 0.02, c = 1.0; end
            q.con = c;

            % axes & traces
            if isfield(UD(tr),'grating') && isfield(UD(tr).grating,'timeBins')
                q.tG = UD(tr).grating.timeBins(:)';    % 1×T
            else
                q.tG = [];
            end
            if isfield(UD(tr),'pupil') && isfield(UD(tr).pupil,'grating') && ~isempty(UD(tr).pupil.grating)
                q.yG = UD(tr).pupil.grating(:)';       % 1×T
            else
                q.yG = [];
            end

            if isfield(UD(tr),'corridor') && isfield(UD(tr).corridor,'binCenters')
                q.xC = UD(tr).corridor.binCenters(:)'; % 1×S
            else
                q.xC = [];
            end
            if isfield(UD(tr),'pupil') && isfield(UD(tr).pupil,'corridor') && ~isempty(UD(tr).pupil.corridor)
                q.yC = UD(tr).pupil.corridor(:)';      % 1×S
            else
                q.yC = [];
            end

            P = [P; q]; %#ok<AGROW>
        end
    end
    if isempty(P), warning('plot_pupil_by_orientation_C1D5_and_avg: no pupil data found.'); return; end

    ORI = unique([P.ori]);
    hasC1D5 = any([P.con]==1 & [P.disp]==5);

    % ------------- (A) C=1, D=5 ----------------
    figure('Name','Pupil vs Orientation — C=1, D=5','Position',[80 80 1200 500]);
    tl = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

    % Grating
    nexttile; hold on; grid on; box off; title('Grating (C=1, D=5)');
    if ~hasC1D5, text(0.5,0.5,'No trials with C=1 & D=5','HorizontalAlignment','center'); axis off;
    else
        for o = ORI
            Y = []; X = [];
            for k = 1:numel(P)
                if P(k).ori ~= o, continue; end
                if P(k).con == 1 && P(k).disp == 5 && ~isempty(P(k).yG)
                    Y = padcat_row(Y, P(k).yG);
                    if isempty(X) && ~isempty(P(k).tG), X = P(k).tG; end
                end
            end
            if ~isempty(Y)
                m = mean(Y,1,'omitnan'); s = sem_omit(Y,1);
                shaded_error_if_avail(X, m, s, 'LineWidth',2);
            end
        end
        xlabel('Time (s)'); ylabel('Pupil (a.u.)'); legend(strsplit(num2str(ORI)),'Location','best');
    end

    % Corridor
    nexttile; hold on; grid on; box off; title('Corridor (C=1, D=5)');
    if ~hasC1D5, text(0.5,0.5,'No trials with C=1 & D=5','HorizontalAlignment','center'); axis off;
    else
        for o = ORI
            Y = []; X = [];
            for k = 1:numel(P)
                if P(k).ori ~= o, continue; end
                if P(k).con == 1 && P(k).disp == 5 && ~isempty(P(k).yC)
                    Y = padcat_row(Y, P(k).yC);
                    if isempty(X) && ~isempty(P(k).xC), X = P(k).xC; end
                end
            end
            if ~isempty(Y)
                m = mean(Y,1,'omitnan'); s = sem_omit(Y,1);
                shaded_error_if_avail(X, m, s, 'LineWidth',2);
            end
        end
        xlabel('Position (cm)'); ylabel('Pupil (a.u.)'); legend(strsplit(num2str(ORI)),'Location','best');
    end

    sgtitle(tl, 'Pupil across orientations — restricted to contrast=1, dispersion=5','Interpreter','none');

    % ------------- (B) Averaged across all contrasts & dispersions -------------
    figure('Name','Pupil vs Orientation — averaged over C & D','Position',[90 90 1200 500]);
    tl = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

    % Grating
    nexttile; hold on; grid on; box off; title('Grating (avg over C & D)');
    for o = ORI
        Y = []; X = [];
        for k = 1:numel(P)
            if P(k).ori ~= o, continue; end
            if ~isempty(P(k).yG)
                Y = padcat_row(Y, P(k).yG);
                if isempty(X) && ~isempty(P(k).tG), X = P(k).tG; end
            end
        end
        if ~isempty(Y)
            m = mean(Y,1,'omitnan'); s = sem_omit(Y,1);
            shaded_error_if_avail(X, m, s, 'LineWidth',2);
        end
    end
    xlabel('Time (s)'); ylabel('Pupil (a.u.)'); legend(strsplit(num2str(ORI)),'Location','best');

    % Corridor
    nexttile; hold on; grid on; box off; title('Corridor (avg over C & D)');
    for o = ORI
        Y = []; X = [];
        for k = 1:numel(P)
            if P(k).ori ~= o, continue; end
            if ~isempty(P(k).yC)
                Y = padcat_row(Y, P(k).yC);
                if isempty(X) && ~isempty(P(k).xC), X = P(k).xC; end
            end
        end
        if ~isempty(Y)
            m = mean(Y,1,'omitnan'); s = sem_omit(Y,1);
            shaded_error_if_avail(X, m, s, 'LineWidth',2);
        end
    end
    xlabel('Position (cm)'); ylabel('Pupil (a.u.)'); legend(strsplit(num2str(ORI)),'Location','best');

    sgtitle(tl, 'Pupil across orientations — averaged across contrasts & dispersions','Interpreter','none');
end



function [xcenters, mY_go, sY_go, mY_ng, sY_ng, r_go, p_go, r_ng, p_ng] = ...
         bin_split_global(x_all, y_all, goMask, noMask, baseValid, nQ)
    % Build a global-valid mask for this pair (exclude NaNs in x or y)
    valid = baseValid & isfinite(x_all) & isfinite(y_all);
    xv = x_all(valid); yv = y_all(valid);

    % --- Global quantile edges from ALL valid trials for this metric ---
    e = quantile(xv, linspace(0,1,nQ+1));
    e = unique(e);
    if numel(e) < nQ+1
        % fallback if many ties collapse quantiles
        e = linspace(min(xv), max(xv), nQ+1);
        e = unique(e);
    end
    e(end) = e(end) + eps(max(e)); % ensure max included
    nBins = numel(e)-1;

    % Global bin centers = mean x in each bin across ALL valid trials
    [~,~,b_all] = histcounts(xv, e);
    xcenters = accumarray(b_all, xv, [nBins 1], @mean, NaN);

    % --- GO side stats in global bins ---
    igo = goMask & valid;
    [~,~,b_go] = histcounts(x_all(igo), e);
    mY_go = nan(nBins,1); sY_go = nan(nBins,1);
    for k = 1:nBins
        yy = y_all(igo); yy = yy(b_go==k);
        if ~isempty(yy)
            mY_go(k) = mean(yy, 'omitnan');
            n = sum(~isnan(yy)); sY_go(k) = std(yy,0,'omitnan')/max(1,sqrt(n));
        end
    end

    % --- NO-GO side stats in global bins ---
    ing = noMask & valid;
    [~,~,b_ng] = histcounts(x_all(ing), e);
    mY_ng = nan(nBins,1); sY_ng = nan(nBins,1);
    for k = 1:nBins
        yy = y_all(ing); yy = yy(b_ng==k);
        if ~isempty(yy)
            mY_ng(k) = mean(yy, 'omitnan');
            n = sum(~isnan(yy)); sY_ng(k) = std(yy,0,'omitnan')/max(1,sqrt(n));
        end
    end

    % Trial-wise correlations (sanity check)
    [r_go, p_go] = corr(x_all(igo), y_all(igo), 'rows','complete');
    [r_ng, p_ng] = corr(x_all(ing), y_all(ing), 'rows','complete');
end




