%% Population Geometry Analysis: Centroid Magnitude & Generalized Variance
%
% This section computes two key geometric measures of population activity
% for each trial during the grating period:
%   1) Centroid magnitude: ||mean activity across neurons||
%   2) Generalized variance: det(Cov) or log(det(Cov + λI))
%
% These are then related to:
%   - Experimental variables: orientation, contrast, dispersion
%   - Behavioral proxy: confidence
%   - Inferred variables: perceptual uncertainty, decision uncertainty

fprintf('\n=== Population Geometry Analysis ===\n');

% --- A) Compute per-trial centroid magnitude and GV ---
% We'll use the grating period data stored in NeuralStore

for a = 1:nAnimals
    tag = NeuralStore(a).tag;
    if isempty(NeuralStore(a).Gspk), continue; end
    
    fprintf('Computing population geometry for %s...\n', tag);
    
    % Get neural data: [nTrials x nNeurons x tBins]
    Gspk = NeuralStore(a).Gspk;
    xG   = NeuralStore(a).xG;
    
    % Time window mask (same as used earlier)
    tmask = xG >= timeWindowG(1) & xG <= timeWindowG(2);
    
    % Extract grating window activity
    Gspk_window = Gspk(:, :, tmask);  % [nTrials x nNeurons x tBins_in_window]
    
    nTrials = size(Gspk_window, 1);
    nNeurons = size(Gspk_window, 2);
    
    % Initialize output arrays
    centroid_mag = nan(nTrials, 1);
    gen_var = nan(nTrials, 1);
    
    % Loop over trials
    for tr = 1:nTrials
        % Get this trial's activity: [nNeurons x tBins]
        act = squeeze(Gspk_window(tr, :, :));  
        
        % Skip if all NaN
        if all(isnan(act(:))), continue; end
        
        % --- 1) Centroid magnitude ---
        % Centroid = time-averaged activity vector across neurons
        centroid = mean(act, 2, 'omitnan');  % [nNeurons x 1]
        centroid_mag(tr) = norm(centroid);   % Euclidean norm
        
        % --- 2) Generalized Variance ---
        % We need the covariance matrix across neurons
        % Each column is a time bin, each row is a neuron
        
        % Remove neurons with all NaN for this trial
        valid_neurons = ~all(isnan(act), 2);
        act_valid = act(valid_neurons, :);
        
        if size(act_valid, 1) < 2  % Need at least 2 neurons for covariance
            continue;
        end
        
        % Compute covariance across time (neurons x neurons)
        % Center each neuron's activity
        act_centered = act_valid - mean(act_valid, 2, 'omitnan');
        
        % Handle remaining NaNs
        act_centered(isnan(act_centered)) = 0;
        
        % Covariance matrix
        C = (act_centered * act_centered') / (size(act_centered, 2) - 1);
        
        % Regularize to ensure positive definite
        lambda = 0.01;  % small regularization
        C_reg = C + lambda * eye(size(C));
        
        % Generalized variance = determinant of covariance matrix
        % Use log to avoid numerical issues
        gen_var(tr) = log(det(C_reg) + eps)/sqrt(size(act_centered, 1));  % log(det(Cov))
    end
    
    % Store in NeuralStore for this animal
    NeuralStore(a).centroid_magnitude = centroid_mag;
    NeuralStore(a).generalized_variance = gen_var;
    
    fprintf('  Computed %d trials\n', sum(isfinite(centroid_mag)));
end

% --- B) Add to TrialTbl_all ---
centroid_mag_all = [];
gen_var_all = [];

for a = 1:nAnimals
    tag = Animals(a).tag;
    animal_mask = strcmp(TrialTbl_all.animal, tag);
    n_animal_trials = sum(animal_mask);
    
    if isempty(NeuralStore) || a > numel(NeuralStore) || isempty(NeuralStore(a).centroid_magnitude)
        centroid_mag_all = [centroid_mag_all; nan(n_animal_trials, 1)];
        gen_var_all = [gen_var_all; nan(n_animal_trials, 1)];
    else
        centroid_mag_all = [centroid_mag_all; NeuralStore(a).centroid_magnitude];
        gen_var_all = [gen_var_all; NeuralStore(a).generalized_variance];
    end
end

TrialTbl_all.centroid_magnitude = centroid_mag_all;
TrialTbl_all.generalized_variance = gen_var_all;

%% C) Line plots: Population geometry vs experimental/inferred variables

% Variables to plot against
x_vars = {'abs_from_go', 'contrast', 'dispersion', 'confidence', ...
          'unc_perceptual', 'unc_decision'};
x_labels = {'|\Delta from Go| (deg)', 'Contrast', 'Dispersion (deg)', ...
            'Confidence (z)', 'Perceptual Uncertainty', 'Decision Uncertainty'};

% Identify which variables should use exact values vs quantile binning
use_exact_values = [true, true, true, false, false, false];

% Measures to plot
y_vars = {'centroid_magnitude', 'generalized_variance'};
y_labels = {'Centroid Magnitude', 'Generalized Variance'};

for yi = 1:numel(y_vars)
    figure('Name', sprintf('Population Geometry: %s', y_labels{yi}), ...
           'Color', 'w', 'Position', [100 + 50*yi, 100, 1400, 500]);
    tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    for xi = 1:numel(x_vars)
        nexttile; hold on; grid on; box off;
        
        % Get data
        x_data = TrialTbl_all.(x_vars{xi});
        y_data = TrialTbl_all.(y_vars{yi});
        
        % Use absolute value for orientation
        if strcmp(x_vars{xi}, 'abs_from_go')
            x_data = abs(x_data);
        end
        
        % Filter valid trials
        valid = isfinite(x_data) & isfinite(y_data);
        if sum(valid) < 10
            title(sprintf('%s vs %s (insufficient data)', y_labels{yi}, x_labels{xi}));
            continue;
        end
        
        x_valid = x_data(valid);
        y_valid = y_data(valid);
        
        % --- Choose binning strategy based on variable type ---
        if use_exact_values(xi)
            % EXACT VALUES: group by actual levels present in data
            
            % Round to avoid floating-point duplicates
            if strcmp(x_vars{xi}, 'contrast')
                x_rounded = round(x_valid, 3);  % 3 decimals for contrast
            elseif strcmp(x_vars{xi}, 'dispersion')
                x_rounded = round(x_valid, 0);  % integers for dispersion
            else  % abs_from_go
                x_rounded = round(x_valid, 1);  % 1 decimal for orientation
            end
            
            % Get unique values and compute stats per level
            [x_unique, ~, x_idx] = unique(x_rounded);
            
            y_mean = accumarray(x_idx, y_valid, [], @(x) mean(x, 'omitnan'));
            y_sem = accumarray(x_idx, y_valid, [], @(x) std(x, 'omitnan')/sqrt(sum(isfinite(x))));
            
            % Plot
            errorbar(x_unique, y_mean, y_sem, 'o-', 'LineWidth', 2, ...
                     'MarkerFaceColor', 'w', 'MarkerSize', 8);
            
        else
            % QUANTILE BINNING: for continuous variables
            
            n_bins = 10;
            edges = quantile(x_valid, linspace(0, 1, n_bins + 1));
            edges = unique(edges);  % Handle ties
            
            if numel(edges) < 3  % Need at least 2 bins
                title(sprintf('%s vs %s (too uniform)', y_labels{yi}, x_labels{xi}));
                continue;
            end
            
            [~, ~, bin_idx] = histcounts(x_valid, edges);
            binned = bin_idx > 0;
            
            % Compute bin centers and stats
            x_mean = accumarray(bin_idx(binned), x_valid(binned), [], @(x) mean(x, 'omitnan'));
            y_mean = accumarray(bin_idx(binned), y_valid(binned), [], @(x) mean(x, 'omitnan'));
            y_sem = accumarray(bin_idx(binned), y_valid(binned), [], @(x) std(x, 'omitnan')/sqrt(sum(isfinite(x))));
            
            % Plot
            errorbar(x_mean, y_mean, y_sem, 'o-', 'LineWidth', 2, ...
                     'MarkerFaceColor', 'w', 'MarkerSize', 8);
        end
        
        % Compute correlation
        [r, p] = corr(x_valid, y_valid, 'rows', 'complete', 'type', 'Spearman');
        
        % Labels and title
        xlabel(x_labels{xi});
        ylabel(y_labels{yi});
        % title(sprintf('%s vs %s (ρ=%.2f, p=%.3f)', y_labels{yi}, x_labels{xi}, r, p), ...
        %       'Interpreter', 'none');
    end
end

%% D) 2D Heatmaps: Population geometry vs Contrast × Dispersion

% Round for binning to exact experimental values
C_round = round(TrialTbl_all.contrast, 3);
D_round = round(TrialTbl_all.dispersion, 0);

% Get unique values (sorted)
C_vals = sort(unique(C_round));
D_vals = sort(unique(D_round));

% Create heatmaps for both measures
y_vars = {'centroid_magnitude', 'generalized_variance'};
y_labels = {'Centroid Magnitude', 'Generalized Variance'};

figure('Name', 'Population Geometry: Contrast × Dispersion Heatmaps', ...
       'Color', 'w', 'Position', [200, 200, 1000, 400]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for yi = 1:numel(y_vars)
    nexttile; hold on;
    
    y_data = TrialTbl_all.(y_vars{yi});
    valid = isfinite(C_round) & isfinite(D_round) & isfinite(y_data);
    
    % Map to indices
    [~, c_idx] = ismember(C_round(valid), C_vals);
    [~, d_idx] = ismember(D_round(valid), D_vals);
    
    % Compute mean in each bin
    M = accumarray([c_idx, d_idx], y_data(valid), [numel(C_vals), numel(D_vals)], ...
                   @(x) mean(x, 'omitnan'), NaN);
    N = accumarray([c_idx, d_idx], 1, [numel(C_vals), numel(D_vals)], @sum, 0);
    
    % Mask low-N cells
    M(N < 3) = NaN;
    
    % Plot with transparency for NaNs
    h = imagesc(M');
    set(h, 'AlphaData', ~isnan(M'));
    
    % Formatting
    cb = colorbar;
    cb.Label.String = y_labels{yi};
    xlabel('Contrast');
    ylabel('Dispersion (deg)');
    title(y_labels{yi});
    colormap(gca, parula);
    % axis xy;  % Origin at bottom-left
    xticks(1:4)
    yticks(1:4)
    xticklabels(num2str(C_vals))
    yticklabels(num2str(D_vals))
    
end

%% E) 2D Heatmaps: Population geometry vs Perceptual × Decision Uncertainty

% Use quantile-based bins for continuous uncertainty variables
n_bins = 10;  % Number of bins per axis

figure('Name', 'Population Geometry: Perceptual × Decision Uncertainty Heatmaps', ...
       'Color', 'w', 'Position', [250, 250, 1000, 400]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for yi = 1:numel(y_vars)
    nexttile; hold on;
    
    y_data = TrialTbl_all.(y_vars{yi});
    U_perc = TrialTbl_all.unc_perceptual;
    U_dec = TrialTbl_all.unc_decision;
    
    valid = isfinite(U_perc) & isfinite(U_dec) & isfinite(y_data);
    
    if sum(valid) < 50
        title(sprintf('%s (insufficient data)', y_labels{yi}));
        continue;
    end
    
    % Create bins using quantiles for better coverage
    perc_edges = quantile(U_perc(valid), linspace(0, 1, n_bins + 1));
    dec_edges = quantile(U_dec(valid), linspace(0, 1, n_bins + 1));
    
    % Ensure unique edges
    perc_edges = unique(perc_edges);
    dec_edges = unique(dec_edges);
    
    if numel(perc_edges) < 3 || numel(dec_edges) < 3
        title(sprintf('%s (insufficient bins)', y_labels{yi}));
        continue;
    end
    
    % Bin the data
    [~, ~, perc_bin] = histcounts(U_perc(valid), perc_edges);
    [~, ~, dec_bin] = histcounts(U_dec(valid), dec_edges);
    
    % Remove unbinned trials
    binned = perc_bin > 0 & dec_bin > 0;
    
    % Compute mean in each bin
    M = accumarray([perc_bin(binned), dec_bin(binned)], y_data(valid & binned), ...
                   [numel(perc_edges)-1, numel(dec_edges)-1], ...
                   @(x) mean(x, 'omitnan'), NaN);
    N = accumarray([perc_bin(binned), dec_bin(binned)], 1, ...
                   [numel(perc_edges)-1, numel(dec_edges)-1], @sum, 0);
    
    % Mask low-N cells
    M(N < 3) = NaN;
    
    % Bin centers for axes
    perc_centers = (perc_edges(1:end-1) + perc_edges(2:end)) / 2;
    dec_centers = (dec_edges(1:end-1) + dec_edges(2:end)) / 2;
    
    % Plot with transparency for NaNs
    h = imagesc(perc_centers, dec_centers, M');
    set(h, 'AlphaData', ~isnan(M'));
    
    % Formatting
    cb = colorbar;
    cb.Label.String = y_labels{yi};
    xlabel('Perceptual Uncertainty');
    ylabel('Decision Uncertainty');
    title(y_labels{yi});
    colormap(gca, "parula");  % Use viridis or parula
    
    axis tight;  % Origin at bottom-left
end

fprintf('=== Population Geometry Analysis Complete ===\n');