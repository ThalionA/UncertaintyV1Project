%% Integrate_GLMHMM_Analysis.m
% Integrates Python GLM-HMM states with Matlab Neural/Behavioral data.
% FIXES: Matches sessions using LOCAL indices to handle Global ID offset.

%% 1. Load HMM Data
fprintf('--- Loading GLM-HMM States ---\n');
% hmm_file = 'GLM_HMM_states_aligned_animal_level.csv';
hmm_file = 'GLM_HMM_states_aligned_hierarchical.csv';

if ~exist(hmm_file, 'file'), error('File %s not found.', hmm_file); end
HMM = readtable(hmm_file);

% Ensure string types for matching
HMM.animal = string(HMM.animal);

% Identify State Columns
state_cols = HMM.Properties.VariableNames(startsWith(HMM.Properties.VariableNames, 'prob_state_'));
n_states = numel(state_cols);

%% 2. Reconstruct Exact Keys for Neural Table
% We use PerAnimalRaw to get the Ground Truth (Session, TrialID) for every row
fprintf('Reconstructing exact trial keys from PerAnimalRaw...\n');

% Initialize key columns
if ~ismember('real_trial_id', TrialTbl_all.Properties.VariableNames)
    TrialTbl_all.real_trial_id = nan(height(TrialTbl_all), 1);
    TrialTbl_all.real_session_id = nan(height(TrialTbl_all), 1);
end

unique_animals = unique(TrialTbl_all.animal);

for i = 1:numel(unique_animals)
    ani_tag = unique_animals{i};
    
    % Find this animal in PerAnimalRaw
    raw_idx = find(strcmp({PerAnimalRaw.tag}, ani_tag));
    if isempty(raw_idx), continue; end
    
    UD = PerAnimalRaw(raw_idx).UD;
    
    % Find rows in table
    table_mask = strcmp(TrialTbl_all.animal, ani_tag);
    
    if sum(table_mask) ~= numel(UD)
        warning('Row mismatch for %s (Table: %d, Raw: %d). Mapping may be unsafe.', ...
            ani_tag, sum(table_mask), numel(UD));
        % Proceeding assuming the subset matches the start of UD is risky, 
        % but usually PerAnimalRaw is the parent of TrialTbl_all.
    end
    
    % Extract Local Session and Trial Number
    s_ids = [UD.session]'; 
    t_ids = [UD.trialNumber]';
    
    % Fill Table
    % Note: If table_mask is smaller/larger, this assignment might error. 
    % We assume TrialTbl_all was built directly from these UDs.
    try
        TrialTbl_all.real_session_id(table_mask) = s_ids;
        TrialTbl_all.real_trial_id(table_mask)   = t_ids;
    catch
        warning('Could not assign keys for %s due to size mismatch.', ani_tag);
    end
end

%% 3. Perform the Merge (With Local Session Correction)
fprintf('Merging HMM states into TrialTbl_all...\n');

TrialTbl_all.GLM_State = nan(height(TrialTbl_all), 1);
for k = 1:n_states
    TrialTbl_all.(sprintf('prob_state_%d', k)) = nan(height(TrialTbl_all), 1);
end

for i = 1:numel(unique_animals)
    ani = unique_animals{i};
    
    % 1. Neural Data Indices
    neur_rows = find(strcmp(TrialTbl_all.animal, ani));
    if isempty(neur_rows), continue; end
    
    % 2. HMM Data Indices
    hmm_rows = find(strcmp(HMM.animal, ani));
    if isempty(hmm_rows), continue; end
    
    % Get Sub-tables
    T_neur = TrialTbl_all(neur_rows, {'real_session_id', 'real_trial_id'});
    T_hmm_sub = HMM(hmm_rows, {'session', 'trial_in_session'});
    probs_hmm_sub = table2array(HMM(hmm_rows, state_cols));
    
    % --- CRITICAL FIX: CONVERT HMM GLOBAL SESSION TO LOCAL SESSION ---
    % HMM sessions might be [3, 4] but Neural are [1, 2].
    % We sort the unique HMM sessions and assign them 1..N
    [~, ~, local_sess_idx] = unique(T_hmm_sub.session, 'sorted');
    T_hmm_sub.local_session = local_sess_idx;
    
    % Now keys match: [Local_Session, Trial_ID]
    keys_neur = [T_neur.real_session_id, T_neur.real_trial_id];
    keys_hmm  = [T_hmm_sub.local_session, T_hmm_sub.trial_in_session];
    
    % Map rows
    [lia, locb] = ismember(keys_neur, keys_hmm, 'rows');
    
    % Apply to main table
    valid_neur_idx = neur_rows(lia);
    valid_hmm_idx  = locb(lia);
    
    if ~isempty(valid_neur_idx)
        % Get dominant state
        [~, dom_state] = max(probs_hmm_sub(valid_hmm_idx, :), [], 2);
        TrialTbl_all.GLM_State(valid_neur_idx) = dom_state;

        % Assign all state probabilities
        for k = 1:n_states
            col_name = sprintf('prob_state_%d', k);
            TrialTbl_all.(col_name)(valid_neur_idx) = probs_hmm_sub(valid_hmm_idx, k);
        end
end
    
    fprintf('  %s: %d/%d trials matched.\n', ani, sum(lia), numel(neur_rows));
end

%% 4. Extract Pupil Data (Robustly)
fprintf('Extracting pupil data...\n');
if ~ismember('pupil_grating', TrialTbl_all.Properties.VariableNames)
    TrialTbl_all.pupil_grating = nan(height(TrialTbl_all), 1);
    TrialTbl_all.pupil_z = nan(height(TrialTbl_all), 1);
    
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
        
        % Assign (assuming row alignment verified above)
        if numel(neur_rows) == numel(pupil_vals)
            TrialTbl_all.pupil_grating(neur_rows) = pupil_vals;
            
            % Z-score
            valid = ~isnan(pupil_vals);
            if any(valid)
                mu = mean(pupil_vals, 'omitnan'); sig = std(pupil_vals, 'omitnan');
                TrialTbl_all.pupil_z(neur_rows) = (pupil_vals - mu) / sig;
            end
        end
    end
end
%% 4.5 Plot Global Weights for ALL K Models (Model Evolution)
fprintf('Plotting Global Weights for all K...\n');
all_k_file = 'GLM_HMM_global_weights_all_K.csv';

if exist(all_k_file, 'file')
    W_all = readtable(all_k_file);
    K_models = unique(W_all.K_model);
    max_K = max(K_models);
    
    % Create a grid figure: Columns = K models, Rows = States
    figure('Name', 'GLM-HMM Weights Across K', 'Color', 'w', 'Position', [50 50 350*length(K_models) 200*max_K]);
    tl_all = tiledlayout(max_K, length(K_models), 'TileSpacing', 'compact', 'Padding', 'compact');
    
    preds = unique(W_all.predictor, 'stable');
    n_preds = length(preds);
    
    for state_idx = 1:max_K
        for k_idx = 1:length(K_models)
            K = K_models(k_idx);
            nexttile((state_idx - 1) * length(K_models) + k_idx);
            
            if state_idx <= K
                % Extract weights for this specific K and State
                mask = (W_all.K_model == K) & (W_all.state == state_idx);
                sub_W = W_all(mask, :);
                
                % Ensure correct ordering of predictors
                vals = zeros(1, n_preds);
                for p = 1:n_preds
                    idx = strcmp(sub_W.predictor, preds{p});
                    if any(idx), vals(p) = sub_W.weight(idx); end
                end
                
                % Bar plot
                bar(1:n_preds, vals, 'FaceColor', [0.3 0.3 0.3], 'EdgeColor', 'none');
                yline(0, 'k-', 'LineWidth', 1);
                
                % Formatting
                title(sprintf('K = %d | State %d', K, state_idx));
                xlim([0.5 n_preds+0.5]);
                
                if state_idx == K % Only put x-labels on the bottom row for each K
                    xticks(1:n_preds);
                    xticklabels(strrep(preds, '_', ' '));
                    xtickangle(25);
                else
                    xticks(1:n_preds);
                    xticklabels({});
                end
                
                if k_idx == 1
                    ylabel('Weight');
                end
                grid on; box off;
            else
                % Leave the tile completely blank if the state doesn't exist for this K
                axis off;
            end
        end
    end
    title(tl_all, 'Evolution of Global GLM Weights Across K', 'FontSize', 16, 'FontWeight', 'bold');
else
    fprintf('File %s not found. Skipping the all-K global weights plot.\n', all_k_file);
end
%% 5. Sophisticated GLM-HMM Plots
% Requires: 'GLM_HMM_weights.csv' (from Python) and TrialTbl_all

figure('Name', 'GLM-HMM Deep Dive', 'Color', 'w', 'Position', [10 10 1600 1000]);
tl = tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- Config ---
% Colors: State 1 (Green/Engaged), State 2 (Red/Biased), State 3 (Blue), State 4 (Grey)
state_cols = [0 0.6 0; 0.8 0 0; 0 0 0.7; 0.5 0.5 0.5]; 
state_names = {'Engaged', 'Biased', 'State 3', 'State 4'};
animals = unique(TrialTbl_all.animal);
n_animals = numel(animals);

% =========================================================================
% ROW 1: MODEL DEFINITION (Weights & Stimulus Dependency)
% =========================================================================

% --- 1A. GLM Weights (The "Receptive Field" of each state) ---
nexttile([1 2]); hold on; 
title('A. GLM Weights per State');
if exist('GLM_HMM_weights.csv', 'file')
    W_Tbl = readtable('GLM_HMM_weights.csv');
    preds = unique(W_Tbl.predictor, 'stable');
    n_preds = numel(preds);
    
    % Grouped bar plot strategy
    group_width = 0.8;
    bar_width = group_width / n_states;
    
    for p = 1:n_preds
        p_name = preds{p};
        center_pos = p;
        
        for k = 1:n_states
            % Offset bars
            x_pos = center_pos + (k - mean(1:n_states)) * bar_width;
            
            % Get data for this predictor & state across animals
            mask = strcmp(W_Tbl.predictor, p_name) & W_Tbl.state == k;
            vals = W_Tbl.weight(mask);
            
            % Plot Mean Bar
            bar(x_pos, mean(vals), bar_width, 'FaceColor', state_cols(k,:), 'EdgeColor', 'none', 'FaceAlpha', 0.7);
            % Plot SEM errorbar
            errorbar(x_pos, mean(vals), std(vals)/sqrt(numel(vals)), 'k.', 'LineWidth', 1.5);
        end
    end
    yline(0, 'k-', 'LineWidth', 0.5);
    xticks(1:n_preds); xticklabels(strrep(preds, '_', ' ')); xtickangle(20);
    ylabel('GLM Weight'); grid on;
else
    text(0.5,0.5,'Weights CSV missing','Horiz','center');
end

% --- 1B. State Occupancy vs Contrast (Heatmap) ---
nexttile; 
plot_state_dependency_heatmap(TrialTbl_all, 'contrast', n_states, state_names);
title('B. P(State | Contrast)');

% --- 1C. State Occupancy vs Dispersion (Heatmap) ---
nexttile; 
plot_state_dependency_heatmap(TrialTbl_all, 'dispersion', n_states, state_names);
title('C. P(State | Dispersion)');


% =========================================================================
% ROW 2: BEHAVIOR & PUPIL (Connected Swarm Plots)
% =========================================================================

% --- 2A. Accuracy ---
nexttile; title('D. Accuracy');
plot_connected_states(TrialTbl_all, animals, n_states, state_cols, 'performance', []);
yline(0.5, ':k'); ylabel('P(Correct)'); ylim([0.3 1]);

% --- 2B. Confidence (Magnitude) ---
nexttile; title('E. Confidence (Abs)');
plot_connected_states(TrialTbl_all, animals, n_states, state_cols, 'confidence', @(x) abs(x));
ylabel('|Conf| (z)');

% --- 2C. Pupil Size ---
nexttile; title('F. Pupil Size');
plot_connected_states(TrialTbl_all, animals, n_states, state_cols, 'pupil_z', []);
ylabel('Pupil (z)'); yline(0, ':k');

% --- 2D. Reaction Time (if available) or Licks ---
nexttile; title('G. Pre-RZ Lick Rate');
if ismember('preRZ_lick_rate', TrialTbl_all.Properties.VariableNames)
    plot_connected_states(TrialTbl_all, animals, n_states, state_cols, 'preRZ_lick_rate', []);
    ylabel('Licks/s');
end


% =========================================================================
% ROW 3: UNCERTAINTY & NEURAL (The Core Hypothesis)
% =========================================================================

% --- 3A. Perceptual Uncertainty ---
nexttile; title('H. Perceptual Uncertainty');
plot_connected_states(TrialTbl_all, animals, n_states, state_cols, 'unc_perceptual', []);
ylabel('Uncertainty (a.u.)');

% --- 3B. Decision Uncertainty ---
nexttile; title('I. Decision Uncertainty');
plot_connected_states(TrialTbl_all, animals, n_states, state_cols, 'unc_decision', []);
ylabel('Uncertainty (a.u.)');

% --- 3C. Neural Gain (Normalized Activity) ---
nexttile; title('J. Neural Activity (Norm)');
% Pre-calculate norm to State 1 per animal
T_norm = normalize_to_state1(TrialTbl_all, animals, 'meanAct_gr');
plot_connected_states(T_norm, animals, n_states, state_cols, 'meanAct_gr', []);
yline(1, ':k'); ylabel('Ratio vs Engaged');

% --- 3D. Neural Variability (Generalized Variance) ---
nexttile; title('K. Gen. Variance (Norm)');
T_norm = normalize_to_state1(TrialTbl_all, animals, 'logGV_gr');
plot_connected_states(T_norm, animals, n_states, state_cols, 'logGV_gr', []);
yline(1, ':k'); ylabel('Ratio vs Engaged');

%% 6. Deep Dive Plots (Psychometrics & Temporal Dynamics)
figure('Name', 'GLM-HMM Deep Dive 2', 'Color', 'w', 'Position', [100 100 1500 450]);
% Create a 1x3 layout for our 3 panels
tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Colors: State 1 (Green), State 2 (Red), State 3 (Blue), State 4 (Grey)
state_cols = [0 0.7 0; 0.8 0 0; 0 0 0.8; 0.5 0.5 0.5]; 

% Pre-calculate the exact GLM stimulus input to be used in Panel B
norm_stim = (TrialTbl_all.stimulus - 45) / 45;
eff_contrast = TrialTbl_all.contrast;
eff_contrast(abs(eff_contrast - 0.99) < 1e-3) = 1.0; 
eff_disp_factor = 5.0 ./ TrialTbl_all.dispersion;

glm_input_stim = norm_stim .* eff_contrast .* eff_disp_factor;
flip_animals = ["Cb21", "Cb22", "Cb24"];
flip_mask = ismember(string(TrialTbl_all.animal), flip_animals);
glm_input_stim(flip_mask) = glm_input_stim(flip_mask) * -1;
glm_input_stim = glm_input_stim * -1; 
TrialTbl_all.glm_stim_input = glm_input_stim;


% =========================================================================
% --- PANEL A: Original Psychometric Curve (|Δ from Go|) ---
% =========================================================================
nexttile; hold on;
title('A. Psychometric Curve (|Δ from Go|)');
xlabel('|Δ from Go Pole| (deg)'); ylabel('P(Go)');

all_x_abs = round(TrialTbl_all.abs_from_go, 4);
unique_x_abs = unique(all_x_abs(~isnan(all_x_abs))); 
n_bins_abs = numel(unique_x_abs);

for k = 1:n_states
    y_means = nan(numel(unique_animals), n_bins_abs);
    for i = 1:numel(unique_animals)
        ani_mask = strcmp(TrialTbl_all.animal, unique_animals{i});
        state_mask = TrialTbl_all.GLM_State == k;
        if sum(ani_mask & state_mask) > 10 
            for b = 1:n_bins_abs
                bin_mask = (all_x_abs == unique_x_abs(b));
                mask = ani_mask & state_mask & bin_mask;
                if sum(mask) > 2
                    y_means(i, b) = mean(TrialTbl_all.goChoice(mask), 'omitnan');
                end
            end
        end
    end
    mu = mean(y_means, 1, 'omitnan');
    se = std(y_means, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(y_means), 1));
    
    valid_pts = ~isnan(mu) & ~isnan(se);
    if sum(valid_pts) > 0
        errorbar(unique_x_abs(valid_pts), mu(valid_pts), se(valid_pts), 'o-', ...
            'Color', state_cols(k,:), 'LineWidth', 2, 'MarkerFaceColor', state_cols(k,:), ...
            'DisplayName', sprintf('State %d', k));
    end
end
xline(45, '--k'); 
ylim([0 1]); xlim([-1 91]); 
legend({'Engaged (S1)', 'Biased (S2)', 'State 3'}, 'Location', 'best'); grid on;

% =========================================================================
% --- PANEL B: Psychometric Curve (Scaled Evidence, Flipped, No Errorbars) ---
% =========================================================================
nexttile; hold on;
title('B. Psychometric Curve (Scaled Evidence)');
xlabel('Scaled Evidence (Categorical Spacing)'); ylabel('P(Go)');

all_x_scaled = round(TrialTbl_all.glm_stim_input, 4);
unique_x_scaled = unique(all_x_scaled(~isnan(all_x_scaled))); 
n_bins_scaled = numel(unique_x_scaled);
x_categorical = 1:n_bins_scaled; 

for k = 1:n_states
    y_means = nan(numel(unique_animals), n_bins_scaled);
    for i = 1:numel(unique_animals)
        ani_mask = strcmp(TrialTbl_all.animal, unique_animals{i});
        state_mask = TrialTbl_all.GLM_State == k;
        if sum(ani_mask & state_mask) > 10 
            for b = 1:n_bins_scaled
                bin_mask = (all_x_scaled == unique_x_scaled(b));
                mask = ani_mask & state_mask & bin_mask;
                if sum(mask) > 2
                    y_means(i, b) = mean(TrialTbl_all.goChoice(mask), 'omitnan');
                end
            end
        end
    end
    mu = mean(y_means, 1, 'omitnan');
    
    valid_pts = ~isnan(mu);
    if sum(valid_pts) > 0
        % PLOT INSTEAD OF ERRORBAR (Cleaner lines)
        plot(x_categorical(valid_pts), mu(valid_pts), 'o-', ...
            'Color', state_cols(k,:), 'LineWidth', 2, 'MarkerFaceColor', state_cols(k,:));
    end
end

[~, zero_idx] = min(abs(unique_x_scaled - 0)); 
xline(x_categorical(zero_idx), '--k', 'Boundary', 'LabelVerticalAlignment', 'bottom'); 

xticks(x_categorical);
xticklabels(arrayfun(@(x) sprintf('%.2g', x), unique_x_scaled, 'UniformOutput', false));
xtickangle(45); 

ylim([0 1]); 
xlim([0.5, n_bins_scaled + 0.5]); 
set(gca, 'XDir', 'reverse'); % FLIP X-AXIS SO GO IS ON THE LEFT
grid on;

% =========================================================================
% --- PANEL C: Temporal Dynamics (Satiety / Engagement over time) ---
% =========================================================================
nexttile; hold on;
title('C. State Occupancy over Session');
xlabel('Trial in Session'); ylabel('State Probability');

state_prob_cols = TrialTbl_all.Properties.VariableNames(startsWith(TrialTbl_all.Properties.VariableNames, 'prob_state_'));
n_states = numel(state_prob_cols); 
if n_states > 0
    state_names_all = {'Engaged (S1)', 'Biased (S2)', 'State 3 (S3)', 'State 4 (S4)'};
    plot_colors = state_cols(1:n_states, :);
    plot_names = state_names_all(1:n_states);
    
    window_total_size = 50; 
    window_half_width = floor(window_total_size / 2);
    
    % Find the max common trial number across all mice
    n_animals = numel(unique_animals);
    max_trials_per_mouse = nan(n_animals, 1);
    for ia = 1:n_animals
        ani_tag = unique_animals{ia};
        max_trials_per_mouse(ia) = max(TrialTbl_all.real_trial_id(strcmp(TrialTbl_all.animal, ani_tag)));
    end
    max_trials_plot = min(max_trials_per_mouse); 
    
    fprintf('Plot C max trials set to: %d (max common across mice)\n', max_trials_plot);
    plot_x_axis = 1:max_trials_plot;
    
    all_session_traces_cell = {}; 
    for ianimal = 1:n_animals
        ani_tag = unique_animals{ianimal};
        T_animal = TrialTbl_all(strcmp(TrialTbl_all.animal, ani_tag), :);
        sessions_in_animal = unique(T_animal.real_session_id(~isnan(T_animal.real_session_id)));
        
        for isession = 1:numel(sessions_in_animal)
            sess_id = sessions_in_animal(isession);
            sess_mask = (T_animal.real_session_id == sess_id);
            
            t_nums = T_animal.real_trial_id(sess_mask);
            p_states = T_animal{sess_mask, state_prob_cols};
            
            p_states_full = nan(max_trials_plot, n_states);
            valid_t_nums = t_nums(t_nums <= max_trials_plot & t_nums > 0);
            valid_p_states = p_states(t_nums <= max_trials_plot & t_nums > 0, :);
            if ~isempty(valid_t_nums)
                p_states_full(valid_t_nums, :) = valid_p_states;
            end
            
            session_trace_smooth = nan(max_trials_plot, n_states);
            for t = 1:max_trials_plot
                win_start = max(1, t - window_half_width);
                win_end   = min(max_trials_plot, t + window_half_width);
                p_window = p_states_full(win_start:win_end, :);
                session_trace_smooth(t, :) = mean(p_window, 1, 'omitnan');
            end
            all_session_traces_cell{end+1} = session_trace_smooth;
        end
    end
    
    all_session_traces = permute(cat(3, all_session_traces_cell{:}), [3, 1, 2]);
    h_lines = gobjects(n_states, 1);
    
    for k = 1:n_states
        traces_k = all_session_traces(:, :, k); 
        mu_k = mean(traces_k, 1, 'omitnan');
        sem_k = std(traces_k, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(traces_k), 1));
        
        valid_plot_points = ~isnan(mu_k) & ~isnan(sem_k);
        if ~any(valid_plot_points), continue; end
        
        plot_x = plot_x_axis(valid_plot_points);
        mu_k = mu_k(valid_plot_points);
        sem_k = sem_k(valid_plot_points);
        
        try
            line_props = {'Color', plot_colors(k,:), 'LineWidth', 2};
            h = shadedErrorBar(plot_x, mu_k, sem_k, 'lineprops', line_props);
            h_lines(k) = h.mainLine;
        catch 
            h = errorbar(plot_x, mu_k, sem_k, 'Color', plot_colors(k,:), 'LineWidth', 2);
            h_lines(k) = h;
        end
    end
    
    yline(1/n_states, ':k', 'Alpha', 0.5, 'DisplayName', 'Chance');
    ylim([0 1]);
    xlim([1 max_trials_plot]);
    grid on;
end

%% --- LOCAL FUNCTIONS FOR SOPHISTICATED PLOTTING ---

function plot_connected_states(T, animals, n_states, colors, field, transform_fn)
    hold on;
    
    % 1. Collect Data
    means_per_animal = nan(numel(animals), n_states);
    for i = 1:numel(animals)
        for k = 1:n_states
            mask = strcmp(T.animal, animals{i}) & T.GLM_State == k;
            if sum(mask) > 10
                vals = T.(field)(mask);
                if ~isempty(transform_fn), vals = transform_fn(vals); end
                means_per_animal(i, k) = mean(vals, 'omitnan');
            end
        end
    end
    
    % 2. Plot Individual "Ghost" Lines
    plot(1:n_states, means_per_animal', '-', 'Color', [0.8 0.8 0.8], 'LineWidth', 0.5);
    
    % 3. Plot Group Mean + Error Bars with Colors
    grp_mean = mean(means_per_animal, 1, 'omitnan');
    grp_sem = std(means_per_animal, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(means_per_animal), 1));
    
    % Error bars (black)
    errorbar(1:n_states, grp_mean, grp_sem, 'k-', 'LineWidth', 1.5);
    
    % Colored Markers on top
    for k = 1:n_states
        plot(k, grp_mean(k), 'o', 'MarkerSize', 8, ...
            'MarkerFaceColor', colors(k,:), 'MarkerEdgeColor', 'k', 'LineWidth', 1);
    end
    
    xticks(1:n_states);
    xticklabels(arrayfun(@(x) sprintf('S%d',x), 1:n_states, 'uni', 0));
    xlim([0.5, n_states + 0.5]);
    grid on; box off;
end

function plot_state_dependency_heatmap(T, condition_field, n_states, state_names)
    % Discretize condition if continuous
    raw_cond = T.(condition_field);
    % If many unique values (continuous), bin them. If discrete (like contrast 0.01, 0.1), keep.
    u_vals = unique(raw_cond);
    if numel(u_vals) > 10 
        % Binning for dispersion if it varies slightly
        raw_cond = round(raw_cond, 2); 
    end
    u_vals = unique(raw_cond(~isnan(raw_cond)));
    
    % Calculate P(State | Condition)
    prob_mat = nan(numel(u_vals), n_states);
    
    for i = 1:numel(u_vals)
        cond_mask = (raw_cond == u_vals(i));
        total_in_cond = sum(cond_mask);
        if total_in_cond > 0
            for k = 1:n_states
                prob_mat(i, k) = sum(cond_mask & T.GLM_State == k) / total_in_cond;
            end
        end
    end
    
    % Plot
    imagesc(prob_mat'); 
    colormap(flipud(bone)); % Dark = High prob
    c = colorbar; c.Label.String = 'P(State | Cond)';
    
    xticks(1:numel(u_vals));
    % Fancy tick labels
    if max(u_vals) <= 1
        xticklabels(arrayfun(@(x) sprintf('%.0f%%', x*100), u_vals, 'uni', 0));
    else
        xticklabels(arrayfun(@(x) sprintf('%g', x), u_vals, 'uni', 0));
    end
    
    xlabel(condition_field);
    yticks(1:n_states); yticklabels(state_names(1:n_states));
end

function T_out = normalize_to_state1(T_in, animals, field)
    T_out = T_in;
    for i = 1:numel(animals)
        mask = strcmp(T_in.animal, animals{i});
        mask1 = mask & T_in.GLM_State == 1;
        base_val = mean(T_in.(field)(mask1), 'omitnan');
        
        % Normalize this animal's data
        T_out.(field)(mask) = T_in.(field)(mask) ./ base_val;
    end
end