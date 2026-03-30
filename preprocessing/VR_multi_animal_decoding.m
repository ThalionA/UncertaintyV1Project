%% Setup and Configuration
cv_folds = 5;
n_time_bins_max = size(NeuralStore(1).Gspk, 3);
animal_names = {Animals.tag};
% Define decoding targets dynamically
targets = struct();
targets(1).name = 'StimulusCategory'; targets(1).type = 'class';     targets(1).source = 'abs_from_go'; % <-- CHANGED
targets(2).name = 'Choice';           targets(2).type = 'class';     targets(2).source = 'goChoice';
targets(3).name = 'Contrast';         targets(3).type = 'reg';       targets(3).source = 'contrast';
targets(4).name = 'Dispersion';       targets(4).type = 'reg';       targets(4).source = 'dispersion';
targets(5).name = 'Confidence';       targets(5).type = 'reg';       targets(5).source = 'confidence';
targets(6).name = 'Performance';      targets(6).type = 'class';     targets(6).source = 'performance'; % Correct/Error
targets(7).name = 'UncPerceptual';    targets(7).type = 'reg';       targets(7).source = 'unc_perceptual';
targets(8).name = 'UncDecision';      targets(8).type = 'reg';       targets(8).source = 'unc_decision';
targets(9).name = 'Outcome';          targets(9).type = 'multiclass';targets(9).source = 'outcome'; % H/M/FA/CR
% Pre-allocate storage
results = struct();
for t = 1:length(targets)
    results.(targets(t).name).linear_time = nan(nAnimals, n_time_bins_max, cv_folds);
    results.(targets(t).name).nonlinear_time = nan(nAnimals, n_time_bins_max, cv_folds);
    results.(targets(t).name).linear_full = nan(nAnimals, cv_folds);
    results.(targets(t).name).nonlinear_full = nan(nAnimals, cv_folds);
end
% Pre-allocate cell array to store trial-by-trial results
all_trial_results = cell(nAnimals, 1);

%% --- New Section: Time-Resolved Choice Decoding with Shuffle Control ---
% Parameters
n_shuffles = 25;       % Number of permutations for significance testing
cv_folds_choice = 5;    % Folds for cross-validation
alpha_sig = 0.01;       % Significance threshold

% Storage for group results
choice_decode = struct();
choice_decode.time_bins = NeuralStore(1).xG;
choice_decode.n_bins = length(choice_decode.time_bins);
choice_decode.linear.real = nan(nAnimals, choice_decode.n_bins);
choice_decode.linear.shuffle_mean = nan(nAnimals, choice_decode.n_bins);
choice_decode.linear.shuffle_upper = nan(nAnimals, choice_decode.n_bins); % 95th percentile
choice_decode.nonlinear.real = nan(nAnimals, choice_decode.n_bins);
choice_decode.nonlinear.shuffle_mean = nan(nAnimals, choice_decode.n_bins);
choice_decode.nonlinear.shuffle_upper = nan(nAnimals, choice_decode.n_bins);
choice_decode.p_vals_lin = nan(nAnimals, choice_decode.n_bins);
choice_decode.p_vals_nlin = nan(nAnimals, choice_decode.n_bins);

fprintf('Starting Time-Resolved Choice Decoding (Linear vs Nonlinear + %d Shuffles)...\n', n_shuffles);

for ianimal = 1:nAnimals
    fprintf('Processing Animal %d (%s)...\n', ianimal, Animals(ianimal).tag);
    
    % 1. Prepare Data
    % Activity: [Neurons x TimeBins x Trials]
    X_raw = permute(NeuralStore(ianimal).Gspk, [2, 3, 1]); 
    [n_neurons, n_bins, n_total_trials] = size(X_raw);
    
    % Get Labels
    animal_idx = strcmp(TrialTbl_all.animal, Animals(ianimal).tag);
    y_choice = TrialTbl_all.goChoice(animal_idx);
    
    % Filter valid trials (non-NaN choice)
    valid_mask = ~isnan(y_choice);
    X = X_raw(:, :, valid_mask);
    y = y_choice(valid_mask);
    n_trials = length(y);
    
    % Pre-allocate temporal storage for this animal
    acc_lin_real = nan(n_bins, cv_folds_choice);
    acc_nlin_real = nan(n_bins, cv_folds_choice);
    
    % Store shuffle results: [n_bins, n_shuffles]
    dist_lin_shuff = nan(n_bins, n_shuffles);
    dist_nlin_shuff = nan(n_bins, n_shuffles);
    
    % 2. Real Decoding Loop
    cv = cvpartition(n_trials, 'KFold', cv_folds_choice);
    
    for ibin = 1:n_bins
        % Prepare feature matrix for this bin: [Trials x Neurons]
        X_bin = squeeze(X(:, ibin, :))';
        
        % Z-score features (crucial for SVM)
        X_bin = zscore(X_bin);
        
        for k = 1:cv_folds_choice
            train_idx = cv.training(k);
            test_idx = cv.test(k);
            
            X_train = X_bin(train_idx, :);
            y_train = y(train_idx);
            X_test = X_bin(test_idx, :);
            y_test = y(test_idx);
            
            % Linear Decoder (SVM)
            Mdl_lin = fitclinear(X_train, y_train, 'Learner', 'svm');
            pred_lin = predict(Mdl_lin, X_test);
            acc_lin_real(ibin, k) = mean(pred_lin == y_test);
            
            % Nonlinear Decoder (Gaussian Naive Bayes)
            % 'DistributionNames', 'normal' assumes z-scored inputs are roughly gaussian
            Mdl_nlin = fitcnb(X_train, y_train, 'DistributionNames', 'normal');
            pred_nlin = predict(Mdl_nlin, X_test);
            acc_nlin_real(ibin, k) = mean(pred_nlin == y_test);
        end
    end
    
    % Average across folds for Real Performance
    choice_decode.linear.real(ianimal, :) = mean(acc_lin_real, 2);
    choice_decode.nonlinear.real(ianimal, :) = mean(acc_nlin_real, 2);
    
    % 3. Shuffle Control Loop
    % We shuffle labels ONCE per iteration, then run through bins/CV
    fprintf('   Running shuffles');
    for ishuff = 1:n_shuffles
        if mod(ishuff, 50) == 0, fprintf('.'); end
        
        % Shuffle labels globally for this iteration
        y_shuff = y(randperm(n_trials));
        
        % Create a new partition for the shuffle to ensure independence
        cv_shuff = cvpartition(n_trials, 'KFold', cv_folds_choice);
        
        % Temporary storage for this shuffle iteration
        tmp_acc_lin = nan(n_bins, 1);
        tmp_acc_nlin = nan(n_bins, 1);
        
        for ibin = 1:n_bins
            X_bin = squeeze(X(:, ibin, :))';
            X_bin = zscore(X_bin); % Recalculating Z-score not strictly needed if handled outside, but safe
            
            fold_acc_lin = nan(cv_folds_choice, 1);
            fold_acc_nlin = nan(cv_folds_choice, 1);
            
            for k = 1:cv_folds_choice
                tr = cv_shuff.training(k);
                te = cv_shuff.test(k);
                
                % Linear
                mdl_l = fitclinear(X_bin(tr,:), y_shuff(tr), 'Learner', 'svm');
                fold_acc_lin(k) = mean(predict(mdl_l, X_bin(te,:)) == y_shuff(te));
                
                % Nonlinear
                mdl_n = fitcnb(X_bin(tr,:), y_shuff(tr), 'DistributionNames', 'normal');
                fold_acc_nlin(k) = mean(predict(mdl_n, X_bin(te,:)) == y_shuff(te));
            end
            tmp_acc_lin(ibin) = mean(fold_acc_lin);
            tmp_acc_nlin(ibin) = mean(fold_acc_nlin);
        end
        dist_lin_shuff(:, ishuff) = tmp_acc_lin;
        dist_nlin_shuff(:, ishuff) = tmp_acc_nlin;
    end
    fprintf('\n');
    
    % 4. Statistics & Storage
    choice_decode.linear.shuffle_mean(ianimal, :) = mean(dist_lin_shuff, 2);
    choice_decode.linear.shuffle_upper(ianimal, :) = prctile(dist_lin_shuff, 95, 2);
    choice_decode.nonlinear.shuffle_mean(ianimal, :) = mean(dist_nlin_shuff, 2);
    choice_decode.nonlinear.shuffle_upper(ianimal, :) = prctile(dist_nlin_shuff, 95, 2);
    
    % Calculate P-values (Time-resolved)
    % P = (Number of shuffles >= Real) / n_shuffles
    for b = 1:n_bins
        choice_decode.p_vals_lin(ianimal, b) = sum(dist_lin_shuff(b, :) >= choice_decode.linear.real(ianimal, b)) / n_shuffles;
        choice_decode.p_vals_nlin(ianimal, b) = sum(dist_nlin_shuff(b, :) >= choice_decode.nonlinear.real(ianimal, b)) / n_shuffles;
    end
end

% --- Plotting the Comparison ---
figure('Name', 'Choice Decoding: Linear vs Nonlinear + Significance', 'Color', 'w', 'Position', [100 100 1200 500]);
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

% Plot 1: Linear Decoder
nexttile; hold on;
mu_real = mean(choice_decode.linear.real, 1);
sem_real = std(choice_decode.linear.real, 0, 1) ./ sqrt(nAnimals);
mu_shuff = mean(choice_decode.linear.shuffle_mean, 1);
bound_shuff = mean(choice_decode.linear.shuffle_upper, 1);

% Draw Shuffle zone (Mean to 95th percentile)
x_vec = [choice_decode.time_bins, fliplr(choice_decode.time_bins)];
fill(x_vec, [mu_shuff, fliplr(bound_shuff)], [0.5 0.5 0.5], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
yline(0.5, 'k--');

% Draw Real Performance
shadedErrorBar(choice_decode.time_bins, mu_real, sem_real, 'lineprops', {'b-', 'LineWidth', 2});

% Mark significant timepoints (Uncorrected p < 0.05 across group)
% Here we assess if the group mean is consistently above chance, 
% or simpler: mark where the average animal p-value is significant
sig_bins = mean(choice_decode.p_vals_lin, 1) < alpha_sig; 
if any(sig_bins)
    plot(choice_decode.time_bins(sig_bins), ones(sum(sig_bins),1)*0.95, 'b*', 'MarkerSize', 5);
end

title('Linear Decoder (SVM)');
xlabel('Time from Grating Onset (s)'); ylabel('Accuracy'); ylim([0.4 1]);
legend('Shuffle (95%)', 'Chance', 'Real Data', 'Significant', 'Location', 'southwest');

% Plot 2: Nonlinear Decoder
nexttile; hold on;
mu_real_nl = mean(choice_decode.nonlinear.real, 1);
sem_real_nl = std(choice_decode.nonlinear.real, 0, 1) ./ sqrt(nAnimals);
mu_shuff_nl = mean(choice_decode.nonlinear.shuffle_mean, 1);
bound_shuff_nl = mean(choice_decode.nonlinear.shuffle_upper, 1);

fill(x_vec, [mu_shuff_nl, fliplr(bound_shuff_nl)], [0.5 0.5 0.5], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
yline(0.5, 'k--');

shadedErrorBar(choice_decode.time_bins, mu_real_nl, sem_real_nl, 'lineprops', {'r-', 'LineWidth', 2});

sig_bins_nl = mean(choice_decode.p_vals_nlin, 1) < alpha_sig;
if any(sig_bins_nl)
    plot(choice_decode.time_bins(sig_bins_nl), ones(sum(sig_bins_nl),1)*0.95, 'r*', 'MarkerSize', 5);
end

title('Nonlinear Decoder (Naive Bayes)');
xlabel('Time from Grating Onset (s)'); ylim([0.4 1]);

%% Main Decoding Loop
for ianimal = 1:nAnimals
    fprintf('Processing Animal %d (%s)...\n', ianimal, animal_names{ianimal});
    
    % 1. Data Prep
    animal_grating_activity = permute(NeuralStore(ianimal).Gspk, [2, 3, 1]); 
    [n_neurons, n_bins, n_trials] = size(animal_grating_activity);
    
    animal_idx = strcmp(TrialTbl_all.animal, animal_names{ianimal});
    
    % Extract Target Variables
    target_data = struct();
    valid_mask_base = true(n_trials, 1);
    
    % --- Load continuous abs_from_go for neurometric plots ---
    target_data.continuous_abs_from_go = TrialTbl_all.abs_from_go(animal_idx);
    target_data.abs_confidence = abs(TrialTbl_all.confidence(animal_idx));
    valid_mask_base = valid_mask_base & ~isnan(target_data.continuous_abs_from_go) & ~isnan(target_data.abs_confidence);
    
    % --- DATA LOADING LOGIC ---
    for t = 1:length(targets)
        raw_vals = TrialTbl_all.(targets(t).source)(animal_idx);
        % --- Binarization rule for StimulusCategory ---
        if strcmp(targets(t).name, 'StimulusCategory')
            raw_vals = double(raw_vals > 45); 
        end
        target_data.(targets(t).name) = raw_vals; % Store raw value first
        
        if strcmp(targets(t).type, 'multiclass')
            target_data.(targets(t).name) = categorical(raw_vals); 
            valid_mask_base = valid_mask_base & ~isundefined(target_data.(targets(t).name));
        elseif iscell(raw_vals) || isstring(raw_vals) 
            target_data.(targets(t).name) = categorical(raw_vals);
            valid_mask_base = valid_mask_base & ~isundefined(target_data.(targets(t).name));
        else
            valid_mask_base = valid_mask_base & ~isnan(raw_vals);
        end
    end
    
    % --- Load GLM_State for post-hoc grouping ---
    if ismember('GLM_State', TrialTbl_all.Properties.VariableNames)
        target_data.GLM_State = TrialTbl_all.GLM_State(animal_idx);
        valid_mask_base = valid_mask_base & ~isnan(target_data.GLM_State);
    else
        warning('GLM_State not found in TrialTbl_all. Skipping state-based analysis.');
        target_data.GLM_State = ones(n_trials, 1); % Fallback
    end
    
    % Apply mask to neural data and all target_data fields
    animal_grating_activity = animal_grating_activity(:, :, valid_mask_base);
    n_trials_clean = sum(valid_mask_base);
    
    all_fields = fieldnames(target_data);
    for f = 1:length(all_fields)
        target_data.(all_fields{f}) = target_data.(all_fields{f})(valid_mask_base);
    end
    % Prepare structure to store this animal's trial-by-trial results
    animal_trial_store = struct();
    animal_trial_store.true_abs_from_go = target_data.continuous_abs_from_go; 
    animal_trial_store.true_contrast = target_data.Contrast;
    animal_trial_store.true_dispersion = target_data.Dispersion;
    animal_trial_store.true_choice = target_data.Choice;
    animal_trial_store.true_abs_confidence = target_data.abs_confidence;
    animal_trial_store.true_unc_perceptual = target_data.UncPerceptual;
    animal_trial_store.true_unc_decision = target_data.UncDecision;
    animal_trial_store.true_GLM_State = target_data.GLM_State;
    
    for t = 1:length(targets)
        animal_trial_store.(['true_' targets(t).name]) = target_data.(targets(t).name);
        animal_trial_store.(['pred_' targets(t).name '_lin']) = nan(n_trials_clean, 1);
        animal_trial_store.(['pred_' targets(t).name '_nlin']) = nan(n_trials_clean, 1);
        if strcmp(targets(t).type, 'multiclass')
            animal_trial_store.(['pred_' targets(t).name '_lin']) = repmat(categorical(missing), n_trials_clean, 1);
            animal_trial_store.(['pred_' targets(t).name '_nlin']) = repmat(categorical(missing), n_trials_clean, 1);
        end
    end
    
    
    cv_idx = cvpartition(length(target_data.Choice), 'KFold', cv_folds);
    
    for ifold = 1:cv_folds
        train_mask = cv_idx.training(ifold);
        test_mask  = cv_idx.test(ifold);
        
        % --- A. Time-Resolved Decoding ---
        for ibin = 1:n_bins
            X_train = squeeze(animal_grating_activity(:, ibin, train_mask))'; 
            X_test  = squeeze(animal_grating_activity(:, ibin, test_mask))';  
            
            % --- STANDARDIZATION (Z-SCORE) for X ---
            [X_train_z, mu, sigma] = zscore(X_train);
            sigma_safe = sigma + 1e-9;
            X_test_z = (X_test - mu) ./ sigma_safe;

            for t = 1:length(targets)
                y_train = target_data.(targets(t).name)(train_mask);
                y_test  = target_data.(targets(t).name)(test_mask);
                
                if strcmp(targets(t).type, 'class')
                    Mdl_lin = fitclinear(X_train_z, y_train, 'Learner', 'svm'); 
                    pred = predict(Mdl_lin, X_test_z);
                    results.(targets(t).name).linear_time(ianimal, ibin, ifold) = mean(pred == y_test);
                    
                    if n_neurons > 0
                         Mdl_nlin = fitcnb(X_train_z, y_train); 
                         pred = predict(Mdl_nlin, X_test_z);
                         results.(targets(t).name).nonlinear_time(ianimal, ibin, ifold) = mean(pred == y_test);
                    end
                    
                elseif strcmp(targets(t).type, 'reg')
                    % --- STANDARDIZE Y for regression ---
                    [y_train_z, mu_y, sigma_y] = zscore(y_train);
                    sigma_y_safe = sigma_y + 1e-9;
                    
                    % Linear regression
                    Mdl_lin = fitrlinear(X_train_z, y_train_z, 'Learner', 'leastsquares', 'Regularization', 'ridge');
                    pred_z = predict(Mdl_lin, X_test_z);
                    % Transform back to original scale
                    pred = pred_z * sigma_y_safe + mu_y;
                    results.(targets(t).name).linear_time(ianimal, ibin, ifold) = ...
                        1 - sum((pred - y_test).^2, 'omitnan') / sum((y_test - mean(y_test, 'omitnan')).^2, 'omitnan');
                    
                    % Gaussian Process regression
                    try
                        Mdl_gp = fitrgp(X_train_z, y_train_z, 'KernelFunction', 'squaredexponential', Standardize=false);
                        pred_z = predict(Mdl_gp, X_test_z);
                        % Transform back to original scale
                        pred = pred_z * sigma_y_safe + mu_y;
                        results.(targets(t).name).nonlinear_time(ianimal, ibin, ifold) = ...
                            1 - sum((pred - y_test).^2, 'omitnan') / sum((y_test - mean(y_test, 'omitnan')).^2, 'omitnan');
                    catch
                        results.(targets(t).name).nonlinear_time(ianimal, ibin, ifold) = NaN;
                    end
                    
                elseif strcmp(targets(t).type, 'multiclass')
                    Mdl_lin = fitcecoc(X_train_z, y_train, 'Learners', 'svm');
                    pred = predict(Mdl_lin, X_test_z);
                    results.(targets(t).name).linear_time(ianimal, ibin, ifold) = mean(pred == y_test);

                    Mdl_nlin = fitcecoc(X_train_z, y_train, 'Learners', 'naivebayes'); 
                    pred = predict(Mdl_nlin, X_test_z);
                    results.(targets(t).name).nonlinear_time(ianimal, ibin, ifold) = mean(pred == y_test);
                end
            end
        end
        
        % --- B. Full-Trial (Average) Decoding ---
        X_full = permute(animal_grating_activity, [3, 1, 2]); 
        X_full = mean(X_full, 3, 'omitnan'); 
        
        X_train_full = X_full(train_mask, :);
        X_test_full  = X_full(test_mask, :);
        
        % --- STANDARDIZATION (Z-SCORE) for X ---
        [X_train_full_z, mu_full, sigma_full] = zscore(X_train_full);
        sigma_full_safe = sigma_full + 1e-9;
        X_test_full_z = (X_test_full - mu_full) ./ sigma_full_safe;

        for t = 1:length(targets)
            y_train = target_data.(targets(t).name)(train_mask);
            y_test  = target_data.(targets(t).name)(test_mask);
            
            if strcmp(targets(t).type, 'class')
                Mdl_lin = fitclinear(X_train_full_z, y_train, 'Learner', 'svm'); 
                pred_lin = predict(Mdl_lin, X_test_full_z);
                results.(targets(t).name).linear_full(ianimal, ifold) = mean(pred_lin == y_test);
                animal_trial_store.(['pred_' targets(t).name '_lin'])(test_mask) = pred_lin; 
                
                Mdl_nlin = fitcnb(X_train_full_z, y_train); 
                pred_nlin = predict(Mdl_nlin, X_test_full_z);
                results.(targets(t).name).nonlinear_full(ianimal, ifold) = mean(pred_nlin == y_test);
                animal_trial_store.(['pred_' targets(t).name '_nlin'])(test_mask) = pred_nlin; 
                
            elseif strcmp(targets(t).type, 'reg')
                % --- STANDARDIZE Y for regression ---
                [y_train_z, mu_y_full, sigma_y_full] = zscore(y_train);
                sigma_y_full_safe = sigma_y_full + 1e-9;
                
                % Linear regression
                Mdl_lin = fitrlinear(X_train_full_z, y_train_z, "Learner", "leastsquares", "Regularization", "ridge");
                pred_lin_z = predict(Mdl_lin, X_test_full_z);
                % Transform back to original scale
                pred_lin = pred_lin_z * sigma_y_full_safe + mu_y_full;
                results.(targets(t).name).linear_full(ianimal, ifold) = ...
                    1 - sum((pred_lin - y_test).^2, 'omitnan') / sum((y_test - mean(y_test, 'omitnan')).^2, 'omitnan');
                animal_trial_store.(['pred_' targets(t).name '_lin'])(test_mask) = pred_lin; 

                % Gaussian Process regression
                Mdl_gp = fitrgp(X_train_full_z, y_train_z, 'KernelFunction', 'squaredexponential', Standardize=false);
                pred_nlin_z = predict(Mdl_gp, X_test_full_z);
                % Transform back to original scale
                pred_nlin = pred_nlin_z * sigma_y_full_safe + mu_y_full;
                results.(targets(t).name).nonlinear_full(ianimal, ifold) = ...
                    1 - sum((pred_nlin - y_test).^2, 'omitnan') / sum((y_test - mean(y_test, 'omitnan')).^2, 'omitnan');
                animal_trial_store.(['pred_' targets(t).name '_nlin'])(test_mask) = pred_nlin; 
            
            elseif strcmp(targets(t).type, 'multiclass')
                Mdl_lin = fitcecoc(X_train_full_z, y_train, 'Learners', 'svm');
                pred_lin = predict(Mdl_lin, X_test_full_z);
                results.(targets(t).name).linear_full(ianimal, ifold) = mean(pred_lin == y_test);
                animal_trial_store.(['pred_' targets(t).name '_lin'])(test_mask) = pred_lin;

                Mdl_nlin = fitcecoc(X_train_full_z, y_train, 'Learners', 'naivebayes'); 
                pred_nlin = predict(Mdl_nlin, X_test_full_z);
                results.(targets(t).name).nonlinear_full(ianimal, ifold) = mean(pred_nlin == y_test);
                animal_trial_store.(['pred_' targets(t).name '_nlin'])(test_mask) = pred_nlin;
            end
        end
        fprintf('.');
    end
    
    all_trial_results{ianimal} = animal_trial_store;
    fprintf('\n');
end

%% --- Analysis Block: Modulation & Neurometric Curves ---
% --- BIN DEFINITIONS ---
abs_go_values = [0, 15, 30, 40, 45, 50, 60, 75, 90];
n_go_bins = length(abs_go_values); % = 9 bins

contrast_values = [0.01, 0.25, 0.5, 1];
n_contrast_bins = length(contrast_values); % = 4 bins

dispersion_values = [5, 30, 45, 90];
n_disp_bins = length(dispersion_values); % = 4 bins

n_quantile_bins = 4; % For Confidence and Uncertainty (Quartiles)

% --- Pre-allocate storage ---
neurometric_p_go = nan(nAnimals, n_go_bins);
psychometric_p_go = nan(nAnimals, n_go_bins);
% abs_from_go_bin_centers no longer needed, we use the values directly

% Storage for Choice decoder (resized for new bin counts)
acc_choice_vs_contrast = nan(nAnimals, n_contrast_bins);
acc_choice_vs_dispersion = nan(nAnimals, n_disp_bins);
acc_choice_vs_abs_conf = nan(nAnimals, n_quantile_bins);
acc_choice_vs_unc_perc = nan(nAnimals, n_quantile_bins);
acc_choice_vs_unc_dec = nan(nAnimals, n_quantile_bins);

% Storage for StimulusCategory decoder modulation (resized for new bin counts)
acc_stim_vs_contrast = nan(nAnimals, n_contrast_bins);
acc_stim_vs_dispersion = nan(nAnimals, n_disp_bins);
acc_stim_vs_abs_conf = nan(nAnimals, n_quantile_bins);
acc_stim_vs_unc_perc = nan(nAnimals, n_quantile_bins);
acc_stim_vs_unc_dec = nan(nAnimals, n_quantile_bins);


for ianimal = 1:nAnimals
    fprintf('Running post-hoc modulation analysis for Animal %d...\n', ianimal);
    
    data = all_trial_results{ianimal};
    
    % Get data
    all_preds_choice = data.pred_Choice_lin;
    y_choice = data.true_Choice;
    all_preds_stim = data.pred_StimulusCategory_lin;
    y_stim = data.true_StimulusCategory;
    
    % Get modulator data
    y_abs_from_go = data.true_abs_from_go; 
    y_contrast = data.true_contrast;
    y_dispersion = data.true_dispersion;
    y_abs_conf = data.true_abs_confidence;
    y_unc_perc = data.true_unc_perceptual;
    y_unc_dec = data.true_unc_decision;
    
    
    valid_mask = ~isnan(all_preds_choice) & ~isnan(y_choice) & ...
                 ~isnan(all_preds_stim) & ~isnan(y_stim) & ... 
                 ~isnan(y_abs_from_go) & ~isnan(y_contrast) & ~isnan(y_dispersion) & ...
                 ~isnan(y_abs_conf) & ~isnan(y_unc_perc) & ~isnan(y_unc_dec);
    
    all_preds_choice = all_preds_choice(valid_mask);
    y_choice = y_choice(valid_mask);
    all_preds_stim = all_preds_stim(valid_mask); 
    y_stim = y_stim(valid_mask);                 
    
    y_abs_from_go = y_abs_from_go(valid_mask);
    y_contrast = y_contrast(valid_mask);
    y_dispersion = y_dispersion(valid_mask);
    y_abs_conf = y_abs_conf(valid_mask);
    y_unc_perc = y_unc_perc(valid_mask);
    y_unc_dec = y_unc_dec(valid_mask);
    
    % --- 1. Neurometric & Psychometric (using specified 'go' values) ---
    for ibin = 1:n_go_bins
        % Use small tolerance for floating point comparison
        trials_in_bin = (abs(y_abs_from_go - abs_go_values(ibin)) < 1e-5);
        if any(trials_in_bin)
            neurometric_p_go(ianimal, ibin) = mean(all_preds_choice(trials_in_bin) == 1);
            psychometric_p_go(ianimal, ibin) = mean(y_choice(trials_in_bin) == 1);
        end
    end
    
    % --- 2. Binning and Modulation ---
    
    % --- A: Specified Values (Contrast, Dispersion) ---
    for ibin = 1:n_contrast_bins
        trials_in_bin = (abs(y_contrast - contrast_values(ibin)) < 1e-5);
        if any(trials_in_bin)
            acc_choice_vs_contrast(ianimal, ibin) = mean(all_preds_choice(trials_in_bin) == y_choice(trials_in_bin));
            acc_stim_vs_contrast(ianimal, ibin) = mean(all_preds_stim(trials_in_bin) == y_stim(trials_in_bin));
        end
    end
    
    for ibin = 1:n_disp_bins
        trials_in_bin = (abs(y_dispersion - dispersion_values(ibin)) < 1e-5);
        if any(trials_in_bin)
            acc_choice_vs_dispersion(ianimal, ibin) = mean(all_preds_choice(trials_in_bin) == y_choice(trials_in_bin));
            acc_stim_vs_dispersion(ianimal, ibin) = mean(all_preds_stim(trials_in_bin) == y_stim(trials_in_bin));
        end
    end

    % --- B: Quantile Bins (Confidence, Uncertainty) ---
    get_bins = @(d, nq) discretize(d, prctile(d, linspace(0, 100, nq + 1)));

    abs_conf_bin_idx = get_bins(y_abs_conf, n_quantile_bins);
    unc_perc_bin_idx = get_bins(y_unc_perc, n_quantile_bins);
    unc_dec_bin_idx  = get_bins(y_unc_dec, n_quantile_bins);
    % 
    % abs_conf_bin_idx = discretize(y_abs_conf, n_quantile_bins);
    % unc_perc_bin_idx = discretize(y_unc_perc, n_quantile_bins);
    % unc_dec_bin_idx = discretize(y_unc_dec, n_quantile_bins);

    for ibin = 1:n_quantile_bins
        trials_in_bin_abs_conf = (abs_conf_bin_idx == ibin);
        trials_in_bin_unc_perc = (unc_perc_bin_idx == ibin);
        trials_in_bin_unc_dec = (unc_dec_bin_idx == ibin);

        if any(trials_in_bin_abs_conf)
            acc_choice_vs_abs_conf(ianimal, ibin) = mean(all_preds_choice(trials_in_bin_abs_conf) == y_choice(trials_in_bin_abs_conf));
            acc_stim_vs_abs_conf(ianimal, ibin) = mean(all_preds_stim(trials_in_bin_abs_conf) == y_stim(trials_in_bin_abs_conf));
        end
        if any(trials_in_bin_unc_perc)
            acc_choice_vs_unc_perc(ianimal, ibin) = mean(all_preds_choice(trials_in_bin_unc_perc) == y_choice(trials_in_bin_unc_perc));
            acc_stim_vs_unc_perc(ianimal, ibin) = mean(all_preds_stim(trials_in_bin_unc_perc) == y_stim(trials_in_bin_unc_perc));
        end
        if any(trials_in_bin_unc_dec)
            acc_choice_vs_unc_dec(ianimal, ibin) = mean(all_preds_choice(trials_in_bin_unc_dec) == y_choice(trials_in_bin_unc_dec));
            acc_stim_vs_unc_dec(ianimal, ibin) = mean(all_preds_stim(trials_in_bin_unc_dec) == y_stim(trials_in_bin_unc_dec));
        end
    end
end
fprintf('All analysis complete.\n');

%% --- Plotting Section 1: Time-Resolved Decoding ---
time_bins = NeuralStore(1).xG;
figure('Name', 'Time-Resolved Decoding', 'Color', 'w', 'Position', [100 100 1600 1000]);
tplot = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
for t = 1:length(targets)
    nexttile;
    hold on;
    res_lin_folds = mean(results.(targets(t).name).linear_time, 3, 'omitnan');
    mu_lin = mean(res_lin_folds, 1, 'omitnan');
    sem_lin = std(res_lin_folds, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(res_lin_folds(:,1))));
    res_nlin_folds = mean(results.(targets(t).name).nonlinear_time, 3, 'omitnan');
    mu_nlin = mean(res_nlin_folds, 1, 'omitnan');
    sem_nlin = std(res_nlin_folds, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(res_nlin_folds(:,1))));
    h1 = shadedErrorBar(time_bins, mu_lin, sem_lin, 'lineprops', {'Color', 'b', 'LineWidth', 2});
    h2 = shadedErrorBar(time_bins, mu_nlin, sem_nlin, 'lineprops', {'Color', 'r', 'LineWidth', 2});
    title(sprintf('Decoding: %s', targets(t).name), 'Interpreter', 'none');
    xlabel('Time from onset (s)');
    if strcmp(targets(t).type, 'class')
        ylabel('Accuracy');
        yline(0.5, 'k--', 'Chance (0.5)');
    elseif strcmp(targets(t).type, 'multiclass')
        ylabel('Accuracy');
        % Get number of classes dynamically (robustly)
        valid_data_idx = find(~cellfun(@isempty, all_trial_results), 1);
        if ~isempty(valid_data_idx)
            n_classes = length(categories(all_trial_results{valid_data_idx}.(['true_' targets(t).name])));
            yline(1/n_classes, 'k--', sprintf('Chance (1/%d)', n_classes));
        else
            yline(0.25, 'k--', 'Chance (fallback)'); % Fallback for 4 outcomes
        end
    else
        ylabel('R^2');
        yline(0, 'k--', 'Mean Model');
    end
    if t == 1
        legend([h1.mainLine, h2.mainLine], {'Linear (SVM)', 'Nonlinear (Naive Bayes)'}, 'Location', 'best'); % <-- Updated Legend
    end
    ylim([0, 1])
end
%% --- Plotting Section 2: Full-Trial Decoding ---
figure('Name', 'Full Trial Decoding Performance', 'Color', 'w', 'Position', [100 200 1600 400]);
tplot_full = tiledlayout(1, length(targets), 'TileSpacing', 'compact');
for t = 1:length(targets)
    nexttile;
    hold on;
    lin_perf = mean(results.(targets(t).name).linear_full, 2, 'omitnan');
    nlin_perf = mean(results.(targets(t).name).nonlinear_full, 2, 'omitnan');
    data_plot = [lin_perf, nlin_perf];
    b = bar([1, 2], mean(data_plot, 1, 'omitnan'), 'FaceColor', 'flat');
    b.CData(1,:) = [0.2 0.2 1];
    b.CData(2,:) = [1 0.2 0.2];
    errorbar([1, 2], mean(data_plot, 1, 'omitnan'), ...
        std(data_plot, 0, 1, 'omitnan')./sqrt(nAnimals), 'k.', 'LineWidth', 2);
    plot([1, 2], data_plot', 'Color', [0.5 0.5 0.5 0.5], 'LineWidth', 1);
    xticks([1 2]);
    xticklabels({'Lin', 'NonLin'});
    title(targets(t).name, 'Interpreter', 'none');
    if strcmp(targets(t).type, 'class')
        yline(0.5, 'k--');
        ylim([0.4 1]);
    elseif strcmp(targets(t).type, 'multiclass')
        valid_data_idx = find(~cellfun(@isempty, all_trial_results), 1);
        if ~isempty(valid_data_idx)
            n_classes = length(categories(all_trial_results{valid_data_idx}.(['true_' targets(t).name])));
            yline(1/n_classes, 'k--', sprintf('Chance (1/%d)', n_classes));
        else
            yline(0.25, 'k--', 'Chance (fallback)'); % Fallback for 4 outcomes
        end
    else
        yline(0, 'k--');
    end
end
ylabel(tplot_full, 'Performance (Acc or R^2)');

figure('Name', 'Full Trial Nonlinear Decoding Performance', 'Color', 'w', 'Position', [100 200 1600 400]);
for t = 1:length(targets)
    % nexttile;
    hold on;
    nlin_perf = mean(results.(targets(t).name).nonlinear_full, 2, 'omitnan');
    data_plot = nlin_perf;
    b = bar(t, mean(data_plot, 1, 'omitnan'), 'FaceColor', 'flat');
    % b.CData(1,:) = [0.2 0.2 1];
    % b.CData(2,:) = [1 0.2 0.2];
    errorbar(t, mean(data_plot, 1, 'omitnan'), ...
        std(data_plot, 0, 1, 'omitnan')./sqrt(nAnimals), 'k.', 'LineWidth', 2);
    plot(t, data_plot', 'Color', [0.5 0.5 0.5 0.5], 'LineWidth', 1);
    
end
xticks(1:length(targets))
xticklabels({targets(:).name});
% ylabel(tplot_full, 'Performance (Acc or R^2)');
%% --- Plotting Section 3: Neurometric & Psychometric ---
figure('Name', 'Neurometric vs. Psychometric Curves', 'Color', 'w', 'Position', [200 200 600 500]);
hold on;
% --- Use actual bin values for x-axis ---
plot_bins = abs_go_values; 
mu_psycho = mean(psychometric_p_go, 1, 'omitnan');
sem_psycho = std(psychometric_p_go, 0, 1, 'omitnan') ./ sqrt(nAnimals);
h_psycho = shadedErrorBar(plot_bins, mu_psycho, sem_psycho, 'lineprops', {'k-', 'LineWidth', 2});
mu_neuro = mean(neurometric_p_go, 1, 'omitnan');
sem_neuro = std(neurometric_p_go, 0, 1, 'omitnan') ./ sqrt(nAnimals);
h_neuro = shadedErrorBar(plot_bins, mu_neuro, sem_neuro, 'lineprops', {'b--', 'LineWidth', 2});
xlabel('Abs(Stimulus - Go Boundary) (deg)');
ylabel('P(Choose ''Go'')');
title('Neurometric vs. Psychometric Choice');
v_boundary = 45; % The 45-degree boundary
yline(0.5, 'k:');
xline(v_boundary, 'k:', 'Category Boundary', 'LabelVerticalAlignment','bottom');
legend([h_psycho.mainLine, h_neuro.mainLine], {'Psychometric (Animal)', 'Neurometric (Decoder)'}, 'Location', 'best');
xlim([min(plot_bins) max(plot_bins)]);

%% --- Plotting Section 4: Choice Accuracy Modulation by Stimulus ---
figure('Name', 'Choice Accuracy Modulation by Stimulus', 'Color', 'w', 'Position', [300 300 1000 450]);
tiledlayout(1, 2);
% --- Plot vs. Contrast ---
nexttile; hold on;
mu_acc_con = mean(acc_choice_vs_contrast, 1, 'omitnan');
sem_acc_con = std(acc_choice_vs_contrast, 0, 1, 'omitnan') ./ sqrt(nAnimals);
plot_bins_x_con = 1:n_contrast_bins;
plot_bin_labels_con = arrayfun(@(x) num2str(x), contrast_values, 'UniformOutput', false);
plot(plot_bins_x_con, acc_choice_vs_contrast', 'Color', [0.5 0.5 0.5 0.3]);
errorbar(plot_bins_x_con, mu_acc_con, sem_acc_con, 'b-o', 'LineWidth', 2, 'CapSize', 0);
xlabel('Stimulus Contrast'); ylabel('Choice Decoder Accuracy'); title('Choice Accuracy vs. Contrast');
yline(0.5, 'k--', 'Chance'); xticks(plot_bins_x_con); xticklabels(plot_bin_labels_con); xlim([0.5, n_contrast_bins + 0.5]);
% --- Plot vs. Dispersion ---
nexttile; hold on;
mu_acc_disp = mean(acc_choice_vs_dispersion, 1, 'omitnan');
sem_acc_disp = std(acc_choice_vs_dispersion, 0, 1, 'omitnan') ./ sqrt(nAnimals);
plot_bins_x_disp = 1:n_disp_bins;
plot_bin_labels_disp = arrayfun(@(x) num2str(x), dispersion_values, 'UniformOutput', false);
plot(plot_bins_x_disp, acc_choice_vs_dispersion', 'Color', [0.5 0.5 0.5 0.3]);
errorbar(plot_bins_x_disp, mu_acc_disp, sem_acc_disp, 'r-o', 'LineWidth', 2, 'CapSize', 0);
xlabel('Stimulus Dispersion (deg)'); ylabel('Choice Decoder Accuracy'); title('Choice Accuracy vs. Dispersion');
yline(0.5, 'k--', 'Chance'); xticks(plot_bins_x_disp); xticklabels(plot_bin_labels_disp); xlim([0.5, n_disp_bins + 0.5]);
%% --- Plotting Section 5: Choice Accuracy Modulation by Uncertainty ---
figure('Name', 'Choice Accuracy Modulation by Uncertainty', 'Color', 'w', 'Position', [400 400 1400 450]);
tiledlayout(1, 3);
plot_bins_x = 1:n_quantile_bins;
plot_bin_labels = {'Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'};
% --- Plot vs. Abs(Confidence) ---
nexttile; hold on;
mu_acc = mean(acc_choice_vs_abs_conf, 1, 'omitnan');
sem_acc = std(acc_choice_vs_abs_conf, 0, 1, 'omitnan') ./ sqrt(nAnimals);
plot(plot_bins_x, acc_choice_vs_abs_conf', 'Color', [0.5 0.5 0.5 0.3]);
errorbar(plot_bins_x, mu_acc, sem_acc, 'k-o', 'LineWidth', 2, 'CapSize', 0);
xlabel('Abs(Behavioral Confidence) (Quantile Bin)'); ylabel('Choice Decoder Accuracy'); title('Choice Accuracy vs. |Confidence|');
yline(0.5, 'k--', 'Chance'); xticks(plot_bins_x); xticklabels(plot_bin_labels); xlim([0.5, n_quantile_bins + 0.5]);
% --- Plot vs. Perceptual Uncertainty ---
nexttile; hold on;
mu_acc = mean(acc_choice_vs_unc_perc, 1, 'omitnan');
sem_acc = std(acc_choice_vs_unc_perc, 0, 1, 'omitnan') ./ sqrt(nAnimals);
plot(plot_bins_x, acc_choice_vs_unc_perc', 'Color', [0.5 0.5 0.5 0.3]);
errorbar(plot_bins_x, mu_acc, sem_acc, 'g-o', 'LineWidth', 2, 'CapSize', 0);
xlabel('Perceptual Uncertainty (Quantile Bin)'); ylabel('Choice Decoder Accuracy'); title('Choice Accuracy vs. Perceptual Uncertainty');
yline(0.5, 'k--', 'Chance'); xticks(plot_bins_x); xticklabels(plot_bin_labels); xlim([0.5, n_quantile_bins + 0.5]);
% --- Plot vs. Decision Uncertainty ---
nexttile; hold on;
mu_acc = mean(acc_choice_vs_unc_dec, 1, 'omitnan');
sem_acc = std(acc_choice_vs_unc_dec, 0, 1, 'omitnan') ./ sqrt(nAnimals);
plot(plot_bins_x, acc_choice_vs_unc_dec', 'Color', [0.5 0.5 0.5 0.3]);
errorbar(plot_bins_x, mu_acc, sem_acc, 'm-o', 'LineWidth', 2, 'CapSize', 0);
xlabel('Decision Uncertainty (Quantile Bin)'); ylabel('Choice Decoder Accuracy'); title('Choice Accuracy vs. Decision Uncertainty');
yline(0.5, 'k--', 'Chance'); xticks(plot_bins_x); xticklabels(plot_bin_labels); xlim([0.5, n_quantile_bins + 0.5]);
%% --- Plotting Section 6: Stimulus Category Accuracy Modulation ---
figure('Name', 'Stimulus Category Accuracy Modulation', 'Color', 'w', 'Position', [400 400 1400 900]);
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
% --- Plot vs. Contrast (Specific Values) ---
nexttile;
hold on;
mu_acc = mean(acc_stim_vs_contrast, 1, 'omitnan');
sem_acc = std(acc_stim_vs_contrast, 0, 1, 'omitnan') ./ sqrt(nAnimals);
plot_bins_x_con = 1:n_contrast_bins;
plot_bin_labels_con = arrayfun(@(x) num2str(x), contrast_values, 'UniformOutput', false);
plot(plot_bins_x_con, acc_stim_vs_contrast', 'Color', [0.5 0.5 0.5 0.3]);
errorbar(plot_bins_x_con, mu_acc, sem_acc, 'b-o', 'LineWidth', 2, 'CapSize', 0);
xlabel('Stimulus Contrast');
ylabel('Stimulus Decoder Accuracy');
title('Stimulus Accuracy vs. Contrast');
yline(0.5, 'k--', 'Chance');
xticks(plot_bins_x_con);
xticklabels(plot_bin_labels_con);
xlim([0.5, n_contrast_bins + 0.5]);
% --- Plot vs. Dispersion (Specific Values) ---
nexttile;
hold on;
mu_acc = mean(acc_stim_vs_dispersion, 1, 'omitnan');
sem_acc = std(acc_stim_vs_dispersion, 0, 1, 'omitnan') ./ sqrt(nAnimals);
plot_bins_x_disp = 1:n_disp_bins;
plot_bin_labels_disp = arrayfun(@(x) num2str(x), dispersion_values, 'UniformOutput', false);
plot(plot_bins_x_disp, acc_stim_vs_dispersion', 'Color', [0.5 0.5 0.5 0.3]);
errorbar(plot_bins_x_disp, mu_acc, sem_acc, 'r-o', 'LineWidth', 2, 'CapSize', 0);
xlabel('Stimulus Dispersion (deg)');
ylabel('Stimulus Decoder Accuracy');
title('Stimulus Accuracy vs. Dispersion');
yline(0.5, 'k--', 'Chance');
xticks(plot_bins_x_disp);
xticklabels(plot_bin_labels_disp);
xlim([0.5, n_disp_bins + 0.5]);
% --- Plot vs. Abs(Confidence) (Quantiles) ---
nexttile;
hold on;
plot_bins_x_q = 1:n_quantile_bins;
plot_bin_labels_q = {'Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'};
mu_acc = mean(acc_stim_vs_abs_conf, 1, 'omitnan');
sem_acc = std(acc_stim_vs_abs_conf, 0, 1, 'omitnan') ./ sqrt(nAnimals);
plot(plot_bins_x_q, acc_stim_vs_abs_conf', 'Color', [0.5 0.5 0.5 0.3]);
errorbar(plot_bins_x_q, mu_acc, sem_acc, 'k-o', 'LineWidth', 2, 'CapSize', 0);
xlabel('Abs(Behavioral Confidence) (Quantile Bin)');
ylabel('Stimulus Decoder Accuracy');
title('Stimulus Accuracy vs. |Confidence|');
yline(0.5, 'k--', 'Chance');
xticks(plot_bins_x_q);
xticklabels(plot_bin_labels_q);
xlim([0.5, n_quantile_bins + 0.5]);
% --- Plot vs. Perceptual Uncertainty (Quantiles) ---
nexttile;
hold on;
mu_acc = mean(acc_stim_vs_unc_perc, 1, 'omitnan');
sem_acc = std(acc_stim_vs_unc_perc, 0, 1, 'omitnan') ./ sqrt(nAnimals);
plot(plot_bins_x_q, acc_stim_vs_unc_perc', 'Color', [0.5 0.5 0.5 0.3]);
errorbar(plot_bins_x_q, mu_acc, sem_acc, 'g-o', 'LineWidth', 2, 'CapSize', 0);
xlabel('Perceptual Uncertainty (Quantile Bin)');
ylabel('Stimulus Decoder Accuracy');
title('Stimulus Accuracy vs. Perceptual Unc.');
yline(0.5, 'k--', 'Chance');
xticks(plot_bins_x_q);
xticklabels(plot_bin_labels_q);
xlim([0.5, n_quantile_bins + 0.5]);
% --- Plot vs. Decision Uncertainty (Quantiles) ---
nexttile;
hold on;
mu_acc = mean(acc_stim_vs_unc_dec, 1, 'omitnan');
sem_acc = std(acc_stim_vs_unc_dec, 0, 1, 'omitnan') ./ sqrt(nAnimals);
plot(plot_bins_x_q, acc_stim_vs_unc_dec', 'Color', [0.5 0.5 0.5 0.3]);
errorbar(plot_bins_x_q, mu_acc, sem_acc, 'm-o', 'LineWidth', 2, 'CapSize', 0);
xlabel('Decision Uncertainty (Quantile Bin)');
ylabel('Stimulus Decoder Accuracy');
title('Stimulus Accuracy vs. Decision Unc.');
yline(0.5, 'k--', 'Chance');
xticks(plot_bins_x_q);
xticklabels(plot_bin_labels_q);
xlim([0.5, n_quantile_bins + 0.5]);


%% --- Plotting Section 7: Decoding Accuracy by GLM-HMM State ---
figure('Name', 'Decoding Accuracy by GLM-HMM State', 'Color', 'w', 'Position', [500 500 1200 450]);
tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
% --- Config ---
% Colors: State 1 (Green/Engaged), State 2 (Red/Biased), State 3 (Blue)
state_cols = [0 0.6 0; 0.8 0 0; 0 0 0.7; 0.5 0.5 0.5];
state_names = {'Engaged (S1)', 'Biased (S2)', 'State 3', 'State 4'};
% Variables to plot (must match 'targets.name' from above)
vars_to_plot = {'StimulusCategory', 'Choice', 'Outcome'};
% --- Check if state data exists ---
if ~isfield(all_trial_results{1}, 'true_GLM_State')
    sgtitle('GLM_State not found in analysis results. Skipping plot.');
    return;
end
n_states_found = max(cellfun(@(x) max(x.true_GLM_State), all_trial_results));
for ivar = 1:length(vars_to_plot)
    var_name = vars_to_plot{ivar};
    % Find the original target info
    target_idx = find(strcmp({targets.name}, var_name));
    if isempty(target_idx)
        warning('Could not find target info for %s', var_name);
        nexttile; title(sprintf('Error: %s not found', var_name));
        continue;
    end
    target_info = targets(target_idx);
    % Store [nAnimals x nStates]
    perf_by_state = nan(nAnimals, n_states_found);
    for ianimal = 1:nAnimals
        data = all_trial_results{ianimal};
        % Get true/predicted data for this variable
        pred_vals = data.(['pred_' var_name '_lin']);
        true_vals = data.(['true_' var_name]);
        glm_state = data.true_GLM_State;
        valid_mask = ~isnan(glm_state);
        if iscategorical(pred_vals)
            valid_mask = valid_mask & ~isundefined(pred_vals);
        else
            valid_mask = valid_mask & ~isnan(pred_vals);
        end
        if iscategorical(true_vals)
            valid_mask = valid_mask & ~isundefined(true_vals);
        else
            valid_mask = valid_mask & ~isnan(true_vals);
        end
        pred_vals = pred_vals(valid_mask);
        true_vals = true_vals(valid_mask);
        glm_state = glm_state(valid_mask);
        for k = 1:n_states_found
            state_mask = (glm_state == k);
            if sum(state_mask) < 10, continue; end
            pred_k = pred_vals(state_mask);
            true_k = true_vals(state_mask);
            % Calculate performance metric
            if strcmp(target_info.type, 'class') || strcmp(target_info.type, 'multiclass')
                % Accuracy
                perf_by_state(ianimal, k) = mean(pred_k == true_k);
            elseif strcmp(target_info.type, 'reg')
                % R-squared
                perf_by_state(ianimal, k) = 1 - sum((pred_k - true_k).^2, 'omitnan') / sum((true_k - mean(true_k, 'omitnan')).^2, 'omitnan');
            end
        end
    end
    % --- Plotting ---
    nexttile; hold on;
    % Plot individual animal lines (ghost lines)
    plot(1:n_states_found, perf_by_state', '-', 'Color', [0.8 0.8 0.8]);
    % Plot group mean + SEM
    grp_mean = mean(perf_by_state, 1, 'omitnan');
    grp_sem = std(perf_by_state, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(perf_by_state), 1));
    % Black line for error bar
    errorbar(1:n_states_found, grp_mean, grp_sem, 'k-', 'LineWidth', 1.5);
    % Colored markers
    for k = 1:n_states_found
        plot(k, grp_mean(k), 'o', 'MarkerSize', 10, ...
            'MarkerFaceColor', state_cols(k,:), 'MarkerEdgeColor', 'k');
    end
    % Formatting
    if strcmp(target_info.type, 'class')
        yline(0.5, 'k--', 'Chance');
        ylabel('Decoding Accuracy');
    elseif strcmp(target_info.type, 'multiclass')
        % Get number of classes dynamically (robustly)
        all_true_cats = [];
        for ia = 1:nAnimals
            if isfield(all_trial_results{ia}, ['true_' var_name])
                all_true_cats = [all_true_cats; all_trial_results{ia}.(['true_' var_name])]; %#ok<AGROW>
            end
        end
        n_classes = length(categories(all_true_cats(~isundefined(all_true_cats))));
        if n_classes > 1
            yline(1/n_classes, 'k--', 'Chance');
        else
            yline(0.25, 'k--', 'Chance (fallback)');
        end
        ylabel('Decoding Accuracy');
    elseif strcmp(target_info.type, 'reg')
        yline(0, 'k--', 'Chance');
        ylabel('Decoding R^2');
    end
    xticks(1:n_states_found);
    xticklabels(state_names(1:n_states_found));
    title(sprintf('Decoding: %s', var_name));
    xlim([0.5, n_states_found + 0.5]);
    grid on;
    ylim([0, 1])
end