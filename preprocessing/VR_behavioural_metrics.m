warning off
sesnames1 = {'20250220_Cb11_3', '20250221_Cb11', '20250222_Cb11', '20250224_Cb11', '20250225_Cb11', '20250225_Cb11_1',...
    '20250226_Cb11', '20250226_Cb11_1', '20250226_Cb11_2', '20250227_Cb11', '20250227_Cb11_1', '20250227_Cb11_2', '20250228_Cb11', '20250303_Cb11', '20250303_Cb11_1',...
    '20250304_Cb11', '20250305_Cb11', '20250306_Cb11', '20250307_Cb11', '20250310_Cb11', '20250311_Cb11', '20250311_Cb11_1', '20250312_Cb11', '20250313_Cb11', '20250314_Cb11', '20250314_Cb11_1', '20250317_Cb11',...
    '20250318_Cb11_1', '20250318_Cb11_2', '20250320_Cb11', '20250321_Cb11'};
sesnames2 = {'20250220_Cb14', '20250221_Cb14', '20250222_Cb14', '20250224_Cb14', '20250225_Cb14', '20250225_Cb14_1', '20250225_Cb14_2',...
    '20250226_Cb14', '20250227_Cb14', '20250228_Cb14', '20250303_Cb14', '20250304_Cb14', '20250305_Cb14', '20250306_Cb14', '20250307_Cb14', '20250310_Cb14', '20250311_Cb14', '20250312_Cb14', '20250313_Cb14', '20250314_Cb14',...
    '20250317_Cb14', '20250318_Cb14', '20250320_Cb14', '20250321_Cb14'};
sesnames3 = {'20250304_Cb15', '20250305_Cb15', '20250306_Cb15', '20250307_Cb15', '20250310_Cb15', '20250311_Cb15', '20250313_Cb15', '20250314_Cb15', '20250317_Cb15', '20250318_Cb15', ...
    '20250320_Cb15', '20250321_Cb15', '20250324_Cb15', '20250325_Cb15', '20250326_Cb15', '20250326_Cb15', '20250327_Cb15', '20250328_Cb15', '20250331_Cb15', '20250401_Cb15'};
sesnames4 = {'20250318_Cb17', '20250320_Cb17', '20250321_Cb17', '20250324_Cb17', '20250325_Cb17', '20250326_Cb17', '20250327_Cb17', '20250328_Cb17', '20250331_Cb17', '20250401_Cb17'};

all_sesnames = {sesnames1, sesnames2, sesnames3, sesnames4};

% Compute total number of sessions across all animals.
total_sessions = sum(cellfun(@numel, all_sesnames));

% Create a waitbar.
hWait = waitbar(0, 'Loading VR sessions...');
idx = 0;

for ianimal = 1:length(all_sesnames)
    vrSessions = cell(1, numel(all_sesnames{ianimal}));
    for iSession = 1:numel(all_sesnames{ianimal})
        idx = idx + 1;
        waitbar(idx / total_sessions, hWait, sprintf('Loading session %d of %d', idx, total_sessions));

        sesname = all_sesnames{ianimal}{iSession};
        load(['vr_', sesname, '.mat']);
        vrSessions{iSession} = vr;
    end
    
    [allPerformance{ianimal}, allRZLick{ianimal}, allRZVel{ianimal}, allDwell{ianimal}, ...
     allPreRZLick{ianimal}, allPreRZVel{ianimal}, allGratingVel{ianimal}, allStimuli{ianimal}, ...
     allGrVelFirst{ianimal}, allGrVelSecond{ianimal}, allCorrLickLatency{ianimal}, ...
     allCorrILI{ianimal}, allGrDecel{ianimal}, allGrMinVel{ianimal}, allTrialDuration{ianimal}, ...
     allGoNogo{ianimal}] = pooledDataVR(vrSessions);
end
close hWait
warning on

% --- Consolidate data after pooledDataVR ---
numAnimals = length(all_sesnames);
animalData = struct(); % Initialize a struct array

for ianimal = 1:numAnimals
    % Store performance and Go/NoGo status
    animalData(ianimal).Performance = allPerformance{ianimal};
    animalData(ianimal).GoNogo = logical(allGoNogo{ianimal}); % Ensure logical

    % Store other metrics using dynamic field names
    % Make sure metricNames includes ALL variables returned by pooledDataVR
    % that you might want to use later. Adjust metricNames definition accordingly.
    allMetricNames = {'allRZLick', 'allRZVel', 'allDwell', ...
                     'allPreRZLick', 'allPreRZVel', 'allGratingVel', 'allStimuli', ...
                     'allGrVelFirst', 'allGrVelSecond', 'allCorrLickLatency', ...
                     'allCorrILI', 'allGrDecel', 'allGrMinVel', 'allTrialDuration'}; % Add any missing ones
    allMetricVariables = {allRZLick, allRZVel, allDwell, allPreRZLick, allPreRZVel, ...
                         allGratingVel, allStimuli, allGrVelFirst, allGrVelSecond, ...
                         allCorrLickLatency, allCorrILI, allGrDecel, allGrMinVel, ...
                         allTrialDuration}; % Corresponding variables

    for k = 1:length(allMetricNames)
        fieldName = strrep(allMetricNames{k}, 'all', ''); % Create cleaner field names like 'RZLick'
        animalData(ianimal).(fieldName) = allMetricVariables{k}{ianimal};
    end
end

%% Define metric names and labels
animalIDs = {'Cb11', 'Cb14', 'Cb15', 'Cb17'};
% Performance is our reference (y-axis) so metrics start at index 2.
metricNames = {'allPerformance', 'allRZLick', 'allRZVel', 'allDwell', ...
    'allPreRZLick', 'allPreRZVel', 'allGratingVel', 'allTrialDuration', 'allGoNogo'};
metricLabels = {'Performance', 'Reward Zone Licks', 'Reward Zone Velocity', 'Dwell Time', ...
    'Pre-RZ Licks', 'Pre-RZ Velocity', 'Grating Velocity', 'Trial Duration', 'Go/NoGo'};
nMetrics = numel(metricNames);

nquantiles = 7;

% For each animal, create a figure with a tiled layout of plots for each metric.
% (We use "allPerformance" as the performance vector.)
for a = 1:length(all_sesnames)
    perfVec = allPerformance{a};    
    % For go/no-go separation, we assume allGoNogo{a} is a logical vector: true = go, false = no-go.
    goIdx = logical(allGoNogo{a});
    nogoIdx = ~allGoNogo{a};
    
    % Create a new figure for this animal.
    figure('Name', ['Animal ', animalIDs{a}]);
    t = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, ['Metrics vs Performance for Animal ', animalIDs{a}]);
    
    % Loop over metrics (skip performance at m=1)
    for m = 2:nMetrics-1  % skip 'allGoNogo' since that is used for grouping
        % Get the metric vector.
        switch metricNames{m}
            case 'allRZLick'
                data = allRZLick{a};
            case 'allRZVel'
                data = allRZVel{a};
            case 'allDwell'
                data = allDwell{a};
            case 'allPreRZLick'
                data = allPreRZLick{a};
            case 'allPreRZVel'
                data = allPreRZVel{a};
            case 'allGratingVel'
                data = allGratingVel{a};
            case 'allTrialDuration'
                data = allTrialDuration{a};
            otherwise
                data = [];
        end
        
        % Separate the data into go and no-go.
        data_go = data(goIdx);
        perf_go = perfVec(goIdx);
        data_nogo = data(nogoIdx);
        perf_nogo = perfVec(nogoIdx);
        
        % Quantile binning for go trials.
        if ~isempty(data_go)
            edges_go = quantile(data_go, linspace(0, 1, nquantiles + 1));
            bins_go = discretize(data_go, edges_go);
            quantCenters_go = nan(nquantiles,1);
            quantPerf_go = nan(nquantiles,1);
            for q = 1:nquantiles
                idx = find(bins_go == q);
                if ~isempty(idx)
                    quantCenters_go(q) = median(data_go(idx));
                    quantPerf_go(q) = mean(perf_go(idx));
                end
            end
        else
            quantCenters_go = []; quantPerf_go = [];
        end
        
        % Quantile binning for no-go trials.
        if ~isempty(data_nogo)
            data_nogo = data_nogo(~isnan(data_nogo));
            edges_nogo = quantile(data_nogo, linspace(0, 1, nquantiles + 1));
            bins_nogo = discretize(data_nogo, edges_nogo);
            quantCenters_nogo = nan(nquantiles,1);
            quantPerf_nogo = nan(nquantiles,1);
            for q = 1:nquantiles
                idx = find(bins_nogo == q);
                if ~isempty(idx)
                    quantCenters_nogo(q) = median(data_nogo(idx));
                    quantPerf_nogo(q) = mean(perf_nogo(idx));
                end
            end
        else
            quantCenters_nogo = []; quantPerf_nogo = [];
        end
        
        % Create a subplot for this metric.
        nexttile;
        hold on;
        % Plot go trials (blue) and no-go trials (red)
        scatter(quantCenters_go, quantPerf_go, 50, 'filled', 'MarkerFaceColor','b');
        lsline
        scatter(quantCenters_nogo, quantPerf_nogo, 50, 'filled', 'MarkerFaceColor','r');
        lsline
        ylim([0, 1])
        
        
        xlabel(metricLabels{m});
        ylabel('Performance');
        title(metricLabels{m});
        hold off;
    end

    % Figure 2: Full Distribution Histograms for Each Metric
    figure('Name', ['Animal ', animalIDs{a}, ' - Distribution']);
    t2 = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t2, ['Full Metric Distributions (Go vs No-Go) for Animal ', animalIDs{a}]);
    
    for m = 2:nMetrics-1  % Again, skip performance and the go/nogo flag
        % Get the metric vector.
        switch metricNames{m}
            case 'allRZLick'
                data = allRZLick{a};
            case 'allRZVel'
                data = allRZVel{a};
            case 'allDwell'
                data = allDwell{a};
            case 'allPreRZLick'
                data = allPreRZLick{a};
            case 'allPreRZVel'
                data = allPreRZVel{a};
            case 'allGratingVel'
                data = allGratingVel{a};
            case 'allTrialDuration'
                data = allTrialDuration{a};
            otherwise
                data = [];
        end
        
        % Separate the data into go and no-go.
        data_go = data(goIdx);
        data_nogo = data(nogoIdx);
        
        nexttile;
        hold on;
        % Plot histograms: adjust number of bins as needed.
        histogram(data_go, 25, 'Normalization', 'probability', 'FaceColor', 'b', 'FaceAlpha', 0.5);
        histogram(data_nogo, 25, 'Normalization', 'probability', 'FaceColor', 'r', 'FaceAlpha', 0.5);
        xlabel(metricLabels{m});
        ylabel('Count');
        title(metricLabels{m});
        
        hold off;
    end
    legend({'Go', 'No-Go'}, 'Location', 'eastoutside');
end

%% --- Correlation Analysis of Predictors ---
predictorNames = {'allRZLick', 'allRZVel', 'allDwell', 'allPreRZLick', ...
    'allPreRZVel', 'allGratingVel', 'allGrVelFirst', 'allGrVelSecond', 'allGrMinVel', 'allTrialDuration'};

for a = 1:length(all_sesnames)
    fprintf('  Calculating predictor correlations...\n');

    % Extract predictor data into a matrix
    predictorData = [allRZLick{a}(:), allRZVel{a}(:), allDwell{a}(:), ...
                     allPreRZLick{a}(:), allPreRZVel{a}(:), ...
                     allGratingVel{a}(:), allGrVelFirst{a}(:), allGrVelSecond{a}(:), ...
                     allGrMinVel{a}(:), allTrialDuration{a}(:)];

    % Calculate the correlation matrix
    corrMatrix = corr(predictorData, 'rows','complete'); % 'rows','complete' handles NaNs if present

    % Visualize the correlation matrix using heatmap
    figure
    h = imagesc(corrMatrix);
    colormap parula
    title(['Correlation Matrix of Predictors - Animal ', animalIDs{a}]);
    colorbar; % Show color scale
    
    % Display correlation values on the heatmap (for better readability)
    textStrings = num2str(corrMatrix(:), '%0.2f'); % Format correlations to 2 decimal places
    textStrings = strtrim(cellstr(textStrings)); % Remove spaces
    [x, y] = meshgrid(1:nPred); % Create x and y coordinates for placing text
    % hStrings = text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center');
    
    xticklabels(predictorNames)
    yticklabels(predictorNames)
end

%% Set up GLM analyses for each animal using 5-fold cross-validation, separated by Go/NoGo.
% Define ALL predictor variable names (as in your pooled data structure).
predictorNames = {'allRZLick', 'allRZVel', 'allDwell', 'allPreRZLick', ...
    'allPreRZVel', 'allGratingVel', 'allGrVelFirst', 'allGrVelSecond', 'allGrMinVel', 'allTrialDuration'};

% --- FLEXIBLE PREDICTOR SELECTION ---
% Choose which predictors to INCLUDE in the GLM analysis.
selectedPredictorNames = {'allPreRZLick', 'allPreRZVel', 'allGratingVel', 'allGrMinVel'};
% -------------------------------------

nPred = numel(selectedPredictorNames); % Number of SELECTED predictors
nFolds = 5;

% Preallocate arrays to hold cross-validated R² for single-predictor models, separated by Go/NoGo.
cvR2_single_go = nan(nPred, length(all_sesnames));
cvR2_single_nogo = nan(nPred, length(all_sesnames));
% Preallocate for the full model R², separated by Go/NoGo.
cvR2_full_go = nan(length(all_sesnames),1);
cvR2_full_nogo = nan(length(all_sesnames),1);
% Preallocate delta-R² (full model minus reduced model) for each predictor, separated by Go/NoGo.
deltaR2_go = nan(nPred, length(all_sesnames));
deltaR2_nogo = nan(nPred, length(all_sesnames));

% Preallocate arrays to hold coefficients
coeff_single_go = nan(nPred, length(all_sesnames));
coeff_single_nogo = nan(nPred, length(all_sesnames));
coeff_full_go = nan(nPred + 1, length(all_sesnames)); % +1 for intercept
coeff_full_nogo = nan(nPred + 1, length(all_sesnames)); % +1 for intercept


% Loop over animals.
for a = 1:length(all_sesnames)
    fprintf('Analyzing animal %d of %d...\n', a, length(all_sesnames));
    % Separate Go/Nogo trials indices
    goIdx = logical(allGoNogo{a});
    nogoIdx = ~allGoNogo{a};
    % Separate data for Go trials
    allPerformance_go = allPerformance{a}(goIdx);
    allRZLick_go = allRZLick{a}(goIdx);
    allRZVel_go = allRZVel{a}(goIdx);
    allDwell_go = allDwell{a}(goIdx);
    allPreRZLick_go = allPreRZLick{a}(goIdx);
    allPreRZVel_go = allPreRZVel{a}(goIdx);
    allGratingVel_go = allGratingVel{a}(goIdx);
    allGrVelFirst_go = allGrVelFirst{a}(goIdx);
    allGrVelSecond_go = allGrVelSecond{a}(goIdx);
    allGrMinVel_go = allGrMinVel{a}(goIdx);
    allTrialDuration_go = allTrialDuration{a}(goIdx);
    % Create table for animal "a" and Go trials
    T_go = table(allPerformance_go(:), ...
        allRZLick_go(:), allRZVel_go(:), allDwell_go(:), ...
        allPreRZLick_go(:), allPreRZVel_go(:), ...
        allGratingVel_go(:), allGrVelFirst_go(:), allGrVelSecond_go(:), ...
        allGrMinVel_go(:), allTrialDuration_go(:), ...
        'VariableNames', [{'Outcome'}, predictorNames]); % Use ALL predictor names to create table
    T_go = rmmissing(T_go); % Optionally remove rows with NaNs.
    cvp_go = cvpartition(height(T_go), 'KFold', nFolds); % Create CV partition for Go trials
    % --- 1) Single Predictor GLMs for Go Trials ---
    fprintf(' Analyzing Go Trials:\n');
    for p = 1:nPred
        currentPredictorName = selectedPredictorNames{p}; % Get the currently SELECTED predictor name
        fprintf('  Single-predictor model using "%s" (%d of %d predictors)\n', currentPredictorName, p, nPred);
        r2_folds_go = nan(nFolds, 1);
        current_coeff_go = nan(nFolds, 1); % Store coefficients for single predictor model
        for fold = 1:nFolds
            trainIdx = training(cvp_go, fold);
            testIdx = test(cvp_go, fold);
            mdl_go = fitglm(T_go(trainIdx, currentPredictorName), T_go.Outcome(trainIdx), 'Distribution', 'binomial', 'LikelihoodPenalty', 'jeffreys-prior', 'Options', statset('MaxIter', 1000));
            ypred_go = predict(mdl_go, T_go(testIdx, currentPredictorName));
            ytrue = T_go.Outcome(testIdx);
            sse = sum((ytrue - ypred_go).^2);
            sst = sum((ytrue - mean(ytrue)).^2);
            r2_folds_go(fold) = 1 - sse/sst;
            current_coeff_go(fold) = mdl_go.Coefficients.Estimate(2); % Store coefficient, index 2 is for the predictor (index 1 is intercept)
        end
        r2_folds_go(isinf(r2_folds_go)) = nan;
        cvR2_single_go(p, a) = mean(r2_folds_go, 'omitmissing');
        coeff_single_go(p, a) = mean(current_coeff_go, 'omitmissing'); % Average coefficient across folds
    end
    % --- 2) Full Model GLM for Go Trials ---
    r2_full_folds_go = nan(nFolds, 1);
    fprintf('  Full model using all %d predictors...\n', nPred);
    for fold = 1:nFolds
        fprintf('   Fold %d of %d...\n', fold, nFolds);
        trainIdx = training(cvp_go, fold);
        testIdx = test(cvp_go, fold);
        mdl_full_go = fitglm(T_go(trainIdx, selectedPredictorNames), T_go.Outcome(trainIdx), 'Distribution', 'binomial', 'LikelihoodPenalty', 'jeffreys-prior', 'Options', statset('MaxIter', 1000));
        ypred_full_go = predict(mdl_full_go, T_go(testIdx, selectedPredictorNames));
        ytrue = T_go.Outcome(testIdx);
        sse = sum((ytrue - ypred_full_go).^2);
        sst = sum((ytrue - mean(ytrue)).^2);
        r2_full_folds_go(fold) = 1 - sse/sst;
        coeff_full_go(:, a) = mean(mdl_full_go.Coefficients.Estimate); % Store coefficients for full model (intercept + all predictors), averaged across folds - although technically coefficients should be similar across folds in CV. Taking mean for consistency.
    end
    r2_full_folds_go(isinf(r2_full_folds_go)) = nan;
    cvR2_full_go(a) = mean(r2_full_folds_go, 'omitmissing');
    % --- 3) Knockout Analysis for Go Trials ---
    fprintf('  Knockout analysis...\n');
    for p = 1:nPred
        currentPredictorName = selectedPredictorNames{p}; % Get the currently SELECTED predictor name
        fprintf('   Removing predictor "%s"\n', currentPredictorName);
        r2_red_folds_go = nan(nFolds, 1);
        predNamesRed = selectedPredictorNames; % Start with the SELECTED predictors
        predNamesRed(p) = []; % Remove the current predictor for knockout
        for fold = 1:nFolds
            fprintf('    Fold %d of %d...\n', fold, nFolds);
            trainIdx = training(cvp_go, fold);
            testIdx = test(cvp_go, fold);
            mdl_red_go = fitglm(T_go(trainIdx, predNamesRed), T_go.Outcome(trainIdx), 'Distribution', 'binomial', 'LikelihoodPenalty', 'jeffreys-prior', 'Options', statset('MaxIter', 1000));
            ypred_red_go = predict(mdl_red_go, T_go(testIdx, predNamesRed));
            ytrue = T_go.Outcome(testIdx);
            sse = sum((ytrue - ypred_red_go).^2);
            sst = sum((ytrue - mean(ytrue)).^2);
            r2_red_folds_go(fold) = 1 - sse/sst;
        end
        r2_red_folds_go(isinf(r2_red_folds_go)) = nan;
        r2_red_go = mean(r2_red_folds_go, 'omitmissing');
        deltaR2_go(p, a) = cvR2_full_go(a) - r2_red_go;
    end
    % Separate data for NoGo trials
    allPerformance_nogo = allPerformance{a}(nogoIdx);
    allRZLick_nogo = allRZLick{a}(nogoIdx);
    allRZVel_nogo = allRZVel{a}(nogoIdx);
    allDwell_nogo = allDwell{a}(nogoIdx);
    allPreRZLick_nogo = allPreRZLick{a}(nogoIdx);
    allPreRZVel_nogo = allPreRZVel{a}(nogoIdx);
    allGratingVel_nogo = allGratingVel{a}(nogoIdx);
    allGrVelFirst_nogo = allGrVelFirst{a}(nogoIdx);
    allGrVelSecond_nogo = allGrVelSecond{a}(nogoIdx);
    allGrMinVel_nogo = allGrMinVel{a}(nogoIdx);
    allTrialDuration_nogo = allTrialDuration{a}(nogoIdx);
    % Create table for animal "a" and NoGo trials
    T_nogo = table(allPerformance_nogo(:), ...
        allRZLick_nogo(:), allRZVel_nogo(:), allDwell_nogo(:), ...
        allPreRZLick_nogo(:), allPreRZVel_nogo(:), ...
        allGratingVel_nogo(:), allGrVelFirst_nogo(:), allGrVelSecond_nogo(:), ...
        allGrMinVel_nogo(:), allTrialDuration_nogo(:), ...
        'VariableNames', [{'Outcome'}, predictorNames]); % Use ALL predictor names to create table
    T_nogo = rmmissing(T_nogo); % Optionally remove rows with NaNs.
    cvp_nogo = cvpartition(height(T_nogo), 'KFold', nFolds); % Create CV partition for NoGo trials
    % --- 1) Single Predictor GLMs for NoGo Trials ---
    fprintf(' Analyzing No-Go Trials:\n');
    for p = 1:nPred
        currentPredictorName = selectedPredictorNames{p}; % Get the currently SELECTED predictor name
        fprintf('  Single-predictor model using "%s" (%d of %d predictors)\n', currentPredictorName, p, nPred);
        r2_folds_nogo = nan(nFolds, 1);
        current_coeff_nogo = nan(nFolds, 1); % Store coefficients for single predictor model
        for fold = 1:nFolds
            trainIdx = training(cvp_nogo, fold);
            testIdx = test(cvp_nogo, fold);
            mdl_nogo = fitglm(T_nogo(trainIdx, currentPredictorName), T_nogo.Outcome(trainIdx), 'Distribution', 'binomial', 'LikelihoodPenalty', 'jeffreys-prior', 'Options', statset('MaxIter', 1000));
            ypred_nogo = predict(mdl_nogo, T_nogo(testIdx, currentPredictorName));
            ytrue = T_nogo.Outcome(testIdx);
            sse = sum((ytrue - ypred_nogo).^2);
            sst = sum((ytrue - mean(ytrue)).^2);
            r2_folds_nogo(fold) = 1 - sse/sst;
            current_coeff_nogo(fold) = mdl_nogo.Coefficients.Estimate(2); % Store coefficient
        end
        cvR2_single_nogo(p, a) = mean(r2_folds_nogo);
        coeff_single_nogo(p, a) = mean(current_coeff_nogo, 'omitmissing'); % Average coefficient across folds
    end
    % --- 2) Full Model GLM for NoGo Trials ---
    r2_full_folds_nogo = nan(nFolds, 1);
    fprintf('  Full model using all %d predictors...\n', nPred);
    for fold = 1:nFolds
        fprintf('   Fold %d of %d...\n', fold, nFolds);
        trainIdx = training(cvp_nogo, fold);
        testIdx = test(cvp_nogo, fold);
        mdl_full_nogo = fitglm(T_nogo(trainIdx, selectedPredictorNames), T_nogo.Outcome(trainIdx), 'Distribution', 'binomial', 'LikelihoodPenalty', 'jeffreys-prior', 'Options', statset('MaxIter', 1000));
        ypred_full_nogo = predict(mdl_full_nogo, T_nogo(testIdx, selectedPredictorNames));
        ytrue = T_nogo.Outcome(testIdx);
        sse = sum((ytrue - ypred_full_nogo).^2);
        sst = sum((ytrue - mean(ytrue)).^2);
        r2_full_folds_nogo(fold) = 1 - sse/sst;
        coeff_full_nogo(:, a) = mean(mdl_full_nogo.Coefficients.Estimate); % Store coefficients for full model
    end
    cvR2_full_nogo(a) = mean(r2_full_folds_nogo);
    % --- 3) Knockout Analysis for NoGo Trials ---
    fprintf('  Knockout analysis...\n');
    for p = 1:nPred
        currentPredictorName = selectedPredictorNames{p}; % Get the currently SELECTED predictor name
        fprintf('   Removing predictor "%s"\n', currentPredictorName);
        r2_red_folds_nogo = nan(nFolds, 1);
        predNamesRed = selectedPredictorNames; % Start with SELECTED predictors
        predNamesRed(p) = []; % Remove the current predictor for knockout
        for fold = 1:nFolds
            fprintf('    Fold %d of %d...\n', fold, nFolds);
            trainIdx = training(cvp_nogo, fold);
            testIdx = test(cvp_nogo, fold);
            mdl_red_nogo = fitglm(T_nogo(trainIdx, predNamesRed), T_nogo.Outcome(trainIdx), 'Distribution', 'binomial', 'LikelihoodPenalty', 'jeffreys-prior', 'Options', statset('MaxIter', 1000));
            ypred_red_nogo = predict(mdl_red_nogo, T_nogo(testIdx, predNamesRed));
            ytrue = T_nogo.Outcome(testIdx);
            sse = sum((ytrue - ypred_red_nogo).^2);
            sst = sum((ytrue - mean(ytrue)).^2);
            r2_red_folds_nogo(fold) = 1 - sse/sst;
        end
        r2_red_nogo = mean(r2_red_folds_nogo);
        deltaR2_nogo(p, a) = cvR2_full_nogo(a) - r2_red_nogo;
    end
end

% Display results for Go Trials
fprintf('\nGo Trials - Single Measure GLMs (CV R²):\n');
disp(cvR2_single_go);
fprintf('Go Trials - Full Model CV R²:\n');
disp(cvR2_full_go);
fprintf('Go Trials - Delta-R² (Unique Variance) for each predictor:\n');
disp(deltaR2_go);
% Display results for NoGo Trials
fprintf('\nNo-Go Trials - Single Measure GLMs (CV R²):\n');
disp(cvR2_single_nogo);
fprintf('No-Go Trials - Full Model CV R²:\n');
disp(cvR2_full_nogo);
fprintf('No-Go Trials - Delta-R² (Unique Variance) for each predictor:\n');
disp(deltaR2_nogo);

figure
subplot(2, 3, 1) % Modified subplot layout to accommodate coefficients
my_errorbar_plot(cvR2_single_go')
ylabel('cvR^2');
xticklabels(selectedPredictorNames) % Use SELECTED predictor names for labels
title('Go Trials - Single Predictor CV R^2')

subplot(2, 3, 2) % New subplot for coefficients - Single Predictor Go
my_errorbar_plot(coeff_single_go')
ylabel('Coefficient Value');
xticklabels(selectedPredictorNames)
title('Go Trials - Single Predictor Coeff')
yline(0, '--', 'Color', [0.5 0.5 0.5]); % Add zero line for reference


subplot(2, 3, 4) % Modified subplot layout
my_errorbar_plot(cvR2_single_nogo')
ylabel('cvR^2');
xticklabels(selectedPredictorNames) % Use SELECTED predictor names for labels
title('No-Go Trials - Single Predictor CV R^2')

subplot(2, 3, 5) % New subplot for coefficients - Single Predictor NoGo
my_errorbar_plot(coeff_single_nogo')
ylabel('Coefficient Value');
xticklabels(selectedPredictorNames)
title('No-Go Trials - Single Predictor Coeff')
yline(0, '--', 'Color', [0.5 0.5 0.5]); % Add zero line


subplot(2, 3, 3)
my_errorbar_plot(-deltaR2_go')
ylabel('\DeltaR^2');
xticklabels(selectedPredictorNames) % Use SELECTED predictor names for labels
title('Go Trials - Delta R^2')


subplot(2, 3, 6)
my_errorbar_plot(-deltaR2_nogo')
ylabel('\DeltaR^2');
xticklabels(selectedPredictorNames) % Use SELECTED predictor names for labels
title('No-Go Trials - Delta R^2')


