%% changeExperimentCode.m
% This script re-links a ViRMEn .mat file to use a new experiment code file.

clearvars;

% 1. Load the experiment file containing your worlds
fprintf('Loading experiment file...\n');
load('C:/Users/roche/Desktop/Experiments/ViRMEn/experiments/NewExperimentTheo25_full1_priorblocks.mat', 'exper');

% 2. Re-initialise callbacks (good practice)
exper.enableCallbacks;

% 3. Re-link the experiment code to your NEW .m file
%    (Make sure 'NewExperimentTheo25_psychometric.m' is on your MATLAB path)
fprintf('Re-linking to new code file...\n');
exper.experimentCode = @NewExperimentTheo25_full1_priorblocks;

% 4. Save as a new experiment file
newFilename = 'C:/Users/roche/Desktop/Experiments/ViRMEn/experiments/NewExperimentTheo25_full1_priorblocks.mat';
fprintf('Saving to %s...\n', newFilename);
save(newFilename, 'exper', '-v7.3');

fprintf('Experiment code successfully relinked!\n');