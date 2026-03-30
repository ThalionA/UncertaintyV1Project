%% batchGenerateStimulusWorlds_named.m
% Assumptions
%   – exper.worlds{1}  : corridor baseline
%   – exper.worlds{2}  : timeout (“black”) world
%   – exper.worlds{3}  : template stimulus world (two walls at indices 1 & 2)
%   – Texture images live in ./stim_textures/
%     and are named: grating_angle%d_contrast%.2f_disp%d.png
%   – ViRMEn is on MATLAB path

clearvars;

%% 1. Load existing experiment
load('C:/Users/roche/Desktop/Experiments/ViRMEn/experiments/NewExperimentTheo25_baseline.mat','exper');      % <- update if needed
exper.enableCallbacks;                           % re‑initialise callbacks

%% 2. Grab template world
baseW   = exper.worlds{3};
wallIdx = [1 2];                                  % adjust if walls differ
imgDir  = 'C:/Users/roche/Desktop/Experiments/2afcanalysis/Actual2AFCexpcode/stimTexturesVR2';

%% 3. Define stimulus grid
angles = [0 15 30 40 45 50 60 75 90];
% pairs  = [ 1    5 ;
%            1   45 ;
%            0.5 30 ;
%            0.5 90 ;
%            0.25 5 ;
%            0.25 45;
%            0.01 30 ;
%            0.01 90];   % [contrast dispersion]

pairs  = [ 1   5 ;
           1   30 ;
           1   90 ;
           0.5 5 ;
           0.5 45 ;
           0.25 30 ;
           0.25 90;
           0.01 5 ;
           0.01 45];   % [contrast dispersion]

%% 4. Duplicate & customise worlds
for ip = 1:size(pairs,1)
    contrast   = pairs(ip,1);
    dispersion = pairs(ip,2);

    for a = 1:numel(angles)
        angle = angles(a);

        % 4a. Determine texture & world names
        fileName   = sprintf('grating_angle%d_contrast%.2f_disp%d.png', ...
                              angle, contrast, dispersion);
        worldLabel = sprintf('a%d_c%.2f_d%d',    ...
                              angle, contrast, dispersion);
        texPath = fullfile(imgDir, fileName);
        assert(isfile(texPath), 'Texture not found: %s', texPath);

        % 4b. Duplicate template
        newW      = baseW.copyItem;      % deep copy
        newW.name = worldLabel;

        % 4c. Load & compute texture  (pass raw RGB/grayscale image)
        tex = virmenTexture;
        img = imread(texPath);      % returns uint8 RGB or grayscale
        if ndims(img)==2           % grayscale → expand to RGB for consistency
            img = repmat(img,1,1,3);
        end
        tex.loadImage(img);        % virmenTexture expects a single img argument
        tex.compute;

        % 4d. Assign texture to both walls
        for w = wallIdx
            newW.objects{w}.setTexture(tex);
        end

        % 4e. Add to experiment
        exper.addWorld(newW);
    end
end

%% 5. Save augmented experiment
% save('C:/Users/roche/Desktop/Experiments/ViRMEn/experiments/NewExperimentTheo25_full1_v2.mat','exper','-v7.3');
save('C:/Users/roche/Desktop/Experiments/ViRMEn/experiments/NewExperimentTheo25_full2_v3.mat','exper','-v7.3');
fprintf('Added %d stimulus worlds \n', numel(angles) * size(pairs,1));