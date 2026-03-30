% Parameters for grating generation
dispersion = 90;    % dispersion value
contrast = 0;    % contrast value
phase = pi/12;
cpd = 0.04;
% angles = [0, 15, 30, 40, 45, 50, 60, 75, 90];
angles = [0];

% Desired output resolution
desiredWidth = 640;
desiredHeight = 360;

for iangle = 1:length(angles)
    angle = angles(iangle);
    grating = generate_mixed_grating_aperture_whisker(dispersion, contrast, angle, phase, cpd);

    % Resize the grating to the desired resolution using bilinear interpolation.
    grating_resized = imresize(grating, [desiredHeight desiredWidth], 'bilinear');

    % Normalize and convert to uint8 for saving
    grating_uint8 = uint8(grating_resized * 255);

    filename = sprintf('grating_angle%d_contrast%.2f_disp%d.png', angle, contrast, dispersion);
    imwrite(grating_uint8, filename);
    fprintf('Saved %s\n', filename);
end


function final_grating = generate_mixed_grating_aperture_whisker(dispersion, contrast, orientation, phase, cpd)
    if nargin < 5 || isempty(cpd)
        cpd = 0.04;
    end

    numAngles = 9;
    screens = Screen('Screens');
    lastScreen = max(screens);
    screenRect = Screen('Rect', lastScreen);
    grey = 0.5;
    
    % Monitor physical dimensions and viewing conditions
    mon_width = 53.2;      % in cm
    viewing_dist = 20;      % in cm
    mon_height = mon_width * 9/16;
    
    % Calculate visual angles
    visual_angle_h = 2 * atand(mon_width/(2*viewing_dist));
    visual_angle_v = 2 * atand(mon_height/(2*viewing_dist));
    
    screenXpixels = screenRect(3);
    screenYpixels = screenRect(4);
    
    scaling_factor = 2;
    screenDiagonal = sqrt(screenXpixels^2 + screenYpixels^2) * scaling_factor;
    
    [x, y] = meshgrid(1:ceil(screenDiagonal), 1:ceil(screenDiagonal));
    center = (size(x,1) + 1) / 2;
    x = x - center;
    y = y - center;
    
    % Convert coordinates to degrees
    ppd_x = screenXpixels / visual_angle_h;
    ppd_y = screenYpixels / visual_angle_v;
    deg_x = x / ppd_x;
    deg_y = y / ppd_y;
    
    % Calculate aspect ratio compensation factor
    aspect_ratio = screenXpixels / screenYpixels;
    orientation_rad = deg2rad(orientation);
    
    % Adjust spatial frequency based on orientation to maintain constant cycles
    % This compensates for the rectangular aspect ratio
    adjusted_cpd = cpd / sqrt((cos(orientation_rad))^2 + (sin(orientation_rad) * aspect_ratio)^2);
    adjusted_cpd = cpd;

    % Use dispersion to weight component gratings
    sig = dispersion;
    baseAngle = 0;
    angles = linspace(baseAngle - 180, baseAngle + 180, numAngles);
    alphaval = exp(-((angles - baseAngle).^2) / (2*sig^2));
    alphaval = alphaval / sum(alphaval);
    
    tot_m = zeros(size(x));
    % Sum sine wave components with different orientations
    for iAngle = 1:numAngles
        theta = deg2rad(angles(iAngle));
        % Use adjusted spatial frequency
        m = sin(2 * pi * adjusted_cpd * (cos(theta) .* deg_x + sin(theta) .* deg_y) + phase) * alphaval(iAngle);
        tot_m = tot_m + m;
    end
    
    % Threshold and set contrast
    tot_m = sign(tot_m);
    high_value = grey + contrast * grey;
    low_value = grey - contrast * grey;
    tot_m(tot_m == 1) = high_value;
    tot_m(tot_m == -1) = low_value;
    
    % Rotate the composite grating
    tot_m = imrotate(tot_m, orientation - 90, "bicubic", "crop");
    
    % Crop to screen dimensions
    center_row = ceil(size(tot_m, 1) / 2);
    center_col = ceil(size(tot_m, 2) / 2);
    start_row = center_row - floor(screenYpixels / 2);
    end_row = start_row + screenYpixels - 1;
    start_col = center_col - floor(screenXpixels / 2);
    end_col = start_col + screenXpixels - 1;
    
    start_row = max(start_row, 1);
    end_row = min(end_row, size(tot_m, 1));
    start_col = max(start_col, 1);
    end_col = min(end_col, size(tot_m, 2));
    
    final_grating = tot_m(start_row:end_row, start_col:end_col);
end