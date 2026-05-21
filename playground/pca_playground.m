%% Load

load("/Users/theoamvr/Desktop/Experiments/UncertaintyV1/data/VR_Decoder_Data_Export.mat")

%% 

ianimal = 3;

current_posterior = IO.animals{ianimal}.inferred.post_s_marginal;
current_likelihood = IO.animals{ianimal}.inferred.L_s_marginal;


%%

figure
t = tiledlayout('flow', 'TileSpacing', 'compact');

nexttile
plot(current_posterior')
axis tight

nexttile
plot(current_likelihood')
axis tight

%% pca
warning off

[coeff,score,latent,tsquared,explained,mu] = pca(current_posterior, "NumComponents", 2, "Centered", true);

figure
t = tiledlayout('flow', 'TileSpacing', 'compact');

nexttile
scatter(score(:, 1), score(:, 2))
hold on
example1 = 50;
example2 = 220;
projected_example1 = current_posterior(example1, :)*coeff;
projected_example2 = current_posterior(example2, :)*coeff;
scatter(projected_example1(1), projected_example1(2), 80, 'red', '*')
scatter(projected_example2(1), projected_example2(2), 80, 'green', '*')
title('non-centred')

nexttile
scatter(score(:, 1), score(:, 2))
hold on
example1 = 50;
example2 = 220;
projected_example1 = (current_posterior(example1, :)-mean(current_posterior))*coeff;
projected_example2 = (current_posterior(example2, :)-mean(current_posterior))*coeff;
scatter(projected_example1(1), projected_example1(2), 80, 'red', '*')
scatter(projected_example2(1), projected_example2(2), 80, 'green', '*')
title('centred')
