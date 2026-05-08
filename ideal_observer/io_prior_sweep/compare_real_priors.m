function compare_real_priors()
%% compare_real_priors  Cross-prior comparison of v2 fits on real data.
%
% Loads all four IOResults_<tag>.mat files in this directory, prints a
% per-animal × prior summary table, and saves figures under figures/.
%
% Reports for each prior:
%   - Stage 1 cross-validated NLL per animal (lower = better fit).
%   - Stage 2 all-data NLL per animal (held-out CV NLL not in IOResults
%     today — flagged in NOTES).
%   - Fitted Stage 1 params: kappa_amp, c_power, d_power, vel_slope,
%     vel_intercept, vel_std.
%   - Fitted Stage 2 params: alpha_r, beta_r, gamma_r, delta_r.

this_dir = fileparts(mfilename('fullpath'));

tags = {'Flat', 'Bimodal_k1', 'Bimodal_k3', 'Bimodal_k5'};
n_priors = numel(tags);

% Load each file
results = cell(n_priors, 1);
for i = 1:n_priors
    fp = fullfile(this_dir, sprintf('IOResults_%s.mat', tags{i}));
    if ~exist(fp, 'file')
        warning('Missing %s — skipping', fp);
        continue;
    end
    L = load(fp, 'IOResults');
    results{i} = L.IOResults;
end

% Determine animal count from the first non-empty result
ref = []; for i = 1:n_priors, if ~isempty(results{i}), ref = results{i}; break; end; end
if isempty(ref), error('No IOResults files found in %s', this_dir); end
n_animals = numel(ref.animals);

stage1_names = ref.meta.model_spec.fit_params;
stage2_names = {'alpha_r','beta_r','gamma_r','delta_r'};
n_p1 = numel(stage1_names);
n_p2 = numel(stage2_names);

% Stack into [n_animals × n_priors] matrices
stage1_nll  = nan(n_animals, n_priors);
stage2_nll  = nan(n_animals, n_priors);
stage1_pars = nan(n_animals, n_p1, n_priors);
stage2_pars = nan(n_animals, n_p2, n_priors);
animal_tags = cell(n_animals, 1);

for i = 1:n_priors
    if isempty(results{i}), continue; end
    for a = 1:n_animals
        ani = results{i}.animals{a};
        if i == 1, animal_tags{a} = ani.tag; end
        stage1_nll(a, i) = ani.fit.avg_test_nll;
        stage2_nll(a, i) = ani.fit.choice_nll;
        stage1_pars(a, :, i) = ani.fit.params_vec(1:n_p1);
        if isfield(ani.fit, 'choice_vec') && numel(ani.fit.choice_vec) >= 4
            stage2_pars(a, :, i) = ani.fit.choice_vec(:)';
        end
    end
end

% =====================================================================
% Print summary tables
% =====================================================================
fprintf('\n================================================================\n');
fprintf(' Stage 1 avg_test_nll  (lower = better; bold = best per row)\n');
fprintf('================================================================\n');
print_table(stage1_nll, animal_tags, tags, true);

fprintf('\n================================================================\n');
fprintf(' Stage 2 choice_nll (all-data) (lower = better; bold = best per row)\n');
fprintf('================================================================\n');
print_table(stage2_nll, animal_tags, tags, true);

% Per-prior aggregate
fprintf('\n================================================================\n');
fprintf(' Aggregate across animals (mean ± SEM)\n');
fprintf('================================================================\n');
fprintf('%-22s', 'Prior');
fprintf('%14s%14s\n', 'Stage 1 NLL', 'Stage 2 NLL');
for i = 1:n_priors
    if all(isnan(stage1_nll(:,i))), continue; end
    s1 = stage1_nll(:,i); s2 = stage2_nll(:,i);
    fprintf('%-22s', tags{i});
    fprintf('  %6.2f ± %5.2f  %6.2f ± %5.2f\n', ...
        mean(s1,'omitnan'), sem_(s1), mean(s2,'omitnan'), sem_(s2));
end

% =====================================================================
% Figures
% =====================================================================
fig_dir = fullfile(this_dir, 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

% --- Figure 1: Stage 1 / Stage 2 NLL per animal across priors ---
f1 = figure('Color','w','Position',[100 100 1200 500],'Name','NLL by prior');
subplot(1,2,1);
plot_per_animal_lines(stage1_nll, animal_tags, tags, ...
    'Stage 1 NLL (avg test, vel only)', 'Prior', 'NLL (nats)');
subplot(1,2,2);
plot_per_animal_lines(stage2_nll, animal_tags, tags, ...
    'Stage 2 NLL (all-data choice)', 'Prior', 'NLL (nats)');
saveas(f1, fullfile(fig_dir, 'real_nll_by_prior.png'));
fprintf('\nSaved %s\n', fullfile(fig_dir, 'real_nll_by_prior.png'));

% --- Figure 2: Fitted Stage 1 params per animal across priors ---
f2 = figure('Color','w','Position',[100 100 1400 700],'Name','Stage 1 params by prior');
nr = 2; nc = ceil(n_p1/2);
for k = 1:n_p1
    subplot(nr, nc, k);
    plot_per_animal_lines(squeeze(stage1_pars(:,k,:)), animal_tags, tags, ...
        stage1_names{k}, 'Prior', stage1_names{k});
end
sgtitle('Stage 1 fitted parameters across priors (per animal)');
saveas(f2, fullfile(fig_dir, 'real_stage1_params_by_prior.png'));
fprintf('Saved %s\n', fullfile(fig_dir, 'real_stage1_params_by_prior.png'));

% --- Figure 3: Fitted Stage 2 params per animal across priors ---
f3 = figure('Color','w','Position',[100 100 1200 600],'Name','Stage 2 params by prior');
for k = 1:n_p2
    subplot(2, 2, k);
    plot_per_animal_lines(squeeze(stage2_pars(:,k,:)), animal_tags, tags, ...
        stage2_names{k}, 'Prior', stage2_names{k});
end
sgtitle('Stage 2 fitted parameters across priors (per animal)');
saveas(f3, fullfile(fig_dir, 'real_stage2_params_by_prior.png'));
fprintf('Saved %s\n', fullfile(fig_dir, 'real_stage2_params_by_prior.png'));

% --- Figure 4: ΔNLL relative to a reference prior (Bimodal_k3) ---
ref_idx = find(strcmp(tags, 'Bimodal_k3'));
if ~isempty(ref_idx) && ~all(isnan(stage1_nll(:,ref_idx)))
    f4 = figure('Color','w','Position',[100 100 1200 500],'Name','ΔNLL vs Bimodal{k=3}');
    subplot(1,2,1);
    delta1 = stage1_nll - stage1_nll(:, ref_idx);
    plot_per_animal_lines(delta1, animal_tags, tags, ...
        sprintf('Stage 1 ΔNLL vs %s', tags{ref_idx}), 'Prior', 'ΔNLL (nats)');
    yline(0,'k--');
    subplot(1,2,2);
    delta2 = stage2_nll - stage2_nll(:, ref_idx);
    plot_per_animal_lines(delta2, animal_tags, tags, ...
        sprintf('Stage 2 ΔNLL vs %s', tags{ref_idx}), 'Prior', 'ΔNLL (nats)');
    yline(0,'k--');
    saveas(f4, fullfile(fig_dir, 'real_delta_nll_vs_bimodal_k3.png'));
    fprintf('Saved %s\n', fullfile(fig_dir, 'real_delta_nll_vs_bimodal_k3.png'));
end

% Save the raw comparison data alongside the figures
out_mat = fullfile(fig_dir, 'real_prior_comparison.mat');
save(out_mat, 'tags', 'animal_tags', 'stage1_nll', 'stage2_nll', ...
    'stage1_pars', 'stage2_pars', 'stage1_names', 'stage2_names', '-v7.3');
fprintf('Saved raw comparison data: %s\n', out_mat);
fprintf('\nDone.\n');
end

% ---------------------------------------------------------------------
function s = sem_(x)
x = x(~isnan(x));
if isempty(x), s = NaN; else, s = std(x) / sqrt(numel(x)); end
end

function print_table(M, row_names, col_names, mark_min)
% M: [n_rows × n_cols] numeric matrix
% Print rows of M with row_names and col_names.
% mark_min = true: prepend '*' to the min in each row.
if nargin < 4, mark_min = false; end
n_rows = size(M,1); n_cols = size(M,2);
fprintf('%-12s', 'Animal');
for c = 1:n_cols, fprintf('%14s', col_names{c}); end
fprintf('\n');
for r = 1:n_rows
    fprintf('%-12s', row_names{r});
    [~, min_c] = min(M(r,:));
    for c = 1:n_cols
        v = M(r,c);
        if isnan(v)
            fprintf('%14s', '—');
        elseif mark_min && c == min_c
            fprintf('       *%6.2f', v);
        else
            fprintf('       %7.2f', v);
        end
    end
    fprintf('\n');
end
end

function plot_per_animal_lines(M, row_names, col_names, ttl, xlab, ylab)
% Each row = one animal; each column = one prior. Lines connect priors
% within-animal so you can see the swing.
[n_rows, n_cols] = size(M);
hold on;
x = 1:n_cols;
colors = lines(n_rows);
for r = 1:n_rows
    plot(x, M(r,:), 'o-', 'Color', colors(r,:), 'LineWidth', 1.2, ...
        'MarkerFaceColor', colors(r,:), 'DisplayName', row_names{r});
end
% group mean
mu = mean(M, 1, 'omitnan');
plot(x, mu, 'k-', 'LineWidth', 2.5, 'DisplayName', 'mean');
set(gca,'XTick', x, 'XTickLabel', col_names, 'XTickLabelRotation', 30);
title(ttl, 'Interpreter', 'none');
xlabel(xlab); ylabel(ylab);
grid on; box on; xlim([0.5, n_cols+0.5]);
if n_rows <= 8, legend('Location','bestoutside','Interpreter','none','FontSize',7); end
end
