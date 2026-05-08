function plot_recovery_results(varargin)
%% plot_recovery_results  Render parameter-recovery validation figures.
%
% Scans ideal_observer/io_prior_sweep/ for RecoveryResults_<tag>.mat files
% and renders per-prior validation figures.
%
% Usage:
%   plot_recovery_results                    % all available tags
%   plot_recovery_results('tag','Bimodal_k3') % single tag
%   plot_recovery_results('verbose',false)
%
% Per tag, writes to ideal_observer/io_prior_sweep/figures/:
%   recovery_<tag>_stage1.png   — 6-panel true-vs-recovered scatter
%   recovery_<tag>_stage2.png   — 4-panel true-vs-recovered scatter
%   recovery_<tag>_kl.png       — KL(Q) and KL(L) summary
%
% Also prints a summary table per tag.

p = inputParser;
p.addParameter('tag', '', @ischar);   % '' = all
p.addParameter('verbose', true, @islogical);
p.parse(varargin{:});
opts = p.Results;

this_dir = fileparts(mfilename('fullpath'));
fig_dir  = fullfile(this_dir, 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

if isempty(opts.tag)
    files = dir(fullfile(this_dir, 'RecoveryResults_*.mat'));
    tags = cell(numel(files),1);
    for i = 1:numel(files)
        nm = files(i).name;
        tags{i} = regexprep(nm, '^RecoveryResults_(.+)\.mat$', '$1');
    end
else
    tags = {opts.tag};
end

if isempty(tags)
    error('plot_recovery_results: no RecoveryResults_*.mat in %s', this_dir);
end

for ti = 1:numel(tags)
    tag = tags{ti};
    fp = fullfile(this_dir, sprintf('RecoveryResults_%s.mat', tag));
    if ~exist(fp, 'file')
        warning('Missing %s — skipping', fp); continue;
    end
    L = load(fp, 'RecoveryResults');
    R = L.RecoveryResults;

    fprintf('\n================================================================\n');
    fprintf(' Recovery validation: %s (prior=%s, strength=%g)\n', ...
        tag, R.meta.prior_type, R.meta.prior_strength);
    fprintf('   n_random=%d, n_yoked=%d, n_trials=%d\n', ...
        R.meta.n_random, R.meta.n_yoked, R.meta.n_trials);
    fprintf('================================================================\n');

    plot_stage_scatter(R, 1, fig_dir, tag);
    plot_stage_scatter(R, 2, fig_dir, tag);
    plot_kl_diagnostics(R, fig_dir, tag);
    print_recovery_table(R);
end
fprintf('\nFigures and summaries written to %s\n', fig_dir);
end

% ---------------------------------------------------------------------
function plot_stage_scatter(R, stage, fig_dir, tag)
% True vs recovered scatter per parameter; random and yoked colored.

if stage == 1
    names = R.meta.stage1_names;
    X = R.true_params_stage1; Y = R.fitted_params_stage1;
    title_prefix = 'Stage 1 (vel-only)';
    fname = sprintf('recovery_%s_stage1.png', tag);
else
    names = R.meta.stage2_names;
    X = R.true_params_stage2; Y = R.fitted_params_stage2;
    title_prefix = 'Stage 2 (choice psych)';
    fname = sprintf('recovery_%s_stage2.png', tag);
end

n_p = numel(names);
n_animals = size(X,1);
yoked = R.is_yoked;

if stage == 1
    nr = 2; nc = ceil(n_p/2);
    fig_w = 1300; fig_h = 700;
else
    nr = 2; nc = 2;
    fig_w = 900; fig_h = 800;
end

f = figure('Color','w','Position',[100 100 fig_w fig_h], ...
    'Name', sprintf('%s — recovery (%s)', title_prefix, tag));

for k = 1:n_p
    subplot(nr, nc, k); hold on;
    x = X(:,k); y = Y(:,k);
    % Random in grey, yoked in red
    rnd = ~yoked; yk = yoked;
    if any(rnd)
        scatter(x(rnd), y(rnd), 50, 'o', ...
            'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.65 0.65 0.65], ...
            'DisplayName', 'Random');
    end
    if any(yk)
        scatter(x(yk), y(yk), 80, 'p', ...
            'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.85 0.20 0.20], ...
            'DisplayName', 'Empirically-yoked');
    end
    % y=x reference
    lo = min([x; y]); hi = max([x; y]);
    pad = (hi - lo) * 0.10 + eps;
    xy_lim = [lo - pad, hi + pad];
    plot(xy_lim, xy_lim, 'k--', 'HandleVisibility','off');

    if n_animals >= 3 && std(x) > 0
        r = corr(x, y);
    else
        r = NaN;
    end
    bias = mean(y - x);
    range_x = max(x) - min(x);
    rel_bias = bias / max(abs(range_x), eps);

    title(sprintf('%s   r=%+.2f, bias=%+.3f', names{k}, r, bias), ...
        'Interpreter','none','FontWeight','normal');
    xlabel(sprintf('true %s', names{k}), 'Interpreter','none');
    ylabel(sprintf('recovered %s', names{k}), 'Interpreter','none');
    axis square; box on; grid on;
    xlim(xy_lim); ylim(xy_lim);
    if k == 1, legend('Location','northwest'); end
end
sgtitle(sprintf('%s — parameter recovery (%s)', title_prefix, tag), 'Interpreter','none');

out_path = fullfile(fig_dir, fname);
saveas(f, out_path);
fprintf(' Saved %s\n', out_path);
end

% ---------------------------------------------------------------------
function plot_kl_diagnostics(R, fig_dir, tag)
% KL(Q_true || Q_fit) + KL(L_true || L_fit), pooled across all trials of
% all animals, summarised four ways: histogram, by orientation, contrast,
% dispersion.

n_animals = numel(R.per_animal);
all_KL_Q = []; all_KL_L = [];
all_ori = []; all_con = []; all_dis = [];
for i = 1:n_animals
    a = R.per_animal{i};
    all_KL_Q = [all_KL_Q; a.KL_Q]; %#ok<AGROW>
    all_KL_L = [all_KL_L; a.KL_L]; %#ok<AGROW>
    all_ori = [all_ori; a.data.orientation]; %#ok<AGROW>
    all_con = [all_con; a.data.contrast]; %#ok<AGROW>
    all_dis = [all_dis; a.data.dispersion]; %#ok<AGROW>
end

f = figure('Color','w','Position',[100 100 1500 800], ...
    'Name', sprintf('KL diagnostics (%s)', tag));

% Row 1: KL(Q)
subplot(2,4,1);
histogram(all_KL_Q, 50, 'FaceColor', [0.30 0.50 0.85]);
title(sprintf('KL(Q_{true} \\| Q_{fit})\nmean=%.3f, median=%.3f', ...
    mean(all_KL_Q,'omitnan'), median(all_KL_Q,'omitnan')));
xlabel('KL (nats)'); ylabel('# trials'); grid on; box on;

subplot(2,4,2);
plot_grouped(all_ori, all_KL_Q, 'Orientation (deg)', 'KL(Q)');
subplot(2,4,3);
plot_grouped(all_con, all_KL_Q, 'Contrast', 'KL(Q)');
subplot(2,4,4);
plot_grouped(all_dis, all_KL_Q, 'Dispersion (deg)', 'KL(Q)');

% Row 2: KL(L)
subplot(2,4,5);
histogram(all_KL_L, 50, 'FaceColor', [0.85 0.50 0.30]);
title(sprintf('KL(L_{true} \\| L_{fit})\nmean=%.3f, median=%.3f', ...
    mean(all_KL_L,'omitnan'), median(all_KL_L,'omitnan')));
xlabel('KL (nats)'); ylabel('# trials'); grid on; box on;

subplot(2,4,6);
plot_grouped(all_ori, all_KL_L, 'Orientation (deg)', 'KL(L)');
subplot(2,4,7);
plot_grouped(all_con, all_KL_L, 'Contrast', 'KL(L)');
subplot(2,4,8);
plot_grouped(all_dis, all_KL_L, 'Dispersion (deg)', 'KL(L)');

sgtitle(sprintf('Inversion KL: true vs fitted (%s)', tag), 'Interpreter','none');

out_path = fullfile(fig_dir, sprintf('recovery_%s_kl.png', tag));
saveas(f, out_path);
fprintf(' Saved %s\n', out_path);
end

% ---------------------------------------------------------------------
function plot_grouped(g, y, xlab, ylab)
% Mean ± SEM of y grouped by unique values of g.
[gu, ~, idx] = unique(g);
mu = nan(numel(gu),1); sem = nan(numel(gu),1);
for i = 1:numel(gu)
    yi = y(idx==i);
    yi = yi(~isnan(yi));
    if isempty(yi), continue; end
    mu(i) = mean(yi); sem(i) = std(yi)/sqrt(numel(yi));
end
errorbar(gu, mu, sem, 'k-o', 'LineWidth', 1.5, 'MarkerFaceColor','k');
xlabel(xlab); ylabel(ylab);
grid on; box on;
end

% ---------------------------------------------------------------------
function print_recovery_table(R)
n_animals = size(R.true_params_stage1,1);
fprintf('\n--- Stage 1 ---\n');
print_param_block(R.meta.stage1_names, R.true_params_stage1, R.fitted_params_stage1, n_animals);
fprintf('\n--- Stage 2 ---\n');
print_param_block(R.meta.stage2_names, R.true_params_stage2, R.fitted_params_stage2, n_animals);

% Per-animal mean KL
fprintf('\n--- Per-animal mean KL ---\n');
fprintf('%-8s %-8s %10s %10s\n', 'Animal', 'Yoked', 'mean KL(Q)', 'mean KL(L)');
for i = 1:n_animals
    a = R.per_animal{i};
    fprintf('%-8d %-8d %10.4f %10.4f\n', ...
        i, a.is_yoked, mean(a.KL_Q,'omitnan'), mean(a.KL_L,'omitnan'));
end
end

function print_param_block(names, X, Y, n_animals)
fprintf('%-15s %8s %8s %10s %10s\n', 'param', 'r', 'bias', 'rel_bias', 'pass(0.5)');
for k = 1:numel(names)
    x = X(:,k); y = Y(:,k);
    if n_animals >= 3 && std(x) > 0
        r = corr(x, y);
    else
        r = NaN;
    end
    bias = mean(y - x);
    range_x = max(x) - min(x);
    rel_bias = bias / max(abs(range_x), eps);
    pass = (r >= 0.5) && (abs(rel_bias) < 0.30);
    fprintf('%-15s %+8.2f %+8.3f %+10.2f %10s\n', ...
        names{k}, r, bias, rel_bias, pass_label(pass));
end
end

function s = pass_label(p)
if p, s = 'PASS'; else, s = 'FAIL'; end
end
