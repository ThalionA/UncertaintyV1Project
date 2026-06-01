# Jensen smoothing — a slide-deck walkthrough

Companion to `jensen_smoothing_explainer.py`. Six standalone PNGs in
`nn_decoder/figures/shuffle_asymmetry/`, plus the talking points below.
Designed so each figure carries one idea: drop a figure on a slide and
the text here is the speaker notes.

## Slide 1 — Two forward passes, same MLP backbone (`fig_01_setup.png`)

> Same MLP, same training data, same hyperparameters. The PPC and
> sampling decoders differ in *one* place: PPC averages the inputs
> across time bins **before** the softmax; sampling averages the
> outputs **after** the softmax.
>
> Softmax is non-linear. So `softmax(mean(x))` and `mean(softmax(x))`
> are different mathematical operations on the same data. Everything
> that follows is consequences of that one fact.

## Slide 2 — Same trial, two predictions (`fig_02_single_trial.png`)

> One trial, T = 10 per-bin logit vectors (top, grey).  Their mean is
> the black line — that's what the PPC pipeline feeds to softmax.
>
> Bottom: the two predictions on the same axis.
> - PPC (blue) is the softmax of the mean logit — a *sharp* curve
>   that simply reflects where the mean logit peaks.
> - SBC (red) is the average of T per-bin softmaxes (faded red).
>   It's *broader*: each per-bin softmax peaks at a slightly
>   different place, and the average across them spreads probability
>   mass over all those places.
>
> Same trial → same input mean → identical PPC. SBC sees the *spread*
> of per-bin logits, not just the mean, and that spread costs it in
> sharpness.

## Slide 3 — The geometry on a 2-class problem (`fig_03_2class_geometry.png`)

> Strip the problem down to two classes: probability of class 1 lives
> on a single line [0, 1]. Each per-bin logit pair gives a softmax
> point on that line, equivalent to evaluating the sigmoid at the
> logit difference.
>
> - PPC picks the logit-difference mean (sigmoid value at $\overline z$).
> - SBC picks the average of the y-values (the chord midpoint, roughly).
>
> Jensen's inequality is exactly the statement that these two
> operations disagree whenever the function is non-linear. On the
> concave side of the sigmoid the chord midpoint sits *below* the
> curve; the offset is the Jensen gap. SBC always lands *closer to
> 0.5* than PPC does — closer to the centre of the simplex (the
> broadest distribution).

## Slide 4 — Within-trial noise is the smoothing dial (`fig_04_noise_drives.png`)

> We construct the within-trial noise so it sums to zero per trial.
> That means `mean(x_t) == s_n` exactly — PPC's input is the trial
> signal, regardless of how noisy the per-bin inputs are.
>
> - **A:** As we crank up the within-trial noise σ, PPC's prediction
>   entropy stays *flat* — it only sees the mean. SBC's prediction
>   broadens monotonically with σ: noisier per-bin logits give
>   per-bin softmaxes that peak in different places; the average
>   across them spreads more.
> - **B:** Trial-to-trial *variance of the prediction*: PPC is
>   constant (its input across trials is fixed). SBC drops as σ grows
>   — at high noise every trial's SBC prediction is pulled toward
>   the *same* broad smoothed distribution, so trial-to-trial
>   variance collapses. At σ=3 the PPC/SBC ratio is **3.10×**.
>
> This is the asymmetry that drives the shuffle-control gap in the
> real experiment.

## Slide 5 — SBC as a Monte-Carlo estimator (`fig_05_T_law.png`)

> SBC = $\frac{1}{T} \sum_t \mathrm{softmax}(\mathrm{MLP}(x_t))$ is a
> Monte-Carlo estimator of the function
> $g(s) := \mathbb{E}_\epsilon\,[\mathrm{softmax}(\mathrm{MLP}(s + \epsilon))]$,
> a *smoothed* version of $\mathrm{softmax}(\mathrm{MLP}(s))$.
>
> - **A:** PPC's trial-to-trial variance is $\mathrm{Var}_s(\mathrm{softmax}(\mathrm{MLP}(s)))$ —
>   the un-smoothed function evaluated at the trial signals
>   (blue ceiling, flat in T). SBC's trial-to-trial variance is
>   $\mathrm{Var}_s(g(s))$ — the *smoothed* function (lower red ceiling),
>   approached as $T \to \infty$.
> - **B:** SBC's *excess variance over its asymptotic floor* falls as
>   1 / T — the textbook Monte-Carlo rate. Averaging T per-bin
>   softmaxes literally is Monte-Carlo integration of g.
>
> Two takeaways for the empirical shuffle gap:
> (i) the gap exists for any T (architecture, not stochastic luck);
> (ii) larger T just gets you closer to the smoothed floor —
> SBC's advantage doesn't disappear in the limit.

## Slide 6 — On real shuffled-target predictions (`fig_06_real_overlay.png`)

> Eight random trials from mouse 1's saved shuffled-target run, both
> models, same trial set.
> - The bold lines are each model's mean prediction across all trials.
>   They're *both* nearly identical to the target marginal (dashed
>   black) — on average, both models recover the right answer for
>   shuffled data.
> - The thin lines are per-trial predictions. Spat_shf (PPC, blue)
>   has visibly wider trial-to-trial spread than temp_shf (SBC, red).
>   Same data, same training, same MLP capacity — the spread
>   difference is the Jensen-smoothing fingerprint.
>
> Numbers in the title: mean per-trial entropy is essentially equal
> (3.91 vs 3.95 nats); trial-to-trial variance is 1.78× larger for
> PPC. Exactly the pattern the synthetic demo predicts from
> architecture alone.

## Why this matters in one line

> When you read a shuffle-normalised "decoder lift" plot
> (`aggregate_*_chance_normalised.png` in the scaling sweep), spatial
> and temporal sit on top of each other on Q-target — meaning *the
> apparent temporal advantage on the raw spat/temp ratio comes from
> this architectural smoothing, not from the temporal code carrying
> more signal*.
