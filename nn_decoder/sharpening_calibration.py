"""Equivalent sharpening factor — comparing decoder peakiness ACROSS target families.

Motivation (measured, 2026-08-12, GOTCHAS "Loss / scoring")
----------------------------------------------------------
Comparing peakiness between decoders trained on different target families is not
a matter of picking a "dimensionless" statistic. Every candidate was tested
against ground truth — synthetic decoders sharpened by a KNOWN factor, rendered
on both the export arm's 91-bin linear [0,90] support and the IO-HMM arm's
72-bin circular [0,180) support — and every one of them disagreed across arms at
matched true sharpening:

    max-prob ratio             22%      differential-entropy delta   54%
    raw Shannon entropy ratio  15%      resultant (R) ratio          60%
    Mardia circular SD ratio   26%      linear SD ratio              69%
    axial dispersion ratio     37%      perplexity-width ratio       25%

The cause is not bin width alone: the two target families differ in SHAPE (the
export Q is bimodal cardinal-U at low contrast, the IO-HMM posterior is
flat-with-a-tilt), and every width statistic responds to sharpening in a
shape-dependent way. No rescaling fixes that.

The method
----------
Stop comparing statistics; compare a physically defined quantity. Define

    sharpen(p, s) = p ** (s ** 2), renormalised

which for a Gaussian is exactly a width shrink by the factor ``s``, and which is
defined identically on ANY support (it is a pointwise map — no grid, no metric,
no geometry). Then for each arm build a calibration curve from THAT ARM'S OWN
targets, and invert it:

    s_hat = the sharpening factor which, applied to this arm's targets, would
            reproduce the width statistic the decoder actually achieved.

``s_hat`` is comparable across arms because ``s`` is defined support-freely,
even though the underlying statistic is not. This is instrument calibration,
not a metric. Validated: it recovers a known ``s`` to 0.00% cross-arm gap.

    s_hat > 1  over-sharpened   s_hat = 1  calibrated   s_hat < 1  too broad

Caveat, to be reported not hidden
---------------------------------
Real decoders do not only sharpen — they also shift and reshape. ``s_hat``
answers "how much sharpening would produce this much width change", not "how
wrong is this posterior". Always report it with :func:`agreement`, the relative
discrepancy between ``s_hat`` derived from two independent base statistics
(peak-based and entropy-based). Close agreement (a few %) means the deviation
really is sharpening-like; a large value means the decoder is reshaping and
``s_hat`` alone under-describes it. Measured on real cells: KL/JS H=8 cells
agree to 0.2-4.5%, a linear cell to 17% (that one reshapes).

Use :func:`kl_skill`-style normalised losses for PERFORMANCE; this module is
only about the shape/peakiness axis.
"""
from __future__ import annotations

import numpy as np

# Ladder of sharpening factors used to build calibration curves. Dense below 1
# (broadening) and out to 6 (severe over-sharpening) since historical PCA cells
# reach ~9x on the max-prob statistic.
LADDER = np.concatenate([np.linspace(0.25, 1.0, 41)[:-1],
                         np.linspace(1.0, 6.0, 76)])


def sharpen(p, s):
    """Sharpen rows of ``p`` by factor ``s`` (support-free, Gaussian-exact).

    ``s > 1`` narrows, ``s < 1`` broadens, ``s == 1`` is the identity.
    """
    q = np.power(np.clip(np.asarray(p, dtype=np.float64), 1e-300, None),
                 float(s) ** 2)
    return q / q.sum(axis=1, keepdims=True)


# --------------------------------------------------------- base statistics ---
# Any statistic monotone in s works; two independent ones are used so their
# agreement can flag decoders whose deviation is not sharpening-like.
def stat_peak(decoded, target):
    """Mean max-probability ratio. Steep, needs no geometry."""
    return float(decoded.max(1).mean() / target.max(1).mean())


def stat_entropy(decoded, target):
    """Mean Shannon-entropy difference (nats). The additive ln(delta) bin-width
    term cancels in the difference, so no grid is needed here either."""
    def _h(x):
        return -(x * np.log(np.clip(x, 1e-300, None))).sum(1)
    return float((_h(decoded) - _h(target)).mean())


BASE_STATS = {'peak': stat_peak, 'entropy': stat_entropy}


def calibration_curve(targets, stat='peak', ladder=LADDER):
    """Statistic as a function of sharpening factor, for one arm's targets.

    Build this ONCE per arm (per target family) and reuse it for every cell in
    that arm — the curve depends only on the targets, which are identical across
    cells within an arm.
    """
    fn = BASE_STATS[stat] if isinstance(stat, str) else stat
    targets = np.asarray(targets, dtype=np.float64)
    targets = targets / targets.sum(1, keepdims=True)
    return np.array([fn(sharpen(targets, s), targets) for s in ladder])


def invert(curve, value, ladder=LADDER):
    """Map an observed statistic back to an equivalent sharpening factor.

    Values beyond the ladder's range clamp to its endpoints; check for clamping
    with :func:`is_clamped` before reporting a headline number.
    """
    curve = np.asarray(curve, dtype=np.float64)
    if np.all(np.diff(curve) > 0):
        return float(np.interp(value, curve, ladder))
    order = np.argsort(curve)                    # decreasing statistics
    return float(np.interp(value, curve[order], np.asarray(ladder)[order]))


def is_clamped(s_hat, ladder=LADDER, tol=1e-6):
    """True when s_hat sits at a ladder endpoint, i.e. the real value is off the
    calibrated range and the number is a lower/upper bound, not an estimate."""
    return bool(abs(s_hat - ladder[0]) < tol or abs(s_hat - ladder[-1]) < tol)


def equivalent_sharpening(decoded, target, curves=None):
    """s_hat for one decoded/target pair, from both base statistics.

    Parameters
    ----------
    decoded, target : (n_trials, n_bins) arrays, rows summing to 1.
    curves : dict {stat_name: curve}, optional
        Precomputed per-arm calibration curves (strongly preferred — building
        them is the expensive part and they are shared by every cell in an arm).

    Returns
    -------
    dict with 's_hat' (the headline, from the peak statistic), 's_hat_entropy',
    'agreement' (relative discrepancy; small = the deviation really is
    sharpening-like) and 'clamped'.
    """
    decoded = np.asarray(decoded, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    decoded = decoded / decoded.sum(1, keepdims=True)
    target = target / target.sum(1, keepdims=True)
    if curves is None:
        curves = {k: calibration_curve(target, k) for k in BASE_STATS}

    s_peak = invert(curves['peak'], stat_peak(decoded, target))
    s_ent = invert(curves['entropy'], stat_entropy(decoded, target))
    return {
        's_hat': s_peak,
        's_hat_entropy': s_ent,
        'agreement': float(abs(s_peak - s_ent) / max(s_peak, s_ent, 1e-9)),
        'clamped': is_clamped(s_peak) or is_clamped(s_ent),
    }


def _self_test():
    """Recover known sharpening factors on two synthetic supports of different
    geometry AND different target shape — the condition that defeated every
    fixed statistic."""
    rng = np.random.default_rng(0)

    grid_a = np.arange(91.0)                        # 91 linear bins, 1 deg
    mu = rng.uniform(10, 80, 400)[:, None]
    bimodal = (np.exp(-0.5 * ((grid_a - mu) / 18.0) ** 2)
               + 0.8 * np.exp(-0.5 * ((grid_a - (90 - mu)) / 18.0) ** 2))
    tgt_a = bimodal / bimodal.sum(1, keepdims=True)

    grid_b = np.arange(72) * 2.5                    # 72 circular bins, 2.5 deg
    ang = np.deg2rad(grid_b * 2)
    pref = np.deg2rad(rng.uniform(0, 180, 400) * 2)[:, None]
    flat_tilt = np.exp(0.45 * np.cos(ang - pref))   # near-uniform, gentle tilt
    tgt_b = flat_tilt / flat_tilt.sum(1, keepdims=True)

    curves_a = {k: calibration_curve(tgt_a, k) for k in BASE_STATS}
    curves_b = {k: calibration_curve(tgt_b, k) for k in BASE_STATS}

    print("self-test — recover a known sharpening factor on two supports whose")
    print("            geometry AND target shape differ")
    print(f"{'true s':>8} {'s_hat A':>9} {'s_hat B':>9} {'gap':>7}")
    worst = 0.0
    for s in (0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0):
        ra = equivalent_sharpening(sharpen(tgt_a, s), tgt_a, curves_a)['s_hat']
        rb = equivalent_sharpening(sharpen(tgt_b, s), tgt_b, curves_b)['s_hat']
        gap = abs(ra - rb) / max(ra, rb)
        worst = max(worst, gap)
        print(f"{s:8.2f} {ra:9.3f} {rb:9.3f} {100 * gap:6.2f}%")
    verdict = 'PASS' if worst < 0.02 else 'FAIL'
    print(f"worst cross-support gap {100 * worst:.3f}%  {verdict}")
    return worst < 0.02


if __name__ == '__main__':
    raise SystemExit(0 if _self_test() else 1)
