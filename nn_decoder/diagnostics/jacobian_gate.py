# -*- coding: utf-8 -*-
"""The softmax Jacobian gate: why a bounded loss stops correcting a sharp posterior.

Pure synthetic analysis of the *loss geometry*, independent of any decoder. The
only real ingredient is a set of ideal-observer targets and the saved PCA basis
(``results/prodfix_v1/A_baseline_pca/.../stratified_balanced.mat``).

The object
----------
Training descends ``dL/dz`` where ``z`` are the logits, ``p = softmax(z)``:

    dL/dz = J g ,   g = dL/dp ,   J = diag(p) - p p^T .

``J`` gates every probability-space gradient component by the local probability
mass. Once a posterior has sharpened, ``p_i ~ 0`` off the spike, so ``J -> 0``
and any loss with a BOUNDED ``g`` loses its ability to correct those bins.

Forward KL is the exception: ``g_i = -t_i / p_i`` diverges exactly as fast as the
gate closes, and the two cancel **exactly**::

    (J g)_i = p_i(-t_i/p_i) - p_i * sum_j p_j(-t_j/p_j) = p_i - t_i .

The sharpening family
---------------------
``p_gamma ∝ t^gamma`` (renormalised). In LOGIT space this is a straight line,
``z_gamma = gamma * log t + const``, so:

  * the width/sharpening direction in logit space is ``v_gamma = centred(log t)``,
    constant along the whole path — and ``(Jg) . v_gamma = dL/dgamma`` is literally
    the restoring force on peakiness;
  * a rigid shift of the posterior is ``v_mu = centred(d log t / d theta)``, also
    gamma-independent.

Both are well conditioned because the real IO targets have no zero bins.

Outputs
-------
``nn_decoder/figures/jacobian/`` (PNG + SVG via ``peakiness_style.save_fig``) and
a printed number dump. Nothing in the repo is modified.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python diagnostics/jacobian_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import peakiness_style as ps  # noqa: E402

MAT = (HERE.parent / 'results' / 'prodfix_v1' / 'A_baseline_pca' /
       'Q_PCA_half_100ms_all' / 'stratified_balanced.mat')
OUT = HERE.parent / 'figures' / 'jacobian'

# float32 machine eps — the value hard-coded in nn_classifier.cross_entropy
EPS32 = float(np.finfo(np.float32).eps)

# gamma grid: 0.5 -> 4 is the brief's range, but for a target with max-prob 0.037
# gamma=4 only reaches ~0.07 peakiness (Gaussian-ish targets sharpen as sqrt(gamma)).
# The trained decoders sit at 0.32 / 0.64, which needs gamma ~ 1e2. Extend the grid
# and report everything against PEAKINESS, which is directly comparable.
GAMMAS = np.unique(np.concatenate([
    np.geomspace(0.5, 2000.0, 220), [1.0, 1.02, 2.0, 4.0]]))
G_REF = 1.02        # scale-matching point ("onset of over-sharpening")
N_TRIALS = 200      # trials sub-sampled per mouse

# empirical operating points (mean over trials of max-prob), prodfix_v1 baseline PCA
EMP = {'spat': 0.3248, 'temp': 0.6433}


# ----------------------------------------------------------------------
# loss gradients:  g = dL/dp  ->  Jg = p*g - p (p.g)
# ----------------------------------------------------------------------
def _jg(p, g):
    """Apply the softmax Jacobian. p, g: (..., n)."""
    return p * g - p * np.sum(p * g, axis=-1, keepdims=True)


def grad_kl_exact(p, t, **kw):
    """Forward KL(t||p) / CE, analytic: the p_i gate and the 1/p_i blow-up cancel."""
    return p - t


def grad_kl_impl(p, t, **kw):
    """KL/CE exactly as coded: log(p + float32-eps). The eps FLOORS the 1/p
    blow-up, so the cancellation fails once p_i << 1.19e-7 and the gate returns."""
    return _jg(p, -t / (p + EPS32))


def grad_js(p, t, **kw):
    """JS: g = 0.5*log(p/M) + const, M = (p+t)/2. Bounded above, log-divergent below.

    Formed as p*g directly so the 0*log(0) bins (p underflows to 0 at large gamma)
    take their correct limit 0 rather than NaN."""
    M = 0.5 * (p + t)
    pg = np.where(p > 0, 0.5 * p * np.log(np.where(p > 0, p, 1.0) / M), 0.0)
    return pg - p * np.sum(pg, axis=-1, keepdims=True)


def grad_wass(p, t, **kw):
    """1-D Wasserstein = sum |CDF_p - CDF_t|; g_i = sum_{j>=i} sign(.), |g| <= n."""
    s = np.sign(np.cumsum(p, -1) - np.cumsum(t, -1))
    g = np.cumsum(s[..., ::-1], -1)[..., ::-1]
    return _jg(p, g)


def grad_proj(p, t, A=None, **kw):
    """Projection ('PCA') loss sum_k evar_k <p-t,u_k>^2 : g = 2 A (p-t), BOUNDED."""
    return _jg(p, 2.0 * (p - t) @ A.T)


def grad_brier(p, t, **kw):
    """Flat L2 / Brier ||p-t||^2 : g = 2(p-t), BOUNDED."""
    return _jg(p, 2.0 * (p - t))


def make_grad_shape(lam):
    """Projection + lam * Brier — as implemented (evar -> evar + shape_lambda/100),
    which with a full orthonormal PC basis is exactly A -> A + lam*I."""
    def f(p, t, A=None, **kw):
        return _jg(p, 2.0 * ((p - t) @ A.T + lam * (p - t)))
    return f


LOSSES = {
    'KL':            (grad_kl_exact, ps.KL,          '-'),
    'KL (as coded)': (grad_kl_impl,  ps.KL,          ':'),
    'JS':            (grad_js,       ps.JS,          '-'),
    'Wasserstein':   (grad_wass,     ps.WASSERSTEIN, '-'),
    'Projection':    (grad_proj,     ps.PCA_EVAR,    '-'),
    'Brier':         (grad_brier,    ps.FLAT_EVAR,   '-'),
    'Proj+0.1 Brier': (make_grad_shape(0.1), ps.SHAPE_GREENS[1], '-'),
    'Proj+0.3 Brier': (make_grad_shape(0.3), ps.SHAPE_GREENS[3], '-'),
}
MAIN = ['KL', 'JS', 'Wasserstein', 'Projection', 'Brier']
SHAPE_SET = ['Projection', 'Proj+0.1 Brier', 'Proj+0.3 Brier', 'Brier', 'KL']


# ----------------------------------------------------------------------
def load_mouse(mkey):
    m = sio.loadmat(str(MAT), simplify_cells=True)['results'][mkey]['Dist']
    T = np.asarray(m['spat']['target'], float)
    T = T / T.sum(1, keepdims=True)
    pcs = np.asarray(m['pcs'], float)
    evar = np.asarray(m['explained_var'], float)
    A = pcs.T @ (evar[:, None] * pcs)
    return T, pcs, evar, A


def family(logt, gammas):
    """p_gamma ∝ t^gamma, computed in log space (stable to gamma ~ 1e3)."""
    z = gammas[:, None, None] * logt[None, :, :]          # (G, K, n)
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(-1, keepdims=True)


def directions(T):
    """Unit logit-space width and location directions, per target (K, n)."""
    lt = np.log(T)
    v_w = lt - lt.mean(-1, keepdims=True)                  # d z / d gamma
    dl = np.gradient(lt, axis=-1)
    v_l = dl - dl.mean(-1, keepdims=True)                  # d z / d shift
    v_w /= np.linalg.norm(v_w, axis=-1, keepdims=True)
    v_l -= (v_l * v_w).sum(-1, keepdims=True) * v_w        # Gram-Schmidt
    v_l /= np.linalg.norm(v_l, axis=-1, keepdims=True)
    return v_w, v_l


# ----------------------------------------------------------------------
def jacobian_spectra(p):
    """Spectrum diagnostics of J = diag(p) - p p^T for a stack of posteriors."""
    out = {k: [] for k in ('lmax', 'trace', 'pr', 'rank_tol', 'l2')}
    eigs = []
    for pi in p:
        J = np.diag(pi) - np.outer(pi, pi)
        w = np.linalg.eigvalsh(J)[::-1]
        w = np.clip(w, 0, None)
        eigs.append(w)
        out['lmax'].append(w[0])
        out['trace'].append(w.sum())
        # p a true delta (reachable in float64 at extreme gamma) => J is exactly 0
        out['pr'].append(w.sum() ** 2 / np.sum(w ** 2) if w.sum() > 0 else np.nan)
        out['rank_tol'].append(int((w > 1e-6 * w[0]).sum()))
        out['l2'].append(w[1] if w.size > 1 else np.nan)
    return {k: np.asarray(v, float) for k, v in out.items()}, np.asarray(eigs)


def run_mouse(mkey, n_trials=N_TRIALS, seed=0):
    T, pcs, evar, A = load_mouse(mkey)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(T), size=min(n_trials, len(T)), replace=False)
    T = T[idx]
    logt = np.log(T)
    v_w, v_l = directions(T)
    P = family(logt, GAMMAS)                              # (G, K, n)
    peak = P.max(-1).mean(-1)                             # (G,) mean max-prob
    res = {'gamma': GAMMAS, 'peak': peak, 'T': T, 'pcs': pcs, 'evar': evar,
           'A': A, 'P': P, 'v_w': v_w, 'v_l': v_l}
    for name, (fn, _, _) in LOSSES.items():
        Jg = fn(P, T[None], A=A)
        res[name] = {
            'norm':  np.linalg.norm(Jg, axis=-1).mean(-1),
            'fwid':  np.abs((Jg * v_w[None]).sum(-1)).mean(-1),
            'floc':  np.abs((Jg * v_l[None]).sum(-1)).mean(-1),
            'pc_lo': np.linalg.norm(Jg @ pcs[:5].T, axis=-1).mean(-1),
            'pc_hi': np.linalg.norm(Jg @ pcs[5:].T, axis=-1).mean(-1),
        }
    return res


# ----------------------------------------------------------------------
# The logit-space Hessian AT the calibrated solution p = t.
#
# At p = t every one of these losses has zero gradient AND the second-order
# term drops out (its multiplier is either 0 or a constant vector, and
# sum_i d^2 p_i/dz^2 = 0), so the logit Hessian is exactly
#
#       H_z = J H_p J          (the gate applied TWICE)
#
# with H_p = d^2 L / dp^2 evaluated at p = t:
#   KL/CE : diag(1/t)  -> H_z = J           exactly
#   JS    : 0.25 diag(1/t) -> H_z = J/4     (same shape, quarter scale)
#   proj  : 2A                              -> rank(A) ~ 8, so H_z has ~82 NULL dirs
#   Brier : 2I                              -> H_z = 2J^2, full rank
#   proj+lam*Brier : 2(A + lam I)           -> full rank restored
#   Wass  : 0 a.e. (piecewise linear)       -> no curvature anywhere
# ----------------------------------------------------------------------
def hessian_p(name, t, A, lam=None):
    n = t.size
    if name in ('KL', 'KL (as coded)'):
        return np.diag(1.0 / t)
    if name == 'JS':
        return 0.25 * np.diag(1.0 / t)
    if name == 'Wasserstein':
        return np.zeros((n, n))
    if name == 'Brier':
        return 2.0 * np.eye(n)
    if name == 'Projection':
        return 2.0 * A
    if name.startswith('Proj+'):
        return 2.0 * (A + float(name.split('+')[1].split()[0]) * np.eye(n))
    raise KeyError(name)


def hz_stats(name, T, A, dz=None, tol=1e-8):
    """Eigen-diagnostics of H_z = J H_p J over a stack of targets."""
    pr, rk, spec, nullfrac, vis = [], [], [], [], []
    for k, t in enumerate(T):
        J = np.diag(t) - np.outer(t, t)
        H = J @ hessian_p(name, t, A) @ J
        w, V = np.linalg.eigh(H)
        w = np.clip(w, 0, None)[::-1]
        V = V[:, ::-1]
        spec.append(w)
        if w.sum() > 0:
            pr.append(w.sum() ** 2 / np.sum(w ** 2))
            rk.append(int((w > tol * w[0]).sum()))
        else:
            pr.append(0.0); rk.append(0)
        if dz is not None and w.sum() > 0:
            d = dz[k] - dz[k].mean()
            c = (V.T @ d) ** 2
            nullfrac.append(c[w < tol * w[0]].sum() / c.sum())
            vis.append((d @ H @ d) / (w[0] * (d @ d)))
    return {'pr': np.mean(pr), 'rank': np.mean(rk), 'spec': np.mean(spec, 0),
            'nullfrac': np.mean(nullfrac) if nullfrac else np.nan,
            'vis': np.mean(vis) if vis else np.nan}


def trained_logit_displacement(mkey, arch, n=40, seed=0):
    """Dz = centred(log p_decoded) - centred(log t) for the trained baseline-PCA
    decoder, plus the matched targets."""
    R = sio.loadmat(str(MAT), simplify_cells=True)['results'][mkey]['Dist'][arch]
    d = np.maximum(np.asarray(R['decoded'], float), 1e-300)
    t = np.asarray(R['target'], float)
    d = d / d.sum(1, keepdims=True)
    t = t / t.sum(1, keepdims=True)
    idx = np.random.default_rng(seed).choice(len(d), min(n, len(d)), replace=False)
    dz = np.log(d[idx]) - np.log(t[idx])
    return t[idx], dz - dz.mean(1, keepdims=True)


def crossover(peak, ratio, level=0.01):
    """First peakiness at which `ratio` (monotonically falling) drops below level."""
    below = np.where(ratio < level)[0]
    if not below.size:
        return np.nan
    i = below[0]
    if i == 0:
        return peak[0]
    x0, x1 = np.log(ratio[i - 1]), np.log(ratio[i])
    f = (np.log(level) - x0) / (x1 - x0)
    return peak[i - 1] + f * (peak[i] - peak[i - 1])


# ----------------------------------------------------------------------
def verify_against_torch(res):
    """Autograd check of every analytic gradient against the loss as coded."""
    import torch
    sys.path.insert(0, str(HERE.parent))
    from nn_classifier import KL_calc, JS_calc, Wasserstein_calc_1D, cross_entropy

    t = torch.tensor(res['T'][0], dtype=torch.float64)
    A = torch.tensor(res['A'], dtype=torch.float64)
    rows = []
    for gam in (0.8, 1.3, 3.0):
        z = (gam * torch.log(t)).clone().requires_grad_(True)
        p = torch.softmax(z, -1)
        terms = {
            'KL (as coded)': KL_calc(p, t),
            'JS':            JS_calc(p, t),
            'Wasserstein':   Wasserstein_calc_1D(p, t),
            'Projection':    ((p - t) @ A * (p - t)).sum(),
            'Brier':         ((p - t) ** 2).sum(),
            'Proj+0.1 Brier': ((p - t) @ A * (p - t)).sum() + 0.1 * ((p - t) ** 2).sum(),
            'Proj+0.3 Brier': ((p - t) @ A * (p - t)).sum() + 0.3 * ((p - t) ** 2).sum(),
        }
        pn = p.detach().numpy()[None]
        tn = res['T'][0][None]
        for nm, L in terms.items():
            gz = torch.autograd.grad(L, z, retain_graph=True)[0].numpy()
            mine = LOSSES[nm][0](pn, tn, A=res['A'])[0]
            rows.append((gam, nm, float(np.abs(gz - mine).max() /
                                        max(np.abs(gz).max(), 1e-300))))
        # exact-KL identity: Jg == p - t
        rows.append((gam, 'KL (exact identity)',
                     float(np.abs(grad_kl_exact(pn, tn)[0] - (pn - tn)[0]).max())))
    return rows


# ======================================================================
def main():
    ps.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    mice = [f'mouse_{i}' for i in range(6)]
    R = {mk: run_mouse(mk) for mk in mice}
    r0 = R['mouse_0']
    gam, peak = r0['gamma'], r0['peak']

    print('=' * 78)
    print('AUTOGRAD VERIFICATION (max rel. deviation, analytic vs torch on the coded loss)')
    for g_, nm, err in verify_against_torch(r0):
        print(f'  gamma={g_:<5}{nm:<22s} {err:.3e}')

    # ---------------- fig 1: the gate itself -------------------------
    T0 = r0['T']
    med = np.argsort(T0.max(1))[len(T0) // 2]
    t_ex = T0[med]
    g_show = np.array([1.0, 2.0, 4.0, 16.0, 64.0, 256.0])
    P_ex = family(np.log(t_ex)[None], g_show)[:, 0]
    spec, eigs = jacobian_spectra(P_ex)

    # a single target needs a much wider gamma range than the ensemble to reach
    # p_max -> 1, so the asymptote tr J ~ 2(1-p_max) is actually visible
    gam_s = np.geomspace(0.5, 3e6, 260)
    P_all = family(np.log(t_ex)[None], gam_s)[:, 0]
    spec_all, _ = jacobian_spectra(P_all)
    pk_ex = P_all.max(-1)

    fig, ax = plt.subplots(1, 4, figsize=ps.figsize(4, 1), constrained_layout=True)
    for i, gg in enumerate(g_show):
        ax[0].plot(P_ex[i], color=plt.cm.magma(0.1 + 0.72 * i / len(g_show)),
                   lw=1.4, label=f'$\\gamma$={gg:g}')
    ps.target_band(ax[0], np.arange(91), t_ex, label='IO target ($\\gamma$=1)')
    ax[0].set(xlabel='orientation bin', ylabel='probability',
              title='$p_\\gamma \\propto t^{\\gamma}$')
    ps.cap_posterior_ylim(ax[0], t_ex.max(), mult=4.0)
    ax[0].legend(ncol=2)

    for i, gg in enumerate(g_show):
        ax[1].semilogy(np.arange(1, 92), np.maximum(eigs[i], 1e-20),
                       color=plt.cm.magma(0.1 + 0.72 * i / len(g_show)), lw=1.3)
    ax[1].set(xlabel='eigenvalue index', ylabel='$\\lambda_k(J)$',
              ylim=(1e-14, 1), title='spectrum of $J=\\mathrm{diag}(p)-pp^\\top$')

    ax[2].loglog(pk_ex, spec_all['trace'], color='k', lw=1.6, label='tr$J$ = $1-\\sum p_i^2$')
    ax[2].loglog(pk_ex, spec_all['lmax'], color=ps.PCA_EVAR, lw=1.6, label='$\\lambda_{max}$')
    ax[2].loglog(pk_ex, spec_all['pr'] / 91, color=ps.FLAT_EVAR, lw=1.6,
                 label='eff. rank / 91')
    for k, v in EMP.items():
        ax[2].axvline(v, color='0.6', ls='--', lw=1)
        ax[2].text(v, 1.3, k, rotation=90, fontsize=7, color='0.4', va='bottom', ha='right')
    ax[2].set(xlabel='peakiness (max prob)', ylabel='value',
              title='the gate closes')
    ax[2].legend()

    off = 1.0 - pk_ex
    ax[3].loglog(off, spec_all['trace'], 'k', lw=1.6, label='tr$J$')
    # asymptotic regime, stopping short of the exact delta (off-peak mass -> 0)
    sel = (pk_ex > 0.9) & (pk_ex < 1 - 1e-9)
    cf = np.polyfit(np.log(off[sel]), np.log(spec_all['trace'][sel]), 1)
    sl = cf[0]
    ax[3].loglog(off[sel], np.exp(np.polyval(cf, np.log(off[sel]))),
                 '--', color=ps.WASSERSTEIN, lw=1.4,
                 label=f'$p_{{max}}>0.9$: slope {sl:.2f}, pre-factor {np.exp(cf[1]):.2f}')
    ax[3].set(xlabel='off-peak mass $1-p_{max}$', ylabel='tr$J$',
              title='tr$J \\approx 2(1-p_{max})$')
    ax[3].legend()
    ps.label_panels(ax)
    fig.suptitle('The softmax Jacobian degenerates as the posterior sharpens '
                 '(mouse_0, median-breadth IO target)')
    ps.save_fig(fig, OUT, 'fig1_jacobian_spectrum')

    # ---------------- fig 2: applied gradient vs sharpness ------------
    fig, ax = plt.subplots(1, 4, figsize=ps.figsize(4, 1), constrained_layout=True)
    for nm in MAIN + ['KL (as coded)']:
        fn, c, lsty = LOSSES[nm]
        ax[0].loglog(peak, r0[nm]['norm'], color=c, ls=lsty, lw=1.6, label=nm)
        ax[1].loglog(peak, r0[nm]['fwid'], color=c, ls=lsty, lw=1.6, label=nm)
        ax[2].loglog(peak, r0[nm]['floc'], color=c, ls=lsty, lw=1.6)
        ax[3].loglog(peak, r0[nm]['fwid'] / r0[nm]['floc'], color=c, ls=lsty, lw=1.6)
    for a in ax:
        for k, v in EMP.items():
            a.axvline(v, color='0.6', ls='--', lw=1)
        a.axvline(r0['peak'][np.argmin(np.abs(gam - 1))], color='k', ls=':', lw=1)
        a.set_xlabel('peakiness (max prob)')
    # the exact zero at gamma=1 (p=t) would otherwise spend 15 decades of axis
    for a in ax[:3]:
        a.set_ylim(bottom=1e-9)
    ax[0].set(ylabel='$\\|Jg\\|$', title='total applied logit gradient')
    ax[1].set(ylabel='$|dL/d\\gamma|$ (unit dir.)', title='WIDTH-direction force')
    ax[2].set(ylabel='$|dL/d\\mu|$ (unit dir.)', title='LOCATION-direction force')
    ax[3].set(ylabel='width / location', title='scale-free: does width still matter?')
    ax[0].legend(fontsize=7)
    ps.label_panels(ax)
    fig.suptitle('Bounded-gradient losses lose their grip on posterior width; '
                 'KL does not (mouse_0, n=200 IO targets)')
    ps.save_fig(fig, OUT, 'fig2_gradient_vs_sharpness')

    # ---------------- fig 3: crossover --------------------------------
    iref = int(np.argmin(np.abs(gam - G_REF)))
    fig, ax = plt.subplots(1, 3, figsize=ps.figsize(3, 1), constrained_layout=True)
    xs = {}
    for nm in MAIN:
        fn, c, lsty = LOSSES[nm]
        y = r0[nm]['fwid'] / r0[nm]['fwid'][iref]
        ax[0].loglog(peak, y, color=c, lw=1.6, label=nm)
        if nm != 'KL':
            rat = y / (r0['KL']['fwid'] / r0['KL']['fwid'][iref])
            ax[1].loglog(peak, rat, color=c, lw=1.6, label=nm)
            xs[nm] = crossover(peak, rat, 0.01)
    ax[1].axhline(0.01, color='k', ls='--', lw=1.2, label='1% of KL')
    ax[0].set_ylim(1e-6, None)      # the exact zero at gamma=1 otherwise eats the axis
    ax[1].set_ylim(1e-6, 1e3)       # ditto: KL->0 at gamma=1 makes the ratio diverge
    for a in ax[:2]:
        # y in AXES fraction so the label can't drag the layout outside the panel
        tr = a.get_xaxis_transform()
        for k, v in EMP.items():
            a.axvline(v, color='0.6', ls='--', lw=1)
            a.text(v, 0.98, k, rotation=90, fontsize=7, transform=tr,
                   color='0.4', va='top', ha='right')
    for nm, x in xs.items():
        if np.isfinite(x):
            ax[1].plot([x], [0.01], 'o', color=LOSSES[nm][1], ms=5, zorder=5)
    ax[0].set(xlabel='peakiness (max prob)', ylabel='width force (norm. at $\\gamma$=1.02)',
              title='scale-matched at onset of over-sharpening')
    ax[1].set(xlabel='peakiness (max prob)', ylabel='width force / KL width force',
              title='crossover: falls below 1% of KL')
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)

    # per-mouse crossover vs its own empirical operating points
    names, cx, sp, tp = [], [], [], []
    for mk in mice:
        rm = R[mk]
        y = rm['Projection']['fwid'] / rm['Projection']['fwid'][iref]
        yk = rm['KL']['fwid'] / rm['KL']['fwid'][iref]
        cx.append(crossover(rm['peak'], y / yk, 0.01))
        names.append(mk.replace('mouse_', 'm'))
    xpos = np.arange(len(mice))
    ax[2].bar(xpos, cx, color=ps.PCA_EVAR, alpha=0.85, label='crossover peakiness')
    D = sio.loadmat(str(MAT), simplify_cells=True)['results']
    for i, mk in enumerate(mice):
        for arch, mk_ in (('spat', 'o'), ('temp', 's')):
            v = np.asarray(D[mk]['Dist'][arch]['decoded'], float).max(1).mean()
            ax[2].plot([i], [v], mk_, color=ps.ARCH[arch], ms=6,
                       label=(f'trained {arch}' if i == 0 else None))
    ax[2].set(xticks=xpos, ylabel='peakiness (max prob)',
              title='temporal decoders sit past it,\nspatial ones just short')
    ax[2].set_xticklabels(names)
    ax[2].legend(fontsize=7)
    ps.label_panels(ax)
    fig.suptitle('Where the projection loss stops being able to fix width '
                 '(1% threshold; scale-matching is a convention, see notes)')
    ps.save_fig(fig, OUT, 'fig3_crossover')

    # ---------------- fig 4: the shape fix ----------------------------
    fig, ax = plt.subplots(1, 3, figsize=ps.figsize(3, 1), constrained_layout=True)
    for nm in SHAPE_SET:
        fn, c, lsty = LOSSES[nm]
        ax[0].loglog(peak, r0[nm]['fwid'], color=c, lw=1.6, label=nm)
        ax[1].loglog(peak, r0[nm]['fwid'] / r0['Projection']['fwid'],
                     color=c, lw=1.6, label=nm)
    ax[1].axhline(1, color='k', ls=':', lw=1)
    for a in ax[:2]:
        for k, v in EMP.items():
            a.axvline(v, color='0.6', ls='--', lw=1)
    ax[0].set_ylim(1e-9, None)
    ax[0].set(xlabel='peakiness (max prob)', ylabel='$|dL/d\\gamma|$',
              title='width force with the Brier floor')
    ax[1].set(xlabel='peakiness (max prob)', ylabel='gain over plain projection',
              title='what $\\lambda\\cdot$Brier buys')
    ax[0].legend(fontsize=7)

    # per-PC gain of the loss operator A (+ lam I) -- WHY the floor helps
    evar = r0['evar']
    ax[2].semilogy(np.arange(1, 92), np.maximum(evar, 1e-35), color=ps.PCA_EVAR,
                   lw=1.6, label='evar$_k$ (projection)')
    for lam, c in ((0.1, ps.SHAPE_GREENS[1]), (0.3, ps.SHAPE_GREENS[3])):
        ax[2].semilogy(np.arange(1, 92), evar + lam, color=c, lw=1.4,
                       label=f'evar$_k$+{lam}')
    ax[2].set(xlabel='PC index', ylabel='loss weight on PC $k$',
              ylim=(1e-35, 3), title='the weight the loss puts on each PC')
    ax[2].legend(fontsize=7)
    ps.label_panels(ax)
    fig.suptitle('The Brier floor restores a non-vanishing width force '
                 '(and a non-zero weight on every PC)')
    ps.save_fig(fig, OUT, 'fig4_shape_fix')

    # ---------------- fig 5: PC-band split ----------------------------
    fig, ax = plt.subplots(1, 3, figsize=ps.figsize(3, 1), constrained_layout=True)
    for nm in MAIN:
        fn, c, lsty = LOSSES[nm]
        ax[0].loglog(peak, r0[nm]['pc_lo'], color=c, lw=1.6, label=nm)
        ax[1].loglog(peak, r0[nm]['pc_hi'], color=c, lw=1.6)
        ax[2].loglog(peak, r0[nm]['pc_hi'] / r0[nm]['pc_lo'], color=c, lw=1.6)
    for a in ax:
        for k, v in EMP.items():
            a.axvline(v, color='0.6', ls='--', lw=1)
        a.set_xlabel('peakiness (max prob)')
    for a in ax[:2]:
        a.set_ylim(1e-9, None)
    ax[0].set(ylabel='$\\|P_{1-5} Jg\\|$', title='evar-carrying PCs 1-5')
    ax[1].set(ylabel='$\\|P_{6-91} Jg\\|$', title='evar-null PCs 6-91 (shape/width)')
    ax[2].set(ylabel='high-PC / low-PC', title='where the gradient lives')
    ax[0].legend(fontsize=7)
    ps.label_panels(ax)
    # NB the hi/lo RATIO (c) is similar for projection and KL -- the projection loss's
    # deficit is in absolute magnitude (b), not in how it splits its gradient.
    fig.suptitle('Applied gradient split by PC band: the projection loss sits ~2 decades '
                 'below KL in the shape PCs')
    ps.save_fig(fig, OUT, 'fig5_pc_band_split')

    # ---------------- fig 6: Hessian at the calibrated solution -------
    # The t^gamma probe answers "can the loss pull back?". This answers the prior
    # question "is the correct answer even pinned?" -- and it is the panel that
    # tracks the empirical calibration ordering, which the gate alone does not.
    T_h, dz_spat = trained_logit_displacement('mouse_0', 'spat')
    _,   dz_temp = trained_logit_displacement('mouse_0', 'temp')
    HSET = ['KL', 'JS', 'Brier', 'Proj+0.3 Brier', 'Proj+0.1 Brier',
            'Projection', 'Wasserstein']
    HS = {n: hz_stats(n, T_h, r0['A'], dz=dz_temp) for n in HSET}
    HS_sp = {n: hz_stats(n, T_h, r0['A'], dz=dz_spat) for n in HSET}

    fig, ax = plt.subplots(1, 3, figsize=ps.figsize(3, 1), constrained_layout=True)
    for n in HSET:
        if n == 'Wasserstein':
            continue
        c = LOSSES[n][1]
        s = np.maximum(HS[n]['spec'] / HS[n]['spec'][0], 1e-38)
        ax[0].semilogy(np.arange(1, 92), s, color=c, lw=1.6, label=n)
    ax[0].axhline(1e-8, color='k', ls=':', lw=1)
    ax[0].set(xlabel='eigenvalue index', ylabel='$\\lambda_k(H_z)/\\lambda_1$',
              ylim=(1e-38, 3), title='$H_z=JH_pJ$ at $p=t$: what the loss pins')
    ax[0].legend(fontsize=7)

    lams = [0.0, 0.03, 0.1, 0.3]
    emp_s = [5.47, 1.34, 1.06, 0.96]
    emp_t = [10.83, 2.07, 1.17, 0.98]
    prs = []
    for lam in lams:
        nm = 'Projection' if lam == 0 else f'Proj+{lam} Brier'
        if nm not in LOSSES:
            LOSSES[nm] = (make_grad_shape(lam), ps.SHAPE, '-')
        prs.append(hz_stats(nm, T_h, r0['A'])['pr'])
    ax[1].plot(prs, emp_s, 'o-', color=ps.SPATIAL, ms=7, label='spatial')
    ax[1].plot(prs, emp_t, 's-', color=ps.TEMPORAL, ms=7, label='temporal')
    for x, lam in zip(prs, lams):
        ax[1].annotate(f'$\\lambda$={lam:g}', (x, emp_t[lams.index(lam)]),
                       textcoords='offset points', xytext=(4, 6), fontsize=7)
    for n, mkr in (('KL', '*'), ('Brier', 'D')):
        ax[1].axvline(HS[n]['pr'], color=LOSSES[n][1], ls='--', lw=1.2)
        ax[1].text(HS[n]['pr'], 9.5, n, rotation=90, fontsize=7,
                   color=LOSSES[n][1], ha='right', va='top')
    ps.target_line(ax[1], 1.0, label='calibrated (1.0x)')
    ax[1].set(xlabel='effective rank of $H_z$ (participation ratio)',
              ylabel='over-sharpening (decoded / IO peakiness)', yscale='log',
              title='dose-response: rank pinned $\\rightarrow$ calibration')
    ax[1].legend(fontsize=7)

    xs_ = np.arange(len(HSET) - 1)
    nf_t = [HS[n]['nullfrac'] * 100 for n in HSET[:-1]]
    nf_s = [HS_sp[n]['nullfrac'] * 100 for n in HSET[:-1]]
    ax[2].bar(xs_ - 0.2, nf_s, 0.38, color=ps.SPATIAL, label='trained spatial')
    ax[2].bar(xs_ + 0.2, nf_t, 0.38, color=ps.TEMPORAL, label='trained temporal')
    ax[2].set(xticks=xs_, ylabel='% of $\\|\\Delta z\\|^2$ in null($H_z$)',
              title='the observed over-sharpening is invisible\nto the projection loss')
    ax[2].set_xticklabels([n.replace(' Brier', '') for n in HSET[:-1]],
                          rotation=35, ha='right', fontsize=8)
    ax[2].legend(fontsize=7)
    ps.label_panels(ax)
    fig.suptitle('What the gate alone misses: the projection loss leaves ~82 of 90 '
                 'logit directions unpinned at the correct answer')
    ps.save_fig(fig, OUT, 'fig6_hessian_at_solution')

    # ---------------- numbers ----------------------------------------
    print('\n' + '=' * 78)
    print('GAMMA -> PEAKINESS (mouse_0, mean over 200 IO targets; target peak = '
          f'{r0["T"].max(1).mean():.4f})')
    for gg in (0.5, 1.0, 2.0, 4.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
        i = int(np.argmin(np.abs(gam - gg)))
        print(f'  gamma={gam[i]:8.2f}  peakiness={peak[i]:.4f}')
    for arch, v in EMP.items():
        gg = np.exp(np.interp(np.log(v), np.log(peak), np.log(gam)))
        print(f'  trained {arch} peakiness {v:.4f} -> equivalent gamma = {gg:.1f}')

    print('\nJACOBIAN SPECTRUM (single median-breadth target; its own wider gamma grid)')
    for gg in (1.0, 2.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 1e5):
        i = int(np.argmin(np.abs(gam_s - gg)))
        print(f'  gamma={gam_s[i]:9.1f} peak={pk_ex[i]:.4f}  trJ={spec_all["trace"][i]:.4e}'
              f'  lmax={spec_all["lmax"][i]:.4e}  effrank(PR)={spec_all["pr"][i]:6.2f}'
              f'  rank@1e-6={int(spec_all["rank_tol"][i]):3d}')
    print(f'  asymptote (p_max>0.9):  tr J = {np.exp(cf[1]):.3f} * (1-p_max)^{sl:.3f}')
    print('  NB tr J = 1 - sum p_i^2 exactly, so the whole gate closes linearly in the '
          'off-peak mass.')

    WCOLS = MAIN + ['KL (as coded)']
    print('\nWIDTH-DIRECTION FORCE |dL/dgamma| (mouse_0, raw as-coded scale)')
    print('  ' + ''.join(f'{h:>16s}' for h in ['peak'] + WCOLS))
    for gg in (1.02, 2.0, 4.0, 16.0, 64.0, 256.0, 1000.0):
        i = int(np.argmin(np.abs(gam - gg)))
        print(f'  {peak[i]:>14.4f}' + ''.join(f'{r0[n]["fwid"][i]:>16.3e}' for n in WCOLS))
    rat = r0['KL (as coded)']['fwid'] / r0['KL']['fwid']
    print('  KL-as-coded / KL-exact -- the hard-coded float32 eps (1.19e-7) inside '
          'log(p+eps)\n  floors the 1/p blow-up, so the exact cancellation fails once '
          'p_i << eps:')
    for lev in (0.9, 0.5, 0.1, 0.01):
        j = np.where(rat < lev)[0]
        pk_ = peak[j[0]] if j.size else np.nan
        print(f'    coded KL retains {lev:.0%} of the exact width force up to '
              f'peakiness {pk_:.3f}')

    print('\nSCALE-MATCHED (each loss = 1 at gamma=1.02) WIDTH FORCE, ratio to KL')
    print('  ' + ''.join(f'{h:>16s}' for h in ['peak'] + MAIN[1:]))
    ykl = r0['KL']['fwid'] / r0['KL']['fwid'][iref]
    for gg in (1.02, 2.0, 4.0, 16.0, 64.0, 256.0):
        i = int(np.argmin(np.abs(gam - gg)))
        row = [(r0[n]['fwid'][i] / r0[n]['fwid'][iref]) / ykl[i] for n in MAIN[1:]]
        print(f'  {peak[i]:>14.4f}' + ''.join(f'{v:>16.3e}' for v in row))

    print('\nCROSSOVER (width force < 1% of KL, scale-matched at gamma=1.02), mouse_0')
    for nm, x in xs.items():
        print(f'  {nm:<16s} peakiness = {x:.4f}'
              f'   ({"BELOW" if x > EMP["spat"] else "ABOVE"} trained spat 0.325;'
              f' {"BELOW" if x > EMP["temp"] else "ABOVE"} trained temp 0.643)')

    print('\nPER-MOUSE crossover for the projection loss')
    for mk, x in zip(mice, cx):
        rm = R[mk]
        sp_ = np.asarray(D[mk]['Dist']['spat']['decoded'], float).max(1).mean()
        tp_ = np.asarray(D[mk]['Dist']['temp']['decoded'], float).max(1).mean()
        print(f'  {mk}: crossover={x:.4f}  target={rm["T"].max(1).mean():.4f}'
              f'  trained spat={sp_:.4f} temp={tp_:.4f}')
    print(f'  MEAN crossover = {np.nanmean(cx):.4f} +/- {np.nanstd(cx):.4f}')

    print('\nSHAPE FIX: width-force gain over plain projection, mouse_0')
    print('  ' + ''.join(f'{h:>18s}' for h in ['peak'] + SHAPE_SET[1:]))
    for gg in (2.0, 4.0, 16.0, 64.0, 256.0):
        i = int(np.argmin(np.abs(gam - gg)))
        row = [r0[n]['fwid'][i] / r0['Projection']['fwid'][i] for n in SHAPE_SET[1:]]
        print(f'  {peak[i]:>16.4f}' + ''.join(f'{v:>18.3f}' for v in row))
    for arch, v in EMP.items():
        i = int(np.argmin(np.abs(peak - v)))
        row = [r0[n]['fwid'][i] / r0['Projection']['fwid'][i] for n in SHAPE_SET[1:]]
        print(f'  at trained {arch} ({v:.3f}): ' +
              ''.join(f'{n}={g_:.3f}  ' for n, g_ in zip(SHAPE_SET[1:], row)))

    print('\nPC-BAND SPLIT at the trained operating points (mouse_0)')
    for arch, v in EMP.items():
        i = int(np.argmin(np.abs(peak - v)))
        print(f'  {arch} (peak {v:.3f}): ' +
              '  '.join(f'{n}: hi/lo={r0[n]["pc_hi"][i]/r0[n]["pc_lo"][i]:.3f}'
                        for n in MAIN))
    print('\n' + '=' * 78)
    print('LOGIT HESSIAN AT THE CALIBRATED SOLUTION  H_z = J H_p J  (mouse_0, 40 targets)')
    print(f"  {'loss':<18}{'eff.rank(PR)':>14}{'rank@1e-8':>11}"
          f"{'%Dz in null (spat)':>20}{'(temp)':>10}{'visibility (temp)':>19}")
    for n in HSET:
        h, hsp = HS[n], HS_sp[n]
        print(f'  {n:<18}{h["pr"]:>14.2f}{h["rank"]:>11.1f}'
              f'{hsp["nullfrac"] * 100:>19.2f}%{h["nullfrac"] * 100:>9.2f}%'
              f'{h["vis"]:>19.3e}')
    print('\nLAMBDA DOSE-RESPONSE (projection + lambda*Brier)')
    for lam, pr_, es, et in zip(lams, prs, emp_s, emp_t):
        print(f'  lambda={lam:<5} PR(H_z)={pr_:7.2f}   empirical over-sharpening: '
              f'spat {es:5.2f}x  temp {et:5.2f}x')
    print(f'  evar participation ratio (PCs the projection loss weights at all) = '
          f'{1 / np.sum(r0["evar"] ** 2):.2f}')

    print('\nIS THE TRAINED DRIFT ON THE t^gamma PATH? (variance of decoded logits '
          'explained by log t)')
    Rm = sio.loadmat(str(MAT), simplify_cells=True)['results']
    for mk in ('mouse_0', 'mouse_3'):
        for arch in ('spat', 'temp'):
            dd = np.maximum(np.asarray(Rm[mk]['Dist'][arch]['decoded'], float), 1e-300)
            tt = np.asarray(Rm[mk]['Dist'][arch]['target'], float)
            dd /= dd.sum(1, keepdims=True); tt /= tt.sum(1, keepdims=True)
            zc = np.log(dd); zc -= zc.mean(1, keepdims=True)
            lt = np.log(tt); lt -= lt.mean(1, keepdims=True)
            gfit = (zc * lt).sum(1) / (lt * lt).sum(1)
            fr = 1 - ((zc - gfit[:, None] * lt) ** 2).sum(1) / (zc ** 2).sum(1)
            print(f'  {mk} {arch}: best-fit gamma median={np.median(gfit):6.2f}, '
                  f'variance explained by the t^gamma path = {np.median(fr) * 100:.1f}%')

    np.save(str(OUT / 'jacobian_gate_curves.npy'),
            {'peak': peak, 'gamma': gam,
             'mouse_0': {n: v for n, v in R['mouse_0'].items() if isinstance(v, dict)},
             'hz_pr': {n: HS[n]['pr'] for n in HSET},
             'lambda_dose': {'lam': lams, 'pr': prs, 'spat': emp_s, 'temp': emp_t},
             'crossover_per_mouse': np.asarray(cx)}, allow_pickle=True)
    print(f'\nfigures + curves -> {OUT}')


if __name__ == '__main__':
    main()
