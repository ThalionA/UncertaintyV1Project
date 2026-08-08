"""Old vs new ideal-observer posteriors, per matched trial, on a common x-axis.

Old: `post_s_marginal` from VR_Decoder_Data_Export.mat — 91 bins, linear 0-90 deg
(1 deg steps). New: `PS_stim_G_tr` from data/fitted_data_and_posteriors.pkl
(collaborator IO-HMM refit) — 72 circular bins x 2.5 deg spanning [0, 180).

X-axis matching: the new posterior is folded about 90 deg (mass at theta and
180-theta summed; orientation is pi-periodic and task stimuli span [0, 90]),
converted to a per-degree density, linearly resampled onto the old 1-deg grid and
renormalised. Both families are then probability vectors on the same 91-point
grid. Mass beyond 90 deg in the raw new posterior (which the old support cannot
represent) is reported separately before folding.

Trials are matched via the stimulus-condition barcode alignment (export trials
are an in-order subsequence of pkl trials). Prefers nn_decoder/io_hmm_data.py for
loading/alignment; falls back to a local partial-file recovery so the comparison
runs on the truncated Slack copy (mouse 0) before the full pkl arrives.

Usage: python diagnostics/compare_io_hmm_vs_export_posteriors.py [--mouse-ids 0 ...]
Figures -> nn_decoder/figures/io_hmm_vs_export/ (png+svg via figsave.save_fig).
"""
from __future__ import annotations

import argparse
import pickle
import sys
import types
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))          # nn_decoder/
from figsave import save_fig                   # noqa: E402
from utils import load_vr_export               # noqa: E402

REPO = _HERE.parent.parent
PKL_DEFAULT = REPO / "data" / "fitted_data_and_posteriors.pkl"
OUT_DIR = _HERE.parent / "figures" / "io_hmm_vs_export"

OLD_GRID = np.arange(91.0)                     # 1-deg bins, 0..90
NEW_GRID_FULL = np.arange(72) * 2.5            # circular, [0, 180)
FOLD_GRID = np.arange(37) * 2.5                # after folding, 0..90


# ---------------------------------------------------------------- loading ----
def _load_new_posteriors(pkl_path):
    """Return {mouse_id: {'ps': (n,72) rows sum 1, 'data': dict}} — via
    io_hmm_data if available, else local partial recovery."""
    try:
        import io_hmm_data                     # written by the wiring session
        mice = io_hmm_data.load_io_hmm_pkl(str(pkl_path), allow_partial=True)
        out = {}
        for m, entry in mice.items():
            d = entry["data"]
            ps = np.asarray(d["PS_stim_G_tr"], dtype=np.float64)
            if ps.shape[0] == 72:              # bins-first layout
                ps = ps.T
            out[m] = {"ps": ps / ps.sum(1, keepdims=True), "data": d}
        return out
    except ImportError:
        warnings.warn("io_hmm_data not importable — using local partial recovery")
        return _recover_partial(pkl_path)


def _recover_partial(pkl_path):
    class _StubBase:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            self.__dict__.update(state if isinstance(state, dict) else {"_state": state})

    def _stub_module(name):
        mod = types.ModuleType(name)

        def __getattr__(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            cls = type(attr, (_StubBase,), {"__module__": name})
            setattr(mod, attr, cls)
            return cls

        mod.__getattr__ = __getattr__
        sys.modules[name] = mod

    _stub_module("datastructs")
    for name in ("jax", "jax._src"):
        sys.modules.setdefault(name, types.ModuleType(name))
    arr_mod = types.ModuleType("jax._src.array")

    def _reconstruct_array(fun, args, arr_state, *rest):
        v = fun(*args)
        v.__setstate__(arr_state)
        return v

    arr_mod._reconstruct_array = _reconstruct_array
    sys.modules["jax._src.array"] = arr_mod

    up = pickle._Unpickler(open(pkl_path, "rb"))
    try:
        up.load()
    except Exception:
        pass                                    # truncated copy — expected
    out = {}
    datas = [v for v in up.memo.values() if type(v).__name__ == "Data"]
    for m, dobj in enumerate(datas):
        d = dobj.__dict__
        if d.get("PS_stim_G_tr") is None:
            continue
        ps = np.asarray(d["PS_stim_G_tr"], dtype=np.float64)
        if ps.shape[0] == 72:
            ps = ps.T
        out[m] = {"ps": ps / ps.sum(1, keepdims=True), "data": d}
    return out


def _align(trials, data):
    """Greedy in-order barcode alignment; returns pkl index per export trial."""
    mo = np.asarray(trials["orientation"]).ravel().astype(float)
    md = np.asarray(trials["dispersion"]).ravel().astype(float)
    mc = np.round(np.asarray(trials["contrast"]).ravel().astype(float), 3)
    po = np.asarray(data["orientation"]).astype(float)
    pdis = np.asarray(data["dispersion"]).astype(float)
    pc = np.round(np.asarray(data["contrast"]).astype(float), 3)
    idx, j = [], 0
    for i in range(len(mo)):
        s = (mo[i], md[i], mc[i])
        while j < len(po) and (po[j], pdis[j], pc[j]) != s:
            j += 1
        if j == len(po):
            raise RuntimeError(f"barcode alignment failed at export trial {i}")
        idx.append(j)
        j += 1
    return np.array(idx)


# ------------------------------------------------------------- x matching ----
def fold_and_resample(ps_new):
    """(n, 72) circular [0,180) -> mass>90 (n,), density on FOLD_GRID (n,37),
    and probability rows on OLD_GRID (n, 91)."""
    mass_beyond = ps_new[:, 37:].sum(1)
    folded = np.empty((ps_new.shape[0], 37))
    folded[:, 0] = ps_new[:, 0]
    folded[:, 36] = ps_new[:, 36]
    for k in range(1, 36):
        folded[:, k] = ps_new[:, k] + ps_new[:, 72 - k]
    # per-degree density: interior bins cover 2.5 deg; the 0 and 90 bins only
    # half of their circular width lands in [0, 90] (edge effect, second-order
    # for posteriors this broad)
    width = np.full(37, 2.5)
    width[[0, 36]] = 1.25
    dens = folded / width
    on_old = np.array([np.interp(OLD_GRID, FOLD_GRID, dens[i]) for i in range(len(dens))])
    on_old /= on_old.sum(1, keepdims=True)
    return mass_beyond, dens, on_old


def _moments(p, grid):
    mu = (p * grid).sum(1)
    sd = np.sqrt((p * (grid - mu[:, None]) ** 2).sum(1))
    return mu, sd


def _diff_entropy(p, delta):
    """Differential entropy in nats (grid-size independent): -sum p ln p + ln(delta)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.log(np.clip(p, 1e-300, None))).sum(1)
    return h + np.log(delta)


# ------------------------------------------------------------------ plots ----
def _panel(ax, old_p, new_dens_on_old, ori, label_old, label_new):
    ax.plot(OLD_GRID, old_p, lw=1.4, color="tab:blue", label=label_old)
    ax.plot(OLD_GRID, new_dens_on_old, lw=1.4, color="tab:orange", label=label_new)
    ax.axvline(ori, color="k", lw=0.7, ls=":")
    ax.set_xlim(0, 90)
    ax.set_ylim(bottom=0)


def make_figures(mouse, old_p, new_on_old, mass_beyond, trials, out_dir):
    ori = np.asarray(trials["orientation"]).ravel().astype(float)
    ct = np.round(np.asarray(trials["contrast"]).ravel().astype(float), 3)
    disp = np.asarray(trials["dispersion"]).ravel().astype(float)
    rng = np.random.default_rng(0)

    mu_o, sd_o = _moments(old_p, OLD_GRID)
    mu_n, sd_n = _moments(new_on_old, OLD_GRID)
    h_o = _diff_entropy(old_p, 1.0)
    h_n = _diff_entropy(new_on_old, 1.0)
    tv = 0.5 * np.abs(old_p - new_on_old).sum(1)

    # fig 1 — example single trials: rows = contrast, cols = orientation
    cts = [1.0, 0.25, 0.01]
    oris = [0.0, 40.0, 50.0, 90.0]
    fig, axes = plt.subplots(len(cts), len(oris), figsize=(11, 7), sharex=True)
    for r, c in enumerate(cts):
        for k, o in enumerate(oris):
            ax = axes[r, k]
            cand = np.where((ct == c) & (ori == o))[0]
            if len(cand) == 0:
                ax.set_axis_off()
                continue
            # most representative trial: min TV distance to the condition mean
            # (single new-model trials swing wrong-way often enough that a random
            # pick misleads — see fig 3 spread)
            cell_mean = new_on_old[cand].mean(0)
            t = cand[np.argmin(np.abs(new_on_old[cand] - cell_mean).sum(1))]
            _panel(ax, old_p[t], new_on_old[t], o,
                   "old (91-bin Q)", "new (IO-HMM, folded)")
            ax.set_title(f"ori {o:.0f}  ctr {c:g}  trial {t}", fontsize=8)
            if r == 0 and k == 0:
                ax.legend(fontsize=7, frameon=False)
            if r == len(cts) - 1:
                ax.set_xlabel("orientation (deg)")
            if k == 0:
                ax.set_ylabel("prob / deg")
    fig.suptitle(f"Mouse {mouse} — example matched trials, old vs new posterior "
                 "(new folded about 90deg, resampled to 1deg grid)", fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, f"m{mouse}_fig1_example_trials")

    # fig 2 — condition means: rows = contrast, cols = orientation (all 9)
    u_oris = np.unique(ori)
    u_cts = sorted(np.unique(ct))[::-1]
    fig, axes = plt.subplots(len(u_cts), len(u_oris), figsize=(13.5, 7),
                             sharex=True, sharey="row")
    for r, c in enumerate(u_cts):
        for k, o in enumerate(u_oris):
            ax = axes[r, k]
            m = (ct == c) & (ori == o)
            if m.sum() == 0:
                ax.set_axis_off()
                continue
            _panel(ax, old_p[m].mean(0), new_on_old[m].mean(0), o, "old", "new")
            ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(f"ori {o:.0f}", fontsize=8)
            if k == 0:
                ax.set_ylabel(f"ctr {c:g}\nprob / deg", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Mouse {mouse} — condition-mean posteriors (n trials per cell vary)",
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, f"m{mouse}_fig2_condition_means")

    # fig 3 — per-trial summary scatters + mass beyond 90
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    ax = axes[0, 0]
    sc = ax.scatter(mu_o, mu_n, c=ori, cmap="viridis", s=8, alpha=0.6)
    ax.plot([0, 90], [0, 90], "k:", lw=0.8)
    ax.set_xlabel("old posterior mean (deg)")
    ax.set_ylabel("new posterior mean (deg)")
    r_mu = np.corrcoef(mu_o, mu_n)[0, 1]
    ax.set_title(f"means, r={r_mu:.2f}", fontsize=9)
    plt.colorbar(sc, ax=ax, label="true ori (deg)")

    ax = axes[0, 1]
    sc = ax.scatter(sd_o, sd_n, c=np.log10(ct), cmap="magma", s=8, alpha=0.6)
    lims = [0, max(sd_o.max(), sd_n.max()) * 1.05]
    ax.plot(lims, lims, "k:", lw=0.8)
    ax.set_xlabel("old posterior SD (deg)")
    ax.set_ylabel("new posterior SD (deg)")
    r_sd = np.corrcoef(sd_o, sd_n)[0, 1]
    ax.set_title(f"widths, r={r_sd:.2f}", fontsize=9)
    plt.colorbar(sc, ax=ax, label="log10 contrast")

    ax = axes[1, 0]
    sc = ax.scatter(h_o, h_n, c=disp, cmap="cividis", s=8, alpha=0.6)
    lims = [min(h_o.min(), h_n.min()) - 0.1, max(h_o.max(), h_n.max()) + 0.1]
    ax.plot(lims, lims, "k:", lw=0.8)
    ax.set_xlabel("old differential entropy (nats)")
    ax.set_ylabel("new differential entropy (nats)")
    r_h = np.corrcoef(h_o, h_n)[0, 1]
    ax.set_title(f"entropies, r={r_h:.2f}", fontsize=9)
    plt.colorbar(sc, ax=ax, label="dispersion (deg)")

    ax = axes[1, 1]
    for c in u_cts:
        m = ct == c
        ax.hist(mass_beyond[m], bins=np.linspace(0, mass_beyond.max() * 1.02, 30),
                histtype="step", lw=1.3, label=f"ctr {c:g}", density=True)
    ax.set_xlabel("new-posterior mass beyond 90 deg (pre-fold)")
    ax.set_ylabel("trial density")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("mass the old support cannot represent", fontsize=9)

    fig.suptitle(f"Mouse {mouse} — per-trial old-vs-new summaries "
                 f"(median TV distance {np.median(tv):.3f})", fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, f"m{mouse}_fig3_trial_summaries")

    return dict(r_mean=r_mu, r_sd=r_sd, r_entropy=r_h, tv_median=float(np.median(tv)),
                sd_old_median=float(np.median(sd_o)), sd_new_median=float(np.median(sd_n)),
                h_old_median=float(np.median(h_o)), h_new_median=float(np.median(h_n)),
                mass_beyond_median=float(np.median(mass_beyond)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse-ids", type=int, nargs="*", default=None)
    ap.add_argument("--pkl", type=Path, default=PKL_DEFAULT)
    args = ap.parse_args()

    new = _load_new_posteriors(args.pkl)
    mice = args.mouse_ids if args.mouse_ids is not None else sorted(new)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for mouse in mice:
        if mouse not in new:
            print(f"mouse {mouse}: not recoverable from pkl copy — skipped")
            continue
        acts, targets_perc, _td, _tl, trials = load_vr_export(mouse)
        old_p = np.asarray(targets_perc, dtype=np.float64)
        old_p /= old_p.sum(1, keepdims=True)
        idx = _align(trials, new[mouse]["data"])
        ps_new = new[mouse]["ps"][idx]
        mass_beyond, _dens, new_on_old = fold_and_resample(ps_new)
        stats = make_figures(mouse, old_p, new_on_old, mass_beyond, trials, OUT_DIR)
        n_drop = new[mouse]["ps"].shape[0] - len(idx)
        print(f"mouse {mouse}: {len(idx)} matched trials ({n_drop} pkl-only), "
              + ", ".join(f"{k}={v:.3f}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
