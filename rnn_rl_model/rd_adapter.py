"""Feed the RL cohort through the project's RD-1 / RD-2 readout tests.

The Hebbian ``si_network_model`` already serves as a positive control inside
``nn_decoder/similarity_readout_tests.py`` (``network_control_arrays``). This
adapter does the same for the RL cohort: it builds the identical array bundle
from each agent's frozen-evaluation block and runs the same estimators, so the
RL model becomes a second, learning-rule-independent control.

RD-1 asks whether covariance-whitening the readout buys choice accuracy beyond
the template; RD-2 asks whether trial-mean SI (and within-trial SI variance)
carry choice information beyond the stimulus. The framework prediction -- and the
single-animal result already seen -- is that whitening adds nothing to choice
(the readout is the template) while mean SI does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .cohort import CONDITIONS, AnimalResult
from .paths import NN_DECODER_DIR

if str(NN_DECODER_DIR) not in sys.path:
    sys.path.insert(0, str(NN_DECODER_DIR))
import similarity_readout_tests as srt  # noqa: E402  (path shim must run first)


def bundle_from_eval(ev: dict) -> dict:
    """The RD array bundle, matching ``srt.network_control_arrays`` exactly."""
    ori = ev["orientation"]
    return {
        "X_bar": ev["r_bar"].astype(float),
        "y_stim": (ori < 45.0).astype(int),
        "y_choice": ev["choice"].astype(int),
        "signed_contrast": ev["contrast"] * np.sign(45.0 - ori),
        "si_mean": ev["mean_si"].astype(float),
        "si_var": ev["var_si"].astype(float),
    }


def run_rd_for_condition(results: list[AnimalResult], cond: str) -> dict:
    """Run RD-1 + RD-2 for every animal under one V1 condition."""
    rd1, rd2 = {}, {}
    for r in results:
        b = bundle_from_eval(r.eval[cond])
        name = f"a{r.animal_id}"
        rd1[name] = srt.rd1_template_vs_whitened(
            b["X_bar"], b["y_stim"], b["y_choice"])
        rd2[name] = srt.rd2_nested_choice_models(
            b["X_bar"], b["y_stim"], b["y_choice"],
            b["signed_contrast"], b["si_mean"], b["si_var"])
    return {"rd1": rd1, "rd2": rd2}


_RD2_M1 = "M1-M0 (meanSI adds | Pred16)"
_RD2_M2 = "M2-M1 (varSI adds | SBC wedge)"
_RD2_M3 = "M3-M1 (whitened adds | premise)"


def _cohort_means(rd: dict) -> dict:
    """Mean over animals of the headline RD contrasts."""
    rd1 = list(rd["rd1"].values())
    rd2 = list(rd["rd2"].values())
    return {
        # RD-1: whitening buys stimulus AUC (>0) but not choice AUC (~0).
        "rd1_dstim_auc": float(np.mean([r["d_auc_stim_whiten"] for r in rd1])),
        "rd1_dchoice_auc": float(np.mean([r["d_auc_choice_whiten"] for r in rd1])),
        # RD-2 nested held-out log-likelihood deltas (nats/trial).
        "rd2_M1_M0": float(np.mean([r["delta"][_RD2_M1] for r in rd2])),
        "rd2_M2_M1": float(np.mean([r["delta"][_RD2_M2] for r in rd2])),
        "rd2_M3_M1": float(np.mean([r["delta"][_RD2_M3] for r in rd2])),
    }


def run_rd_on_cohort(results: list[AnimalResult], out_dir: Path) -> dict:
    """Run RD-1/RD-2 for both V1 conditions, render figures, return a summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    for cond in CONDITIONS:
        rd = run_rd_for_condition(results, cond)
        srt.render_rd1(rd["rd1"], out_dir, stem=f"rd1_template_vs_whitened__{cond}")
        srt.render_rd2(rd["rd2"], out_dir, stem=f"rd2_nested_choice__{cond}")
        summary[cond] = _cohort_means(rd)
    return summary
