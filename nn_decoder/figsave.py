# -*- coding: utf-8 -*-
"""The single figure-save sink for the repo.

CLAUDE.md mandates that every figure is written as **both** a full-detail SVG
(for the manuscript / research vault) and a PNG capped so its longest side stays
≤ 1600 px (so the Claude Code image reader — which rejects images > 2000 px on a
side — can always preview it). This module is the one place that contract lives.

It deliberately depends only on matplotlib + pathlib (no project imports, no
seaborn/style), so any plotter can ``from figsave import save_fig`` without
pulling in heavy modules or risking an import cycle. ``peakiness_style.save_fig``
re-exports this function, so its existing callers are unchanged.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_fig(fig, out_dir, stem, max_px=1600, svg_dpi=140, close=True, verbose=True,
             layout='constrained'):
    """Write ``stem.svg`` (full vector detail) and ``stem.png`` (rasterised at a
    dpi chosen so the longest side is ≤ ``max_px``) under ``out_dir``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    out_dir : str | pathlib.Path
        Created (with parents) if it doesn't exist.
    stem : str
        Basename without extension; both ``stem.svg`` and ``stem.png`` are written.
    max_px : int
        Cap on the PNG's longest side in pixels (default 1600; the hard reader
        limit is 2000). The PNG dpi is ``min(svg_dpi, max_px / max(w_in, h_in))``.
    svg_dpi : int
        Upper bound on the PNG dpi (the SVG is vector, so dpi-independent).
    close : bool
        Close the figure after saving (default True) to free memory in loops.
    verbose : bool
        Print the written stem + chosen dpi.
    layout : str | None
        Matplotlib layout engine to apply before saving so panels/legends/titles/
        colourbars/twin-axes don't collide (default ``'constrained'`` — it makes a
        prior ``fig.tight_layout()`` a harmless no-op). Pass ``None`` for figures
        that position their own axes (insets, manual ``subplots_adjust``/GridSpec
        placement) and would fight an auto layout engine.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if layout:
        try:
            fig.set_layout_engine(layout)
        except Exception:
            pass   # older mpl / incompatible engine — fall back to whatever's set
    w, h = fig.get_size_inches()
    dpi = min(svg_dpi, int(max_px / max(w, h)))

    def _write():
        fig.savefig(out_dir / f'{stem}.svg', bbox_inches='tight')
        # bbox_inches='tight' grows the canvas for outside legends / long titles,
        # so a dpi set from the NOMINAL size can overshoot max_px (measured
        # 2026-08-23: 6 of 13 figures at 1603-1822 px). Measure the tight bbox
        # and shrink the dpi to fit it before rasterising.
        try:
            fig.canvas.draw()
            bb = fig.get_tightbbox(fig.canvas.get_renderer())
            tw, th = bb.width, bb.height          # inches
            # savefig pads the tight bbox by pad_inches (0.1 each side) — include it
            dpi_fit = min(dpi, int(max_px / (max(tw, th) + 0.25)))
        except Exception:
            dpi_fit = dpi
        fig.savefig(out_dir / f'{stem}.png', bbox_inches='tight', dpi=dpi_fit)

    try:
        _write()
    except Exception:
        # constrained layout crashes on some twin-axis / colourbar combos
        # (matplotlib ZeroDivisionError); fall back to tight so saving never fails.
        try:
            fig.set_layout_engine('none')
            fig.tight_layout()
        except Exception:
            pass
        _write()
    if close:
        plt.close(fig)
    if verbose:
        print(f'  -> {stem}.png/.svg  (png dpi={dpi})')
