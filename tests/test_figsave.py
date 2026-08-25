# -*- coding: utf-8 -*-
"""First tests for ``nn_decoder/figsave.py`` (2026-08-25 audit, item B8).

The ≤ max_px PNG cap is a hard contract (CLAUDE.md: a figure the reader
cannot open is a figure that cannot be checked), but until now nothing in CI
verified it — and the tight-bbox estimate it relied on silently fell back to
the nominal dpi on failure (the measured 2026-08-23 escape: 6/13 figures at
1603–1822 px). These tests pin: both formats written, the cap held on the
actual bytes (IHDR-measured, not estimated), the cap held even when the
bbox measurement is unavailable, and the verbose line reporting the dpi
actually used.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NN_DECODER = os.path.join(REPO_ROOT, 'nn_decoder')
if NN_DECODER not in sys.path:
    sys.path.insert(0, NN_DECODER)

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from figsave import save_fig, _png_size  # noqa: E402


def _fig(w=6, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.plot([0, 1], [0, 1])
    return fig


def test_writes_both_formats(tmp_path):
    save_fig(_fig(), tmp_path, 'basic', verbose=False)
    assert (tmp_path / 'basic.svg').is_file()
    assert (tmp_path / 'basic.png').is_file()


def test_png_cap_holds_on_actual_pixels(tmp_path):
    save_fig(_fig(), tmp_path, 'capped', max_px=800, verbose=False)
    assert max(_png_size(tmp_path / 'capped.png')) <= 800


def test_png_cap_holds_with_outside_legend(tmp_path):
    # bbox_inches='tight' grows the canvas for an outside legend — the case
    # where a nominal-size dpi overshoots.
    fig, ax = plt.subplots(figsize=(11, 4))
    for i in range(4):
        ax.plot([0, 1], [i, i + 1], label=f'a rather long series label {i}')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.set_title('a long, long, long title that stretches the tight bbox out')
    save_fig(fig, tmp_path, 'legend', max_px=1600, verbose=False)
    assert max(_png_size(tmp_path / 'legend.png')) <= 1600


def test_png_cap_holds_when_bbox_measurement_fails(tmp_path, monkeypatch):
    # The pre-fix escape: get_tightbbox raising left the nominal dpi in
    # place and the PNG overshot. The post-save pixel check must catch it.
    fig = _fig(w=12, h=3)
    # Break only figsave's measurement seam, not savefig itself.
    import figsave as fs
    monkeypatch.setattr(
        fs, '_tight_size_inches',
        lambda f: (_ for _ in ()).throw(RuntimeError('boom')))
    save_fig(fig, tmp_path, 'nobbox', max_px=1000, verbose=False)
    assert max(_png_size(tmp_path / 'nobbox.png')) <= 1000


def test_verbose_reports_the_dpi_actually_used(tmp_path, capsys):
    save_fig(_fig(w=12, h=3), tmp_path, 'verbose', max_px=600, svg_dpi=140)
    out = capsys.readouterr().out
    assert 'verbose.png/.svg' in out
    # nominal dpi would be min(140, 600/12)=50; the printed dpi must be the
    # one that produced the file, i.e. consistent with the measured pixels.
    import re
    m = re.search(r'png dpi=(\d+)', out)
    assert m, out
    dpi_reported = int(m.group(1))
    w_px, _ = _png_size(tmp_path / 'verbose.png')
    # savefig pads the tight bbox, so allow slack — but the reported dpi must
    # be of the right magnitude for the actual file, not the svg_dpi bound.
    assert dpi_reported <= 60
    assert w_px <= 600


def test_png_size_rejects_non_png(tmp_path):
    p = tmp_path / 'fake.png'
    p.write_bytes(b'not a png at all, definitely')
    with pytest.raises(ValueError):
        _png_size(p)
