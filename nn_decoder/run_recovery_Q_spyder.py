# -*- coding: utf-8 -*-
"""Run the Q-only crossover recovery from Spyder.

Wraps ``run_fixed_recovery.run_recovery_experiment`` so you can step
through it cell-by-cell with ``%runcell -i`` instead of running the
full multi-target driver. The crossover trains, for each mouse and
each target-architecture (spat / temp), a fresh decoder against the
**saved full-population predictions** of the production Q decoder
under ``results/clean_2026_05_19/Q_PCA_half_100ms_condmean/``.

Defensive-builtin-imports note: `%runcell -i` shares the kernel
namespace; if a previous cell ran ``from numpy import *`` the
``all``/``any`` builtins are shadowed. We re-import explicitly at
the top of each cell that uses them.

Output paths
------------
- Recovery cache : ``nn_decoder/recovery_cache_fixed_perception.npy``
- Plots          : ``Recovery_Plots_Fixed/`` (next to the script)

Set ``FORCE_RERUN = True`` to delete the cache and re-train; leave
``False`` to just regenerate plots from the existing cache.
"""
# %% Setup -- imports + defensive builtins
from builtins import all, any, sum, min, max, abs   # in case numpy.* shadows
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paths
from run_fixed_recovery import (
    run_recovery_experiment, plot_recovery_matrix, plot_recovery_scatter,
)

print(f'sys.path[0] = {sys.path[0]}')
print(f'cwd        = {os.getcwd()}')

# %% Config -- pick the Q base file + flag for re-run
SPLIT = 'stratified_balanced'
RUN_NAME = 'clean_2026_05_19'
SLUG = 'Q_PCA_half_100ms_condmean'      # the only Q production slug on disk
TARGET_NAME = 'perception'              # 'perception' == Q (the recovery-cache key)
FORCE_RERUN = True                      # set False to load the existing cache

BASE_FILE = paths.RESULTS / RUN_NAME / SLUG / f'{SPLIT}.mat'
CACHE_FILE = paths.recovery_cache(TARGET_NAME)

print(f'BASE_FILE  = {BASE_FILE}')
print(f'  exists?  = {BASE_FILE.exists()}')
print(f'CACHE_FILE = {CACHE_FILE}')
print(f'  exists?  = {CACHE_FILE.exists()}')

# %% Move stale cache aside if we want a fresh run
import shutil
from datetime import datetime
if FORCE_RERUN and CACHE_FILE.exists():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = CACHE_FILE.with_suffix(f'.npy.bak.{stamp}')
    shutil.move(str(CACHE_FILE), str(backup))
    print(f'Moved old cache aside  -> {backup.name}')
else:
    print('No cache to move (or FORCE_RERUN=False).')

# %% Run the crossover experiment (Q only)
# This trains, for each mouse, the spat & temp decoders against
# spat-target ('Ground Truth = Fitted SPAT') and temp-target
# ('Ground Truth = Fitted TEMP') outputs.  Roughly:
#   6 mice  x  2 target archs  x  (1 run_animal_decoder call that
#   internally trains 4 networks: spat, spat_shf, temp, temp_shf)
# Expect tens of minutes on a Mac GPU; an hour-ish on CPU.
if not BASE_FILE.exists():
    raise FileNotFoundError(
        f'{BASE_FILE} not found.  Either pull the production Q .mat '
        f'from gpu1 or point SLUG at a different run.')

recov_data = run_recovery_experiment(str(BASE_FILE), TARGET_NAME)
print('\nDone training. Branches:', [k for k in recov_data if k != 'base_config'])

# %% Render the recovery plots
plot_recovery_matrix(recov_data, TARGET_NAME)
plot_recovery_scatter(recov_data, TARGET_NAME)
print('Plots written under  ./Recovery_Plots_Fixed/')

# %% (Optional) Quick numeric peek at the loss matrix
import numpy as np
import scipy.io as sio
from decoder_plotting_utils import calc_fit_loss

base = sio.loadmat(str(BASE_FILE), simplify_cells=True)
base_cfg = base['config']
loss_fn = base_cfg.get('custom_loss_func', base_cfg.get('loss_func', 'PCA'))
pcs = base['results']['mouse_0']['Dist']['pcs']
evar = base['results']['mouse_0']['Dist']['explained_var']

rows = []
for target_arch in ('spat', 'temp'):
    branch = recov_data[target_arch]
    for mid, res in branch.items():
        for fit_arch in ('spat', 'temp'):
            d = res['Dist'][fit_arch]
            fl = calc_fit_loss(d, loss_fn, pcs, evar)
            rows.append((mid, target_arch, fit_arch, float(np.nanmean(fl))))
import pandas as pd
df = pd.DataFrame(rows, columns=['mouse', 'target_arch', 'fit_arch', 'mean_loss'])
print('\nCrossover loss matrix (mean across trials, all 6 mice):')
print(df.pivot_table(index='target_arch', columns='fit_arch',
                     values='mean_loss', aggfunc='mean').round(4))
