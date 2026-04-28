"""Run frechet(iterations=1) using the module name in argv[1].
Pickles {time, mu, betas_aligned} to argv[2]. Run in its own subprocess
so geomstats / CUDA state is clean.
"""
import os
os.environ['NUMEXPR_MAX_THREADS'] = '35'
import sys
import time
import pickle
import importlib

import numpy as np
import pandas as pd

mod_name = sys.argv[1]
out_path = sys.argv[2]

mod = importlib.import_module(mod_name)
import torch

tslen = 200
data_folder = "/mnt/sdb/arafat/stroke_riemann/csv_r"
files = sorted(os.listdir(data_folder), key=lambda x: int(x.split('_')[0][2:]))

all_data = {}
for f in files:
    g = pd.read_csv(os.path.join(data_folder, f))
    gait_cycles = g.iloc[:, :-1].values
    n_rows = gait_cycles.shape[0]
    result = gait_cycles.reshape(n_rows, 32, 3).transpose(1, 2, 0)
    pid = f.split('_')[0][2:]
    all_data[pid] = result

keys = list(all_data.keys())
data_stroke = {k: all_data[k] for k in keys[:44]}
data_healthy = {k: all_data[k] for k in keys[44:]}
gamma_t = np.linspace(0, 1, tslen)

np.random.seed(42)
betas_resampled_stroke = mod.process_kinematic({k: data_stroke[k].copy() for k in data_stroke}, gamma_t)
betas_resampled_healthy = mod.process_kinematic({k: data_healthy[k].copy() for k in data_healthy}, gamma_t)
betas_all = betas_resampled_stroke + betas_resampled_healthy
mu_init = betas_resampled_healthy[np.random.choice(range(len(betas_resampled_healthy)))]

torch.cuda.synchronize()
t0 = time.perf_counter()
mu, betas_aligned, gammas, tangent_vec, history = mod.frechet(
    betas_all, gamma_t, mu_init, iterations=1, plot=False
)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

with open(out_path, "wb") as f:
    pickle.dump({"time": elapsed, "mu": mu, "betas_aligned": betas_aligned}, f)

print(f"{mod_name}: {elapsed:.2f} s")
