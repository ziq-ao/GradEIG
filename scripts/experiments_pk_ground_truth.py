from bed.apnmc import APNMC
from simulator.pk import PK
import torch
from utils.plot import plot_hist_marginals
import torch.multiprocessing as mp
import pickle
import numpy as np
import matplotlib.pyplot as plt

inner_noise_std=0.1
noise_std=0.3162

# =============================================================================
# inner_noise_std=0.1
# noise_std=0.3162
# =============================================================================

# =============================================================================
# inner_noise_std=0
# noise_std=0.03162
# =============================================================================

with open("results/pk/gradbed_noise_std_{0}_{1}.pkl".format(inner_noise_std, noise_std), "rb") as file:
    gradbed_results = pickle.load(file)
try:
    with open("results/pk/ace_noise_std_{0}_{1}.pkl".format(inner_noise_std, noise_std), "rb") as file:
        ace_results = pickle.load(file)
except:
    ace_results = None
with open("results/pk/mcmc_noise_std_{0}_{1}.pkl".format(inner_noise_std, noise_std), "rb") as file:
    mcmc_results = pickle.load(file)
with open("results/pk/apnmc_noise_std_{0}_{1}.pkl".format(inner_noise_std, noise_std), "rb") as file:
    apnmc_results= pickle.load(file)
with open("results/pk/pce_noise_std_{0}_{1}.pkl".format(inner_noise_std, noise_std), "rb") as file:
    pce_results = pickle.load(file)

lims = [0, 5e6]
label_list = ["BEEG-AP", "UEEG-MCMC", "ACE", "PCE", "gradbed"]
marker_list = ["x", "+", "*", "o", "^"]
color_list = ['blue', 'red', 'green', 'purple', 'orange']
gt_list = []
for i, results in enumerate([apnmc_results, mcmc_results, ace_results, pce_results, gradbed_results]):    
    if results is None:
        pass
    else:
        design_num, dim = results.shape
        max_idx = int(design_num*2/5)
        compute_num = 100
        n_out = 10000
        compute_idx = np.round(np.linspace(1, max_idx, compute_num))
        gt = np.zeros(compute_num)
        for j in range(compute_num):
            init_design = results[int(compute_idx[j])]
            init_design = torch.Tensor(init_design)
            simulator = PK(init_design)
            simulator.inner_noise_std = inner_noise_std
            simulator.noise_std = noise_std
            apnmc = APNMC(simulator)
            gt[j] = apnmc.nmc_reuse(n_out).detach()
        gt_list.append(gt)

        with open("results/pk/ground_truth_noise_std_{0}_{1}.pkl".format(inner_noise_std, noise_std), "wb") as file:
            pickle.dump(gt_list, file)