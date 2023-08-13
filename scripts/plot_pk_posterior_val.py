from bed.apnmc import APNMC
from simulator.pk import PK
import torch
from utils.plot import plot_hist_marginals
import torch.multiprocessing as mp
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import distributions

import torch
from scipy.stats import gaussian_kde

inner_noise_std=0
noise_std=0.03162

def plot_mean_errorbar_entries(num_sims_arr, data_list, label, color):
    # Calculate the mean and standard deviation of each entry
    mean_list = [np.mean(entry) for entry in data_list]
    std_list = [np.std(entry)/np.sqrt(entry.shape[0]) for entry in data_list]

    # Create x-axis values for each entry index
    entry_indices = range(len(data_list))

    # Plot the mean and error bar for each entry
    plt.errorbar(num_sims_arr, mean_list, yerr=std_list, label=label,color=color, fmt='o', capsize=4)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('No. simulations', fontsize=18)
    plt.ylabel('Posterior entropy', fontsize=18)
    #plt.xscale('log')
    plt.grid(linestyle='--')
    #plt.title('Mean and Error Bar of Entries')
    #plt.show()
    
with open("results/pk/posterior_val_noise_std_{0}_{1}.pkl".format(inner_noise_std, noise_std), "rb") as file:
    stats_list = pickle.load(file)

lims = [0, 5e6]
label_list = ["BEEG-AP", "UEEG-MCMC", "PCE", "GradBED"]
marker_list = ["x", "+", "*", "^"]
color_list = ['blue', 'red', 'purple', 'orange']

design_num = 50000
max_idx = int(design_num*2/5)
compute_num = 5
compute_idx = np.round(np.logspace(np.log(1000), np.log(max_idx), compute_num, base=2.72))
num_sims_arr = 100*compute_idx

for i, stats in enumerate(stats_list):
    plot_mean_errorbar_entries(num_sims_arr, stats, label_list[i], color_list[i])
plt.legend(fontsize=16)
path = 'figs/pk/posterior_val_{0}_{1}'.format(inner_noise_std, noise_std)+'.png'
plt.savefig(path, format='png', bbox_inches='tight', dpi=500)