from bed.apnmc import APNMC
from simulator.stats5 import STATS5
import torch
from utils.plot import plot_hist_marginals
import torch.multiprocessing as mp
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import distributions
import argparse

import torch
from scipy.stats import gaussian_kde


def plot_mean_errorbar_columns(tensor):
    # Calculate the mean and standard deviation of each column
    mean = torch.mean(tensor, dim=0)
    std = torch.std(tensor, dim=0)/np.sqrt(tensor.shape[0])

    # Create x-axis values for each column index
    column_indices = range(tensor.shape[1])

    # Plot the mean and error bar
    for i, color in enumerate(color_list):       
        plt.errorbar(column_indices[i], mean[i], yerr=std[i], fmt='o', capsize=4, color=color, label=label_list[i])
        
    plt.xticks(column_indices, label_list, fontsize=18)
    plt.yticks(fontsize=16)
    plt.ylabel('Posterior entropy', fontsize=18)
    plt.grid(linestyle='--')
    
label_list = ["BEEG-AP", "UEEG-MCMC", "PCE", "GradBED"]
marker_list = ["x", "+", "o", "^"]
color_list = ['blue', 'red', 'purple', 'orange']

even =  True
even = False
noise_std = 0.01

if even is True:
    head = '_even'
else:
    head = ''
    
entropy_stats = []
for i, method in enumerate(['apnmc', 'mcmc', 'pce', 'gradbed']):
    with open("results/STATS5/posterior_val_{0}_noise_std_{1}{2}.pkl".format(method, noise_std,head), "rb") as file:
        entropy_list = pickle.load(file)
        entropy_stats.append(entropy_list)

entropy_stats=np.concatenate(entropy_stats).reshape(4,-1).T
plot_mean_errorbar_columns(torch.Tensor(entropy_stats))
path = 'figs/STATS5/posterior_val_{0}{1}'.format(noise_std,head)+'.eps'
plt.savefig(path, format='eps', bbox_inches='tight', dpi=300)