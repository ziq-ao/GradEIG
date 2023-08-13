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

label_list = ["BEEG-AP", "UEEG-MCMC", "ACE", "PCE", "GredBED"]
marker_list = ["x", "+", "*", "o", "^"]
color_list = ['blue', 'red', 'green', 'purple', 'orange']

with open("results/pk/ground_truth_noise_std_{0}_{1}.pkl".format(inner_noise_std, noise_std), "rb") as file:
    results = pickle.load(file)

import numpy as np
import matplotlib.pyplot as plt

def plot_moving_average_with_error_bar(x, y, window_size, color='r', label=None):
    # Calculate the moving average and standard deviation
    moving_avg = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
    std_dev = np.std([y[i:i+window_size] for i in range(len(y)-window_size+1)], axis=1)

    # Plot the moving average with continuous transparent error bar
    plt.plot(x[window_size-1:], moving_avg, color=color, label=label, linewidth=2)
    plt.fill_between(x[window_size-1:], moving_avg - std_dev, moving_avg + std_dev, color=color, alpha=0.15)
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('No. simulations', fontsize=18)
    plt.ylabel('Estimated EIG', fontsize=18)
    plt.grid(linestyle='--')
    
    
num_sims_arr = np.linspace(0,2000000,100)
for i in range(len(results)):
    plot_moving_average_with_error_bar(
                num_sims_arr,
                results[i],
                #alpha=1,
                #marker=marker_list[i],
                color=color_list[i],
                label = label_list[i],
                window_size=10,)
    plt.legend(fontsize=16)

path = 'figs/pk/ground_truth_{0}_{1}'.format(inner_noise_std, noise_std)+'.png'
plt.savefig(path, format='png', bbox_inches='tight', dpi=500)