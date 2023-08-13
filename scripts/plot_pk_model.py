import torch
from utils.plot import plot_hist_marginals
import torch.multiprocessing as mp
import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_matrix_vs_vector(matrix, lims, color, title):
    n, d = matrix.shape
    
    vector = torch.linspace(lims[0], lims[1], n)
    # Iterate over each column of the matrix
    for col in range(d):
        plt.plot(vector, matrix[:, col], color=color)

    # Add labels and title
    plt.xlabel("No. simulations")
    plt.ylabel("Design value")
    plt.title(title)
    plt.grid(linestyle='--')

    # Show the plot
    #plt.show()
    
# =============================================================================
# inner_noise_std=0.1
# noise_std=0.3162
# =============================================================================

inner_noise_std=0.1
noise_std=0.3162

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
for i, results in enumerate([apnmc_results, mcmc_results, ace_results, pce_results, gradbed_results]):    
    if results is None:
        pass
    else:
        plot_matrix_vs_vector(results, lims, color=color_list[i], title=label_list[i])
        path = 'figs/pk/'+label_list[i]+'-'+str(inner_noise_std)+'-'+str(noise_std)+'.eps'
        plt.savefig(path, format='eps')
        plt.show()
        plt.close()