from bed.apnmc import APNMC, IS_Proposal
from bed.vnmc import VNMC
from bed.mcmc_gradient import MCMC_Gradient
from simulator.stats5 import STATS5
import torch
from bed.ace import ACE
from bed.gradbed import GradBED
from utils.get_models import get_proposal
from nn_.nets.fullyconnected import FullyConnected

import matplotlib.pyplot as plt
from nn_.nde import MixtureOfGaussiansMADE, MultivariateGaussianMDN
from torch import nn
import pickle

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

even =  True
#even = False
noise_std = 0.001

if even is True:
    head = 'even_'
else:
    head = ''
    
try:
    with open("results/STATS5/{1}gradbed_noise_std_{0}.pkl".format(noise_std, head), "rb") as file:
        gradbed_results = pickle.load(file)
except:
    gradbed_results = None
    
try:
    with open("results/STATS5/{1}ace_noise_std_{0}.pkl".format(noise_std, head), "rb") as file:
        ace_results = pickle.load(file)
except:
    ace_results = None

try:
    with open("results/STATS5/{1}mcmc_noise_std_{0}.pkl".format(noise_std, head), "rb") as file:
        mcmc_results = pickle.load(file)
except:
    mcmc_results = None
        
try:
    with open("results/STATS5/{1}apnmc_noise_std_{0}.pkl".format(noise_std, head), "rb") as file:
        apnmc_results= pickle.load(file)
except:
    apnmc_results = None

try:    
    with open("results/STATS5/{1}pce_noise_std_{0}.pkl".format(noise_std, head), "rb") as file:
        pce_results = pickle.load(file)
except:
    pce_results = None
    
lims = [0, 5e5]
label_list = ["BEEG-AP", "UEEG-MCMC", "ACE", "PCE", "gradbed"]
marker_list = ["x", "+", "*", "o", "^"]
color_list = ['blue', 'red', 'green', 'purple', 'orange']
for i, results in enumerate([apnmc_results, mcmc_results, ace_results, pce_results, gradbed_results]):    
    if results is None:
        pass
    else:
        plot_matrix_vs_vector(results, lims, color=color_list[i], title=label_list[i])
        path = 'figs/STATS5/'+label_list[i]+'-'+'-'+str(noise_std)+head+'.eps'
        plt.savefig(path, format='eps')
        plt.show()
        plt.close()