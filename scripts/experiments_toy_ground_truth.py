from bed.apnmc import APNMC, IS_Proposal
from bed.vnmc import VNMC
from simulator.toy2 import Toy
import torch
import numpy as np
import pickle

import matplotlib.pyplot as plt

def compute_ground_truth_eig(args):
    design1, design2, noise_std, n_out, n_in = args
    init_design = torch.cat((design1,design2))
    simulator = Toy(init_design, noise_std=noise_std)
    vnmc = VNMC(simulator)
    return vnmc.pce( n_out, n_in)
    
if __name__ == "__main__":
    # results for noise_std = 0.01
    n_grids = 41
    x = torch.linspace(0,1,n_grids)
    y = torch.linspace(0,1,n_grids)
    X, Y = torch.meshgrid(x, y)
    noise_std = 0.01
    n_out = 10000
    n_in = 10000
    zipper = zip(
        X.reshape(n_grids**2,1),
        Y.reshape(n_grids**2,1), 
        n_grids**2*[noise_std],
        n_grids**2*[n_out],
        n_grids**2*[n_in])
    
    results = np.zeros(n_grids**2)
    for i, args in enumerate(zipper):
        results[i] = compute_ground_truth_eig(args)
        if i % n_grids == 1:
            with open("results/toy/ground_truth_noise_std_{0}.pkl".format(noise_std), "wb") as file:
                pickle.dump(results.reshape(n_grids, n_grids), file)
    