from bed.apnmc import APNMC, IS_Proposal
from bed.vnmc import VNMC
from bed.mcmc_gradient import MCMC_Gradient
from simulator.toy2 import Toy
import torch
import numpy as np
from bed.ace import ACE
from bed.gradbed import GradBED
from utils.get_models import get_proposal
from nn_.nets.conditionalgaussian import ConditionalGaussian
import pickle

import matplotlib.pyplot as plt

#init_design = torch.rand(2)
#noise_std = 0.01
sim_budget = 200*100
n_steps = 100000
lr = 5e-3


## method 1
def apnmc_design(args, plot_history=False):
    init_design, noise_std = args
    simulator = Toy(init_design, noise_std=noise_std)
    apnmc = APNMC(simulator, sim_budget=sim_budget)
    apnmc.learn_design(lr=lr, n_out=100, n_in=1, n_steps=n_steps, sampling_method="prior")
    
    if plot_history:
        design_history = torch.cat(apnmc.design_history).view(-1,2).numpy()
        sim_count_history = apnmc.sim_count_history
        plt.plot(sim_count_history, design_history[:,0], color="blue")
        plt.plot(sim_count_history, design_history[:,1], color="blue")
    return simulator.design.detach()

## method 2
def vnmc_design(args, plot_history=False):
    init_design, noise_std = args
    simulator = Toy(init_design, noise_std=noise_std)
    vnmc = VNMC(simulator, sim_budget=sim_budget)
    vnmc.learn_design(lr=lr, n_out=10, n_in=10, n_steps=n_steps, sampling_method="pce")
    
    if plot_history:
        design_history = torch.cat(vnmc.design_history).view(-1,2).numpy()
        sim_count_history = vnmc.sim_count_history
        plt.plot(sim_count_history, design_history[:,0], color="red")
        plt.plot(sim_count_history, design_history[:,1], color="red")
    return simulator.design.detach()


## method mcmc-gradient
def mcmc_design(args, plot_history=False):
    init_design, noise_std = args
    simulator = Toy(init_design, noise_std=noise_std)
    sampling_method="slice"
    mcmc_param = np.sqrt(noise_std)
    #sampling_method = "mh"
    #mcmc_param = 50*noise_std**2
    mcmcgradient = MCMC_Gradient(simulator, mcmc_param=mcmc_param, sampling_method=sampling_method, sim_budget=sim_budget)
    mcmcgradient.learn_design(lr=lr, n_out=1, n_in=1, n_steps=n_steps,)
    acceptance_rate_list = simulator.acceptance_rate
    print(sum(acceptance_rate_list)/len(acceptance_rate_list))
    
    if plot_history:
        design_history = torch.cat(mcmcgradient.design_history).view(-1,2).numpy()
        sim_count_history = mcmcgradient.sim_count_history
        plt.plot(sim_count_history, design_history[:,0], color="black")
        plt.plot(sim_count_history, design_history[:,1], color="black")
    
    return simulator.design.detach()

## method gradbed
def gradbed_design(args, plot_history=False):
    init_design, noise_std = args
    simulator = Toy(init_design, noise_std=noise_std)
    proposal = get_proposal("fullyconnected", simulator.observation_dim, simulator.parameter_dim)

    gradbed = GradBED(simulator, net=proposal, sim_budget=sim_budget)
    gradbed.learn_design(lr=lr, batch_size=100, n_steps=n_steps)
    
    if plot_history:
        design_history = torch.cat(gradbed.design_history).view(-1, 2).numpy()
        sim_count_history = gradbed.sim_count_history
        for i in range(2):
            plt.plot(sim_count_history, design_history[:,i], color="yellow")
            plt.plot(sim_count_history, design_history[:,i], color="yellow")
    return simulator.design.detach()

## method ace 
def ace_design(args, plot_history=False):
    init_design, noise_std = args
    simulator = Toy(init_design, noise_std=noise_std)
    proposal = ConditionalGaussian(simulator.observation_dim, 50)

    ace = ACE(simulator, is_proposal=proposal,sim_budget=sim_budget)
    ace.learn_design(lr=lr, n_out=10, n_in=10, n_steps=n_steps)
    
    if plot_history:
        design_history = torch.cat(ace.design_history).view(-1, 2).numpy()
        sim_count_history = ace.sim_count_history
        for i in range(2):
            plt.plot(sim_count_history, design_history[:,i], color="green")
            plt.plot(sim_count_history, design_history[:,i], color="green")
        
    return simulator.design.detach()

if __name__ == "__main__":
    n_trials = 20
    torch.manual_seed(2255)
    design_list = torch.rand(n_trials, 2)
    
    noise_std = 0.0001
    noise_list = [noise_std]*n_trials    
    gradbed_results = map(gradbed_design, zip(design_list, noise_list))
    ace_results = map(ace_design, zip(design_list, noise_list))
    mcmc_results = map(mcmc_design, zip(design_list, noise_list))
    apnmc_results = map(apnmc_design, zip(design_list, noise_list))
    pce_results = map(vnmc_design, zip(design_list, noise_list))
    gradbed_results = list(gradbed_results)
    ace_results = list(ace_results)
    mcmc_results = list(mcmc_results)
    apnmc_results = list(apnmc_results)
    pce_results = list(pce_results)
    with open("results/toy/noise_std_{0}.pkl".format(noise_std), "wb") as file:
        pickle.dump([apnmc_results, mcmc_results, ace_results, pce_results, gradbed_results], file)
        
    noise_std = 0.01
    noise_list = [noise_std]*n_trials    
    gradbed_results = map(gradbed_design, zip(design_list, noise_list))
    ace_results = map(ace_design, zip(design_list, noise_list))
    mcmc_results = map(mcmc_design, zip(design_list, noise_list))
    apnmc_results = map(apnmc_design, zip(design_list, noise_list))
    pce_results = map(vnmc_design, zip(design_list, noise_list))
    gradbed_results = list(gradbed_results)
    ace_results = list(ace_results)
    mcmc_results = list(mcmc_results)
    apnmc_results = list(apnmc_results)
    pce_results = list(pce_results)
    with open("results/toy/noise_std_{0}.pkl".format(noise_std), "wb") as file:
        pickle.dump([apnmc_results, mcmc_results, ace_results, pce_results, gradbed_results], file)
        