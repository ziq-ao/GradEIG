from bed.apnmc import APNMC, IS_Proposal
from bed.vnmc import VNMC
from bed.mcmc_gradient import MCMC_Gradient
from simulator.pk import PK
import torch
from bed.ace import ACE
from bed.gradbed import GradBED
from utils.get_models import get_proposal
from nn_.nets.fullyconnected import FullyConnected

import matplotlib.pyplot as plt
from nn_.nde import MixtureOfGaussiansMADE, MultivariateGaussianMDN
from torch import nn
import pickle

dim = 10
init_design = torch.linspace(0.5, 20, dim)
n_steps = 1000000
lr = 1e-2
sim_budget = 50000*100


## method 1
def apnmc_design(args, plot_history=False):
    init_design, inner_noise_std, noise_std = args
    simulator = PK(init_design)
    simulator.inner_noise_std = inner_noise_std
    simulator.noise_std = noise_std
    apnmc = APNMC(simulator, sim_budget=sim_budget)
    apnmc.learn_design(lr=lr, n_out=100, n_in=1, n_steps=n_steps, sampling_method="prior")
    design_history = torch.cat(apnmc.design_history).view(-1,dim).numpy()
    if plot_history:
        sim_count_history = apnmc.sim_count_history
        for i in range(dim):
            plt.plot(sim_count_history,design_history[:,i], color="blue")
            plt.plot(sim_count_history,design_history[:,i], color="blue")
    return design_history

## method 2
def vnmc_design(args, plot_history=False):
    init_design, inner_noise_std, noise_std = args
    simulator = PK(init_design)
    simulator.inner_noise_std = inner_noise_std
    simulator.noise_std = noise_std
    vnmc = VNMC(simulator, sim_budget=sim_budget)
    vnmc.learn_design(lr=lr, n_out=10, n_in=10, n_steps=n_steps, sampling_method="pce")
    design_history = torch.cat(vnmc.design_history).view(-1,dim).numpy()
    if plot_history:
        sim_count_history = vnmc.sim_count_history
        for i in range(dim):
            plt.plot(sim_count_history, design_history[:,i], color="red")
            plt.plot(sim_count_history, design_history[:,i], color="red")
    return design_history

## method mcmc-gradient
def mcmc_design(args, plot_history=False):
    init_design, inner_noise_std, noise_std = args
    simulator = PK(init_design)
    simulator.inner_noise_std = inner_noise_std
    simulator.noise_std = noise_std
    mcmcgradient = MCMC_Gradient(simulator, mcmc_param=1, sim_budget=sim_budget)
    mcmcgradient.sampling_method="adaptive_mh2"
    mcmcgradient.learn_design(lr=lr, n_out=1, n_in=1, n_steps=n_steps)
    design_history = torch.cat(mcmcgradient.design_history).view(-1,dim).numpy()
    if plot_history:
        sim_count_history = mcmcgradient.sim_count_history
        for i in range(dim):
            plt.plot(sim_count_history, design_history[:,i], color="black")
            plt.plot(sim_count_history, design_history[:,i], color="black")
    
    return design_history

## method gradbed
def gradbed_design(args, plot_history=False):
    init_design, inner_noise_std, noise_std = args
    simulator = PK(init_design)
    simulator.inner_noise_std = inner_noise_std
    simulator.noise_std = noise_std
    proposal = FullyConnected(
        var1_dim=simulator.parameter_dim,
        var2_dim=simulator.observation_dim, 
        L=1,
        H=300)
   
    gradbed = GradBED(simulator, net=proposal, sim_budget=sim_budget)
    gradbed.learn_design(lr=lr, batch_size=100, n_steps=n_steps,lr_net=1e-3)
    design_history = torch.cat(gradbed.design_history).view(-1, dim).numpy()
    if plot_history:
        sim_count_history = gradbed.sim_count_history
        for i in range(dim):
            plt.plot(sim_count_history, design_history[:,i], color="yellow")
            plt.plot(sim_count_history, design_history[:,i], color="yellow")
    return design_history 

## method ace 
def ace_design(args, plot_history=False):
    init_design, inner_noise_std, noise_std = args
    simulator = PK(init_design)
    simulator.inner_noise_std = inner_noise_std
    simulator.noise_std = noise_std
    # build neural proposal
    hidden_features = 50
    proposal = MultivariateGaussianMDN(
        features=simulator.parameter_dim,
        context_features=simulator.observation_dim,
        hidden_features=hidden_features,
        hidden_net=nn.Sequential(
            nn.Linear(simulator.observation_dim, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
        ),
        num_components=1,
        custom_initialization=True,
    )
    
    ace = ACE(simulator, is_proposal=proposal,sim_budget=sim_budget)
    ace.learn_design(lr=lr, n_out=10, n_in=10, n_steps=n_steps)
    design_history = torch.cat(ace.design_history).view(-1, dim).numpy()
    if plot_history:
        sim_count_history = ace.sim_count_history
        for i in range(dim):
            plt.plot(sim_count_history, design_history[:,i], color="green")
            plt.plot(sim_count_history, design_history[:,i], color="green")
        
    return design_history


if __name__ == "__main__":
    torch.manual_seed(2377)
    
    def run_experiments(inner_noise_std, noise_std):
        gradbed_results = gradbed_design([init_design, inner_noise_std, noise_std])
        with open("results/pk/gradbed_noise_std_{0}_{1}.pkl".format(inner_noise_std, noise_std), "wb") as file:
            pickle.dump(gradbed_results, file)
        try:
            ace_results = ace_design([init_design, inner_noise_std, noise_std])
            with open("results/pk/ace_noise_std_{0}_{1}.pkl".format(inner_noise_std, noise_std), "wb") as file:
                pickle.dump(ace_results, file)
        except:
            pass
        mcmc_results = mcmc_design([init_design, inner_noise_std, noise_std])
        with open("results/pk/mcmc_noise_std_{0}_{1}.pkl".format(inner_noise_std, noise_std), "wb") as file:
            pickle.dump(mcmc_results, file)
        apnmc_results = apnmc_design([init_design, inner_noise_std, noise_std])
        with open("results/pk/apnmc_noise_std_{0}_{1}.pkl".format(inner_noise_std, noise_std), "wb") as file:
            pickle.dump(apnmc_results, file)
        pce_results = vnmc_design([init_design, inner_noise_std, noise_std])
        with open("results/pk/pce_noise_std_{0}_{1}.pkl".format(inner_noise_std, noise_std), "wb") as file:
            pickle.dump(pce_results, file)
    
    #run_experiments(inner_noise_std=0.1, noise_std=0.3162)
    #run_experiments(inner_noise_std=0.01, noise_std=0.03162)
    run_experiments(inner_noise_std=0, noise_std=0.03162)
             