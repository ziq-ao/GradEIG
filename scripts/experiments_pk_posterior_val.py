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

def estimate_entropy_kde(tensor):
    # Reshape the tensor to a 2D array
    data = tensor.view(-1, tensor.shape[-1]).numpy()

    # Estimate the density using Gaussian KDE
    kde = gaussian_kde(data.T)

    # Calculate the entropy using the estimated density
    entropy =  -np.mean(kde.logpdf(data.T))

    return entropy


def adaptive_mh2(pair, simulator, n_in, step):
    parameters, ys = pair
    parameters = parameters.view(1,-1)
    ys = ys.view(1,-1)
    
    def potential(z):
        target_log_prob = (simulator.log_prob(ys, torch.Tensor(z), True)+
            simulator.prior.log_prob(torch.Tensor(z).view(1,-1))
            )
        return -target_log_prob
    
    init_state = parameters.view(-1).requires_grad_(True)
    hessian_matrix = compute_hessian(potential, init_state)   
    simulator.sim_count += init_state.shape[0]
    # The computation cost of second-order derivative is proportional to 
    # the dimentionality

    alpha = 0.1
    covariance_matrix_new = torch.inverse(hessian_matrix)
    if simulator.covariance_matrix is None:
        aver_cov_matrix = covariance_matrix_new
    else:
        aver_cov_matrix = (1-alpha)*simulator.covariance_matrix + alpha*covariance_matrix_new
    try:
        proposal = distributions.MultivariateNormal(
            loc=torch.zeros_like(init_state),
            covariance_matrix=aver_cov_matrix)
        simulator.covariance_matrix = aver_cov_matrix
    except:
        try:
            proposal = distributions.MultivariateNormal(
                loc=torch.zeros_like(init_state),
                covariance_matrix=simulator.covariance_matrix)
        except:
            parameters = simulator.prior.sample((1,))
            _, ys = simulator.forward(parameters)
            return adaptive_mh2([parameters, ys], simulator, n_in, step)
    #print(proposal.covariance_matrix)
        
    def log_Q(z_prime, z, step):
        return proposal.log_prob((z_prime-z)/step)
    
    thin = 2
    burn_in = thin
    Z0 = parameters+step*proposal.sample((1,))
    Zi = Z0
    samples = []
    acceptance_num = 0 
    pbar = tqdm(range(n_in*thin), disable=(True))
    Zi.requires_grad_()
    potential_Zi = potential(Zi).mean()
    simulator.sim_count -= 1
    # The potential of Z0 has already been computed before, so 1 count is substrated
    #sim number from the simulation counts. Just for the convenience of coding.
    for i in pbar:
        prop_Zi = Zi.detach() + step*proposal.sample((1,))
        prop_Zi.requires_grad_()
        potential_prop_Zi = potential(prop_Zi)
        log_ratio = -potential_prop_Zi.mean() + potential_Zi.mean() +\
                    log_Q(Zi, prop_Zi, step) - log_Q(prop_Zi, Zi, step)
        if torch.rand(1) < torch.exp(log_ratio):
            Zi = prop_Zi
            potential_Zi = potential_prop_Zi
            acceptance_num += 1
        samples.append(Zi.detach())
    ar = acceptance_num/(n_in*thin)
    simulator.acceptance_rate.append(ar)
    #print(ar)
    return torch.cat(samples, 0)[burn_in-1::thin]


from torch.autograd.functional import hessian
def compute_hessian(loss, variables):
    # Compute Hessian matrix
    hessian_matrix = hessian(loss, variables, create_graph=False)

    return hessian_matrix

inner_noise_std=0
noise_std=0.03162

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
stats_list = []
for i, results in enumerate([apnmc_results, mcmc_results, ace_results, pce_results, gradbed_results]):    
    if results is None:
        pass
    else:
        design_num, dim = results.shape
        max_idx = int(design_num*2/5)
        compute_num = 5
        n_mc = 1000
        compute_idx = np.round(np.logspace(np.log(1000), np.log(max_idx), compute_num, base=2.72))
        gt = np.zeros(compute_num)
        
        def compute_stats(init_design):
            entropy_arr = np.zeros(n_mc)
            init_design = torch.Tensor(init_design)
            simulator = PK(init_design)
            simulator.inner_noise_std = inner_noise_std
            simulator.noise_std = noise_std
            for i in range(n_mc):
                parameters = simulator.prior.sample((1,))
                _, ys = simulator.forward(parameters)
                parameters_prime = adaptive_mh2([parameters, ys], simulator, 1000, 1)
                entropy = estimate_entropy_kde(parameters_prime)
                entropy_arr[i] = entropy
            print(entropy.mean())
            return entropy_arr
                
        entropy_list = list(map(compute_stats, results[[int(idx) for idx in compute_idx]]))
        stats_list.append(entropy_list)
        with open("results/pk/posterior_val_noise_std_{0}_{1}.pkl".format(inner_noise_std, noise_std), "wb") as file:
            pickle.dump(stats_list, file)