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

    
    
def compute_posterior_entropy(method, noise_std): 
    if method == 'gradbed':
        torch.manual_seed(2377)
        with open("results/STATS5/gradbed_noise_std_{0}.pkl".format(noise_std), "rb") as file:
            results = pickle.load(file)
    elif method == 'ace':
        torch.manual_seed(2377)
        try:
            with open("results/STATS5/ace_noise_std_{0}.pkl".format(noise_std), "rb") as file:
                results = pickle.load(file)
        except:
            results = None
    elif method == 'mcmc':
        torch.manual_seed(2377)
        with open("results/STATS5/mcmc_noise_std_{0}.pkl".format(noise_std), "rb") as file:
            results = pickle.load(file)
    elif method == 'apnmc':
        torch.manual_seed(2377)
        with open("results/STATS5/apnmc_noise_std_{0}.pkl".format(noise_std), "rb") as file:
            results= pickle.load(file)
    elif method == 'pce':
        torch.manual_seed(2377)
        with open("results/STATS5/pce_noise_std_{0}.pkl".format(noise_std), "rb") as file:
            results = pickle.load(file)
        
    if results is None:
        pass
    else:
        design_num, dim = results.shape
        n_mc = 100
        
        def compute_stats(init_design):
            entropy_arr = np.zeros(n_mc)
            init_design = torch.Tensor(init_design)
            simulator = STATS5(init_design)
            simulator.noise_std = noise_std
            for i in range(n_mc):
                parameters = simulator.prior.sample((1,))
                _, ys = simulator.forward(parameters)
                parameters_prime = adaptive_mh2([parameters, ys], simulator, 1000, 1)
                entropy = estimate_entropy_kde(parameters_prime)
                entropy_arr[i] = entropy
            print(entropy.mean())
            return entropy_arr
                
        entropy_list = compute_stats(results[-1])
        with open("results/STATS5/posterior_val_{0}_noise_std_{1}.pkl".format(method, noise_std), "wb") as file:
            pickle.dump(entropy_list, file)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='STATS experiments.')
    parser.add_argument('method', type=str, help='the name of the model')
    parser.add_argument('noise_std', type=float, help='the number of repeated trials')
    args = parser.parse_args()

    compute_posterior_entropy(args.method, args.noise_std)