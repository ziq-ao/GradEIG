import torch
from utils.plot import plot_hist_marginals
import torch.multiprocessing as mp
import pickle
import numpy as np
import matplotlib.pyplot as plt
from simulator.toy2 import Toy
from bed.mcmc_gradient import slice_sampling

from tqdm import tqdm
from torch import distributions

label_list = ["BEEG-AP", "UEEG-MCMC", "ACE", "PCE", "gradbed"]
marker_list = ["x", "+", "*", "o", "^"]
color_list = ['blue', 'red', 'green', 'purple', 'orange']

from scipy.stats import gaussian_kde

def estimate_entropy_and_plot(data, plot=False):
    # Estimate the density function using Gaussian KDE
    kde = gaussian_kde(data)
    x = np.linspace(min(data), max(data), 100)
    density_estimation = kde(x)

    # Calculate the entropy using the estimated density function
    entropy = -np.mean(kde.logpdf(data))

    # Plot the estimated density function and real data
    if plot:
        plt.figure()
        plt.hist(data, bins='auto', density=True, alpha=0.5, label='Real Data')
        plt.plot(x, density_estimation, label='Estimated Density')
        plt.xlabel('Data')
        plt.ylabel('Density')
        plt.title('Density Estimation')
        plt.legend()
        plt.show()

    return entropy


def compute_mse(prediction, ground_truth):
    # Ensure both tensors have the same shape
    assert prediction.shape[1] == ground_truth.shape[0], "Tensors must have compatible shapes."

    # Compute the squared difference between the prediction and ground truth
    squared_diff = (prediction - ground_truth) ** 2

    # Compute the mean squared error
    mse = torch.mean(squared_diff)

    return mse

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


noise_std = 0.0001
with open("results/toy/noise_std_{0}.pkl".format(noise_std), "rb") as file:
    torch.manual_seed(1710)
    results = pickle.load(file)
    n_mc = 500
    mse_stats = torch.zeros(20*n_mc, 5)
    entropy_stats = torch.zeros(20*n_mc, 5)
    
    def compute_mse_and_entropy(init_design):
        simulator = Toy(init_design, noise_std=noise_std)
        theta = simulator.prior.sample((1,))
        _, ys = simulator.forward(theta)
        #parameters_prime = slice_sampling((theta, ys), simulator, 10, 0.01)
        parameters_prime = adaptive_mh2((theta, ys), simulator, 100, 1)
        mse = compute_mse(parameters_prime, theta)
        entropy = estimate_entropy_and_plot(parameters_prime.view(-1))
        return mse, entropy
    
    for i, single_results in enumerate(results):
        n=0
        for result in single_results:
            mse_and_entropy_list = list(map(compute_mse_and_entropy, [result]*n_mc))
            for mse, entropy in mse_and_entropy_list:
                mse_stats[n, i] = mse
                entropy_stats[n, i] = entropy
                n+=1
            print(n)

        with open("results/toy/posterior_val_noise_std_{0}.pkl".format(noise_std), "wb") as file:
            pickle.dump([mse_stats, entropy_stats], file)

