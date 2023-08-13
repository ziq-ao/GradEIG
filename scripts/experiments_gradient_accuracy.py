from bed.mcmc_gradient import *
from bed.apnmc import APNMC
from bed.vnmc import VNMC
from simulator.regression import Regression
import torch
from utils.plot import plot_hist_marginals
import torch.multiprocessing as mp
import pickle


design_dim = 3

global ground_truth_gradient
def ground_truth_gradient(args):
    design, noise_std = args
    simulator = Regression(torch.Tensor(design), noise_std=noise_std)
    eig = simulator.EIG()
    gradient = torch.autograd.grad(simulator.EIG(), simulator.design)[0]
    return eig.detach(), gradient

global apnmc_gradient
def apnmc_gradient(args):
    design, noise_std, n_trials = args
    simulator = Regression(torch.Tensor(design), noise_std=noise_std)
    apnmc = APNMC(simulator)
    m_grad = []
    for i in range(n_trials):
        m_grad.append(torch.autograd.grad(apnmc.nmc_reuse(100), simulator.design)[0])
    return torch.cat(m_grad).view(-1, design_dim).mean(0)

global vnmc_gradient
def vnmc_gradient(args):
    design, noise_std, n_trials = args
    simulator = Regression(torch.Tensor(design), noise_std=noise_std)
    vnmc = VNMC(simulator)
    m_grad = []
    for i in range(n_trials):
        m_grad.append(torch.autograd.grad(vnmc.pce(100, 100), simulator.design)[0])
    return torch.cat(m_grad).view(-1, design_dim).mean(0)

global mcmc_gradient
def mcmc_gradient(args):
    design, noise_std, n_trials = args
    simulator = Regression(torch.Tensor(design), noise_std=noise_std)
    mcgrad = MCMC_Gradient(simulator, mcmc_param=1.0)
    mcgrad.sampling_method = "adaptive_mh"
    m_grad = []
    for i in range(n_trials):
        m_grad.append(mcgrad.get_mcmc_gradient(100, 1, use_multiprocessing=(False)))
    return torch.cat(m_grad).view(-1, design_dim).mean(0)

def expr(design, noise_std, n_trials):
    eig, gradient = ground_truth_gradient(design, noise_std)
    apnmc_est = apnmc_gradient(design, noise_std, n_trials)
    mcmc_est = mcmc_gradient(design, noise_std, n_trials)
    return eig, gradient, apnmc_est, mcmc_est
    

if __name__ == '__main__':
    torch.manual_seed(666)
    use_multiprocess = False
    num_process = mp.cpu_count()
    if num_process > 32:
        num_process = 32
        
    n_trials = 100
    
    noise_std_list = [2,1/2,1/8]
    num_designs = 20
    design_list = list(2*torch.rand(num_designs,design_dim)-1) 
    for noise_std in noise_std_list:
        
        #func_gt = lambda design: ground_truth_gradient(design, noise_std)
        #func_apnmc = lambda design: apnmc_gradient(design, noise_std, n_trials)
        #func_mcmc = lambda design: mcmc_gradient(design, noise_std, n_trials)
       
        if use_multiprocess:
            pool = mp.Pool(num_process-1)
            gt_results = pool.map(ground_truth_gradient, zip(design_list, [noise_std]*num_designs))
            apnmc_results = pool.map(apnmc_gradient, zip(design_list, [noise_std]*num_designs, [n_trials]*num_designs))
            mcmc_results = pool.map(mcmc_gradient, zip(design_list, [noise_std]*num_designs, [n_trials]*num_designs))
        else:
            gt_results = list(map(ground_truth_gradient, zip(design_list, [noise_std]*num_designs)))
            apnmc_results = list(map(apnmc_gradient, zip(design_list, [noise_std]*num_designs, [n_trials]*num_designs)))
            vnmc_results = list(map(vnmc_gradient, zip(design_list, [noise_std]*num_designs, [n_trials]*num_designs)))
            mcmc_results = list(map(mcmc_gradient, zip(design_list, [noise_std]*num_designs, [n_trials]*num_designs)))
        with open("results/gradient_accuracy/noise_std_{0}.pkl".format(noise_std), "wb") as file:
            pickle.dump([gt_results, apnmc_results, vnmc_results, mcmc_results], file)
        