import torch
from torch import distributions
import multiprocessing as mp
from mcmc import Slice, SliceSampler, SVGDSampler
import utils
from tqdm import tqdm
import numpy as np

"""
Codes for unbiased gradient estimation with MCMC
"""

class MCMC_Gradient():
     
    def __init__(self, simulator, proposal=None, sims_bank = None, mcmc_param=0.5, sim_budget=None, sampling_method="slice"):
        
        self.simulator = simulator
        self.simulation_count = 0
        self.proposal = proposal
        self.design_history = []
        self.sims_bank = sims_bank
        self.n_in = None
        self.mcmc_param = mcmc_param
        self.sim_count_history = []
        self.sim_budget = sim_budget
        self.sampling_method = sampling_method
       
    def learn_design(self, lr, n_out, n_in, n_steps):
        optimizer = torch.optim.RMSprop(self.simulator.parameters(), lr=lr)
        self.n_in = n_in

        gradient_estimator = self.prior_gradient_estimator
        
        design_lims = torch.Tensor(self.simulator.design_lims)
        for i in range(n_steps):
            optimizer.zero_grad()
            param_grad = gradient_estimator(n_out, n_in)
            self.simulator.design.grad = param_grad
            optimizer.step()
            for p in self.simulator.parameters():
                p.data.clamp_(min=design_lims[:,0], max=design_lims[:,1])
            self.design_history.append(self.simulator.design.clone().detach())
            self.sim_count_history.append(self.simulator.sim_count)
            if self.simulator.sim_count >= self.sim_budget:
                break
            
    def proposal_sampling_estimator(self, n_out, n_in):
        pass
        
    def prior_gradient_estimator(self, n_out, n_in):
        return -(self.get_mcmc_gradient(n_out, n_in))
    
    def get_mcmc_gradient(self, n_out, n_in, use_multiprocessing=False):
        self.n_in = n_in
        parameters = self.simulator.prior.sample((n_out,))
        true_ys, ys = self.simulator.forward(parameters)
        #func = self.gradient_est
        if self.sampling_method == "slice":
            func = lambda pair: slice_sampling(pair, self.simulator, self.n_in, self.mcmc_param)
        elif self.sampling_method == "mala":
            func = lambda pair: mala(pair, self.simulator, self.n_in, self.mcmc_param)
        elif self.sampling_method == "mh":
            func = lambda pair: mh(pair, self.simulator, self.n_in, self.mcmc_param)
        elif self.sampling_method == "adaptive_mh":
            func = lambda pair: adaptive_mh(pair, self.simulator, self.n_in, self.mcmc_param)
        elif self.sampling_method == "adaptive_mh2":
            func = lambda pair: adaptive_mh2(pair, self.simulator, self.n_in, self.mcmc_param)
   
            
        design_requires_grad = True
        if use_multiprocessing:
            pool = mp.Pool(10)
            parameters_prime_list = pool.map(func, list(zip(parameters.detach(),ys.detach())))
        else:
            parameters_prime_list = map(func, list(zip(parameters.detach(),ys.detach())))
        parameters_prime = torch.cat(list(parameters_prime_list))
        log_ratio = self.simulator.log_prob_reuse(ys, true_ys, design_requires_grad).mean()-self.simulator.log_prob(torch.tile(ys, [1, self.n_in]).view(-1, self.simulator.observation_dim), parameters_prime, design_requires_grad).mean() 
        self.simulator.sim_count -= n_out*n_in
        # The forward modes for parameters_prime have already been computed in MCMC
        # so we minus the simulation costs here. Just for the convenience of coding
        return torch.autograd.grad(log_ratio, self.simulator.design)[0].detach()
    
        
    def gradient_est(self, pair):
        parameters, ys = pair
        target_log_prob = (
            lambda parameters: self.simulator.log_prob(ys, torch.Tensor(parameters))+
            self.simulator.prior.log_prob(torch.Tensor(parameters).view(1,-1))
            )
        # create sampler
        posterior_sampler = SliceSampler(
            utils.tensor2numpy(parameters).reshape(-1),
            lp_f=target_log_prob,
            thin=3,
            #max_width=0.1,
        )
        posterior_sampler.width=torch.Tensor([self.mcmc_param]*self.simulator.parameter_dim)
        parameters_prime = posterior_sampler.gen(self.n_in)
        parameters_prime = torch.Tensor(parameters_prime)
        return parameters_prime
        #log_ratio = self.simulator.log_prob(ys, parameters, False) - self.simulator.log_prob(torch.tile(ys, [self.n_in,1]), parameters_prime, False).mean() 
        #return torch.autograd.grad(log_ratio, self.simulator.design)[0].detach()
    
    
    def nmc_reuse(self, n_out, design_requires_grad=True):
        parameters = self.simulator.prior.sample((n_out,))
        true_ys, ys = self.simulator.forward(parameters)
        lp_outer = self.simulator.log_prob_reuse(ys, true_ys, design_requires_grad).mean()
        nested_ys = torch.tile(true_ys, [1,n_out]).view(n_out**2, -1)
        lp_nested = self.simulator.log_prob_reuse(torch.tile(ys,[n_out,1]), nested_ys, design_requires_grad)
        lp_nested = lp_nested.exp().view(n_out, n_out).mean(0).log().mean()
        return lp_outer - lp_nested
    

def slice_sampling(pair, simulator, n_in, mcmc_param):
    parameters, ys = pair
    init_num_sim = simulator.sim_count
    target_log_prob = (
        lambda parameters: simulator.log_prob(ys, torch.Tensor(parameters))+
        simulator.prior.log_prob(torch.Tensor(parameters).view(1,-1))
        )
    # create sampler
    posterior_sampler = SliceSampler(
        utils.tensor2numpy(parameters).reshape(-1),
        lp_f=target_log_prob,
        thin=2,
        #max_width=0.1,
    )
    posterior_sampler.width=torch.Tensor([mcmc_param]*simulator.parameter_dim)
    parameters_prime = posterior_sampler.gen(n_in)
    parameters_prime = torch.Tensor(parameters_prime)
    final_num_sim = simulator.sim_count
    simulator.acceptance_rate.append(final_num_sim-init_num_sim)
    return parameters_prime



def mala(pair, simulator, n_in, step):
    parameters, ys = pair
    parameters = parameters.view(1,-1)
    ys = ys.view(1,-1)
    
    def potential(z):
        target_log_prob = (simulator.log_prob(ys, torch.Tensor(z), True)+
            simulator.prior.log_prob(torch.Tensor(z).view(1,-1))
            )
        return -target_log_prob
    
    def log_Q(grad_z, z_prime, z, step):
        return -(torch.norm(z_prime - z + step * grad_z, p=2, dim=1) ** 2) / (4 * step)
    
    thin = 50
    burn_in = thin
    Z0 = parameters
    #Z0 = simulator.prior.sample((1,))
    Zi = Z0
    samples = []
    acceptance_num = 0 
    pbar = tqdm(range(n_in*thin), disable=(True))
    Zi.requires_grad_()
    potential_Zi = potential(Zi).mean()
    grad_Zi = torch.autograd.grad(potential_Zi, Zi)[0]
    # The potential of Z0 has already been computed before, so no sim number is 
    # needed to be counted here. Just for the convenience of coding
    for i in pbar:
        prop_Zi = Zi.detach() - step * grad_Zi + np.sqrt(2 * step) * torch.randn(1, simulator.parameter_dim)
        prop_Zi.requires_grad_()
        potential_prop_Zi = potential(prop_Zi)
        grad_prop_Zi = torch.autograd.grad(potential_prop_Zi, prop_Zi)[0]
        simulator.sim_count+=1
        # we regard the cost of gradient computation is similar to the forward mode
        # here.
        log_ratio = -potential_prop_Zi.mean() + potential_Zi.mean() +\
                    log_Q(grad_prop_Zi, Zi, prop_Zi, step) - log_Q(grad_Zi, prop_Zi, Zi, step)
        if torch.rand(1) < torch.exp(log_ratio):
            Zi = prop_Zi
            potential_Zi = potential_prop_Zi
            grad_Zi = grad_prop_Zi
            acceptance_num += 1
        samples.append(Zi.detach())
    simulator.acceptance_rate.append(acceptance_num/(n_in*thin))
    return torch.cat(samples, 0)[burn_in-1::thin]



def mh(pair, simulator, n_in, step):
    parameters, ys = pair
    parameters = parameters.view(1,-1)
    ys = ys.view(1,-1)
    
    def potential(z):
        target_log_prob = (simulator.log_prob(ys, torch.Tensor(z))+
            simulator.prior.log_prob(torch.Tensor(z).view(1,-1))
            )
        return -target_log_prob
    
    def log_Q(z_prime, z, step):
        return -(torch.norm(z_prime - z, p=2, dim=1) ** 2) / (2 * step)
  
    thin = 99
    burn_in = thin
    Z0 = parameters+0.5*np.sqrt(step)*torch.randn_like(parameters)
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
        prop_Zi = Zi.detach() + np.sqrt(step) * torch.randn(1, simulator.parameter_dim)
        prop_Zi.requires_grad_()
        potential_prop_Zi = potential(prop_Zi)
        log_ratio = -potential_prop_Zi.mean() + potential_Zi.mean() +\
                    log_Q(Zi, prop_Zi, step) - log_Q(prop_Zi, Zi, step)
        if torch.rand(1) < torch.exp(log_ratio):
            Zi = prop_Zi
            potential_Zi = potential_prop_Zi
            acceptance_num += 1
        samples.append(Zi.detach())
    simulator.acceptance_rate.append(acceptance_num/(n_in*thin))
    return torch.cat(samples, 0)[burn_in-1::thin]


def adaptive_mh(pair, simulator, n_in, step):
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
    try:
        proposal = distributions.MultivariateNormal(
            loc=torch.zeros_like(init_state),
            covariance_matrix=torch.inverse(hessian_matrix))
        #print(torch.inverse(hessian_matrix))
    except:
        print("failed")
        print(torch.inverse(hessian_matrix))
        parameters = simulator.prior.sample((1,))
        _, ys = simulator.forward(parameters)
        return adaptive_mh([parameters, ys], simulator, n_in, step)
        
    def log_Q(z_prime, z, step):
        return proposal.log_prob((z_prime-z)/step)
    
    thin = 95
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
    print(ar)
    return torch.cat(samples, 0)[burn_in-1::thin]


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
        proposal = distributions.MultivariateNormal(
            loc=torch.zeros_like(init_state),
            covariance_matrix=simulator.covariance_matrix)
    #print(proposal.covariance_matrix)
        
    def log_Q(z_prime, z, step):
        return proposal.log_prob((z_prime-z)/step)
    
    thin = 95
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
    print(ar)
    return torch.cat(samples, 0)[burn_in-1::thin]


from torch.autograd.functional import hessian
def compute_hessian(loss, variables):
    # Compute Hessian matrix
    hessian_matrix = hessian(loss, variables, create_graph=False)

    return hessian_matrix