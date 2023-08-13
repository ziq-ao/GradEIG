import torch
from torch import distributions
from torch import nn
import numpy as np

class Toy(nn.Module):
    
    def __init__(self, init_design, noise_std=0.01, sims_bank=None):
        super().__init__()
        self.prior = distributions.Normal(0, 1, validate_args=False)
        #self.prior = distributions.Uniform(low=0.0, high=1, validate_args=False)
        self.design = nn.Parameter(init_design.clone())
        self.design_lims = [[0.0, 1.0], [0.0, 1.0]]
        self.noise_std = noise_std
        self.sim_count = 0
        self.sims_bank = sims_bank
        self.acceptance_rate = []
    
    def forward(self, parameters):
        if parameters.ndim == 1:
            return self.forward(parameters.unsqueeze(1))
        parameters = 1*self.prior.cdf(parameters)
        true_obs = parameters**3*self.design+parameters*torch.exp(-torch.abs(0.2-self.design))
        noised_obs = true_obs + self.noise_std*torch.randn_like(true_obs)
        self.sim_count = self.sim_count+parameters.shape[0]
        return true_obs, noised_obs
    
    def log_prob_reuse(self, noised_obs, true_obs, true_obs_requires_grad=False):
        if not true_obs_requires_grad:
            true_obs = true_obs.detach()
        return distributions.Normal(0.0, self.noise_std).log_prob(noised_obs-true_obs).sum(1)
    
    def log_prob(self, ys, parameters, design_requires_grad=False):
        if parameters.ndim == 1:
            return self.log_prob(ys, parameters.unsqueeze(1), design_requires_grad)
        true_obs, _ = self.forward(parameters)
        #self.sim_count = self.sim_count+parameters.shape[0]
        return self.log_prob_reuse(ys, true_obs, true_obs_requires_grad=design_requires_grad)
    
    def nmc_resample(self, n_out, n_in, design_requires_grad=True):
        parameters = self.prior.sample((n_out,))
        true_ys, ys = self.forward(parameters)
        lp_outer = self.log_prob_reuse(ys, true_ys, design_requires_grad).mean()
        parameters_prime = self.prior.sample((n_out*n_in,))
        lp_nested = self.log_prob(torch.tile(ys,[n_in,1]), parameters_prime, design_requires_grad)
        lp_nested = lp_nested.exp().view(n_in, n_out).mean(0).log().mean()
        return lp_outer - lp_nested
    
    def nmc_reuse(self, n_out, design_requires_grad=True):
        parameters = self.prior.sample((n_out,))
        true_ys, ys = self.forward(parameters)
        lp_outer = self.log_prob_reuse(ys, true_ys, design_requires_grad).mean()
        nested_ys = torch.tile(true_ys, [1,n_out]).view(n_out**2, -1)
        lp_nested = self.log_prob_reuse(torch.tile(ys,[n_out,1]), nested_ys, design_requires_grad)
        lp_nested = lp_nested.exp().view(n_out, n_out).mean(0).log().mean()
        return lp_outer - lp_nested
    
    def nmc_reuse_sims(self, n_out, design_requires_grad=True):
        if self.sims_bank is None:
            self.sims_bank = self.prior.sample((n_out,))
        parameters = self.sims_bank
        true_ys, ys = self.forward(parameters)
        lp_outer = self.log_prob_reuse(ys, true_ys, design_requires_grad).mean()
        nested_ys = torch.tile(true_ys, [1,n_out]).view(n_out**2, -1)
        lp_nested = self.log_prob_reuse(torch.tile(ys,[n_out,1]), nested_ys, design_requires_grad)
        lp_nested = lp_nested.exp().view(n_out, n_out).mean(0).log().mean()
        return lp_outer - lp_nested
    
    def pce(self, n_out, n_in, design_requires_grad=True):
        parameters = self.prior.sample((n_out,))
        true_ys, ys = self.forward(parameters)
        lp_outer = self.log_prob_reuse(ys, true_ys, design_requires_grad)
        parameters_prime = self.prior.sample((n_out*n_in,))
        lp_nested = self.log_prob(torch.tile(ys,[n_in,1]), parameters_prime, design_requires_grad)
        lp_nested = torch.cat((lp_nested, lp_outer))
        lp_nested = lp_nested.exp().view(n_in+1, n_out).mean(0).log().mean()
        return lp_outer.mean() - lp_nested
    
    @property
    def parameter_dim(self):
        return 1
    
    @property
    def observation_dim(self):
        return 2
    
if __name__ == "__main__":
    n_trials = 1000
    list1 = []
    list2 = []
    for i in range(n_trials):
        sim_model = Toy(torch.Tensor([0.3,0.7]))
        nmc_resample_value = sim_model.pce(10, 10, True)
        grad = torch.autograd.grad(nmc_resample_value, sim_model.design)[0]
        list1.append(grad)
        #print(nmc_resample_value, grad)
        nmc_reuse_value = sim_model.nmc_reuse(100, True)
        grad = torch.autograd.grad(nmc_reuse_value, sim_model.design)[0]
        list2.append(grad)
        print(nmc_reuse_value, grad)
        
    list1 = torch.cat(list1).view(-1,2)
    list2 = torch.cat(list2).view(-1,2)
    print(list1.mean(0), list2.mean(0))
    print(list1.std(0), list2.std(0))
    