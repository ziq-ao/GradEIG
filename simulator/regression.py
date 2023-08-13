import torch
from torch import distributions
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from math import pi

class Regression(nn.Module):
    '''
    The implementation of Generalized Linear Model
    '''
    
    def __init__(self, init_design, noise_std=0.01, sims_bank=None):
        super().__init__()
        self.noise_std = noise_std
        self.prior = distributions.MultivariateNormal(loc=torch.zeros(3), covariance_matrix=torch.eye(3), validate_args=(False))
        self.design = nn.Parameter(init_design.clone())
        self.param_dim = 3
        self.obs_dim = self.design.shape[0]
        self.design_lims = [[0.0, np.inf] for i in range(self.obs_dim)]
        self.noise_std = noise_std
        self.sim_count = 0
        self.sims_bank = sims_bank
        self.acceptance_rate = []
    
    def forward(self, parameters):
        if parameters.ndim == 1:
            return self.forward(parameters.unsqueeze(0))
        design_matrix = torch.cat((torch.ones_like(self.design),self.design, self.design**2)).view(-1, self.obs_dim)
        true_obs = parameters @ design_matrix
        noised_obs = true_obs + self.noise_std*torch.randn_like(true_obs)
        self.sim_count = self.sim_count+parameters.shape[0]
        return true_obs, noised_obs
    
    def log_prob_reuse(self, noised_obs, true_obs, true_obs_requires_grad=False):
        if not true_obs_requires_grad:
            true_obs = true_obs.detach()
        noise_std = self.noise_std
        return distributions.MultivariateNormal(loc=torch.zeros(self.obs_dim),
                                                covariance_matrix=torch.eye(self.obs_dim), 
                                                validate_args=(False)).log_prob((noised_obs-true_obs)/noise_std)
    
    def log_prob(self, ys, parameters, design_requires_grad=False):
        if parameters.ndim == 1:
            return self.log_prob(ys, parameters.unsqueeze(0), design_requires_grad)
        true_obs, _ = self.forward(parameters)
        return self.log_prob_reuse(ys, true_obs, true_obs_requires_grad=design_requires_grad)
    
    def EIG(self):
        design_matrix = torch.cat((torch.ones_like(self.design),self.design, self.design**2)).view(-1, self.obs_dim)
        ent = self.obs_dim/2*(np.log(2*pi)+1)+0.5*torch.log(torch.det(design_matrix.T @ design_matrix + self.noise_std**2*torch.eye(self.obs_dim)))
        cond_ent = self.obs_dim/2*(np.log(2*pi)+1)+0.5*torch.log(torch.det(self.noise_std**2*torch.eye(self.obs_dim)))
        return ent-cond_ent
        
    def posterior_sampler(self, ys):
        pass
    
    @property
    def parameter_dim(self):
        return self.param_dim
    
    @property
    def observation_dim(self):
        return self.obs_dim