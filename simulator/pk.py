import torch
from torch import distributions
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class PK(nn.Module):
    '''
    The implementation of Pharmacokinetic Model
    '''
    
    def __init__(self, init_design, sims_bank=None):
        super().__init__()
        self.prior = distributions.MultivariateNormal(loc=torch.zeros(3), covariance_matrix=torch.eye(3), validate_args=(False))
        self.design = nn.Parameter(init_design.clone())
        self.obs_dim = self.design.shape[0]
        self.design_lims = [[0.0, np.inf] for i in range(self.obs_dim)]
        self.inner_noise_std = 0.1
        self.noise_std = np.sqrt(0.1)
        self.sim_count = 0
        self.sims_bank = sims_bank
        self.Dv = 400
        self.acceptance_rate = []
        self.covariance_matrix = None
        
    def forward(self, parameters):
        if parameters.ndim == 1:
            return self.forward(parameters.unsqueeze(0))
        parameters = torch.Tensor([np.log(1), np.log(0.1), np.log(20)]) +torch.sqrt(0.05*torch.ones(3))*parameters
        parameters = parameters.exp()
        ka, ke, V = parameters[:,[0]], parameters[:,[1]], parameters[:,[2]]
        true_obs = self.Dv/V*ka/(ka-ke)*(torch.exp(-ke*self.design)-torch.exp(-ka*self.design))
        noised_obs = true_obs*(1+self.inner_noise_std*torch.randn_like(true_obs)) + self.noise_std*torch.randn_like(true_obs)
        self.sim_count = self.sim_count+parameters.shape[0]
        return true_obs, noised_obs
    
    def log_prob_reuse(self, noised_obs, true_obs, true_obs_requires_grad=False):
        if not true_obs_requires_grad:
            true_obs = true_obs.detach()
        noise_std = torch.sqrt((true_obs*self.inner_noise_std)**2+self.noise_std**2)
        return distributions.MultivariateNormal(loc=torch.zeros(self.obs_dim),
                                                covariance_matrix=torch.eye(self.obs_dim), 
                                                validate_args=(False)).log_prob((noised_obs-true_obs)/noise_std)
    
    def log_prob(self, ys, parameters, design_requires_grad=False):
        if parameters.ndim == 1:
            return self.log_prob(ys, parameters.unsqueeze(0), design_requires_grad)
        true_obs, _ = self.forward(parameters)
        return self.log_prob_reuse(ys, true_obs, true_obs_requires_grad=design_requires_grad)
    
    @property
    def parameter_dim(self):
        return 3
    
    @property
    def observation_dim(self):
        return self.obs_dim
    
    

if __name__ == "__main__":
    from utils.plot import plot_hist_marginals
    init_design = 5*torch.rand(1)
    sim_model = PK(init_design)
    parameters = sim_model.prior.sample((10000,))
    true_obs, noised_obs = sim_model.forward(parameters)
    lp = sim_model.log_prob_reuse(noised_obs, true_obs)
    lp = sim_model.log_prob(noised_obs, parameters)
    plt.hist(lp.detach().numpy(),50)
    plot_hist_marginals(noised_obs.detach().numpy())
    
    plot_hist_marginals(torch.cat((parameters,noised_obs.detach()),1))
