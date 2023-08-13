import torch
from torch import distributions

"""
Codes for basic nested monte carlo with atomic proposal
"""

class APNMC():
     
    def __init__(self, simulator, proposal=None, sims_bank = None, sim_budget=None):
        
        self.simulator = simulator
        self.simulation_count = 0
        self.proposal = proposal
        self.design_history = []
        self.sim_count_history = []
        self.sims_bank = sims_bank
        self.sim_budget = sim_budget
        
    def learn_design(self, lr, n_out, n_in, n_steps, sampling_method="prior"):
        optimizer = torch.optim.RMSprop(self.simulator.parameters(), lr=lr)
        if sampling_method == "proposal":
            eig_estimator = self.proposal_sampling_estimator
        elif sampling_method == "prior":
            eig_estimator = self.prior_estimator
        elif sampling_method == "prior_reuse_sims":
            eig_estimator = self.prior_reuse_sims_estimator
        
        design_lims = torch.Tensor(self.simulator.design_lims)
        for i in range(n_steps):
            optimizer.zero_grad()
            loss = eig_estimator(n_out, n_in)
            loss.backward()
            optimizer.step()
            for p in self.simulator.parameters():
                p.data.clamp_(min=design_lims[:,0], max=design_lims[:,1])
            self.design_history.append(self.simulator.design.clone().detach())
            self.sim_count_history.append(self.simulator.sim_count)
            if self.simulator.sim_count >= self.sim_budget:
                break
            
    def proposal_sampling_estimator(self, n_out, n_in):
        return -(self.nmc_reuse_proposal(n_out, True))
        
    def prior_estimator(self, n_out, n_in):
        return -(self.nmc_reuse(n_out, True))
    
    def prior_reuse_sims_estimator(self, n_out, n_in):
        return -(self.nmc_reuse_sims(n_out, True))
    
    def nmc_reuse_proposal(self, n_out, design_requires_grad=True):
        parameters = self.proposal.sample((n_out,))
        ws = (self.simulator.prior.log_prob(parameters)-self.proposal.log_prob(parameters)).exp()
        true_ys, ys = self.simulator.forward(parameters)
        lp_outer = (ws*self.simulator.log_prob_reuse(ys, true_ys, design_requires_grad)).mean()
        nested_ys = torch.tile(true_ys, [1,n_out]).view(n_out**2, -1)
        nested_ws = torch.tile(ws, [1,n_out]).view(n_out**2, -1).view(-1)
        lp_nested = self.simulator.log_prob_reuse(torch.tile(ys,[n_out,1]), nested_ys, design_requires_grad)
        lp_nested = (ws*(nested_ws*lp_nested.exp()).view(n_out, n_out).mean(0).log()).mean()
        return lp_outer - lp_nested
    
    def nmc_reuse(self, n_out, design_requires_grad=True):
        parameters = self.simulator.prior.sample((n_out,))
        true_ys, ys = self.simulator.forward(parameters)
        lp_outer = self.simulator.log_prob_reuse(ys, true_ys, design_requires_grad).mean()
        nested_ys = torch.tile(true_ys, [1,n_out]).view(n_out**2, -1)
        lp_nested = self.simulator.log_prob_reuse(torch.tile(ys,[n_out,1]), nested_ys, design_requires_grad)
        lp_nested = lp_nested.exp().view(n_out, n_out).mean(0).log().mean()
        return lp_outer - lp_nested
    
    def nmc_reuse_sims(self, n_out, design_requires_grad=True):
        if self.sims_bank is None:
            self.sims_bank = self.simulator.prior.sample((n_out,))
        parameters = self.sims_bank
        true_ys, ys = self.simulator.forward(parameters)
        lp_outer = self.simulator.log_prob_reuse(ys, true_ys, design_requires_grad).mean()
        nested_ys = torch.tile(true_ys, [1,n_out]).view(n_out**2, -1)
        lp_nested = self.simulator.log_prob_reuse(torch.tile(ys,[n_out,1]), nested_ys, design_requires_grad)
        lp_nested = lp_nested.exp().view(n_out, n_out).mean(0).log().mean()
        return lp_outer - lp_nested 
    
    
      
class IS_Proposal():
    
    def __init__(self, covariance_matrix):
        self.covariance_matrix = covariance_matrix
    
    def sample(self, parameters):
        return distributions.MultivariateNormal(loc=parameters,covariance_matrix=self.covariance_matrix).sample()
    
    def log_prob(self, parameters_prime, parameters):
        return distributions.MultivariateNormal(loc=parameters,covariance_matrix=self.covariance_matrix).log_prob(parameters_prime)
        

       
      
    
    