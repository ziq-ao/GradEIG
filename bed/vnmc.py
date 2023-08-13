import torch
from torch import distributions

class VNMC():
     
    def __init__(self, simulator, is_proposal=None, sim_budget=None):
        
        self.simulator = simulator
        self.simulation_count = 0
        self.is_proposal = is_proposal
        self.design_history = []
        self.sim_count_history = []
        self.sim_budget = sim_budget
       
    def learn_design(self, lr, n_out, n_in, n_steps, sampling_method="pce"):
        optimizer = torch.optim.RMSprop(self.simulator.parameters(), lr=lr)
        if sampling_method == "pce":
            eig_estimator = self.nmc_pce
        
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
            
    def nmc_pce(self, n_out, n_in):
        return -(self.pce(n_out,n_in, True))
    
        
    def pce(self, n_out, n_in, design_requires_grad=True):
        parameters = self.simulator.prior.sample((n_out,))
        true_ys, ys = self.simulator.forward(parameters)
        lp_outer = self.simulator.log_prob_reuse(ys, true_ys, design_requires_grad)
        parameters_prime = self.simulator.prior.sample((n_out*n_in,))
        lp_nested = self.simulator.log_prob(torch.tile(ys,[n_in,1]), parameters_prime, design_requires_grad)
        lp_nested = torch.cat((lp_nested, lp_outer))
        lp_nested = lp_nested.exp().view(n_in+1, n_out).mean(0).log().mean()
        return lp_outer.mean() - lp_nested
    
    def nmc_reuse(self, n_out, design_requires_grad=True):
        parameters = self.simulator.prior.sample((n_out,))
        true_ys, ys = self.simulator.forward(parameters)
        lp_outer = self.simulator.log_prob_reuse(ys, true_ys, design_requires_grad).mean()
        nested_ys = torch.tile(true_ys, [1,n_out]).view(n_out**2, -1)
        lp_nested = self.simulator.log_prob_reuse(torch.tile(ys,[n_out,1]), nested_ys, design_requires_grad)
        lp_nested = lp_nested.exp().view(n_out, n_out).mean(0).log().mean()
        return lp_outer - lp_nested
