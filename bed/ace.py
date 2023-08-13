import torch
from torch import distributions

class ACE():
     
    def __init__(self, simulator, is_proposal=None, sim_budget=None):
        
        self.simulator = simulator
        self.simulation_count = 0
        self.is_proposal = is_proposal
        self.design_history = []
        self.sim_count_history = []
        self.sim_budget = sim_budget
       
    def learn_design(self, lr, n_out, n_in, n_steps, sampling_method="ace"):
        optimizer = torch.optim.RMSprop(self.simulator.parameters(), lr=lr)
        optimizer_net = torch.optim.RMSprop(self.is_proposal.parameters(), lr=1e-3)
        if sampling_method == "ace":
            eig_estimator = self.nmc_ace
        
        design_lims = torch.Tensor(self.simulator.design_lims)
        for i in range(n_steps):
            optimizer.zero_grad()
            optimizer_net.zero_grad()
            loss = eig_estimator(n_out, n_in)
            loss.backward()
            optimizer.step()
            for p in self.simulator.parameters():
                p.data.clamp_(min=design_lims[:,0], max=design_lims[:,1])
            optimizer_net.step()
            self.design_history.append(self.simulator.design.clone().detach())
            self.sim_count_history.append(self.simulator.sim_count)
            if self.simulator.sim_count >= self.sim_budget:
                break
            
    def nmc_ace(self, n_out, n_in):
        return -(self.ace(n_out,n_in, True))
    
        
    def ace(self, n_out, n_in, design_requires_grad=True):
        parameters = self.simulator.prior.sample((n_out,))
        if parameters.ndim == 1:
            parameters = parameters.unsqueeze(1)
        true_ys, ys = self.simulator.forward(parameters)
        lp_outer = self.simulator.log_prob_reuse(ys, true_ys, design_requires_grad)
        parameters_prime = self.is_proposal.sample(n_in, context=ys)
        parameters_prime = parameters_prime.view(-1, self.simulator.parameter_dim)
        stacked_ys = torch.tile(ys,[1,n_in]).view(-1, self.simulator.observation_dim)
        ws = (self.simulator.prior.log_prob(parameters_prime)-self.is_proposal.log_prob(parameters_prime, context=stacked_ys)).exp()
        ws_ps = (self.simulator.prior.log_prob(parameters)-self.is_proposal.log_prob(parameters, context=ys)).exp()
        ws_matrix = torch.cat((ws.view(n_in, n_out), ws_ps.view(1,-1)), 0)
        lp_nested = self.simulator.log_prob(stacked_ys, parameters_prime, design_requires_grad)
        lp_nested = torch.cat((lp_nested.view(n_out, n_in), lp_outer.view(n_out,1)), 1)
        lp_nested = (ws_matrix*lp_nested.transpose(0,1).exp()).mean(0).log().mean()
        return lp_outer.mean() - lp_nested
    
    def nmc_reuse(self, n_out, design_requires_grad=True):
        parameters = self.simulator.prior.sample((n_out,))
        true_ys, ys = self.simulator.forward(parameters)
        lp_outer = self.simulator.log_prob_reuse(ys, true_ys, design_requires_grad).mean()
        nested_ys = torch.tile(true_ys, [1,n_out]).view(n_out**2, -1)
        lp_nested = self.simulator.log_prob_reuse(torch.tile(ys,[n_out,1]), nested_ys, design_requires_grad)
        lp_nested = lp_nested.exp().view(n_out, n_out).mean(0).log().mean()
        return lp_outer - lp_nested
