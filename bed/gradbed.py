import torch
from torch import distributions
import numpy as np

class GradBED():
     
    def __init__(self, simulator, net=None, sim_budget=None):
        
        self.simulator = simulator
        self.simulation_count = 0
        self.net = net
        self.design_history = []
        self.sim_count_history = []
        self.sim_budget = sim_budget
       
    def learn_design(self, lr, batch_size, n_steps, bound="nwj", lr_net=1e-3):
        optimizer = torch.optim.RMSprop(self.simulator.parameters(), lr=lr)
        optimizer_net = torch.optim.RMSprop(self.net.parameters(), lr=lr_net)
        if bound == "nwj":
            eig_estimator = self.nwj_estimator
        
        design_lims = torch.Tensor(self.simulator.design_lims)
        for i in range(n_steps):
            optimizer.zero_grad()
            optimizer_net.zero_grad()
            loss = eig_estimator(batch_size)
            loss.backward()
            optimizer.step()
            for p in self.simulator.parameters():
                p.data.clamp_(min=design_lims[:,0], max=design_lims[:,1])
            optimizer_net.step()
            self.design_history.append(self.simulator.design.clone().detach())
            self.sim_count_history.append(self.simulator.sim_count)
            if self.simulator.sim_count >= self.sim_budget:
                break
    
    def nwj_estimator(self, batch_size):
        parameters = self.simulator.prior.sample((batch_size,))
        if parameters.ndim == 1:
            parameters = parameters.unsqueeze(1)
        _, ys = self.simulator.forward(parameters)
        return self.nwj_loss(parameters, ys, self.net)
    
    def nwj_loss(self, x_sample, y_sample, model):
    
        # Shuffle y-data for the second expectation
        idxs = np.random.choice(
            range(len(y_sample)), size=len(y_sample), replace=False)
        # We need y_shuffle attached to the design d
        y_shuffle = y_sample[idxs]
    
        # Get predictions from network
        pred_joint = model(x_sample, y_sample)
        pred_marginals = model(x_sample, y_shuffle)
    
        # Compute the MINE-f (or NWJ) lower bound
        Z = torch.tensor(np.exp(1), dtype=torch.float)
        mi_ma = torch.mean(pred_joint) - torch.mean(
            torch.exp(pred_marginals) / Z + torch.log(Z) - 1)
    
        # we want to maximize the lower bound; PyTorch minimizes
        loss = - mi_ma
    
        return loss
         
    def nmc_reuse(self, n_out, design_requires_grad=True):
        parameters = self.simulator.prior.sample((n_out,))
        true_ys, ys = self.simulator.forward(parameters)
        lp_outer = self.simulator.log_prob_reuse(ys, true_ys, design_requires_grad).mean()
        nested_ys = torch.tile(true_ys, [1,n_out]).view(n_out**2, -1)
        lp_nested = self.simulator.log_prob_reuse(torch.tile(ys,[n_out,1]), nested_ys, design_requires_grad)
        lp_nested = lp_nested.exp().view(n_out, n_out).mean(0).log().mean()
        return lp_outer - lp_nested
