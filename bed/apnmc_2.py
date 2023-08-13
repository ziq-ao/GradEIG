import torch
from torch import distributions
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch import nn, optim

"""
Codes for basic nested monte carlo with double atomic proposals
"""

class APNMC_2():
     
    def __init__(self, simulator, proposal=None, sims_bank = None):
        
        self.simulator = simulator
        self.simulation_count = 0
        self.proposal = proposal
        self.design_history = []
        self.sims_bank = sims_bank
       
    def learn_design(self, lr, n_out, n_in, n_steps, sampling_method="proposal"):
        optimizer = torch.optim.RMSprop(self.simulator.parameters(), lr=lr)
        if sampling_method == "proposal":
            eig_estimator = self.proposal_sampling_estimator
        elif sampling_method == "adaptive_proposal":
            eig_estimator = self.adaptive_proposal_sampling_estimator
        elif sampling_method == "proposal_reuse_sims":
            eig_estimator = self.proposal_reuse_sims_estimator
        elif sampling_method == "adaptive_proposal_reuse_sims":
            eig_estimator = self.adaptive_proposal_reuse_sims_estimator    
        
        design_lims = torch.Tensor(self.simulator.design_lims)
        for i in range(n_steps):
            optimizer.zero_grad()
            _, loss = eig_estimator(n_out, n_in)
            loss.backward()
            optimizer.step()
            for p in self.simulator.parameters():
                p.data.clamp_(min=design_lims[:,0], max=design_lims[:,1])
            self.design_history.append(self.simulator.design.clone().detach())
    
    def pre_train_proposal(self, lr=5e-3, batch_size=100, n_steps=10):
        parameters = self.simulator.prior.sample((1000,))
        _, ys = self.simulator.forward(parameters)
        joint = torch.cat((parameters, ys), 1).detach()
        dataset = data.TensorDataset(
            joint
        )

        # Create train and validation loaders using a subset sampler.
        train_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
            #drop_last=True,
        )
        
        optimizer = optim.Adam(self.proposal.parameters(), lr=lr)
        epochs = 0
        while epochs<n_steps:
            # Train for a single epoch.
            self.proposal.train()
            for batch in train_loader:
                optimizer.zero_grad()
                device = "cpu"
                inputs = batch[0].to(device).requires_grad_(True)
                log_prob = self.proposal.log_prob(inputs)
                loss = -torch.mean(log_prob)
                loss.backward()
                clip_grad_norm_(self.proposal.parameters(), max_norm=5.0)
                optimizer.step()
            epochs += 1
            print(loss)
        
    def learn_proposal(self, lr, n_out, n_in, n_steps, sampling_method="proposal"):
        optimizer = torch.optim.RMSprop(self.proposal.parameters(), lr=lr)
        if sampling_method == "proposal":
            eig_estimator = self.proposal_sampling_estimator
        if sampling_method == "joint_proposal":
            eig_estimator = self.joint_proposal_sampling_estimator
        for i in range(n_steps):
            optimizer.zero_grad()
            mc_ests, _ = eig_estimator(n_out, n_in)
            loss = torch.var(mc_ests)
            if i % 100 == 1: 
                print(loss, mc_ests.mean())
            loss.backward()
            clip_grad_norm_(self.proposal.parameters(), max_norm=3.0)
            optimizer.step()
    
    def proposal_sampling_estimator(self, n_out, n_in):
        mc_ests, mean_est = self.nmc_reuse_proposal(n_out, n_in, True)
        return -mc_ests, -mean_est
    
    def joint_proposal_sampling_estimator(self, n_out, n_in):
        mc_ests, mean_est = self.nmc_reuse_joint_proposal(n_out, n_in, True)
        return -mc_ests, -mean_est
    
    def nmc_reuse_proposal(self, n_out, n_in, design_requires_grad=True):
        try:
            parameters = self.proposal.sample((n_out,))
        except:
            parameters = self.proposal.sample(n_out)
        ws = (self.simulator.prior.log_prob(parameters)-self.proposal.log_prob(parameters)).exp()
        true_ys, ys = self.simulator.forward(parameters)
        lp_outer = self.simulator.log_prob_reuse(ys, true_ys, design_requires_grad)
        
        parameters_prime = self.simulator.prior.sample((n_in,))
        parameters_prime = torch.tile(parameters_prime, [1,n_out]).view(n_out*n_in, -1)
        lp_nested = self.simulator.log_prob(torch.tile(ys,[n_in,1]), parameters_prime, design_requires_grad)
        lp_nested = torch.cat((lp_nested, lp_outer))
        lp_nested = (lp_nested.exp()).view(n_in+1, n_out).mean(0).log()
        
        mc_ests = ws*(lp_outer-lp_nested)
        return mc_ests, (mc_ests).mean()
    
    def nmc_reuse_joint_proposal(self, n_out, n_in, design_requires_grad=True):
        try:
            joint = self.proposal.sample((n_out,))
        except:
            joint = self.proposal.sample(n_out)
        parameters, ys = joint[:,0:self.simulator.parameter_dim], joint[:,self.simulator.parameter_dim:]
        lp_outer = self.simulator.log_prob(ys, parameters, design_requires_grad)
        ws = (self.simulator.prior.log_prob(parameters)+lp_outer-self.proposal.log_prob(joint)).exp()
        
        parameters_prime = self.simulator.prior.sample((n_in,))
        parameters_prime = torch.tile(parameters_prime, [1,n_out]).view(n_out*n_in, -1)
        lp_nested = self.simulator.log_prob(torch.tile(ys,[n_in,1]), parameters_prime, design_requires_grad)
        lp_nested = torch.cat((lp_nested, lp_outer))
        lp_nested = (lp_nested.exp()).view(n_in+1, n_out).mean(0).log()
        
        mc_ests = ws*(lp_outer-lp_nested)
        return mc_ests, (mc_ests).mean()
    
    def nmc_reuse(self, n_out, design_requires_grad=True):
        parameters = self.simulator.prior.sample((n_out,))
        true_ys, ys = self.simulator.forward(parameters)
        lp_outer = self.simulator.log_prob_reuse(ys, true_ys, design_requires_grad)
        nested_ys = torch.tile(true_ys, [1,n_out]).view(n_out**2, -1)
        lp_nested = self.simulator.log_prob_reuse(torch.tile(ys,[n_out,1]), nested_ys, design_requires_grad)
        lp_nested = lp_nested.exp().view(n_out, n_out).mean(0).log()
        mc_ests = lp_outer - lp_nested
        return mc_ests, mc_ests.mean()
    
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
        

       
      
    
    