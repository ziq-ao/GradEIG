import torch
from torch import nn
from torch import distributions
from scipy.optimize import minimize

class Gradient_Free():
     
    def __init__(self, simulator, is_proposal=None):
        
        self.simulator = simulator
        self.simulation_count = 0
        self.is_proposal = is_proposal
        self.design_history = []
       
    def learn_design(self, lr, n_out, n_in, n_steps, optim_method="Nelder-Mead"):
        objective = lambda x: self.objective(x, n_out)
        if optim_method == "Nelder-Mead":
            init_point = self.simulator.design.clone().detach().numpy()
            method="Nelder-Mead"
            bounds = self.simulator.design_lims
            options = {'maxfev': n_steps}
            res = minimize(objective, init_point, method=method, bounds=bounds, options=options)
            print(res)
            return res
                
        elif optim_method == "spsa":
            pass
            
        
    def objective(self, x, n_out):
        x = torch.Tensor(x)
        self.simulator.design = nn.Parameter(x)
        y = self.simulator.nmc_reuse(n_out, True)
        return y.detach().numpy()
        
    
if __name__ == "__main__":
    import numpy as np
    from simulator.toy import Toy
    
    init_design = torch.rand(2)
    simulator = Toy(init_design, noise_std=0.01)
    gradient_free = Gradient_Free(simulator)
    res = gradient_free.learn_design(lr=5e-3, n_out=100, n_in=1, n_steps=100)
    