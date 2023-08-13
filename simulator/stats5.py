import torch
import torchdiffeq
import matplotlib.pyplot as plt
import numpy
from scipy import interpolate
from utils.interpolate import linear_interpolation
from utils.plot import plot_hist_marginals

from torch import distributions
from torch import nn
import numpy as np
import matplotlib.pyplot as plt



class STATS5(nn.Module):
    '''
    The implementation of the STAT5 (signal transducer and activator of transcription 5)
    signalling cascade Model
    '''
    
    def __init__(self, init_design, noise_std=0.01, sims_bank=None):
        super().__init__()
        self.prior = distributions.MultivariateNormal(loc=torch.zeros(3), covariance_matrix=torch.eye(3), validate_args=(False))
        self.design = nn.Parameter(init_design.clone())
        self.obs_dim = self.design.shape[0]
        self.design_lims = [[0.0, 60.0] for i in range(self.obs_dim)]
        self.noise_std = noise_std
        self.sim_count = 0
        self.acceptance_rate = []
        self.covariance_matrix = None
        
    def forward(self, parameters):
        if parameters.ndim == 1:
            return self.forward(parameters.unsqueeze(0))
        #parameters = torch.Tensor([1.9, 0.094, 4.6]) + distributions.Normal(0,1).cdf(parameters)*torch.Tensor([0.44,0.03,1.2])
        #parameters = torch.Tensor([1.68, 0.079, 4.0]) + distributions.Normal(0,1).cdf(parameters)*torch.Tensor([0.88,0.06,2.4])
        parameters = torch.Tensor([0.5, 0.05, 4.0]) + distributions.Normal(0,1).cdf(parameters)*torch.Tensor([2.5,0.15,6])
        
        t0 = torch.linspace(0, 60, 70).requires_grad_(True)
        #t0 = torch.cat((torch.linspace(0, 10, 20),torch.linspace(10.1, 60, 50)))
        x0 = torch.Tensor([3.71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        def measurement(ps):    
            ode = lambda t, x: self.ode(t, x, ps)
            solution = torchdiffeq.odeint(ode, x0, t0,
                                          method='rk4',
                                          )
            solution = solution.T
            y1 = 0.33*solution[[1,2]].sum(0)
            y2 = 0.26*solution[[0,1,2]].sum(0)
            y1 = linear_interpolation(self.design, t0, y1)
            y2 = linear_interpolation(self.design, t0, y2)
            return torch.cat((y1, y2))
        true_obs = torch.cat(list(map(measurement, parameters))).view(parameters.shape[0], -1)
        noised_obs = true_obs + self.noise_std*torch.randn_like(true_obs)
        self.sim_count = self.sim_count+parameters.shape[0]
        return true_obs, noised_obs
    
    def ode(self, t, x, ps):
        k1, k2, tau = ps
        x1, x2, x3, x4, q1, q2, q3, q4, q5, q6, q7, out = x
        #EpoRA = 1
        x = self.default_times
        y = self.EpoRA_data
        EpoRA = linear_interpolation(t, x, y)
        
        dx1_dt = -k1*x1*EpoRA + k2*out
        dx2_dt = -x2**2 + k1*x1*EpoRA
        dx3_dt = -k2*x3 + x2**2
        dx4_dt = -k2*out + k2*x3
        dq1_dt = 8/tau*(x3-q1)
        dq2_dt = 8/tau*(q1-q2)
        dq3_dt = 8/tau*(q2-q3)
        dq4_dt = 8/tau*(q3-q4)
        dq5_dt = 8/tau*(q4-q5)
        dq6_dt = 8/tau*(q5-q6)
        dq7_dt = 8/tau*(q6-q7)
        dout_dt = 8/tau*(q7-out)
        return torch.cat((dx1_dt.view(1),
                          dx2_dt.view(1),
                          dx3_dt.view(1),
                          dx4_dt.view(1),
                          dq1_dt.view(1),
                          dq2_dt.view(1),
                          dq3_dt.view(1),
                          dq4_dt.view(1),
                          dq5_dt.view(1),
                          dq6_dt.view(1),
                          dq7_dt.view(1),
                          dout_dt.view(1),
                          ))
        
    
    def log_prob_reuse(self, noised_obs, true_obs, true_obs_requires_grad=False):
        if not true_obs_requires_grad:
            true_obs = true_obs.detach()
        noise_std = self.noise_std
        return distributions.MultivariateNormal(loc=torch.zeros(self.observation_dim),
                                                covariance_matrix=torch.eye(self.observation_dim), 
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
        return self.obs_dim*2
    
    @property
    def default_times(self):
        return torch.Tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60])
    
    @property
    def EpoRA_data(self):
        return torch.Tensor([0.01713, 0.145, 0.2442, 0.7659, 1, 0.8605,
         0.7829, 0.5705, 0.6217, 0.331, 0.3388, 0.3116, 0.05062, 0.02504,
         0.01163, 0.01163])


def plot_trajectories():
    init_design = torch.linspace(0,60,20)
    simulator = STATS5(init_design)
    parameters = simulator.prior.sample((100,)).requires_grad_(True)
    true_ys,_ = simulator.forward(parameters)
    for s in true_ys.detach():
        plt.plot(init_design, s[:20], 'blue')
        plt.plot(init_design, s[20:], 'red')
    

def test_stats5():
    simulator = STATS5(torch.Tensor([5,10,20,30,40]))
    parameters = simulator.prior.sample((100,)).requires_grad_(True)
    true_ys, ys = simulator.forward(parameters)
    print(torch.autograd.grad(simulator.log_prob(ys, parameters, True).sum(), simulator.design))
    #print(torch.autograd.grad(ys[-1,-1], parameters))
    plot_hist_marginals(ys.detach())
    

def test_example_torchdiffeq():
    # Define ODE function as an nn.Module
    class StiffODE(nn.Module):
        
        def __init__(self):
            super().__init__()
            self.params = nn.Parameter(torch.Tensor([0.1, 2]))
        
        def forward(self, t, y):
            dy1_dt = -self.params[0] * y[0] + self.params[1] * y[1]
            dy2_dt = -self.params[1] * y[0] - self.params[0] * y[1]
            return torch.cat((dy1_dt.view(1), dy2_dt.view(1)))
    
    # Define time interval
    t0 = 0.0
    t_max = 5.0
    t = torch.linspace(t0, t_max, 20).requires_grad_(True)
    #t = torch.cat((torch.linspace(t0, t0+0.005, 20), torch.linspace(t_max-0.005, t_max, 20))).requires_grad_(True)
    
    
    # Define initial conditions
    y0 = torch.tensor([1.0, 0.0]).requires_grad_(True)
    
    # Create an instance of the ODE function
    stiff_ode = StiffODE()
    
    # Use torchdiffeq to solve the ODE with the adjoint method
    solution = torchdiffeq.odeint(stiff_ode.forward, y0, t,
                                          #method='dopri5',
                                          #method='rk4',
                                          #method='euler',
                                          #method='midpoint',
                                          #method='implicit_adams',
                                          #tol=1e-6, 
                                          #rtol=1e-6,
                                          )
    
    # Extract the solution values and convert them to NumPy arrays
    y1_values = solution[:, 0].detach().numpy()
    y2_values = solution[:, 1].detach().numpy()
    
    #print(torch.autograd.grad(solution[-1,1], stiff_ode.parameters()))
    print(torch.autograd.grad(solution[-1,1], t))
    # Plot the solution
    plt.plot(t.detach(), y1_values, label='y1(t)')
    plt.plot(t.detach(), y2_values, label='y2(t)')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('Solution of 2D stiff ODE')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
   #test_example_torchdiffeq()
   #test_stats5()
   plot_trajectories()