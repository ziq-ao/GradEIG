import torch
import torch.nn as nn
from utils.plot import plot_hist_marginals

class ConditionalGaussian(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ConditionalGaussian, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Conditional MLP
        self.condition_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Parameters of the Gaussian distribution
        self.mean_net = nn.Linear(hidden_dim, 1)
        self.std_net = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # Condition model
        condition = self.condition_net(x)
        
        # Compute mean and standard deviation
        mean = self.mean_net(condition)
        std = torch.exp(self.std_net(condition))
        
        # Create Gaussian distribution
        dist = torch.distributions.normal.Normal(mean, std)
        
        return dist
    
    def log_prob(self, inputs, context):
        # Compute log probability of y given x
        dist = self.forward(context)
        log_prob = dist.log_prob(inputs)
        
        return log_prob
    
    def sample(self, num_samples, context):
        # Sample from the conditional Gaussian distribution
        batch_size = context.shape[0]
        samples = []
        
        for i in range(batch_size):
            sample = self.forward(context[i:i+1]).sample((num_samples,))
            samples.append(sample)
        
        samples = torch.cat(samples, dim=0)
        
        return samples.view(batch_size, num_samples,-1)

if __name__ == "__main__":
    # example
    input_dim = 5
    hidden_dim = 10
    
    # construct conditional gaussian model
    model = ConditionalGaussian(input_dim, hidden_dim)
    
    # generate inputs
    batch_size = 10000
    x = torch.randn(batch_size, input_dim)
    
    # sample from model
    num_samples = 1
    samples = model.sample(num_samples, context=x)
    samples = samples.view(-1,1)
    print(samples)
    
    model.log_prob(samples, context=x)
    plot_hist_marginals(torch.cat((x,samples),1))

