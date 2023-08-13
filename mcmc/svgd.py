import math
import torch
from torch.optim import Adam

def median(tensor):
    """
    torch.median() acts differently from np.median(). We want to simulate numpy implementation.
    """
    tensor = tensor.detach().flatten()
    tensor_max = tensor.max()[None]
    return (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.


def kernel_rbf(inputs, h=None):
    n = inputs.shape[0]
    pairwise_distance = torch.norm(inputs[:, None] - inputs, dim=2).pow(2)
    if h is None:
        h = median(pairwise_distance) / math.log(n)
    kernel_matrix = torch.exp(-pairwise_distance / h)
    return kernel_matrix


def get_gradient(lp_f, inputs, h=None):
    n = inputs.size(0)
    inputs = inputs.detach().requires_grad_(True)

    log_prob = lp_f(inputs)
    log_prob_grad = torch.autograd.grad(log_prob.sum(), inputs)[0]

    # See https://github.com/activatedgeek/svgd/issues/1#issuecomment-649235844 for why there is a factor -0.5
    kernel = kernel_rbf(inputs, h)
    kernel_grad = -0.5 * torch.autograd.grad(kernel.sum(), inputs)[0]

    gradient = -(kernel.mm(log_prob_grad) + kernel_grad) / n

    return gradient

class SVGDSampler:
    """
    class for SVGD sampler
    """
    def __init__(self, lp_f):
        """
        :param lp_f: function that returns the log prob
        """

        self.lp_f = lp_f
        
    def gen(self, x, n_iter, lr=None, h=None):
        """
        :param x: initial samples
        :param n_iter: the number of iterations
        :param lr: learning rate of samples updating
        :param h: bandwidth
        """

        for n in range(n_iter):
            if lr is None:
                lr=0.1 * 0.5 ** (n // 250)
            optimizer = Adam([x], lr=lr)
            optimizer.zero_grad()
            x.grad = get_gradient(self.lp_f, x, h)
            optimizer.step()
            if n % 100 == 99:
                print("svgd: ", torch.mean(x, dim=0))
                
        return x
        