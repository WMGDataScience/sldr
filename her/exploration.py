import numpy as np

import torch as K
import torch.nn.functional as F


def onehot_from_logits(logits):
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    return argmax_acs

def sample_gumbel(logits, eps=1e-20):
    U = K.rand(logits.shape, dtype=logits.dtype, device=logits.device, requires_grad=False)
    return -K.log(-K.log(U + eps) + eps)

def gumbel_softmax_sample(logits, exploration=False, temperature=1.0):
    if exploration:
        y = logits + sample_gumbel(logits)
    else:
        y = logits
    return F.softmax(y / temperature, dim=1)

def gumbel_softmax(logits, exploration=False, hard=True, temperature=1.0):
    y = gumbel_softmax_sample(logits, exploration, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y

# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale

    def get_noisy_action(self, x):
        return  x+self.noise() 


class Noise:
    def __init__(self,action_dimension, sigma=0.2, eps=0.3, max_u=1.0, min_u=None):
        self.action_dimension = action_dimension
        self.sigma = sigma
        self.eps = eps
        self.max_u = max_u
        self.min_u = -max_u if min_u is None else min_u
        

    def noise(self, size):
        return self.sigma*np.random.randn(*size)

    def random_action(self, size):
        return np.random.uniform(low=self.min_u , high=self.max_u, size=size)

    def get_noisy_action(self, x):
        size = (x.shape[0], self.action_dimension)
        p = np.random.binomial(1, self.eps, size=(x.shape[0],1))
        return p*self.random_action(size) + (1-p)*(x+self.noise(size)) 

