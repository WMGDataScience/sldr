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
