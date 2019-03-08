import torch as K
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from her.agents.basic import BackwardDyn

import pdb


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG_BD(object):
    def __init__(self, observation_space, action_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=K.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, object_Qfunc=None, backward_dyn=None, 
                 dtype=K.float32, device="cuda"):

        super(DDPG_BD, self).__init__()

        optimizer, lr = optimizer
        actor_lr, critic_lr = lr

        self.loss_func = loss_func
        self.gamma = gamma
        self.tau = tau
        self.out_func = out_func
        self.discrete = discrete
        self.regularization = regularization
        self.normalized_rewards = normalized_rewards
        self.dtype = dtype
        self.device = device
        self.action_space = action_space

        # model initialization
        self.entities = []
        
        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []
        
        self.actors.append(Actor(observation_space, action_space, discrete, out_func).to(device))
        self.actors_target.append(Actor(observation_space, action_space, discrete, out_func).to(device))
        self.actors_optim.append(optimizer(self.actors[0].parameters(), lr = actor_lr))

        hard_update(self.actors_target[0], self.actors[0])

        self.entities.extend(self.actors)
        self.entities.extend(self.actors_target)
        self.entities.extend(self.actors_optim) 
        
        # critics   
        self.critics = []
        self.critics_target = []
        self.critics_optim = []
        
        self.critics.append(Critic(observation_space, action_space).to(device))
        self.critics_target.append(Critic(observation_space, action_space).to(device))
        self.critics_optim.append(optimizer(self.critics[0].parameters(), lr = critic_lr))

        hard_update(self.critics_target[0], self.critics[0])
            
        self.entities.extend(self.critics)
        self.entities.extend(self.critics_target)
        self.entities.extend(self.critics_optim)

        # backward dynamics model
        if backward_dyn is None:
            self.backward = BackwardDyn(observation_space, action_space).to(device)
            self.backward_optim = optimizer(self.backward.parameters(), lr = critic_lr)
            self.entities.append(self.backward)
            self.entities.append(self.backward_optim)
        else:
            self.backward = backward_dyn.to(device)
            self.backward.eval()
            self.entities.append(self.backward)

        # Learnt Q function for object
        if object_Qfunc is not None:
            self.object_Qfunc = object_Qfunc
            self.object_Qfunc.eval()
            self.entities.append(self.object_Qfunc)

    def to_cpu(self):
        for entity in self.entities:
            if type(entity) != type(self.actors_optim[0]):
                entity.cpu()
        self.device = 'cpu'

    def to_cuda(self):        
        for entity in self.entities:
            if type(entity) != type(self.actors_optim[0]):
                entity.cuda()
        self.device = 'cuda'    

    def select_action(self, state, exploration=False):
        self.actors[0].eval()
        with K.no_grad():
            mu = self.actors[0](state.to(self.device))
        self.actors[0].train()
        if exploration:
            mu = K.tensor(exploration.get_noisy_action(mu.cpu().numpy()), dtype=self.dtype, device=self.device)
        mu = mu.clamp(int(self.action_space.low[0]), int(self.action_space.high[0]))

        return mu

    def update_parameters(self, batch, normalizer=None, use_object_Qfunc=False):

        V = K.zeros((len(batch['o']), 1), dtype=self.dtype, device=self.device)

        s = K.cat([K.tensor(batch['o'], dtype=self.dtype, device=self.device),
                   K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)
        a = K.tensor(batch['u'], dtype=self.dtype, device=self.device)
        r = K.tensor(batch['r'], dtype=self.dtype, device=self.device).unsqueeze(1)
        s_ = K.cat([K.tensor(batch['o_2'], dtype=self.dtype, device=self.device),
                    K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)
        a_ = K.zeros_like(a)
        
        if normalizer is not None:
            s = normalizer.preprocess(s)
            s_ = normalizer.preprocess(s_)
        
        Q = self.critics[0](s, a)    
            
        a_ = self.actors_target[0](s_)
        V = self.critics_target[0](s_, a_).detach()

        if use_object_Qfunc:
            r = self.get_obj_reward(s, s_)
            target_Q = (V * self.gamma) + r
        else:
            target_Q = (V * self.gamma) + r
            target_Q = target_Q.clamp(-1./(1.-self.gamma), 0.)
        
        loss_critic = self.loss_func(Q, target_Q)

        self.critics_optim[0].zero_grad()
        loss_critic.backward()
        self.critics_optim[0].step()

        a = self.actors[0](s)

        loss_actor = -self.critics[0](s, a).mean()
        
        if self.regularization:
            loss_actor += (self.actors[0](s)**2).mean()*1

        self.actors_optim[0].zero_grad()        
        loss_actor.backward()
        self.actors_optim[0].step()
                
        return loss_critic.item(), loss_actor.item()

    def update_target(self):

        soft_update(self.actors_target[0], self.actors[0], self.tau)
        soft_update(self.critics_target[0], self.critics[0], self.tau)

    def estimate_obj_action(self, state, next_state):
        with K.no_grad():
            action = self.backward(state.to(self.device), next_state.to(self.device))

        return action

    def get_obj_reward(self, state, next_state):
        with K.no_grad():
            action = self.backward(state.to(self.device), next_state.to(self.device))
            reward = self.object_Qfunc(state.to(self.device), action)
        
        return reward

    def update_backward(self, batch, normalizer=None):

        s = K.cat([K.tensor(batch['o'], dtype=self.dtype, device=self.device),
                   K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)
        a = K.tensor(batch['u'], dtype=self.dtype, device=self.device)
        s_ = K.cat([K.tensor(batch['o_2'], dtype=self.dtype, device=self.device),
                    K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)

        if normalizer is not None:
            s = normalizer.preprocess(s)
            s_ = normalizer.preprocess(s_)

        a_pred = self.backward(s, s_)

        loss_backward = self.loss_func(a_pred, a)

        self.backward_optim.zero_grad()
        loss_backward.backward()
        self.backward_optim.step()

        return loss_backward.item()





