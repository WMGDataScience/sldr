import torch as K
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import pdb


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class MADDPG(object):
    def __init__(self, observation_space, action_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=K.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, dtype=K.float32, device="cuda"):

        super(MADDPG, self).__init__()

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
        self.observation_space = observation_space
        self.action_space = action_space

        # model initialization
        self.entities = []
        
        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []

        for i in range(2):
            self.actors.append(Actor(observation_space, action_space[i], discrete, out_func).to(device))
            self.actors_target.append(Actor(observation_space, action_space[i], discrete, out_func).to(device))
            self.actors_optim.append(optimizer(self.actors[i].parameters(), lr = actor_lr))

        for i in range(2):
            hard_update(self.actors_target[i], self.actors[i])

        self.entities.extend(self.actors)
        self.entities.extend(self.actors_target)
        self.entities.extend(self.actors_optim) 
        
        # critics   
        self.critics = []
        self.critics_target = []
        self.critics_optim = []

        for i in range(2):
            self.critics.append(Critic(observation_space, action_space[i]).to(device))
            self.critics_target.append(Critic(observation_space, action_space[i]).to(device))
            self.critics_optim.append(optimizer(self.critics[i].parameters(), lr = critic_lr))
        
        for i in range(2):
            hard_update(self.critics_target[i], self.critics[i])
            
        self.entities.extend(self.critics)
        self.entities.extend(self.critics_target)
        self.entities.extend(self.critics_optim)

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

    def select_action(self, state, i_agent, exploration=False):
        self.actors[i_agent].eval()
        with K.no_grad():
            mu = self.actors[i_agent](state.to(self.device))
        self.actors[i_agent].train()
        if exploration:
            mu = K.tensor(exploration.get_noisy_action(mu.cpu().numpy()), dtype=self.dtype, device=self.device)
        mu = mu.clamp(int(self.action_space[i_agent].low[0]), int(self.action_space[i_agent].high[0]))

        return mu

    def update_parameters(self, batch, normalizer=None):

        observation_space = self.observation_space - 3
        action_space = self.action_space[0].shape[0]

        V = K.zeros((len(batch['o']), 1), dtype=self.dtype, device=self.device)

        s1 = K.cat([K.tensor(batch['o'], dtype=self.dtype, device=self.device)[:, 0:observation_space],
                    K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)
        s2 = K.cat([K.tensor(batch['o'], dtype=self.dtype, device=self.device)[:, observation_space:],
                    K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)

        a1 = K.tensor(batch['u'], dtype=self.dtype, device=self.device)[:, 0:action_space]
        a2 = K.tensor(batch['u'], dtype=self.dtype, device=self.device)[:, action_space:]

        r = K.tensor(batch['r'], dtype=self.dtype, device=self.device).unsqueeze(1)

        s1_ = K.cat([K.tensor(batch['o_2'], dtype=self.dtype, device=self.device)[:, 0:observation_space],
                     K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)
        s2_ = K.cat([K.tensor(batch['o_2'], dtype=self.dtype, device=self.device)[:, observation_space:],
                     K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)

        
        if normalizer[0] is not None:
            s1 = normalizer[0].preprocess(s1)
            s1_ = normalizer[0].preprocess(s1_)

        if normalizer[1] is not None:
            s2 = normalizer[1].preprocess(s2)
            s2_ = normalizer[1].preprocess(s2_)

        a1_ = self.actors_target[0](s1_)
        a2_ = self.actors_target[1](s2_)
        
        # Critics
        # updating the critic of the robot
        Q = self.critics[0](s1, a1)     
        V = self.critics_target[0](s1_, a1_).detach()

        target_Q = (V * self.gamma) + r
        target_Q = target_Q.clamp(-1./(1.-self.gamma), 0.)
        loss_critic = self.loss_func(Q, target_Q)

        self.critics_optim[0].zero_grad()
        loss_critic.backward()
        self.critics_optim[0].step()

        loss_critic_robot = loss_critic.item()

        # updating the critic of the object
        Q = self.critics[1](s2, a2)     
        V = self.critics_target[1](s2_, a2_).detach()

        target_Q = (V * self.gamma) + r
        target_Q = target_Q.clamp(-1./(1.-self.gamma), 0.)
        loss_critic = self.loss_func(Q, target_Q)

        self.critics_optim[1].zero_grad()
        loss_critic.backward()
        self.critics_optim[1].step()

        # Actors 
        # updating the actor of the robot
        a1 = self.actors[0](s1)

        loss_actor = -self.critics[0](s1, a1).mean()
        
        if self.regularization:
            #loss_actor += (self.actors[0].get_preactivations(s)**2).mean()*1
            loss_actor += (self.actors[0](s1)**2).mean()*1

        self.actors_optim[0].zero_grad()        
        loss_actor.backward()
        #K.nn.utils.clip_grad_norm_(self.actors[0].parameters(), 0.5)
        self.actors_optim[0].step()

        loss_actor_robot = loss_actor.item()

        # updating the actor of the object
        a2 = self.actors[1](s2)

        loss_actor = -self.critics[1](s2, a2).mean()
        
        if self.regularization:
            #loss_actor += (self.actors[0].get_preactivations(s)**2).mean()*1
            loss_actor += (self.actors[1](s2)**2).mean()*1

        self.actors_optim[1].zero_grad()        
        loss_actor.backward()
        #K.nn.utils.clip_grad_norm_(self.actors[0].parameters(), 0.5)
        self.actors_optim[1].step()
                
        return loss_critic_robot, loss_actor_robot

    def update_target(self):

        soft_update(self.actors_target[0], self.actors[0], self.tau)
        soft_update(self.critics_target[0], self.critics[0], self.tau)

        soft_update(self.actors_target[1], self.actors[1], self.tau)
        soft_update(self.critics_target[1], self.critics[1], self.tau)



