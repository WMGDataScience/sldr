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
            self.critics.append(Critic(observation_space*2, action_space[2]).to(device))
            self.critics_target.append(Critic(observation_space*2, action_space[2]).to(device))
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

    def update_parameters(self, batch, i_agent, normalizer=None):

        observation_space = self.observation_space - K.tensor(batch['g'], dtype=self.dtype, device=self.device).shape[1]
        action_space = self.action_space[0].shape[0]

        mask = K.tensor(tuple(map(lambda ai_object: ai_object==0, K.tensor(batch['o'][:,-1]))), dtype=K.uint8, device=self.device)

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

        s = [s1, s2]

        a1_ = self.actors_target[0](s1_)
        a2_ = self.actors_target[1](s2_)
        a2_[mask] *= 0.00
        
        # Critics
        # updating the critic of the robot
        Q = self.critics[i_agent](K.cat([s1,s2],dim=1), K.cat([a1,a2],dim=1))     
        V = self.critics_target[i_agent](K.cat([s1_,s2_],dim=1), K.cat([a1_,a2_],dim=1)).detach()

        target_Q = (V * self.gamma) + r
        target_Q = target_Q.clamp(-1./(1.-self.gamma), 0.)
        loss_critic = self.loss_func(Q, target_Q)

        self.critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        self.critics_optim[i_agent].step()

        # Actors 
        a1 = self.actors[0](s1)
        a2 = self.actors[1](s2)
        a2[mask] *= 0.00

        loss_actor = -self.critics[i_agent](K.cat([s1,s2],dim=1), K.cat([a1,a2],dim=1)).mean()
        
        if self.regularization:
            loss_actor += (self.actors[i_agent](s[i_agent])**2).mean()*1

        self.actors_optim[i_agent].zero_grad()        
        loss_actor.backward()
        #K.nn.utils.clip_grad_norm_(self.actors[0].parameters(), 0.5)
        self.actors_optim[i_agent].step()

        return loss_critic.item(), loss_actor.item()

    def update_target(self):

        soft_update(self.actors_target[0], self.actors[0], self.tau)
        soft_update(self.critics_target[0], self.critics[0], self.tau)

        soft_update(self.actors_target[1], self.actors[1], self.tau)
        soft_update(self.critics_target[1], self.critics[1], self.tau)


class DDPG(object):
    def __init__(self, observation_space, action_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=K.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, dtype=K.float32, device="cuda"):

        super(DDPG, self).__init__()

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

    def update_parameters(self, batch, i_agent, normalizer=None):

        observation_space = self.observation_space - K.tensor(batch['g'], dtype=self.dtype, device=self.device).shape[1]
        action_space = self.action_space[0].shape[0]

        mask = K.tensor(tuple(map(lambda ai_object: ai_object==0, K.tensor(batch['o'][:,-1]))), dtype=K.uint8, device=self.device)

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
        a2_[mask] *= 0.00            

        s = [s1, s2]
        s_ = [s1_, s2_]
        a = [a1, a2]
        a_ = [a1_, a2_]
        
        # Critics
        Q = self.critics[i_agent](s[i_agent], a[i_agent])     
        V = self.critics_target[i_agent](s_[i_agent], a_[i_agent]).detach()

        target_Q = (V * self.gamma) + r
        target_Q = target_Q.clamp(-1./(1.-self.gamma), 0.)
        loss_critic = self.loss_func(Q, target_Q)

        self.critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        self.critics_optim[i_agent].step()

        # Actors 
        a1 = self.actors[0](s1)
        a2 = self.actors[1](s2)
        a2[mask] *= 0.00
        a = [a1, a2]

        loss_actor = -self.critics[i_agent](s[i_agent], a[i_agent]).mean()
        
        if self.regularization:
            loss_actor += (self.actors[i_agent](s[i_agent])**2).mean()*1

        self.actors_optim[i_agent].zero_grad()        
        loss_actor.backward()
        self.actors_optim[i_agent].step()

        return loss_critic.item(), loss_actor.item()

    def update_target(self):

        soft_update(self.actors_target[0], self.actors[0], self.tau)
        soft_update(self.critics_target[0], self.critics[0], self.tau)

        soft_update(self.actors_target[1], self.actors[1], self.tau)
        soft_update(self.critics_target[1], self.critics[1], self.tau)


class MADDPG_R(object):
    def __init__(self, observation_space, action_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=K.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, dtype=K.float32, device="cuda"):

        super(MADDPG_R, self).__init__()

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
            self.critics.append(Critic(observation_space, action_space[2]).to(device))
            self.critics_target.append(Critic(observation_space, action_space[2]).to(device))
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

    def update_parameters(self, batch, i_agent, normalizer=None):

        observation_space = self.observation_space - K.tensor(batch['g'], dtype=self.dtype, device=self.device).shape[1]
        action_space = self.action_space[0].shape[0]

        mask = K.tensor(tuple(map(lambda ai_object: ai_object==0, K.tensor(batch['o'][:,-1]))), dtype=K.uint8, device=self.device)

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
        a2_[mask] *= 0.00            

        s = [s1, s2]
        s_ = [s1_, s2_]
        
        # Critics
        Q = self.critics[i_agent](s[i_agent], K.cat([a1, a2],dim=1))     
        V = self.critics_target[i_agent](s_[i_agent], K.cat([a1_, a2_],dim=1)).detach()

        target_Q = (V * self.gamma) + r
        target_Q = target_Q.clamp(-1./(1.-self.gamma), 0.)
        loss_critic = self.loss_func(Q, target_Q)

        self.critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        self.critics_optim[i_agent].step()

        # Actors 
        a1 = self.actors[0](s1)
        a2 = self.actors[1](s2)
        a2[mask] *= 0.00

        loss_actor = -self.critics[i_agent](s[i_agent], K.cat([a1, a2],dim=1)).mean()
        
        if self.regularization:
            loss_actor += (self.actors[i_agent](s[i_agent])**2).mean()*1

        self.actors_optim[i_agent].zero_grad()        
        loss_actor.backward()
        self.actors_optim[i_agent].step()

        return loss_critic.item(), loss_actor.item()

    def update_target(self):

        soft_update(self.actors_target[0], self.actors[0], self.tau)
        soft_update(self.critics_target[0], self.critics[0], self.tau)

        soft_update(self.actors_target[1], self.actors[1], self.tau)
        soft_update(self.critics_target[1], self.critics[1], self.tau)


class MADDPG_RAE(object):
    def __init__(self, observation_space, action_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=K.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, dtype=K.float32, device="cuda"):

        super(MADDPG_RAE, self).__init__()

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
            self.critics.append(Critic(observation_space, action_space[2]).to(device))
            self.critics_target.append(Critic(observation_space, action_space[2]).to(device))
            self.critics_optim.append(optimizer(self.critics[i].parameters(), lr = critic_lr))
        
        for i in range(2):
            hard_update(self.critics_target[i], self.critics[i])
            
        self.entities.extend(self.critics)
        self.entities.extend(self.critics_target)
        self.entities.extend(self.critics_optim)

        # backward dynamics model
        self.backward = BackwardDyn(3, action_space[1]).to(device)
        self.backward_optim = optimizer(self.backward.parameters(), lr = critic_lr)

        self.entities.append(self.backward)
        self.entities.append(self.backward_optim)

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

    def update_parameters(self, batch, i_agent, normalizer=None):

        observation_space = self.observation_space - K.tensor(batch['g'], dtype=self.dtype, device=self.device).shape[1]
        action_space = self.action_space[0].shape[0]

        mask = K.tensor(tuple(map(lambda ai_object: ai_object==0, K.tensor(batch['o'][:,-1]))), dtype=K.uint8, device=self.device)

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
        s2__ = K.cat([K.tensor(batch['o_3'], dtype=self.dtype, device=self.device)[:, observation_space:],
                     K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)

        ag = K.tensor(batch['ag'], dtype=self.dtype, device=self.device)
        ag_2 = K.tensor(batch['ag_2'], dtype=self.dtype, device=self.device)  
        ag_3 = K.tensor(batch['ag_3'], dtype=self.dtype, device=self.device)             


        if normalizer[0] is not None:
            s1 = normalizer[0].preprocess(s1)
            s1_ = normalizer[0].preprocess(s1_)

        if normalizer[1] is not None:
            s2 = normalizer[1].preprocess(s2)
            s2_ = normalizer[1].preprocess(s2_)
            s2__ = normalizer[1].preprocess(s2__)

        a1_ = self.actors_target[0](s1_)
        a2_ = self.actors_target[1](s2_)
        #a2_[mask] = self.estimate_obj_action(s2_[mask], s2__[mask])   
        a2_[mask] = self.estimate_obj_action(ag_2[mask], ag_3[mask])         

        s = [s1, s2]
        s_ = [s1_, s2_]
        
        # Critics
        Q = self.critics[i_agent](s[i_agent], K.cat([a1, a2],dim=1))     
        V = self.critics_target[i_agent](s_[i_agent], K.cat([a1_, a2_],dim=1)).detach()

        target_Q = (V * self.gamma) + r
        target_Q = target_Q.clamp(-1./(1.-self.gamma), 0.)
        loss_critic = self.loss_func(Q, target_Q)

        self.critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        self.critics_optim[i_agent].step()

        # Actors 
        a1 = self.actors[0](s1)
        a2 = self.actors[1](s2)
        #a2_[mask] = self.estimate_obj_action(s2[mask], s2_[mask])
        a2[mask] = self.estimate_obj_action(ag[mask], ag_2[mask])

        loss_actor = -self.critics[i_agent](s[i_agent], K.cat([a1, a2],dim=1)).mean()
        
        if self.regularization:
            loss_actor += (self.actors[i_agent](s[i_agent])**2).mean()*1

        self.actors_optim[i_agent].zero_grad()        
        loss_actor.backward()
        self.actors_optim[i_agent].step()

        return loss_critic.item(), loss_actor.item()

    def update_target(self):

        soft_update(self.actors_target[0], self.actors[0], self.tau)
        soft_update(self.critics_target[0], self.critics[0], self.tau)

        soft_update(self.actors_target[1], self.actors[1], self.tau)
        soft_update(self.critics_target[1], self.critics[1], self.tau)

    def estimate_obj_action(self, state2, next_state2):
        self.backward.eval()
        with K.no_grad():
            action2 = self.backward(state2.to(self.device), next_state2.to(self.device))
        self.backward.train()

        return action2

    def update_backward(self, batch, normalizer=None):

        observation_space = self.observation_space - K.tensor(batch['g'], dtype=self.dtype, device=self.device).shape[1]
        action_space = self.action_space[0].shape[0]

        mask = K.tensor(tuple(map(lambda ai_object: ai_object>0, K.tensor(batch['o'][:,-1]))), dtype=K.uint8, device=self.device)

        #s2 = K.cat([K.tensor(batch['o'], dtype=self.dtype, device=self.device)[:, observation_space:],
        #            K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)

        a2 = K.tensor(batch['u'], dtype=self.dtype, device=self.device)[:, action_space:]

        #s2_ = K.cat([K.tensor(batch['o_2'], dtype=self.dtype, device=self.device)[:, observation_space:],
        #             K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)

        #if normalizer[1] is not None:
        #    s2 = normalizer[1].preprocess(s2)
        #    s2_ = normalizer[1].preprocess(s2_)

        #a2_pred = self.backward(s2[mask], s2_[mask])

        ag = K.tensor(batch['ag'], dtype=self.dtype, device=self.device)
        ag_2 = K.tensor(batch['ag_2'], dtype=self.dtype, device=self.device)

        a2_pred = self.backward(ag[mask], ag_2[mask])

        loss_backward = self.loss_func(a2_pred, a2[mask])

        self.backward_optim.zero_grad()
        loss_backward.backward()
        #K.nn.utils.clip_grad_norm_(self.forward.parameters(), 0.5)
        self.backward_optim.step()

        return loss_backward.item()




