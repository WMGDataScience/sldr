import torch as K
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from her.exploration import gumbel_softmax
from her.agents.basic import ForwardDynReg, AutoEncReg, AutoEncNextReg

import pdb


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


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

    def update_parameters(self, batch, normalizer=None):
        
        mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)

        V = K.zeros((len(batch.state), 1), device=self.device)

        s = K.cat(batch.state, dim=0).to(self.device)
        a = K.cat(batch.action, dim=0).to(self.device)
        r = K.cat(batch.reward, dim=0).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=0)
        a_ = K.zeros_like(a)[:,0:s_.shape[0],]
        
        if normalizer[0] is not None:
            s = normalizer[0].preprocess(s)
            s_ = normalizer[0].preprocess(s_)

        if self.normalized_rewards:
            if normalizer[1] is not None:
                r = r
                #r = normalizer[1].preprocess(r)
            else:
                r -= r.mean()
                r /= r.std()
        
        Q = self.critics[0](s, a)        
        a_ = self.actors_target[0](s_)
        V[mask] = self.critics_target[0](s_, a_).detach()

        #loss_critic = self.loss_func(Q, (V * self.gamma) + r)
        target_Q = (V * self.gamma) + r
        target_Q = target_Q.clamp(-1./(1.-self.gamma), 0.)
        
        loss_critic = self.loss_func(Q, target_Q)

        self.critics_optim[0].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.critics[0].parameters(), 0.5)
        self.critics_optim[0].step()

        a = self.actors[0](s)

        loss_actor = -self.critics[0](s, a).mean()
        
        if self.regularization:
            #loss_actor += (self.actors[0].get_preactivations(s)**2).mean()*1
            loss_actor += (self.actors[0](s)**2).mean()*1

        self.actors_optim[0].zero_grad()        
        loss_actor.backward()
        K.nn.utils.clip_grad_norm_(self.actors[0].parameters(), 0.5)
        self.actors_optim[0].step()

        soft_update(self.actors_target[0], self.actors[0], self.tau)
        soft_update(self.critics_target[0], self.critics[0], self.tau)

        return loss_critic.item(), loss_actor.item()


class DDPG_GAR(object):
    def __init__(self, observation_space, action_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=K.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, dtype=K.float32, device="cuda"):

        super(DDPG_GAR, self).__init__()

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

        self.critics.append(Critic(observation_space, action_space).to(device))
        self.critics_target.append(Critic(observation_space, action_space).to(device))
        self.critics_optim.append(optimizer(self.critics[1].parameters(), lr = critic_lr))

        hard_update(self.critics_target[1], self.critics[1])
            
        self.entities.extend(self.critics)
        self.entities.extend(self.critics_target)
        self.entities.extend(self.critics_optim)

        # Forward Dynamics Network
        self.forward = AutoEncNextReg(observation_space, action_space).to(device)
        #self.forward = AutoEncReg(observation_space, action_space).to(device)
        #self.forward = ForwardDynReg(observation_space, action_space).to(device)
        self.forward_optim = optimizer(self.forward.parameters(), lr = critic_lr)

        self.entities.append(self.forward)
        self.entities.append(self.forward_optim)

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

    def update_parameters(self, batch, normalizer=None):
        
        mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)

        V = K.zeros((len(batch.state), 1), device=self.device)

        s = K.cat(batch.state, dim=0).to(self.device)
        a = K.cat(batch.action, dim=0).to(self.device)
        r = K.cat(batch.reward, dim=0).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=0)
        a_ = K.zeros_like(a)[:,0:s_.shape[0],]

        if self.normalized_rewards:
            if normalizer[1] is not None:
                r = r
                #r = normalizer[1].preprocess(r)
            else:
                r -= r.mean()
                r /= r.std()
        
        if normalizer[0] is not None:
            normed_s = normalizer[0].preprocess(s)
            normed_s_ = normalizer[0].preprocess(s_)

        #p = self.forward.get_intr_rewards(s,a)
        p = K.zeros_like(r)
        p[mask], M = self.forward.get_intr_rewards(normed_s_)
        if self.normalized_rewards:
            if normalizer[2] is not None:
                p = normalizer[2].preprocess_with_update(p.to('cpu')).to(self.device)          
            else:
                p -= p.mean()
                p /= p.std()
        
        # extrinsic Q update
        Q = self.critics[0](normed_s, a)

        a_ = self.actors_target[0](normed_s_)

        V[mask] = self.critics_target[0](normed_s_, a_).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r + 0.1*p)

        self.critics_optim[0].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.critics[0].parameters(), 0.5)
        self.critics_optim[0].step()

        # intrinsic Q update
        Q = self.critics[1](normed_s, a)

        V[mask] = self.critics_target[1](normed_s_, a_).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + p)
        # in case of reward propagation
        #reg_critic =  self.critics[0].GAR(1, 1, 0.000001)
        #loss_critic += reg_critic

        self.critics_optim[1].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.critics[1].parameters(), 0.5)
        self.critics_optim[1].step()

        # actor update
        a = self.actors[0](normed_s)

        loss_actor = - self.critics[0](normed_s, a).mean() - 0.1*self.critics[1](normed_s, a).mean() #- self.critics[0].GAR(1, 1, 0.000001)
        
        if self.regularization:
            loss_actor += (self.actors[0].get_preactivations(normed_s)**2).mean()*1

        self.actors_optim[0].zero_grad()        
        loss_actor.backward()
        K.nn.utils.clip_grad_norm_(self.actors[0].parameters(), 0.5)
        self.actors_optim[0].step()

        soft_update(self.actors_target[0], self.actors[0], self.tau)
        soft_update(self.critics_target[0], self.critics[0], self.tau)
        soft_update(self.critics_target[1], self.critics[1], self.tau)

        p = p.squeeze(1).cpu().numpy()
        r = r.squeeze(1).cpu().numpy()

        ind = np.argsort(p)

        return loss_critic.item(), loss_actor.item(), p[ind], r[ind], M

    def update_forward(self, batch, normalizer=None):
        mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)

        s = K.cat(batch.state, dim=0).to(self.device)
        a = K.cat(batch.action, dim=0).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=0)
        
        if normalizer[0] is not None:
            s = normalizer[0].preprocess(s)
            s_ = normalizer[0].preprocess(s_)

        #pred = self.forward(s,a)
        pred = self.forward(s_)
        #loss_forward = self.loss_func(pred[mask], s_)
        loss_forward = self.loss_func(pred, s_)
        #loss_forward = self.loss_func(pred, K.cat([s, a], dim=1))
        reg_forward = self.forward.GAR(0.03, 0.03, 0.001, False)
        #reg_forward = self.forward.GAR(1, 1, 1)
        loss_forward += reg_forward

        self.forward_optim.zero_grad()
        loss_forward.backward()
        #K.nn.utils.clip_grad_norm_(self.forward.parameters(), 0.5)
        self.forward_optim.step()

        return loss_forward.item()-reg_forward.item(), reg_forward.item()


class HDDPG(object):
    def __init__(self, observation_space, action_space, goal_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=K.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, dtype=K.float32, device="cuda"):

        super(HDDPG, self).__init__()

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
        self.goal_space = goal_space

        # model initialization
        self.entities = []
        
        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []
        
        self.actors.append(Actor(observation_space+goal_space.shape[0], action_space, discrete, out_func).to(device))
        self.actors_target.append(Actor(observation_space+goal_space.shape[0], action_space, discrete, out_func).to(device))

        self.actors.append(Actor(observation_space, goal_space, discrete, out_func).to(device))
        self.actors_target.append(Actor(observation_space, goal_space, discrete, out_func).to(device))

        for i in range(len(self.actors)):
            self.actors_optim.append(optimizer(self.actors[i].parameters(), lr = actor_lr))
            hard_update(self.actors_target[i], self.actors[i])

        self.entities.extend(self.actors)
        self.entities.extend(self.actors_target)
        self.entities.extend(self.actors_optim) 
        
        # critics   
        self.critics = []
        self.critics_target = []
        self.critics_optim = []
        
        self.critics.append(Critic(observation_space+goal_space.shape[0], action_space).to(device))
        self.critics_target.append(Critic(observation_space+goal_space.shape[0], action_space).to(device))

        self.critics.append(Critic(observation_space, goal_space).to(device))
        self.critics_target.append(Critic(observation_space, goal_space).to(device))

        for i in range(len(self.critics)):
            self.critics_optim.append(optimizer(self.critics[i].parameters(), lr = critic_lr))
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

    def select_goal(self, state, exploration=False):
        self.actors[1].eval()
        with K.no_grad():
            mu = self.actors[1](state.to(self.device))
        self.actors[1].train()
        if exploration:
            mu = K.tensor(exploration.get_noisy_action(mu.cpu().numpy()), dtype=self.dtype, device=self.device)

        #mu = mu * K.tensor(self.goal_space.high[0], dtype=self.dtype, device=self.device)
        mu = mu.clamp(-1, 1)

        return mu

    def select_action(self, state, exploration=False):
        self.actors[0].eval()
        with K.no_grad():
            mu = self.actors[0](state.to(self.device))
        self.actors[0].train()
        if exploration:
            mu = K.tensor(exploration.get_noisy_action(mu.cpu().numpy()), dtype=self.dtype, device=self.device)
        
        mu = mu * K.tensor(self.action_space.high[0], dtype=self.dtype, device=self.device)
        mu = mu.clamp(int(self.action_space.low[0]), int(self.action_space.high[0]))

        return mu

    def update_parameters(self, batch):

        for i in range(2):
            
            mask = K.tensor(tuple(map(lambda s: s is not None, batch[i].next_state)), dtype=K.uint8, device=self.device)

            V = K.zeros((len(batch[i].state), 1), device=self.device)

            s = K.cat(batch[i].state, dim=0).to(self.device)
            a = K.cat(batch[i].action, dim=0).to(self.device)
            r = K.cat(batch[i].reward, dim=0).to(self.device)
            s_ = K.cat([i.to(self.device) for i in batch[i].next_state if i is not None], dim=0)
            a_ = K.zeros_like(a)[:,0:s_.shape[0],]

            if self.normalized_rewards:
                r -= r.mean()
                r /= r.std()
            
            Q = self.critics[i](s, a)
            
            a_ = self.actors_target[i](s_)

            V[mask] = self.critics_target[i](s_, a_).detach()

            loss_critic = self.loss_func(Q, (V * self.gamma) + r) 

            self.critics_optim[i].zero_grad()
            loss_critic.backward()
            K.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 0.5)
            self.critics_optim[i].step()

            a = self.actors[i](s)

            loss_actor = -self.critics[i](s, a).mean()
            
            if self.regularization:
                loss_actor += (self.actors[i].get_preactivations(s)**2).mean()*1

            self.actors_optim[i].zero_grad()        
            loss_actor.backward()
            K.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actors_optim[i].step()

            soft_update(self.actors_target[i], self.actors[i], self.tau)
            soft_update(self.critics_target[i], self.critics[i], self.tau)

        
        return loss_critic.item(), loss_actor.item()


class DDPGC(object):
    def __init__(self, observation_space, action_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=K.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, dtype=K.float32, device="cuda"):

        super(DDPGC, self).__init__()

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


        # actors
        self.cactors = []
        self.cactors_target = []
        self.cactors_optim = []
        
        self.cactors.append(Actor(observation_space, action_space, discrete, out_func).to(device))
        self.cactors_target.append(Actor(observation_space, action_space, discrete, out_func).to(device))
        self.cactors_optim.append(optimizer(self.cactors[0].parameters(), lr = actor_lr))

        hard_update(self.cactors_target[0], self.cactors[0])

        self.entities.extend(self.cactors)
        self.entities.extend(self.cactors_target)
        self.entities.extend(self.cactors_optim) 
        
        # critics   
        self.ccritics = []
        self.ccritics_target = []
        self.ccritics_optim = []
        
        self.ccritics.append(Critic(observation_space, action_space).to(device))
        self.ccritics_target.append(Critic(observation_space, action_space).to(device))
        self.ccritics_optim.append(optimizer(self.ccritics[0].parameters(), lr = critic_lr))

        hard_update(self.ccritics_target[0], self.ccritics[0])
            
        self.entities.extend(self.ccritics)
        self.entities.extend(self.ccritics_target)
        self.entities.extend(self.ccritics_optim)

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
        self.cactors[0].eval()
        with K.no_grad():
            mu = self.actors[0](state.to(self.device))
            cmu = self.cactors[0](state.to(self.device))
        self.actors[0].train()
        self.cactors[0].train()

        mu = mu*(1 - cmu)

        mu = K.tensor(self.action_space.low[0], dtype=mu.dtype, device=mu.device) + mu*K.tensor((self.action_space.high[0]-self.action_space.low[0]), dtype=mu.dtype, device=mu.device)
        
        if exploration:
            mu += K.tensor(exploration.noise(), dtype=self.dtype, device=self.device)
            cmu += K.tensor(exploration.noise(), dtype=self.dtype, device=self.device)

        mu = mu.clamp(int(self.action_space.low[0]), int(self.action_space.high[0]))

        return mu

    def update_parameters(self, batch):
        
        mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)

        V = K.zeros((len(batch.state), 1), device=self.device)

        s = K.cat(batch.state, dim=0).to(self.device)
        a = K.cat(batch.action, dim=0).to(self.device)
        r = K.cat(batch.reward, dim=0).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=0)
        a_ = K.zeros_like(a)[:,0:s_.shape[0],]

        if self.normalized_rewards:
            r -= r.mean()
            r /= r.std()
        
        Q = self.critics[0](s, a)
        
        a_ = self.actors_target[0](s_)

        V[mask] = self.critics_target[0](s_, a_).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r) 

        self.critics_optim[0].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.critics[0].parameters(), 0.5)
        self.critics_optim[0].step()

        a = self.actors[0](s)

        loss_actor = -self.critics[0](s, a).mean()
        
        if self.regularization:
            loss_actor += (self.actors[0].get_preactivations(s)**2).mean()*1e-3

        self.actors_optim[0].zero_grad()        
        loss_actor.backward()
        K.nn.utils.clip_grad_norm_(self.actors[0].parameters(), 0.5)
        self.actors_optim[0].step()

        soft_update(self.actors_target[0], self.actors[0], self.tau)
        soft_update(self.critics_target[0], self.critics[0], self.tau)


        mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)

        V = K.zeros((len(batch.state), 1), device=self.device)

        s = K.cat(batch.state, dim=0).to(self.device)
        a = K.cat(batch.action, dim=0).to(self.device)
        r = K.cat(batch.reward, dim=0).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=0)
        a_ = K.zeros_like(a)[:,0:s_.shape[0],]

        if self.normalized_rewards:
            r -= r.mean()
            r /= r.std()

        Q = self.ccritics[0](s, a)
        
        a_ = self.cactors_target[0](s_)

        V[mask] = self.ccritics_target[0](s_, a_).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + (1-r)) 

        self.ccritics_optim[0].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.ccritics[0].parameters(), 0.5)
        self.ccritics_optim[0].step()

        a = self.cactors[0](s)

        loss_actor = -self.ccritics[0](s, a).mean()
        
        if self.regularization:
            loss_actor += (self.cactors[0].get_preactivations(s)**2).mean()*1e-3

        self.cactors_optim[0].zero_grad()        
        loss_actor.backward()
        K.nn.utils.clip_grad_norm_(self.cactors[0].parameters(), 0.5)
        self.cactors_optim[0].step()

        soft_update(self.cactors_target[0], self.cactors[0], self.tau)
        soft_update(self.ccritics_target[0], self.ccritics[0], self.tau)

        
        return loss_critic.item(), loss_actor.item()


# class MAHCDDPG(PARENT):
#     def __init__(self, num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=K.sigmoid,
#                  discrete=True, regularization=False, normalized_rewards=False, communication=None, discrete_comm=True, Comm_Actor=None, Comm_Critic=None, dtype=K.float32, device="cuda"):
        
#         super().__init__(num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func,
#                          discrete, regularization, normalized_rewards, communication, discrete_comm, Comm_Actor, Comm_Critic, dtype, device)

#         optimizer, lr = optimizer
#         actor_lr, critic_lr, comm_actor_lr, comm_critic_lr = lr

#         # model initialization
#         self.entities = []
        
#         # actors
#         self.actors = []
#         self.actors_target = []
#         self.actors_optim = []
        
#         for i in range(num_agents):
#             self.actors.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
#             self.actors_target.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
#             self.actors_optim.append(optimizer(self.actors[i].parameters(), actor_lr))

#         for i in range(num_agents):
#             hard_update(self.actors_target[i], self.actors[i])

#         self.entities.extend(self.actors)
#         self.entities.extend(self.actors_target)
#         self.entities.extend(self.actors_optim) 
        
#         # critics   
#         self.critics = []
#         self.critics_target = []
#         self.critics_optim = []
        
#         for i in range(num_agents):
#             self.critics.append(Critic(observation_space+medium_space, action_space).to(device))
#             self.critics_target.append(Critic(observation_space+medium_space, action_space).to(device))
#             self.critics_optim.append(optimizer(self.critics[i].parameters(), lr = critic_lr))

#         for i in range(num_agents):
#             hard_update(self.critics_target[i], self.critics[i])
            
#         self.entities.extend(self.critics)
#         self.entities.extend(self.critics_target)
#         self.entities.extend(self.critics_optim)    

#         # communication actors
#         self.comm_actors = []
#         self.comm_actors_target = []
#         self.comm_actors_optim = []
        
#         for i in range(num_agents):
#             self.comm_actors.append(Comm_Actor(observation_space, 1, discrete_comm, K.sigmoid).to(device))
#             self.comm_actors_target.append(Comm_Actor(observation_space, 1, discrete_comm, K.sigmoid).to(device))
#             self.comm_actors_optim.append(optimizer(self.comm_actors[i].parameters(), lr = comm_actor_lr))
            
#         for i in range(num_agents):
#             hard_update(self.comm_actors_target[i], self.comm_actors[i])

#         self.entities.extend(self.comm_actors)
#         self.entities.extend(self.comm_actors_target)
#         self.entities.extend(self.comm_actors_optim)

#         # communication critics   
#         self.comm_critics = []
#         self.comm_critics_target = []
#         self.comm_critics_optim = []
        
#         for i in range(num_agents):
#             self.comm_critics.append(Comm_Critic(observation_space*num_agents, 1*num_agents).to(device))
#             self.comm_critics_target.append(Comm_Critic(observation_space*num_agents, 1*num_agents).to(device))
#             self.comm_critics_optim.append(optimizer(self.comm_critics[i].parameters(), lr = comm_critic_lr))

#         for i in range(num_agents):
#             hard_update(self.comm_critics_target[i], self.comm_critics[i]) 

#         print('amanin')

#     def select_comm_action(self, state, i_agent, exploration=False):
#         self.comm_actors[i_agent].eval()
#         with K.no_grad():
#             mu = self.comm_actors[i_agent](state.to(self.device))
#         self.actors[i_agent].train()
#         if self.discrete_comm:
#             mu = gumbel_softmax(mu, exploration=exploration)
#         else:
#             if exploration:
#                 mu += K.tensor(exploration.noise(), dtype=self.dtype, device=self.device)
#                 mu = mu.clamp(0, 1) 
#         return mu

#     def update_parameters(self, batch, batch2, i_agent):
        
#         mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)

#         V = K.zeros((len(batch.state), 1), device=self.device)

#         s = K.cat(batch.state, dim=1).to(self.device)
#         a = K.cat(batch.action, dim=1).to(self.device)
#         r = K.cat(batch.reward, dim=1).to(self.device)
#         s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=1)
#         a_ = K.zeros_like(a)[:,0:s_.shape[1],]
        
#         m = K.cat(batch.medium, dim=1).to(self.device)

#         if self.normalized_rewards:
#             r -= r.mean()
#             r /= r.std()

#         Q = self.critics[i_agent](K.cat([s[[i_agent],], m], dim=-1),
#                                   a[[i_agent],])
        
#         for i in range(self.num_agents):
#             a_[i,] = self.actors_target[i](K.cat([s_[[i],], m[:,mask,]], dim=-1))

#         V[mask] = self.critics_target[i_agent](K.cat([s_[[i_agent],], m[:,mask,]], dim=-1),
#                                                a_[[i_agent],]).detach()

#         loss_critic = self.loss_func(Q, (V * self.gamma) + r[[i_agent],].squeeze(0)) 

#         self.critics_optim[i_agent].zero_grad()
#         loss_critic.backward()
#         K.nn.utils.clip_grad_norm_(self.critics[i_agent].parameters(), 0.5)
#         self.critics_optim[i_agent].step()

#         for i in range(self.num_agents):
#             a[i,] = self.actors[i](K.cat([s[[i],], m], dim=-1))

#         loss_actor = -self.critics[i_agent](K.cat([s[[i_agent],], m], dim=-1), 
#                                             a[[i_agent],]).mean()
        
#         if self.regularization:
#             loss_actor += (self.actors[i_agent].get_preactivations(K.cat([s[[i_agent],], m], dim=-1))**2).mean()*1e-3

#         self.actors_optim[i_agent].zero_grad()        
#         loss_actor.backward()
#         K.nn.utils.clip_grad_norm_(self.actors[i_agent].parameters(), 0.5)
#         self.actors_optim[i_agent].step()

#         soft_update(self.actors_target[i_agent], self.actors[i_agent], self.tau)
#         soft_update(self.critics_target[i_agent], self.critics[i_agent], self.tau)

#         ## update of the communication part

#         mask = K.tensor(tuple(map(lambda s: s is not None, batch2.next_state)), dtype=K.uint8, device=self.device)
#         V = K.zeros((len(batch2.state), 1), device=self.device)

#         s = K.cat(batch2.state, dim=1).to(self.device)
#         r = K.cat(batch2.reward, dim=1).to(self.device)
#         s_ = K.cat([i.to(self.device) for i in batch2.next_state if i is not None], dim=1)
        
#         c = K.cat(batch2.comm_action, dim=1).to(self.device)
#         c_ = K.zeros_like(c)[:,0:s_.shape[1],]
        
#         if self.normalized_rewards:
#             r -= r.mean()
#             r /= r.std()

#         Q = self.comm_critics[i_agent](s, c)

#         for i in range(self.num_agents):
#             if self.discrete_comm:
#                 c_[i,] = gumbel_softmax(self.comm_actors_target[i](s_[[i],]), exploration=False)
#             else:
#                 c_[i,] = self.comm_actors_target[i](s_[[i],])

#         V[mask] = self.comm_critics_target[i_agent](s_, c_).detach()

#         loss_critic = self.loss_func(Q, (V * self.gamma) + r[[i_agent],].squeeze(0)) 

#         self.comm_critics_optim[i_agent].zero_grad()
#         loss_critic.backward()
#         K.nn.utils.clip_grad_norm_(self.comm_critics[i_agent].parameters(), 0.5)
#         self.comm_critics_optim[i_agent].step()

#         for i in range(self.num_agents):
#             if self.discrete_comm:
#                 c[i,] = gumbel_softmax(self.comm_actors[i](s[[i],]), exploration=False)
#             else:
#                 c[i,] = self.comm_actors[i](s[[i],])

#         loss_actor = -self.comm_critics[i_agent](s, c).mean()
        
#         if self.regularization:
#             loss_actor += (self.comm_actors[i_agent].get_preactivations(s[[i_agent],])**2).mean()*1e-3

#         self.comm_actors_optim[i_agent].zero_grad()        
#         loss_actor.backward()
#         K.nn.utils.clip_grad_norm_(self.comm_actors[i_agent].parameters(), 0.5)
#         self.comm_actors_optim[i_agent].step()

#         soft_update(self.comm_actors_target[i_agent], self.comm_actors[i_agent], self.tau)
#         soft_update(self.comm_critics_target[i_agent], self.comm_critics[i_agent], self.tau)

        
#         return loss_critic.item(), loss_actor.item()
