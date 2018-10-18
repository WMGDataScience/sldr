import torch as K
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from macomm.exploration import gumbel_softmax

import pdb


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG(object):
    def __init__(self, observation_space, action_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=F.sigmoid,
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
            mu += K.tensor(exploration.noise(), dtype=self.dtype, device=self.device)
        
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

        
        return loss_critic.item(), loss_actor.item()


class DDPGC(object):
    def __init__(self, observation_space, action_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=F.sigmoid,
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

