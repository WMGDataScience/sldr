import torch as K
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from sldr.agents.basic import BackwardDyn
from sldr.utils import get_obj_obs, get_rob_obs

import pdb


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG_BD(object):
    def __init__(self, observation_space, action_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=K.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, agent_id=0, object_Qfunc=None, backward_dyn=None, 
                 object_policy=None, reward_fun=None, n_objects=[3,2,0], clip_Q_neg=None, dtype=K.float32, device="cuda"):

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
        self.observation_space = observation_space
        self.action_space = action_space
        self.agent_id = agent_id
        self.object_Qfunc = object_Qfunc
        self.object_policy = object_policy
        self.object_backward = backward_dyn
        self.n_objects = n_objects
        self.clip_Q_neg = clip_Q_neg if clip_Q_neg is not None else -1./(1.-self.gamma)
        self.n_aux_critics = 1+n_objects[0]-n_objects[2] if len(object_Qfunc) > 1 else n_objects[0]-n_objects[2]

        # model initialization
        self.entities = []
        
        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []
        
        self.actors.append(Actor(observation_space, action_space[agent_id], discrete, out_func).to(device))
        self.actors_target.append(Actor(observation_space, action_space[agent_id], discrete, out_func).to(device))
        self.actors_optim.append(optimizer(self.actors[0].parameters(), lr = actor_lr))

        hard_update(self.actors_target[0], self.actors[0])

        self.entities.extend(self.actors)
        self.entities.extend(self.actors_target)
        self.entities.extend(self.actors_optim) 
        
        # critics   
        self.critics = []
        self.critics_target = []
        self.critics_optim = []
        
        self.critics.append(Critic(observation_space, action_space[agent_id]).to(device))
        self.critics_target.append(Critic(observation_space, action_space[agent_id]).to(device))
        self.critics_optim.append(optimizer(self.critics[0].parameters(), lr = critic_lr))

        hard_update(self.critics_target[0], self.critics[0])

        for i_critic in range(1,self.n_aux_critics+1):
            self.critics.append(Critic(observation_space, action_space[agent_id]).to(device))
            self.critics_target.append(Critic(observation_space, action_space[agent_id]).to(device))
            self.critics_optim.append(optimizer(self.critics[i_critic].parameters(), lr = critic_lr))

            hard_update(self.critics_target[i_critic], self.critics[i_critic])
            
        self.entities.extend(self.critics)
        self.entities.extend(self.critics_target)
        self.entities.extend(self.critics_optim)

        # backward dynamics model for object actions and others 
        if self.object_backward is not None:
            for backward in self.object_backward:
                self.entities.append(backward)

        # Learnt Q function for object and other 
        if self.object_Qfunc is not None:
            for Qfunc in self.object_Qfunc:
                self.entities.append(Qfunc)

        # Learnt policy for object and others 
        if self.object_policy is not None:
            for policy in self.object_policy:
                self.entities.append(policy)

        if reward_fun is not None:
            self.get_obj_reward = reward_fun
        else:
            self.get_obj_reward = self.reward_fun

        # backward dynamics model for object actions
        self.backward = BackwardDyn(observation_space, action_space[0]).to(device)
        self.backward_optim = optimizer(self.backward.parameters(), lr = critic_lr)
        self.entities.append(self.backward)
        self.entities.append(self.backward_optim)

        self.backward_otw = BackwardDyn(observation_space, action_space[0]).to(device)
        self.backward_otw_optim = optimizer(self.backward_otw.parameters(), lr = critic_lr)
        self.entities.append(self.backward_otw)
        self.entities.append(self.backward_otw_optim)

        print('seperaate Qs')
        print('using actual actions')

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
        mu = mu.clamp(int(self.action_space[self.agent_id].low[0]), int(self.action_space[self.agent_id].high[0]))

        return mu

    def update_parameters(self, batch, normalizer=None):

        observation_space = self.observation_space - K.tensor(batch['g'], dtype=self.dtype, device=self.device).shape[1]
        action_space = self.action_space[0].shape[0]

        V = K.zeros((len(batch['o']), 1), dtype=self.dtype, device=self.device)
        
        s1 = K.cat([K.tensor(batch['o'], dtype=self.dtype, device=self.device)[:, 0:observation_space],
                    K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)
        
        if self.n_objects[0] <= 1:
            s2 = K.cat([K.tensor(batch['o'], dtype=self.dtype, device=self.device)[:, observation_space:],
                        K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)
        else:
            s2 = get_obj_obs(K.tensor(batch['o'], dtype=self.dtype, device=self.device)[:, observation_space:],
                             K.tensor(batch['g'], dtype=self.dtype, device=self.device), 
                             n_object=self.n_objects[0])

        a1 = K.tensor(batch['u'], dtype=self.dtype, device=self.device)[:, 0:action_space]
        a2 = K.tensor(batch['u'], dtype=self.dtype, device=self.device)[:, action_space:]

        s1_ = K.cat([K.tensor(batch['o_2'], dtype=self.dtype, device=self.device)[:, 0:observation_space],
                     K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)

        if self.n_objects[0] <= 1:
            s2_ = K.cat([K.tensor(batch['o_2'], dtype=self.dtype, device=self.device)[:, observation_space:],
                         K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)
        else:
            s2_ = get_obj_obs(K.tensor(batch['o_2'], dtype=self.dtype, device=self.device)[:, observation_space:],
                              K.tensor(batch['g'], dtype=self.dtype, device=self.device), 
                              n_object=self.n_objects[0])
        
        if normalizer[0] is not None:
            s1 = normalizer[0].preprocess(s1)
            s1_ = normalizer[0].preprocess(s1_)

        if normalizer[1] is not None:
            if self.n_objects[0] <= 1:
                s2 = normalizer[1].preprocess(s2)
                s2_ = normalizer[1].preprocess(s2_)
            else:
                for i_object in range(self.n_objects[0]):
                    s2[:,:,i_object] = normalizer[1].preprocess(s2[:,:,i_object])
                    s2_[:,:,i_object] = normalizer[1].preprocess(s2_[:,:,i_object])

        s3 = get_obj_obs(K.tensor(batch['o'], dtype=self.dtype, device=self.device)[:, 0:observation_space],
                         K.tensor(batch['g'], dtype=self.dtype, device=self.device), 
                         n_object=self.n_objects[0])
        s3 = s3[:,:,0:self.n_objects[1]]
        s3 = get_rob_obs(s3, self.n_objects[1])
        
        s3_ = get_obj_obs(K.tensor(batch['o_2'], dtype=self.dtype, device=self.device)[:, 0:observation_space],
                         K.tensor(batch['g'], dtype=self.dtype, device=self.device), 
                         n_object=self.n_objects[0])
        s3_ = s3_[:,:,0:self.n_objects[1]]
        s3_ = get_rob_obs(s3_, self.n_objects[1])

        if normalizer[2] is not None:
            s3 = normalizer[2].preprocess(s3)
            s3_ = normalizer[2].preprocess(s3_)

        s, s_, a = (s1, s1_, a1) if self.agent_id == 0 else (s2, s2_, a2)
        a_ = self.actors_target[0](s_)
    
        r_all = []
        if self.object_Qfunc is None:
            r = K.tensor(batch['r'], dtype=self.dtype, device=self.device).unsqueeze(1)
            r_all.append(r)
        else:
            r = K.tensor(batch['r'], dtype=self.dtype, device=self.device).unsqueeze(1)
            r_all.append(r)
            if len(self.object_Qfunc) > 1:
                # estimated actions
                #r_intr.append(self.get_obj_reward(s3, s3_, index=1))
                # actual actions
                r_all.append(self.get_obj_reward(s3, s3_, index=1, action=a1))
            for i_object in range(self.n_objects[2], self.n_objects[0]):
                r_all.append(self.get_obj_reward(s2[:,:,i_object], s2_[:,:,i_object], index=0))

        # first critic for main rewards
        Q = self.critics[0](s, a)       
        V = self.critics_target[0](s_, a_).detach()

        target_Q = (V * self.gamma) + r_all[0]
        target_Q = target_Q.clamp(self.clip_Q_neg, 0.)

        loss_critic = self.loss_func(Q, target_Q)

        self.critics_optim[0].zero_grad()
        loss_critic.backward()
        self.critics_optim[0].step()
        
        #r_sum = K.zeros_like(r_all[0])
        #for i_object in range(self.n_objects[2], self.n_objects[1]):
        #    r_sum += r_all[i_object+2]
        #r_mask = r_sum < -0.0001
        #r_all[1] *= K.tensor(r_mask, dtype=r_all[1].dtype, device=r_all[1].device)

        # other critics for intrinsic
        for i_critic in range(1,self.n_aux_critics+1):
            Q = self.critics[i_critic](s, a)       
            V = self.critics_target[i_critic](s_, a_).detach()

            target_Q = (V * self.gamma) + r_all[i_critic]
            target_Q = target_Q.clamp(self.clip_Q_neg, 0.)

            loss_critic = self.loss_func(Q, target_Q)

            self.critics_optim[i_critic].zero_grad()
            loss_critic.backward()
            self.critics_optim[i_critic].step()

        # actor update
        a = self.actors[0](s)

        loss_actor = -self.critics[0](s, a).mean()
        for i_critic in range(1,self.n_aux_critics+1):
            loss_actor += -self.critics[i_critic](s, a).mean()

        if self.regularization:
            loss_actor += (self.actors[0](s)**2).mean()*1

        self.actors_optim[0].zero_grad()        
        loss_actor.backward()
        self.actors_optim[0].step()
                
        return loss_critic.item(), loss_actor.item()

    def update_target(self):

        soft_update(self.actors_target[0], self.actors[0], self.tau)
        soft_update(self.critics_target[0], self.critics[0], self.tau)
        for i_critic in range(1,self.n_aux_critics+1):
            soft_update(self.critics_target[i_critic], self.critics[i_critic], self.tau)

    def reward_fun(self, state, next_state, index=0, action=None):
        with K.no_grad():
            if action is None:
                action = self.object_backward[index](state.to(self.device), next_state.to(self.device))
            opt_action = self.object_policy[index](state.to(self.device))

            reward = self.object_Qfunc[index](state.to(self.device), action) - self.object_Qfunc[index](state.to(self.device), opt_action)
        return reward.clamp(min=-1.0, max=0.0)

    def update_backward(self, batch, normalizer=None):

        observation_space = self.observation_space - K.tensor(batch['g'], dtype=self.dtype, device=self.device).shape[1]
        action_space = self.action_space[0].shape[0]
        
        s1 = K.cat([K.tensor(batch['o'], dtype=self.dtype, device=self.device)[:, 0:observation_space],
                    K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)

        a1 = K.tensor(batch['u'], dtype=self.dtype, device=self.device)[:, 0:action_space]

        s1_ = K.cat([K.tensor(batch['o_2'], dtype=self.dtype, device=self.device)[:, 0:observation_space],
                     K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)

        if normalizer[0] is not None:
            s1 = normalizer[0].preprocess(s1)
            s1_ = normalizer[0].preprocess(s1_)

        a1_pred = self.backward(s1, s1_)

        loss_backward = self.loss_func(a1_pred, a1)

        self.backward_optim.zero_grad()
        loss_backward.backward()
        self.backward_optim.step()

        return loss_backward.item()

    def update_backward_otw(self, batch, normalizer=None):

        observation_space = self.observation_space - K.tensor(batch['g'], dtype=self.dtype, device=self.device).shape[1]
        action_space = self.action_space[0].shape[0]
        
        s1 = K.cat([K.tensor(batch['o'], dtype=self.dtype, device=self.device)[:, 0:observation_space],
                    K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)

        a1 = K.tensor(batch['u'], dtype=self.dtype, device=self.device)[:, 0:action_space]

        s1_ = K.cat([K.tensor(batch['o_2'], dtype=self.dtype, device=self.device)[:, 0:observation_space],
                     K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)

        if normalizer[0] is not None:
            s1 = normalizer[0].preprocess(s1)
            s1_ = normalizer[0].preprocess(s1_)

        a1_pred = self.backward_otw(s1, s1_)

        loss_backward_otw = self.loss_func(a1_pred, a1)

        self.backward_otw_optim.zero_grad()
        loss_backward_otw.backward()
        self.backward_otw_optim.step()

        return loss_backward_otw.item()