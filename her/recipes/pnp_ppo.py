import numpy as np
import scipy as sc
import time

import torch as K

from her.utils import get_params as get_params, running_mean
from her.main_ppo import init, run


device = K.device("cuda" if K.cuda.is_available() else "cpu")
dtype = K.float32

import matplotlib.pyplot as plt

env_name = 'FetchPickAndPlaceMulti-v1'
model_name = 'DDPG_BD'

exp_args=['--env_id', env_name,
          '--exp_id', model_name + '_dgx1_foo760_' + str(0),
          '--random_seed', str(0), 
          '--agent_alg', model_name,
          '--verbose', '2',
          '--render', '0',
          '--episode_length', '50',
          '--gamma', '0.98',
          '--n_episodes', '20',
          '--n_cycles', '50',
          '--batch_size', '256',
          '--reward_normalization', 'False', 
          '--obj_action_type', '012',
          '--max_nb_objects', '1',
          '--observe_obj_grp', 'True']


config = get_params(args=exp_args)
model, experiment_args = init(config, agent='object', her=True, 
                              object_Qfunc=None, 
                              backward_dyn=None,
                              object_policy=None)
env, memory, noise, config, normalizer, agent_id = experiment_args

path = '~/Dropbox/Jupyter/notebooks/Reinforcement_Learning/models/obj/obj_model_norm_slide_pnp/'
model.critics[0].load_state_dict(K.load(path + 'object_Qfunc.pt'))
model.backward.load_state_dict(K.load(path + 'backward_dyn.pt'))
model.actors[0].load_state_dict(K.load(path + 'object_policy.pt'))

import pickle
with open(path + 'normalizer.pkl', 'rb') as file:
    normalizer = pickle.load(file)

experiment_args = (env, memory, noise, config, normalizer, agent_id)  

model_name = 'PPO_BD'
exp_args2=['--env_id', env_name,
          '--exp_id', model_name + '_dgx1_foo861_' + str(0),
          '--random_seed', str(0), 
          '--agent_alg', model_name,
          '--verbose', '2',
          '--render', '0',
          '--episode_length', '50',
          '--gamma', '0.98',
          '--n_episodes', '50',
          '--n_cycles', '50',
          '--n_rollouts', '38',
          '--n_batches', '4',
          '--batch_size', '256',
          '--reward_normalization', 'False', 
          '--ai_object_rate', '0.0',
          '--obj_action_type', '012',
          '--max_nb_objects', '1',
          '--observe_obj_grp', 'True',
          '--plcy_lr', '3e-4',
          '--crtc_lr', '3e-4',
          '--ppo_epoch', '3',
          '--entropy_coef', '0.00',
          '--clip_param', '0.1',
          '--use_gae', "True",]

config2 = get_params(args=exp_args2)
model2, experiment_args2 = init(config2, agent='robot', her=False, 
                                object_Qfunc=model.critics[0], 
                                backward_dyn=model.backward,
                                object_policy=model.actors[0]
                               )
env2, memory2, noise2, config2, normalizer2, agent_id2 = experiment_args2
normalizer2[1] = normalizer[1]
experiment_args2 = (env2, memory2, noise2, config2, normalizer2, agent_id2)

monitor2 = run(model2, experiment_args2, train=True)

path = '~/Dropbox/Jupyter/notebooks/Reinforcement_Learning/models/recent/rob_model_PnP_v4P_Norm_Slide_Clipped_Both_Masked_PlusR/'
K.save(model2.critics[0].state_dict(), path + 'robot_Qfunc.pt')
K.save(model2.actors[0].state_dict(), path + 'robot_policy.pt')
K.save(model2.object_Qfunc.state_dict(), path + 'object_Qfunc.pt')
K.save(model2.backward.state_dict(), path + 'backward_dyn.pt')
K.save(model2.object_policy.state_dict(), path + 'object_policy.pt')
import pickle
with open(path + 'normalizer.pkl', 'wb') as file:
    pickle.dump(normalizer2, file)

np.save('~/Dropbox/Jupyter/notebooks/Reinforcement_Learning/monitors/recent/monitor_FetchPickandPlaceMulti-v1_Rew_v4P_Norm_Slide_Clipped_Both_Masked_PlusR.npy', monitor2)

