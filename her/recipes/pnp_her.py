import numpy as np
import scipy as sc
import time

import torch as K

from her.utils import get_params as get_params, running_mean
from her.main import init, run

import os
filepath='/jmain01/home/JAD022/grm01/oxk28-grm01/Dropbox/Jupyter/notebooks/Reinforcement_Learning/'
os.chdir(filepath)

device = K.device("cuda" if K.cuda.is_available() else "cpu")
dtype = K.float32

import matplotlib.pyplot as plt

env_name = 'FetchPickAndPlaceMulti-v1'
model_name = 'DDPG_BD'

exp_args2=['--env_id', env_name,
          '--exp_id', model_name + '_dgx1_foo761_' + str(0),
          '--random_seed', str(0), 
          '--agent_alg', model_name,
          '--verbose', '2',
          '--render', '0',
          '--episode_length', '50',
          '--gamma', '0.98',
          '--n_episodes', '50',
          '--n_cycles', '50',
          '--batch_size', '256',
          '--reward_normalization', 'False', 
          '--ai_object_rate', '0.0',
          '--obj_action_type', '012',
          '--max_nb_objects', '1',
          '--observe_obj_grp', 'True']

config2 = get_params(args=exp_args2)
model2, experiment_args2 = init(config2, agent='robot', her=True, 
                                object_Qfunc=None, 
                                backward_dyn=None,
                                object_policy=None
                               )
env2, memory2, noise2, config2, normalizer2, agent_id2 = experiment_args2

monitor2 = run(model2, experiment_args2, train=True)

path = './models/recent/rob_model_PnP_HER_Norm_Slide/'
K.save(model2.critics[0].state_dict(), path + 'robot_Qfunc.pt')
K.save(model2.actors[0].state_dict(), path + 'robot_policy.pt')
K.save(model2.object_Qfunc.state_dict(), path + 'object_Qfunc.pt')
K.save(model2.backward.state_dict(), path + 'backward_dyn.pt')
K.save(model2.object_policy.state_dict(), path + 'object_policy.pt')
import pickle
with open(path + 'normalizer.pkl', 'wb') as file:
    pickle.dump(normalizer2, file)

np.save('./monitors/recent/monitor_FetchPickandPlaceMulti-v1_HER_Norm_Slide.npy', monitor2)

