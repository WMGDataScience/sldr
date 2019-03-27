import numpy as np
import scipy as sc
import time

import torch as K

from her.utils import get_params as get_params, running_mean, get_exp_params
from her.main import init, run
from her.main_ppo import init as init_ppo
from her.main_ppo import run as run_ppo
import matplotlib.pyplot as plt

import os
import pickle

filepath='/jmain01/home/JAD022/grm01/oxk28-grm01/Dropbox/Jupyter/notebooks/Reinforcement_Learning/'
os.chdir(filepath)

device = K.device("cuda" if K.cuda.is_available() else "cpu")
dtype = K.float32

exp_config = get_exp_params()
####################### loading object ###########################
if exp_config['env'] is 'Push':
    env_name = 'FetchPickAndPlaceMulti-v1'
elif exp_config['env'] is 'PnP':
    env_name = 'FetchPushMulti-v1'

model_name = 'DDPG_BD'
exp_args=['--env_id', env_name,
          '--exp_id', model_name + '_fooobj_' + str(0),
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

#loading the object model
if exp_config['env'] is 'Push':
    path = './models/obj/obj_model_norm_slide/'
elif exp_config['env'] is 'PnP':
    path = './models/obj/obj_model_norm_slide_pnp/'

model.critics[0].load_state_dict(K.load(path + 'object_Qfunc.pt'))
model.backward.load_state_dict(K.load(path + 'backward_dyn.pt'))
model.actors[0].load_state_dict(K.load(path + 'object_policy.pt'))
with open(path + 'normalizer.pkl', 'rb') as file:
    normalizer = pickle.load(file)

experiment_args = (env, memory, noise, config, normalizer, agent_id)  
####################### loading object ###########################  

####################### training robot ###########################  
masked_with_r = exp_config['masked_with_r']
if exp_config['use_her'] is 'True':
    use_her = True
else:
    use_her = False

if exp_config['rob_model'] is 'DDPG':
    model_name = 'DDPG_BD'
    exp_args2=['--env_id', env_name,
            '--exp_id', model_name + '_foorob_' + str(0),
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
            '--observe_obj_grp', 'True',
            '--masked_with_r', masked_with_r,
            '--n_test_rollouts', '100'
            ]

    config2 = get_params(args=exp_args2)
    model2, experiment_args2 = init(config2, agent='robot', her=use_her, 
                                    object_Qfunc=model.critics[0], 
                                    backward_dyn=model.backward,
                                    object_policy=model.actors[0]
                                )
    env2, memory2, noise2, config2, normalizer2, agent_id2 = experiment_args2
    normalizer2[1] = normalizer[1]
    experiment_args2 = (env2, memory2, noise2, config2, normalizer2, agent_id2)

    monitor2 = run(model2, experiment_args2, train=True)

elif exp_config['rob_model'] is 'PPO':
    model_name = 'PPO_BD'
    exp_args2=['--env_id', env_name,
            '--exp_id', model_name + '_foorob_' + str(0),
            '--random_seed', str(0), 
            '--agent_alg', model_name,
            '--verbose', '2',
            '--render', '1',
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
            '--masked_with_r', masked_with_r,
            '--n_test_rollouts', '100',
            '--plcy_lr', '3e-4',
            '--crtc_lr', '3e-4',
            '--ppo_epoch', '3',
            '--entropy_coef', '0.00',
            '--clip_param', '0.1',
            '--use_gae', "False"
    ]

    config2 = get_params(args=exp_args2)
    model2, experiment_args2 = init_ppo(config2, agent='robot', her=False, 
                                    object_Qfunc=model.critics[0], 
                                    backward_dyn=model.backward,
                                    object_policy=model.actors[0]
                                )
    env2, memory2, noise2, config2, normalizer2, agent_id2 = experiment_args2
    normalizer2[1] = normalizer[1]
    experiment_args2 = (env2, memory2, noise2, config2, normalizer2, agent_id2)

    monitor2 = run_ppo(model2, experiment_args2, train=True)

rob_name = exp_config['env']
if exp_config['rob_model'] is 'PPO':
    rob_name = rob_name + '_v4P_'
elif exp_config['rob_model'] is 'DDPG':
    if use_her:
        rob_name = rob_name + '_v4H_'
    else:
        rob_name = rob_name + '_v4_'
rob_name = rob_name + 'Norm_Slide_Clipped_Both_'
if masked_with_r:
    rob_name = rob_name + 'Masked_PlusR'
else:
    rob_name = rob_name + 'PlusR'

path = './models/recent/rob_model_' + rob_name
try:  
    os.makedirs(path)
except OSError:  
    print ("Creation of the directory %s failed" % path)
else:  
    print ("Successfully created the directory %s" % path)

K.save(model2.critics[0].state_dict(), path + 'robot_Qfunc.pt')
K.save(model2.actors[0].state_dict(), path + 'robot_policy.pt')
K.save(model2.object_Qfunc.state_dict(), path + 'object_Qfunc.pt')
K.save(model2.backward.state_dict(), path + 'backward_dyn.pt')
K.save(model2.object_policy.state_dict(), path + 'object_policy.pt')
with open(path + 'normalizer.pkl', 'wb') as file:
    pickle.dump(normalizer2, file)

path = './monitors/recent/monitor_' + rob_name + '.npy'
np.save(path, monitor2)

