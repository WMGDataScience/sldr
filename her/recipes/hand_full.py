import numpy as np
import scipy as sc
import time

import torch as K

from her.utils import get_params as get_params, running_mean, get_exp_params
from her.main import init, run
from her.main_q_rt import init as init_q
from her.main_q_rt import run as run_q
import matplotlib.pyplot as plt

import os
import pickle
import sys

K.set_num_threads(1)

filepath='/jmain01/home/JAD022/grm01/oxk28-grm01/Dropbox/Jupyter/notebooks/Reinforcement_Learning/'
#filepath='/home/ok18/Jupyter/notebooks/Reinforcement_Learning/'
os.chdir(filepath)

device = K.device("cuda" if K.cuda.is_available() else "cpu")
dtype = K.float32

exp_config = get_exp_params(sys.argv[1:])

ENV = exp_config['env']

if exp_config['shaped'] == 'True':
    use_dist = True
else:
    use_dist = False

suffix = 'Dense' if use_dist else ''

if ENV == 'Egg':
     env_name_0 = ['HandManipulateEggRotateMulti{}-v0'.format(suffix)]
     env_name_1 = ['HandManipulateEggTranslateMulti{}-v0'.format(suffix)]
     env_name_2 = ['HandManipulateEggFullMulti{}-v0'.format(suffix)]
elif ENV == 'Block':
     env_name_0 = ['HandManipulateBlockRotateXYZMulti{}-v0'.format(suffix)]
     env_name_1 = ['HandManipulateBlockTranslateMulti{}-v0'.format(suffix)]
     env_name_2 = ['HandManipulateBlockFullMulti{}-v0'.format(suffix)]
elif ENV == 'Pen':
     env_name_0 = ['HandManipulatePenRotateXYZMulti{}-v0'.format(suffix)]
     env_name_1 = ['HandManipulatePenTranslateMulti{}-v0'.format(suffix)]
     env_name_2 = ['HandManipulatePenFullMulti{}-v0'.format(suffix)]
elif ENV == 'All':
     env_name_0 = ['HandManipulateEggRotateMulti{}-v0'.format(suffix), 'HandManipulateBlockRotateXYZMulti{}-v0'.format(suffix), 'HandManipulatePenRotateXYZMulti{}-v0'.format(suffix)]
     env_name_1 = ['HandManipulateEggTranslateMulti{}-v0'.format(suffix), 'HandManipulateBlockTranslateMulti{}-v0'.format(suffix), 'HandManipulatePenTranslateMulti{}-v0'.format(suffix)]
     env_name_2 = ['HandManipulateEggFullMulti{}-v0'.format(suffix), 'HandManipulateBlockFullMulti{}-v0'.format(suffix), 'HandManipulatePenFullMulti{}-v0'.format(suffix)]


if exp_config['use_her'] == 'True':
    use_her = True
    print("training with HER")
else:
    use_her = False
    print("training without HER")

model_name = 'DDPG_BD'

for i_env in range(len(env_name_0)):

    if env_name_0[i_env].replace('Dense','') == 'HandManipulatePenRotateXYZMulti-v0':
        gamma = 0.98
        clip_Q_neg = -100
    else:
        gamma = 0.99
        clip_Q_neg = -100

    for i_exp in range(int(exp_config['start_n_exp']), int(exp_config['n_exp'])):
        if exp_config['obj_rew'] == 'True':
        ####################### loading object ###########################
            exp_args_0 = ['--env_id', env_name_0[i_env],
                    '--exp_id', model_name + '_fooobj_' + str(0),
                    '--random_seed', str(0), 
                    '--agent_alg', model_name,
                    '--verbose', '2',
                    '--render', '0',
                    '--gamma', '0.99',
                    '--n_episodes', '50',
                    '--n_cycles', '50',
                    '--n_rollouts', '38',
                    '--n_test_rollouts', '38',
                    '--n_envs', '1',
                    '--n_batches', '40',
                    '--batch_size', '4864',
                    '--obj_action_type', '0123456',
                    '--max_nb_objects', '1',
                    '--observe_obj_grp', 'False',
                    '--rob_policy', '01']

            config_0 = get_params(args=exp_args_0)
            model_0, experiment_args_0 = init(config_0, agent='object', her=True, 
                                                    object_Qfunc=None, 
                                                    backward_dyn=None,
                                                    object_policy=None)
            env_0, memory_0, noise_0, config_0, normalizer_0, agent_id_0 = experiment_args_0

            #loading the object model
            if env_name_0[i_env].replace('Dense','') == 'HandManipulateEggRotateMulti-v0':
                path = './models_paper/obj/egg_rotate_7d_50ep/'
            elif env_name_0[i_env].replace('Dense','') == 'HandManipulateBlockRotateXYZMulti-v0':
                path = './models_paper/obj/block_rotate_7d_50ep/'
            elif env_name_0[i_env].replace('Dense','') == 'HandManipulatePenRotateXYZMulti-v0':
                path = './models_paper/obj/pen_rotate_7d_50ep/'

            print('loading object model for rotation')
            print(path)
            model_0.critics[0].load_state_dict(K.load(path + 'object_Qfunc.pt'))
            model_0.actors[0].load_state_dict(K.load(path + 'object_policy.pt'))
            model_0.backward.load_state_dict(K.load(path + 'backward_dyn.pt'))
            with open(path + 'normalizer.pkl', 'rb') as file:
                normalizer_0 = pickle.load(file)

            experiment_args_0 = (env_0, memory_0, noise_0, config_0, normalizer_0, agent_id_0)


            exp_args_1 = ['--env_id', env_name_1[i_env],
                    '--exp_id', model_name + '_fooobj_' + str(0),
                    '--random_seed', str(0), 
                    '--agent_alg', model_name,
                    '--verbose', '2',
                    '--render', '0',
                    '--gamma', '0.99',
                    '--n_episodes', '50',
                    '--n_cycles', '50',
                    '--n_rollouts', '38',
                    '--n_test_rollouts', '38',
                    '--n_envs', '1',
                    '--n_batches', '40',
                    '--batch_size', '4864',
                    '--obj_action_type', '0123456',
                    '--max_nb_objects', '1',
                    '--observe_obj_grp', 'False',
                    '--rob_policy', '01']

            config_1 = get_params(args=exp_args_1)
            model_1, experiment_args_1 = init(config_1, agent='object', her=True, 
                                                    object_Qfunc=None, 
                                                    backward_dyn=None,
                                                    object_policy=None)
            env_1, memory_1, noise_1, config_1, normalizer_1, agent_id_1 = experiment_args_1

            #loading the object model
            if env_name_0[i_env].replace('Dense','') == 'HandManipulateEggRotateMulti-v0':
                path = './models_paper/obj/egg_translate_7d_20ep/'
            elif env_name_0[i_env].replace('Dense','') == 'HandManipulateBlockRotateXYZMulti-v0':
                path = './models_paper/obj/block_translate_7d_20ep/'
            elif env_name_0[i_env].replace('Dense','') == 'HandManipulatePenRotateXYZMulti-v0':
                path = './models_paper/obj/pen_translate_7d_20ep/'

            print('loading object model for translation')
            print(path)
            model_1.critics[0].load_state_dict(K.load(path + 'object_Qfunc.pt'))
            model_1.actors[0].load_state_dict(K.load(path + 'object_policy.pt'))
            model_1.backward.load_state_dict(K.load(path + 'backward_dyn.pt'))
            with open(path + 'normalizer.pkl', 'rb') as file:
                normalizer_1 = pickle.load(file)

            experiment_args_1 = (env_1, memory_1, noise_1, config_1, normalizer_1, agent_id_1)
            
            obj_rew = True
            object_Qfunc = (model_0.critics[0], model_1.critics[0])
            object_policy = (model_0.actors[0], model_1.actors[0])
            backward_dyn = (model_0.backward, model_1.backward)
            init_2 = init_q   
            run_2 = run_q
            print("training with object based rewards")
        ####################### loading object ###########################
        elif exp_config['obj_rew'] == 'False':
            obj_rew = False
            object_Qfunc = None
            object_policy = None  
            backward_dyn = None
            init_2 = init
            run_2 = run
            print("training without object based rewards")

        ####################### training robot ###########################  
        exp_args_2=['--env_id', env_name_2[i_env],
                '--exp_id', model_name + '_foorob_' + str(i_exp),
                '--random_seed', str(i_exp), 
                '--agent_alg', model_name,
                '--verbose', '2',
                '--render', '0',
                '--gamma', str(gamma),
                '--clip_Q_neg', '-100',
                '--n_episodes', '200',
                '--n_cycles', '50',
                '--n_rollouts', '38',
                '--n_test_rollouts', '380',
                '--n_envs', '38',
                '--n_batches', '40',
                '--batch_size', '4864',
                '--obj_action_type', '0123456',
                '--max_nb_objects', '1',
                '--observe_obj_grp', 'False',
                ]

        config_2 = get_params(args=exp_args_2)
        model_2, experiment_args_2 = init_2(config_2, agent='robot', her=use_her, 
                                        object_Qfunc=object_Qfunc, 
                                        object_policy=object_policy,
                                        backward_dyn=backward_dyn,
                                    )
        env_2, memory_2, noise_2, config_2, normalizer_2, running_rintr_mean_2 = experiment_args_2
        if obj_rew:
            normalizer_2[1] = normalizer_0[1]
            normalizer_2[2] = normalizer_1[1]
        experiment_args_2 = (env_2, memory_2, noise_2, config_2, normalizer_2, running_rintr_mean_2)

        monitor_2, bestmodel = run_2(model_2, experiment_args_2, train=True)

        rob_name = env_name_2[i_env].replace('Dense','')
        if obj_rew:
            if use_her:
                rob_name = rob_name + '_DDPG_OURS_HER_'
            else:
                rob_name = rob_name + '_DDPG_OURS_'
        else:
            if use_her:
                rob_name = rob_name + '_DDPG_HER_'
            else:
                rob_name = rob_name + '_DDPG_'

        if use_dist:
            rob_name = rob_name + 'DIST_'

        path = './models_paper/batch3/' + rob_name + str(i_exp)
        try:  
            os.makedirs(path)
        except OSError:  
            print ("Creation of the directory %s failed" % path)
        else:  
            print ("Successfully created the directory %s" % path)

        K.save(model_2.critics[0].state_dict(), path + '/robot_Qfunc.pt')
        K.save(model_2.actors[0].state_dict(), path + '/robot_policy.pt')

        K.save(bestmodel[0], path + '/robot_Qfunc_best.pt')
        K.save(bestmodel[1], path + '/robot_policy_best.pt')
        
        with open(path + '/normalizer.pkl', 'wb') as file:
            pickle.dump(normalizer_2, file)

            with open(path + '/normalizer_best.pkl', 'wb') as file:
                pickle.dump(bestmodel[2], file)

        path = './models_paper/batch3/monitor_' + rob_name + str(i_exp) + '.npy'
        np.save(path, monitor_2)

