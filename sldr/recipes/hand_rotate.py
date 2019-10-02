import numpy as np
import scipy as sc
import time

import torch as K

from sldr.utils import get_params as get_params, running_mean, get_exp_params
from sldr.main import init, run
from sldr.main_q import init as init_q
from sldr.main_q import run as run_q
from sldr.main_q_rnd import init as init_q_rnd
from sldr.main_q_rnd import run as run_q_rnd
import matplotlib.pyplot as plt

import os
import pickle
import sys
import sldr

#for compatibility to the name change
sys.modules['her.experience'] = sldr.experience

K.set_num_threads(1)

device = K.device("cuda" if K.cuda.is_available() else "cpu")
dtype = K.float32

exp_config = get_exp_params(sys.argv[1:])
filepath = exp_config['filepath'] 
os.chdir(filepath)

if exp_config['shaped'] == 'True':
    use_dist = True
else:
    use_dist = False

suffix = 'Dense' if use_dist else ''

if exp_config['env'] == 'Egg':
     env_name_list = ['HandManipulateEggRotateMulti{}-v0'.format(suffix)]
elif exp_config['env'] == 'Block':
     env_name_list = ['HandManipulateBlockRotateXYZMulti{}-v0'.format(suffix)]
elif exp_config['env'] == 'Pen':
     env_name_list = ['HandManipulatePenRotateMulti{}-v0'.format(suffix)]
elif exp_config['env'] == 'All':
     env_name_list = ['HandManipulateEggRotateMulti{}-v0'.format(suffix), 'HandManipulateBlockRotateXYZMulti{}-v0'.format(suffix), 'HandManipulatePenRotateMulti{}-v0'.format(suffix)]

if exp_config['use_her'] == 'True':
    use_her = True
else:
    use_her = False

if exp_config['use_rnd'] == 'True':
    use_rnd = True
else:
    use_rnd = False

for env_name in env_name_list:

    for i_exp in range(int(exp_config['start_n_exp']), int(exp_config['n_exp'])):

        if use_her:
            print("training with HER")
        else:
            print("training without HER")

        if env_name.replace('Dense','') == 'HandManipulatePenRotateMulti-v0':
            gamma = 0.98
            clip_Q_neg = -100
        else:
            gamma = 0.99
            clip_Q_neg = -100

        if exp_config['obj_rew'] == 'True':
        ####################### loading object ###########################
            print("training with object based rewards")
            model_name = 'DDPG_BD'
            exp_args=['--env_id', env_name,
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

            config = get_params(args=exp_args)
            model, experiment_args = init(config, agent='object', her=True, 
                                        object_Qfunc=None, 
                                        backward_dyn=None,
                                        object_policy=None)
            env, memory, noise, config, normalizer, agent_id = experiment_args

            #loading the object model
            if env_name.replace('Dense','') == 'HandManipulateEggRotateMulti-v0':
                path = './models_paper/obj/egg_rotate_7d_50ep/'
            elif env_name.replace('Dense','') == 'HandManipulateBlockRotateXYZMulti-v0':
                path = './models_paper/obj/block_rotate_7d_50ep/'
            elif env_name.replace('Dense','') == 'HandManipulatePenRotateMulti-v0':
                path = './models_paper/obj/pen_rotate_7d_50ep/'

            print('loading object model')
            print(path)
            model.critics[0].load_state_dict(K.load(path + 'object_Qfunc.pt'))
            model.actors[0].load_state_dict(K.load(path + 'object_policy.pt'))
            model.backward.load_state_dict(K.load(path + 'backward_dyn.pt'))
            with open(path + 'normalizer.pkl', 'rb') as file:
                normalizer = pickle.load(file)

            experiment_args = (env, memory, noise, config, normalizer, agent_id)
            
            obj_rew = True
            object_Qfunc = model.critics[0]
            object_policy = model.actors[0]  
            backward_dyn = model.backward
            init_2 = init_q   
            run_2 = run_q
        ####################### loading object ###########################
        elif exp_config['obj_rew'] == 'False':
            obj_rew = False
            object_Qfunc = None
            object_policy = None  
            backward_dyn = None
            print("training without object based rewards")
            if use_rnd:
                init_2 = init_q_rnd
                run_2 = run_q_rnd
                print("training with RND rewards")
            else:
                init_2 = init
                run_2 = run

        ####################### training robot ###########################  
        model_name = 'DDPG_BD'
        exp_args2=['--env_id', env_name,
                '--exp_id', model_name + '_foorob_' + str(i_exp),
                '--random_seed', str(i_exp), 
                '--agent_alg', model_name,
                '--verbose', '2',
                '--render', '0',
                '--gamma', str(gamma),
                '--clip_Q_neg', '-100',
                '--n_episodes', '100',
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

        config2 = get_params(args=exp_args2)
        model2, experiment_args2 = init_2(config2, agent='robot', her=use_her, 
                                        object_Qfunc=object_Qfunc, 
                                        object_policy=object_policy,
                                        backward_dyn=backward_dyn,
                                    )
        env2, memory2, noise2, config2, normalizer2, running_rintr_mean2 = experiment_args2
        if obj_rew:
            normalizer2[1] = normalizer[1]
        experiment_args2 = (env2, memory2, noise2, config2, normalizer2, running_rintr_mean2)

        monitor2, bestmodel = run_2(model2, experiment_args2, train=True)

        rob_name = env_name.replace('Dense','')
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

        if use_rnd:
            rob_name = rob_name + 'RND_'

        if use_dist:
            rob_name = rob_name + 'DIST_'

        path = './models_paper/batch3/' + rob_name + str(i_exp)

        try:  
            os.makedirs(path)
        except OSError:  
            print ("Creation of the directory %s failed" % path)
        else:  
            print ("Successfully created the directory %s" % path)

        K.save(model2.critics[0].state_dict(), path + '/robot_Qfunc.pt')
        K.save(model2.actors[0].state_dict(), path + '/robot_policy.pt')

        K.save(bestmodel[0], path + '/robot_Qfunc_best.pt')
        K.save(bestmodel[1], path + '/robot_policy_best.pt')

        if obj_rew:
            K.save(model2.object_Qfunc.state_dict(), path + '/object_Qfunc.pt')
            K.save(model2.backward.state_dict(), path + '/backward_dyn.pt')
            K.save(model2.object_policy.state_dict(), path + '/object_policy.pt')
        
        with open(path + '/normalizer.pkl', 'wb') as file:
            pickle.dump(normalizer2, file)

        with open(path + '/normalizer_best.pkl', 'wb') as file:
            pickle.dump(bestmodel[2], file)

        path = './models_paper/batch3/monitor_' + rob_name  + str(i_exp) + '.npy'
        np.save(path, monitor2)

