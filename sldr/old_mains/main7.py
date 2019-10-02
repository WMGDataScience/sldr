import numpy as np
import scipy as sc
import time
import imageio

import torch as K
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym_wmgds as gym

from sldr.algorithms3 import MADDPG, DDPG, MADDPG_R, MADDPG_RAE
from sldr.experience import Normalizer
from sldr.replay_buffer import ReplayBuffer_v2 as ReplayBuffer
from sldr.her_sampler import make_sample_her_transitions_v2 as make_sample_her_transitions
from sldr.exploration import Noise
from sldr.utils import Saver, Summarizer, get_params, running_mean
from sldr.agents.basic import Actor 
from sldr.agents.basic import Critic

import pdb

import matplotlib
import matplotlib.pyplot as plt

device = K.device("cuda" if K.cuda.is_available() else "cpu")
dtype = K.float32

def init(config):
    
    if config['resume'] != '':
        resume_path = config['resume']
        saver = Saver(config)
        config, start_episode, save_dict = saver.resume_ckpt()
        config['resume'] = resume_path
    else:
        start_episode = 0
        
    #hyperparameters
    ENV_NAME = config['env_id'] #'simple_spread'
    SEED = config['random_seed'] # 1

    GAMMA = config['gamma'] # 0.95
    TAU = config['tau'] # 0.01

    ACTOR_LR = config['plcy_lr'] # 0.01
    CRITIC_LR = config['crtc_lr'] # 0.01

    MEM_SIZE = config['buffer_length'] # 1000000 

    REGULARIZATION = config['regularization'] # True
    NORMALIZED_REWARDS = config['reward_normalization'] # True

    if (ENV_NAME == 'FetchStackMulti-v1') or (ENV_NAME == 'FetchPushMulti-v1'):
        env = gym.make(ENV_NAME, n_objects=config['max_nb_objects'], obj_action_type=config['obj_action_type'])
    else:
        env = gym.make(ENV_NAME)
    env.seed(SEED)

    if config['obj_action_type'] == 'all':
        n_actions = config['max_nb_objects'] * 7 + 4
    elif config['obj_action_type'] == 'slide_only':
        n_actions = config['max_nb_objects'] * 3 + 4
    elif config['obj_action_type'] == 'rotation_only':
        n_actions = config['max_nb_objects'] * 4 + 4

    observation_space = env.observation_space.spaces['observation'].shape[1] + env.observation_space.spaces['desired_goal'].shape[0]
    action_space = (gym.spaces.Box(-1., 1., shape=(4,), dtype='float32'),
                    gym.spaces.Box(-1., 1., shape=(n_actions-4,), dtype='float32'),
                    gym.spaces.Box(-1., 1., shape=(n_actions,), dtype='float32'))
    if env.action_space.low[0] == -1 and env.action_space.high[0] == 1:
        OUT_FUNC = K.tanh 
    elif env.action_space.low[0] == 0 and env.action_space.high[0] == 1:
        OUT_FUNC = K.sigmoid
    else:
        OUT_FUNC = K.sigmoid

    K.manual_seed(SEED)
    np.random.seed(SEED)
    
    if config['agent_alg'] == 'MADDPG':
        MODEL = MADDPG
    elif config['agent_alg'] == 'DDPG':
        MODEL = DDPG
    elif config['agent_alg'] == 'MADDPG_R':
        MODEL = MADDPG_R    
    elif config['agent_alg'] == 'MADDPG_RAE':
        MODEL = MADDPG_RAE            

    if config['verbose'] > 1:
        # utils
        summaries = (Summarizer(config['dir_summary_train'], config['port'], config['resume']),
                    Summarizer(config['dir_summary_test'], config['port'], config['resume']))
        saver = Saver(config)
    else:
        summaries = None
        saver = None

    #exploration initialization
    noise = (Noise(action_space[0].shape[0], sigma=0.2, eps=0.3),
             Noise(action_space[1].shape[0], sigma=0.05, eps=0.1))
    #noise = OUNoise(action_space.shape[0])

    #model initialization
    optimizer = (optim.Adam, (ACTOR_LR, CRITIC_LR)) # optimiser func, (actor_lr, critic_lr)
    loss_func = F.mse_loss
    model = MODEL(observation_space, action_space, optimizer, 
                  Actor, Critic, loss_func, GAMMA, TAU, out_func=OUT_FUNC,
                  discrete=False, regularization=REGULARIZATION, normalized_rewards=NORMALIZED_REWARDS)
    
    if config['resume'] != '':
        for i, param in enumerate(save_dict['model_params']):
            model.entities[i].load_state_dict(param)
    
    #memory initilization
    #memory = ReplayMemory(MEM_SIZE)
    def reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)
        
    sample_her_transitions = make_sample_her_transitions('future', 4, reward_fun)
    buffer_shapes = {
        'o' : (env._max_episode_steps, env.observation_space.spaces['observation'].shape[1]*2),
        'ag' : (env._max_episode_steps, env.observation_space.spaces['achieved_goal'].shape[0]),
        'g' : (env._max_episode_steps, env.observation_space.spaces['desired_goal'].shape[0]),
        'u' : (env._max_episode_steps-1, action_space[2].shape[0])
        }
    memory = ReplayBuffer(buffer_shapes, MEM_SIZE, env._max_episode_steps, sample_her_transitions)

    normalizer = (Normalizer(), Normalizer())

    experiment_args = (env, memory, noise, config, summaries, saver, start_episode, normalizer)
          
    return model, experiment_args

def rollout(env, model, noise, normalizer=None, render=False, ai_object=False):
    trajectories = []

    # monitoring variables
    episode_reward = 0
    frames = []
    
    env.env.ai_object = ai_object
    state_all = env.reset()

    for i_agent in range(2):
        trajectories.append([])
    
    for i_step in range(env._max_episode_steps):

        model.to_cpu()

        obs = K.tensor(state_all['observation'][0], dtype=K.float32).unsqueeze(0)
        goal = K.tensor(state_all['desired_goal'], dtype=K.float32).unsqueeze(0)

        obs_goal = K.cat([obs, goal], dim=-1)
        # Observation normalization
        if normalizer[0] is not None:
            obs_goal = normalizer[0].preprocess_with_update(obs_goal)

        action = model.select_action(obs_goal, 0, noise[0]).cpu().numpy().squeeze(0)

        # get the action of the object
        obs2 = K.tensor(state_all['observation'][1], dtype=K.float32).unsqueeze(0)
        
        obs2_goal = K.cat([obs2, goal], dim=-1)
        # Observation normalization
        if normalizer[1] is not None:
            obs2_goal = normalizer[1].preprocess_with_update(obs2_goal)

        action2 = model.select_action(obs2_goal, 1, noise[1]).cpu().numpy().squeeze(0)
        if not ai_object:
            action2 *= 0.00

        next_state_all, reward, done, info = env.step(np.concatenate((action, action2)))
        reward = K.tensor(reward, dtype=dtype).view(1,1)

        if not ai_object:
            #next_obs2 = K.tensor(next_state_all['observation'][1], dtype=K.float32).unsqueeze(0)
            #next_obs2_goal = K.cat([next_obs2, goal], dim=-1)
            #if normalizer[1] is not None:
            #    next_obs2_goal = normalizer[1].preprocess(next_obs2_goal)
            #action2 = model.estimate_obj_action(obs2_goal, next_obs2_goal).cpu().numpy().squeeze(0)
            ag = K.tensor(state_all['achieved_goal'], dtype=K.float32).unsqueeze(0)
            ag_2 = K.tensor(next_state_all['achieved_goal'], dtype=K.float32).unsqueeze(0)
            action2 = model.estimate_obj_action(ag, ag_2).cpu().numpy().squeeze(0)
        # for monitoring
        episode_reward += reward

        for i_agent in range(2):
            state = {
                'observation'   : state_all['observation'][i_agent],
                'achieved_goal' : state_all['achieved_goal'],
                'desired_goal'  : state_all['desired_goal']   
                }
            next_state = {
                'observation'   : next_state_all['observation'][i_agent],
                'achieved_goal' : next_state_all['achieved_goal'],
                'desired_goal'  : next_state_all['desired_goal']    
                }
            trajectories[i_agent].append((state.copy(), action if i_agent==0 else action2, reward, next_state.copy(), done))

        # Move to the next state
        state_all = next_state_all

        # Record frames
        if render:
            frames.append(env.render(mode='rgb_array')[0])

    obs, ags, goals, acts = [], [], [], []

    for trajectory in trajectories:
        obs.append([])
        ags.append([])
        goals.append([])
        acts.append([])
        for i_step in range(env._max_episode_steps):
            obs[-1].append(trajectory[i_step][0]['observation'])
            ags[-1].append(trajectory[i_step][0]['achieved_goal'])
            goals[-1].append(trajectory[i_step][0]['desired_goal'])
            if (i_step < env._max_episode_steps - 1): 
                acts[-1].append(trajectory[i_step][1])

    trajectories = {
        'o'    : np.concatenate(obs,axis=1)[np.newaxis,:], #np.asarray(obs),
        'ag'   : np.asarray(ags)[0:1,],
        'g'    : np.asarray(goals)[0:1,],
        'u'    : np.concatenate(acts,axis=1)[np.newaxis,:] #np.asarray(acts)
    }

    return trajectories, episode_reward, info['is_success'], frames

def run(model, experiment_args, train=True):

    total_time_start =  time.time()

    env, memory, noise, config, summaries, saver, start_episode, normalizer = experiment_args
    
    start_episode = start_episode if train else 0
    NUM_EPISODES = config['n_episodes'] if train else config['n_episodes_test'] 
    EPISODE_LENGTH = env._max_episode_steps #config['episode_length'] if train else config['episode_length_test'] 
    
    episode_reward_all = []
    episode_success_all = []
    critic_losses = []
    actor_losses = []
    critic_losses_obj = []
    actor_losses_obj = []
    backward_losses = []
        
    for i_episode in range(start_episode, NUM_EPISODES):
        
        episode_time_start = time.time()
        #noise.scale = get_noise_scale(i_episode, config)
        if train:
            for i_cycle in range(50):
                
                for i_rollout in range(38):
                    # Initialize the environment and state
                    ai_object = 1 if np.random.rand() < config['ai_object_rate']  else 0 #np.random.randint(2)
                    trajectories, _, _, _ = rollout(env, model, noise, normalizer, render=(i_rollout==37), ai_object=ai_object)
                    memory.store_episode(trajectories.copy())   
              
                for i_batch in range(40):  
                    model.to_cuda()

                    batch = memory.sample(config['batch_size'])
                    backward_loss = model.update_backward(batch, normalizer)  

                    batch = memory.sample(config['batch_size'])
                    critic_loss_robot, actor_loss_robot = model.update_parameters(batch, 0, normalizer)

                    batch = memory.sample(config['batch_size'])
                    critic_loss_obj, actor_loss_obj = model.update_parameters(batch, 1, normalizer)

                    if i_batch == 39:
                        critic_losses.append(critic_loss_robot)
                        actor_losses.append(actor_loss_robot)

                        critic_losses_obj.append(critic_loss_obj)
                        actor_losses_obj.append(actor_loss_obj)

                        backward_losses.append(backward_loss)

                model.update_target()

                #print(normalizer.get_stats())

            # <-- end loop: i_cycle
        
        plot_durations(np.asarray(critic_losses), np.asarray(actor_losses))
        plot_durations(np.asarray(critic_losses_obj), np.asarray(actor_losses_obj))
        plot_durations(np.asarray(backward_losses), np.asarray(backward_losses))
        #pdb.set_trace()
        episode_reward_cycle = []
        episode_succeess_cycle = []
        for i_rollout in range(10):
            # Initialize the environment and state
            render = config['render'] > 0 and i_episode % config['render'] == 0
            _, episode_reward, success, frames = rollout(env, model, noise=(False,False), normalizer=normalizer, render=render, ai_object=False)
                
            episode_reward_cycle.append(episode_reward)
            episode_succeess_cycle.append(success)
        # <-- end loop: i_rollout 
            
        ### MONITORIRNG ###
        episode_reward_all.append(np.mean(episode_reward_cycle))
        episode_success_all.append(np.mean(episode_succeess_cycle))

        plot_durations(np.asarray(episode_reward_all), np.asarray(episode_success_all))
        
        if config['verbose'] > 0:
        # Printing out
            if (i_episode+1)%1 == 0:
                print("==> Episode {} of {}".format(i_episode + 1, NUM_EPISODES))
                print('  | Id exp: {}'.format(config['exp_id']))
                print('  | Exp description: {}'.format(config['exp_descr']))
                print('  | Env: {}'.format(config['env_id']))
                print('  | Process pid: {}'.format(config['process_pid']))
                print('  | Tensorboard port: {}'.format(config['port']))
                print('  | Episode total reward: {}'.format(episode_reward))
                print('  | Running mean of total reward: {}'.format(episode_reward_all[-1]))
                print('  | Success rate: {}'.format(episode_success_all[-1]))
                #print('  | Running mean of total reward: {}'.format(running_mean(episode_reward_all)[-1]))
                print('  | Time episode: {}'.format(time.time()-episode_time_start))
                print('  | Time total: {}'.format(time.time()-total_time_start))
                        
        if config['verbose'] > 0:    
            ep_save = i_episode+1 if (i_episode == NUM_EPISODES-1) else None  
            is_best_save = None
            is_best_avg_save = None   
                
            if (not train) or ((np.asarray([ep_save, is_best_save, is_best_avg_save]) == None).sum() == 3):
                to_save = False
            else:
                model.to_cpu()
                saver.save_checkpoint(save_dict   = {'model_params': [entity.state_dict() for entity in model.entities]},
                                      episode     = ep_save,
                                      is_best     = is_best_save,
                                      is_best_avg = is_best_avg_save
                                      )
                to_save = True
            
    # <-- end loop: i_episode
    if train:
        print('Training completed')
    else:
        print('Test completed')
    
    return episode_reward_all, episode_success_all

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def plot_durations(p, r):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(p)
    plt.plot(r)

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

if __name__ == '__main__':

    monitor_macddpg_p2 = []
    monitor_macddpg_p2_test = []
    for i in range(0,5):
        config = get_params(args=['--exp_id','MACDDPG_P2_120K_'+ str(i+1), 
                                '--random_seed', str(i+1), 
                                '--agent_alg', 'MACDDPG',
                                '--protocol_type', str(2),
                                '--n_episodes', '120000',
                                '--verbose', '2',
                                ]
                        )
        model, experiment_args = init(config)

        env, memory, ounoise, config, summaries, saver, start_episode = experiment_args

        tic = time.time()
        monitor = run(model, experiment_args, train=True)
        monitor_test = run(model, experiment_args, train=False)

        toc = time.time()

        env.close()
        for summary in summaries:
            summary.close()
            
        monitor_macddpg_p2.append(monitor)
        monitor_macddpg_p2_test.append(monitor_test)
        
        np.save('./monitor_macddpg_p2.npy', monitor_macddpg_p2)
        np.save('./monitor_macddpg_p2_test.npy', monitor_macddpg_p2_test)
        
        print(toc-tic)