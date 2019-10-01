import numpy as np
import scipy as sc
import time
import imageio

import torch as K
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym_wmgds as gym

from sldr.algorithms4 import DDPG_BD
from sldr.experience import Normalizer
from sldr.replay_buffer import ReplayBuffer
from sldr.her_sampler import make_sample_her_transitions
from sldr.exploration import Noise
from sldr.utils import Saver, Summarizer, get_params, running_mean
from sldr.agents.basic import Actor 
from sldr.agents.basic import Critic

import pdb

import matplotlib
import matplotlib.pyplot as plt

device = K.device("cuda" if K.cuda.is_available() else "cpu")
dtype = K.float32

def init(config, agent='robot', her=False, object_Qfunc=None, backward_dyn=None):
        
    #hyperparameters
    ENV_NAME = config['env_id'] 
    SEED = config['random_seed']

    if (ENV_NAME == 'FetchStackMulti-v1') or (ENV_NAME == 'FetchPushMulti-v1'):
        env = gym.make(ENV_NAME, n_objects=config['max_nb_objects'], obj_action_type=config['obj_action_type'])
    else:
        env = gym.make(ENV_NAME)

    def reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)
    
    env.seed(SEED)
    K.manual_seed(SEED)
    np.random.seed(SEED)

    if config['obj_action_type'] == 'all':
        n_actions = config['max_nb_objects'] * 7 + 4
    elif config['obj_action_type'] == 'slide_only':
        n_actions = config['max_nb_objects'] * 3 + 4
    elif config['obj_action_type'] == 'rotation_only':
        n_actions = config['max_nb_objects'] * 4 + 4

    observation_space = env.observation_space.spaces['observation'].shape[1] + env.observation_space.spaces['desired_goal'].shape[0]
    action_space = (gym.spaces.Box(-1., 1., shape=(4,), dtype='float32'),
                    gym.spaces.Box(-1., 1., shape=(n_actions-4,), dtype='float32'))

    GAMMA = config['gamma'] 
    TAU = config['tau'] 
    ACTOR_LR = config['plcy_lr'] 
    CRITIC_LR = config['crtc_lr'] 

    MEM_SIZE = config['buffer_length']

    REGULARIZATION = config['regularization']
    NORMALIZED_REWARDS = config['reward_normalization']

    OUT_FUNC = K.tanh 
    MODEL = DDPG_BD         

    #exploration initialization
    if agent == 'robot':
        i_agent = 0
        noise = Noise(action_space[0].shape[0], sigma=0.2, eps=0.3)
    elif agent == 'object':
        i_agent = 1
        noise = Noise(action_space[1].shape[0], sigma=0.05, eps=0.1)
             
    #model initialization
    optimizer = (optim.Adam, (ACTOR_LR, CRITIC_LR)) # optimiser func, (actor_lr, critic_lr)
    loss_func = F.mse_loss
    model = MODEL(observation_space, action_space[i_agent], optimizer, 
                  Actor, Critic, loss_func, GAMMA, TAU, out_func=OUT_FUNC, discrete=False, 
                  regularization=REGULARIZATION, normalized_rewards=NORMALIZED_REWARDS,
                  object_Qfunc=object_Qfunc, backward_dyn=backward_dyn)
    normalizer = Normalizer()

    #memory initilization  
    if her:
        sample_her_transitions = make_sample_her_transitions('future', 4, reward_fun)
    else:
        sample_her_transitions = make_sample_her_transitions('none', 4, reward_fun)

    buffer_shapes = {
        'o' : (env._max_episode_steps, env.observation_space.spaces['observation'].shape[1]),
        'ag' : (env._max_episode_steps, env.observation_space.spaces['achieved_goal'].shape[0]),
        'g' : (env._max_episode_steps, env.observation_space.spaces['desired_goal'].shape[0]),
        'u' : (env._max_episode_steps-1, action_space[i_agent].shape[0])
        }
    memory = ReplayBuffer(buffer_shapes, MEM_SIZE, env._max_episode_steps, sample_her_transitions)

    experiment_args = (env, memory, noise, config, normalizer, i_agent)
          
    return model, experiment_args

def rollout(env, model, noise, normalizer=None, render=False, i_agent=0):

    # monitoring variables
    episode_reward = 0
    frames = []
    
    env.env.ai_object = True if i_agent==1 else False
    state_all = env.reset()

    trajectory = []
    
    for i_step in range(env._max_episode_steps):

        model.to_cpu()

        obs = K.tensor(state_all['observation'][i_agent], dtype=K.float32).unsqueeze(0)
        goal = K.tensor(state_all['desired_goal'], dtype=K.float32).unsqueeze(0)

        obs_goal = K.cat([obs, goal], dim=-1)
        # Observation normalization
        if normalizer is not None:
            obs_goal = normalizer.preprocess_with_update(obs_goal)

        action = model.select_action(obs_goal, noise).cpu().numpy().squeeze(0)

        if i_agent == 0:
            action_to_env = np.zeros_like(env.action_space.sample())
            action_to_env[0:action.shape[0]] = action
        else:
            action_to_env = np.ones_like(env.action_space.sample())
            action_to_env[-action.shape[0]::] = action

        next_state_all, reward, done, info = env.step(action_to_env)
        reward = K.tensor(reward, dtype=dtype).view(1,1)

        # for monitoring
        if i_agent == 0:
            next_obs = K.tensor(next_state_all['observation'][i_agent], dtype=K.float32).unsqueeze(0)
            next_obs_goal = K.cat([next_obs, goal], dim=-1)
            if normalizer is not None:
                next_obs_goal = normalizer.preprocess(next_obs_goal)     
            episode_reward += model.get_obj_reward(obs_goal, next_obs_goal)
        else:
            episode_reward += reward

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
        
        trajectory.append((state.copy(), action, reward, next_state.copy(), done))

        # Move to the next state
        state_all = next_state_all

        # Record frames
        if render:
            frames.append(env.render(mode='rgb_array')[0])

    obs, ags, goals, acts = [], [], [], []

    for i_step in range(env._max_episode_steps):
        obs.append(trajectory[i_step][0]['observation'])
        ags.append(trajectory[i_step][0]['achieved_goal'])
        goals.append(trajectory[i_step][0]['desired_goal'])
        if (i_step < env._max_episode_steps - 1): 
            acts.append(trajectory[i_step][1])

    trajectory = {
        'o'    : np.asarray(obs)[np.newaxis,:,:],
        'ag'   : np.asarray(ags)[np.newaxis,:,:],
        'g'    : np.asarray(goals)[np.newaxis,:,:],
        'u'    : np.asarray(acts)[np.newaxis,:,:]
        }

    return trajectory, episode_reward, info['is_success'], frames

def run(model, experiment_args, train=True):

    total_time_start =  time.time()

    env, memory, noise, config, normalizer, i_agent = experiment_args
    
    N_EPISODES = config['n_episodes'] if train else config['n_episodes_test']
    N_CYCLES = config['n_cycles']
    N_ROLLOUTS = config['n_rollouts']
    N_BATCHES = config['n_batches']
    N_TEST_ROLLOUTS = config['n_test_rollouts']
    BATCH_SIZE = config['batch_size']
    
    episode_reward_all = []
    episode_success_all = []
    critic_losses = []
    actor_losses = []
    backward_losses = []
        
    for i_episode in range(N_EPISODES):
        
        episode_time_start = time.time()
        #noise.scale = get_noise_scale(i_episode, config)
        if train:
            for i_cycle in range(N_CYCLES):
                
                for i_rollout in range(N_ROLLOUTS):
                    # Initialize the environment and state
                    trajectories, _, _, _ = rollout(env, model, noise, normalizer, render=(i_rollout==N_ROLLOUTS-1), i_agent=i_agent)
                    memory.store_episode(trajectories.copy())   
              
                for i_batch in range(N_BATCHES):  
                    model.to_cuda()

                    batch = memory.sample(BATCH_SIZE)
                    critic_loss, actor_loss = model.update_parameters(batch, normalizer, use_object_Qfunc=True if i_agent==0 else False)

                    if i_batch == N_BATCHES - 1:
                        critic_losses.append(critic_loss)
                        actor_losses.append(actor_loss)

                model.update_target()

            # <-- end loop: i_cycle
        plot_durations(np.asarray(critic_losses), np.asarray(actor_losses))

        episode_reward_cycle = []
        episode_succeess_cycle = []
        for i_rollout in range(N_TEST_ROLLOUTS):
            # Initialize the environment and state
            render = config['render'] > 0 and i_episode % config['render'] == 0
            _, episode_reward, success, _ = rollout(env, model, noise=False, normalizer=normalizer, render=render, i_agent=i_agent)
                
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
                print("==> Episode {} of {}".format(i_episode + 1, N_EPISODES))
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
            
    # <-- end loop: i_episode

    if train and i_agent==1:
        print('Training Backward Model')
        for _ in range(N_EPISODES*N_CYCLES):
            for i_batch in range(N_BATCHES):
                batch = memory.sample(BATCH_SIZE)
                backward_loss = model.update_backward(batch, normalizer)  
                if i_batch == N_BATCHES - 1:
                    backward_losses.append(backward_loss)
        plot_durations(np.asarray(backward_losses), np.asarray(backward_losses))

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