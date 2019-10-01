import numpy as np
import scipy as sc
import time
import imageio

import torch as K
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym_wmgds as gym

from sldr.algorithms.ddpg import DDPG_BD
from sldr.algorithms.maddpg import MADDPG_BD
from sldr.algorithms.ppo import PPO_BD
from sldr.experience import Normalizer, RunningMean
from sldr.exploration import Noise
from sldr.utils import Saver, Summarizer, get_params
from sldr.agents.basic import Critic
from sldr.agents.basic import ActorStoch as Actor 
from sldr.replay_buffer import RolloutStorage as ReplayBuffer

import pdb

import matplotlib
import matplotlib.pyplot as plt

device = K.device("cuda" if K.cuda.is_available() else "cpu")
dtype = K.float32

def init(config, agent='robot', her=False, object_Qfunc=None, 
                                           backward_dyn=None, 
                                           object_policy=None, 
                                           reward_fun=None,
                                           rnd_models=None):
        
    #hyperparameters
    ENV_NAME = config['env_id'] 
    SEED = config['random_seed']
    N_ENVS = config['n_envs']

    env = []
    if 'Fetch' in ENV_NAME and 'Multi' in ENV_NAME:
        for i_env in range(N_ENVS):
            env.append(gym.make(ENV_NAME, n_objects=config['max_nb_objects'], 
                                          obj_action_type=config['obj_action_type'], 
                                          observe_obj_grp=config['observe_obj_grp'],
                                          obj_range=config['obj_range']))
        n_rob_actions = 4
        n_actions = config['max_nb_objects'] * len(config['obj_action_type']) + n_rob_actions
    elif 'HandManipulate' in ENV_NAME and 'Multi' in ENV_NAME:
        for i_env in range(N_ENVS):
            env.append(gym.make(ENV_NAME, obj_action_type=config['obj_action_type']))
        n_rob_actions = 20
        n_actions = 1 * len(config['obj_action_type']) + n_rob_actions
    else:
        for i_env in range(N_ENVS):
            env.append(gym.make(ENV_NAME))
    
    for i_env in range(N_ENVS):
        env[i_env].seed(SEED+10*i_env)
    K.manual_seed(SEED)
    np.random.seed(SEED)

    observation_space = env[0].observation_space.spaces['observation'].shape[1] + env[0].observation_space.spaces['desired_goal'].shape[0]
    action_space = (gym.spaces.Box(-1., 1., shape=(n_rob_actions,), dtype='float32'),
                    gym.spaces.Box(-1., 1., shape=(n_actions-n_rob_actions,), dtype='float32'),
                    gym.spaces.Box(-1., 1., shape=(n_actions,), dtype='float32'))

    GAMMA = config['gamma'] 
    TAU = config['tau'] 
    ACTOR_LR = config['plcy_lr'] 
    CRITIC_LR = config['crtc_lr'] 

    MEM_SIZE = config['buffer_length']

    REGULARIZATION = config['regularization']
    NORMALIZED_REWARDS = config['reward_normalization']

    MODEL = PPO_BD
    OUT_FUNC = 'linear'

    #exploration initialization
    noise = True
    env[0]._max_episode_steps *= config['max_nb_objects']

    #model initialization
    optimizer = (optim.Adam, (ACTOR_LR, CRITIC_LR)) # optimiser func, (actor_lr, critic_lr)
    loss_func = F.mse_loss          
    model = MODEL(observation_space, action_space, optimizer, Actor, Critic, 
            config['clip_param'], config['ppo_epoch'], config['n_batches'], config['value_loss_coef'], config['entropy_coef'],
            eps=config['eps'], max_grad_norm=config['max_grad_norm'], use_clipped_value_loss=True,
            out_func=OUT_FUNC, discrete=False, 
            object_Qfunc=object_Qfunc, backward_dyn=backward_dyn, 
            object_policy=object_policy, reward_fun=reward_fun, masked_with_r=config['masked_with_r'],
            rnd_models=rnd_models, pred_th=config['pred_th'])
    normalizer = [Normalizer(), Normalizer()]

    #memory initilization  
    memory = ReplayBuffer(env[0]._max_episode_steps-1, config['n_rollouts'], (observation_space,), action_space[0])

    running_rintr_mean = RunningMean()

    experiment_args = (env, memory, noise, config, normalizer, running_rintr_mean)
          
    return model, experiment_args

ALL_COUNT = 0.
NC_COUNT = 0.

def rollout(env, model, noise, i_env, normalizer=None, render=False, running_rintr_mean=None):
    trajectories = []
    for i_agent in range(2):
        trajectories.append([])
    
    # monitoring variables
    episode_reward = 0
    frames = []
    
    env[i_env].env.ai_object = False
    env[i_env].env.deactivate_ai_object() 
    state_all = env[i_env].reset()

    for i_step in range(env[0]._max_episode_steps):

        model.to_cpu()

        obs = [K.tensor(obs, dtype=K.float32).unsqueeze(0) for obs in state_all['observation']]
        goal = K.tensor(state_all['desired_goal'], dtype=K.float32).unsqueeze(0)

        # Observation normalization
        obs_goal = []
        for i_agent in range(2):
            obs_goal.append(K.cat([obs[i_agent], goal], dim=-1))
            if normalizer[i_agent] is not None:
                if i_agent == 0:
                    obs_goal[i_agent] = normalizer[i_agent].preprocess_with_update(obs_goal[i_agent])
                else:
                    obs_goal[i_agent] = normalizer[i_agent].preprocess(obs_goal[i_agent])

        value, action, log_probs = model.select_action(obs_goal[0], noise)
        value = value.cpu().numpy().squeeze(0)
        action = action.cpu().numpy().squeeze(0)
        log_probs = log_probs.cpu().numpy().squeeze(0)

        action_to_env = np.zeros_like(env[0].action_space.sample())
        action_to_env[0:action.shape[0]] = action

        next_state_all, reward, done, info = env[i_env].step(action_to_env)

        next_obs = [K.tensor(next_obs, dtype=K.float32).unsqueeze(0) for next_obs in next_state_all['observation']]
        
        # Observation normalization
        next_obs_goal = []
        for i_agent in range(2):
            next_obs_goal.append(K.cat([next_obs[i_agent], goal], dim=-1))
            if normalizer[i_agent] is not None:
                next_obs_goal[i_agent] = normalizer[i_agent].preprocess(next_obs_goal[i_agent])
   
        pred_error = model.get_pred_error(next_obs_goal[1]).cpu().numpy().squeeze(0)

        global ALL_COUNT
        global NC_COUNT

        ALL_COUNT += 1.
        if pred_error > model.pred_th:
            NC_COUNT +=1.
            if running_rintr_mean is not None:
                r_intr = running_rintr_mean.get_stats()
            else:
                r_intr = -0.5
        else:
            r_intr = model.get_obj_reward(obs_goal[1], next_obs_goal[1]).cpu().numpy().squeeze(0)
            if running_rintr_mean is not None:
                running_rintr_mean.update_stats(r_intr)

        if model.masked_with_r:
            reward = r_intr * np.abs(reward) + (reward)
        else:
            reward = r_intr + (reward)

        # for monitoring
        episode_reward += reward

        for i_agent in range(2):
            trajectories[i_agent].append((obs_goal[i_agent].cpu().numpy().squeeze(0), 
                                            action_to_env[0:action.shape[0]], 
                                            reward, 
                                            next_obs_goal[i_agent].cpu().numpy().squeeze(0), 
                                            done, 
                                            value, 
                                            log_probs))

        # Move to the next state
        state_all = next_state_all

        # Record frames
        if render:
            frames.append(env[i_env].render(mode='rgb_array')[0])

    obs, acts, rews, vals, logs = [], [], [], [], []

    for i_step in range(env[0]._max_episode_steps):
        obs.append(trajectories[0][i_step][0])
        vals.append(trajectories[0][i_step][5])    
        if (i_step < env[0]._max_episode_steps - 1): 
            acts.append(trajectories[0][i_step][1])
            rews.append(trajectories[0][i_step][2])
            logs.append(trajectories[0][i_step][6])
    
    trajectories = {
        'o'    : np.asarray(obs)[np.newaxis,:],
        'u'    : np.asarray(acts)[np.newaxis,:],
        'r'    : np.asarray(rews)[np.newaxis,:],
        'v'    : np.asarray(vals)[np.newaxis,:],
        'l'    : np.asarray(logs)[np.newaxis,:]
    }

    return trajectories, episode_reward, info['is_success'], frames

def run(model, experiment_args, train=True):

    total_time_start =  time.time()

    env, memory, noise, config, normalizer, running_rintr_mean = experiment_args
    
    N_EPISODES = config['n_episodes'] if train else config['n_episodes_test']
    N_CYCLES = config['n_cycles']
    N_ROLLOUTS = config['n_rollouts']
    N_BATCHES = config['n_batches']
    N_TEST_ROLLOUTS = config['n_test_rollouts']
    BATCH_SIZE = config['batch_size']
    
    episode_reward_all = []
    episode_success_all = []
    episode_reward_mean = []
    episode_success_mean = []
    critic_losses = []
    actor_losses = []
    dist_entropies = []
        
    for i_episode in range(N_EPISODES):

        lr = config['plcy_lr']  - (config['plcy_lr']  * (i_episode / float(N_EPISODES)))
        for param_group in model.optims[0].param_groups:
            param_group['lr'] = lr
        
        episode_time_start = time.time()
        if train:
            for i_cycle in range(N_CYCLES):
                
                trajectories = None
                for i_rollout in range(N_ROLLOUTS):
                    # Initialize the environment and state
                    i_env = i_rollout % config['n_envs']
                    render = config['render'] > 0 and i_rollout==0
                    trajectory, _, _, _ = rollout(env, model, noise, i_env, normalizer, render=render, running_rintr_mean=running_rintr_mean)
                    if trajectories is None:
                        trajectories = trajectory.copy()
                    else:
                        for key in trajectories.keys():
                            trajectories[key] = np.concatenate((trajectories[key], trajectory.copy()[key]), axis=0)

                memory.obs.copy_(K.tensor(np.swapaxes(trajectories['o'], 0, 1), dtype=model.dtype, device=model.device))
                memory.actions.copy_(K.tensor(np.swapaxes(trajectories['u'], 0, 1), dtype=model.dtype, device=model.device))
                memory.rewards.copy_(K.tensor(np.swapaxes(trajectories['r'], 0, 1), dtype=model.dtype, device=model.device))
                memory.value_preds.copy_(K.tensor(np.swapaxes(trajectories['v'], 0, 1), dtype=model.dtype, device=model.device))
                memory.action_log_probs.copy_(K.tensor(np.swapaxes(trajectories['l'], 0, 1), dtype=model.dtype, device=model.device))

                memory.compute_returns(config['use_gae'], config['gamma'], config['gae_lambda'])

                model.to_cuda()
                memory.to(model.device)

                critic_loss, actor_loss, dist_entropy = model.update(memory)

                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
                dist_entropies.append(dist_entropy)

                if NC_COUNT > 0:
                    print(NC_COUNT/ALL_COUNT)
                    print(running_rintr_mean.get_stats())

            # <-- end loop: i_cycle
        plot_durations(np.asarray(critic_losses), np.asarray(critic_losses))
        plot_durations(np.asarray(actor_losses), np.asarray(actor_losses))
        plot_durations(np.asarray(dist_entropies), np.asarray(dist_entropies))

        episode_reward_cycle = []
        episode_succeess_cycle = []
        for i_rollout in range(N_TEST_ROLLOUTS):
            # Initialize the environment and state
            rollout_per_env = N_TEST_ROLLOUTS // config['n_envs']
            i_env = i_rollout // rollout_per_env
            render = config['render'] > 0 and i_episode % config['render'] == 0 and i_env == 0
            _, episode_reward, success, _ = rollout(env, model, False, i_env, normalizer=normalizer, render=render, running_rintr_mean=running_rintr_mean)
                
            episode_reward_cycle.append(episode_reward.item())
            episode_succeess_cycle.append(success)
        # <-- end loop: i_rollout 
            
        ### MONITORIRNG ###
        episode_reward_all.append(episode_reward_cycle)
        episode_success_all.append(episode_succeess_cycle)

        episode_reward_mean.append(np.mean(episode_reward_cycle))
        episode_success_mean.append(np.mean(episode_succeess_cycle))
        plot_durations(np.asarray(episode_reward_mean), np.asarray(episode_success_mean))
        
        if config['verbose'] > 0:
        # Printing out
            if (i_episode+1)%1 == 0:
                print("==> Episode {} of {}".format(i_episode + 1, N_EPISODES))
                print('  | Id exp: {}'.format(config['exp_id']))
                print('  | Exp description: {}'.format(config['exp_descr']))
                print('  | Env: {}'.format(config['env_id']))
                print('  | Process pid: {}'.format(config['process_pid']))
                print('  | Episode total reward: {}'.format(episode_reward))
                print('  | Running mean of total reward: {}'.format(episode_reward_mean[-1]))
                print('  | Success rate: {}'.format(episode_success_mean[-1]))
                print('  | Time episode: {}'.format(time.time()-episode_time_start))
                print('  | Time total: {}'.format(time.time()-total_time_start))
            
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