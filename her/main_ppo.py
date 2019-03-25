import numpy as np
import scipy as sc
import time
import imageio

import torch as K
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym_wmgds as gym

from her.algorithms.ddpg import DDPG_BD
from her.algorithms.maddpg import MADDPG_BD
from her.algorithms.ppo import PPO_BD
from her.experience import Normalizer
from her.exploration import Noise
from her.utils import Saver, Summarizer, get_params, running_mean
from her.agents.basic import Critic

import pdb

import matplotlib
import matplotlib.pyplot as plt

device = K.device("cuda" if K.cuda.is_available() else "cpu")
dtype = K.float32

def init(config, agent='robot', her=False, object_Qfunc=None, backward_dyn=None, object_policy=None, reward_fun=None):
        
    #hyperparameters
    ENV_NAME = config['env_id'] 
    SEED = config['random_seed']

    if (ENV_NAME == 'FetchStackMulti-v1') or (ENV_NAME == 'FetchPushMulti-v1') or (ENV_NAME == 'FetchPickAndPlaceMulti-v1'):
        env = gym.make(ENV_NAME, n_objects=config['max_nb_objects'], obj_action_type=config['obj_action_type'], observe_obj_grp=config['observe_obj_grp'])
    else:
        env = gym.make(ENV_NAME)

    def her_reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)
    
    env.seed(SEED)
    K.manual_seed(SEED)
    np.random.seed(SEED)

    #if config['obj_action_type'] == 'all':
    #    n_actions = config['max_nb_objects'] * 7 + 4
    #elif config['obj_action_type'] == 'slide_only':
    #    n_actions = config['max_nb_objects'] * 3 + 4
    #elif config['obj_action_type'] == 'rotation_only':
    #    n_actions = config['max_nb_objects'] * 4 + 4
    n_actions = config['max_nb_objects'] * len(config['obj_action_type']) + 4

    observation_space = env.observation_space.spaces['observation'].shape[1] + env.observation_space.spaces['desired_goal'].shape[0]
    action_space = (gym.spaces.Box(-1., 1., shape=(4,), dtype='float32'),
                    gym.spaces.Box(-1., 1., shape=(n_actions-4,), dtype='float32'),
                    gym.spaces.Box(-1., 1., shape=(n_actions,), dtype='float32'))

    GAMMA = config['gamma'] 
    TAU = config['tau'] 
    ACTOR_LR = config['plcy_lr'] 
    CRITIC_LR = config['crtc_lr'] 

    MEM_SIZE = config['buffer_length']

    REGULARIZATION = config['regularization']
    NORMALIZED_REWARDS = config['reward_normalization']

    if config['agent_alg'] == 'DDPG_BD':
        MODEL = DDPG_BD
        OUT_FUNC = K.tanh 
        from her.agents.basic import Actor 
        from her.replay_buffer import ReplayBuffer
        from her.her_sampler import make_sample_her_transitions
    elif config['agent_alg'] == 'MADDPG_BD':
        MODEL = MADDPG_BD
        OUT_FUNC = K.tanh 
        from her.agents.basic import Actor 
        from her.replay_buffer import ReplayBuffer_v2 as ReplayBuffer
        from her.her_sampler import make_sample_her_transitions_v2 as make_sample_her_transitions
    elif config['agent_alg'] == 'PPO_BD':
        MODEL = PPO_BD
        OUT_FUNC = 'linear'
        from her.agents.basic import ActorStoch as Actor 
        from her.replay_buffer import RolloutStorage as ReplayBuffer

    #exploration initialization
    if agent == 'robot':
        agent_id = 0
        if config['agent_alg'] == 'PPO_BD':
            noise = True
        else:
            noise = Noise(action_space[0].shape[0], sigma=0.2, eps=0.3)
    elif agent == 'object':
        agent_id = 1
        #noise = Noise(action_space[1].shape[0], sigma=0.05, eps=0.1)
        noise = Noise(action_space[1].shape[0], sigma=0.2, eps=0.3)

    #model initialization
    optimizer = (optim.Adam, (ACTOR_LR, CRITIC_LR)) # optimiser func, (actor_lr, critic_lr)
    loss_func = F.mse_loss
    if config['agent_alg'] == 'PPO_BD':            
        model = MODEL(observation_space, action_space, optimizer, Actor, Critic, 
                config['clip_param'], config['ppo_epoch'], config['n_batches'], config['value_loss_coef'], config['entropy_coef'],
                eps=config['eps'], max_grad_norm=config['max_grad_norm'], use_clipped_value_loss=True,
                out_func=OUT_FUNC, discrete=False, 
                agent_id=agent_id, object_Qfunc=object_Qfunc, backward_dyn=backward_dyn, 
                object_policy=object_policy, reward_fun=reward_fun)
    else:  
        model = MODEL(observation_space, action_space, optimizer, Actor, Critic, 
                    loss_func, GAMMA, TAU, out_func=OUT_FUNC, discrete=False, 
                    regularization=REGULARIZATION, normalized_rewards=NORMALIZED_REWARDS,
                    agent_id=agent_id, object_Qfunc=object_Qfunc, backward_dyn=backward_dyn, 
                    object_policy=object_policy, reward_fun=reward_fun)  
   
    normalizer = [Normalizer(), Normalizer()]

    
    #memory initilization  
    if config['agent_alg'] == 'PPO_BD':
        memory = ReplayBuffer(env._max_episode_steps-1, config['n_rollouts'], (observation_space,), action_space[0])
    else:
        if her:
            sample_her_transitions = make_sample_her_transitions('future', 4, her_reward_fun)
        else:
            sample_her_transitions = make_sample_her_transitions('none', 4, her_reward_fun)

        buffer_shapes = {
            'o' : (env._max_episode_steps, env.observation_space.spaces['observation'].shape[1]*2),
            'ag' : (env._max_episode_steps, env.observation_space.spaces['achieved_goal'].shape[0]),
            'g' : (env._max_episode_steps, env.observation_space.spaces['desired_goal'].shape[0]),
            'u' : (env._max_episode_steps-1, action_space[2].shape[0]),
            'r' : (env._max_episode_steps-1, 1)
            }
        memory = ReplayBuffer(buffer_shapes, MEM_SIZE, env._max_episode_steps, sample_her_transitions)

    experiment_args = (env, memory, noise, config, normalizer, agent_id)

    print('clipped between -1 and 0, and masked with abs(r), and + r')
          
    return model, experiment_args

def rollout(env, model, noise, normalizer=None, render=False, agent_id=0, ai_object=False):
    trajectories = []
    
    # monitoring variables
    episode_reward = 0
    frames = []
    
    env.env.ai_object = True if agent_id==1 else ai_object
    state_all = env.reset()

    for i_agent in range(2):
        trajectories.append([])
    
    for i_step in range(env._max_episode_steps):

        model.to_cpu()

        obs = [K.tensor(obs, dtype=K.float32).unsqueeze(0) for obs in state_all['observation']]
        goal = K.tensor(state_all['desired_goal'], dtype=K.float32).unsqueeze(0)

        #rew_distance = - K.sqrt(((obs[0][0,0:3] - goal)*(obs[0][0,0:3] - goal)).sum(1)).cpu().numpy()

        # Observation normalization
        obs_goal = []
        for i_agent in range(2):
            obs_goal.append(K.cat([obs[i_agent], goal], dim=-1))
            if normalizer[i_agent] is not None:
                obs_goal[i_agent] = normalizer[i_agent].preprocess_with_update(obs_goal[i_agent])

        value, action, log_probs = model.select_action(obs_goal[agent_id], noise)
        value = value.cpu().numpy().squeeze(0)
        action = action.cpu().numpy().squeeze(0)
        log_probs = log_probs.cpu().numpy().squeeze(0)

        if agent_id == 0:
            action_to_env = np.zeros_like(env.action_space.sample())
            action_to_env[0:action.shape[0]] = action
            if ai_object:
                action_to_env[action.shape[0]::] = model.get_obj_action(obs_goal[1]).cpu().numpy().squeeze(0)
        else:
            action_to_env = env.action_space.sample()
            action_to_env[-action.shape[0]::] = action

        next_state_all, reward, done, info = env.step(action_to_env)

        next_obs = [K.tensor(next_obs, dtype=K.float32).unsqueeze(0) for next_obs in next_state_all['observation']]
        
        # Observation normalization
        next_obs_goal = []
        for i_agent in range(2):
            next_obs_goal.append(K.cat([next_obs[i_agent], goal], dim=-1))
            if normalizer[i_agent] is not None:
                next_obs_goal[i_agent] = normalizer[i_agent].preprocess(next_obs_goal[i_agent])

        if agent_id == 0:
            #reward = model.get_obj_reward(obs_goal[1], next_obs_goal[1]).cpu().numpy().squeeze(0) * np.abs(reward) * 0.90 + (reward)
            #reward = model.get_obj_reward(obs_goal[1], next_obs_goal[1]).cpu().numpy().squeeze(0) * np.abs(1) * 1 + (reward)
            reward = model.get_obj_reward(obs_goal[1], next_obs_goal[1]).cpu().numpy().squeeze(0) * np.abs(reward) + (reward)

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
            frames.append(env.render(mode='rgb_array')[0])

    obs, acts, rews, vals, logs = [], [], [], [], []

    for i_step in range(env._max_episode_steps):
        obs.append(trajectories[0][i_step][0])
        vals.append(trajectories[0][i_step][5])    
        if (i_step < env._max_episode_steps - 1): 
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

    env, memory, noise, config, normalizer, agent_id = experiment_args
    
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
    dist_entropies = []
    backward_losses = []
        
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
                    ai_object = 1 if np.random.rand() < config['ai_object_rate']  else 0
                    trajectory, _, _, _ = rollout(env, model, noise, normalizer, render=(i_rollout==N_ROLLOUTS-1), agent_id=agent_id, ai_object=ai_object)
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

            # <-- end loop: i_cycle
        plot_durations(np.asarray(critic_losses), np.asarray(critic_losses))
        plot_durations(np.asarray(actor_losses), np.asarray(actor_losses))
        plot_durations(np.asarray(dist_entropies), np.asarray(dist_entropies))

        episode_reward_cycle = []
        episode_succeess_cycle = []
        for i_rollout in range(N_TEST_ROLLOUTS):
            # Initialize the environment and state
            render = config['render'] > 0 and i_episode % config['render'] == 0
            if config['agent_alg'] == 'PPO_BD':
                _, episode_reward, success, _ = rollout(env, model, noise=False, normalizer=normalizer, render=render, agent_id=agent_id, ai_object=False)
            else:
                _, episode_reward, success, _ = rollout(env, model, noise=False, normalizer=normalizer, render=render, agent_id=agent_id, ai_object=False)
                
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

    if train and agent_id==1:
        print('Training Backward Model')
        model.to_cuda()
        for _ in range(N_EPISODES*N_CYCLES*10):
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