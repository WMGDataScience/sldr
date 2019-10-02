import numpy as np
import scipy as sc
import time
import imageio
import copy

import torch as K
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym_wmgds as gym

from sldr.algorithms.ddpg import DDPG_BD
from sldr.experience import Normalizer
from sldr.exploration import Noise
from sldr.utils import Saver, Summarizer, get_params, running_mean
from sldr.agents.basic import Actor 
from sldr.agents.basic import Critic

import pdb

import matplotlib
import matplotlib.pyplot as plt

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

device = K.device("cuda" if K.cuda.is_available() else "cpu")
dtype = K.float32

def init(config, agent='robot', her=False, object_Qfunc=None, backward_dyn=None, object_policy=None, reward_fun=None):
        
    #hyperparameters
    ENV_NAME = config['env_id'] 
    SEED = config['random_seed']
    N_ENVS = config['n_envs']

    def make_env(env_id, i_env, env_type='Fetch', ai_object=False):
        def _f():
            if env_type == 'Fetch':
                env = gym.make(env_id, n_objects=config['max_nb_objects'], 
                                    obj_action_type=config['obj_action_type'], 
                                    observe_obj_grp=config['observe_obj_grp'],
                                    obj_range=config['obj_range'])
            elif env_type == 'Hand':
                env = gym.make(env_id, obj_action_type=config['obj_action_type'])
            elif env_type == 'Others':
                env = gym.make(env_id)
            
            keys = env.observation_space.spaces.keys()
            env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))
            env.seed(SEED+10*i_env)
            env.unwrapped.ai_object = ai_object
            return env
        return _f   

    if 'Fetch' in ENV_NAME and 'Multi' in ENV_NAME and 'Flex' not in ENV_NAME:
        dummy_env = gym.make(ENV_NAME, n_objects=config['max_nb_objects'], 
                                    obj_action_type=config['obj_action_type'], 
                                    observe_obj_grp=config['observe_obj_grp'],
                                    obj_range=config['obj_range'])
        envs = SubprocVecEnv([make_env(ENV_NAME, i_env, 'Fetch', agent == 'object') for i_env in range(N_ENVS)])
        envs_render = SubprocVecEnv([make_env(ENV_NAME, i_env, 'Fetch', agent == 'object') for i_env in range(1)])
        n_rob_actions = 4
        n_actions = config['max_nb_objects'] * len(config['obj_action_type']) + n_rob_actions
    elif 'Fetch' in ENV_NAME and 'Multi' in ENV_NAME and 'Flex' in ENV_NAME:
        dummy_env = gym.make(ENV_NAME, n_objects=config['max_nb_objects'], 
                                    obj_action_type=config['obj_action_type'], 
                                    observe_obj_grp=config['observe_obj_grp'],
                                    obj_range=config['obj_range'])
        envs = SubprocVecEnv([make_env(ENV_NAME, i_env, 'Fetch', agent == 'object') for i_env in range(N_ENVS)])
        envs_render = SubprocVecEnv([make_env(ENV_NAME, i_env, 'Fetch', agent == 'object') for i_env in range(1)])
        n_rob_actions = 4
        n_actions = 2 * len(config['obj_action_type']) + n_rob_actions
    elif 'HandManipulate' in ENV_NAME and 'Multi' in ENV_NAME:
        dummy_env = gym.make(ENV_NAME, obj_action_type=config['obj_action_type'])
        envs = SubprocVecEnv([make_env(ENV_NAME, i_env, 'Hand', agent == 'object') for i_env in range(N_ENVS)])
        envs_render = SubprocVecEnv([make_env(ENV_NAME, i_env, 'Hand', agent == 'object') for i_env in range(N_ENVS)])
        n_rob_actions = 20
        n_actions = 1 * len(config['obj_action_type']) + n_rob_actions
    else:
        dummy_env = gym.make(ENV_NAME)
        envs = SubprocVecEnv([make_env(ENV_NAME, i_env, 'Others', agent == 'object') for i_env in range(N_ENVS)])
        envs_render = None

    def her_reward_fun(ag_2, g, info):  # vectorized
        return dummy_env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)


    K.manual_seed(SEED)
    np.random.seed(SEED)

    observation_space = dummy_env.observation_space.spaces['observation'].shape[1] + dummy_env.observation_space.spaces['desired_goal'].shape[0]
    action_space = (gym.spaces.Box(-1., 1., shape=(n_rob_actions,), dtype='float32'),
                    gym.spaces.Box(-1., 1., shape=(n_actions-n_rob_actions,), dtype='float32'),
                    gym.spaces.Box(-1., 1., shape=(n_actions,), dtype='float32'))

    GAMMA = config['gamma']
    clip_Q_neg = config['clip_Q_neg'] if config['clip_Q_neg'] < 0 else None
    TAU = config['tau'] 
    ACTOR_LR = config['plcy_lr'] 
    CRITIC_LR = config['crtc_lr'] 

    MEM_SIZE = config['buffer_length']

    REGULARIZATION = config['regularization']
    NORMALIZED_REWARDS = config['reward_normalization']

    OUT_FUNC = K.tanh 
    if config['agent_alg'] == 'DDPG_BD':
        MODEL = DDPG_BD
        from sldr.replay_buffer import ReplayBuffer
        from sldr.her_sampler import make_sample_her_transitions
    elif config['agent_alg'] == 'MADDPG_BD':
        MODEL = MADDPG_BD
        from sldr.replay_buffer import ReplayBuffer_v2 as ReplayBuffer
        from sldr.her_sampler import make_sample_her_transitions_v2 as make_sample_her_transitions

    #exploration initialization
    if agent == 'robot':
        agent_id = 0
        noise = Noise(action_space[0].shape[0], sigma=0.2, eps=0.3)
    elif agent == 'object':
        agent_id = 1
        #noise = Noise(action_space[1].shape[0], sigma=0.2, eps=0.3)    
        noise = Noise(action_space[1].shape[0], sigma=0.05, eps=0.2)   
    config['episode_length'] = dummy_env._max_episode_steps
    config['observation_space'] = dummy_env.observation_space

    #model initialization
    optimizer = (optim.Adam, (ACTOR_LR, CRITIC_LR)) # optimiser func, (actor_lr, critic_lr)
    loss_func = F.mse_loss
    model = MODEL(observation_space, action_space, optimizer, 
                  Actor, Critic, loss_func, GAMMA, TAU, out_func=OUT_FUNC, discrete=False, 
                  regularization=REGULARIZATION, normalized_rewards=NORMALIZED_REWARDS,
                  agent_id=agent_id, object_Qfunc=object_Qfunc, backward_dyn=backward_dyn, 
                  object_policy=object_policy, reward_fun=reward_fun, clip_Q_neg=clip_Q_neg
                  )
    normalizer = [Normalizer(), Normalizer()]

    for _ in range(1):
        state_all = dummy_env.reset()
        for _ in range(config['episode_length']):

            model.to_cpu()

            obs = [K.tensor(obs, dtype=K.float32).unsqueeze(0) for obs in state_all['observation']]
            goal = K.tensor(state_all['desired_goal'], dtype=K.float32).unsqueeze(0)

            # Observation normalization
            obs_goal = []
            obs_goal.append(K.cat([obs[agent_id], goal], dim=-1))
            if normalizer[agent_id] is not None:
                obs_goal[0] = normalizer[agent_id].preprocess_with_update(obs_goal[0])

            action = model.select_action(obs_goal[0], noise).cpu().numpy().squeeze(0)
            action_to_env = np.zeros_like(dummy_env.action_space.sample())
            if agent_id == 0:
                action_to_env[0:action.shape[0]] = action
            else:
                action_to_env[-action.shape[0]::] = action

            next_state_all, _, _, _ = dummy_env.step(action_to_env)

            # Move to the next state
            state_all = next_state_all

    #memory initilization  
    if her:
        sample_her_transitions = make_sample_her_transitions('future', 4, her_reward_fun)
    else:
        sample_her_transitions = make_sample_her_transitions('none', 4, her_reward_fun)

    buffer_shapes = {
        'o' : (config['episode_length'], dummy_env.observation_space.spaces['observation'].shape[1]*2),
        'ag' : (config['episode_length'], dummy_env.observation_space.spaces['achieved_goal'].shape[0]),
        'g' : (config['episode_length'], dummy_env.observation_space.spaces['desired_goal'].shape[0]),
        'u' : (config['episode_length']-1, action_space[2].shape[0])
        }
    memory = ReplayBuffer(buffer_shapes, MEM_SIZE, config['episode_length'], sample_her_transitions)

    experiment_args = ((envs, envs_render), memory, noise, config, normalizer, agent_id)
          
    return model, experiment_args

def back_to_dict(state, config):

    goal_len = config['observation_space'].spaces['desired_goal'].shape[0]
    obs_len = config['observation_space'].spaces['observation'].shape[1]
    n_agents = config['observation_space'].spaces['observation'].shape[0]

    state_dict = {}
    state_dict['achieved_goal'] = state[:,0:goal_len]
    state_dict['desired_goal'] = state[:,goal_len:goal_len*2]
    state_dict['observation'] = state[:,goal_len*2::].reshape(-1,n_agents,obs_len).swapaxes(0,1)

    return state_dict

def rollout(env, model, noise, config, normalizer=None, render=False, agent_id=0, ai_object=False, rob_policy=[0., 0.]):
    trajectories = []
    for i_agent in range(2):
        trajectories.append([])
    
    # monitoring variables
    episode_reward = np.zeros(env.num_envs)
    frames = []
    
    state_all = env.reset()
    state_all = back_to_dict(state_all, config)

    for i_step in range(config['episode_length']):

        model.to_cpu()

        obs = [K.tensor(obs, dtype=K.float32) for obs in state_all['observation']]
        goal = K.tensor(state_all['desired_goal'], dtype=K.float32)

        # Observation normalization
        obs_goal = []
        for i_agent in range(2):
            obs_goal.append(K.cat([obs[i_agent], goal], dim=-1))
            if normalizer[i_agent] is not None:
                obs_goal[i_agent] = normalizer[i_agent].preprocess_with_update(obs_goal[i_agent])

        action = model.select_action(obs_goal[agent_id], noise).cpu().numpy()

        if agent_id == 0:
            action_to_env = np.zeros((len(action), len(env.action_space.sample())))
            action_to_env[:,0:action.shape[1]] = action
            if ai_object:
                action_to_env[:, action.shape[1]::] = model.get_obj_action(obs_goal[1]).cpu().numpy()
            action_to_mem = action_to_env
        else:
            action_to_env = np.zeros((len(action), len(env.action_space.sample())))
            action_to_env[:,] = env.action_space.sample() * rob_policy[0] + np.ones_like(env.action_space.sample()) * rob_policy[1]
            action_to_env[:,-action.shape[1]::] = action
            action_to_mem = action_to_env

        next_state_all, reward, done, info = env.step(action_to_env)
        next_state_all = back_to_dict(next_state_all, config)
        reward = K.tensor(reward, dtype=dtype).view(-1,1)

        next_obs = [K.tensor(next_obs, dtype=K.float32) for next_obs in next_state_all['observation']]
        
        # Observation normalization
        next_obs_goal = []
        for i_agent in range(2):
            next_obs_goal.append(K.cat([next_obs[i_agent], goal], dim=-1))
            if normalizer[i_agent] is not None:
                next_obs_goal[i_agent] = normalizer[i_agent].preprocess(next_obs_goal[i_agent])

        # for monitoring
        if model.object_Qfunc is None:            
            episode_reward += reward.squeeze(1).cpu().numpy()
        else:
            r_intr = model.get_obj_reward(obs_goal[1], next_obs_goal[1])
            if model.masked_with_r:
                episode_reward += (r_intr * K.abs(reward) + reward).squeeze(1).cpu().numpy()
            else:
                episode_reward += (r_intr + reward).squeeze(1).cpu().numpy()

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
            
            trajectories[i_agent].append((state.copy(), action_to_mem, reward, next_state.copy(), done))
        
        goal_a = state_all['achieved_goal']
        goal_b = state_all['desired_goal']
        ENV_NAME = config['env_id'] 
        if 'Rotate' in ENV_NAME:
            goal_a = goal_a[:,3:]
            goal_b = goal_b[:,3:]

        # Move to the next state
        state_all = next_state_all

        # Record frames
        if render:
            frames.append(env.render(mode='rgb_array')[0])

    distance = np.linalg.norm(goal_a - goal_b, axis=-1)
    
    obs, ags, goals, acts = [], [], [], []

    for trajectory in trajectories:
        obs.append([])
        ags.append([])
        goals.append([])
        acts.append([])
        for i_step in range(config['episode_length']):
            obs[-1].append(trajectory[i_step][0]['observation'])
            ags[-1].append(trajectory[i_step][0]['achieved_goal'])
            goals[-1].append(trajectory[i_step][0]['desired_goal'])
            if (i_step < config['episode_length'] - 1): 
                acts[-1].append(trajectory[i_step][1])

    trajectories = {
        'o'    : np.concatenate(obs,axis=-1).swapaxes(0,1),
        'ag'   : np.asarray(ags)[0,].swapaxes(0,1),
        'g'    : np.asarray(goals)[0,].swapaxes(0,1),
        'u'    : np.asarray(acts)[0,].swapaxes(0,1),
    }

    info = np.asarray([i_info['is_success'] for i_info in info])

    return trajectories, episode_reward, info, distance

def run(model, experiment_args, train=True):

    total_time_start =  time.time()

    envs, memory, noise, config, normalizer, agent_id = experiment_args
    envs_train, envs_render = envs
    
    N_EPISODES = config['n_episodes'] if train else config['n_episodes_test']
    N_CYCLES = config['n_cycles']
    N_BATCHES = config['n_batches']
    N_TEST_ROLLOUTS = config['n_test_rollouts']
    BATCH_SIZE = config['batch_size']
    
    episode_reward_all = []
    episode_success_all = []
    episode_distance_all = []
    episode_reward_mean = []
    episode_success_mean = []
    episode_distance_mean = []
    critic_losses = []
    actor_losses = []
    backward_losses = []
    backward_otw_losses = []

    best_succeess = -1
        
    for i_episode in range(N_EPISODES):
        
        episode_time_start = time.time()
        if train:
            for i_cycle in range(N_CYCLES):
                
                ai_object = 1 if np.random.rand() < config['ai_object_rate']  else 0
                trajectories, _, _, _ = rollout(envs_train, model, noise, config, normalizer, render=False, agent_id=agent_id, ai_object=ai_object, rob_policy=config['rob_policy'])
                memory.store_episode(trajectories.copy())   
              
                for i_batch in range(N_BATCHES):  
                    model.to_cuda()

                    batch = memory.sample(BATCH_SIZE)
                    critic_loss, actor_loss = model.update_parameters(batch, normalizer)
                    if i_batch == N_BATCHES - 1:
                        critic_losses.append(critic_loss)
                        actor_losses.append(actor_loss)

                model.update_target()

                if agent_id == 1:
                    for i_batch in range(N_BATCHES):
                        batch = memory.sample(BATCH_SIZE)
                        backward_otw_loss = model.update_backward_otw(batch, normalizer)  
                        if i_batch == N_BATCHES - 1:
                            backward_otw_losses.append(backward_otw_loss)

            # <-- end loop: i_cycle
        plot_durations(np.asarray(critic_losses), np.asarray(actor_losses))

        episode_reward_cycle = []
        episode_succeess_cycle = []
        episode_distance_cycle = []
        rollout_per_env = N_TEST_ROLLOUTS // config['n_envs']
        for i_rollout in range(rollout_per_env):
            render = config['render'] == 2 and i_episode % config['render'] == 0
            _, episode_reward, success, distance = rollout(envs_train, model, False, config, normalizer=normalizer, render=render, agent_id=agent_id, ai_object=False, rob_policy=config['rob_policy'])
                
            episode_reward_cycle.extend(episode_reward)
            episode_succeess_cycle.extend(success)
            episode_distance_cycle.extend(distance)

        render = (config['render'] == 1) and (i_episode % config['render'] == 0) and (envs_render is not None)
        if render:
            for i_rollout in range(10):
                _, _, _, _ = rollout(envs_render, model, False, config, normalizer=normalizer, render=render, agent_id=agent_id, ai_object=False, rob_policy=config['rob_policy'])
        # <-- end loop: i_rollout 
            
        ### MONITORIRNG ###
        episode_reward_all.append(episode_reward_cycle)
        episode_success_all.append(episode_succeess_cycle)
        episode_distance_all.append(episode_distance_cycle)

        episode_reward_mean.append(np.mean(episode_reward_cycle))
        episode_success_mean.append(np.mean(episode_succeess_cycle))
        episode_distance_mean.append(np.mean(episode_distance_cycle))
        plot_durations(np.asarray(episode_reward_mean), np.asarray(episode_success_mean))
        
        if best_succeess < np.mean(episode_succeess_cycle):
            bestmodel_critic = model.critics[0].state_dict()
            bestmodel_actor = model.actors[0].state_dict()
            bestmodel_normalizer = copy.deepcopy(normalizer)
            best_succeess = np.mean(episode_succeess_cycle)

        if config['verbose'] > 0:
        # Printing out
            if (i_episode+1)%1 == 0:
                print("==> Episode {} of {}".format(i_episode + 1, N_EPISODES))
                print('  | Id exp: {}'.format(config['exp_id']))
                print('  | Exp description: {}'.format(config['exp_descr']))
                print('  | Env: {}'.format(config['env_id']))
                print('  | Process pid: {}'.format(config['process_pid']))
                print('  | Running mean of total reward: {}'.format(episode_reward_mean[-1]))
                print('  | Success rate: {}'.format(episode_success_mean[-1]))
                print('  | Distance to target {}'.format(episode_distance_mean[-1]))
                print('  | Time episode: {}'.format(time.time()-episode_time_start))
                print('  | Time total: {}'.format(time.time()-total_time_start))

    # <-- end loop: i_episode

    if train and agent_id==1:
        print('Training Backward Model')
        model.to_cuda()
        for _ in range(N_EPISODES*N_CYCLES):
            for i_batch in range(N_BATCHES):
                batch = memory.sample(BATCH_SIZE)
                backward_loss = model.update_backward(batch, normalizer)  
                if i_batch == N_BATCHES - 1:
                    backward_losses.append(backward_loss)

        plot_durations(np.asarray(backward_otw_losses), np.asarray(backward_losses))

    if train:
        print('Training completed')
    else:
        print('Test completed')

    return (episode_reward_all, episode_success_all, episode_distance_all), (bestmodel_critic, bestmodel_actor, bestmodel_normalizer)

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