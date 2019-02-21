import numpy as np
import scipy as sc
import time
import imageio

import torch as K
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym_wmgds as gym

from her.algorithms import DDPG
from her.experience import ReplayMemory, Transition, Normalizer
from her.exploration import Noise
from her.utils import Saver, Summarizer, get_noise_scale, get_params, running_mean
from her.agents.basic import Actor 
from her.agents.basic import Critic

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

    env = gym.make(ENV_NAME)
    env.seed(SEED)

    observation_space = env.observation_space.spaces['observation'].shape[0] + env.observation_space.spaces['desired_goal'].shape[0]
    action_space = env.action_space
    if env.action_space.low[0] == -1 and env.action_space.high[0] == 1:
        OUT_FUNC = K.tanh 
    elif env.action_space.low[0] == 0 and env.action_space.high[0] == 1:
        OUT_FUNC = K.sigmoid
    else:
        OUT_FUNC = K.sigmoid

    K.manual_seed(SEED)
    np.random.seed(SEED)
    
    MODEL = DDPG

    if config['verbose'] > 1:
        # utils
        summaries = (Summarizer(config['dir_summary_train'], config['port'], config['resume']),
                    Summarizer(config['dir_summary_test'], config['port'], config['resume']))
        saver = Saver(config)
    else:
        summaries = None
        saver = None

    #exploration initialization
    noise = Noise(action_space.shape[0], sigma=0.2, eps=0.3)
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
    memory = ReplayMemory(MEM_SIZE)

    normalizer = (Normalizer(), None)

    experiment_args = (env, memory, noise, config, summaries, saver, start_episode, normalizer)
          
    return model, experiment_args

def rollout(env, model, noise, normalizer=None, render=False, nb_objects=1):
    trajectory = []

    # monitoring variables
    episode_reward = 0
    frames = []
    
    env.env.nb_objects = nb_objects
    state = env.reset()
    achieved_init = state['achieved_goal']
    
    for i_step in range(env._max_episode_steps):

        model.to_cpu()

        obs = K.tensor(state['observation'][0], dtype=K.float32).unsqueeze(0)
        goal = K.tensor(state['desired_goal'], dtype=K.float32).unsqueeze(0)

        obs_goal = K.cat([obs, goal], dim=-1)
        # Observation normalization
        if normalizer[0] is not None:
            obs_goal = normalizer[0].preprocess_with_update(obs_goal)
            #_ = normalizer.preprocess_with_update(obs_goal)

        action = model.select_action(obs_goal, noise)

        next_state, reward, done, info = env.step(action.squeeze(0).numpy())
        reward = K.tensor(reward, dtype=dtype).view(1,1)
        if normalizer[1] is not None:
            reward = normalizer[1].preprocess_with_update(reward)

        # for monitoring
        episode_reward += reward
        trajectory.append((state.copy(), action, reward, next_state.copy(), done))

        # Move to the next state
        state = next_state

        # Record frames
        if render:
            frames.append(env.render(mode='rgb_array')[0])
        
        #achieved_final = state['achieved_goal']
        #moved_step[np.where(np.logical_and(np.linalg.norm((achieved_init - achieved_final), axis=1) > 1e-4, moved_step==-1))[0]] = i_step

    achieved_final = state['achieved_goal']
    moved_index = np.concatenate((K.zeros(1), np.where((achieved_init[1::] != achieved_final[1::]).all(axis=1))[0]+1)).astype(int)
    #moved_index = np.where(np.linalg.norm((achieved_init - achieved_final), axis=1) > 1e-4)[0]

    if render:
        print(moved_index)


    return trajectory, episode_reward, info['is_success'], frames, moved_index

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
    
    max_nb_objects = config['max_nb_objects']
        
    for i_episode in range(start_episode, NUM_EPISODES):
        
        episode_time_start = time.time()
        #noise.scale = get_noise_scale(i_episode, config)
        if train:
            for i_cycle in range(50):
            
                trajectories = []
                moved_indices = []
                for i_rollout in range(1):
                    # Initialize the environment and state
                    nb_objects = np.random.randint(1, max_nb_objects+1)
                    trajectory, _, _, _, moved_index = rollout(env, model, noise, normalizer, render=(i_cycle%10==1000), nb_objects=nb_objects)
                    trajectories.append(trajectory)
                    moved_indices.append(moved_index)

                for trajectory, moved_index in zip(trajectories, moved_indices):

                    T = len(trajectory) - 1
                    t_samples = np.random.randint(T, size=(max_nb_objects,T))
                    future_offset = np.random.uniform(size=(max_nb_objects,T)) * (T - t_samples)
                    future_offset = future_offset.astype(int)
                    her_sample = (np.random.uniform(size=(max_nb_objects,T)) < 0.8)
                    future_t = (t_samples + 1 + future_offset)
                    
                    # for i_step in range(len(trajectory)):
                    #     state, action, reward, next_state, done = trajectory[i_step]
                        
                    #     obs = K.tensor(state['observation'][0], dtype=K.float32).unsqueeze(0)
                    #     goal = K.tensor(state['desired_goal'], dtype=K.float32).unsqueeze(0)

                    #     next_obs = K.tensor(next_state['observation'][0], dtype=K.float32).unsqueeze(0)
                    #     next_achieved = K.tensor(next_state['achieved_goal'][0], dtype=K.float32).unsqueeze(0)
                        
                    #     # regular sample
                    #     obs_goal = K.cat([obs, goal], dim=-1)
                    #     if done:
                    #         next_obs_goal = None
                    #     else:
                    #         next_obs_goal = K.cat([next_obs, goal], dim=-1)
                        
                    #     memory.push(obs_goal, action, next_obs_goal, reward)
                    # <-- end loop: i_step
                    
                    for i_object in moved_index:
                        for i_step in t_samples[i_object]:
                            state, action, reward, next_state, done = trajectory[i_step]

                            #pdb.set_trace()
                            obs = K.tensor(state['observation'][i_object], dtype=K.float32).unsqueeze(0)
                            goal = K.tensor(state['desired_goal'], dtype=K.float32).unsqueeze(0)
                                
                            next_obs = K.tensor(next_state['observation'][i_object], dtype=K.float32).unsqueeze(0)
                            next_achieved = K.tensor(next_state['achieved_goal'][i_object], dtype=K.float32).unsqueeze(0)

                            if her_sample[i_object, i_step]:
                                _, _, _, next_state, _ = trajectory[future_t[i_object, i_step]]
                                aux_goal = K.tensor(next_state['achieved_goal'][i_object], dtype=K.float32).unsqueeze(0)
                                obs_goal = K.cat([obs, aux_goal], dim=-1)

                                if done:
                                    next_obs_goal = None
                                else:
                                    next_obs_goal = K.cat([next_obs, aux_goal], dim=-1)
                                        
                                reward = env.compute_reward(next_achieved, aux_goal, None)
                                reward = K.tensor(reward, dtype=dtype).view(1,1)
                                if normalizer[1] is not None:
                                    reward = normalizer[1].preprocess_with_update(reward)
                            else:
                                # regular sample
                                obs_goal = K.cat([obs, goal], dim=-1)
                                if done:
                                    next_obs_goal = None
                                else:
                                    next_obs_goal = K.cat([next_obs, goal], dim=-1)
                                    
                            memory.push(obs_goal, action, next_obs_goal, reward)    
                        # <-- end loop: i_step
                    # <-- end loop: i_object  
                # <-- end loop: i_rollout 

                for i_batch in range(40):
                    if len(memory) > config['batch_size']-1:
                        
                        model.to_cuda()  
                        batch = Transition(*zip(*memory.sample(config['batch_size'])))
                        critic_loss, actor_loss = model.update_parameters(batch, normalizer)

                        if i_batch == 39:
                            critic_losses.append(critic_loss)
                            actor_losses.append(actor_loss)

                #print(normalizer.get_stats())

            # <-- end loop: i_cycle
        
        plot_durations(np.asarray(critic_losses), np.asarray(actor_losses))
        #pdb.set_trace()
        episode_reward_cycle = []
        episode_succeess_cycle = []
        for i_rollout in range(10):
            # Initialize the environment and state
            render = config['render'] > 0 and i_episode % config['render'] == 0
            _, episode_reward, success, frames, _ = rollout(env, model, noise=False, normalizer=normalizer, render=render, nb_objects=1)
                
            # Save gif
            dir_monitor = config['dir_monitor_train'] if train else config['dir_monitor_test']
            if config['render'] > 0 and i_episode % config['render'] == 0:
                imageio.mimsave('{}/{}.gif'.format(dir_monitor, i_episode), frames)

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
    
            #if (i_episode+1)%100 == 0:
            #    summary = summaries[0] if train else summaries[1]
            #    summary.update_log(i_episode, 
            #                    episode_reward, 
            #                    list(episode_reward.reshape(-1,)), 
            #                    critic_loss        = critic_loss, 
            #                    actor_loss         = actor_loss,
            #                    to_save            = to_save
            #                    )
            
    # <-- end loop: i_episode
    if train:
        print('Training completed')
    else:
        print('Test completed')
    
    return episode_reward_all

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