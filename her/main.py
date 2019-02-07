import numpy as np
import scipy as sc
import time
import imageio

import torch as K
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym

from her.algorithms import DDPG, DDPGC
from her.experience import ReplayMemory, Transition, Normalizer
from her.exploration import Noise, OUNoise
from her.utils import Saver, Summarizer, get_noise_scale, get_params, running_mean
from her.agents.basic import Actor 
from her.agents.basic import CriticReg as Critic

import pdb

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
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space
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
    noise = Noise(action_space.shape[0], sigma=0.2, eps=0.3, max_u=env.action_space.high[0])
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

    experiment_args = (env, memory, noise, config, summaries, saver, start_episode)
          
    return model, experiment_args

def rollout(env, model, noise, normalizer=None, render=False):
    trajectory = []

    # monitoring variables
    episode_reward = 0
    frames = []
    
    obs = env.reset()
    nb_step = 100#env._max_episode_steps

    for i_step in range(nb_step):

        model.to_cpu()

        obs = K.tensor(obs, dtype=K.float32).unsqueeze(0)
        # Observation normalization
        if normalizer is not None:
            obs = normalizer.preprocess_with_update(obs)

        action = model.select_action(obs, noise)

        next_obs, reward, done, info = env.step(action.squeeze(0).numpy())
        reward = K.tensor(reward, dtype=dtype).view(1,1)

        # for monitoring
        episode_reward += reward
        done = 1 if (i_step == (nb_step - 1)) else 0

        trajectory.append((obs, action, reward, K.tensor(next_obs, dtype=K.float32).unsqueeze(0), done))

        # Move to the next state
        obs = next_obs

        # Record frames
        if render:
            frames.append(env.render(mode='rgb_array')[0])

    success = 0 if (info.get('is_success') is None) else info.get('is_success')

    return trajectory, episode_reward, success, frames

def run(model, experiment_args, train=True):

    total_time_start =  time.time()

    env, memory, noise, config, summaries, saver, start_episode = experiment_args

    #normalizer = Normalizer()
    
    start_episode = start_episode if train else 0
    NUM_EPISODES = config['n_episodes'] if train else config['n_episodes_test'] 
    EPISODE_LENGTH = 100#env._max_episode_steps #config['episode_length'] if train else config['episode_length_test'] 
   
    episode_reward_all = []
    episode_success_all = []
        
    for i_episode in range(start_episode, NUM_EPISODES):
        
        episode_time_start = time.time()
        #noise.scale = get_noise_scale(i_episode, config)
        if train:
            for i_cycle in range(25):
            
                trajectories = []
                for i_rollout in range(16):
                    # Initialize the environment and state
                    trajectory, _, _, _ = rollout(env, model, noise, render=0)
                    trajectories.append(trajectory)

                for trajectory in trajectories:

                    for i_step in range(len(trajectory)):
                        obs, action, reward, next_obs, done = trajectory[i_step]                    
                        if done:
                            next_obs = None

                        memory.push(obs, action, next_obs, reward)
        
            
                for _ in range(40):
                    if len(memory) > config['batch_size']-1:

                        model.to_cuda()        
                        batch = Transition(*zip(*memory.sample(config['batch_size'])))
                        critic_loss, actor_loss = model.update_parameters(batch)

                #print(normalizer.get_stats())

            # <-- end loop: i_cycle

        episode_reward_cycle = []
        episode_succeess_cycle = []
        for i_rollout in range(10):
            # Initialize the environment and state
            render = config['render'] > 0 and i_episode % config['render'] == 0
            _, episode_reward, success, frames = rollout(env, model, noise=False, normalizer=None, render=render)
                
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