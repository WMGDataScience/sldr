import numpy as np
import scipy as sc
import time
import imageio

import torch as K
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym

from macomm.algorithms3 import DDPG, DDPGC
from macomm.experience import ReplayMemory, Transition
from macomm.exploration import OUNoise
from macomm.utils3 import Saver, Summarizer, get_noise_scale, get_params, running_mean

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
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space
    if env.action_space.low[0] == -1 and env.action_space.high[0] == 1:
        OUT_FUNC = K.tanh 
    elif env.action_space.low[0] == 0 and env.action_space.high[0] == 1:
        OUT_FUNC = K.sigmoid
    else:
        OUT_FUNC = K.sigmoid


    env.seed(SEED)
    K.manual_seed(SEED)
    np.random.seed(SEED)
    
    if config['agent_alg'] == 'DDPG':
        MODEL = DDPG
    elif config['agent_alg'] == 'DDPGC':
        MODEL = DDPGC
        
    if config['agent_type'] == 'basic':
        from macomm.agents3.basic import Actor 
        from macomm.agents3.basic import Critic_Chaos as Critic 
    elif config['agent_type'] == 'deep':
        from macomm.agents3.deep import Actor, Critic
    
    if config['verbose'] > 1:
        # utils
        summaries = (Summarizer(config['dir_summary_train'], config['port'], config['resume']),
                    Summarizer(config['dir_summary_test'], config['port'], config['resume']))
        saver = Saver(config)
    else:
        summaries = None
        saver = None

    #exploration initialization
    ounoise = OUNoise(action_space.shape[0])

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

    experiment_args = (env, memory, ounoise, config, summaries, saver, start_episode)
          
    return model, experiment_args

def run(model, experiment_args, train=True):

    total_time_start =  time.time()

    env, memory, ounoise, config, summaries, saver, start_episode = experiment_args
    
    start_episode = start_episode if train else 0
    NUM_EPISODES = config['n_episodes'] if train else config['n_episodes_test'] 
    EPISODE_LENGTH = config['episode_length'] if train else config['episode_length_test'] 
    
    t = 0
    episode_reward_all = []
        
    for i_episode in range(start_episode, NUM_EPISODES):
        
        episode_time_start = time.time()
        
        frames = []
        
        # Initialize the environment and state
        observation = env.reset()
        observation = K.tensor(observation, dtype=K.float32).unsqueeze(0)

        ounoise.scale = get_noise_scale(i_episode, config)
        
        # monitoring variables
        episode_reward = 0
        
        for i_step in range(EPISODE_LENGTH):

            model.to_cpu()
            action = model.select_action(observation, ounoise if train else False)

            next_observation, reward, done, info = env.step(action.squeeze(0))
            next_observation = K.tensor(next_observation, dtype=dtype).unsqueeze(0)
            reward = K.tensor(reward, dtype=dtype).view(1,1)

            # for monitoring
            episode_reward += reward

            # if it is the last step we don't need next obs
            if i_step == EPISODE_LENGTH-1:
                next_observation = None

            # Store the transition in memory
            if train:
                memory.push(observation, action, next_observation, reward)

            # Move to the next state
            observation = next_observation
            t += 1
            
            # Use experience replay and train the model
            if train:
                if len(memory) > config['batch_size']-1 and t%config['steps_per_update'] == 0:
                    model.to_cuda()        
                    batch = Transition(*zip(*memory.sample(config['batch_size'])))
                    critic_loss, actor_loss = model.update_parameters(batch)

                        
            # Record frames
            if config['render'] > 0 and i_episode % config['render'] == 0:
                frames.append(env.render(mode='rgb_array')[0]) 

        # <-- end loop: i_step 

        ### MONITORIRNG ###

        episode_reward_all.append(episode_reward)
        
        if config['verbose'] > 0:
        # Printing out
            if (i_episode+1)%100 == 0:
                print("==> Episode {} of {}".format(i_episode + 1, NUM_EPISODES))
                print('  | Id exp: {}'.format(config['exp_id']))
                print('  | Exp description: {}'.format(config['exp_descr']))
                print('  | Env: {}'.format(config['env_id']))
                print('  | Process pid: {}'.format(config['process_pid']))
                print('  | Tensorboard port: {}'.format(config['port']))
                print('  | Episode total reward: {}'.format(episode_reward))
                print('  | Running mean of total reward: {}'.format(running_mean(episode_reward_all)[-1]))
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
        

        # Save gif
        dir_monitor = config['dir_monitor_train'] if train else config['dir_monitor_test']
        if config['render'] > 0 and i_episode % config['render'] == 0:
            if config['env_id'] == 'waterworld':
                imageio.mimsave('{}/{}.gif'.format(dir_monitor, i_episode), frames[0::3])
            else:
                imageio.mimsave('{}/{}.gif'.format(dir_monitor, i_episode), frames)
            
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