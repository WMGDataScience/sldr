import numpy as np
import scipy as sc
import time
import imageio

import torch as K
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym

from her.algorithms import DDPG, HDDPG
from her.experience import ReplayMemory, Transition
from her.exploration import Noise
from her.utils import Saver, Summarizer, get_noise_scale, get_params, running_mean
from her.agents.basic import Actor 
from her.agents.basic import Critic

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

    observation_space = env.observation_space.spaces['observation'].shape[0] + env.observation_space.spaces['desired_goal'].shape[0]
    #goal_space = gym.spaces.box.Box(low=-1, high=1, shape=(8,), dtype='float32')
    #goal_space = gym.spaces.box.Box(low=-2, high=2, shape=env.observation_space.spaces['desired_goal'].shape, dtype='float32')
    goal_space = gym.spaces.box.Box(low=-1, high=1, shape=env.observation_space.spaces['desired_goal'].shape, dtype='float32')
    action_space = env.action_space
    if env.action_space.low[0] == -1 and env.action_space.high[0] == 1:
        OUT_FUNC = K.tanh 
    elif env.action_space.low[0] == 0 and env.action_space.high[0] == 1:
        OUT_FUNC = K.sigmoid
    else:
        OUT_FUNC = K.sigmoid

    K.manual_seed(SEED)
    np.random.seed(SEED)
    
    MODEL = HDDPG

    if config['verbose'] > 1:
        # utils
        summaries = (Summarizer(config['dir_summary_train'], config['port'], config['resume']),
                    Summarizer(config['dir_summary_test'], config['port'], config['resume']))
        saver = Saver(config)
    else:
        summaries = None
        saver = None

    #exploration initialization
    noise = (Noise(action_space.shape[0], sigma=0.05, eps=0.2),
             Noise(goal_space.shape[0], sigma=0.05, eps=0.2))
    #noise = OUNoise(action_space.shape[0])

    #model initialization
    optimizer = (optim.Adam, (ACTOR_LR, CRITIC_LR)) # optimiser func, (actor_lr, critic_lr)
    loss_func = F.mse_loss
    model = MODEL(observation_space, action_space, goal_space, optimizer, 
                  Actor, Critic, loss_func, GAMMA, TAU, out_func=OUT_FUNC,
                  discrete=False, regularization=REGULARIZATION, normalized_rewards=NORMALIZED_REWARDS)
    
    if config['resume'] != '':
        for i, param in enumerate(save_dict['model_params']):
            model.entities[i].load_state_dict(param)
    
    #memory initilization
    memory = [ReplayMemory(MEM_SIZE), ReplayMemory(MEM_SIZE)]

    experiment_args = (env, memory, noise, config, summaries, saver, start_episode)
          
    return model, experiment_args

def rollout(env, model, noise, mask, ratio, mapping, C=25, render=False, sparse_rewards=True):
    trajectory = [[], []]

    # monitoring variables
    episode_reward = [0, 0]
    frames = []
    
    state = env.reset()
    success = 0
    
    for i_step in range(env._max_episode_steps):

        model.to_cpu()

        obs = K.tensor(state['observation'], dtype=K.float32).unsqueeze(0)
        goal = K.tensor(state['desired_goal'], dtype=K.float32).unsqueeze(0)
        obs = K.cat([obs, goal], dim=-1)

        obs_f = obs*mask
        obs_s = obs#*(1-mask)

        if i_step % C  == 0:
            state_init = state.copy()
            #goal_f = model.select_goal(obs_s, noise[1])
            #pdb.set_trace()
            goal_f = goal
            #goal_f = goal - obs[:,0:3]
            #goal_f *= (ratio[0:3]*C*2)
            #goal_f += obs_s[:,0:3]
            #pdb.set_trace()
            #goal_f *= (ratio[K.tensor(mask, dtype=K.uint8)]*C)
            #goal_f += obs_s[:,K.tensor(mask, dtype=K.uint8)]            
            cumm_reward = K.zeros((1,1), dtype=K.float32)

        action = model.select_action(K.cat([obs_f, goal_f], dim=-1), noise[0])

        next_state, reward, done, info = env.step(action.squeeze(0).numpy())

        next_obs = K.tensor(next_state['observation'], dtype=K.float32).unsqueeze(0)
        #next_goal = K.tensor(next_state['desired_goal'], dtype=K.float32).unsqueeze(0)
        next_obs = K.cat([next_obs, goal], dim=-1)

        next_obs_f = next_obs*mask
        #next_obs_s = next_obs*(1-mask)

        reward = K.tensor(reward, dtype=K.float32).view(1,1)
        cumm_reward += reward
        #old_distance = F.mse_loss(goal_f, mapping(obs_f, mask)).view(1,1)
        #new_distance = F.mse_loss(goal_f, mapping(next_obs_f, mask)).view(1,1)
        #old_distance = goal_f - mapping(obs_f, mask)
        #new_distance = goal_f - mapping(next_obs_f, mask)
        #old_distance /= (ratio[K.tensor(mask, dtype=K.uint8)]*C)
        #new_distance /= (ratio[K.tensor(mask, dtype=K.uint8)]*C)

        #old_distance = K.tensor(np.linalg.norm(old_distance, axis=-1), dtype=K.float32).view(1,1)
        #new_distance = K.tensor(np.linalg.norm(new_distance, axis=-1), dtype=K.float32).view(1,1)
        distance = K.tensor(np.linalg.norm(mapping(goal_f) - mapping(next_obs_f), axis=-1), dtype=K.float32).view(1,1)
        #distance = K.tensor(np.linalg.norm(mapping(goal_f) - 0, axis=-1), dtype=K.float32).view(1,1)
        success = distance < 0.05
        if sparse_rewards:
            intr_reward = - K.tensor(distance > 0.05, dtype=K.float32)
        else:
            intr_reward = old_distance - new_distance
        #K.sqrt(K.pow((goal_f - mapping(next_obs_f)), 2).sum(-1))
        
        #print(K.sqrt(K.pow((goal_f - mapping(next_obs_f)), 2).sum(-1)))
        #print(F.mse_loss(goal_f, mapping(next_obs_f)).view(1,1))

        # for monitoring
        episode_reward[0] = episode_reward[0] + intr_reward
        episode_reward[1] = episode_reward[1] + reward

        trajectory[0].append((state, action, intr_reward, next_state, goal_f, done))
        if (i_step+1) % C  == 0:
            trajectory[1].append((state_init, None, cumm_reward, next_state, goal_f, done))

        # Move to the next state
        state = next_state

        # Record frames
        if render:
            frames.append(env.render(mode='rgb_array')[0])

    return trajectory, episode_reward, (success, info['is_success']), frames, 
  
# def get_of_mask(env, nb_episodes=400, nb_steps=50, threshold=0.08):
#     states = []
#     actions = []
#     for _ in range(nb_episodes):
#         env.reset()
#         for _ in range(nb_steps):    
#             action = env.action_space.sample()
#             state, _, _, _ = env.step(action)
#             observation = state['observation']
#             desired_goal = state['desired_goal']
#             state = np.concatenate([observation, desired_goal])
#             states.append(state)
#             actions.append(action)

#     states = np.asarray(states)
#     actions = np.asarray(actions)

#     corr = np.zeros((states.shape[1], actions.shape[1]))
#     for i in range(states.shape[1]):
#         for j in range(actions.shape[1]):    
#             corr[i,j] = np.corrcoef(states[:,i], actions[:,j])[0,1]
    
#     mask = (K.tensor(abs(corr))>threshold).any(dim=1)
    
#     return mask

def get_of_mask(env, nb_episodes=400, nb_steps=50, threshold=0.08):
    states = []
    actions = []
    for _ in range(nb_episodes):
        old_state = env.reset()
        for _ in range(nb_steps):    
            action = env.action_space.sample()
            new_state, _, _, _ = env.step(action)
            observation_diff = new_state['observation'] - old_state['observation']
            desired_goal_diff = new_state['desired_goal'] - old_state['desired_goal'] 
            state = np.concatenate([observation_diff, desired_goal_diff])
            states.append(state)
            actions.append(action)
            old_state = new_state.copy()

    states = np.asarray(states)
    actions = np.asarray(actions)

    corr = np.zeros((states.shape[1], actions.shape[1]))
    for i in range(states.shape[1]):
        for j in range(actions.shape[1]):    
            corr[i,j] = np.corrcoef(states[:,i], actions[:,j])[0,1]
            
    ratio = np.zeros((states.shape[1], actions.shape[1]))
    for i in range(states.shape[1]):
        for j in range(actions.shape[1]):    
            ratio[i,j] = np.median(states[:,i]/actions[:,j])
    
    mask = (K.tensor(abs(corr))>threshold).any(dim=1)
    
    return K.tensor(mask, dtype=K.float32), K.max(K.tensor(ratio, dtype=K.float32), dim=-1)[0]

def mapping(x, mask=None):
    if mask is None:
        return x[:,0:3]
    else:
        return x[:, K.tensor(mask, dtype=K.uint8)]

def run(model, experiment_args, train=True):

    total_time_start =  time.time()

    env, memory, noise, config, summaries, saver, start_episode = experiment_args
    
    start_episode = start_episode if train else 0
    NUM_EPISODES = config['n_episodes'] if train else config['n_episodes_test'] 
    EPISODE_LENGTH = env._max_episode_steps #config['episode_length'] if train else config['episode_length_test'] 
    
    episode_intr_reward_all = []
    episode_reward_all = []
    episode_intr_success_all = []
    episode_success_all = []

    mask, ratio = get_of_mask(env, nb_episodes=40)

    hier_C = 1
    sparse_rewards = True
        
    for i_episode in range(start_episode, NUM_EPISODES*5):
        
        episode_time_start = time.time()
        #noise.scale = get_noise_scale(i_episode, config)
        if train:
            for i_cycle in range(10):
            
                trajectories = []
                for i_rollout in range(16):
                    # Initialize the environment and state
                    trajectory, _, _, _ = rollout(env, model, noise, mask, ratio, mapping, hier_C, render=0, sparse_rewards=sparse_rewards)
                    trajectories.append(trajectory)

                for trajectory in trajectories:

                    for i_step in range(len(trajectory[0])):
                        state, action, reward, next_state, goal_f, done = trajectory[0][i_step]
                        
                        obs = K.tensor(state['observation'], dtype=K.float32).unsqueeze(0)
                        goal = K.tensor(state['desired_goal'], dtype=K.float32).unsqueeze(0)
                        obs = K.cat([obs, goal], dim=-1)
                        obs_f = obs*mask

                        next_obs = K.tensor(next_state['observation'], dtype=K.float32).unsqueeze(0)
                        next_obs = K.cat([next_obs, goal], dim=-1)
                        next_obs_f = next_obs*mask
                        next_achieved_f = mapping(next_obs_f)
                        
                        # regular sample
                        obs_goal_f = K.cat([obs_f, goal_f], dim=-1)
                        if done:
                            next_obs_goal_f = None
                        else:
                            next_obs_goal_f = K.cat([next_obs_f, goal_f], dim=-1)
                        
                        memory[0].push(obs_goal_f, action, next_obs_goal_f, reward)

                        # HER sample 
                        for _ in range(4):
                            future = np.random.randint(i_step, len(trajectory[0]))
                            _, _, _, next_state, _, _ = trajectory[0][future]

                            aux_goal_f = mapping(K.cat([K.tensor(next_state['observation'], dtype=K.float32).unsqueeze(0), goal], dim=-1)*mask)

                            obs_goal_f = K.cat([obs_f, aux_goal_f], dim=-1)#*(1-mask)
                            if done:
                                next_obs_goal_f = None
                            else:
                                next_obs_goal_f = K.cat([next_obs_f, aux_goal_f], dim=-1)#*(1-mask)
                                
                            reward = env.compute_reward(next_achieved_f, aux_goal_f, None)
                            reward = K.tensor(reward, dtype=dtype).view(1,1)
                            
                            memory[0].push(obs_goal_f, action, next_obs_goal_f, reward)

                    for i_step in range(len(trajectory[1])):
                        state, _, reward, next_state, goal_f, done = trajectory[1][i_step]
                        
                        obs = K.tensor(state['observation'], dtype=K.float32).unsqueeze(0)
                        goal = K.tensor(state['desired_goal'], dtype=K.float32).unsqueeze(0)

                        next_obs = K.tensor(next_state['observation'], dtype=K.float32).unsqueeze(0)
                        next_achieved = K.tensor(next_state['achieved_goal'], dtype=K.float32).unsqueeze(0)
                        
                        # regular sample
                        obs_goal_s = K.cat([obs, goal], dim=-1)#*(1-mask)
                        if done:
                            next_obs_goal_s = None
                        else:
                            next_obs_goal_s = K.cat([next_obs, goal], dim=-1)#*(1-mask)
                        
                        memory[1].push(obs_goal_s, goal_f, next_obs_goal_s, reward)
                        # HER sample 
                        for _ in range(4):
                            future = np.random.randint(i_step, len(trajectory[1]))
                            _, _, _, next_state, _, _ = trajectory[1][future]
                            aux_goal = K.tensor(next_state['achieved_goal'], dtype=K.float32).unsqueeze(0)

                            obs_goal_s = K.cat([obs, aux_goal], dim=-1)#*(1-mask)
                            if done:
                                next_obs_goal_s = None
                            else:
                                next_obs_goal_s = K.cat([next_obs, aux_goal], dim=-1)#*(1-mask)
                                
                            reward = env.compute_reward(next_achieved, aux_goal, None)
                            #reward = K.tensor(reward, dtype=dtype).view(1,1)*(hier_C/2) - hier_C/2
                            reward = K.tensor(reward, dtype=dtype).view(1,1)
                            
                            memory[1].push(obs_goal_s, goal_f, next_obs_goal_s, reward)

                    # <-- end loop: i_step
                # <-- end loop: i_rollout 

                for _ in range(40):
                    if len(memory[0]) > config['batch_size']-1 and len(memory[1]) > config['batch_size']-1:  
                        batch = []
                        model.to_cuda()  
                        batch.append(Transition(*zip(*memory[0].sample(config['batch_size']))))
                        batch.append(Transition(*zip(*memory[1].sample(config['batch_size']))))
                        critic_loss, actor_loss = model.update_parameters(batch)

            # <-- end loop: i_cycle

        episode_intr_reward_cycle = []
        episode_reward_cycle = []
        episode_intr_success_cycle = []
        episode_succeess_cycle = []
        for i_rollout in range(10):
            # Initialize the environment and state
            render = config['render'] > 0 and i_episode % config['render'] == 0
            _, episode_reward, success, frames = rollout(env, model, noise=(False, False), mask=mask, ratio=ratio, mapping=mapping, C=hier_C, render=render, sparse_rewards=sparse_rewards)
                
            # Save gif
            dir_monitor = config['dir_monitor_train'] if train else config['dir_monitor_test']
            if config['render'] > 0 and i_episode % config['render'] == 0:
                imageio.mimsave('{}/{}.gif'.format(dir_monitor, i_episode), frames)

            
            episode_intr_reward_cycle.append(episode_reward[0])
            episode_reward_cycle.append(episode_reward[1])
            episode_intr_success_cycle.append(success[0])
            episode_succeess_cycle.append(success[1])
        # <-- end loop: i_rollout 
            
        ### MONITORIRNG ###
        episode_intr_reward_all.append(np.mean(episode_intr_reward_cycle))
        episode_reward_all.append(np.mean(episode_reward_cycle))
        episode_intr_success_all.append(np.mean(episode_intr_success_cycle))
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
                print('  | Running mean of total intrinsic reward: {}'.format(episode_intr_reward_all[-1]))
                print('  | Success rate: {}'.format(episode_success_all[-1]))
                print('  | Intrinsic success rate: {}'.format(episode_intr_success_all[-1]))
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