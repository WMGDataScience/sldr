import matplotlib
matplotlib.use('agg')
import numpy as np
import os, sys, threading, shutil, argparse, json, time, datetime, pickle
import pdb
import matplotlib.pylab as plt

from pathlib import Path
from tensorboardX import SummaryWriter

import torch as K


def running_mean(durations, threshold=100):
    durations_t = K.tensor(durations, dtype=K.float32)
    # take 100 episode averages and plot them too
    if len(durations_t) >= threshold:
        means = durations_t.unfold(0, threshold, 1).mean(1).view(-1)
        means = K.cat((K.zeros(threshold-1), means)).numpy()
    return means


def get_noise_scale(i_episode, config):
    
    coef = (config['init_noise_scale'] - config['final_noise_scale'])
    decay = max(0, config['n_exploration_eps'] - i_episode) / config['n_exploration_eps']
    offset = config['final_noise_scale']
    
    scale = coef * decay + offset
    
    return scale

def plot_durations(durations):
    plt.figure(2)
    plt.clf()
    durations_t = K.tensor(durations, dtype=K.float32)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = K.cat((K.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def get_obj_obs(obs, goal, n_object):
    
    l_x = 2
    l_y = 3
    l_z = 1
    
    l_a = 3
    l_b = l_a+3*l_x*n_object
    l_c = l_b+2
    l_d = l_c+3*l_y*n_object
    l_e = l_d+5
    
    ind1 = slice(0,l_a)
    ind2 = slice(l_a,l_b)
    ind3 = slice(l_b,l_c)
    ind4 = slice(l_c,l_d)
    ind5 = slice(l_d,l_e)
    
    obj_obs_all = []
    for i in range (n_object):
        obj_obs = K.cat((obs[:,ind1],
                         obs[:,ind2].view(-1,l_x,n_object,3)[:,:,i,:].contiguous().view(-1,3*l_x),
                         obs[:,ind3],
                         obs[:,ind4].view(-1,l_y,n_object,3)[:,:,i,:].contiguous().view(-1,3*l_y),
                         obs[:,ind5],
                         goal.view(-1,l_z,n_object,3)[:,:,i,:].contiguous().view(-1,3*l_z)), dim=-1)
        
        obj_obs_all.append(obj_obs)
        
    obj_obs_all = K.stack(obj_obs_all, dim=-1)
    
    return obj_obs_all

def get_rob_obs(obj_obs_all, n_object):
    
    l_x = 2
    l_y = 3
    l_z = 1

    l_a = 3
    l_b = l_a+3*l_x*1
    l_c = l_b+2
    l_d = l_c+3*l_y*1
    l_e = l_d+5
    l_f = l_e+3*l_z*1
    
    ind1 = slice(0,l_a)
    ind2 = slice(l_a,l_b)
    ind3 = slice(l_b,l_c)
    ind4 = slice(l_c,l_d)
    ind5 = slice(l_d,l_e)
    ind6 = slice(l_e,l_f)

    obj = []
    obj.append(obj_obs_all[:,ind1,0])
    obj.append(K.cat([obj_obs_all[:,ind2,i].view(-1,l_x,1,3) for i in range(n_object)], dim=2).view(-1, l_x*n_object*3))
    obj.append(obj_obs_all[:,ind3,0])
    obj.append(K.cat([obj_obs_all[:,ind4,i].view(-1,l_y,1,3) for i in range(n_object)], dim=2).view(-1, l_y*n_object*3))
    obj.append(obj_obs_all[:,ind5,0])

    obj = K.cat(obj, dim=-1)
    goal = K.cat([obj_obs_all[:,ind6,i].view(-1,l_z,1,3) for i in range(n_object)], dim=2).view(-1, l_z*n_object*3)
    
    return K.cat([obj, goal], dim=-1)


def get_params(args=[], verbose=False):

    print("\n\n\n\n")
    print("==============================")
    print("Acquiring Arguments")
    print("==============================")


    parser = argparse.ArgumentParser(description='Arguments')

    # positional
    parser.add_argument("--env_id", default='simple_spread', help="Name of environment")

    # general settings
    parser.add_argument('--random_seed', default=1,type=int,
                        help='random seed for repeatability')
    parser.add_argument("--buffer_length", default=int(1e6), type=int,
                        help="replay memory buffer capacity")
    parser.add_argument("--n_episodes", default=200, type=int,
                        help="number of episodes")
    parser.add_argument("--n_episodes_test", default=1000, type=int,
                        help="number of episodes")
    parser.add_argument("--episode_length", default=50, type=int,
                        help="number of steps for episode")
    parser.add_argument("--episode_length_test", default=50, type=int,
                        help="number of steps for episode")
    parser.add_argument("--steps_per_update", default=100, type=int,
                        help="target networks update frequency")
    parser.add_argument("--batch_size", default=256, type=int,
                        help="batch size for model training")
    parser.add_argument("--n_exploration_eps", default=-1, type=int,
                        help="exploration epsilon, -1: n_episodes")
    parser.add_argument("--init_noise_scale", default=0.3, type=float,
                        help="noise initialization")
    parser.add_argument("--final_noise_scale", default=0.0, type=float,
                        help="noise stop updates value")
    parser.add_argument("--save_epochs", default=5000, type=int,
                        help="save model interval")
    parser.add_argument("--plcy_lr", default=0.001, type=float,
                        help="learning rate")
    parser.add_argument("--crtc_lr", default=0.001, type=float,
                        help="learning rate")
    parser.add_argument("--tau", default=0.05, type=float,
                        help="soft update parameter")
    parser.add_argument("--gamma", default=0.95, type=float,
                        help="discount factor")
    parser.add_argument("--agent_alg",
                        default="DDPG", type=str,
                        choices=['DDPG', 'MADDPG', 'MADDPG_R', 'MADDPG_RAE',
                                 'DDPG_BD', 'MADDPG_BD', 'PPO_BD', 'DDPG_BD_v2'])
    parser.add_argument("--device", default='cuda',
                        choices=['cpu','cuda'], 
                        help="device type")
    parser.add_argument("--plcy_hidden_dim", default=256, type=int, 
                        help="actor hidden state dimension")
    parser.add_argument("--crtc_hidden_dim", default=256, type=int, 
                        help="critic hidden state dimension")      

    parser.add_argument("--agent_type", default='basic',
                        choices=['basic', 'deep'], help="agent type")
    # path arguments
    parser.add_argument('--exp_id', default='no_id',
                        help='experiment ID')
    parser.add_argument('--dir_base', default='./experiments',
                        help='path of the experiment directory')
    parser.add_argument('--port', default=0, type=int,\
                         help='tensorboardX port')
    parser.add_argument('--exp_descr', default='',
                         help='short experiment description')

    # experiment modality
    parser.add_argument('--resume', default='',
                        help='path in case resume is needed')
    parser.add_argument('--expmode', default='normal',
                        help='fast exp mode is usually used to try is code run')
    parser.add_argument("--render", default=0, type=int,
                        help="epochs interval for rendering, 0: no rendering")
    parser.add_argument("--render_color_change", default="False",
                        choices=['True', 'False'],
                        help="Changing the color of leader during rendering") 
    parser.add_argument("--benchmark", action="store_true",
                        help="benchmark mode")
    parser.add_argument("--regularization", default="True",
                        choices=['True', 'False'],
                        help="Applying regulation to action preactivations")
    parser.add_argument("--reward_normalization", default="False",
                        choices=['True', 'False'],
                        help="Normalizing the rewards")
    parser.add_argument("--discrete_action", default="False",
                        choices=['True', 'False'],
                        help="discrete actions")

    parser.add_argument('--verbose', default=1, type=int,\
                         help='monitoring level')

    parser.add_argument('--max_nb_objects', default=1, type=int,\
                         help='number of objects to be used in the training')

    parser.add_argument('--ai_object_rate', default=0.00, type=float,\
                         help='the probability of intelligent object')

    parser.add_argument('--obj_action_type', default=[0,1,2], type=list,
                         help='the indices of objects actions')

    parser.add_argument("--observe_obj_grp", default="False",
                        choices=['True', 'False'],
                        help="wheather or not robot can observe object type") 

    parser.add_argument("--obj_range", default=0.15, type=float,
                        help="placement range of the objects")

    parser.add_argument('--ai_object_fine_tune_rate', default=10, type=int,\
                         help='number of robot updates for each fine object tune')

    parser.add_argument('--rob_policy', default=[0., 0.], type=list,
                         help='robot policy while training the objects')  

    parser.add_argument('--n_cycles', default=50, type=int,\
                         help='number of cycles per iteration')

    parser.add_argument('--n_rollouts', default=38, type=int,\
                         help='number of rollouts per cycle')

    parser.add_argument('--n_batches', default=40, type=int,\
                         help='number of batch updates per cycle')

    parser.add_argument('--n_bd_batches', default=1000000, type=int,\
                         help='number of batch updates per cycle')
    
    parser.add_argument('--n_test_rollouts', default=380, type=int,\
                         help='number of test rollouts per cycle')

    parser.add_argument('--n_envs', default=38, type=int,\
                         help='number of envs')

    parser.add_argument('--use_gae', default="True",
                        choices=['True', 'False'],
                        help="wheather or not robot can observe object type") 

    parser.add_argument("--gae_lambda", default=0.95, type=float,
                        help="PPO lambda factor")

    parser.add_argument('--ppo_epoch', default=10, type=int,\
                         help='PPO epoch')

    parser.add_argument("--entropy_coef", default=0, type=float,
                        help="PPO entropy_coef")

    parser.add_argument("--value_loss_coef", default=0.5, type=float,
                        help="PPO value loss coef")

    parser.add_argument('--clip_param', default=0.2, type=float,
                        help='ppo clip parameter (default: 0.2)')

    parser.add_argument('--eps', type=float, default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')

    parser.add_argument('--max_grad_norm',type=float,default=0.5,
        help='max norm of gradients (default: 0.5)')

    parser.add_argument("--masked_with_r", default="False",
                        choices=['True', 'False'],
                        help="Masking intrinsic rewards with true reward")

    parser.add_argument('--pred_th', type=float, default=0.0002,
                        help='prediction error threshold)')

    parser.add_argument('--clip_Q_neg', default=0, type=int,\
                         help='Negative boundary for Target Q')

    parser.add_argument("--change_stack_order", default="False",
                        choices=['True', 'False'],
                        help="Whether or not to change the object order in stacking")

    parser.add_argument("--use_step_reward_fun", default="False",
                        choices=['True', 'False'],
                        help="Whether or not to use step reward instead of sparse")

    parser.add_argument('--train_stack_prob', default=0.5, type=float,
                        help='stacking task probability during training (default: 0.5)')

    parser.add_argument('--test_stack_prob', default=1.0, type=float,
                        help='stacking task probability during testing (default: 1.0)')

    parser.add_argument("--train_robot_backward", default="False",
                        choices=['True', 'False'],
                        help="Whether or not to train backward model for the robot")
    
    parser.add_argument('--nb_objects_in_prebot', default=2, type=int,\
                         help='number of objects used in the training of prebot')
    
    parser.add_argument('--nb_objects_to_ignore', default=0, type=int,\
                         help='number of objects to be ignored in aux rewards')
    # acquire in a dict
    config = parser.parse_args(args)
    args   = vars(config)

    # set arguments which need dependencies
    dir_exp_name = '{}_{}_{}_{}'.format(str([datetime.date.today()][0]),
                                  args['env_id'],
                                  args['agent_type'],
                                  args['exp_id'])

    args['dir_exp'] = '{}/{}'.format(args['dir_base'],dir_exp_name)
    args['dir_saved_models'] = '{}/saved_models'.format(args['dir_exp'])
    args['dir_summary_train'] = '{}/summary/train'.format(args['dir_exp'])
    args['dir_summary_test'] = '{}/summary/test'.format(args['dir_exp'])
    args['dir_monitor_train'] = '{}/monitor/train'.format(args['dir_exp'])
    args['dir_monitor_test'] = '{}/monitor/test'.format(args['dir_exp'])
    # get current process pid
    args['process_pid'] = os.getpid()

    # # creating folders:
    # directory = args['dir_exp']
    # if os.path.exists(directory) and args['resume'] == '':
    #     toDelete= input("{} already exists, delete it if do you want to continue. Delete it? (yes/no) ".\
    #         format(directory))

    #     if toDelete.lower() == 'yes':
    #         shutil.rmtree(directory)
    #         print("Directory removed")
    #     if toDelete == 'No':
    #         print("It was not possible to continue, an experiment \
    #                folder is required.Terminiting here.")
    #         sys.exit()
    # if os.path.exists(directory) == False and args['resume'] == '':
    #     os.makedirs(directory)
    #     os.makedirs(args['dir_saved_models'])
    #     os.makedirs(args['dir_summary_train'])
    #     os.makedirs(args['dir_summary_test'])
    #     os.makedirs(args['dir_monitor_train'])
    #     os.makedirs(args['dir_monitor_test'])

    # time.sleep(1)
    # with open(os.path.expanduser('{}/arguments.txt'.format(args['dir_exp'])), 'w+') as file:
    #     file.write(json.dumps(args, indent=4, sort_keys=True))

    if args['expmode'] == 'fast':
        args['batch_size'] = 8
        args['max_episode_len'] = 50

    # noise
    if args['n_exploration_eps'] < 0:
        args['n_exploration_eps'] = args['n_episodes']

    # discrete actions
    if args['discrete_action'] == 'True':
        args['discrete_action'] = True
    else:
        args['discrete_action'] = False
    
    # actor l2 regularization
    if args['regularization'] == 'True':
        args['regularization'] = True
    else:
        args['regularization'] = False

    # reward normalization
    if args['reward_normalization'] == 'True':
        args['reward_normalization'] = True
    else:
        args['reward_normalization'] = False

    # object type observablity
    if args['observe_obj_grp'] == 'True':
        args['observe_obj_grp'] = True
    else:
        args['observe_obj_grp'] = False

    # masking aux reward with reward 
    if args['masked_with_r'] == 'True':
        args['masked_with_r'] = True
    else:
        args['masked_with_r'] = False

    # using gae in PPO
    if args['use_gae'] == 'True':
        args['use_gae'] = True
    else:
        args['use_gae'] = False

    # Whether or not to change the object order in stacking
    if args['change_stack_order'] == 'True':
        args['change_stack_order'] = True
    else:
        args['change_stack_order'] = False

    # Whether or not to use step reward instead of sparse
    if args['use_step_reward_fun'] == 'True':
        args['use_step_reward_fun'] = True
    else:
        args['use_step_reward_fun'] = False
        

    if args['train_robot_backward'] == 'False':
        args['train_robot_backward'] = False
    else:
        args['train_robot_backward'] = True

    obj_action_type = []
    for i in args['obj_action_type']:
        obj_action_type.append(int(i))
    args['obj_action_type'] = obj_action_type

    rob_policy = []
    for i in args['rob_policy']:
        rob_policy.append(float(i))
    rob_policy[1] -= 1.0
    args['rob_policy'] = rob_policy
        
    if verbose:
        print("\n==> Arguments:")
        for k,v in sorted(args.items()):
            print('{}: {}'.format(k,v))
        print('\n')


    return args


class Summarizer:
    """
        Class for saving the experiment log files
    """ 
    def __init__(self, path_summary, port, resume=''):

        if resume == '':
            self.__init__from_config(path_summary,port)
        else:
            self.__init__from_file(path_summary,port,resume)


    def __init__from_config(self, path_summary, port):
        self.path_summary = path_summary
        self.writer = SummaryWriter(self.path_summary)
        self.port = port
        self.list_rwd = []
        self.list_comm_rwd = []
        self.list_pkl = []

        if self.port != 0:
            t = threading.Thread(target=self.launchTensorBoard, args=([]))
            t.start()
 

    def __init__from_file(self, path_summary, port, resume):
        
        p = Path(resume).parents[1]
        #print('./{}/summary/log_record.pickle'.format(Path(resume).parents[1]))
        self.path_summary = '{}/summary/'.format(p)
        self.writer = SummaryWriter(self.path_summary)
        self.port = port

        #print(path_summary)
        with open('{}/summary/log_record.pickle'.format(p),'rb') as f:
            pckl = pickle.load(f)
            self.list_rwd = [x['reward_total'] for x in pckl]
            self.list_comm_rwd = [x['comm_reward_total'] for x in pckl]
            self.list_pkl = [x for x in pckl]

        if self.port != 0:
            t = threading.Thread(target=self.launchTensorBoard, args=([]))
            t.start()

    def update_log(self,
        idx_episode, 
        reward_total,
        reward_agents,
        critic_loss = None,
        actor_loss = None,
        to_save=True,
        to_save_plot = 10
        ):
            
        self.writer.add_scalar('reward_total',reward_total,idx_episode)

        self.writer.add_scalar('reward_agent', reward_agents,idx_episode)
        if critic_loss != None:
            self.writer.add_scalar('critic_loss', critic_loss,idx_episode)
        if actor_loss != None:
            self.writer.add_scalar('actor_loss', actor_loss,idx_episode)

        # save raw values on file
        self.list_rwd.append(reward_total)
        with open('{}/reward_total.txt'.format(self.path_summary), 'w') as fp:
            for el in self.list_rwd:
                fp.write("{}\n".format(round(el, 2)))

        # save in pickle format
        dct = {
            'idx_episode'  : idx_episode,
            'reward_total' : reward_total,
            'reward_agents': reward_agents,
            'critic_loss'  : critic_loss,
            'actor_loss'   : actor_loss
        }
        self.list_pkl.append(dct)


        # save things on disk
        if to_save:
            self.writer.export_scalars_to_json(
                '{}/summary.json'.format(self.path_summary))
            with open('{}/log_record.pickle'.format(self.path_summary), 'wb') as fp:
                pickle.dump(self.list_pkl, fp)

        if idx_episode % to_save_plot==0:
            self.plot_fig(self.list_rwd, 'reward_total')


    def plot_fig(self, record, name):
        durations_t = K.FloatTensor(np.asarray(record))

        fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(20,15))
        ax.grid(True)
        ax.set_ylabel('Duration')
        ax.set_xlabel('Episode')
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        # plt.yticks(np.arange(-200, 10, 10.0))

        ax.plot(durations_t.numpy())
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = K.cat((K.zeros(99), means))
            ax.plot(means.numpy())

        plt.draw()
        # plt.ylim([-200,10])
        
        fig.savefig('{}/{}.png'.format(self.path_summary,name))
        plt.close(fig)


    def save_fig(self, idx_episode, list_rwd):
        fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(20,15))
        ax.plot(range(idx_episode+1), list_rwd)
        ax.grid(True)
        ax.set_ylabel('Total reward')
        ax.set_xlabel('Episode')
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        fig.savefig('{}/reward_total.png'.format(self.path_summary))
        plt.draw()
        plt.close(fig)

    def close(self):
        self.writer.close()

    def launchTensorBoard(self):    
        os.system("tensorboard --logdir={} --port={}".format(
                                            self.path_summary,
                                            self.port))
        return


class Saver:
    """
        Class for saving and resuming the framework
    """ 

    def __init__(self, args):
        self.args = args

    def save_checkpoint(self,
                        save_dict,
                        episode,
                        filename = 'ckpt_last.pth.tar', 
                        is_best = False,
                        is_best_avg = False):
        """
            save on file
        """
        ckpt = self.build_state_ckpt(save_dict, episode)
        path_ckpt = os.path.join(self.args['dir_saved_models'], filename)
        K.save(ckpt, path_ckpt)

        if episode is not None:
            path_ckpt_ep = os.path.join(self.args['dir_saved_models'], 
                                        'ckpt_ep{}.pth.tar'.format(episode))
            shutil.copyfile(path_ckpt, path_ckpt_ep)

        if is_best:
            path_ckpt_best = os.path.join(self.args['dir_saved_models'], 
                                     'ckpt_best.pth.tar')
            shutil.copyfile(path_ckpt, path_ckpt_best)

        if is_best_avg:
            path_ckpt_best_avg = os.path.join(self.args['dir_saved_models'], 
                                     'ckpt_best_avg.pth.tar')
            shutil.copyfile(path_ckpt, path_ckpt_best_avg)


    def build_state_ckpt(self, save_dict, episode):
        """
            build a proper structure with all the info for resuming
        """
        ckpt = ({
            'args'       : self.args,
            'episode'    : episode,
            'save_dict'  : save_dict
            })

        return ckpt


    def resume_ckpt(self, resume_path=''):
        """
            build a proper structure with all the info for resuming
        """
        if resume_path =='':     
            ckpt = K.load(self.args['resume'])
        else:
            ckpt = K.load(resume_path)    

        self.args = ckpt['args']
        save_dict = ckpt['save_dict']
        episode   = ckpt['episode']

        return self.args, episode, save_dict


def get_exp_params(args=[], verbose=False):

    print("\n\n\n\n")
    print("==============================")
    print("Acquiring Arguments")
    print("==============================")


    parser = argparse.ArgumentParser(description='Arguments')

    # positional
    parser.add_argument("--env")
    parser.add_argument("--rob_model")
    parser.add_argument("--obj_rew")
    parser.add_argument("--use_her")
    parser.add_argument("--n_exp")
    parser.add_argument("--start_n_exp")
    parser.add_argument("--change_stack_order")
    parser.add_argument("--use_step_reward_fun")
    parser.add_argument("--shaped")
    parser.add_argument("--use_rnd")
    parser.add_argument("--filepath", default="/home/ok18/Jupyter/notebooks/Reinforcement_Learning/")

    # acquire in a dict
    config = parser.parse_args(args)
    args   = vars(config)


    return args