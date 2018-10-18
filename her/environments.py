import torch as K

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import pdb

class communication(object):
    def __init__(self, protocol_type=2, consecuitive_limit=5, num_agents=3, medium_type='obs_only'):
        self.protocol_type = protocol_type
        self.consecuitive_limit = consecuitive_limit
        self.num_agents = num_agents
        self.previous_granted_agent = None
        self.consecuitive_count = 0
        self.comm_hist = K.zeros((num_agents, consecuitive_limit, 1), dtype=K.uint8)
        self.medium_type = medium_type

    def reset(self):
        self.previous_granted_agent = None
        self.consecuitive_count = 0
        self.comm_hist = K.zeros((self.num_agents, self.consecuitive_limit, 1), dtype=K.uint8)

    def step(self, observations, comm_actions, prev_actions=None):
        
        comm_rewards = K.zeros(self.num_agents, dtype=observations.dtype).view(-1,1,1)
        
        if (comm_actions>0.5).sum().item() == 0: # channel stays idle
            comm_rewards -= 1
            if self.medium_type is 'obs_only':
                medium = K.cat([K.rand_like(observations[[0], ]), K.zeros((1,1,1), dtype=observations.dtype)], dim=-1)
            else:
                medium = K.cat([K.rand_like(observations[[0], ]),
                                K.rand_like(prev_actions[[0], ]),
                                K.zeros((1,1,1), dtype=observations.dtype)
                                ], dim=-1)
        elif (comm_actions>0.5).sum().item() > 1: # collision
            comm_rewards[comm_actions>0.5] -= 1
            if self.medium_type is 'obs_only':
                medium = K.cat([K.rand_like(observations[[0], ]), 
                                (self.num_agents+1)*K.ones((1,1,1), dtype=observations.dtype)], dim=-1)
            else:
                medium = K.cat([K.rand_like(observations[[0], ]),
                                K.rand_like(prev_actions[[0], ]), 
                                (self.num_agents+1)*K.ones((1,1,1), dtype=observations.dtype)], dim=-1)
        else:                                     # success
            granted_agent = K.argmax((comm_actions>0.5)).item()
            if self.protocol_type == 1:
                if self.previous_granted_agent == granted_agent:
                    self.consecuitive_count += 1
                    if self.consecuitive_count > self.consecuitive_limit:
                        comm_rewards -= 1#(self.consecuitive_count - self.consecuitive_limit)
                else:
                    self.previous_granted_agent = granted_agent
                    self.consecuitive_count = 0
            elif self.protocol_type == 2:
                grant = K.zeros((self.num_agents,1,1), dtype=K.uint8)
                grant[granted_agent,] = 1
                self.comm_hist = K.cat((self.comm_hist[:,1::,], grant), dim=1)
                comm_rewards[(self.comm_hist.sum(dim=1, keepdim=True) == self.consecuitive_limit) + 
                             (self.comm_hist.sum(dim=1, keepdim=True) == 0)] -= 1
                
            if self.medium_type is 'obs_only':
                medium = K.cat([observations[[granted_agent], ], (granted_agent+1)*K.ones((1,1,1), dtype=observations.dtype)], dim=-1)
            else:
                medium = K.cat([observations[[granted_agent], ],
                                prev_actions[[granted_agent], ], 
                                (granted_agent+1)*K.ones((1,1,1), dtype=observations.dtype)], dim=-1)


            #if competitive_comm:
            #    comm_rewards -= 0.75
            #    comm_rewards[granted_agent] += 1.5  
            #medium = K.cat([observations[[granted_agent], ], (granted_agent+1)*K.ones((1,1,1), dtype=observations.dtype)], dim=-1)

        comm_rewards = comm_rewards.numpy()
        medium = medium.numpy()
        
        return medium, comm_rewards


    def get_m(self, observations, comm_actions, prev_actions=None):
        
        #comm_rewards = K.zeros((observations.shape[0], observations.shape[1], 1), 
        #                    dtype=observations.dtype, device=observations.device)

        if self.medium_type is 'obs_only':
            medium = K.rand((1, observations.shape[1], observations.shape[2]+1), 
                            dtype=observations.dtype, device=observations.device)
        else:
            medium = K.rand((1, observations.shape[1], observations.shape[2]+prev_actions.shape[2]+1), 
                            dtype=observations.dtype, device=observations.device)           


        granted_agent = (comm_actions>0.5).argmax(dim=0)[:,0]
        for i in range(self.num_agents):
            #if competitive_comm:
            #    comm_rewards[i, granted_agent == i, :] = 1
            if self.medium_type is 'obs_only':
                medium[:, granted_agent == i, :] = K.cat([observations[[i],][:, granted_agent==i, :], 
                                                          (i+1)*K.ones((1,(granted_agent==i).sum().item(),1), 
                                                                    dtype=observations.dtype, device=observations.device)], dim=-1)
            else:
                medium[:, granted_agent == i, :] = K.cat([observations[[i],][:, granted_agent==i, :], 
                                                          prev_actions[[i],][:, granted_agent==i, :], 
                                                          (i+1)*K.ones((1,(granted_agent==i).sum().item(),1), 
                                                                    dtype=observations.dtype, device=observations.device)], dim=-1)               


        if K.is_nonzero(((comm_actions>0.5).sum(dim=0) == 0)[:,0].sum()):
            #comm_rewards[:,((comm_actions>0.5).sum(dim=0) == 0)[:,0],:] = -1
            if self.medium_type is 'obs_only':
                medium[:,((comm_actions>0.5).sum(dim=0) == 0)[:,0], :] = K.cat([K.rand((1,1,observations.shape[2]),
                                                                                        dtype=observations.dtype, 
                                                                                        device=observations.device),
                                                                                K.zeros((1,1,1), 
                                                                                        dtype=observations.dtype, 
                                                                                        device=observations.device)], 
                                                                            dim=-1)
            else:
                medium[:,((comm_actions>0.5).sum(dim=0) == 0)[:,0], :] = K.cat([K.rand((1,1,observations.shape[2]),
                                                                                        dtype=observations.dtype, 
                                                                                        device=observations.device),
                                                                                K.zeros((1,1,prev_actions.shape[2]),
                                                                                        dtype=observations.dtype, 
                                                                                        device=observations.device),
                                                                                K.zeros((1,1,1), 
                                                                                        dtype=observations.dtype, 
                                                                                        device=observations.device)], 
                                                                            dim=-1)

            
        if K.is_nonzero(((comm_actions>0.5).sum(dim=0) > 1)[:,0].sum()):
            #comm_rewards[:,((comm_actions>0.5).sum(dim=0) > 1)[:,0],:] = -1
            if self.medium_type is 'obs_only':
                medium[:,((comm_actions>0.5).sum(dim=0) > 1)[:,0], :] = K.cat([K.zeros((1,1,observations.shape[2]),
                                                                                        dtype=observations.dtype, 
                                                                                        device=observations.device),
                                                                               K.ones((1,1,1), 
                                                                                        dtype=observations.dtype, 
                                                                                        device=observations.device)*(self.num_agents+1)], 
                                                                            dim=-1)
            else:
                medium[:,((comm_actions>0.5).sum(dim=0) > 1)[:,0], :] = K.cat([K.zeros((1,1,observations.shape[2]),
                                                                                        dtype=observations.dtype, 
                                                                                        device=observations.device),
                                                                               K.zeros((1,1,prev_actions.shape[2]),
                                                                                        dtype=observations.dtype, 
                                                                                        device=observations.device),
                                                                               K.ones((1,1,1), 
                                                                                        dtype=observations.dtype, 
                                                                                        device=observations.device)*(self.num_agents+1)], 
                                                                            dim=-1)

        return K.tensor(medium, requires_grad=True)

class communication_v2(object):
    def __init__(self, protocol_type=2, consecuitive_limit=5, num_agents=3, medium_type='obs_only'):
        self.protocol_type = protocol_type
        self.consecuitive_limit = consecuitive_limit
        self.num_agents = num_agents
        self.previous_granted_agent = None
        self.consecuitive_count = 0
        self.comm_hist = K.zeros((num_agents, consecuitive_limit, 1), dtype=K.uint8)
        self.medium_type = medium_type

    def reset(self):
        self.previous_granted_agent = None
        self.consecuitive_count = 0
        self.comm_hist = K.zeros((self.num_agents, self.consecuitive_limit, 1), dtype=K.uint8)

    def step(self, observations, comm_actions, prev_actions=None):
        
        #pdb.set_trace()
        comm_rewards = K.zeros(self.num_agents, dtype=observations.dtype).view(-1,1,1)
        
        if (comm_actions<0.00001).sum().item() == self.num_agents: # channel stays idle
            comm_rewards -= 1
            if self.medium_type is 'obs_only':
                medium = K.cat([K.zeros_like(observations[[0], ]), K.zeros((1,1,1), dtype=observations.dtype)], dim=-1)
            else:
                medium = K.cat([K.zeros_like(observations[[0], ]),
                                K.zeros_like(prev_actions[[0], ]),
                                K.zeros((1,1,1), dtype=observations.dtype)
                                ], dim=-1)
        elif (comm_actions>0.99999).sum().item() == self.num_agents: # collision
            comm_rewards -= 1
            if self.medium_type is 'obs_only':
                medium = K.cat([K.zeros_like(observations[[0], ]), 
                                (self.num_agents+1)*K.ones((1,1,1), dtype=observations.dtype)], dim=-1)
            else:
                medium = K.cat([K.zeros_like(observations[[0], ]),
                                K.zeros_like(prev_actions[[0], ]), 
                                (self.num_agents+1)*K.ones((1,1,1), dtype=observations.dtype)], dim=-1)
        else:                                     # success
            granted_agent = K.argmax(comm_actions).item()
            if self.protocol_type == 1:
                if self.previous_granted_agent == granted_agent:
                    self.consecuitive_count += 1
                    if self.consecuitive_count > self.consecuitive_limit:
                        comm_rewards -= 1#(self.consecuitive_count - self.consecuitive_limit)
                else:
                    self.previous_granted_agent = granted_agent
                    self.consecuitive_count = 0
            elif self.protocol_type == 2:
                grant = K.zeros((self.num_agents,1,1), dtype=K.uint8)
                grant[granted_agent,] = 1
                self.comm_hist = K.cat((self.comm_hist[:,1::,], grant), dim=1)
                comm_rewards[(self.comm_hist.sum(dim=1, keepdim=True) == self.consecuitive_limit) + 
                             (self.comm_hist.sum(dim=1, keepdim=True) == 0)] -= 1
                
            if self.medium_type is 'obs_only':
                medium = K.cat([observations[[granted_agent], ], (granted_agent+1)*K.ones((1,1,1), dtype=observations.dtype)], dim=-1)
            else:
                medium = K.cat([observations[[granted_agent], ],
                                prev_actions[[granted_agent], ], 
                                (granted_agent+1)*K.ones((1,1,1), dtype=observations.dtype)], dim=-1)


            #if competitive_comm:
            #    comm_rewards -= 0.75
            #    comm_rewards[granted_agent] += 1.5  
            #medium = K.cat([observations[[granted_agent], ], (granted_agent+1)*K.ones((1,1,1), dtype=observations.dtype)], dim=-1)

        comm_rewards = comm_rewards.numpy()
        medium = medium.numpy()
        
        return medium, comm_rewards


    def get_m(self, observations, comm_actions, prev_actions=None):
        
        #comm_rewards = K.zeros((observations.shape[0], observations.shape[1], 1), 
        #                    dtype=observations.dtype, device=observations.device)

        if self.medium_type is 'obs_only':
            medium = K.zeros((1, observations.shape[1], observations.shape[2]+1), 
                            dtype=observations.dtype, device=observations.device)
        else:
            medium = K.zeros((1, observations.shape[1], observations.shape[2]+prev_actions.shape[2]+1), 
                            dtype=observations.dtype, device=observations.device)           


        granted_agent = comm_actions.argmax(dim=0)[:,0]
        for i in range(self.num_agents):
            #if competitive_comm:
            #    comm_rewards[i, granted_agent == i, :] = 1
            if self.medium_type is 'obs_only':
                medium[:, granted_agent == i, :] = K.cat([observations[[i],][:, granted_agent==i, :], 
                                                          (i+1)*K.ones((1,(granted_agent==i).sum().item(),1), 
                                                                    dtype=observations.dtype, device=observations.device)], dim=-1)
            else:
                medium[:, granted_agent == i, :] = K.cat([observations[[i],][:, granted_agent==i, :], 
                                                          prev_actions[[i],][:, granted_agent==i, :], 
                                                          (i+1)*K.ones((1,(granted_agent==i).sum().item(),1), 
                                                                    dtype=observations.dtype, device=observations.device)], dim=-1)               


        if K.is_nonzero(((comm_actions<0.00001).sum(dim=0) == self.num_agents)[:,0].sum()):

            #comm_rewards[:,((comm_actions>0.5).sum(dim=0) == 0)[:,0],:] = -1
            if self.medium_type is 'obs_only':
                medium[:,((comm_actions<0.00001).sum(dim=0) == self.num_agents)[:,0], :] = K.cat([K.zeros((1,1,observations.shape[2]),
                                                                                        dtype=observations.dtype, 
                                                                                        device=observations.device),
                                                                                K.zeros((1,1,1), 
                                                                                        dtype=observations.dtype, 
                                                                                        device=observations.device)], 
                                                                            dim=-1)
            else:
                medium[:,((comm_actions<0.00001).sum(dim=0) == self.num_agents)[:,0], :] = K.cat([K.zeros((1,1,observations.shape[2]),
                                                                                        dtype=observations.dtype, 
                                                                                        device=observations.device),
                                                                                K.zeros((1,1,prev_actions.shape[2]),
                                                                                        dtype=observations.dtype, 
                                                                                        device=observations.device),
                                                                                K.zeros((1,1,1), 
                                                                                        dtype=observations.dtype, 
                                                                                        device=observations.device)], 
                                                                            dim=-1)

            
        if K.is_nonzero(((comm_actions>0.99999).sum(dim=0) == self.num_agents)[:,0].sum()):
            #comm_rewards[:,((comm_actions>0.5).sum(dim=0) > 1)[:,0],:] = -1
            if self.medium_type is 'obs_only':
                medium[:,((comm_actions>0.99999).sum(dim=0) == self.num_agents)[:,0], :] = K.cat([K.zeros((1,1,observations.shape[2]),
                                                                                        dtype=observations.dtype, 
                                                                                        device=observations.device),
                                                                               K.ones((1,1,1), 
                                                                                        dtype=observations.dtype, 
                                                                                        device=observations.device)*(self.num_agents+1)], 
                                                                            dim=-1)
            else:
                medium[:,((comm_actions>0.99999).sum(dim=0) == self.num_agents)[:,0], :] = K.cat([K.zeros((1,1,observations.shape[2]),
                                                                                        dtype=observations.dtype, 
                                                                                        device=observations.device),
                                                                               K.zeros((1,1,prev_actions.shape[2]),
                                                                                        dtype=observations.dtype, 
                                                                                        device=observations.device),
                                                                               K.ones((1,1,1), 
                                                                                        dtype=observations.dtype, 
                                                                                        device=observations.device)*(self.num_agents+1)], 
                                                                            dim=-1)

        return K.tensor(medium, requires_grad=True)

def make_env_cont(scenario_name, benchmark=False):

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:        
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env