import torch as K
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Critic, self).__init__()
        
        input_size = observation_space + action_space.shape[0]
        hidden_size = 128
        output_size = 1
        
        self.FC = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, output_size))
        

    def forward(self, s, a):

        x = K.cat([s, a], dim=1)     
        x = self.FC(x)
        return x


class Critic_Chaos(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Critic_Chaos, self).__init__()
        
        input_size = observation_space + action_space.shape[0]
        hidden_size = 128
        output_size = 1
        
        self.FC = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True))

        self.N = nn.Linear(hidden_size, output_size)

        self.C = nn.Linear(hidden_size, output_size)

        self.A = nn.Linear(hidden_size, output_size)


    def forward(self, s, a):

        x = K.cat([s, a], dim=1)     
        x = self.FC(x)
        n = self.N(x)
        c = self.C(x)
        a = self.A(x)

        x = n - a*(n*n)
        return x
    
    
class Actor(nn.Module):

    def __init__(self, observation_space, action_space, discrete=True, out_func=F.sigmoid, ):
        super(Actor, self).__init__()
        
        input_size = observation_space
        hidden_size = 64
        output_size = action_space.shape[0]

        self.discrete = discrete
        self.out_func = out_func
        self.action_space = action_space
        
        self.FC = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, output_size))
        
    def forward(self, s):

        x = s
        if self.discrete:
            x = F.softmax(self.FC(x), dim=1)
        else:
            if self.out_func == 'linear':
                x = self.FC(x)
            else:
                x = self.out_func(self.FC(x))

        x = K.tensor(self.action_space.low[0], dtype=x.dtype, device=x.device) + x*K.tensor((self.action_space.high[0]-self.action_space.low[0]), dtype=x.dtype, device=x.device)

        return x

    def get_preactivations(self, s):
        
        x = s
        x = self.FC(x)
   
        return x
