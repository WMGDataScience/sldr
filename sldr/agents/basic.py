import torch as K
import torch.nn as nn
import torch.nn.functional as F

import pdb


class Critic(nn.Module):
    def __init__(self, observation_space, action_space=None):
        super(Critic, self).__init__()
        
        if action_space is None:
            input_size = observation_space
        else:
            input_size = observation_space + action_space.shape[0]
        hidden_size = 256
        output_size = 1
        
        BN = nn.BatchNorm1d(input_size)
        BN.weight.data.fill_(1)
        BN.bias.data.fill_(0)

        self.FC = nn.Sequential(#BN, 
                                nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, output_size))
        

    def forward(self, s, a=None):

        x = s if (a is None) else K.cat([s, a], dim=1)
        x = self.FC(x)
        return x


class Critic5L(nn.Module):
    def __init__(self, observation_space, action_space=None):
        super(Critic5L, self).__init__()
        
        if action_space is None:
            input_size = observation_space
        else:
            input_size = observation_space + action_space.shape[0]
        hidden_size = 256
        output_size = 1
        
        BN = nn.BatchNorm1d(input_size)
        BN.weight.data.fill_(1)
        BN.bias.data.fill_(0)

        self.FC = nn.Sequential(#BN, 
                                nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, output_size))
        

    def forward(self, s, a=None):

        x = s if (a is None) else K.cat([s, a], dim=1)
        x = self.FC(x)
        return x


class CriticReg(nn.Module):
    def __init__(self, observation_space, action_space):
        super(CriticReg, self).__init__()
        
        input_size = observation_space + action_space.shape[0]
        hidden_size = 256
        output_size = 1

        self.k = 1
        self.n_p = hidden_size

        BN = nn.BatchNorm1d(input_size)
        BN.weight.data.fill_(1)
        BN.bias.data.fill_(0)
        
        # BN1 = nn.BatchNorm1d(observation_space)
        # BN1.weight.data.fill_(1)
        # BN1.bias.data.fill_(0)

        # BN2 = nn.BatchNorm1d(action_space.shape[0])
        # BN2.weight.data.fill_(1)
        # BN2.bias.data.fill_(0)

        # self.FC1 = nn.Sequential(BN1, 
        #                         nn.Linear(observation_space, hidden_size//2), nn.ReLU(True),
        #                         nn.Linear(hidden_size//2, hidden_size//2), nn.ReLU(True),
        #                         nn.Linear(hidden_size//2, hidden_size//2))

        # self.FC2 = nn.Sequential(BN2, 
        #                         nn.Linear(action_space.shape[0], hidden_size//2), nn.ReLU(True),
        #                         nn.Linear(hidden_size//2, hidden_size//2), nn.ReLU(True),
        #                         nn.Linear(hidden_size//2, hidden_size//2))

        self.FC = nn.Sequential(#BN, 
                                nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size))

        self.output = nn.Linear(hidden_size, output_size)
        

    def forward(self, s, a):

        #s = self.FC1(s)
        #a = self.FC2(a)

        #self.z = s

        x = K.cat([s, a], dim=1)
        x = self.FC(x)

        self.z = x

        x = self.output(F.relu(x))

        return x

    def GAR(self, c_alpha=1, c_beta=1, c_F=0.000001):

        Z = self.z
        B = F.relu(Z)
        N = K.matmul(B.t(),B)

        v = N.diag().view(1, -1)
        V = K.matmul(v.t(), v)

        affinity = (N.sum() - N.trace()) / ((self.k*self.n_p-1) * N.trace() + 1e-8)
        balance = (V.sum() - V.trace()) / ((self.k*self.n_p-1) * V.trace() + 1e-8)
        frob = K.pow(Z, 2).sum()

        reg = (c_alpha * affinity + c_beta * (1 - balance) + c_F * frob)

        return reg


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

    def __init__(self, observation_space, action_space, discrete=True, out_func=K.sigmoid):
        super(Actor, self).__init__()
        
        input_size = observation_space
        hidden_size = 256
        output_size = action_space.shape[0]

        self.discrete = discrete
        self.out_func = out_func
        self.action_space = action_space

        BN = nn.BatchNorm1d(input_size)
        BN.weight.data.fill_(1)
        BN.bias.data.fill_(0)
        
        self.FC = nn.Sequential(#BN,
                                nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
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
                if self.out_func == K.sigmoid:
                    x = K.tensor(self.action_space.low[0], dtype=x.dtype, device=x.device) + x*K.tensor((self.action_space.high[0]-self.action_space.low[0]), dtype=x.dtype, device=x.device)
                elif self.out_func == K.tanh:
                    x = x*K.tensor((self.action_space.high[0]), dtype=x.dtype, device=x.device)

        #x = x*K.tensor((self.action_space.high[0]), dtype=x.dtype, device=x.device)

        return x

    def get_preactivations(self, s):
        
        x = s
        x = self.FC(x)
   
        return x


class Actor5L(nn.Module):

    def __init__(self, observation_space, action_space, discrete=True, out_func=K.sigmoid):
        super(Actor5L, self).__init__()
        
        input_size = observation_space
        hidden_size = 256
        output_size = action_space.shape[0]

        self.discrete = discrete
        self.out_func = out_func
        self.action_space = action_space

        BN = nn.BatchNorm1d(input_size)
        BN.weight.data.fill_(1)
        BN.bias.data.fill_(0)
        
        self.FC = nn.Sequential(#BN,
                                nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
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
                if self.out_func == K.sigmoid:
                    x = K.tensor(self.action_space.low[0], dtype=x.dtype, device=x.device) + x*K.tensor((self.action_space.high[0]-self.action_space.low[0]), dtype=x.dtype, device=x.device)
                elif self.out_func == K.tanh:
                    x = x*K.tensor((self.action_space.high[0]), dtype=x.dtype, device=x.device)

        #x = x*K.tensor((self.action_space.high[0]), dtype=x.dtype, device=x.device)

        return x

    def get_preactivations(self, s):
        
        x = s
        x = self.FC(x)
   
        return x


# Normal
FixedNormal = K.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropies = lambda self: normal_entropy(self).sum(-1)
FixedNormal.mode = lambda self: self.mean

class ActorStoch(nn.Module):

    def __init__(self, observation_space, action_space, discrete=True, out_func=K.sigmoid):
        super(ActorStoch, self).__init__()
        
        input_size = observation_space
        hidden_size = 256
        output_size = action_space.shape[0]

        self.discrete = discrete
        self.out_func = out_func
        self.action_space = action_space

        BN = nn.BatchNorm1d(input_size)
        BN.weight.data.fill_(1)
        BN.bias.data.fill_(0)
        
        self.FC = nn.Sequential(#BN,
                                nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, output_size))

        bias = K.zeros(output_size)
        self._bias = nn.Parameter(bias.unsqueeze(1))
        
    def forward(self, s):

        x = s
        if self.discrete:
            x = F.softmax(self.FC(x), dim=1)
        else:
            if self.out_func == 'linear':
                x = self.FC(x)
            else:
                x = self.out_func(self.FC(x))
                if self.out_func == K.sigmoid:
                    x = K.tensor(self.action_space.low[0], dtype=x.dtype, device=x.device) + x*K.tensor((self.action_space.high[0]-self.action_space.low[0]), dtype=x.dtype, device=x.device)
                elif self.out_func == K.tanh:
                    x = x*K.tensor((self.action_space.high[0]), dtype=x.dtype, device=x.device)

        #x = x*K.tensor((self.action_space.high[0]), dtype=x.dtype, device=x.device)

        action_logstd = self._bias.t().view(1, -1)

        return FixedNormal(x, action_logstd.exp())

    def get_preactivations(self, s):
        
        x = s
        x = self.FC(x)
   
        return x


class ForwardDynReg(nn.Module):
    def __init__(self, observation_space, action_space):
        super(ForwardDynReg, self).__init__()
        
        input_size = observation_space + action_space.shape[0]
        hidden_size = 256
        output_size = observation_space

        self.k = 1
        self.n_p = hidden_size

        BN = nn.BatchNorm1d(input_size)
        BN.weight.data.fill_(1)
        BN.bias.data.fill_(0)

        self.FC = nn.Sequential(#BN, 
                                nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size))

        self.output = nn.Linear(hidden_size, output_size)
        self.maxM = 1.
        

    def forward(self, s, a):

        x = K.cat([s, a], dim=1)
        x = self.FC(x)

        self.z = x

        x = self.output(F.relu(x))

        return x

    def GAR(self, c_alpha=1, c_beta=1, c_F=0.000001):

        Z = self.z
        B = F.relu(Z)
        N = K.matmul(B.t(),B)

        v = N.diag().view(1, -1)
        V = K.matmul(v.t(), v)

        affinity = (N.sum() - N.trace()) / ((self.k*self.n_p-1) * N.trace() + 1e-8)
        balance = (V.sum() - V.trace()) / ((self.k*self.n_p-1) * V.trace() + 1e-8)
        frob = K.pow(Z, 2).sum()

        reg = (c_alpha * affinity + c_beta * (1 - balance) + c_F * frob)

        return reg

    def get_maxM(self):

        Z = self.z
        B = F.relu(Z)
        M = K.matmul(B,B.t())
        M *= (1-K.eye(M.shape[0], dtype=M.dtype, device=M.device))
        if M.max().item() > self.maxM:
            self.maxM = M.max().item()

        return self.maxM

    def get_intr_rewards(self, s, a):
        with K.no_grad():
            x = K.cat([s, a], dim=1)
            Z = self.FC(x)
            B = F.relu(Z)
            M = K.matmul(B, B.t())

            #M = M/(M.max(1)[0].unsqueeze(1))
            M /= M.diag().unsqueeze(1)
            M *= (1-K.eye(M.shape[0], dtype=M.dtype, device=M.device))
            p = - M.mean(1).unsqueeze(1)

        return p


class AutoEncReg(nn.Module):
    def __init__(self, observation_space, action_space):
        super(AutoEncReg, self).__init__()
        
        input_size = observation_space + action_space.shape[0]
        hidden_size = 256
        output_size = 20#observation_space

        self.k = 1
        self.n_p = output_size

        BN = nn.BatchNorm1d(input_size)
        BN.weight.data.fill_(1)
        BN.bias.data.fill_(0)

        self.ENC = nn.Sequential(#BN, 
                                nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, output_size))

        self.DEC = nn.Sequential(#BN, 
                                nn.Linear(output_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, input_size), nn.Tanh())


        self.maxM = 1.
        
    def encode(self, s, a):
        x = K.cat([s, a], dim=1)
        x = self.ENC(x)

        self.z = x

        x = F.relu(x)
        return x

    def decode(self, x):
        x = self.DEC(x)

        return x

    def forward(self, s, a):

        z = self.encode(s, a)
        x = self.decode(z)

        return x

    def GAR(self, c_alpha=1, c_beta=1, c_F=0.000001):

        Z = self.z
        B = F.relu(Z)
        N = K.matmul(B.t(),B)

        v = N.diag().view(1, -1)
        V = K.matmul(v.t(), v)

        affinity = (N.sum() - N.trace()) / ((self.k*self.n_p-1) * N.trace() + 1e-8)
        balance = (V.sum() - V.trace()) / ((self.k*self.n_p-1) * V.trace() + 1e-8)
        frob = K.pow(Z, 2).sum()

        reg = (c_alpha * affinity + c_beta * (1 - balance) + c_F * frob)

        return reg

    def get_maxM(self):

        Z = self.z
        B = F.relu(Z)
        M = K.matmul(B,B.t())
        M *= (1-K.eye(M.shape[0], dtype=M.dtype, device=M.device))
        if M.max().item() > self.maxM:
            self.maxM = M.max().item()

        return self.maxM

    def get_intr_rewards(self, s, a):
        with K.no_grad():
            x = K.cat([s, a], dim=1)
            Z = self.ENC(x)
            B = F.relu(Z)
            M = K.matmul(B, B.t())

            #M = M/(M.max(1)[0].unsqueeze(1))
            M /= M.diag().unsqueeze(1)
            M *= (1-K.eye(M.shape[0], dtype=M.dtype, device=M.device))
            p = - M.mean(1).unsqueeze(1)

        return p


class AutoEncNextReg(nn.Module):
    def __init__(self, observation_space, action_space):
        super(AutoEncNextReg, self).__init__()
        
        input_size = observation_space
        hidden_size = 256
        output_size = 20#observation_space

        self.k = 1
        self.n_p = output_size

        self.k = self.k  * self.n_p
        self.n_p = 1

        BN = nn.BatchNorm1d(input_size)
        BN.weight.data.fill_(1)
        BN.bias.data.fill_(0)

        self.ENC = nn.Sequential(#BN, 
                                nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, output_size))

        self.DEC = nn.Sequential(#BN, 
                                nn.Linear(output_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, input_size), nn.Tanh())


        self.maxM = 1.
        
    def encode(self, s):
        x = self.ENC(s)

        self.z = x

        x = F.relu(x)
        return x

    def decode(self, x):
        x = self.DEC(x)

        return x

    def forward(self, s):

        z = self.encode(s)
        x = self.decode(z)

        return x

    def GAR(self, c_alpha=.03, c_beta=.03, c_F=0., symmetric=False):

        Z = self.z

        #pdb.set_trace()

        Z = Z.view(-1, self.k, self.n_p)
        Bs = [F.relu(Z)]
        if symmetric:
            Bs.append(F.relu(-Z))
            
        affinity = 0
        balance = 0
    
        for B in Bs:

            N = K.bmm(B.permute(2,1,0), B.permute(2,0,1))
            N_sum = N.sum(dim=[1,2])
            N_trace = N.diagonal(dim1=1, dim2=2).sum(-1)

            v = N.diagonal(dim1=1, dim2=2).unsqueeze(1)
            V = K.bmm(v.permute(0,2,1), v.permute(0,1,2))
            V_sum = V.sum(dim=[1,2])
            V_trace = V.diagonal(dim1=1, dim2=2).sum(-1)

            affinity += ((N_sum - N_trace) / ((self.k-1) * N_trace + 1e-8)).sum()
            balance += ((V_sum - V_trace) / ((self.k-1) * V_trace + 1e-8)).sum()

        affinity /= self.n_p
        balance /= self.n_p
        frob = K.pow(Z, 2).mean()

        self.affinity = affinity
        self.balance = balance
        self.frob = frob

        reg = c_alpha * affinity + c_beta * (len(Bs) - balance) + c_F * frob

        return reg

    def get_maxM(self):

        Z = self.z
        B = F.relu(Z)
        M = K.matmul(B,B.t())
        M *= (1-K.eye(M.shape[0], dtype=M.dtype, device=M.device))
        if M.max().item() > self.maxM:
            self.maxM = M.max().item()

        return self.maxM

    def get_intr_rewards(self, s):
        with K.no_grad():
            Z = self.ENC(s)
            B = F.relu(Z)
            M = K.matmul(B, B.t())
            #M2 = M * (1-K.eye(M.shape[0], dtype=M.dtype, device=M.device))
            #p = - M2.mean(1).unsqueeze(1)/M.diag().unsqueeze(1)

            #p = M.mean(1).unsqueeze(1)/M.diagonal().unsqueeze(1)
            
            #p = (p-p.min())/(p.max()-p.min())
            p = (M>0).to(M.dtype).sum(1).unsqueeze(1)
            #p = (p-p.min())/(p.max()-p.min()+1e-8)
            
        return p, M


class BackwardDyn(nn.Module):
    def __init__(self, observation_space, action_space):
        super(BackwardDyn, self).__init__()
        
        input_size = observation_space*2
        hidden_size = 256
        output_size = action_space.shape[0]

        BN = nn.BatchNorm1d(input_size)
        BN.weight.data.fill_(1)
        BN.bias.data.fill_(0)

        self.FC = nn.Sequential(#BN, 
                                nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, output_size))
        

    def forward(self, s, s_):

        x = K.cat([s, s_], dim=1)
        x = self.FC(x)

        return x


class RandomNetDist(nn.Module):
    def __init__(self, observation_space):
        super(RandomNetDist, self).__init__()
        
        input_size = observation_space
        hidden_size = 256
        output_size = hidden_size

        self.FC = nn.Sequential(
                                nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, output_size))
        

    def forward(self, x):

        x = self.FC(x)

        return x
