import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
from rewards import compute_param_reward
import numpy as np
__all__ = ['ActorCritic']


def init_action_space():
    # alpha = [-2.*np.pi/180., 0., 2.*np.pi/180. ]
    # beta =  [-2. * np.pi / 180., 0., 2. * np.pi / 180.]
    # gamma = [-2. * np.pi / 180., 0., 2. * np.pi / 180.]
    # delta_d = [-0.05, 0, 0.05]
    alpha = [-2. * np.pi / 180.,  2. * np.pi / 180.]
    beta = [-2. * np.pi / 180.,  2. * np.pi / 180.]
    gamma = [-2. * np.pi / 180., 2. * np.pi / 180.]
    delta_d = [-0.05,  0.05]

    action_space = []
    for i in range(len(alpha)):
        for j in range(len(beta)):
            for k in range(len(gamma)):
                for l in range(len(delta_d)):
                     action_space.append((alpha[i], beta[j],gamma[k], delta_d[l]))

    return action_space, len(action_space)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs=128, num_outputs=16, hidden_size=512, std=0.0):
        super(ActorCritic, self).__init__()
        self.feat_transfer = nn.Sequential(
                            nn.Linear(2048, num_inputs),
                            nn.ReLU(),
                            #nn.Linear(512, num_inputs), #num_inputs <=512
                            #nn.ReLU()
                            )

        self.critic = nn.Sequential(
            nn.Linear(num_inputs+4, hidden_size), #num_inputs+
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs+4, hidden_size),  #num_inputs+
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )

    # def forward(self, x, param):
    #     x = self.feat_transfer(x)
    #     x = torch.cat([x,param.permute([1,0])],dim=1)
    #     value = self.critic(x)
    #     probs = self.actor(x)
    #     dist = Categorical(probs)
    #     return dist, value

    def forward(self,  param):
        # x = self.feat_transfer(x)
        x = param.permute([1,0]) #torch.cat([x,param.permute([1,0])],dim=1)
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value