import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

# Source: https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter07/lib/dqn_model.py
class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight.data, bias)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)

class QNetworkSmall(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetworkSmall, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        """
        self.fc = nn.Sequential(
            nn.Linear(self.state_size[0], 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.action_size),
        )
        """

        self.noisy_layers = [
            NoisyLinear(self.state_size[0], 512),
            NoisyLinear(512, 256),
            NoisyLinear(256, 64),
            NoisyLinear(64, self.action_size)
        ]

        self.fc = nn.Sequential(
            self.noisy_layers[0],
            nn.LeakyReLU(),
            self.noisy_layers[1],
            nn.LeakyReLU(),
            self.noisy_layers[2],
            nn.LeakyReLU(),
            self.noisy_layers[3]
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc(state)

        return x

class QNetworkMedium(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetworkMedium, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        self.fc = nn.Sequential(
            nn.Linear(self.state_size[0], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.action_size),
        )

    def get_fc_input_size(self):
        return self.conv(autograd.Variable(torch.zeros(1, *self.state_size))).view(1, -1).size(1)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc(state)

        return x


class QNetworkOriginal(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetworkOriginal, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        self.fc = nn.Sequential(
            nn.Linear(self.state_size[0], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.action_size),
        )

    def get_fc_input_size(self):
        return self.conv(autograd.Variable(torch.zeros(1, *self.state_size))).view(1, -1).size(1)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc(state)

        return x