import torch
import torch.nn as nn
import torch.autograd as autograd


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

        self.fc = nn.Sequential(
            nn.Linear(self.state_size[0], 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.action_size),
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