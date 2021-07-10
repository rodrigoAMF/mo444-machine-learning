import torch
import torch.nn as nn
import torch.autograd as autograd


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)

class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )
        self.conv.apply(weights_init)

        self.fc_input_size = self.get_fc_input_size()
        print("Linear layer input size: ", self.fc_input_size)

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.action_size),
        )
        self.fc.apply(weights_init)

    def get_fc_input_size(self):
        return self.conv(autograd.Variable(torch.zeros(1, *self.state_size))).view(1, -1).size(1)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.conv(state)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x