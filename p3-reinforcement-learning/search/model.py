import torch
import torch.nn as nn
import torch.autograd as autograd

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
            nn.Conv2d(1, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU()
        )

        self.fc_input_size = self.get_fc_input_size()

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size),
        )

    def get_fc_input_size(self):
        return self.conv(autograd.Variable(torch.zeros(1, *self.state_size))).view(1, -1).size(1)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.conv(state)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x