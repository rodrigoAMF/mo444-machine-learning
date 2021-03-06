import os
import numpy as np
import random
from collections import namedtuple, deque

from model import QNetworkSmall, QNetworkMedium, QNetworkOriginal

import torch
import torch.nn.functional as F
import torch.optim as optim

from game import Agent
from pacman import Directions

UPDATE_EVERY = 4  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

action_to_direction = {
    0: Directions.NORTH,
    1: Directions.SOUTH,
    2: Directions.EAST,
    3: Directions.WEST,
    4: Directions.STOP
}

direction_to_action = {
    Directions.NORTH: 0,
    Directions.SOUTH: 1,
    Directions.EAST: 2,
    Directions.WEST: 3,
    Directions.STOP: 4
}


class DQNAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, params, layout_used, seed, test_only=True,
                 checkpoint_to_use=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.params = params

        # Q-Network
        if not test_only:
            if layout_used == "smallClassic":
                self.qnetwork_local = QNetworkSmall(state_size, action_size, seed).to(device)
                self.qnetwork_target = QNetworkSmall(state_size, action_size, seed).to(device)
            elif layout_used == "mediumClassic":
                self.qnetwork_local = QNetworkMedium(state_size, action_size, seed).to(device)
                self.qnetwork_target = QNetworkMedium(state_size, action_size, seed).to(device)
            elif layout_used == "originalClassic":
                self.qnetwork_local = QNetworkOriginal(state_size, action_size, seed).to(device)
                self.qnetwork_target = QNetworkOriginal(state_size, action_size, seed).to(device)
            else:
                raise ValueError("Unkown layout", layout_used)
            self.optimizer = optim.AdamW(self.qnetwork_local.parameters(), lr=params["lr"], amsgrad=False)

            # Replay memory
            self.memory = ReplayBuffer(state_size, action_size, self.params["buffer_size"], self.params["batch_size"], seed)
            # Initialize time step (for updating every params["update_every"] steps)
            self.t_step = 0
        else:
            if checkpoint_to_use:
                layout_used = checkpoint_to_use
            self.qnetwork_local = QNetworkSmall(state_size, action_size, seed).to(device)
            weights_filename = "checkpoint_{}.pth".format(layout_used)
            self.qnetwork_local.load_state_dict(torch.load(os.path.join("models", weights_filename)))

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every params["update_every"] time steps.
        self.t_step = (self.t_step + 1) % self.params["update_every"]
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.params["batch_size"]:
                experiences = self.memory.sample()
                self.learn(experiences, self.params["gamma"])

    def getAction(self, state, legal_actions):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        action_values = action_values.cpu().data.numpy()[0] # Action values from Neural Network
        legal_actions = np.array([direction_to_action[action] for action in legal_actions]) # Possible action in the current state as number
        possible_actions = action_values[legal_actions] # Values of legal actions
        best_action_index = np.argmax(possible_actions)
        action = legal_actions[best_action_index]
        action = action_to_direction[action]

        return action

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.params["tau"])

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        ??_target = ??*??_local + (1 - ??)*??_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_size, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.vstack([e.state for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        if self.state_size[0] == 1:
            states = np.expand_dims(states, axis=1)
            next_states = np.expand_dims(next_states, axis=1)
        states = torch.from_numpy(states).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)

        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)

        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)