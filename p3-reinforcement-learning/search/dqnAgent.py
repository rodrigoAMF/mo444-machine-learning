import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

from game import Agent
from pacman import Directions
from SumTree import SumTree

BUFFER_SIZE = 2*int(1e4)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
TRAIN_START = 1000 # Number of samples needed to start training
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 2e-3  # learning rate
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

    def __init__(self, state_size, action_size, seed):
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

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = PrioritizedExperienceReplay(state_size, BUFFER_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        state_ = torch.from_numpy(state).float().unsqueeze(0).to(device)
        next_state_ = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        action_ = torch.from_numpy(np.array(action, dtype=np.int64)).view(-1, 1).to(device)
        reward_ =  torch.from_numpy(np.array(reward)).view(-1, 1).to(device)
        done_ = torch.from_numpy(np.array(done)).view(-1, 1).to(device)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_state_).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = reward_ + (GAMMA * Q_targets_next * (1 - done_))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(state_).gather(1, action_)
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        loss = loss.cpu().data.numpy()

        self.memory.add(loss, (state, action, reward, next_state, done))

        # Save experience in replay memory
        #self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.tree.n_entries >= TRAIN_START:
                experiences, idxs, is_weights  = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, idxs, is_weights, GAMMA)

    def getAction(self, state, legal_actions, eps=0.):
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
        if random.random() > eps:
            action_values = action_values.cpu().data.numpy()[0] # Action values from Neural Network
            legal_actions = np.array([direction_to_action[action] for action in legal_actions]) # Possible action in the current state as number
            possible_actions = action_values[legal_actions] # Values of legal actions
            best_action_index = np.argmax(possible_actions)
            action = legal_actions[best_action_index]
            action = action_to_direction[action]
        else:
            action = random.choice(legal_actions)

        return action

    def learn(self, experiences, idxs, is_weights, gamma):
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
        Q_expected = self.qnetwork_local(states).gather(1, actions).type(torch.DoubleTensor).to(device)
        errors = torch.abs(Q_expected - Q_targets).cpu().data.numpy()

        # update priority
        for i in range(BATCH_SIZE):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        # Compute loss
        loss = (is_weights * F.mse_loss(Q_expected, Q_targets)).mean()
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
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

class PrioritizedExperienceReplay:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, state_size, capacity):
        self.state_size = state_size
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        is_weight = torch.from_numpy(is_weight.reshape(-1, 1).astype(np.float64)).to(device)

        batch = np.array(batch, dtype=object).transpose()

        states = np.vstack(batch[0])
        next_states = np.vstack(batch[3])
        if self.state_size[0] == 1:
            states = np.expand_dims(states, axis=1)
            next_states = np.expand_dims(next_states, axis=1)

        states = torch.from_numpy(states.astype(np.float64)).float().to(device)
        next_states = torch.from_numpy(next_states.astype(np.float64)).float().to(device)

        actions = torch.from_numpy(batch[1].reshape(-1, 1).astype(np.int64)).to(device)
        rewards = torch.from_numpy(batch[2].reshape(-1, 1).astype(np.float64)).to(device)
        dones = torch.from_numpy(batch[4].reshape(-1, 1).astype(np.float64)).to(device)

        return (states, actions, rewards, next_states, dones), idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)