from collections import defaultdict
import random

import numpy as np

from game import Agent
from pacman import Directions


class QLearningAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, action_size):
        self.Q = defaultdict(lambda: np.zeros(action_size))
        self.epsilon = 0.1
        self.lr = 0.01
        self.discount_factor = 0.99

    def get_Q_value_best_action(self, state):
        best_action = self.get_best_action(state)

        return self.Q[state][best_action]

    def get_best_action(self, state):
        return np.argmax(self.Q[state])

    def act(self, state, legal):
        if random.uniform(0, 1) > self.epsilon:
            return random.choice(legal)
        else:
            return self.get_best_action(state)

    def learn(self, state, action, reward, next_state):
        Q_current_state = self.Q[state, action]
        Q_next_state = self.get_Q_value_best_action(next_state)
        self.Q[state, action] += self.lr * (reward + self.discount * Q_next_state - Q_current_state)


class ApproximateQLearningAgent(Agent):

    def __init__(self, action_size):
        self.Q = defaultdict(lambda: np.zeros(action_size))
        self.epsilon = 0.1
        self.lr = 0.01
        self.discount_factor = 0.99

    def get_Q_value_best_action(self, state):
        best_action = self.get_best_action(state)

        return self.Q[state][best_action]

    def get_best_action(self, state):
        return np.argmax(self.Q[state])

    def act(self, state, legal):
        if random.uniform(0, 1) > self.epsilon:
            return random.choice(legal)
        else:
            return self.get_best_action(state)

    def learn(self, state, action, reward, next_state):
        Q_current_state = self.Q[state, action]
        Q_next_state = self.get_Q_value_best_action(next_state)
        self.Q[state, action] += self.lr * (reward + self.discount * Q_next_state - Q_current_state)