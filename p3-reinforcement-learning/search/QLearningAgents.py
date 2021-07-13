import random
from collections import defaultdict

import numpy as np
from sklearn.linear_model import SGDRegressor

from game import Agent
from pacman import Directions

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


class Estimator():
    def __init__(self, state_size, action_size):
        self.models = []
        state = np.zeros(state_size)
        for _ in range(action_size):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([state], [0])
            self.models.append(model)

    def predict(self, state, action=None):
        if not action:
            return np.array([m.predict([state])[0] for m in self.models])
        else:
            return self.models[action].predict([state])[0]

    def update(self, state, action, td_target):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        self.models[action].partial_fit([state], [td_target])

class ApproximateQLearningAgent(Agent):

    def __init__(self, state_size, action_size):
        self.discount = 1.0
        self.estimator = Estimator(state_size, action_size)
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.

    def getAction(self, state, legal, epsilon):
        if random.uniform(0, 1) > epsilon:
            return random.choice(legal)
        else:
            q_values_state = self.estimator.predict(state)
            # action = np.argmax(q_values_state)
            legal_actions = np.array([direction_to_action[action] for action in
                                      legal])  # Possible action in the current state as number
            possible_actions = q_values_state[legal_actions]  # Values of legal actions
            best_action_index = np.argmax(possible_actions)
            action = legal_actions[best_action_index]
            action = action_to_direction[action]

            return action

    def step(self, state, action, reward, next_state, done=None):
        Q_values_next_state = self.estimator.predict(next_state)
        td_target = reward + self.discount * np.max(Q_values_next_state)
        self.estimator.update(state, action, td_target)