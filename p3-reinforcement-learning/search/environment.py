from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from util import *
from pacman import Directions
import pacman as pm
import layout as l
import textDisplay
import dqnAgent, ghostAgents

try:
    import boinc
    _BOINC_ENABLED = True
except:
    _BOINC_ENABLED = False

class Environment:
    def __init__(self, params, layout="mediumClassic", use_features=False, seed=27):
        self.layout = l.getLayout(layout)
        if not use_features:
            self.state_size = [1, self.layout.height - 2, self.layout.width - 2]
        else:
            self.state_size = [16]
        self.beQuiet=True
        self.use_features = use_features
        self.catchExceptions = False
        self.rules = pm.ClassicGameRules(timeout=30)
        self.pacman = dqnAgent.DQNAgent(self.state_size, action_size=5, params=params, layout_used=layout, seed=seed)
        self.reset()
        # To keep track of progress
        self.wins = []
        self.wins_window = deque(maxlen=100)
        self.scores = []
        self.scores_window = deque(maxlen=100)
        self.average_scores = []

        print("Initial state:")
        initial_state = self.convert_state_to_image(self.get_current_state())
        initial_state = np.moveaxis(initial_state, [0, 1, 2], [-1, -3, -2])
        plt.imshow(initial_state, cmap="gray", vmin=0, vmax=1.0)
        plt.show()
        print("Shape: ", initial_state.shape)

    def reset(self):
        self.display = textDisplay.NullGraphics()
        self.ghosts = [ghostAgents.RandomGhost(i+1) for i in range(self.layout.getNumGhosts())]
        self.agents = [self.pacman] + self.ghosts
        self.game = self.rules.newGame(self.layout, self.pacman, self.ghosts,
                                       self.display, self.beQuiet, self.catchExceptions)

        # inform learning agents of the game start
        for i in range(len(self.agents)):
            agent = self.game.agents[i]
            if not agent:
                self.game.mute(i)
                # this is a null agent, meaning it failed to load
                # the other team wins
                print("Agent %d failed to load" % i, file=sys.stderr)
                self.game.unmute()
                self.game._agentCrash(i, quiet=True)
                return

        self.agentIndex = self.game.startingIndex
        self.numAgents = len(self.game.agents)


    def get_current_state(self):
        return self.game.state.deepCopy()

    def get_reward(self):
        return self.game.state.getScore()

    def update_game_state(self, action):
        # Execute the action
        self.game.state = self.game.state.generateSuccessor( self.agentIndex, action )
        # Change the display
        self.game.display.update( self.game.state.data )
        # Allow for game specific conditions (winning, losing, etc.)
        self.game.rules.process(self.game.state, self.game)
        # Track progress
        if self.agentIndex == self.numAgents + 1: self.game.numMoves += 1
        # Next agent
        self.agentIndex = ( self.agentIndex + 1 ) % self.numAgents

        if _BOINC_ENABLED:
            boinc.set_fraction_done(self.game.getProgress())

    def convert_state_to_image(self, state):
        state = str(state).split("\n")[:-2]
        new_state = np.zeros([1, self.layout.height - 2, self.layout.width - 2])
        state_dict = {
            '%': 0, '.': 225, 'o': 255,
            'G': 50, '<': 100, '>': 100,
            '^': 100, 'v': 100, ' ': 150,
            'P': 100
        }

        for i in range(1, self.layout.height - 1):
            for j in range(1, self.layout.width - 1):
                new_state[0][i - 1][j - 1] = state_dict[state[i][j]]

        #new_state = new_state.reshape(-1)
        new_state /= 255.0

        return new_state

    def convert_state_to_features(self, state):
        pacman_position = state.getPacmanPosition()
        ghosts_position = state.getGhostPositions()

        def has_food(x, y, state):
            if x >= self.layout.width or x < 0:
                return 0
            elif y >= self.layout.height or y < 0:
                return 0
            else:
                return int(state.hasFood(x, y))

        foods = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0: continue
                x, y = pacman_position[0] + i, pacman_position[1] + j
                foods.append(has_food(x, y, state))

        pacman_position = np.array(pacman_position, dtype=np.float32)
        pacman_position[0] = pacman_position[0] / self.layout.width
        pacman_position[1] = pacman_position[1] / self.layout.height

        ghosts_position = np.array(ghosts_position, dtype=np.float32)
        ghosts_position[:, 0] = ghosts_position[:, 0] / self.layout.width
        ghosts_position[:, 1] = ghosts_position[:, 1] / self.layout.height

        def euclidian_distance(x1, x2):
            return np.sqrt((x2[0] - x1[0]) ** 2 + (x2[1] - x1[1]) ** 2)

        distance_ghosts = []
        for ghost_position in ghosts_position:
            distance_ghosts.append(euclidian_distance(pacman_position, ghost_position))
        distance_ghosts = np.array(distance_ghosts)

        features = np.hstack((pacman_position, ghosts_position[0]))
        for ghost_position in ghosts_position[1:]:
            features = np.hstack((features, ghost_position))
        features = np.hstack((features, distance_ghosts))
        features = np.hstack((features, foods))

        return features

    def get_action_as_number(self, action):
        direction_to_action = {
            Directions.NORTH: 0,
            Directions.SOUTH: 1,
            Directions.EAST: 2,
            Directions.WEST: 3,
            Directions.STOP: 4
        }

        return direction_to_action[action]

    def step(self, eps):
        initial_reward = self.get_reward()
        if not self.use_features:
            state_pacman = self.convert_state_to_image(self.get_current_state())
        else:
            state_pacman = self.convert_state_to_features(self.get_current_state())
        action_pacman = 0

        for agentIndex, agent in enumerate(self.game.agents):
            if not self.done():
                state = self.get_current_state()

                if agentIndex == 0:
                    legal = state.getLegalPacmanActions()
                    legal.remove(Directions.STOP)
                    action = agent.getAction(state_pacman, legal, eps)
                    action_pacman = self.get_action_as_number(action)
                else:
                    action = agent.getAction(state)

                self.update_game_state(action)

        # Pacman agent learn
        if not self.use_features:
            next_state = self.convert_state_to_image(self.get_current_state())
        else:
            next_state = self.convert_state_to_features(self.get_current_state())

        reward = self.get_reward() - initial_reward
        done = self.done()

        self.pacman.step(state_pacman, action_pacman, reward, next_state, done)

    def done(self, fast_check=False):
        if not self.game.gameOver:
            return False
        else:
            if fast_check: self.game.display.finish()
            return True