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
            self.state_size = [4, self.layout.height - 2, self.layout.width - 2]
        else:
            self.state_size = [11]
        self.beQuiet=True
        self.use_features = use_features
        self.catchExceptions = False
        self.rules = pm.ClassicGameRules(timeout=30)
        self.pacman = dqnAgent.DQNAgent(self.state_size, action_size=5, params=params, layout_used=layout, seed=seed)
        self.reset()
        # To keep track of progress
        self.wins = []
        self.wins_window = deque(maxlen=100)
        self.average_wins = []
        self.scores = []
        self.scores_window = deque(maxlen=100)
        self.average_scores = []

        print("Initial state:")
        self.show_current_state(show_shape=True)

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
        self.num_actions = 0
        self.last_states = deque(maxlen=4)
        self.last_next_states = deque(maxlen=4)

    def get_current_state(self):
        return self.game.state.deepCopy()

    def get_reward(self):
        return self.game.state.getScore()

    def show_current_state(self, show_shape=False):
        initial_state = self.convert_state_to_image(self.get_current_state())
        initial_state = np.moveaxis(initial_state, [0, 1, 2], [-1, -3, -2])
        plt.imshow(initial_state, cmap="gray", vmin=0, vmax=1.0)
        plt.show()
        if show_shape:
            print("Shape: ", initial_state.shape)

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
        def get_distance(x1, x2):
            return x2 - x1

        pacman_position = np.array(state.getPacmanPosition(), dtype=np.float32)
        ghosts_position = np.array(state.getGhostPositions(), dtype=np.float32)
        capsules_position = np.array(state.getCapsules(), dtype=np.float32)
        number_of_foods = self.layout.totalFood
        number_of_ghosts = self.layout.numGhosts
        width = self.layout.width
        height = self.layout.height

        ## Foods
        foods_positions = []
        distances_pacman = []
        distances = []
        foods = state.getFood().data
        for i in range(1, len(foods) - 1):
            for j in range(1, len(foods[0]) - 1):
                if foods[i][j]:
                    food_position = np.array([i, j], dtype=np.float32)
                    distance_pacman = get_distance(pacman_position, food_position)

                    distances_pacman.append(distance_pacman)
                    distances.append(np.abs(distance_pacman).sum())
                    foods_positions.append(food_position)
        foods_positions = np.array(foods_positions)
        distances_pacman = np.array(distances_pacman)
        distances = np.array(distances)

        number_of_foods_remaining = len(foods_positions)/number_of_foods

        if len(foods_positions) > 0:
            distance_closest_food = distances_pacman[np.argmin(distances)]
        else:
            distance_closest_food = np.array([width, height])

        ## Ghosts
        distances_pacman = []
        distances = []
        for ghost_position in ghosts_position:
            distance_pacman = get_distance(pacman_position, ghost_position)
            distances_pacman.append(distance_pacman)
            distances.append(np.abs(distance_pacman).sum())
        distances_pacman = np.array(distances_pacman)
        distances = np.array(distances)

        # food_near = distance_closest_food[0] <= 2 and distance_closest_food[1] <= 2
        food_near = len(distances[distances <= 2])

        # Check if ghosts are scared
        scared_timers = []
        for agent in state.data.agentStates[1:]:
            scared_timers.append(agent.scaredTimer)
        scared_timers = np.array(scared_timers)
        distances_scareds = distances[scared_timers > 0]
        distances_pacman_scareds = distances_pacman[scared_timers > 0]

        distance_closest_ghost = distances_pacman[np.argmin(distances)]
        num_ghosts_close = len([True for distance in distances_pacman if distance[0] <= 2 and distance[1] <= 2])/number_of_ghosts
        num_ghosts_2steps_away = len(distances[distances <= 2])/number_of_ghosts
        num_ghosts_1steps_away = len(distances[distances <= 1])/number_of_ghosts

        if len(distances_scareds) > 0:
            distance_closest_scared_ghost = distances_pacman_scareds[np.argmin(distances_scareds)]
            num_ghosts_scared_2steps_away = len(distances_scareds[distances_scareds <= 2])
            num_ghosts_scared_1steps_away = len(distances_scareds[distances_scareds <= 1])
        else:
            distance_closest_scared_ghost = np.array([width, height])
            num_ghosts_scared_2steps_away = 0
            num_ghosts_scared_1steps_away = 0

        #ghost_near_2 = distance_closest_ghost[0] <= 2 and distance_closest_ghost[1] <= 2
        #ghost_near_1 = distance_closest_ghost[0] <= 1 and distance_closest_ghost[1] <= 1
        ghost_near_2 = len(distances[distances <= 2]) > 0
        ghost_near_1 = len(distances[distances <= 1]) > 0

        safe_eat = not ghost_near_2 and not ghost_near_1 and food_near

        ## Capsules
        if len(capsules_position) > 0:
            distances_pacman = []
            distances = []
            for capsule_position in capsules_position:
                distance_pacman = get_distance(pacman_position, capsule_position)
                distances_pacman.append(distance_pacman)
                distances.append(np.abs(distance_pacman).sum())
            distances_pacman = np.array(distances_pacman)
            distances = np.array(distances)

            distance_closest_capsule = distances_pacman[np.argmin(distances)]
        else:
            distance_closest_capsule = np.array([width, height])

        distance_closest_food[0] = distance_closest_food[0] / width
        distance_closest_food[1] = distance_closest_food[1] / height
        distance_closest_ghost[0] = distance_closest_ghost[0] / width
        distance_closest_ghost[1] = distance_closest_ghost[1] / height
        distance_closest_capsule[0] = distance_closest_capsule[0] / width
        distance_closest_capsule[1] = distance_closest_capsule[1] / height
        distance_closest_scared_ghost[0] = distance_closest_scared_ghost[0] / width
        distance_closest_scared_ghost[1] = distance_closest_scared_ghost[1] / height
        """
        features = np.hstack((food_near, distance_closest_food,
                              distance_closest_ghost, ghost_near_1, ghost_near_2,
                              num_ghosts_close, num_ghosts_1steps_away, num_ghosts_2steps_away,
                              safe_eat,
                              distance_closest_capsule, distance_closest_scared_ghost,
                              num_ghosts_scared_2steps_away, num_ghosts_scared_1steps_away))
        """

        features = np.hstack((food_near, distance_closest_food, number_of_foods_remaining,
                              distance_closest_ghost, ghost_near_1, ghost_near_2,
                              #num_ghosts_close, num_ghosts_1steps_away, num_ghosts_2steps_away,
                              safe_eat, distance_closest_capsule))

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
        self.num_actions += 1
        initial_reward = self.get_reward()
        if not self.use_features:
            state_pacman = self.convert_state_to_image(self.get_current_state())
            self.last_states.append(state_pacman)
        else:
            state_pacman = self.convert_state_to_features(self.get_current_state())
        action_pacman = 0

        for agentIndex, agent in enumerate(self.game.agents):
            if not self.done():
                state = self.get_current_state()

                if agentIndex == 0:
                    legal = state.getLegalPacmanActions()
                    legal.remove(Directions.STOP)

                    if not self.use_features:
                        if self.num_actions > 5:
                            state_pacman = np.array(list(self.last_states))[:, 0, :, :]
                            action = agent.getAction(state_pacman, legal, eps)
                            action_pacman = self.get_action_as_number(action)
                        else:
                            action = random.choice(legal)
                    else:
                        action = agent.getAction(state_pacman, legal, eps)
                        action_pacman = self.get_action_as_number(action)
                else:
                    action = agent.getAction(state)

                self.update_game_state(action)

        # Pacman agent learn
        if not self.use_features:
            next_state = self.convert_state_to_image(self.get_current_state())
            self.last_next_states.append(next_state)
        else:
            next_state = self.convert_state_to_features(self.get_current_state())

        reward = self.get_reward() - initial_reward
        done = self.done()

        if not self.use_features:
            if self.num_actions > 5:
                next_state = np.array(list(self.last_next_states))[:, 0, :, :]
                self.pacman.step(state_pacman, action_pacman, reward, next_state, done)
        else:
            self.pacman.step(state_pacman, action_pacman, reward, next_state, done)

    def done(self, fast_check=False):
        if not self.game.gameOver:
            return False
        else:
            if fast_check: self.game.display.finish()
            return True