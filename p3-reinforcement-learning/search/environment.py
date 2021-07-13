from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from util import *
from pacman import Directions
import pacman as pm
import layout as l
import textDisplay
import pacmanAgents, QLearningAgents, dqnAgent, ghostAgents

try:
    import boinc
    _BOINC_ENABLED = True
except:
    _BOINC_ENABLED = False

class Environment:
    def __init__(self, params, layout="mediumClassic", pacman_algorithm="DQN",
                 use_features=False, number_features=400, scaler=None, featurizer=None,
                 seed=27, save_states=False):
        self.layout = l.getLayout(layout)
        if not use_features:
            self.state_size = [4, self.layout.height - 2, self.layout.width - 2]
        else:
            self.state_size = [number_features]
            self.scaler = scaler
            self.featurizer = featurizer
        self.beQuiet=True
        self.use_features = use_features
        self.catchExceptions = False
        self.rules = pm.ClassicGameRules(timeout=30)
        self.pacman_algorithm = pacman_algorithm
        self.save_states = save_states
        if pacman_algorithm == "DQN":
            self.pacman = dqnAgent.DQNAgent(self.state_size, action_size=4, params=params, layout_used=layout, seed=seed)
        elif pacman_algorithm == "ApproxQLearning":
            self.pacman = QLearningAgents.ApproximateQLearningAgent(self.state_size, action_size=4)
        else:
            self.pacman = pacmanAgents.GreedyAgent()
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
        if self.save_states:
            self.states = []

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

        ## Features - Food
        if len(foods_positions) > 0:
            distance_closest_food = distances_pacman[np.argmin(distances)]
        else:
            distance_closest_food = np.array([width, height])
        number_of_foods_remaining = len(foods_positions)/number_of_foods
        is_food_near = distance_closest_food[0] <= 2 and distance_closest_food[1] <= 2

        ## Ghosts
        distance_ghosts_to_pacman = []
        distances = []
        for ghost_position in ghosts_position:
            distance_pacman = get_distance(pacman_position, ghost_position)
            distance_ghosts_to_pacman.append(distance_pacman)
            distances.append(np.abs(distance_pacman).sum())
        distance_ghosts_to_pacman = np.array(distance_ghosts_to_pacman)
        distances = np.array(distances)

        distance_closest_ghost = distance_ghosts_to_pacman[np.argmin(distances)]

        num_ghosts_close = len([True for distance in distance_ghosts_to_pacman if distance[0] <= 2 and distance[1] <= 2])/number_of_ghosts
        #num_ghosts_2_steps_away = len(distances[distances <= 2])/number_of_ghosts
        #num_ghosts_1_step_away = len(distances[distances <= 1])/number_of_ghosts

        ghost_2_steps_away = len(distances[distances <= 2]) > 0
        ghost_1_step_away = len(distances[distances <= 1]) > 0

        is_safe_to_eat = not ghost_2_steps_away and not ghost_1_step_away and is_food_near

        # Check if ghosts are scared
        scared_timers = []
        for agent in state.data.agentStates[1:]:
            scared_timers.append(agent.scaredTimer)
        scared_timers = np.array(scared_timers)
        distances_scareds = distances[scared_timers > 0]
        distances_pacman_scareds = distance_ghosts_to_pacman[scared_timers > 0]

        if len(distances_scareds) > 0:
            distance_closest_scared_ghost = distances_pacman_scareds[np.argmin(distances_scareds)]
            ghost_scared_2_steps_away = len(distances_scareds[distances_scareds <= 2]) > 0
            ghost_scared_1_step_away = len(distances_scareds[distances_scareds <= 1]) > 0
            is_any_ghost_scared = True
        else:
            distance_closest_scared_ghost = np.array([width, height])
            ghost_scared_2_steps_away = False
            ghost_scared_1_step_away = False
            is_any_ghost_scared = False

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
            is_any_capsule_available = True
        else:
            distance_closest_capsule = np.array([width, height])
            is_any_capsule_available = False

        features = np.hstack((distance_closest_food, is_food_near, is_safe_to_eat,
                              distance_closest_ghost, ghost_1_step_away, ghost_2_steps_away,
                              is_any_ghost_scared, distance_closest_scared_ghost, ghost_scared_1_step_away,
                              ghost_scared_2_steps_away,
                              is_any_capsule_available, distance_closest_capsule))

        if self.scaler and self.featurizer:
            scaled = self.scaler.transform(features.reshape(1, -1))
            return scaled[0]
        else:
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

    def adjust_reward(self, reward):
        rewards = {
            "win": 100,                 # Win the game             (Default: 500)
            "eat_capsule": 25,          # Eat a capsule            (Default: 200)
            "eat_food": 5,              # Eat a food               (Default: 10)
            "lose": -25,                # Lose the game            (Default: -500)
            "walking_without_eat": -1,  # Walk without eating food (Default: -1)
        }

        if reward >= 400: # Win the game
            reward = rewards["win"]
        elif reward >= 100: # Eat a capsule
            reward = rewards["eat_capsule"]
        elif reward >= 0: # Eat a food
            reward = rewards["eat_food"]
        elif reward >= -10: # Walk without eating food
            reward = rewards["walking_without_eat"]
        else: # Lose the game
            reward = rewards["lose"]

        return reward

    def step(self, eps):
        self.num_actions += 1
        initial_reward = self.get_reward()
        if not self.use_features:
            state_pacman = self.convert_state_to_image(self.get_current_state())
            self.last_states.append(state_pacman)
        else:
            state_pacman = self.convert_state_to_features(self.get_current_state())
        action_pacman = 0

        if self.save_states:
            self.states.append(state_pacman)

        for agentIndex, agent in enumerate(self.game.agents):
            if not self.done():
                state = self.get_current_state()

                if agentIndex == 0 and (self.pacman_algorithm == "DQN" or self.pacman_algorithm == "ApproxQLearning"):
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
        reward = self.adjust_reward(reward)
        done = self.done()

        if self.pacman_algorithm == "DQN" or self.pacman_algorithm == "ApproxQLearning":
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