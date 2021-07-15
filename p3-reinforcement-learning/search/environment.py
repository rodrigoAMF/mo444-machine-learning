from collections import deque

import random
import numpy as np
import matplotlib.pyplot as plt

from util import *
from pacman import Directions
import pacman as pm
import layout as l
import textDisplay
import dqnAgent, ghostAgents, gaAgent

try:
    import boinc
    _BOINC_ENABLED = True
except:
    _BOINC_ENABLED = False

class Environment:

    def __init__(self, params, layout="mediumClassic", pacman_algorithm="DQN", use_features=True, seed=27,
                 print_steps=False, test_only=False, checkpoint_to_use=None):
        self.layout = l.getLayout(layout)
        if not use_features:
            self.state_size = [1, self.layout.height - 2, self.layout.width - 2]
        else:
            self.state_size = [11]
        self.beQuiet=True
        self.use_features = use_features
        self.catchExceptions = False
        self.rules = pm.ClassicGameRules(timeout=30)
        self.print_steps = print_steps
        self.test_only = test_only

        self.pacman_algorithm = pacman_algorithm
        if pacman_algorithm == "DQN":
            self.pacman = dqnAgent.DQNAgent(self.state_size, action_size=4, params=params, layout_used=layout, seed=seed,
                                            test_only=test_only, checkpoint_to_use=checkpoint_to_use)
        if pacman_algorithm == "GA":
            self.pacman = gaAgent.gaAgent(self.state_size, action_size=4, params=params, layout_used=layout, seed=seed)
#            self.pacman = pacmanAgents.GreedyAgent()
#        else:
#            self.pacman = pacmanAgents.GreedyAgent()

        self.reset()
        # To keep track of progress
        self.wins = []
        self.wins_window = deque(maxlen=100)
        self.average_wins = []
        self.scores = []
        self.scores_window = deque(maxlen=100)
        self.average_scores = []
        self.rewards_window = deque(maxlen=100)
        self.average_rewards = []
        self.foods_eaten_window = deque(maxlen=100)
        self.average_foods_eaten = []
        self.num_actions_window = deque(maxlen=100)
        self.average_num_actions = []

        print("Initial state of the environment looks like:")
        self.print_env_maze()

    def print_env_maze(self):
        initial_state = self.convert_state_to_image(self.get_current_state())
        initial_state = np.moveaxis(initial_state, [0, 1, 2], [-1, -3, -2])
        plt.imshow(initial_state, cmap="gray", vmin=0, vmax=1.0)
        plt.show()
        print("Layout Height: ", initial_state.shape[0])
        print("Layout Width: ", initial_state.shape[1])
        print()

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
        self.foods_eaten = 0
        self.total_reward = 0

    def get_current_state(self):
        return self.game.state.deepCopy()

    def get_reward(self, state, done):
        if self.layout == "smallClassic":
            if done:
                if state.data._win:
                    return 500.0 * state.data.combo
                else:
                    return -50.0
            elif state.data.eaten_food or state.data.eaten_capsule or state.data.eaten_ghost:
                if state.data.eaten_food:
                    reward = 10.0 * state.data.combo
                elif state.data.eaten_capsule:
                    reward = 50.0 * state.data.combo
                elif state.data.eaten_ghost:
                    reward = 100.0 * state.data.combo

                state.data.combo += 0.2

                return reward
            else:
                state.data.combo = 1.0

                return -1.0
        else:
            if done:
                if state.data._win:
                    return 500.0*state.data.combo
                else:
                    return -50.0
            elif state.data.eaten_food or state.data.eaten_capsule or state.data.eaten_ghost:
                if state.data.eaten_food:
                    self.foods_eaten += 1
                    reward = 10.0
                elif state.data.eaten_capsule:
                    reward = 50.0
                elif state.data.eaten_ghost:
                    reward = 100.0

                state.data.combo += 0.2

                return reward
            else:
                state.data.combo = 1.0

                return -1.0

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
        new_state = np.zeros([1, self.layout.height, self.layout.width])
        state_dict = {
            '%': 0, '.': 225, 'o': 255,
            'G': 50, '<': 100, '>': 100,
            '^': 100, 'v': 100, ' ': 150,
            'P': 100
        }

        for i in range(self.layout.height):
            for j in range(self.layout.width):
                new_state[0][i][j] = state_dict[state[i][j]]

        #new_state = new_state.reshape(-1)
        new_state /= 255.0

        return new_state

    def convert_state_to_features(self, state):
        def get_distance(x1, x2):
            return x2 - x1

        pacman_position = np.array(state.getPacmanPosition(), dtype=np.float32)
        ghosts_position = np.array(state.getGhostPositions(), dtype=np.float32)
        capsules_position = np.array(state.getCapsules(), dtype=np.float32)
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

        distance_closest_ghost = distances_pacman[np.argmin(distances)]

        ghost_near_2 = distance_closest_ghost[0] <= 2 and distance_closest_ghost[1] <= 2
        ghost_near_1 = distance_closest_ghost[0] <= 1 and distance_closest_ghost[1] <= 1
        food_near = distance_closest_food[0] <= 2 and distance_closest_food[1] <= 2

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

        scared_timers = [0, 0]
        # for agent in state.data.agentStates[1:]:
        #    scared_timers.append(agent.scaredTimer)
        scared_timers = np.array(scared_timers)
        # If one of the ghosts is scared
        is_scared = np.any(scared_timers) > 0


        distance_closest_food[0] = distance_closest_food[0] / width
        distance_closest_food[1] = distance_closest_food[1] / height
        distance_closest_ghost[0] = distance_closest_ghost[0] / width
        distance_closest_ghost[1] = distance_closest_ghost[1] / height
        distance_closest_capsule[0] = distance_closest_capsule[0] / width
        distance_closest_capsule[1] = distance_closest_capsule[1] / height

        features = np.hstack((food_near, distance_closest_food,
                              distance_closest_ghost, ghost_near_1, ghost_near_2,
                              safe_eat,
                              distance_closest_capsule, is_scared))

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

    def step(self):
        self.num_actions += 1
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
                    if self.pacman_algorithm == "GA":
                        self.pacman.features = state_pacman
                    action = agent.getAction(state_pacman, legal)
                    action_pacman = self.get_action_as_number(action)
                else:
                    action = agent.getAction(state)

                self.update_game_state(action)

        # Pacman agent learn
        if not self.use_features:
            next_state = self.convert_state_to_image(self.get_current_state())
        else:
            next_state = self.convert_state_to_features(self.get_current_state())

        done = self.done()
        reward = self.get_reward(self.game.state, done)
        self.total_reward += reward

        if self.pacman_algorithm != "GA" and not self.test_only:
            self.pacman.step(state_pacman, action_pacman, reward, next_state, done)
        if self.print_steps:
            self.print_env_maze()

    def done(self, fast_check=False):
        if not self.game.gameOver:
            return False
        else:
            if not fast_check: self.game.display.finish()
            return True

    def compute_scores(self):
        def calculate_average(deque_object):
            return sum(deque_object) / float(len(deque_object))

        self.scores.append(self.game.state.getScore())
        self.scores_window.append(self.game.state.getScore())

        self.rewards_window.append(self.total_reward)

        self.wins.append(self.game.state.isWin())
        self.wins_window.append(self.game.state.isWin())

        self.foods_eaten_window.append(self.foods_eaten)
        self.num_actions_window.append(self.num_actions)

        average_score = calculate_average(self.scores_window)
        average_reward = calculate_average(self.rewards_window)
        average_foods_eaten = calculate_average(self.foods_eaten_window)
        average_num_actions = calculate_average(self.num_actions_window)
        winrate = calculate_average(self.wins_window) * 100.0

        self.average_scores.append(average_score)
        self.average_rewards.append(average_reward)
        self.average_foods_eaten.append(average_foods_eaten)
        self.average_num_actions.append(average_num_actions)
        self.average_wins.append(winrate)

        return average_score, average_reward, average_foods_eaten, average_num_actions, winrate

    def choose_random_position_for_ghosts(self):
        state = self.convert_state_to_image(self.get_current_state())
        state = state > 0.39216
        available_positions = []
        for i in range(self.layout.height):
            for j in range(self.layout.width):
                if state[0][i][j]:
                    available_positions.append((False, (j, self.layout.height - 1 - i)))

        import random

        return random.sample(available_positions, len(self.agents) - 1)

    def set_random_positions_to_ghosts(self):
        positions = self.choose_random_position_for_ghosts()

        for i, position in enumerate(positions):
            self.layout.agentPositions[i + 1] = position