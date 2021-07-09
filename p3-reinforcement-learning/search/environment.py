import numpy as np

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
    def __init__(self, layout="mediumClassic", seed=27):
        self.layout = l.getLayout(layout)
        state_size = [1, self.layout.height, self.layout.width]
        self.beQuiet=True
        self.catchExceptions = False
        self.rules = pm.ClassicGameRules(timeout=30)
        self.pacman = dqnAgent.DQNAgent(state_size, action_size=5, seed=seed)
        self.reset()

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
        new_state = np.zeros((1, self.layout.height, self.layout.width))
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

        for agentIndex, agent in enumerate(self.game.agents):
            if not self.done():
                state = self.get_current_state()

                if agentIndex == 0:
                    state_as_image = self.convert_state_to_image(state)
                    legal = state.getLegalPacmanActions()
                    legal.remove(Directions.STOP)
                    action = agent.getAction(state_as_image, legal, eps)
                else:
                    action = agent.getAction(state)

                self.update_game_state(action)

                if agentIndex == 0:
                    state = state_as_image
                    action = self.get_action_as_number(action)
                    next_state = self.get_current_state()
                    next_state = self.convert_state_to_image(next_state)

                    reward = self.get_reward() - initial_reward
                    if reward >= 100: reward = 20
                    if reward <= -100: reward = -20
                    done = self.done()
                    agent.step(state, action, reward, next_state, done)

    def done(self, fast_check=False):
        if not self.game.gameOver:
            return False
        else:
            if fast_check: self.game.display.finish()
            return True