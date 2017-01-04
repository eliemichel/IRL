"""
This module implements the "student dilemma" as briefly presented in RL lecture
as a gym module.
"""

# code structure is from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

import sys
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class StudentDilemmaEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        k = 2
        n = 7

        REST, WORK = 0, 1
        self.action_space = spaces.Discrete(k)

        self.state_space = range(n)

        self.transition = np.ndarray((k, n, n))
        self.transition[REST] = np.array([
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.6, 0.0, 0.0, 0.4, 0.0, 0.0],
            [0.0, 0.4, 0.6, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.0, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ])
        self.transition[WORK] = np.array([
            [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])

        assert (self.transition.sum(axis=2) == 1.0).all(), "invalid transition matrix"

        self.reward = np.array([[[
            0, 1, -1, -10, -10, 100, -1000
        ]]]).repeat(k, axis=0).repeat(n, axis=1)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        next_state = np.random.choice(self.state_space, p=self.transition[action, self.state])
        reward = self.reward[action, self.state, next_state]
        self.state = next_state
        done = False
        return self.state, reward, done, {}

    def _reset(self):
        self.state = 0
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write("state: {}\n".format(self.state))
        return outfile
