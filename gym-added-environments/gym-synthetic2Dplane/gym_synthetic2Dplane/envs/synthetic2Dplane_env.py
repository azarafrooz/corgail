import random, math
from collections import deque, namedtuple
import itertools

import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

class Synthetic2DPlane(gym.Env):
    metadata = {'render.modes': ['human']}
    """
    Leaf rose World Environment:
    Actions are the directions P_t at discrete time t.
    Observations at time t are cartesian coordinates from t-4 to t.
    The (unlabeled) expert demonstrations contain three distinct modes, each generate with a stochastic expert policy
    that produces a rose-like trajectories.
    """
    def __init__(self):
        Mode = namedtuple('Mode', ['A', 'B', 'C', 'K', 'Color'])
        self.modes = [Mode(10, 0, 1, 10, 'b'), Mode(10, 0, 1, -10, 'g'),
                      Mode(20, 0, 1, 20, 'r')]  # for circle
        self.action_space = np.zeros((1, 1)) # actions are directions coordinates.
        # observations are positions from t-5 to t, each of which is (x_t,y_t)
        self.observation_space = np.zeros(10)
        self.mode = random.choice(self.modes)
        self.A, self.B, self.K, self.C, self.color = self.mode
        self.rads = np.arange(0, (3 * np.pi), 0.1)
        # We keep the past coordinates form rendering/visualization
        self.polar_coordinates_histories = []
        self.state = deque(maxlen=5)
        radius = self.A + (self.B * np.cos(self.K * self.rads[0]))
        self.polar_coordinates_histories.append(radius)
        self.state.append([radius*np.cos(self.rads[0]), radius*np.sin(self.rads[0]) + self.C])
        for time in range(1, 5):
            radius = self.A + (self.B * np.cos(self.K * self.rads[time]))
            self.polar_coordinates_histories.append(radius)
            self.state.append([radius * np.cos(self.rads[time]), radius * np.sin(self.rads[time] + self.C)])

        self.time = 4
        self.done = 0

    def stochastic_synthetic_policy(self):
        """
        Built-in expert
        :param action:
        :return: actions which are numpy array of polar coordinates (directions)
        """
        delta_radius = self.A + (self.B * np.cos(self.K * self.rads[self.time])) - \
                       self.A + (self.B * np.cos(self.K * self.rads[self.time-1]))

        noise = np.random.normal(0.0, 0.01)
        return np.array([delta_radius + noise])

    def step(self, action):
        """
        :param action:  which are polar coordinates
        :return:  states which are the the cartesian coordinates
        """

        self.time += 1
        if self.time == len(self.rads):
            reward = 1/self.time
            self.done = 1
            return [np.array(list(itertools.chain.from_iterable(self.state))), reward, self.done, self.mode]
        else:
            delta_radius = action[0]
            past_radius = self.polar_coordinates_histories[-1]
            radius = delta_radius+past_radius
            self.polar_coordinates_histories.append(radius)
            self.state.append([radius * np.cos(self.rads[self.time]), radius * np.sin(self.rads[self.time])+ self.C])
            reward = 0
            return [np.array(list(itertools.chain.from_iterable(self.state))), reward, self.done, self.mode]

    def reset(self):
        Mode = namedtuple('Mode', ['A', 'B', 'K', 'C', 'Color'])
        self.modes = [Mode(10, 0, 1, 10, 'b'), Mode(10, 0, 1, -10, 'g'),
                      Mode(20, 0, 1, 20, 'r')]  # for circle
        self.action_space = np.zeros((1, 1))  
        self.observation_space = np.zeros(10)
        self.mode = random.choice(self.modes)
        self.A, self.B, self.K, self.C , self.color = self.mode
        self.rads = np.arange(0, (1 * 3 * np.pi), 0.1)
        self.polar_coordinates_histories = []
        self.state = deque(maxlen=5)
        radius = self.A + (self.B * np.cos(self.K * self.rads[0]))
        self.polar_coordinates_histories.append(radius)
        self.state.append([radius * np.cos(self.rads[0]), radius * np.sin(self.rads[0])+ self.C])
        for time in range(1, 5):
            radius = self.A + (self.B * np.cos(self.K * self.rads[time]))
            self.polar_coordinates_histories.append(radius)
            self.state.append([radius * np.cos(self.rads[time]), radius * np.sin(self.rads[time]) + self.C])

        self.time = 4
        self.done = 0
        return np.array(list(itertools.chain.from_iterable(self.state)))

    def render(self,mode,**kwargs):
        points = list(zip(self.rads[:len(self.polar_coordinates_histories)], self.polar_coordinates_histories))
        xs = [r * np.cos(p) for p,r in points]
        ys = [r * np.sin(p) + self.C for p, r in points]
        plt.plot(xs,ys,color=self.color, linestyle='dotted')
        plt.ioff()
        plt.savefig('circle_world_gail')

