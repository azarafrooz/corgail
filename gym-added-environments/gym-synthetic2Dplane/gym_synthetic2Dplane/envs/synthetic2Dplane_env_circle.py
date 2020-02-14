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


def generate_random_point_on_circle(radius=0.1):
    """
    # using universality of uniform : assuming a random points on the circumvent of a unit, the distribution of points on
    # the circle should follow p(x)=2x \for 0<x<1 (linear and it should sum up to 1). universality of uniform implies:
    #  x~p(x) then cdf(x)~uniform ,
    # inverse of it implies cdf^-1(uniform) = p(x)
    # it is easy to note that cdf^-1 in the circle distribution is sqrt()
    # also more info is on the stackflow :
    # https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly/50746409#50746409
    :return: returns random coordinates on a circle
    """
    r = radius * math.sqrt(random.random())
    return r


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
        self.rads = np.arange(0, (2*np.pi), 0.1)
        Mode = namedtuple('Mode', ['radius'])
        self.modes = [Mode(1), Mode(2), Mode(3)]
        self.action_space = np.zeros((1, 1)) # actions are directions coordinates.
        # observations are positions from t-4 to t, each of which is (x_t,y_t)
        self.observation_space = np.zeros((4, 1))
        self.mode = random.choice(self.modes)
        self.Radius= self.mode.radius
        self.Radius += generate_random_point_on_circle(radius=0.0)
        # We keep the past coordinates form rendering/visualization
        self.polar_coordinates_histories = []
        self.state = deque(maxlen=4)
        radius = self.rads[0]
        # stochastic distance to generate random policy
        self.polar_coordinates_histories.append(radius)
        self.state.append(radius)
        # self.state.append([self.radius + random_delta_distance, theta])
        # Observation/states at time t constitutes of the coordinate positions from t-4 to t
        for time in range(1, 4):
            # stochastic distance to generate random policy
            radius = self.rads[time]
            self.polar_coordinates_histories.append(radius)
            self.state.append(radius)

        self.time = 4
        self.done = 0

    def stochastic_synthetic_policy(self):
        """
        Built-in expert
        :param action:
        :return: actions which are numpy array of polar coordinates (directions)
        """
        return np.array([self.rads[self.time]])

    def step(self, action):
        """
        :param action:  which are polar coordinates
        :return:  states which are the the cartesian coordinates
        """

        self.time += 1
        if self.time == len(self.rads):
            reward = 1/self.time
            self.done = 1
            return [np.array(list(self.state)), reward, self.done, self.mode]
        else:
            radius = action[0]
            self.polar_coordinates_histories.append(radius)
            self.state.append(radius)
            reward = 0
            return [np.array(list(self.state)), reward, self.done, self.mode]

    def reset(self):
        self.rads = np.arange(0, (2 * np.pi), 0.1)
        Mode = namedtuple('Mode', ['radius'])
        self.modes = [Mode(1), Mode(2), Mode(3)]
        self.action_space = np.zeros((1, 1))  # actions are directions coordinates.
        # observations are positions from t-4 to t, each of which is (x_t,y_t)
        self.observation_space = np.zeros((4, 1))
        self.mode = random.choice(self.modes)
        self.Radius = self.mode.radius
        self.Radius += generate_random_point_on_circle(radius=0.0)
        # We keep the past coordinates form rendering/visualization
        self.polar_coordinates_histories = []
        self.state = deque(maxlen=4)
        radius = self.rads[0]
        # stochastic distance to generate random policy
        self.polar_coordinates_histories.append(radius)
        self.state.append(radius)
        # self.state.append([self.radius + random_delta_distance, theta])
        # Observation/states at time t constitutes of the coordinate positions from t-4 to t
        for time in range(1, 4):
            # stochastic distance to generate random policy
            radius = self.rads[time]
            self.polar_coordinates_histories.append(radius)
            self.state.append(radius)

        self.time = 4
        self.done = 0
        return np.array(list(self.state))

    def render(self):
        if self.Radius < 2:
            color = 'b'
        else:
            color = 'g'
        plt.polar(self.rads[:len(self.polar_coordinates_histories)], self.polar_coordinates_histories, color=color, linestyle='dotted')
        plt.gca().set_aspect('equal', adjustable='box')
        # Turn interactive plotting off
        plt.ioff()
        plt.savefig('circle_world')
        # plt.clf()

