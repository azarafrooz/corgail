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
    Circle World Environment (Used in InfoGAIL paper):
    Actions are the polar coordinates, directions A_t and P_t at discrete time t.
    Observations at time t are cartesian coordinates from t-4 to t.
    The (unlabeled) expert demonstrations contain three distinct modes, each generate with a stochastic expert policy
    that produces a circle-like trajectories.
    """
    def __init__(self):
        Mode = namedtuple('Mode', ['center', 'radius'])
        self.action_space = np.zeros((2,1)) # actions are (r_t, p_t), the polar coordinates.
        # observations are positions from t-4 to t, each of which is (x_t,y_t)
        self.observation_space = np.zeros((8, 1))
        self.modes = [Mode((0, 10), 10), Mode((0, -10), 10), Mode((0, 20), 20)]
        self.random_radius = 0.0  # generates random perturbation to radius r
        self.mode = random.choice(self.modes)
        self.center, self.radius = self.mode
        self.num_points_per_circle = 1 * self.radius  # Number of steps to circle back to where it started.
        self.delta_theta = 2 * math.pi / self.num_points_per_circle
        # We keep the past coordinates form rendering/visualization
        self.polar_coordinates_histories = []
        self.state = deque(maxlen=4)
        # random angle initialization
        theta = 2 * math.pi * random.random()
        theta = 0
        # stochastic distance to generate random policy
        random_delta_distance = generate_random_point_on_circle(radius=self.random_radius*self.radius)
        self.polar_coordinates_histories.append([self.radius + random_delta_distance, theta])
        self.state.append([self.center[0] + (self.radius + random_delta_distance) * math.cos(theta),
                           self.center[1] + (self.radius + random_delta_distance) * math.sin(theta)])
        # self.state.append([self.radius + random_delta_distance, theta])
        # Observation/states at time t constitutes of the coordinate positions from t-4 to t
        for time in range(3):
            # It is assumed that observations are
            theta = theta + self.delta_theta
            # stochastic distance to generate random policy
            random_delta_distance = generate_random_point_on_circle(radius=self.random_radius*self.radius)
            self.polar_coordinates_histories.append([self.radius + random_delta_distance, theta])
            self.state.append([self.center[0] + (self.radius + random_delta_distance) * math.cos(theta),
                               self.center[1] + (self.radius + random_delta_distance) * math.sin(theta)])

        self.time = 0
        self.done = 0

    def stochastic_synthetic_policy(self):
        """
        Built-in expert
        :param action:
        :return: actions which are numpy array of polar coordinates (directions)
        """

        _, theta = self.polar_coordinates_histories[-1]
        theta = theta + self.delta_theta
        # stochastic distance to generate random policy
        random_delta_distance = generate_random_point_on_circle(radius=self.random_radius*self.radius)
        action = [random_delta_distance, self.delta_theta]
        return np.array(action)

    def step(self, action):
        """
        :param action:  which are polar coordinates
        :return:  states which are the the cartesian coordinates
        """

        self.time += 1
        if self.time > self.num_points_per_circle+1:
            reward = 1/self.time
            self.done = 1
            return [np.array(list(itertools.chain.from_iterable(self.state))), reward, self.done, self.mode]
        else:
            delta_radius, delta_theta = action
            _, theta = self.polar_coordinates_histories[-1]
            theta = theta + delta_theta
            radius = self.radius + delta_radius
            new_coordinate = [radius, theta]
            self.polar_coordinates_histories.append(new_coordinate)
            self.state.append([self.center[0] + radius * math.cos(theta),
                               self.center[1] + radius * math.sin(theta)])
            reward = 0
            return [np.array(list(itertools.chain.from_iterable(self.state))), reward, self.done, self.mode]

    def reset(self):
        # restarts the polar coordinates
        self.mode = random.choice(self.modes)
        self.center, self.radius = self.mode
        self.num_points_per_circle = 1 * self.radius  # Number of steps to circle back to where it started.
        self.delta_theta = 2 * math.pi / self.num_points_per_circle
        self.polar_coordinates_histories = []
        self.state = deque(maxlen=4)
        # random angle initialization
        theta = 2 * math.pi * random.random()
        theta = 0
        # stochastic distance to generate random policy
        random_delta_distance = generate_random_point_on_circle(radius=self.random_radius*self.radius)
        self.polar_coordinates_histories.append([self.radius + random_delta_distance, theta])
        self.state.append([self.center[0] + (self.radius + random_delta_distance) * math.cos(theta),
                           self.center[1] + (self.radius + random_delta_distance) * math.sin(theta)])

        # Observation/states at time t constitutes of the positions from t-4 to t
        for time in range(3):
            # It is assumed that observations are
            theta = theta + self.delta_theta
            # stochastic distance to generate random policy
            random_delta_distance = generate_random_point_on_circle(radius=self.random_radius*self.radius)
            self.polar_coordinates_histories.append([self.radius + random_delta_distance, theta])
            self.state.append([self.center[0] + (self.radius + random_delta_distance) * math.cos(theta),
                               self.center[1] + (self.radius + random_delta_distance) * math.sin(theta)])

        self.time = 0
        self.done = 0
        return np.array(list(itertools.chain.from_iterable(self.state)))

    def render(self):
        # polar to cartesian
        x_cartesian_coordinates = [self.center[0] + r * math.cos(p) for r, p in self.polar_coordinates_histories[4:]]
        y_cartesian_coordinates = [self.center[1] + r * math.sin(p) for r, p in self.polar_coordinates_histories[4:]]
        if self.center[1]==10:
            color ='b'
            zorder = 10
        elif self.center[1]==-10:
            color ='g'
            zorder = 5
        elif self.center[1]==20:
            color='r'
            zorder = 1
        plt.plot(x_cartesian_coordinates, y_cartesian_coordinates, color=color, linestyle='dotted', zorder = zorder)
        plt.gca().set_aspect('equal', adjustable='box')
        # Turn interactive plotting off
        plt.ioff()
        plt.savefig('circle_world')
        # plt.clf()

