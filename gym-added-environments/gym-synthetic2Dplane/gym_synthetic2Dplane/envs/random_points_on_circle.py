import random
import math
from collections import namedtuple

from matplotlib import pyplot as plt


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


if __name__ == '__main__':
    seed = 10
    random.seed(seed)
    Modes = namedtuple("Modes",['center', 'radius'])
    M1 = Modes((0, 1),1)
    M2 = Modes((0, -1), 1)
    M3 = Modes((0, 2), 2)
    modes = [M1, M2, M3]
    num_epochs = 50
    polar_coordinates = []
    random_radius = 0.2
    num_points_per_circle = 60
    for epochs in range(num_epochs):
        x_cartesian_coordinates = []
        y_cartesian_coordinates = []
        center, radius = random.choice(modes)
        if center[1]==1:
            color ='b'
        elif center[1]==-1:
            color ='g'
        elif center[1]==2:
            color='r'
        # random angle initialization
        theta = 2 * math.pi * random.random()
        # stochastic distance to generate random policy
        random_delta_distance = generate_random_point_on_circle(radius=random_radius)
        polar_coordinates.append((radius+random_delta_distance, theta))
        # polar to cartesian
        x_cartesian_coordinates.append(center[0] + (radius+random_delta_distance) * math.cos(theta))
        y_cartesian_coordinates.append(center[1] + (radius + random_delta_distance) * math.sin(theta))
        delta_theta = 2 * math.pi / num_points_per_circle
        for time in range(num_points_per_circle):
            theta = theta + delta_theta
            # stochastic distance to generate random policy
            random_delta_distance = generate_random_point_on_circle(radius=random_radius)
            polar_coordinates.append((radius + random_delta_distance, theta))
            # polar to cartesian
            x_cartesian_coordinates.append(center[0] + (radius + random_delta_distance) * math.cos(theta))
            y_cartesian_coordinates.append(center[1] + (radius + random_delta_distance) * math.sin(theta))
        plt.plot(x_cartesian_coordinates, y_cartesian_coordinates, color=color)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('circle')



