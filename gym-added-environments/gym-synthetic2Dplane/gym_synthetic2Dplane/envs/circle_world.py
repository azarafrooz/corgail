import random
import numpy as np
import math


def sample_start(set_diff):
    return random.choice(set_diff)


class State(object):
    def __init__(self, coordinates, feat_type='view'):
        assert(len(coordinates) == 2)
        self.coordinates = coordinates

        self.state = self.set_features(coordinates,
                                       feat_type=feat_type)
        self.feat_type = feat_type

    def get_features(self):
        assert self.state is not None, 'Feature not set'
        return self.state

    def feature_size(self, feat_type):
        if feat_type == 'view':
            return 2
        else:
            raise ValueError("Incorrect feat_type: {}".format(feat_type))

    def set_features(self, coordinates, feat_type='view'):
        state = np.zeros(self.feature_size(feat_type))
        state[0], state[1] = coordinates
        if feat_type == 'view':
            pass
        else:
            raise ValueError('Incorrect feature type {}'.format(feat_type))
        return state


class StateVector(State):
    def __init__(self, coordinates_arr, feat_type='view'):
        super(StateVector, self).__init__((1, 1),  # Fake coordinates not used
                                          feat_type=feat_type)
        self.coordinates_arr = coordinates_arr
        self.coordinates = coordinates_arr
        self.state_arr = self.set_features_array(coordinates_arr,
                                                 feat_type=feat_type)

    def set_features_array(self, coordinates_arr, feat_type='view'):
        batch_size = coordinates_arr.shape[0]
        state_arr = np.zeros((batch_size,
                              self.feature_size(feat_type)))
        for b in range(batch_size):
            state_arr[b, :] = self.set_features(
                    coordinates_arr[b, :],
                    feat_type=feat_type)
        return state_arr

    def get_features(self):
        assert self.state_arr is not None, 'Feature not set'
        return self.state_arr


class Action(object):
    def __init__(self, delta):
        self.delta = delta


class ActionVector(Action):
    def __init__(self, delta_arr):
        # Pass in a dummy action
        super(ActionVector, self).__init__(delta_arr[0])
        self.delta_arr = delta_arr


class TransitionFunction:
    def __init__(self):
        """
         Transition function for the grid world.
        """
        pass

    def __call__(self, state, action, batch_radius, time):
        if type(state) is StateVector:
            assert state.coordinates_arr.shape[0] == action.delta_arr.shape[0], \
                    "(State, Action) batch sizes do not match"
            batch_size = state.coordinates_arr.shape[0]
            new_coord_arr = np.zeros(state.coordinates.shape)
            for b in range(batch_size):
                state_coord = state.coordinates_arr[b, :]
                radius = batch_radius[b]
                if type(state_coord) is not type([]):
                    state_coord = state_coord.tolist()
                action_delta = action.delta_arr[b]
                new_coord = self.next_state(state_coord, action_delta, radius, time)
                new_coord_arr[b, :] = new_coord

            new_state = StateVector(new_coord_arr, feat_type=state.feat_type)
        elif type(state) is State:
            radius = batch_radius[0]
            new_coord = self.next_state(state.coordinates, action.delta, radius, time)
            new_state = State(new_coord, feat_type=state.feat_type)

        else:
            raise ValueError('Incorrect state type: {}'.format(type(state)))

        return new_state

    def next_state(self, state_coord, action_delta, radius, time):
        radius_idx, w = time // 60, (2 * math.pi) / 60
        # if time < 60:
            # radius_idx, w = 0, (2*math.pi)/60.0
        # elif time >= 60 and time < 90:
            # radius_idx, w = 1, (2*math.pi)/30.0
        # else:
            # radius_idx, w = 2, (2*math.pi)/15.0
        assert radius_idx < 3, "Invalid time input"
        radius = radius[radius_idx]
        dist = w * radius
        new_state = (state_coord[0] + action_delta[0] * dist,
                     state_coord[1] + action_delta[1] * dist)
        return new_state


class RewardFunction:
    def __init__(self, penalty, reward):
        self.terminal = False
        self.penalty = penalty
        self.reward = reward
        self.t = 0 # timer

    def __call__(self, state, action, c):
        self.t += 1
        if action.delta != np.argmax(c):
            return self.penalty
        else:
            return self.reward

    def reset(self):
        self.terminal = False
        self.t = 0



