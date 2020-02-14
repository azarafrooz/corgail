import glob
import os

import math

import torch
import torch.nn as nn

from a2c_ppo_acktr.envs import VecNormalize

import copy


def conv2d_size_out(size, kernel_size, stride):
    """
    Helper function for adjusting the size correctly in CNNs.
    :param size:
    :param kernel_size:
    :param stride:
    :return:
    """
    return (size - kernel_size) // stride + 1


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def queue_update(queue, m, K, t, ft, inc=2000):
    """
    :param queue: Queue containing the models as game strategies
    :param m: spacing parameter
    :param K: Queue size
    :param inc: hyper-parameter that needs to be adjusted
    :param t: current step
    :param ft: new model
    :return: updated queue and new spacing parameter
    """
    if t % m ==0 and len(queue)==K:
        queue.append(ft)  # we remove a model from the end of the queue, which is the oldest one
        # print("the new interval for queue update is: {}".format(inc))
        return queue, m+inc
    elif len(queue) ==K: # override the head (first) element with the current model
        queue.pop()
        queue.append(ft)
        return queue, m
    else:
        queue.append(ft)
        return queue, m
