import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd
from torch.distributions import Normal

from baselines.common.running_mean_std import RunningMeanStd

from a2c_ppo_acktr.utils import init, Flatten, conv2d_size_out

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# # Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Correlator(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=100, embed_dim=1, lr=3e-5, base=None, base_kwargs=None, device='cpu'):
        super(Correlator, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(input_dim) == 3:
                base = CNNBase
            elif len(input_dim) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(input_dim[0], input_dim[1:], action_dim, hidden_dim, embed_dim, lr, device=device, **base_kwargs)
        self.parameters = self.base.parameters()

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        "size of rnn_hx"
        return self.base.recurrent_hidden_state_size

    def act(self, state, action):
        return self.base.act(state, action)

    def update(self, rollouts, agent_gains, max_grad_norm):
        return self.base.update(rollouts, agent_gains, max_grad_norm)


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size, device='cpu'):
        super(NNBase, self).__init__()
        self.device = device

        self.__hidden_size = hidden_size
        self.__recurrent = recurrent

        if recurrent:
            return NotImplementedError

    @property
    def is_recurrent(self):
        return self.__recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self.__recurrent:
            return self.__hidden_size
        else:
            return 1

    @property
    def output_size(self):
        return self.__hidden_size

    def _forward_gru(self, x, hxs, masks):
        return NotImplementedError


class MLPBase(NNBase):
    def __init__(self, num_inputs, input_size, action_space, hidden_size=64, embed_size=1, lr= 3e-5,
                 recurrent=False, device='cpu'):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        self.device = device

        if recurrent:
            num_inputs = hidden_size


        self.trunk = nn.Sequential(nn.Linear(num_inputs + action_space.shape[0], hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size), nn.ReLU())


        self.embed_mean = nn.Linear(hidden_size, embed_size)
        self.embed_log_std = nn.Linear(hidden_size, embed_size)

        self.apply(weights_init_)

        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        x = self.trunk(x)
        mean, log_std = self.embed_mean(x), self.embed_log_std(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = torch.exp(log_std)
        return mean, std

    def act(self, obs, action):
        """
        pathwise derivative estimator for taking embedding actions.
        :param x:
        :return:
        """
        mean, std = self.forward(obs, action)
        normal = Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean

    def update(self, rollouts, agent_gains, max_grad_norm):
        self.train()
        reward = rollouts.correlated_reward + agent_gains # agent gains are per an state trajectory but discrim gains/rewards are per state-action
        loss = -(rollouts.embeds_log_probs[1:].to(self.device) * reward).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        self.optimizer.step()
        return loss


# the CNN correlator/mediator. # we use the term correlator and mediator interchangeably. 
class CNNBase(NNBase):
    def __init__(self, num_inputs, input_size, action_space, hidden_size=512, embed_size=1, lr= 3e-5,
                 recurrent=False, device='cpu'):

        super(CNNBase, self).__init__(recurrent, num_inputs, hidden_size)

        self.device = device
        self.action_space = action_space

        h, w = input_size
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4)
        w_out = conv2d_size_out(w, kernel_size=8, stride=4)
        h_out = conv2d_size_out(h, kernel_size=8, stride=4)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        w_out = conv2d_size_out(w_out, kernel_size=4, stride=2)
        h_out = conv2d_size_out(h_out, kernel_size=4, stride=2)

        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        w_out = conv2d_size_out(w_out, kernel_size=3, stride=1)
        h_out = conv2d_size_out(h_out, kernel_size=3, stride=1)

        init_cnn_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0), nn.init.calculate_gain('relu'))

        self.cnn_trunk = nn.Sequential(
            init_cnn_(self.conv1), nn.ReLU(),
            init_cnn_(self.conv2), nn.ReLU(),
            init_cnn_(self.conv3), nn.ReLU(), Flatten(),
            init_cnn_(nn.Linear(32 * h_out * w_out, hidden_size)), nn.ReLU())

        init__ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

    
        self.trunk = nn.Sequential(
            init__(nn.Linear(hidden_size + self.action_space.n, hidden_size // 2)), nn.ReLU())

        self.embed_mean = nn.Linear(hidden_size // 2, embed_size)
        self.embed_log_std = nn.Linear(hidden_size // 2, embed_size)

        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

        self.device = device

    def forward(self, obs, policy_action):
        with torch.autograd.set_detect_anomaly(True):
            obs_embedding = self.cnn_trunk(obs/255.0)
            x = torch.nn.functional.one_hot(policy_action, self.action_space.n).squeeze(1).float()
            x = torch.cat([obs_embedding, x], dim=1)
            x = self.trunk(x)
            mean, log_std = self.embed_mean(x), self.embed_log_std(x)
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            std = torch.exp(log_std)
            return mean, std


    def act(self, obs, policy_action):
        """
        pathwise derivative estimator for taking embedding actions.
        :param x:
        :return:
        """
        with torch.autograd.set_detect_anomaly(True):
            mean, std = self.forward(obs, policy_action)
            normal = Normal(mean, std)
            x = normal.rsample()
            y = torch.tanh(x)
            action = y * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x)
            # Enforcing Action Bound
            log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + epsilon)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean

    def update(self, rollouts, agent_gains, max_grad_norm):
         self.train()
         with torch.autograd.set_detect_anomaly(True):
            reward = rollouts.correlated_reward + agent_gains # agent gains are per an state trajectory but discrim gains/rewards are per state-action
            loss = -(rollouts.embeds_log_probs[1:].to(self.device) * reward).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()
