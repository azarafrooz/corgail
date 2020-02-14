import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd
from torch.distributions.exponential import Exponential

from baselines.common.running_mean_std import RunningMeanStd

from a2c_ppo_acktr.utils import init, conv2d_size_out, Flatten
import math

class NoRegretDiscriminator(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=100, base=None, base_kwargs=None, device='cpu'):
        super(NoRegretDiscriminator, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(input_dim) == 3:
                base = CNNBase
            elif len(input_dim) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(input_dim[0], input_dim[1:], action_dim, hidden_dim, device=device, **base_kwargs)
        self.parameters = self.base.parameters()
        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        "size of rnn_hx"
        return self.base.recurrent_hidden_state_size

    def compute_grad_pen(self, expert_state, expert_action, policy_state, policy_action, lambda_=10):
        return self.base.compute_grad_pen(expert_state, expert_action, policy_state, policy_action, lambda_)

    def update(self, expert_loader, rollouts, discrim_queue, max_grad_norm, obsfilt=None, i_iter=0):
        return self.base.update(expert_loader, rollouts, discrim_queue, max_grad_norm, obsfilt , i_iter)

    def predict_reward(self, state, action, gamma, masks, discrim_queue, update_rms=True):
        return self.base.predict_reward( state, action, gamma, masks, discrim_queue, update_rms)

    def get_state_dict(self):
        return copy.deepcopy(self.base.state_dict())


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
    def __init__(self, num_inputs, input_size, action_space, hidden_size=64, recurrent=False, device='cpu'):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        self.device = device

        if recurrent:
            num_inputs = hidden_size

        init__ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.trunk = nn.Sequential(
            init__(nn.Linear(num_inputs + action_space.shape[0], hidden_size)), nn.Tanh(),
            init__(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init__(nn.Linear(hidden_size, 1)))

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr = 5e-5)

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        params = torch.Tensor().to(device)
        for param in self.parameters():
            params = torch.cat((params, param.view(-1)))
        self.randomness = Exponential(torch.ones(len(params))).sample().to(device)

        self.train()


    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen


    def strategy_update(self, expert_loader, rollouts, obsfilt):
        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[2]

            policy_d = self.trunk(
                torch.cat([policy_state, policy_action], dim=1))

            expert_state, expert_action = expert_batch
            expert_state = obsfilt(expert_state.numpy(), update=False)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(
                torch.cat([expert_state, expert_action], dim=1))

            expert_loss = -expert_d.mean()

            policy_loss = policy_d.mean()

            gail_loss = expert_loss + policy_loss

            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)
            #
            gail_loss = gail_loss + grad_pen

        return gail_loss

    def update(self, expert_loader, rollouts, discr_queue, max_grad_norm, obsfilt, i_iter = 0):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train()
        no_regret_gail_loss = torch.tensor(0.0).to(device)
        if len(discr_queue)>0: # First strategy get reward 0
            # Iterate through strategies in the queue.
            # Note: we loaded the parameters in order,
            # therefore we ended up with latest param, which optimizer updates.
            for strategy in discr_queue:
                self.load_state_dict(strategy)
                no_regret_gail_loss = no_regret_gail_loss + self.strategy_update(expert_loader, rollouts, obsfilt)

            no_regret_gail_loss = no_regret_gail_loss / len(discr_queue)

            # # in case, one need to regularize in the convex FTRL way.
            # reg = 1e-3
            # for param in self.parameters():
            #     no_regret_gail_loss = no_regret_gail_loss + param.pow(2).sum() * reg * (1 / math.log(1 + i_iter))

            # Discriminator loss is non-convex and not suitable for FTRL.
            # One recent technique is to inject Randomness for non-convex FTRL.
            params = torch.Tensor().to(device)
            for param in self.parameters():
                params = torch.cat((params, param.view(-1)))
            no_regret_gail_loss -= torch.dot(params, self.randomness)

            self.optimizer.zero_grad()
            no_regret_gail_loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()

        return copy.deepcopy(self.state_dict())

    def predict_strategy_reward(self, state, action, gamma, masks, update_rms):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            # 0 for expert-like states, goes to -inf for non-expert-like states
            # compatible with envs with traj cutoffs for good (expert-like) behavior
            # e.g. mountain car, which gets cut off when the car reaches the destination
            # s = torch.sigmoid(d)

            # 0 for non-expert-like states, goes to +inf for expert-like states
            # compatible with envs with traj cutoffs for bad (non-expert-like) behavior
            # e.g. walking simulations that get cut off when the robot falls over
            # s = -(1. - torch.sigmoid(d))

            # reward = s.log() - (1 - s).log()
            reward = d

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
         
    def predict_reward(self, state, action, gamma, masks, discr_queue, update_rms=True):
        """
        :param state:
        :param action:
        :param gamma:
        :param masks:
        :param discrim_queue:
        :param update_rms:
        :return: returns actor_reward
        """
        actor_reward = 0
        strategy_rewards = []
        if len(discr_queue) > 0:
            for strategy in discr_queue:
                self.load_state_dict(strategy)
                reward = self.predict_strategy_reward(state, action, gamma, masks, update_rms)
                strategy_rewards.append(reward)

            actor_reward = strategy_rewards[-1]
        return actor_reward

