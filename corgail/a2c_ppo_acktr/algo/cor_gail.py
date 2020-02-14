import random, copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd
from torch.distributions.exponential import Exponential

from baselines.common.running_mean_std import RunningMeanStd

from a2c_ppo_acktr.utils import init, conv2d_size_out, Flatten


class CorDiscriminator(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_size=100, embed_size=0, base=None, base_kwargs=None, device='cpu'):
        super(CorDiscriminator, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(input_dim) == 3:
                base = CNNBase
            elif len(input_dim) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(input_dim[0], input_dim[1:], action_dim, hidden_size, embed_size, device=device, **base_kwargs)
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

    def update(self, expert_loader, rollouts, discrim_queue, max_grad_norm, obsfilt=None, i_iter=0):
        return self.base.update(expert_loader, rollouts, discrim_queue, max_grad_norm, obsfilt, i_iter)

    def predict_reward(self, state, action, gamma, masks, discrim_queue, embedding, update_rms=True):
        return self.base.predict_reward( state, action, gamma, masks, discrim_queue, embedding, update_rms)

    def get_state_dict(self):
        return copy.deepcopy(self.base.state_dict())


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size, embed_size, device='cpu'):
        super(NNBase, self).__init__()
        self.device = device

        self.__hidden_size = hidden_size
        self.__recurrent = recurrent
        self.__embed_size = embed_size

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
    def __init__(self, num_inputs, input_size, action_space, hidden_size=64, embed_size=0, recurrent=False, device='cpu'):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size, embed_size)

        self.device = device

        if recurrent:
            num_inputs = hidden_size

        init__ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.trunk = nn.Sequential(
            init__(nn.Linear(num_inputs + action_space.shape[0] + embed_size, hidden_size)), nn.Tanh(),
            init__(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init__(nn.Linear(hidden_size, 1)))

        # self.optimizer = torch.optim.Adam(self.parameters(), lr= 3e-5)
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr = 5e-5)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

        self.train()

    def update(self, expert_loader, rollouts, discr_queue, max_grad_norm, obsfilt, i_iter = 0):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train()

        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action, embeddings = policy_batch[0], policy_batch[2], policy_batch[3]
            loss = torch.tensor(0.0).to(device)
            # Iterate through strategies in the queue.
            # Note: we loaded the parameters in order, therefore we end up with latest param, which optimizer updates.
            if len(discr_queue) < 1:
                return copy.deepcopy(self.state_dict())

            for strategy in discr_queue:
                self.load_state_dict(strategy)
                policy_d = self.trunk(
                    torch.cat([policy_state, policy_action, embeddings], dim=1))

                expert_state, expert_action = expert_batch
                expert_state = obsfilt(expert_state.numpy(), update=False)
                expert_state = torch.FloatTensor(expert_state).to(self.device)
                expert_action = expert_action.to(self.device)
                expert_d = self.trunk(
                    torch.cat([expert_state, expert_action, embeddings], dim=1))

                expert_loss = -expert_d.mean()

                policy_loss = policy_d.mean()

                loss = loss + expert_loss + policy_loss

            loss = loss/len(discr_queue)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm) # not necessary but it is conistently used across all NN modeules.
            self.optimizer.step()
            

        return copy.deepcopy(self.state_dict())

    def predict_strategy_reward(self, state, action, embedding, gamma, masks, update_rms):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action, embedding], dim=1))
            reward = d

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)

    def predict_reward(self, state, action, embedding, gamma, masks, discr_queue, update_rms=True):
        """
        :param state:
        :param action:
        :param gamma:
        :param masks:
        :param discrim_queue:
        :param update_rms:
        :return: returns actor_reward
        """
        actor_reward = gains = 0.0
        strategy_rewards = []
        if len(discr_queue) > 0:
            for strategy in discr_queue:
                self.load_state_dict(strategy)
                reward = self.predict_strategy_reward(state, action, embedding, gamma, masks, update_rms)
                strategy_rewards.append(reward)


            # gain gets used by correlator to compute maxEnt corEQ loss.
            # It quantifies, how much overall gain would be achieved by switching strategies.
            for i in range(len(strategy_rewards)):
                for j in range(i + 1, len(strategy_rewards)):
                    gains = gains - torch.pow(strategy_rewards[i] - strategy_rewards[j], 2)

            gains = gains / (len(discr_queue) * len(discr_queue) / 4)
            actor_reward = strategy_rewards[-1]

        return actor_reward, gains


class CNNBase(NNBase):
    def __init__(self, num_inputs, input_size, action_space, hidden_size=512, embed_size=0, recurrent=False, device='cpu'):

        super(CNNBase, self).__init__(recurrent, num_inputs, hidden_size, embed_size)

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
            init__(nn.Linear(hidden_size + self.action_space.n + embed_size, hidden_size // 2)), nn.Tanh(),
            init__(nn.Linear(hidden_size // 2, hidden_size // 2)), nn.Tanh(),
            init__(nn.Linear(hidden_size // 2, 1)))

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-5)
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=5e-5) # To be conistent with the wgan optimizer, althougt not necessary

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def update(self, expert_loader, rollouts, discr_queue, max_grad_norm, obsfilt, i_iter = 0):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for expert_batch, policy_batch in zip(expert_loader, policy_data_generator):
            policy_state, policy_action, correlated_embeddings = policy_batch[0], policy_batch[2], policy_batch[3]
            loss = torch.tensor(0.0).to(device)
            # Iterate through strategies in the queue.
            # Note: we loaded the parameters in order, therefore we ended up with latest param, which optimizer update.
            if len(discr_queue) < 1:
                return copy.deepcopy(self.state_dict())

            for strategy in discr_queue:
                self.load_state_dict(strategy)

                policy_state_embedding = self.cnn_trunk(policy_state/255.0)
                policy_d = self.trunk(
                    torch.cat([policy_state_embedding,
                               torch.nn.functional.one_hot(policy_action, self.action_space.n).squeeze(1).float(),
                               correlated_embeddings], dim=1))

                expert_state, expert_action = expert_batch
                expert_state = torch.FloatTensor(expert_state).to(self.device)
                expert_action = expert_action.to(self.device)

                expert_state_embedding = self.cnn_trunk(expert_state/255.)
                expert_d = self.trunk(torch.cat([expert_state_embedding, expert_action,
                                                 correlated_embeddings], dim=1))

                expert_loss = -expert_d.mean()

                policy_loss = policy_d.mean()

                loss = loss + expert_loss + policy_loss

            loss = loss/len(discr_queue)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()

        return copy.deepcopy(self.state_dict())

    def predict_strategy_reward(self, state, action, embedding, gamma, masks, update_rms):
        with torch.no_grad():
            self.eval()
            state_embedding = self.cnn_trunk(state/255.)
            d = self.trunk(torch.cat([state_embedding,
                                      torch.nn.functional.one_hot(action,self.action_space.n).squeeze(1).float(),
                                      embedding], dim=1))

            reward = d

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())
            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)

    def predict_reward(self, state, action, embedding, gamma, masks, discr_queue, update_rms=True):
        """
        :param state:
        :param action:
        :param gamma:
        :param masks:
        :param discrim_queue:
        :param update_rms:
        :return: returns actor_reward
        """
        actor_reward = gains = 0.0
        strategy_rewards = []
        if len(discr_queue) > 0:
            for strategy in discr_queue:
                self.load_state_dict(strategy)
                reward = self.predict_strategy_reward(state, action, embedding, gamma, masks, update_rms)
                strategy_rewards.append(reward)


            # gain gets used by correlator to compute maxEnt corEQ loss.
            # It quantifies, how much overall gain would be achieved by switching strategies.
            for i in range(len(strategy_rewards)):
                for j in range(i + 1, len(strategy_rewards)):
                    gains = gains - torch.pow(strategy_rewards[i] - strategy_rewards[j], 2)

            gains = gains / (len(discr_queue) * len(discr_queue) / 4)
            actor_reward = strategy_rewards[-1]

        return actor_reward, gains


