import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd

from baselines.common.running_mean_std import RunningMeanStd

from a2c_ppo_acktr.utils import init

import cv2

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


class Discriminator(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=100, base=None, base_kwargs=None, device = 'cpu'):
        super(Discriminator, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(input_dim) == 3:
                base = CNNBase
            elif len(input_dim) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(input_dim[0], input_dim[1:], action_dim, hidden_dim, device = device, **base_kwargs)
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
        return self.base.compute_grad_pen(self, expert_state, expert_action, policy_state, policy_action, lambda_=10)

    def update(self, expert_loader, rollouts, obsfilt=None):
        return self.base.update(expert_loader, rollouts, obsfilt)

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        return self.base.predict_reward(state, action, gamma, masks, update_rms=True)


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size, device = 'cpu'):
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
    def __init__(self, num_inputs, input_size, action_space, hidden_size=64, recurrent=False, device = 'cpu'):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        self.device = device

        if recurrent:
            num_inputs = hidden_size

        init__ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.trunk = nn.Sequential(
            init__(nn.Linear(num_inputs + action_space.shape[0], hidden_size)), nn.Tanh(),
            init__(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init__(nn.Linear(hidden_size, 1)))

        self.optimizer = torch.optim.Adam(self.parameters())

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

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

    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        n = 0
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

            # expert_loss = F.binary_cross_entropy_with_logits(
            #     expert_d,
            #     torch.ones(expert_d.size()).to(self.device))
            # policy_loss = F.binary_cross_entropy_with_logits(
            #     policy_d,
            #     torch.zeros(policy_d.size()).to(self.device))

            expert_loss = -expert_d.mean()

            policy_loss = policy_d.mean()

            gail_loss = expert_loss + policy_loss

            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()

            n += 1
            # before = list(self.parameters())[0].sum().clone()
            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
            # after = list(self.parameters())[0].sum().clone()
            # print(after, before)
        return loss / n

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
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
            # return reward


class CNNBase(NNBase):
    def __init__(self, num_inputs, input_size, action_space, hidden_size= 512 ,recurrent= False, device = 'cpu'):

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
            init__(nn.Linear(hidden_size + self.action_space.n, hidden_size//2)), nn.Tanh(),
            init__(nn.Linear(hidden_size//2, hidden_size//2)), nn.Tanh(),
            init__(nn.Linear(hidden_size//2, 1)))

        self.optimizer = torch.optim.Adam(self.parameters())

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):

            policy_state, policy_action = policy_batch[0], policy_batch[2]
            policy_state_embedding = self.cnn_trunk(policy_state/255.0)
            policy_d = self.trunk(torch.cat([policy_state_embedding,
                                  torch.nn.functional.one_hot(policy_action,self.action_space.n).squeeze(1).float()],
                                            dim=1))

            expert_state, expert_action = expert_batch
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_state_embedding = self.cnn_trunk(expert_state/255.0)
            expert_d = self.trunk(torch.cat([expert_state_embedding, expert_action], dim=1))

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            loss += gail_loss.item()
            n += 1

            self.optimizer.zero_grad()
            gail_loss.backward()
            self.optimizer.step()
        return loss / n

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            state_embedding = self.cnn_trunk(state/255.)
            d = self.trunk(torch.cat([state_embedding,
                       torch.nn.functional.one_hot(action,
                                                   self.action_space.n).squeeze(1).float()], dim=1))
            s = torch.sigmoid(d)
            reward = s.log() - (1 - s).log()
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)


class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, num_trajectories=4, subsample_frequency=20):
        try:
            all_trajectories = torch.load(file_name)
        except:
            #https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
            import os, pickle
            max_bytes = 2 ** 31 - 1
            bytes_in = bytearray(0)
            input_size = os.path.getsize(file_name)
            with open(file_name, 'rb') as f_in:
                for _ in range(0, input_size, max_bytes):
                    bytes_in += f_in.read(max_bytes)
            all_trajectories = pickle.loads(bytes_in)
        
        perm = torch.randperm(len(all_trajectories['states']))
        idx = perm[:num_trajectories]

        self.trajectories = {}
        
        # See https://github.com/pytorch/pytorch/issues/14886
        # .long() for fixing bug in torch v0.4.1
        start_idx = torch.randint(
            0, subsample_frequency, size=(num_trajectories, )).long()

        for k, v in all_trajectories.items():
            data = v[idx]

            if k != 'lengths':
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i][start_idx[i]::subsample_frequency])
                self.trajectories[k] = np.stack(samples)
            else:
                self.trajectories[k] = torch.from_numpy(data).long() // subsample_frequency

        self.i2traj_idx = {}
        self.i2i = {}
        
        self.length = self.trajectories['lengths'].sum().item()

        traj_idx = 0
        i = 0

        self.get_idx = []
        
        for j in range(self.length):
            
            while self.trajectories['lengths'][traj_idx].item() <= i:
                i -= self.trajectories['lengths'][traj_idx].item()
                traj_idx += 1

            self.get_idx.append((traj_idx, i))

            i += 1

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        traj_idx, i = self.get_idx[i]

        state = self.trajectories['states'][traj_idx][i]

        if isinstance(state,str):
            image_path = state
            state = cv2.imread(image_path)
            state = torch.from_numpy(state).float().permute(2, 0, 1)
        else:
            state = torch.from_numpy(state).float()

        return state, torch.from_numpy(self.trajectories['actions'][traj_idx][i]).float()

