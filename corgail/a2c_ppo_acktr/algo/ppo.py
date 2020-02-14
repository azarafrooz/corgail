import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.exponential import Exponential

import random, copy, math

class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 ftrl_mode = False,
                 correlated_mode=False):

        self.correlated_mode = correlated_mode # we inject embedding in this case
        self.ftrl_mode = ftrl_mode # we use regularization for FTRL.

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        params = torch.Tensor().to(self.device)
        for param in self.actor_critic.parameters():
            params = torch.cat((params, param.view(-1)))
        self.randomness = Exponential(torch.ones(len(params))).sample().to(self.device)

    def strategy_update_per_epoch(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        if self.actor_critic.is_recurrent:
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch)
        else:
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)

        for sample in data_generator:
            obs_batch, recurrent_hidden_states_batch, actions_batch, embeds_batch, \
               value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                    adv_targ = sample

            # Reshape to do in a single forward pass for all steps
            if self.correlated_mode:
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, embeds_batch)

            else:
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, None)

            ratio = torch.exp(action_log_probs -
                              old_action_log_probs_batch)
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                1.0 + self.clip_param) * adv_targ
            action_loss = -torch.min(surr1, surr2).mean()

            if self.use_clipped_value_loss:
                value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (
                    value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,
                                             value_losses_clipped).mean()

            else:
                value_loss = 0.5 * (return_batch - values).pow(2).mean()

        return value_loss, action_loss, dist_entropy

    def update(self, rollouts, i_iter = 0):
        value_loss_epoch, action_loss_epoch, dist_entropy_epoch = 0.0, 0.0, 0.0
        for epoch in range(self.ppo_epoch):
            value_loss, action_loss, dist_entropy = self.strategy_update_per_epoch(rollouts)
            self.optimizer.zero_grad()
            (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            value_loss_epoch += value_loss.item()
            action_loss_epoch += action_loss.item()
            dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch


    def mixed_update(self, rollouts, agent_queue, i_iter=0):
        value_loss_epoch, action_loss_epoch, dist_entropy_epoch = 0.0, 0.0, 0.0
        gains = 0.0
        gain_vector = []
        num_updates = self.num_mini_batch * self.ppo_epoch
        if len(agent_queue) < 1:
            return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, 0.0,\
                    copy.deepcopy(self.actor_critic.state_dict())
        for strategy in agent_queue:
            value_loss_epoch, action_loss_epoch, dist_entropy_epoch = 0.0, 0.0, 0.0
            self.actor_critic.load_state_dict(strategy)
            for epoch in range(self.ppo_epoch):
                value_loss, action_loss, dist_entropy = self.strategy_update_per_epoch(rollouts)
                action_loss_epoch += action_loss
                value_loss_epoch += value_loss
                dist_entropy_epoch += dist_entropy
                if not self.correlated_mode:
                    l2_reg = 1e-3
                    for param in self.actor_critic.parameters():
                        action_loss = action_loss + param.pow(2).sum() * l2_reg * (1/math.sqrt(1+i_iter))
                # params = torch.Tensor().to(self.device)
                # for param in self.actor_critic.parameters():
                #     params = torch.cat((params, param.view(-1).to(self.device)))
                # action_loss -= torch.dot(params, self.randomness)
                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
            gain_vector.append((-action_loss_epoch/num_updates).detach())


        action_loss_epoch = action_loss_epoch.item() / num_updates
        value_loss_epoch = value_loss_epoch.item() / num_updates
        dist_entropy_epoch = dist_entropy_epoch.item() / num_updates

        for i in range(len(gain_vector)):
            for j in range(i + 1, len(gain_vector)):
                gains = gains - torch.pow(gain_vector[i] - gain_vector[j], 2)

        gains = gains / (len(agent_queue) * len(agent_queue) / 4)

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch,\
               gains, copy.deepcopy(self.actor_critic.state_dict())

