import math
import os
import time
from collections import deque

import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail, regret_gail, cor_gail
from a2c_ppo_acktr.algo.correlator import Correlator
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

from torch.distributions import Normal

from baselines.common.vec_env.vec_normalize import VecNormalize

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # coinrun environments need to be treated differently.
    coinrun_envs = {'CoinRun': 'standard', 'CoinRun-Platforms': 'platform', 'Random-Mazes': 'maze'}

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma, args.log_dir, device, False,
                         coin_run_level=args.num_levels,difficulty=args.high_difficulty, coin_run_seed=args.seed)
    if args.env_name in coinrun_envs.keys():
        observation_space_shape = (3, 64, 64)
        args.save_dir = args.save_dir + "/NUM_LEVELS_{}".format(args.num_levels)  # Save the level info in the

    else:
        observation_space_shape = envs.observation_space.shape

    # trained model name
    if args.continue_ppo_training:
        actor_critic, _ = torch.load(os.path.join(args.check_point, args.env_name + ".pt"),
                                     map_location=torch.device(device))
    elif args.cor_gail:
        embed_size = args.embed_size
        actor_critic = Policy(
        observation_space_shape,
        envs.action_space,
        hidden_size = args.hidden_size,
        embed_size= embed_size,
        base_kwargs={'recurrent': args.recurrent_policy})
        actor_critic.to(device)
        correlator = Correlator(observation_space_shape, envs.action_space, hidden_dim=args.hidden_size,
                                embed_dim=embed_size, lr= args.lr,
                                device=device)

        correlator.to(device)
        embeds = torch.zeros(1, embed_size)
    else:
        embed_size = 0
        actor_critic = Policy(
            observation_space_shape,
            envs.action_space,
            hidden_size=args.hidden_size,
            base_kwargs={'recurrent': args.recurrent_policy})
        actor_critic.to(device)
        embeds = None

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            use_clipped_value_loss=True,
            ftrl_mode=args.cor_gail or args.no_regret_gail,
            correlated_mode=args.cor_gail)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail or args.no_regret_gail or args.cor_gail:
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=50, subsample_frequency=1) #if subsample set to a different number,
        # grad_pen might need adjustment
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)
        if args.gail:
            discr = gail.Discriminator(observation_space_shape, envs.action_space, device=device)
        if args.no_regret_gail or args.cor_gail:
            queue = deque(maxlen=args.queue_size)  # Strategy Queues: Each element of a queue is a dicr strategy
            agent_queue = deque(maxlen=args.queue_size)  # Strategy Queues: Each element of a queue is an agent strategy
            pruning_frequency= 1
        if args.no_regret_gail:
            discr = regret_gail.NoRegretDiscriminator(observation_space_shape, envs.action_space, device=device)
        if args.cor_gail:
            discr = cor_gail.CorDiscriminator(observation_space_shape, envs.action_space, hidden_size=args.hidden_size,
                                              embed_size=embed_size, device=device)
        discr.to(device)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              observation_space_shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size, embed_size)

    obs = envs.reset()

    rollouts.obs[0].copy_(obs)
    if args.cor_gail:
        rollouts.embeds[0].copy_(embeds)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions # Roll-out
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], rollouts.embeds[step])

            obs, reward, done, infos = envs.step(action.to('cpu'))
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
            # Sample mediating/correlating actions # Correlated Roll-out
            if args.cor_gail:
                embeds, embeds_log_prob, mean = correlator.act(rollouts.obs[step], rollouts.actions[step])
                rollouts.insert_embedding(embeds, embeds_log_prob)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1], rollouts.embeds[-1]).detach()

        if args.gail or args.no_regret_gail or args.cor_gail:
            if args.env_name not in {'CoinRun','Random-Mazes'}:
                if j >= 10:
                    envs.venv.eval()

            gail_epoch = args.gail_epoch
            if args.gail:
                if j < 10:
                    gail_epoch = 100 # Warm up

                # no need for gail epoch or warm up in the no-regret case and cor_gail.
            for _ in range(gail_epoch):
                if utils.get_vec_normalize(envs):
                    obfilt = utils.get_vec_normalize(envs)._obfilt
                else:
                    obfilt = None

                if args.gail:
                    discr.update(gail_train_loader, rollouts, obfilt)

                if args.no_regret_gail or args.cor_gail:
                    last_strategy = discr.update(gail_train_loader, rollouts, queue, args.max_grad_norm, obfilt, j)

            for step in range(args.num_steps):
                if args.gail:
                    rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])
                if args.no_regret_gail:
                    rollouts.rewards[step] = discr.predict_reward(
                        rollouts.obs[step], rollouts.actions[step], args.gamma,
                        rollouts.masks[step], queue)
                if args.cor_gail:
                    rollouts.rewards[step], correlator_reward = discr.predict_reward(
                        rollouts.obs[step], rollouts.actions[step], rollouts.embeds[step], args.gamma,
                        rollouts.masks[step], queue)

                    rollouts.correlated_reward[step] = correlator_reward


        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        if args.gail:
            value_loss, action_loss, dist_entropy = agent.update(rollouts, j)

        elif args.no_regret_gail or args.cor_gail:
            value_loss, action_loss, dist_entropy, agent_gains, agent_strategy = \
                agent.mixed_update(rollouts, agent_queue, j)

        if args.cor_gail:
            correlator.update(rollouts, agent_gains, args.max_grad_norm)

        if args.no_regret_gail or args.cor_gail:
            queue, _ = utils.queue_update(queue, pruning_frequency, args.queue_size, j,
                                          last_strategy)
            agent_queue, pruning_frequency = utils.queue_update(agent_queue, pruning_frequency, args.queue_size,
                                                                       j, agent_strategy)



        rollouts.after_update()
        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            if not args.cor_gail:
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], os.path.join(save_path, args.env_name + ".pt"))

            else:
                print("saving models in {}".format(os.path.join(save_path, args.env_name)))
                torch.save(correlator.state_dict()
                    ,os.path.join(save_path, args.env_name + "correlator.pt"))
                torch.save([actor_critic.state_dict(), getattr(utils.get_vec_normalize(envs), 'ob_rms', None)]
                           , os.path.join(save_path, args.env_name + "actor.pt"))


        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f},"
                " value loss/action loss {:.1f}/{}".format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), value_loss, action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
