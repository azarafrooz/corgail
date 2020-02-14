import os
import warnings
from typing import Dict

import cv2  # pytype:disable=import-error
import numpy as np
import gym
import gym_synthetic2Dplane
from gym import spaces


from baselines.common.vec_env import VecEnv, VecFrameStack

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

import torch

from tqdm import tqdm


def generate_expert_traj(save_path= None, env_name= None, n_episodes= 100, seed =0):
    """
    Train expert controller (if needed) and record expert trajectories.
    .. note::
        only Box and Discrete spaces are supported for now.
    :param load_dir: The path to the saved expert model.
    :param save_path: (str) Path without the extension where the expert dataset will be saved
        (ex: 'expert_cartpole' -> creates 'expert_cartpole.npz').
        If not specified, it will not save, and just return the generated expert trajectories.
        This parameter must be specified for image-based environments.
    :param env_name: (gym.Env) The name of the environment
    :param n_episodes: (int) Number of trajectories (episodes) to record
    :param save_images: bool, set to True if you want to save images on disk instead of saving the observations.
    If set true, you may need to pass observation as numpy instead of torch.tensor.
    :param image_folder: (str) When using images, folder that will be used to record images.
    :return: (dict) the generated expert trajectories.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_name)
    traj_lens = []
    traj_actions = []
    traj_observations = []


    actions = []
    observations = []
    episode_starts = []

    ep_idx = 0
    obs = env.reset()

    episode_starts.append(True)
    idx = 0

    pbar = tqdm(total=n_episodes)

    num_steps_per_ecoch = 200

    while ep_idx in range(n_episodes):
        observations.append(obs.tolist())
        # select an action based on pre-defined/synthetic stochastic policy. Since the env is synthetic, we have
        # access to the optimal stochastic policy,i.e no need learn any policy.
        action = env.stochastic_synthetic_policy()
        # Observe reward and next obs
        obs, reward, done, _ = env.step(action)
        # Use only first env
        action = action.tolist()
        actions.append(action)
        idx += 1
        if done:
            traj_lens.append(idx)

            while idx<num_steps_per_ecoch:# padd the rest of the sequence with the latest info, to keep the tensors the same size
                actions.append(action)
                observations.append(obs.tolist())
                idx+=1

            traj_actions.append(np.array(actions))
            traj_observations.append(np.array(observations))

            # reset the episode records
            idx = 0
            actions = []
            observations = []
            ep_idx += 1
            pbar.update(1)
            env.render(mode='Human')
            obs = env.reset()

        elif idx>=num_steps_per_ecoch: # is not done but is more than the number of allowed demonstrations per epoch
            traj_lens.append(idx)

            traj_actions.append(np.array(actions))
            traj_observations.append(np.array(observations))

            idx = 0
            actions = []
            observations = []
            ep_idx += 1
            pbar.update(1)
            obs = env.reset()

    pbar.close()

    assert len(traj_observations) == len(traj_actions) == len(traj_lens)

    traj_actions = np.array(traj_actions, dtype=float)
    traj_observations = np.array(traj_observations, dtype=float)
    traj_lens = np.array(traj_lens)

    data = {
        'actions': traj_actions,
         'states': traj_observations,
         'lengths': traj_lens}
    # type: Dict[str, np.ndarray]

    for key, val in data.items():
        print(key, val.shape)

    torch.save(data, save_path)

    env.close()


if __name__=='__main__':
    env_name = 'synthetic2Dplane-v0'
    generate_expert_traj(save_path='gail_experts/trajs_{}.pt'.format(env_name.lower()),
                         env_name=env_name, n_episodes=100, seed=0)
