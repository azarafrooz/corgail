import os
import warnings
from typing import Dict

import cv2  # pytype:disable=import-error
import numpy as np
from gym import spaces

from baselines.common.vec_env import VecEnv, VecFrameStack

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

import torch

from tqdm import tqdm

def generate_expert_traj(load_dir, save_path= None, env_name= None, n_episodes= 100, save_images = False,
                         image_folder='recorded_images', seed =0):
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
    # We need to use the same statistics for normalization as used in training
    model, ob_rms = torch.load(os.path.join(load_dir, env_name + ".pt"), map_location=torch.device(device))
    model.eval()
    # coinrun environments need to be treated differently.
    env = make_vec_envs(
        env_name,
        seed + 1000,
        1,
        None,
        None,
        device='cpu',
        allow_early_resets=False)

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    # Sanity check
    assert (isinstance(env.observation_space, spaces.Box) or
            isinstance(env.observation_space, spaces.Discrete)), "Observation space type not supported"

    assert (isinstance(env.action_space, spaces.Box) or
            isinstance(env.action_space, spaces.Discrete)), "Action space type not supported"

    # Check if we need to record images
    obs_space = env.observation_space
    record_images = len(obs_space.shape) == 3 and obs_space.shape[-1] in [1, 3, 4] \
                    and obs_space.dtype == np.uint8
    if record_images and not save_images:
        warnings.warn("Observations are images but save_images was set False, so will save in numpy archive; "
                      "this can lead to higher memory usage.")
        record_images = False

    if not record_images and len(obs_space.shape) == 3 and obs_space.dtype == np.uint8:
        warnings.warn("The observations looks like images (shape = {}) "
                      "but the number of channel > 4, so it will be saved in the numpy archive "
                      "which can lead to high memory usage".format(obs_space.shape))

    image_ext = 'jpg'
    if record_images:
        # We save images as jpg or png, that have only 3/4 color channels
        if isinstance(env, VecFrameStack) and env.n_stack == 4:
            # assert env.n_stack < 5, "The current data recorder does no support"\
            #                          "VecFrameStack with n_stack > 4"
            image_ext = 'png'

        folder_path = os.path.dirname(save_path)
        image_folder = os.path.join(folder_path, image_folder)
        os.makedirs(image_folder, exist_ok=True)
        print("=" * 10)
        print("Images will be recorded to {}/".format(image_folder))
        print("Image shape: {}".format(obs_space.shape))
        print("=" * 10)


    traj_lens = []
    traj_actions = []
    traj_observations = []
    traj_rewards = []


    actions = []
    observations = []
    rewards = []
    episode_starts = []

    ep_idx = 0
    obs = env.reset()

    episode_starts.append(True)
    idx = 0

    # state and mask for recurrent policies
    state, mask = torch.zeros(1, model.recurrent_hidden_state_size), torch.zeros(1, 1)

    pbar = tqdm(total=n_episodes)

    num_steps_per_ecoch = 100

    while ep_idx in range(n_episodes):
        if record_images:
            image_path = os.path.join(image_folder, "{}.{}".format(idx, image_ext))
            obs_ = obs[0]
            obs_ = obs_.permute(1, 2, 0).numpy()
            # Convert from RGB to BGR
            # which is the format OpenCV expect
            if obs_.shape[-1] == 3:
                obs_ = cv2.cvtColor(obs_, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, obs_)
            observations.append(image_path)
        else:
            observations.append(obs.tolist())

        with torch.no_grad():
            value, action, _, recurrent_hidden_states = model.act(obs, state, mask)

        # Observe reward and next obs
        obs, reward, done, _ = env.step(action)

        # Use only first env
        mask = [done[0] for _ in range(env.num_envs)]
        action = action[0].tolist()
        reward = reward[0].item()
        done = done[0]
        actions.append(action)
        rewards.append(reward)
        idx += 1
        if done:
            traj_lens.append(idx)

            while idx<num_steps_per_ecoch:# padd the rest of the sequence with the latest info, to keep the tensors the same size
                actions.append(action)
                rewards.append(reward)
                observations.append(obs.tolist())
                idx+=1

            traj_actions.append(np.array(actions))
            traj_observations.append(np.array(observations))
            traj_rewards.append(rewards)

            # reset the episode records
            idx = 0
            actions = []
            observations = []
            rewards = []
            ep_idx += 1
            pbar.update(1)
            state = torch.zeros(1, model.recurrent_hidden_state_size)

        elif idx>=num_steps_per_ecoch: # is not done but is more than the number of allowed demonstrations per epoch
            traj_lens.append(idx)

            traj_actions.append(np.array(actions))
            traj_observations.append(np.array(observations))
            traj_rewards.append(rewards)

            idx = 0
            actions = []
            observations = []
            rewards = []
            ep_idx += 1
            pbar.update(1)
            state = torch.zeros(1, model.recurrent_hidden_state_size)

    pbar.close()

    assert len(traj_observations) == len(traj_actions) == len(traj_lens) == len(traj_rewards)

    traj_actions = np.array(traj_actions, dtype=float)
    traj_observations = np.array(traj_observations, dtype=float).squeeze(2)
    traj_rewards = np.array(traj_rewards, dtype=float)
    traj_lens = np.array(traj_lens)

    data = {
        'actions': traj_actions,
         'states': traj_observations,
         'rewards': traj_rewards,
         'lengths': traj_lens}
    # type: Dict[str, np.ndarray]

    for key, val in data.items():
        print(key, val.shape)

    torch.save(data, save_path)

    env.close()


if __name__=='__main__':
    load_dir = 'trained_models/ppo/'
    env_name = 'Pendulum-v0'
    generate_expert_traj(load_dir=load_dir, save_path='gail_experts/trajs_{}.pt'.format(env_name.lower()),
                         env_name=env_name, n_episodes=300, save_images=False, image_folder='recorded_images', seed=0)
