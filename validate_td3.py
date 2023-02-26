import argparse
import datetime
import gym
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sys
import scipy.io as sio

from itertools import count
from tensorflow import keras
from td3_agent import Agent
from utils import plot_save_cdf, plot_SEs
from gym.envs.registration import make


np.random.seed(2022)
tf.random.set_seed(2022)


def validate_train_process(mimo_net, td3_agent, num_tests=2500):
    # Load test data
    ref_data = sio.loadmat("D:\Work\matlab_ref\Deep-Learning-Power-Allocation-in-Massive-MIMO-master\MyDataFile.mat")
    ref_pos = ref_data["input_positions"][:, :, :num_tests]
    ref_maxprod_sumSE = np.sum(ref_data['SE_MMMSE_maxprod'], axis=(0, 1))[:num_tests]

    episode_esc_sumSE = []
    episode_max_sumSE = []

    episode_durations = []
    episode_total_reward = []

    for episode in tqdm(range(num_tests)):
        # This is a 1-D numpy array.
        state = mimo_net.reset(ref_data=ref_pos[:, :, episode])
        done = False
        # Initialize total reward for this episode
        total_reward = 0
        for time_step in count():
            action = td3_agent.choose_action(state).numpy()
            # Take action
            next_state, reward, done, _ = mimo_net.step(action)
            total_reward += reward

            state = next_state
            if done:
                episode_total_reward.append(total_reward)
                episode_durations.append(time_step)
                episode_esc_sumSE.append(mimo_net.compute_sum_se())
                episode_max_sumSE.append(mimo_net.max_sumSE)
                break

    plot_SEs(episode_esc_sumSE, ref_maxprod_sumSE, save_name="Sum_SE_diff")

    # min_SE_diff = np.abs(episode_esc_sumSE - ref_maxprod_sumSE)/ref_maxprod_sumSE
    # np.save('data/min_SE_diff.npy', min_SE_diff)
    # plot_save_cdf(min_SE_diff, save_name="actual_diff",
    #                  xlabel="Actual sum SE difference")
    # potential_diff = np.abs(episode_max_sumSE - ref_maxprod_sumSE)/ref_maxprod_sumSE
    # np.save('data/potential_diff.npy', potential_diff)
    # plot_save_cdf(potential_diff, save_name="potential_diff",
    #               xlabel="Potential sum SE difference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train mimo network continuous power allocator deep reinforcement learning model')
    parser.add_argument(
        '-t', '--threshold', help='Maximum power allocatable in mW', type=int, default=100)
    parser.add_argument(
        '-V', '--verbose', help='Display debug messages while training', action='store_true')
    parser.add_argument(
        '-v', '--version', help='Specify environment version to work with', default=0)
    parser.add_argument(
        '-c', '--config', help='Specify hyperparameters config file', type=str)
    parser.add_argument(
        '-n', '--numtest', help='Number of samples to compare between ref and TD3', type=int, default=2500)

    args = parser.parse_args()

    #Hyper paremeters
    config_file = open(f'config/{args.config}.json')
    config = json.load(config_file)
    alpha = config["alpha"]
    beta = config["beta"]
    tau = config["tau"]
    gamma = config["gamma"]
    batch_size = config["batch_size"]
    warmup = config["warmup"]
    memory_size = config["memory_size"]
    smooth_bound = config["smooth_bound"]
    smooth_std = config["smooth_std"]
    warmup_noise_std = config["warmup_noise_std"]
    noise_start = config["noise_start"]
    noise_end = config["noise_end"]
    eps_decay = config["eps_decay"]
    updt_delay = config["updt_delay"]
    actors_hsize = config["actors_hsize"]
    critic_hsize = config["critic_hsize"]
    num_episodes = config["num_episodes"]
    delta_rho = config["delta_rho"]
    if args.version == 0:
        env_ver = config["env_ver"]
    else:
        env_ver = args.version
    # End of hyperparameters

    nbr_of_BSs = 4
    nbr_of_UEs = 5

    # Create and extract environment information
    mimo_net = make('gym_cont_mimo_env:mimo-v{}'.format(env_ver),
                    L=nbr_of_BSs, K=nbr_of_UEs, M=100, ASD_deg=10,
                    delta_rho=delta_rho, val=True)

    obs_dim = 3*nbr_of_UEs*nbr_of_BSs
    action_dim = mimo_net.action_space.shape[0]
    action_bound = mimo_net.action_space.high[0]

    # Initialize agent
    td3_agent = Agent(alpha, beta, obs_dim, action_dim, action_bound,
                      memory_capacity=memory_size,
                      warmup_noise_std=warmup_noise_std,
                      noise_start=noise_start, noise_end=noise_end, eps_decay=eps_decay,
                      gamma=gamma, delay_factor=updt_delay,
                      batch_size=batch_size, tau=tau, warmup=warmup,
                      tgt_actor_noise_bound=smooth_bound,
                      tgt_actor_smooth_std=smooth_std,
                      actor_hdims=actors_hsize.copy(),
                      critic_hdims=critic_hsize.copy())

    print("Loading model from latest checkpoint")
    # Load model's weights
    td3_agent.load_models()

    validate_train_process(mimo_net, td3_agent, num_tests=args.numtest)


