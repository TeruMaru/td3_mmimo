import datetime
import sys
import argparse
import gym
import json
import tensorflow as tf
import numpy as np

from itertools import count
from tensorflow import keras
from td3_agent import Agent
from utils import plot_learning_curve, visualize_eps_length
from validate_maxmin import validate_train_process
from gym.envs.registration import make


np.random.seed(2022)
tf.random.set_seed(2022)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train mimo network continuous power allocator deep reinforcement learning model')
    parser.add_argument(
        '-m', '--mode', help='Case-insensitive training mode, either "scratch" or "continue"')
    parser.add_argument(
        '-t', '--threshold', help='Maximum power allocatable in mW', type=int, default=100)
    parser.add_argument(
        '-V', '--verbose', help='Display debug messages while training', action='store_true')
    parser.add_argument(
        '-v', '--version', help='Specify environment version to work with', default=0)
    parser.add_argument(
        '-c', '--config', help='Specify hyperparameters config file', type=str)

    args = parser.parse_args()
    assert args.mode != None, "Please define training mode. Available modes are:\n\t+ Scratch: Train from scratch\n\t+ Continue: Continue to train from previous saved models"

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
    # Must use env version 5 and above
    if env_ver < 6:
        env_ver = 6
    # End of hyperparameters

    nbr_of_BSs = 4
    nbr_of_UEs = 5

    save_weights = 100
    # train_log = train_logger(filepath='logs/mimo_maxmin_trainer.txt')

    # Create and extract environment information
    mimo_net = make('gym_cont_mimo_env:mimo-v{}'.format(env_ver),
                    L=nbr_of_BSs, K=nbr_of_UEs, M=100, ASD_deg=10,
                    delta_rho=delta_rho)

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

    start_eps = 0
    episode_total_reward = []
    episode_durations = []

    # evaluate every 1000 episodes
    eval_interval = 1000

    if args.mode.lower() == "continue":
        # train_log.logger.info("Continue from latest checkpoint")
        # Load models
        td3_agent.load_models()
        # Load starting episode
        start_eps = np.load('data/elapsed_episodes.npy') + 1
        # Load episodes total reward
        episode_total_reward = \
                      np.load('data/elapsed_episodes_rewards.npy').tolist()
        # Load episodes durations
        episode_durations = \
                    np.load('data/elapsed_episodes_durations.npy').tolist()

    np.set_printoptions(threshold=sys.maxsize, suppress=True)

    for episode in range(start_eps, num_episodes):
        # train_log.episode_entrance(episode+1)
        '''
        Reset environment
        '''
        state = mimo_net.reset()  # This is a 1-D numpy array.
        print(f'Episode: {episode+1} - Initial power:\n{mimo_net.rho}\n')
        # train_log.logger.info("Environment is reset to a new state!")
        done = False
        # Initialize total reward for this episode
        total_reward = 0
        # Initialize timestep per epsiode
        time_step = 0
        while not done:
            time_step += 1
            # train_log.step_entrance(timestep+1)
            action = td3_agent.choose_action(state).numpy()
            # print(f'Selected action at time step {time_step}:\n{action}')
            # Take action
            next_state, reward, done, _ = mimo_net.step(action)
            td3_agent.memorize(state, action, reward, next_state, done)
            td3_agent.learn()
            total_reward += reward
            state = next_state

        episode_total_reward.append(total_reward)
        episode_durations.append(time_step)

        avg_score = np.mean(episode_total_reward[-100:])
        # train_log.maxmin_episode_exit(
        #     episode+1, time_step+1, mimo_net.min_SE, mimo_net.max_min_SE)


        if (episode+1) % save_weights == 0:
            # Save models
            td3_agent.save_models()
            # Save elasped episodes
            np.save('data/elapsed_episodes.npy', episode)
            # Save episodes' total rewards
            np.save('data/elapsed_episodes_rewards.npy', episode_total_reward)
            # Save episodes' durations
            np.save('data/elapsed_episodes_durations.npy', episode_durations)
            
            x = [i+1 for i in range(episode+1)]
            plot_learning_curve(x, episode_total_reward, "data/td3_mimo.png")
            # visualize_loss(cummulative_loss)
            visualize_eps_length(x, episode_durations, "data/eps_length.png")

        print((f'Score: {total_reward:.5f}'
               f' - Steps taken: {time_step}'
               f' - Cummulative agent steps: {td3_agent.agent_step}'
               f' - Last 100 eps avg score: {avg_score:.2f}'
               f' - Start sum SE: {mimo_net.start_sum_SE:.3f}'
               f' - Max sum SE: {mimo_net.max_sumSE:.3f}'
               f' - Max at step: {mimo_net.peek_sumse_step}'
               f' - Stop sum SE: {mimo_net.sum_SE:.3f}'
               f' - Stopped power:\n{mimo_net.rho}')
              )

        if (episode+1) % eval_interval == 0:
            validate_train_process(mimo_net, td3_agent, num_tests=2500)

    # x = [i+1 for i in range(num_episodes)]
    # plot_learning_curve(x, episode_total_reward, "data/td3_mimo.png")
    # # visualize_loss(cummulative_loss)
    # visualize_eps_length(x, episode_durations, "data/eps_length.png")
    # train_log.logger.info("Agent took {} steps, with last exploration rate = {}".format(
    #     agent.current_step, agent.strategy.get_exploration_rate(agent.current_step)))
