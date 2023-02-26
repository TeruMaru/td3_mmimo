import gym
import numpy as np
from td3_agent import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    np.random.seed(1207)
    #env = gym.make('LunarLanderContinuous-v2')
    env = gym.make('Pendulum-v0')
    # env = gym.make('BipedalWalker-v3')
    agent = Agent(0.001, 0.001,
            env.observation_space.shape[0], env.action_space.shape[0],
            env.action_space.high[0], eps_start=1, eps_end=1, eps_decay=1,
            tau=0.005, noise_std=0.1, warmup_noise_std=0.1,
            batch_size=100, actor_hdims=[400,300], critic_hdims=[400,300]
            )
    n_games = 1000
    # filename = 'plots/' + 'walker_' + str(n_games) + '_games.png'

    best_score = env.reward_range[0]
    score_history = []

    #agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        time_step = 0
        while not done:
            time_step += 1
            action = agent.choose_action(observation)
            print(f'Selected action at time step {time_step}:\n{action}')
            observation_, reward, done, info = env.step(action)
            agent.memorize(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        # if avg_score > best_score:
        #     best_score = avg_score
        #     agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)

    x = [i+1 for i in range(n_games)]
    # plot_learning_curve(x, score_history, filename)
