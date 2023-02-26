import numpy as np
import tensorflow as tf

from tensorflow import keras
from pathlib import Path
from tensorflow.keras.losses import mean_squared_error
from replay_memory import MemoryBuffer
from actor_critic import Actor, Critic

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
            np.exp(-1. * current_step * self.decay)

class Agent(object):
    def __init__(self, alpha, beta, obs_dim, action_dim, action_bound,
                 memory_capacity=int(1e6), noise_start=5,
                 noise_end = 0.0001, eps_decay=1e-6, gamma=0.99,
                 delay_factor=2, batch_size=64, tau=0.05, warmup=1000,
                 warmup_noise_std = 5, tgt_actor_noise_bound=0.5,
                 tgt_actor_smooth_std=0.2,actor_hdims=[], critic_hdims=[]):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.d = delay_factor
        self.warmup = warmup
        self.warmup_noise_std = warmup_noise_std
        self.strategy = EpsilonGreedyStrategy(noise_start,
                                              noise_end,
                                              eps_decay)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.tgt_noise_bound = tgt_actor_noise_bound
        self.tgt_noise_std = tgt_actor_smooth_std

        self.learn_step_cntr = 0
        self.agent_step = 0

        self.memory = MemoryBuffer(memory_capacity, obs_dim, action_dim)

        self.actor = Actor(alpha, obs_dim, action_dim, action_bound,
                           name='real_actor', hidden_dims=actor_hdims.copy())

        self.target_actor = Actor(alpha, obs_dim, action_dim, action_bound,
                                  name='target_actor', hidden_dims=actor_hdims.copy())

        self.critic_1 = Critic(beta, obs_dim, action_dim, name='real_critic_1',
                               hidden_dims=critic_hdims.copy())

        self.target_critic_1 = Critic(beta, obs_dim, action_dim,
                                      name='target_critic_1',
                                      hidden_dims=critic_hdims.copy())

        self.critic_2 = Critic(beta, obs_dim, action_dim, name='real_critic_2',
                               hidden_dims=critic_hdims.copy())

        self.target_critic_2 = Critic(beta, obs_dim, action_dim,
                                      name='target_critic_2',
                                      hidden_dims=critic_hdims.copy())

        self.actor.model.compile(optimizer=self.actor.optimizer)
        self.target_actor.model.compile(optimizer=self.target_actor.optimizer)

        self.critic_1.model.compile(optimizer=self.critic_1.optimizer)
        self.target_critic_1.model.compile(
            optimizer=self.target_critic_1.optimizer)

        self.critic_2.model.compile(optimizer=self.critic_2.optimizer)
        self.target_critic_2.model.compile(
            optimizer=self.target_critic_2.optimizer)

        self.update_target_networks(tau=1)

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        self.update_target(self.target_actor.model.variables,
                           self.actor.model.variables,
                           tau)

        self.update_target(self.target_critic_1.model.variables,
                           self.critic_1.model.variables,
                           tau)

        self.update_target(self.target_critic_2.model.variables,
                           self.critic_2.model.variables,
                           tau)

    # @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def choose_action(self, state):
        if self.agent_step < self.warmup:
            mu = np.random.normal(scale=self.warmup_noise_std,
                                  size=(self.action_dim,))
        else:
            if not tf.is_tensor(state):
                state = tf.convert_to_tensor(state, dtype=tf.float32)

            if state.shape is not (1, self.obs_dim):
                state = tf.expand_dims(state, axis=0)
            mu = self.actor.predict(state)[0]
            noise_std = self.strategy.get_exploration_rate(self.agent_step - self.warmup)
            mu += np.random.normal(scale=noise_std,size=(self.action_dim,))

        # epsilon = self.strategy.get_exploration_rate(
        #   self.agent_step - self.warmup)
        # referee = np.random.random()
        # if referee < epsilon: #Exploring
        #     mu_prime = mu + np.random.normal(scale=self.noise_std,
        #                                     size=(self.action_dim,))
        # else: #Exploting
        #     mu_prime = mu
        mu_prime = tf.clip_by_value(mu,
                                    -self.action_bound, self.action_bound)

        self.agent_step += 1
        return mu_prime

    def memorize(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    #@tf.function
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        epsilon = tf.clip_by_value(np.random.normal(scale=self.tgt_noise_std),
                                   -self.tgt_noise_bound, self.tgt_noise_bound)
        with tf.GradientTape(persistent=True) as critic_tape:
            # Sample mini-batch of N transitions (s, a, r, s′) from replay memory
            states, next_states, actions, rewards, terminals = \
                self.memory.sample_memory(self.batch_size)

            # Compute tilde_a in the algorithm
            tilde_a = self.target_actor.predict(next_states, in_train=True) + epsilon
            tilde_a = tf.clip_by_value(tilde_a,
                                       -self.action_bound, self.action_bound)
            # Select which target critic to use in bellman operator
            tgt_critic_val_1 = self.target_critic_1.predict(next_states,
                                                            tilde_a,
                                                            in_train=True)

            tgt_critic_val_2 = self.target_critic_1.predict(next_states,
                                                            tilde_a,
                                                            in_train=True)
            # shape is [batch_size, 1], want to collapse to (batch_size,)
            tgt_critic_val_1 = tf.squeeze(tgt_critic_val_1, axis=1)
            tgt_critic_val_2 = tf.squeeze(tgt_critic_val_2, axis=1)

            tgt_critic_val = tf.math.minimum(
                tgt_critic_val_1, tgt_critic_val_2)

            # Execute Bellman operation, which means calculate y
            # in the paper's algorithm
            y = rewards + self.gamma * tgt_critic_val * terminals

            # Compute Q values given by real critics
            critic_val_1 = tf.squeeze(self.critic_1.predict(states, actions),
                                      axis=1)
            critic_val_2 = tf.squeeze(self.critic_2.predict(states, actions),
                                      axis=1)
            # Compute MSE losses
            loss_1 = mean_squared_error(y, critic_val_1)
            loss_2 = mean_squared_error(y, critic_val_2)

        print(f"Critic 1's loss: {loss_1}")
        print(f"Critic 2's loss: {loss_2}")
        
        critic_1_grads = critic_tape.gradient(loss_1,
                                       self.critic_1.model.trainable_variables)

        critic_2_grads = critic_tape.gradient(loss_2,
                                       self.critic_2.model.trainable_variables)

        self.critic_1.model.optimizer.apply_gradients(
            zip(critic_1_grads,  self.critic_1.model.trainable_variables)
        )

        self.critic_2.model.optimizer.apply_gradients(
            zip(critic_2_grads,  self.critic_2.model.trainable_variables)
        )

        self.learn_step_cntr += 1
        # Update policy network every d iterations
        if (self.learn_step_cntr % self.d) != 0:
            return

        with tf.GradientTape() as actor_tape:
            predicted_actions = self.actor.predict(states, in_train=True)
            # Not clipping here so that the network can figure out the bounds
            # by itself
            critic1_vals = self.critic_1.predict(states, predicted_actions)
            loss = -tf.math.reduce_mean(critic1_vals)
        print(f"Actor gain: {loss}")
        actor_grads = actor_tape.gradient(loss,
                                    self.actor.model.trainable_variables)
        self.actor.model.optimizer.apply_gradients(
            zip(actor_grads, self.actor.model.trainable_variables)
        )
        self.update_target_networks()

    def save_models(self):
        # Create parent dirs
        Path('models').mkdir(exist_ok=True)
        Path('data').mkdir(exist_ok=True)
        # Save Neural Nets
        print('... saving models ...')
        self.actor.model.save(self.actor.checkpoint_dir)
        self.critic_1.model.save(self.critic_1.checkpoint_dir)
        self.critic_2.model.save(self.critic_2.checkpoint_dir)
        self.target_actor.model.save(self.target_actor.checkpoint_dir)
        self.target_critic_1.model.save(self.target_critic_1.checkpoint_dir)
        self.target_critic_2.model.save(self.target_critic_2.checkpoint_dir)

        # Save memory buffer
        self.memory.save_memory()

        # Save counters
        cntr_arr = np.array([self.agent_step, self.learn_step_cntr,
                             self.memory.mem_cntr])
        np.save("data/cntr_arr.npy",cntr_arr)

    def load_models(self):
        print('... loading models ...')
        # Load Neural nets
        self.actor.model = keras.models.load_model(
            self.actor.checkpoint_dir)
        self.critic_1.model = keras.models.load_model(
            self.critic_1.checkpoint_dir)
        self.critic_2.model = keras.models.load_model(
            self.critic_2.checkpoint_dir)
        self.target_actor.model = keras.models.load_model(
            self.target_actor.checkpoint_dir)
        self.target_critic_1.model = keras.models.load_model(
            self.target_critic_1.checkpoint_dir)
        self.target_critic_2.model = keras.models.load_model(
            self.target_critic_2.checkpoint_dir)

        # Load memory buffer
        self.memory.load_memory()

        # Load agent's step counter
        self.agent_step, self.learn_step_cntr, self.memory.mem_cntr =\
                     np.load("data/cntr_arr.npy")
