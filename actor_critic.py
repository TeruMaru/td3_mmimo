import tensorflow as tf
import numpy as np
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

class Actor(object):
    def __init__(self, lr, obs_dim, action_dim, action_bound, name='actor',
                 hidden_dims=[], ckpt_parent_dir="models"):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.action_bound = action_bound
        self.name = name

        self.optimizer = Adam(learning_rate=lr)
        self.build_network()

        self.checkpoint_dir = f'{ckpt_parent_dir}/{self.name}'

    def build_network(self):
        tanh_init = tf.keras.initializers.GlorotUniform(seed=2022)
        actor_inputs = keras.Input(shape=(self.obs_dim,),name='actor_inputs')
        for id, dim in enumerate(self.hidden_dims):
            if id == 0:
                layer_inputs = actor_inputs
            else:
                layer_inputs = dense_layer
            dense_layer = layers.Dense(units=dim,
                                       activation='relu',
                                       name=f'{self.name}_hidden_layer_{id}')(layer_inputs)
        mu_hat = layers.Dense(units=self.action_dim,
                              activation='tanh',
                              kernel_initializer=tanh_init,
                              name=f'{self.name}_mu_hat')(dense_layer)
        mu_prime = tf.multiply(mu_hat, self.action_bound/2)
        mu = tf.add(mu_prime, self.action_bound/2)
        self.model = keras.Model(inputs=actor_inputs, outputs=mu,
                                name = self.name)


    def predict(self, state, in_train=False):
        return self.model(state, training=in_train)

class Critic(object):
    def __init__(self, lr, obs_dim, action_dim, name='critic',
                 ckpt_parent_dir="models", hidden_dims=[]):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.name = name
        self.hidden_dims = hidden_dims

        self.optimizer = Adam(learning_rate=lr)
        self.build_network()

        self.checkpoint_dir = f'{ckpt_parent_dir}/{self.name}'

    def build_network(self):
        state_inputs = keras.Input(shape=(self.obs_dim,))
        action_inputs = keras.Input(shape=(self.action_dim,))

        for id, dim in enumerate(self.hidden_dims):
            if id == 0:
                layer_inputs = tf.concat([state_inputs, action_inputs], axis=1)
            else:
                layer_inputs = dense_layer
            dense_layer = layers.Dense(units=dim,
                                       activation='relu',
                                       name=f'{self.name}_hidden_layer_{id}')(layer_inputs)

        q_hat = layers.Dense(units=1, activation=None,
                             name=f'{self.name}_q_hat')(dense_layer)

        self.model = keras.Model(inputs=[state_inputs, action_inputs],
                                outputs=q_hat,
                                name=self.name)

    def predict(self, state, action, in_train=False):
        return self.model([state,action], training=in_train)