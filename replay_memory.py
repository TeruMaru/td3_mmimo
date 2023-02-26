import tensorflow as tf
import numpy as np

class MemoryBuffer(object):
    def __init__(self, capacity, obs_dim, action_dim):
        assert isinstance(obs_dim, int), ("Observation dimension is not given"
                                        " as an integer")
        assert isinstance(action_dim, int), ("Action dimension is not given "
                                        "as an integer")
        self.capacity = capacity
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.capacity, obs_dim))
        self.next_state_memory = np.zeros((self.capacity, obs_dim))
        self.reward_memory = np.zeros(self.capacity)
        self.terminal_memory = np.zeros(self.capacity, dtype=np.float32)
        self.action_memory = np.zeros((self.capacity, action_dim))

    def store_transition(self, state, action, reward, next_state, done):
        assert isinstance(done,bool), "Done is not a boolean variable"
        store_index = self.mem_cntr % self.capacity # wrap-around mechanism
        self.state_memory[store_index] = state
        self.next_state_memory[store_index] = next_state
        self.action_memory[store_index] = action
        self.reward_memory[store_index] = reward
        self.terminal_memory[store_index] = 1.0 - float(done)
        self.mem_cntr += 1
        
    def sample_memory(self, batch_size):
        sample_range = min(self.mem_cntr, self.capacity)
        sample_indices = np.random.choice(sample_range, size=batch_size)
        states = tf.convert_to_tensor(self.state_memory[sample_indices],
                                    dtype=tf.float32)
        next_states = tf.convert_to_tensor(
                        self.next_state_memory[sample_indices],
                        dtype=tf.float32)
        actions = tf.convert_to_tensor(self.action_memory[sample_indices],
                                        dtype = tf.float32)
        rewards = tf.convert_to_tensor(self.reward_memory[sample_indices],
                                        dtype = tf.float32)
        termnials = tf.convert_to_tensor(self.terminal_memory[sample_indices],
                                        dtype = tf.float32)

        return states, next_states, actions, rewards, termnials

    def save_memory(self, path='data'):
        np.save(f'{path}/state_memory.npy', self.state_memory)
        np.save(f'{path}/next_state_memory.npy', self.next_state_memory)
        np.save(f'{path}/action_memory.npy', self.action_memory)
        np.save(f'{path}/reward_memory.npy', self.reward_memory)
        np.save(f'{path}/terminal_memory.npy', self.terminal_memory)

    def load_memory(self, path='data'):
        self.state_memory = np.load(f'{path}/state_memory.npy')
        self.next_state_memory = np.load(f'{path}/next_state_memory.npy')
        self.action_memory = np.load(f'{path}/action_memory.npy')
        self.reward_memory = np.load(f'{path}/reward_memory.npy')
        self.terminal_memory = np.load(f'{path}/terminal_memory.npy')


