import numpy as np
import tensorflow as tf
import random


def default_train_fn(loss, learning_rate, var_list=None):
    if var_list:
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss, var_list=var_list)
    else:
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)


class DQN(object):
    def __init__(self, network_definition, state_dim, action_dim, gamma=0.99, epsilon_max=1.0, epsilon_min=0.1,
                 epsilon_steps=1000000, learning_rate=0.001, num_stacked=1, train_fn=default_train_fn):
        # Cache important info
        self._network = network_definition
        self._state_dim = state_dim * num_stacked
        self._action_dim = action_dim
        self._gamma = gamma
        self._epsilon = epsilon_max
        self._epsilon_max = epsilon_max
        self._epsilon_min = epsilon_min
        self._epsilon_steps = epsilon_steps
        self._epsilon_current_steps = 0.0
        self._learning_rate = learning_rate

        # Create networks
        self._state = tf.placeholder(tf.float32, [None, self._state_dim])
        self._action = tf.placeholder(tf.float32, [None, self._action_dim])  # One hot
        self._reward = tf.placeholder(tf.float32, [None, 1])
        self._state_prime = tf.placeholder(tf.float32, [None, self._state_dim])
        self._terminal = tf.placeholder(tf.float32, [None, 1])

        # Learning net is updated every step, target net is updated every self._copy_weights_interval steps
        self._learning_net = self._network.create(self._state, self._state_dim, self._action_dim, scope="LearningNet",
                                                  reuse=False)
        self._target_net = self._network.create(self._state_prime, self._state_dim, self._action_dim,
                                                scope="TargetNet", reuse=False)

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self._learning_vars = [var for var in var_list if "LearningNet" in var.name]
        self._target_vars = [var for var in var_list if "TargetNet" in var.name]
        self._replace_ops = []

        # For replacing target_vars with learning_vars
        for i, x in enumerate(self._target_vars):
            self._replace_ops.append(self._target_vars[i].assign((self._learning_vars[i])))

        # Loss and optimization
        self._predicted_return = self._reward + (1 - self._terminal) * self._gamma * tf.reduce_max(self._target_net, 1,
                                                                                                   keep_dims=True)
        self._loss = tf.reduce_mean((self._predicted_return - tf.reduce_sum(self._action * self._learning_net, 1,
                                                                            keep_dims=True)) ** 2.0)
        self._optim = train_fn(self._loss, self._learning_rate, self._learning_vars)

        # Tensorflow init
        self._saver = tf.train.Saver()
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        self.reassign_target_weights()

    def train(self, s, a, r, s_prime, t):
        self._sess.run(self._optim, feed_dict={self._state: s, self._action: a, self._state_prime: s_prime,
                                               self._reward: r, self._terminal: t})

    def reassign_target_weights(self):
        for x in self._replace_ops:
            self._sess.run(x)

    # return one hot with selected action
    def select_action(self, state, decay=True):
        q_vals = self._sess.run(self._learning_net, feed_dict={self._state: state})
        best = np.argmax(q_vals, 1)[0]
        if random.random() < self._epsilon:
            best = np.random.choice(np.array([i for i in range(self._action_dim) if i != best]))

        if decay and self._epsilon_current_steps < self._epsilon_steps:
            self._epsilon_current_steps += 1
            ratio = self._epsilon_current_steps / self._epsilon_steps
            self._epsilon = (1 - ratio) * self._epsilon_max + ratio * self._epsilon_min

        a = np.zeros([1, self._action_dim])
        a[0, best] = 1.0
        return a

    def select_greedy_action(self, state):
        q_vals = self._sess.run(self._target_net, feed_dict={self._state_prime: state})
        a = np.zeros([1, self._action_dim])
        a[0, np.argmax(q_vals, 1)[0]] = 1.0
        return a

    def episode_step(self, state):
        return self.select_action(self._target_net.forward(state))

    def save(self, path):
        self._saver.save(self._sess, path)

    def load(self, path):
        self._saver.restore(self._sess, path)


class MemoryReplay(object):
    def __init__(self, state_dim, action_dim, max_saved=10000, num_stacked=1):
        self._max_saved = max_saved
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._num_stacked = num_stacked

        self._states = np.empty([max_saved, num_stacked * state_dim])
        self._actions = np.empty([max_saved, action_dim])
        self._state_primes = np.empty([max_saved, num_stacked * state_dim])
        self._rewards = np.empty([max_saved, 1])
        self._terminals = np.empty([max_saved, 1])

        self._index = 0
        self._max_index = 0

    def add(self, s, a, r, s_prime, t):
        self._states[self._index, :] = s
        self._actions[self._index, :] = a
        self._rewards[self._index, 0] = r
        self._state_primes[self._index, :] = s_prime
        self._terminals[self._index, :] = t

        self._index = (self._index + 1) % self._max_saved
        self._max_index = max(self._index, self._max_index)

    def get_batch(self, batch_size=32):
        indices = np.random.choice(self._max_index, batch_size)
        return (self._states[indices, :], self._actions[indices, :], self._rewards[indices, :],
                self._state_primes[indices, :], self._terminals[indices, :])


class DQNNetworkDef(object):
    def __init__(self):
        pass

    def create(self, inputs, input_dim, output_dim, scope="CartPoleNetwork", reuse=False):
        raise NotImplementedError("Must be implemented by child class")
