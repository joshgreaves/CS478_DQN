import numpy as np
import tensorflow as tf
import random


class DQN(object):
    def __init__(self, network_definition, state_dim, action_dim, gamma=0.95, copy_weight_interval=1000, epsilon=0.1,
                 epsilon_decay=1.0, learning_rate=0.001):
        # Cache important info
        self._network = network_definition
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._gamma = gamma
        self._copy_weights_interval = copy_weight_interval
        self._target_update_countdown = copy_weight_interval
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
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

        # Loss and optimization
        self._predicted_return = self._reward + self._terminal * self._gamma * tf.reduce_max(self._target_net, 1)
        self._loss = tf.reduce_mean((self._predicted_return - (self._learning_net * self._action))**2.0)
        self._optim = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss, var_list=self._learning_vars)

        # Tensorflow init
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

    def train(self, s, a, r, s_prime, t):

        learning_net_q = self._learning_net.forward(s)
        pred = learning_net_q[:, a]

        target_net_q = self._target_net.forward(s_prime)
        target = r + self._gamma * np.max(target_net_q, axis=1)

        loss_vector = target - pred
        loss = np.sum(np.power(np.clip(loss_vector, -1, 1), 2))

        # learning_net.backwards(loss) Something like this.. Whatever is normal for Tensorflow in backprop

        self._target_update_countdown -= 1
        if self._target_update_countdown <= 0:
            self._target_update_countdown = self._copy_weights_interval

<<<<<<< Updated upstream
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.learning_vars = [var for var in vars if "learning" in var.name]
            self.target_vars = [var for var in vars if "target" in var.name]
            self.assign_ops = []
            for i, x in enumerate(self.target_vars):
                self.assign_ops.append(self.target_vars[i].assign((self.learning_vars[i])))

=======
>>>>>>> Stashed changes
            # replace wieghts of target_net with weights of learning_net

    # return index of selected action
    def select_action(self, state):
        q_vals = self._sess.run(self._learning_net, feed_dict={self._state: state})
        best = np.argmax(q_vals, 1)[0]
        if self._epsilon < random.random():
            return np.random.choice(np.array([i for i in range(self._action_dim) if i != best]))
        else:
            return best

    def select_greedy_action(self, state):
        q_vals = self._sess.run(self._learning_net, feed_dict={self._state: state})
        return np.argmax(q_vals, 1)[0]

    def episode_step(self, state):
        return self.select_action(self._target_net.forward(state))


class DQNNetworkDef(object):
    def __init__(self):
        pass

    def create(self, inputs, input_dim, output_dim, scope="CartPoleNetwork", reuse=False):
        raise NotImplementedError("Must be implemented by child class")
