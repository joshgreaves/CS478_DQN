import tensorflow.contrib.slim as slim
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt

from DQN import *


def SGD_train_fn(loss, learning_rate, var_list=None):
    if var_list:
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=var_list)
    else:
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


def adam_train_fn(loss, learning_rate, var_list=None):
    if var_list:
        return tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=var_list)
    else:
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)


def rms_prop_fn(loss, learning_rate, var_list=None):
    if var_list:
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss, var_list=var_list)
    else:
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)


# Create network architecture
class CartPoleNetwork(DQNNetworkDef):
    def __init__(self):
        super(CartPoleNetwork, self).__init__()

    def create(self, inputs, input_dim, output_dim, scope="CartPoleNetwork", reuse=False):
        h1 = slim.fully_connected(inputs=inputs, num_outputs=10, scope=scope + "_h1", reuse=reuse)
        h2 = slim.fully_connected(inputs=h1, num_outputs=10, scope=scope + "_h2", reuse=reuse)
        h3 = slim.fully_connected(inputs=h2, num_outputs=10, scope=scope + "_h3", reuse=reuse)
        return slim.fully_connected(inputs=h3, num_outputs=output_dim, scope=scope + "_o", reuse=reuse,
                                    activation_fn=None)


def main():
    # Setup
    env = gym.make("CartPole-v0")
    train_funcs = [SGD_train_fn, adam_train_fn, rms_prop_fn]
    train_names = ["SGD", "Adam", "RMSProp"]
    learning_rates = [0.001, 0.01, 0.1]

    results = np.zeros([len(train_names), len(learning_rates), 500])

    # For each test and learning rate
    for k, train_method in enumerate(train_names):
        for lr_i, lr in enumerate(learning_rates):

            dqn = DQN(CartPoleNetwork(), 4, 2, train_fn=train_funcs[k], learning_rate=lr, epsilon_steps=10000)
            memory = MemoryReplay(4, 2, max_saved=10000)
            steps = 0
            for epoch in range(500):

                # Gain experience
                s = env.reset()
                for i in range(200):
                    a = dqn.select_action(np.reshape(s, [1, -1]))
                    s_prime, r, t, _ = env.step(np.argmax(a))
                    memory.add(s, a, 1 * (-1 * t), s_prime, t)
                    s = s_prime

                    # Train on that experience
                    dqn.train(*memory.get_batch())
                    steps += 1
                    if steps % 25 == 0:
                        dqn.reassign_target_weights()

                    if t:
                        break

                s = env.reset()
                greedy_success = 0
                for i in range(200):
                    a = dqn.select_greedy_action(np.reshape(s, [1, -1]))
                    s, _, t, _ = env.step(np.argmax(a))
                    greedy_success += 1
                    if t:
                        break

                results[k, lr_i, epoch] = greedy_success
                print(train_method, lr, ":", epoch, ", ", greedy_success, dqn._epsilon)

            dqn.close()

    for k, train_method in enumerate(train_names):
        for lr_i, lr in enumerate(learning_rates):
            plt.title("CartPole Time Up")
            plt.xlabel("Epoch")
            plt.ylabel("Time Up in Frames (Max 200)")
            plt.plot(results[k, lr_i, :], label=train_method + " " + str(lr))
    plt.show()


if __name__ == "__main__":
    main()
