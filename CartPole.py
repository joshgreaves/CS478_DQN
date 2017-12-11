import tensorflow.contrib.slim as slim
import gym
from tqdm import tqdm

from DQN import *


def cartpole_train_fn(loss, learning_rate, var_list=None):
    if var_list:
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=var_list)
    else:
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


# Create network architecture
class CartPoleNetwork(DQNNetworkDef):
    def __init__(self):
        super().__init__()

    def create(self, inputs, input_dim, output_dim, scope="CartPoleNetwork", reuse=False):
        h1 = slim.fully_connected(inputs=inputs, num_outputs=10, scope=scope + "_h1", reuse=reuse)
        h2 = slim.fully_connected(inputs=h1, num_outputs=10, scope=scope + "_h2", reuse=reuse)
        h3 = slim.fully_connected(inputs=h2, num_outputs=10, scope=scope + "_h3", reuse=reuse)
        return slim.fully_connected(inputs=h1, num_outputs=output_dim, scope=scope + "_o", reuse=reuse,
                                    activation_fn=None)


def main():
    env = gym.make("CartPole-v0")
    # CartPole has an 4 dimensional observation space and 2 dimensional action space
    dqn = DQN(CartPoleNetwork(), 4, 2, learning_rate=0.01, train_fn=cartpole_train_fn, epsilon_max=1.0, epsilon_min=0.1,
              epsilon_steps=10000)
    memory = MemoryReplay(4, 2, max_saved=10000)
    steps = 0

    for epoch in range(1000):

        # Gain experience
        s = env.reset()
        for i in range(200):
            a = dqn.select_action(np.reshape(s, [1, -1]))
            s_prime, r, t, _ = env.step(np.argmax(a))
            memory.add(s, a, -1 * t, s_prime, t)
            s = s_prime

            dqn.train(*memory.get_batch())
            steps = (steps + 1) % 25
            if steps == 0:
                dqn.reassign_target_weights()

            if t:
                break

        s = env.reset()
        num_steps = 0
        for i in range(200):
            a = dqn.select_greedy_action(np.reshape(s, [1, -1]))
            env.render()
            s, _, t, _ = env.step(np.argmax(a))
            num_steps += 1
            if t:
                break
        print(num_steps, ", ", dqn._epsilon)


if __name__ == "__main__":
    main()