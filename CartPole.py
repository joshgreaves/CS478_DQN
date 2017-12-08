import tensorflow.contrib.slim as slim
import gym
from tqdm import tqdm

from DQN import *


# Create network architecture
class CartPoleNetwork(DQNNetworkDef):
    def __init__(self):
        super().__init__()

    def create(self, inputs, input_dim, output_dim, scope="CartPoleNetwork", reuse=False):
        h1 = slim.fully_connected(inputs=inputs, num_outputs=10, scope=scope + "_h1", reuse=reuse)
        # h2 = slim.fully_connected(inputs=h1, num_outputs=50, scope=scope + "_h2", reuse=reuse)
        # h3 = slim.fully_connected(inputs=h2, num_outputs=50, scope=scope + "_h3", reuse=reuse)
        return slim.fully_connected(inputs=h1, num_outputs=output_dim, scope=scope + "_o", reuse=reuse,
                                    activation_fn=None)


def main():
    env = gym.make("CartPole-v0")
    # CartPole has an 4 dimensional observation space and 2 dimensional action space
    dqn = DQN(CartPoleNetwork(), 4, 2, epsilon=0.05, gamma=0.9)
    memory = MemoryReplay(4, 2, max_saved=1000)

    for epoch in tqdm(range(1000)):

        # Gain experience
        s = env.reset()
        for i in range(100):
            a = dqn.select_action(np.reshape(s, [1, -1]))
            s_prime, r, t, _ = env.step(np.argmax(a))
            memory.add(s, a, 1 * (-1 * t), s_prime, t)
            s = s_prime

            if t:
                break

        # Train on that experience
        for i in range(20):
            dqn.train(*memory.get_batch())

        dqn.reassign_target_weights()

        s = env.reset()
        for i in range(1000):
            a = dqn.select_greedy_action(np.reshape(s, [1, -1]))
            env.render()
            s, _, t, _ = env.step(np.argmax(a))
            if t:
                break


if __name__ == "__main__":
    main()