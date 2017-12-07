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
        return slim.fully_connected(inputs=h1, num_outputs=output_dim, scope=scope + "_o", reuse=reuse,
                                    activation_fn=None)


def main():
    env = gym.make("CartPole-v0")
    # CartPole has an 4 dimensional observation space and 2 dimensional action space
    dqn = DQN(CartPoleNetwork(), 4, 2)
    memory = MemoryReplay(4, 2)

    for _ in tqdm(range(1000)):

        # Gain experience
        s = env.reset()
        for i in range(100):
            a = dqn.select_action(np.reshape(s, [1, -1]))
            s_prime, r, t, _ = env.step(a)
            if t:
                r = -1.0
            memory.add(s, a, r, s_prime, t)
            s = s_prime

            if t:
                break

        # Train on that experience
        for i in range(100):
            dqn.train(*memory.get_batch())

        s = env.reset()
        for i in range(100):
            a = dqn.select_greedy_action(np.reshape(s, [1, -1]))
            env.render()
            s, _, t, _ = env.step(a)
            if t:
                break


if __name__ == "__main__":
    main()
