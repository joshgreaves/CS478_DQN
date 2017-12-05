import tensorflow.contrib.slim as slim
import gym

from DQN import *


# Create network architecture
class CartPoleNetwork(DQNNetworkDef):
    def __init__(self):
        super().__init__()

    def create(self, inputs, input_dim, output_dim, scope="CartPoleNetwork", reuse=False):
        h1 = slim.fully_connected(inputs=inputs, num_outputs=10, scope=scope + "_h", reuse=reuse)
        return slim.fully_connected(inputs=h1, num_outputs=output_dim, scope=scope + "_o", reuse=reuse,
                                    activation_fn=None)


def main():
    # Create environment
    env = gym.make("CartPole-v0")
    # CartPole has an 4 dimensional observation space and 2 dimensional action space
    dqn = DQN(CartPoleNetwork(), 4, 2)

    for _ in range(10):
        s = env.reset()
        for i in range(100):
            s, r, t, _ = env.step(dqn.select_action(np.reshape(s, [1, -1])))
            env.render()


if __name__ == "__main__":
    main()
