import tensorflow as tf
import tensorflow.contrib.slim as slim

from DQN import DQN
from DQN import DQNNetworkDef


# Create network architecture
class CartPoleNetwork(DQNNetworkDef):
    def __init__(self):
        super().__init__()

    def create(self, inputs, input_dim, output_dim, scope="CartPoleNetwork", reuse=False):
        h1 = slim.fully_connected(inputs=inputs, num_outputs=10, scope=scope + "_h", reuse=reuse)
        return slim.fully_connected(inputs=h1, num_outputs=output_dim, scope=scope + "_o", reuse=reuse,
                                    activation_fn=None)


# Here we should pass in some function which creates our network
dqn = DQN()

# Then we create a train loop where we alternate gathering experience and calling dqn.train
