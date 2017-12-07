import tensorflow.contrib.slim as slim
import tensorflow as tf
import gym
import ppaquette_gym_doom
from tqdm import tqdm
import numpy as np
import cv2

from DQN import *


# Create network architecture
class DoomNetwork(DQNNetworkDef):
    def __init__(self, height=480, width=640):
        super().__init__()
        self._height = height
        self._width = width

    def create(self, inputs, input_dim, output_dim, scope="CartPoleNetwork", reuse=False):
        img = tf.reshape(inputs, [-1, self._height, self._width, 1])
        conv1 = slim.conv2d(inputs=img, num_outputs=32, kernel_size=3, stride=1, scope=scope + "_conv1", reuse=reuse)
        conv2 = slim.conv2d(inputs=conv1, num_outputs=64, kernel_size=4, stride=2,
                            scope=scope + "_conv2", reuse=reuse)  # 120, 160 -> 60, 80
        conv3 = slim.conv2d(inputs=conv2, num_outputs=64, kernel_size=3, stride=1, scope=scope + "_conv3", reuse=reuse)
        conv4 = slim.conv2d(inputs=conv3, num_outputs=128, kernel_size=4, stride=2,
                            scope=scope + "_conv4", reuse=reuse)  # 60, 80 -> 30, 40
        conv5 = slim.conv2d(inputs=conv4, num_outputs=128, kernel_size=3, stride=1, scope=scope + "_conv5", reuse=reuse)
        conv6 = slim.conv2d(inputs=conv5, num_outputs=256, kernel_size=4, stride=2,
                            scope=scope + "_conv6", reuse=reuse)  # 30, 40-> 15, 20
        flattened = tf.reshape(conv6, [-1, 15 * 20 * 256])
        h1 = slim.fully_connected(inputs=flattened, num_outputs=500, scope=scope + "_h1", reuse=reuse)
        h2 = slim.fully_connected(inputs=h1, num_outputs=100, scope=scope + "_h2", reuse=reuse)
        h3 = slim.fully_connected(inputs=h2, num_outputs=100, scope=scope + "_h3", reuse=reuse)
        return slim.fully_connected(inputs=h3, num_outputs=output_dim, scope=scope + "_o", reuse=reuse,
                                    activation_fn=None)


def process_image(img):
    # Convert to grayscale and flatten
    grayscale = np.mean(img, 2)
    smaller = cv2.resize(grayscale, (120, 160))
    return np.reshape(smaller, [-1])


def main():
    # Doom has an 480x640x3 dimensional observation space and 43 multi discrete action space
    # However, we resize it to 1/4 the size (120, 160)
    env = gym.make('ppaquette/DoomDefendLine-v0')
    height = 120
    width = 160
    num_actions = 43

    dqn = DQN(DoomNetwork(height, width), height * width, num_actions, epsilon=0.05)
    memory = MemoryReplay(height * width, num_actions, max_saved=100000)

    for epoch in range(1000):

        # Gain experience
        total_reward = 0
        s = process_image(env.reset())
        for i in range(10000):
            a = dqn.select_action(np.reshape(s, [1, -1]))
            s_prime, r, t, _ = env.step(a.astype(np.int32).reshape([-1]))
            total_reward += r
            s_prime = process_image(s_prime)
            memory.add(s, a, r, s_prime, t)
            env.render()
            s = s_prime

            if t:
                break

        print(epoch, ": ", total_reward)

        # Train on that experience
        for i in range(100):
            dqn.train(*memory.get_batch())

        dqn.reassign_target_weights()

        if (epoch + 1) % 10 == 0:
            dqn.save(".saves/doom_" + str(epoch) + ".ckpt")


if __name__ == "__main__":
    main()
