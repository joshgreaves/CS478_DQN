import tensorflow.contrib.slim as slim
import tensorflow as tf
import gym
import ppaquette_gym_doom
from tqdm import tqdm
import numpy as np
import cv2

from DQN import *

# List of actions at https://github.com/ppaquette/gym-doom/blob/master/ppaquette_gym_doom/controls.md
ACTION_NAMES = ["Attack", "Use", "Reload", "Zoom", "Move Right", "Move Left", "Move Forward", "Move Backward",
                "Turn Right", "Turn Left", "Nothing"]
NUM_ACTIONS = len(ACTION_NAMES)
DOOM_ACTIONS = np.zeros([NUM_ACTIONS, 43])
DOOM_ACTIONS[0, 0] = 1.0  # Attack
DOOM_ACTIONS[1, 1] = 1.0  # Use
DOOM_ACTIONS[2, 6] = 1.0  # Reload
DOOM_ACTIONS[3, 7] = 1.0  # Zoom
DOOM_ACTIONS[4, 10] = 1.0  # Move Right
DOOM_ACTIONS[5, 11] = 1.0  # Move Left
DOOM_ACTIONS[6, 12] = 1.0  # Move Forward
DOOM_ACTIONS[7, 13] = 1.0  # Move Backward
DOOM_ACTIONS[8, 14] = 1.0  # Turn Right
DOOM_ACTIONS[9, 15] = 1.0  # Turn Left


# Create network architecture
class DoomNetwork(DQNNetworkDef):
    def __init__(self, height=480, width=640, channels=1):
        super().__init__()
        self._height = height
        self._width = width
        self._channels = channels

    def create(self, inputs, input_dim, output_dim, scope="CartPoleNetwork", reuse=False):
        img = tf.reshape(inputs, [-1, self._height, self._width, self._channels])
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
        h1 = slim.fully_connected(inputs=flattened, num_outputs=100, scope=scope + "_h1", reuse=reuse)
        h2 = slim.fully_connected(inputs=h1, num_outputs=100, scope=scope + "_h2", reuse=reuse)
        h3 = slim.fully_connected(inputs=h2, num_outputs=100, scope=scope + "_h3", reuse=reuse)
        return slim.fully_connected(inputs=h3, num_outputs=output_dim, scope=scope + "_o", reuse=reuse,
                                    activation_fn=None)


def process_image(img):
    # Convert to grayscale and flatten
    grayscale = np.mean(img, 2)
    smaller = cv2.resize(grayscale, (160, 120))
    return smaller


def main():
    # Doom has an 480x640x3 dimensional observation space and 43 multi discrete action space
    # However, we resize it to 1/4 the size (120, 160)
    env = gym.make('ppaquette/DoomBasic-v0')
    height = 120
    width = 160
    channels = 4
    num_actions = NUM_ACTIONS

    dqn = DQN(DoomNetwork(height, width, channels), height * width, num_actions, epsilon=0.05, num_stacked=channels)
    # dqn.load(".saves/doom_basic_719.ckpt")
    memory = MemoryReplay(height * width, num_actions, max_saved=10000, num_stacked=channels)

    for epoch in range(1000):

        # Gain experience
        total_reward = 0
        s = np.zeros([120, 160, 4])
        s_prime = np.zeros([120, 160, 4])
        s[:, :, 0] = process_image(env.reset())
        for i in range(10000):
            a = dqn.select_action(np.reshape(s, [1, -1]))
            action = DOOM_ACTIONS[a.reshape([-1]) == 1.0]
            s2, r, t, _ = env.step(action.reshape([-1]))
            total_reward += r
            s_prime[:, :, 1:] = s_prime[:, :, :3]
            s_prime[:, :, 0] = process_image(s_prime)
            memory.add(np.reshape(s, [-1]), a, r, np.reshape(s_prime, [-1]), t)
            env.render()
            s = s_prime

            if t:
                break

        print(epoch, ": ", total_reward)

        # Train on that experience
        # for i in range(min((epoch + 1) * 5, 250)):
        for i in range(25):
            dqn.train(*memory.get_batch())

        dqn.reassign_target_weights()

        if (epoch + 1) % 20 == 0:
            dqn.save(".saves/doom_basic_" + str(epoch) + ".ckpt")


if __name__ == "__main__":
    main()
