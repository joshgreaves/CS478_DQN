import tensorflow.contrib.slim as slim
import gym
from tqdm import tqdm
import numpy as np
from time import sleep
import cv2
from DQN import *

np.set_printoptions(threshold=np.nan)

# Create network architecture
class AtariNetwork(DQNNetworkDef):
    def __init__(self, height=210, width=160, channels=1):
        super().__init__()
        self._height = height
        self._width = width
        self._channels = channels

    def create(self, inputs, input_dim, output_dim, scope="CartPoleNetwork", reuse=False):
        img = tf.reshape(inputs, [-1, self._height, self._width, self._channels])
        conv1 = slim.conv2d(inputs=img, num_outputs=32, kernel_size=8, stride=4, scope=scope + "_conv1", reuse=reuse, padding="VALID")
        conv2 = slim.conv2d(inputs=conv1, num_outputs=64, kernel_size=4, stride=2, scope=scope + "_conv2", reuse=reuse, padding="VALID")
        conv3 = slim.conv2d(inputs=conv2, num_outputs=64, kernel_size=3, stride=1, scope=scope + "_conv3", reuse=reuse)
        flattened = tf.reshape(conv3, [-1, 81 * 64])
        h1 = slim.fully_connected(inputs=flattened, num_outputs=500, scope=scope + "_h1", reuse=reuse)
        return slim.fully_connected(inputs=h1, num_outputs=output_dim, scope=scope + "_o", reuse=reuse, activation_fn=None)


def preprocess(s):
    luminance = 0.299 * s[:, :, 0] + 0.587 * s[:, :, 1] + 0.114 * s[:, :, 2]
    return cv2.resize(luminance, (84, 84))


def main():
    env = gym.make("Boxing-v0")
    height = 84
    width = 84
    channels = 4
    num_actions = 18
    dqn = DQN(AtariNetwork(height, width, channels), height * width, num_actions, epsilon=1.0, epsilon_decay=0.999, num_stacked=channels, learning_rate=0.1)
    memory = MemoryReplay(height * width, num_actions, max_saved=10000, num_stacked=channels)

    for epoch in tqdm(range(1000)):

        # Gain experience
        for _ in range(1):
            s = env.reset()
            s = preprocess(s)
            s = np.array([s, s, s, s])
            for i in range(100):
                # if epoch % 5 == 0:
                #     env.render()
                a = dqn.select_action(np.reshape(s, [1, -1]))
                s_prime, r, t, _ = env.step(np.argmax(a))
                s_prime = preprocess(s_prime)
                s_prime = np.roll(s, 1, axis=0)
                s_prime[0] = np.maximum(s_prime[1], s_prime[0])
                memory.add(s.reshape([-1]), a, r-1, s_prime.reshape([-1]), t)
                s = s_prime

                if t:
                    break

        #print(epoch, ": ", total_reward)

        # Train on that experience
        # for i in range(min((epoch + 1) * 5, 250)):
        for i in range(25):
            dqn.train(*memory.get_batch())

        dqn.reassign_target_weights()

        if (epoch + 1) % 25 == 0:
            s = env.reset()
            s = preprocess(s)
            s = np.array([s, s, s, s])
            for i in range(100):
                a = dqn.select_greedy_action(np.reshape(s, [1, -1]))
                env.render()
                s_prime, _, t, _ = env.step(np.argmax(a))
                s = np.roll(s, 1, axis=0)
                s[0] = preprocess(s_prime)
                if t:
                    break

if __name__ == "__main__":
    main()
