from __future__ import print_function

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
    return cv2.resize(luminance / 255.0, (84, 84))

def multiframe_step(env, a, render=False):
    r_total = 0
    for i in range(3):
        s_prime, r, t, _ = env.step(a)
        r_total += r
        if render:
            env.render()

    s_current_prime, r, t, _ = env.step(a)
    r_total += r
    if render:
        env.render()
    current_frame = np.maximum(s_prime, s_current_prime)
    return current_frame, r_total, t, _


def main():
    env = gym.make("Boxing-v0")
    height = 84
    width = 84
    channels = 4
    num_actions = 18
    dqn = DQN(AtariNetwork(height, width, channels), height * width, num_actions, num_stacked=channels)
    memory = MemoryReplay(height * width, num_actions, max_saved=200000, num_stacked=channels)

    for epoch in tqdm(range(10000)):
        total_reward = 0
        # Gain experience
        for _ in range(1):
            s = np.zeros([84, 84, 4])
            s[:, :, 0] = preprocess(env.reset())
            for i in range(100):
                a = dqn.select_action(np.reshape(s, [1, -1]))
                current_frame, r, t, _ = multiframe_step(env, np.argmax(a))
                total_reward += r
                s_prime = np.roll(s, 1, axis=2)
                s_prime[:, :, 0] = preprocess(current_frame)
                memory.add(s.reshape([-1]), a, r-1, s_prime.reshape([-1]), t)

                if i > 4:
                    for j in range(4):
                        cv2.imshow("test s " + str(j), s[:, :, j])
                        cv2.imshow("test s' " + str(j), s_prime[:, :, j])
                    cv2.waitKey()
                    cv2.destroyAllWindows()

                s = s_prime

                if t:
                    break

        #print(epoch, ": ", total_reward)

        for i in range(25):
            dqn.train(*memory.get_batch())

        dqn.reassign_target_weights()

        print(epoch, ": ", total_reward, " Epsilon: ", dqn._epsilon)
        
        if (epoch + 1) % 100 == 0:
            s = env.reset()
            s = preprocess(s)
            s = np.array([s, s, s, s])
            for i in range(100):
                a = dqn.select_greedy_action(np.reshape(s, [1, -1]))
                s_prime, _, t, _ = multiframe_step(env, np.argmax(a), render=True)
                s = np.roll(s, 1, axis=0)
                s[0] = preprocess(s_prime)
                if t:
                    break

        if (epoch + 1) % 50 == 0:
            dqn.save(".saves/atari_boxing" + str(epoch) + ".ckpt")

    for i in range(1000):
        s = env.reset()
        s = preprocess(s)
        s = np.array([s, s, s, s])
        for i in range(100):
            a = dqn.select_greedy_action(np.reshape(s, [1, -1]))
            s_prime, _, t, _ = multiframe_step(env, np.argmax(a), render=True)
            s = np.roll(s, 1, axis=0)
            s[0] = preprocess(s_prime)
            if t:
                break


if __name__ == "__main__":
    main()
