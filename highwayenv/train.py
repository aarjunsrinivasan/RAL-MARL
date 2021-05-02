import gym
from gym import wrappers
import sys
from tqdm.notebook import trange
from IPython import display
from matplotlib import pyplot as plt

import highway_env


def train():
    env = gym.make('highway-adv-v0')
    obs = env.reset()

    NUM_ITERATIONS = 1000

    for i in range(NUM_ITERATIONS):
        action = env.action_type.actions_indexes[ACTIONS_ALL[(i+1)%5]]
        obs, reward, done, info = env.step(action)
        if done: break
        #env.render()

    env.close()
    #plt.imshow(env.render(mode="rgb_array"))
    #plt.show()


if __name__ == "__main__":

    # args parser if any
    train()
