import gym
import numpy as np

env_name = "DoubleDunk-v0"

env = gym.make(env_name)
# print(env.observation_space)
# print(env.action_space)
# print(env.action_space.sample())

def generate_random_batch(batch_size):
    L=[]
    obs=env.reset()
    for i in range(batch_size):
        L.append(env.step(env.action_space.sample()))
    return L

# print(env.step(env.action_space.sample()))
