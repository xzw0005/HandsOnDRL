'''
Created on Jan 2, 2019

@author: wangxing
'''
import gym
import pybullet_envs

ENV_ID = 'MinitaurBulletEnv-v0'
RENDER = True

spec = gym.envs.registry.spec(ENV_ID)
spec._kwargs['render'] = RENDER
env = gym.make(ENV_ID)

print(env.observation_space)
print(env.action_space)
print(env)
print(env.reset())
env.close()