'''
Created on Jan 4, 2019

@author: wangxing
'''
import gym
import numpy as np
import collections 
import argparse
import time 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter 

ENV_NAME = 'HalfCheetah-v2'
NOISE_STD = 0.05
LEARNING_RATE = 0.01


class Net(object):
    def __init__(self, obs_size, act_size, hid_size=64):
        super(Net, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(obs_size, hid_size),
                nn.Tanh(),
                nn.Linear(hid_size, hid_size),
                nn.Tanh(),
                nn.Linear(hid_size, act_size),
                nn.Tanh()
            )
    def forward(self, x):
        return self.net(x)
    
def make_env():
    return gym.make(ENV_NAME)        
        
def evaluate(env, net, device='cpu'):
    