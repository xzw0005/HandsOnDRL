import argparse
import os
import gym
import numpy as np
import random

import tensorflow as tf

from model import *

def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    parser.add_argument('--env', type=str, default='Seaquest')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--replay-buffer-size', type=int, default=int(1e6))
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-steps', type=int, default=int(4e7))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning-freq', type=int, default=4)
    parser.add_argument('--target-update-freq', type=int, default=10000)

    boolean_flag(parser, 'double-q', default=True)
    boolean_flag(parser, 'dueling', default=False)
    boolean_flag(parser, 'bootstrap', default=True)
    boolean_flag(parser, 'prioritized', default=False)
    parser.add_argument('--prioritized-alpha', type=float, default=.6)
    parser.add_argument('--prioritized-beta0', type=float, default=.4)
    parser.add_argument('--prioritized-eps', type=float, default=1e-6)

    parser.add_argument('--save_dir', type=str, default='./logs')
    parser.add_argument('--save_freq', type=int, default=1e6)
    return parser.parse_args()

def make_env(env_name):
    env = gym.make(env_name + 'NoFrameskip-v4')
    monitored_env = SimpleMonitor(env)
    env = wrap_dqn(monitored_env)
    return env, monitored_env

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

if __name__=='__main__':
    args = parse_args()
    save_dir = args.save_dir + '_' + args.env
    env, monitored_env = make_env(args.env)
    set_global_seed(seed)









