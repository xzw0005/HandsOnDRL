'''
Created on Dec 24, 2018

@author: wangxing
'''
import gym
import time
import argparse
import numpy as np
import collections
import torch 

import wrappers 
import dqn_model

DEFAULT_ENV_NAME = 'PongNoFrameskip-v4'
FPS = 25

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='Model file to load')
    parser.add_argument('-e', '--env', default=DEFAULT_ENV_NAME)
    parser.add_argument('-r', '--record', help='Dir to store video recording')
    parser.add_argument('-nv', '--no-visualize', default=True, \
                        action='store_false', dest='visualize', help='Disable visualization of game play')
    args = parser.parse_args()
    
    env = wrappers.make_env(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
    
    s = env.reset()
    total_rewards = 0.
    cnt = collections.Counter()
    while True:
        tic = time.time()
        if args.visualize:
            env.render()
        state_v = torch.tensor(np.array([s], copy=False))
        Qs = net(state_v).data.numpy()[0]
        a = np.argmax(Qs)
        cnt[a] += 1
        sp, r, done, _ = env.step(a)
        total_rewards += r 
        if done:
            break
        s = sp 
        if args.visualize:
            delta = 1./FPS - (time.time() - tic)
            if delta > 0:
                time.sleep(delta)
    
    print("Total reward: %.2f"%total_rewards)
    print("Actions counts: ", cnt)
    if args.record:
        env.env.close()
