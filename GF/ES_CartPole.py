'''
Created on Jan 1, 2019

@author: wangxing
'''
import gym
import time
import numpy as np

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter 

ENV_NAME = 'CartPole-v0'
MAX_BATCH_EPISODES = 100
MAX_BATCH_STEPS = 10000
NOISE_STD = 0.01
LEARNING_RATE = 0.001

class Net(nn.Module):
    def __init__(self, obs_size, act_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(obs_size, 32), 
                nn.ReLU(),
                nn.Linear(32, act_size),
                nn.Softmax(dim=1)
            )
        
    def forward(self, x):
        return self.net(x)
    
def evaluate(env, net):
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    while True:
        obs_v = torch.FloatTensor([obs])
        act_prob = net(obs_v)
        act = act_prob.max(dim=1)[1]
        sp, r, done, _ = env.step(act.data.numpy()[0])
        total_reward += r 
        steps += 1
        if done:
            break
        obs = sp
    return total_reward, steps

def sample_noise(net):
    """ Mirrored Sampling to improve stability of convergence"""
    pos, neg = [], []
    for p in net.parameters():
        noise = np.random.normal(size=p.data.size()).astype(np.float32)
        noise_v = torch.from_numpy(noise)
        pos.append(noise_v)
        neg.append(-noise_v)
    return pos, neg

def eval_with_noise(env, net, noise):
    old_params = net.state_dict()
    for p, pn in zip(net.parameters(), noise):
        p.data += NOISE_STD * pn
    rets, steps = evaluate(env, net)
    net.load_state_dict(old_params)
    return rets, steps

def train(net, batch_noise, batch_fitness, writer, itr):
#     normalize fitness with zero mean and unit variance
    batch_fitness = np.array(batch_fitness)
    batch_fitness -= np.mean(batch_fitness)
    batch_sigma = np.std(batch_fitness)
    if batch_sigma > 1e-6:
        batch_fitness /= batch_sigma
     
    weighted_noise = None
    for noise, fitness in zip(batch_noise, batch_fitness):
        if weighted_noise is None:
            weighted_noise = [fitness * pn for pn in noise]
        else:
            for wn, pn in zip(weighted_noise, noise):
                wn += fitness * pn
    batch_updates = []
    for p, dp in zip(net.parameters(), weighted_noise):
        update = dp / (len(batch_fitness) * NOISE_STD)
        p.data += LEARNING_RATE * update 
        batch_updates.append(torch.norm(update))
    writer.add_scalar('update_L2', np.mean(batch_updates), itr)

if __name__=='__main__':
    writer = SummaryWriter(comment='-ES-'+ENV_NAME)
    env = gym.make(ENV_NAME)
    
    net = Net(env.observation_space.shape[0], env.action_space.n)
    print(net)
    
    itr = 0 
    while True:
        tic = time.time()
        batch_noise, batch_fitness = [],  []
        batch_steps = 0
        for _ in range(MAX_BATCH_EPISODES):
            pos_noise, neg_noise = sample_noise(net)
#             batch_noise.append(pos_noise)
#             batch_noise.append(neg_noise)
#             fitness, steps = eval_with_noise(env, net, pos_noise)
#             batch_fitness.append(fitness)
#             batch_steps += steps
#             fitness, steps = eval_with_noise(env, net, neg_noise)
#             batch_fitness.append(fitness)
#             batch_steps += steps
            for noise in (pos_noise, neg_noise):
                batch_noise.append(noise)
                fitness, steps = eval_with_noise(env, net, noise)
                batch_fitness.append(fitness)
                batch_steps += steps 
            if batch_steps > MAX_BATCH_STEPS:
                break
        itr += 1
        m_rew = np.mean(batch_fitness)
        if m_rew > 199:
            print('Solved in %d steps'%itr)
            break
        
        train(net, batch_noise, batch_fitness, writer, itr)
        writer.add_scalar('rewards_mean', m_rew, itr)
        writer.add_scalar('rewards_std', np.std(batch_fitness), itr)
        writer.add_scalar('reward_max', np.max(batch_fitness), itr)
        writer.add_scalar('batch_episodes', len(batch_fitness), itr)
        writer.add_scalar('batch_steps', batch_steps, itr)
        speed = batch_steps / (time.time() -tic)
        writer.add_scalar('speed', speed, itr)
        print("%d: reward=%.2f, speed=%.2f f/s"%(itr, m_rew, speed))
        