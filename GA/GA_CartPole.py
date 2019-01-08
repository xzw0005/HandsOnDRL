'''
Created on Jan 3, 2019

@author: wangxing
'''
import gym
import numpy as np
import copy

import torch 
import torch.nn as nn

from tensorboardX import SummaryWriter 

ENV_NAME = 'CartPole-v0'
NOISE_STD = 0.01
POPULATION_SIZE = 50
PARENTS_COUNT = 10

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
    s = env.reset()
    tot_rew = 0.
    while True:
#         s_v = torch.FloatTensor([s])
        s_v = torch.tensor(np.array([s]).astype(np.float32))
        a_probs = net(s_v)
        a_v = a_probs.max(dim=1)[1]
        a = a_v.data.numpy()[0]
        sp, r, done, _ = env.step(a)
        tot_rew += r 
        if done:
            break
        s = sp
    return tot_rew

def mutate(net):
    child = copy.deepcopy(net)
    for p in child.parameters():
        noise_np = np.random.normal(size=p.data.size()).astype(np.float32)
        noise_v = torch.tensor(noise_np)
        p.data += NOISE_STD * noise_v 
    return child

if __name__=='__main__':
    writer = SummaryWriter(comment='CartPole-GA')
    env = gym.make(ENV_NAME)
    itr = 0
    nets = [Net(env.observation_space.shape[0], env.action_space.n) for _ in range(POPULATION_SIZE)]
    population = [(net, evaluate(env, net)) for net in nets]
    while True:
        population.sort(key=lambda p: p[1], reverse=True)
        fitnesses = [p[1] for p in population[:PARENTS_COUNT]]
        
        fit_mean = np.mean(fitnesses)
        fit_std = np.std(fitnesses)
        fit_max = np.max(fitnesses)
        writer.add_scalar('fitness_mean', fit_mean, itr)
        writer.add_scalar('fitness_std', fit_std, itr)
        writer.add_scalar('fitness_max', fit_max, itr)
        print('%d: fitness_mean=%.2f, fitness_std=%.2f, fitness_max=%.2f'%(itr, fit_mean, fit_std, fit_max))
        if fit_mean > 199:
            print('Solved in %d generations'%itr)
            break 
        
        # Next generation
        prev_population = population 
        population = [population[0]]
        for _ in range(POPULATION_SIZE-1):
            parent_idx = np.random.randint(0, PARENTS_COUNT)
            parent_net = prev_population[parent_idx][0]
            child = mutate(parent_net)
            population.append((child, evaluate(env, child)))
        itr += 1
    pass