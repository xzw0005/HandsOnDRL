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
SIGMA = .01
COG_LR = 5
SOCIAL_LR = 5
INERTIA = .95
POPULATION_SIZE = 100

class Net(nn.Module):
    def __init__(self, obs_size, act_size):
        super(Net, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(obs_size, 32),
                nn.ReLU(),
                nn.Linear(32, act_size), 
                nn.Softmax(dim=1)
            )
#         self.names = self.state_dict().keys()
#         for key in self.names:
            
        self.velocity = {}
        for name, param in self.named_parameters():
            vel = np.random.uniform(-1, 1, size=param.size()).astype(np.float32)
            self.velocity[name] = torch.tensor(vel)
        self.pbest = None
        self.gbest = None
        self.pval = None 
        self.gval = None
        
    def forward(self, x):
        return self.model(x)

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
    if net.pval is None or net.pval <= tot_rew:
        net.pval = tot_rew
        net.pbest = net.state_dict()
    return tot_rew

def move(net):
#     for p, pb, gb in zip(net.parameters(), net.pbest, net.gbest):
    params = net.state_dict()
    pb_params = net.pbest
    gb_params = net.gbest
    vel_params = net.velocity
#     for key in net.names:
    for name, p in net.named_parameters():
        pb = pb_params[name]
        gb = gb_params[name]
        vel = vel_params[name]
#         vel = np.random.uniform(-1, 1, size=p.data.size()).astype(np.float32)
        lr_cog = np.random.uniform(size=p.data.size()).astype(np.float32) 
        lr_cog *= COG_LR
#         print(lr_cog)
        lr_social = np.random.uniform(size=p.data.size()).astype(np.float32) 
        lr_social *= SOCIAL_LR
#         print(pb.data.size(), p.data.size())
        vel_cog = (pb - p) * torch.tensor(lr_cog)
        vel_social = (gb - p) * torch.tensor(lr_social)
        vel = INERTIA * vel + vel_cog + vel_social
        vel_params[name].data = vel
        p.data += SIGMA * vel
    net.load_state_dict(params)
    return net

if __name__=='__main__':
    writer = SummaryWriter(comment='CartPole-PS')
    env = gym.make(ENV_NAME)
    itr = 0
    nets = [Net(env.observation_space.shape[0], env.action_space.n) for _ in range(POPULATION_SIZE)]
    population = [(net, evaluate(env, net)) for net in nets]
    best_params = None 
    best_val = None
    while True:
        population.sort(key=lambda p: p[1], reverse=True)
        fitnesses = [i[1] for i in population]
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
        
        if best_val is None or best_val <= fitnesses[0]:
            best_val = fitnesses[0]
            best_params = population[0][0].state_dict()
#             print(fitnesses)
#             print(best_val)
            for net in nets:
                net.gval = best_val
                net.gbest = best_params
        nets = [move(net) for net in nets]
        population = [(net, evaluate(env, net)) for net in nets]
        itr += 1
    pass