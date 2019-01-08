'''
Created on Jan 6, 2019

@author: wangxing
'''
import numpy as np
import copy 
import gym
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter

ENVNAME = 'CartPole-v0'
SIGMA = 0.01
POPSIZE = 50
PROB_CRS = 0.7
PARENTS_COUNT = int(.2*POPSIZE)

class Net(nn.Module):
    def __init__(self, obs_size, act_size, seed=None):
        super(Net, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(obs_size, 32),
                nn.ReLU(),
                nn.Linear(32, act_size), 
                nn.Softmax(dim=1)
            )
        self.fitness = None
        
    def forward(self, x):
        return self.net(x)
    
def mutate(net, sigma=SIGMA, seed=None):
    if sigma is None:
        return net
    if seed is not None:
        np.random.seed(seed)
    for p in net.parameters():
        noise_np = np.random.normal(size=p.data.size()).astype(np.float32)
        noise_v = torch.tensor(noise_np)
        p.data += sigma * noise_v 
    return net
    
def crossover(fnet, mnet, ffit=None, mfit=None, seed=None):
    pm = 0.5
    if ffit is not None and mfit is not None:
        pm = mfit / (ffit + mfit + 1e-5)
    if seed is not None:
        torch.manual_seed(seed)
    child = copy.deepcopy(mnet)
    for param_tensor, ftensor in zip(child.parameters(), fnet.parameters()):
#         param_tensor = pm * param_tensor
#         param_tensor += (1-pm) * ftensor
        mask = torch.bernoulli(pm * torch.ones(param_tensor.data.size()))
        param_tensor = param_tensor * mask 
        param_tensor += (ftensor * (1 - mask))
    return child
            
def evaluate(net, env):
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

def selection():
    pass

if __name__=='__main__':
    writer = SummaryWriter(comment='-'+ENVNAME+'-Crossover')
    envs = [gym.make(ENVNAME) for _ in range(POPSIZE)]
    nets = [Net(envs[0].observation_space.shape[0], envs[0].action_space.n) for _ in range(POPSIZE)]
    population = [(net,evaluate(net, env)) for net, env in zip(nets, envs)]
    itr = 0
    while True:
        population.sort(key=lambda i: i[1], reverse=True)
        fitnesses = [p[1] for p in population]
        
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
        
        old = copy.copy(population)
#         n_elites = int(POPSIZE * (1.-PROB_CRS))
#         nets = [mutate(indiv[0]) for indiv in population[:n_elites]]
#         population = [(net, evaluate(net, env)) for net, env in zip(nets, envs)]
#         probs = np.array(range(POPSIZE, 0, -1), dtype=np.float)
#         probs /= sum(probs)
#         while len(population) < POPSIZE:
#             # Already sorted, use rank softmax as probs for selection
#             parents = np.random.choice(POPSIZE, size=2, p=probs)
#             fnet, ffit = old[parents[0]]
#             mnet, mfit = old[parents[1]]
#             child_net = crossover(fnet, mnet)#, ffit, mfit)
#             child_net = mutate(child_net)
#             population.append( (child_net, evaluate(child_net, envs[-1]) ) )

        elite = old[0][0]
        population = [(elite, evaluate(elite, envs[-1]) )]            
        for _ in range(POPSIZE-1):
            parents = np.random.choice(PARENTS_COUNT, size=2)
            fnet, ffit = old[parents[0]]
            mnet, mfit = old[parents[1]]
            child_net = crossover(fnet, mnet)#, ffit, mfit)
            child_net = mutate(child_net)
            population.append( (child_net, evaluate(child_net, envs[-1]) ) )
        itr += 1