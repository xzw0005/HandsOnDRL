'''
Created on Jan 4, 2019

@author: wangxing
'''
import gym
import numpy as np
import collections 
import copy 
import time 
import sys 

import torch
import torch.nn as nn
import torch.multiprocessing as mp 

from tensorboardX import SummaryWriter 

ENV_NAME = 'HalfCheetah-v2'
NOISE_STD = 0.01
POPULATION_SIZE = 200
PARENTS_COUNT = 10
WORKERS_COUNT = 6
SEEDS_PER_WORKER = POPULATION_SIZE // WORKERS_COUNT
MAX_SEED = 2**31 - 1

class Net(nn.Module):
    def __init__(self, obs_size, act_size, hid_size=64):
        super(Net, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(obs_size, hid_size),
                nn.Tanh(),
                nn.Linear(hid_size, hid_size),
                nn.Tanh(),
                nn.Linear(hid_size, act_size),
                nn.Tanh(),
            )
    def forward(self, x):
        return self.net(x)
    
def evaluate(env, net):
    s = env.reset()
    tot_rew = 0.
    steps = 0
    while True:
        s_v = torch.tensor(np.array([s]).astype(np.float32))
        a_v = net(s_v)
        a = a_v.data.numpy()[0]
        sp, r, done, _ = env.step(a)
        tot_rew += r
        steps += 1
        if done:
            break
        s = sp 
    return tot_rew, steps 

def mutate(net, seed, copy_net=True):
    new_net = copy.deepcopy(net) if copy_net else net 
    np.random.seed(seed)
    for p in new_net.parameters():
        noise_np = np.random.normal(size=p.data.size()).astype(np.float32)
        noise_v = torch.tensor(noise_np)
        p.data += NOISE_STD * noise_v 
    return new_net

def build_net(env, seeds):
    torch.manual_seed(seeds[0])
    net = Net(env.observation_space.shape[0], env.action_space.shape[0])
    for seed in seeds[1:]:
        net = mutate(net, seed, copy_net=False)
    return net 

OutputItem = collections.namedtuple(typename='OutputItem', field_names=['seeds', 'fitness', 'steps'])

def worker_func(input_queue, output_queue):
    env = gym.make(ENV_NAME)
    cache = {} 
    
    while True:
        parents = input_queue.get()
        if parents is None:
            break 
        new_cache = {}
        for seeds in parents:
            if len(seeds) == 1:
                net = build_net(env, seeds)
            else:
                net = cache.get(seeds[:-1])
                if net is None:
                    net = build_net(env, seeds)
                else:
                    net = mutate(net, seeds[-1],  copy_net=False)
            new_cache[seeds] = net 
            fitness, steps = evaluate(env, net)
            output_queue.put(OutputItem(seeds=seeds, fitness=fitness, steps=steps))
        cache = new_cache
        
if __name__=='__main__':
    mp.set_start_method('spawn')
    writer = SummaryWriter(comment='cheetah-ga')
    
    input_queues = []
    output_queue = mp.Queue(maxsize=WORKERS_COUNT)
    workers = []
    for _ in range(WORKERS_COUNT):
        input_queue = mp.Queue(maxsize=1)
        input_queues.append(input_queue)
        proc = mp.Process(target=worker_func, args=(input_queue, output_queue))
        proc.start()
        seeds = [(np.random.randint(MAX_SEED), ) for _ in range(SEEDS_PER_WORKER)]
        input_queue.put(seeds)
        
    itr = 0 
    elite = None 
    while True:
        tic = time.time()
        batch_steps = 0 
        population = [] 
        while len(population) < SEEDS_PER_WORKER*WORKERS_COUNT:
            output_item = output_queue.get()
            population.append((output_item.seeds, output_item.fitness))
            batch_steps += output_item.steps 
        if elite is not None:
            population.append(elite)
        population.sort(key=lambda p: p[1], reverse=True)
        
        fitnesses = [p[1] for p in population[:PARENTS_COUNT]]
        fit_mean = np.mean(fitnesses)
        fit_std = np.std(fitnesses)
        fit_max = np.max(fitnesses)
        writer.add_scalar('fitness_mean', fit_mean, itr)
        writer.add_scalar('fitness_std', fit_std, itr)
        writer.add_scalar('fitness_max', fit_max, itr)
        writer.add_scalar("batch_steps", batch_steps, itr)
        gen_time = time.time() - tic 
        writer.add_scalar("gen_seconds", gen_time, itr)
        speed = batch_steps / gen_time
        writer.add_scalar('speed', speed, itr)
        print("%d: fitness_mean=%.2f, fitness_std=%.2f, fitness_max=%.2f, speed=%.2f f/s, gen_time=%.2f s"%(itr, fit_mean, fit_std, fit_max, speed, gen_time))
        
        elite = population[0]
        for worker_queue in input_queues:
            seeds = []
            for _ in range(SEEDS_PER_WORKER):
                parent = np.random.randint(PARENTS_COUNT)
                next_seed = np.random.randint(MAX_SEED)
                seeds.append(tuple( list(population[parent][0]) + [next_seed] ))
            worker_queue.put(seeds)
        itr += 1
    pass