'''
Created on Jan 2, 2019

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
PROCESSES_COUNT = 5
ITERS_PER_UPDATE = 10
MAX_ITERS = int(1e5)

RewardsItem = collections.namedtuple(typename='RewardsItem', field_names=['seed', 'pos_reward', 'neg_reward', 'steps'])

class Net(nn.Module):
    def __init__(self, obs_size, act_size, hid_size=64):
        super(Net, self).__init__()
        self.mu = nn.Sequential(
                nn.Linear(obs_size, hid_size),
                nn.Tanh(),
                nn.Linear(hid_size, hid_size),
                nn.Tanh(),
                nn.Linear(hid_size, act_size),
                nn.Tanh(),
            )
        
    def forward(self, x):
        return self.mu(x)
    
def make_env():
    return gym.make(ENV_NAME)

def default_states_preprocessor(states):
    if len(states) == 1:
        np_states = np.expand_dims(states[0], axis=0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states)

def evaluate(env, net, device='cpu'):
    obs_size = env.observation_space.shape[0]
    s = env.reset()
    rets = 0.
    steps = 0
    while True:
        s = np.array(s, dtype=np.float32).reshape(1, obs_size)
        s_v = torch.tensor(s).to(device)
        a_v = net(s_v)
        a = a_v.data.cpu().numpy()[0]
        sp, r, done, _ = env.step(a)
        rets += r 
        steps += 1
        if done:
            break
        s = sp
    return rets, steps

def sample_noise(net, seed=None, device='cpu'):
    if seed is not None:
        np.random.seed(seed)
    pos, neg = [], []
    for p in net.parameters():
        noise = np.random.normal(size=p.data.size()).astype(np.float32)
        noise_v = torch.FloatTensor(noise).to(device)
        pos.append(noise_v)
        neg.append(-noise_v)
    return pos, neg

def eval_with_noise(env, net, noise, sigma, device='cpu'):
    for p, pn in zip(net.parameters(), noise):
        p.data += sigma * pn 
    rets, steps = evaluate(env, net, device)
    for p, pn in zip(net.parameters(), noise):
        p.data -= sigma * pn 
    return rets, steps

def compute_ranks(x):
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks 

def compute_centered_ranks(x):
    ranks = compute_ranks(x.ravel())
    ranks = ranks.reshape(x.shape).astype(np.float32)
    ranks /= (x.size-1)
    ranks -= 0.5
    return ranks

def train(optimizer, net, batch_noise, batch_fit, writer, itr, sigma):
    batch_reward = compute_centered_ranks(np.array(batch_fit))
    weighted_noise = None 
    for noise, reward in zip(batch_noise, batch_reward):
        if weighted_noise is None:
            weighted_noise = [reward * pn for pn in noise]
        else:
            for wn, pn in zip(weighted_noise, noise):
                wn += reward*pn
    optimizer.zero_grad()
    batch_updates = []
    for p, wn in zip(net.parameters(), weighted_noise):
        update = wn / (len(batch_fit) * sigma)
        p.grad = -update 
        batch_updates.append(torch.norm(update))
    writer.add_scalar('update_L2', np.mean(batch_updates), itr)
    optimizer.step()
    
def worker_func(worker_id, params_queue, rewards_queue, sigma, device):
    env = make_env()
    net = Net(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    net.eval()
    while True:
        params = params_queue.get()
        if params is None:
            break
        net.load_state_dict(params)
        
        for _ in range(ITERS_PER_UPDATE):
            seed = np.random.randint(low=0, high=65535)
            np.random.seed(seed)
            pos_noise, neg_noise = sample_noise(net, device=device)
            pos_fit, pos_steps = eval_with_noise(env, net, pos_noise, sigma, device)
            neg_fit, neg_steps = eval_with_noise(env, net, neg_noise, sigma, device)
            rewards_queue.put(RewardsItem(seed=seed, pos_reward=pos_fit, neg_reward=neg_fit, steps=pos_steps+neg_steps))
    pass
            
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable CUDA mode")
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--sigma', type=float, default=NOISE_STD)
    parser.add_argument('--iters', type=int, default=MAX_ITERS)

    args = parser.parse_args()
    device = 'cuda' if args.cuda else 'cpu'
    
    writer = SummaryWriter(comment='-cheetah-es_lr=%.3e_sigma=%.3e'%(args.lr, args.sigma))
    env = make_env()
    net = Net(env.observation_space.shape[0], env.action_space.shape[0])
    print(net)

    mp.set_start_method('spawn')
    params_queues = [mp.Queue(maxsize=1) for _ in range(PROCESSES_COUNT)]
    rewards_queue = mp.Queue(maxsize=ITERS_PER_UPDATE)
    
    workers = []
    for idx, params_queue in enumerate(params_queues):
        proc = mp.Process(target=worker_func, args=(idx, params_queue, rewards_queue, args.sigma, device))
        proc.start()
        workers.append(proc)
        
    print('All started!')
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    
    for itr in range(args.iters):
        # Broadcast network params
        params = net.state_dict()
        for q in params_queues:
            q.put(params)
        
        # Wait for results
        tic = time.time()
        batch_noise = []
        batch_fit = []
        cnt = 0
        batch_steps = 0 
        batch_steps_data = []
        while True:
            while not rewards_queue.empty():
                rew_item = rewards_queue.get_nowait()
#                 np.random.seed(rew_item.seed)
                pos_noise, neg_noise = sample_noise(net, seed=rew_item.seed)
                batch_noise.append(pos_noise)
                batch_fit.append(rew_item.pos_reward)
                batch_noise.append(neg_noise)
                batch_fit.append(rew_item.neg_reward)
                cnt += 1
                batch_steps += rew_item.steps
                batch_steps_data.append(rew_item.steps)
                
            if cnt == PROCESSES_COUNT*ITERS_PER_UPDATE:
                break
            time.sleep(0.01)
            
        worker_time = time.time() - tic 
        m_fit = np.mean(batch_fit)
        train(optimizer, net, batch_noise, batch_fit, writer, itr, args.sigma)
        master_time = time.time() - tic - worker_time
        
        writer.add_scalar('fitness_mean', m_fit, itr)
        writer.add_scalar('fitness_std', np.std(batch_fit), itr)
        writer.add_scalar('fitness_max', np.max(batch_fit), itr)
        writer.add_scalar('batch_episodes', len(batch_fit), itr)
        writer.add_scalar('batch_steps', batch_steps, itr)
        speed = batch_steps / (worker_time)
        writer.add_scalar('speed', speed, itr)
        
        print('%d: fitness=%.2f, speed=%.2f f/s, worker time=%.3f s, master train time=%.3f s, steps_mean=%.2f, min=%.2f, max=%.2f, steps_std=%.2f' \
              % (itr, m_fit, speed, worker_time, master_time, np.mean(batch_steps_data), np.min(batch_steps_data), np.max(batch_steps_data), np.std(batch_steps_data)) )
              
    for worker, params_queue in zip(workers, params_queues):
        params_queue.put(None)
        worker.join()