'''
Created on Dec 31, 2018

@author: wangxing
'''
import gym
import numpy as np
import argparse
import collections
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

import ptan
import common

GAMMA = 0.99
LEARNING_RATE = 1e-3
ENTROPY_BETA = 0.01
BATCH_SIZE = 128

REWARD_STEPS = 4
CLIP_GRAD = 0.1

PROCESSES_COUNT = 3
NUM_ENVS = 5

ENV_NAME = 'PongNoFrameskip-v4'
REWARD_BOUND = 18


def make_env():
    env = gym.make(ENV_NAME)
    return ptan.common.wrappers.wrap_dqn(env)

TotalReward = collections.namedtuple(typename='TotalReward', \
                                     field_names='reward')

def data_func(net, device, train_queue):
    envs = [make_env() for _ in range(NUM_ENVS)]
    agent = PGAgent(lambda s: net(s)[0], apply_softmax=True, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(\
                envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    for ex in exp_source:
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            train_queue.put(TotalReward(reward=np.mean(new_rewards)))
        train_queue.put(ex)            
    
class PGAgent:
    def __init__(self, model, apply_softmax=False, device='cpu'):
        self.model = model
        self.apply_softmax = apply_softmax
        self.device = device 
        
    def initial_state(self):
        return None
    
    def __call__(self, states, agent_states= None):
        if agent_states is None:
            agent_states = [None] * len(states)
        if len(states) == 1:
            states = np.expand_dims(states[0], axis=0)
        else:
            states = np.array([np.array(s, copy=False) for s in states], copy=False)
        states = torch.tensor(states).to(self.device)
        probs = self.model(states)
        if self.apply_softmax:
            probs = F.softmax(probs, dim=1)
        probs = probs.data.cpu().numpy()
        actions = []
        for p in probs:
            actions.append(np.random.choice(len(p), p=p))
        return np.array(actions), agent_states

class AtariActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(AtariActorCritic, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
        o = self.conv(torch.zeros(1, *input_shape))
        conv_out_size = int(np.prod(o.size()))
        self.policy = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )
        self.value = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
        
    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)
    
def unpack_batch(batch, net, device='cpu'):
    states, actions, rewards, not_dones, next_states = [],[],[],[],[]
    for i, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_dones.append(i)
            next_states.append(np.array(exp.last_state, copy=False))
    states_v = torch.FloatTensor(states).to(device)
    actions_v = torch.LongTensor(actions).to(device)
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_dones:
        next_states_v = torch.FloatTensor(next_states).to(device)
        next_vals_v = net(next_states_v)[1]
        next_vals = next_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_dones] += GAMMA ** REWARD_STEPS * next_vals
    qvals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, qvals_v
    
if __name__=='__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
    args = parser.parse_args()
    
    device = 'cuda' if args.cuda else 'cpu'
    writer = SummaryWriter(comment='-a3c-data-'+ENV_NAME)
    
    env = make_env()
    net = AtariActorCritic(env.observation_space.shape, \
            env.action_space.n).to(device)
    net.share_memory()

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
    
    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    proc_list = []
    for _ in range(PROCESSES_COUNT):
        proc = mp.Process(target=data_func, args=(net, device, train_queue))
        proc.start()
        proc_list.append(proc)
        
    batch = []
    step_idx = 0
    try:
        with common.RewardTracker(writer, stop_reward=REWARD_BOUND) as tracker:
            with common.TBMeanTracker(writer, batch_size=100) as tb_tracker:
                while True:
                    train_entry = train_queue.get()
                    if isinstance(train_entry, TotalReward):
                        if tracker.reward(train_entry.reward, step_idx):
                            break
                        continue
                    
                    step_idx +=1
                    batch.append(train_entry)
                    if len(batch) < BATCH_SIZE:
                        continue
                    
                    states_v, actions_v, qvals_v = unpack_batch(batch, net, device)
                    batch.clear()
                    
                    optimizer.zero_grad()
                    logits, Vs = net(states_v)
                    
                    loss_value = F.mse_loss(input=Vs.squeeze(-1), target=qvals_v)
                    
                    log_pi = F.log_softmax(logits, dim=1)
                    adv = qvals_v - Vs.detach()
                    
                    weighted_log_pi = adv * log_pi[range(BATCH_SIZE), actions_v]
                    loss_policy = -weighted_log_pi.mean()
                    
                    pi = F.softmax(logits, dim=1)
                    entropy = (-pi * log_pi).sum(dim=1).mean()
                    loss_entropy = ENTROPY_BETA * entropy 
                    
                    loss = loss_value + loss_policy + loss_entropy
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                    optimizer.step()
                    
                    tb_tracker.track('advantage', adv, step_idx)
                    tb_tracker.track('values', Vs, step_idx)
                    tb_tracker.track('batch_rewards', qvals_v, step_idx)
                    tb_tracker.track('loss_value', loss_value, step_idx)
                    tb_tracker.track('loss_policy', loss_policy, step_idx)
                    tb_tracker.track('loss_entropy', loss_entropy, step_idx)                    
                    tb_tracker.track('loss_total', loss, step_idx)
                    
    finally:
        for p in proc_list:
            p.terminate()
            p.join()