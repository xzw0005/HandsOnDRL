"""
Created on Jan 28, 2019

@author: wangxing
"""
import gym
import argparse
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'next_state'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        sb, ab, rb, db, spb = zip(*[self.buffer[idx] for idx in indices])
        return np.array(sb), np.array(ab), np.array(rb, dtype=np.float32), \
                np.array(db, dtype=np.uint8), np.array(spb)

class Agent:
    def __init__(self, env, replay_buffer):
        self.env = env
        self.replay_buffer = replay_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.

    def play_step(self, net, epsilon):

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
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
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.conv_out

def unpack(batch):
    state_batch, action_batch, reward_batch, done_batch, next_state_batch = [], [], [], [], []
    for experience in batch:
        s = np.array(experience.state, copy=False)
        state_batch.append(s)
        action_batch.append(experience.action)
        reward_batch.append(experience.reward)
        done_batch.append(experience.next_state is None)
        if experience.next_state is None:
            next_state_batch.append(s)
        else:
            sp = np.array(experience.next_state, copy=False)
            next_state_batch.append(sp)
    state_batch = np.array(state_batch, copy=False)
    action_batch = np.array(action_batch)
    reward_batch = np.array(reward_batch, dtype=np.float32)
    done_batch = np.array(done_batch, dtype=np.uint8)
    next_state_batch = np.array(next_state_batch, copy=False)
    return state_batch, action_batch, reward_batch, done_batch, next_state_batch

def calc_loss(batch, net, target_net, gamma, device='cpu', double=True):
    sb, ab, rb, db, spb = unpack(batch)
    states = torch.tensor(sb).to(device)
    actions = torch.tensor(ab).to(device)
    rewards = torch.tensor(rb).to(device)
    next_states = torch.tensor(spb).to(device)
    done_mask = torch.ByteTensor(db).to(device)
    Qs = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    if double:
        actions_sp = net(next_states).max(1)[1]
        Qsp = target_net(next_states).gather(1, actions_sp.unsqueeze(-1)).squeeze(-1)
    else:
        Qsp = target_net(next_states).max(1)[0]
    Qsp[done_mask] = 0.
    Qs_ = rewards + gamma * Qsp.detach()
    loss = nn.MSELoss(Qs, Qs_)
    return loss

def eval_state_vals(states, net, device='cpu'):
    mean_vals = []
    for batch in np.array_split(states, 64):
        s_v = torch.tensor(batch).to(device)
        Qs = net(s_v)
        Qstar = Qs.max(1)[0]
        mean_vals.append(Qstar.mean().item())
    return np.mean(mean_vals)

if __name__=='__main__':


