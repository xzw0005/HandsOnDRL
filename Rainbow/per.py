'''
Created on Dec 26, 2018

@author: wangxing
'''
import gym 
import numpy as np
import argparse 

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import ptan 
import common
import PARAMS
import utils.agent as agent
import utils.actions as actions
# from ptan.experience import PrioritizedReplayBuffer

PER_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000

class PrioritizedReplayBuffer:
    def __init__(self, exp_source, buffer_size, prob_alpha=0.6):
        self.exp_source_iter = iter(exp_source)
        self.prob_alpha = prob_alpha
        self.capacity = buffer_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buffer_size, ), dtype=np.float32)
        
    def __len__(self):
        return len(self.buffer)
    
    def populate(self, count):
        max_priority = self.priorities.max() if self.buffer else 1.
        for _ in range(count):
            sample = next(self.exp_source_iter)
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.pos] = sample 
            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity
            
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        probs = priorities ** self.prob_alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (probs[indices] * len(self.buffer)) ** (-beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority
            
def calc_loss(batch, batch_weights, net, target_net, gamma, device='cpu', double=True):
    sb, ab, rb, db, spb = common.unpack_batch(batch)
    states = torch.tensor(sb).to(device)
    actions = torch.tensor(ab).to(device)
    rewards = torch.tensor(rb).to(device)
    next_states = torch.tensor(spb).to(device)
    done_mask = torch.ByteTensor(db).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)
    
    Qs = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    if double:
        actions_sp = net(next_states).max(1)[1]
        Qsp = target_net.gather(1, actions_sp.unsqueeze(-1)).squeeze(-1)
    else:
        Qsp = target_net(next_states).max(1)[0]
    Qsp[done_mask] = 0.
    Qs_ = rewards + gamma * Qsp.detach()
    loss = batch_weights_v * (Qs - Qs_)**2
    return loss.mean(), loss + 1e-5

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
        return self.fc(conv_out)

if __name__=='__main__':
    params = PARAMS.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
    parser.add_argument('--double', default=False, action='store_true', help='Enable Double Learning')
    parser.add_argument('-n', default=1, type=int, help='Number of steps unrolling Bellman')
    
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)
    
    comment_str = '-'+params['run_name']
    if args.double:
        comment_str += '-double'
    if args.n:
        comment_str += '-%d-step'%args.n
    writer = SummaryWriter(comment=comment_str+'-per')
    
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = agent.TargetNet(net)
    selector = actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = agent.DQNAgent(net, selector, device=device)
    
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, \
                            gamma=params['gamma'], steps_count=args.n)
    buffer = PrioritizedReplayBuffer(exp_source, params['replay_size'], PER_ALPHA)
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])
    
    frame_idx = 0
    beta = BETA_START
    
    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)
            beta = min(1., BETA_START+frame_idx*(1.-BETA_START)/BETA_FRAMES)
            
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break
                
            if len(buffer) < params['replay_initial']:
                continue
            
            optimizer.zero_grad()
            batch, batch_indices, batch_weights = buffer.sample(params['batch_size'], beta)
            loss, sample_priorities = calc_loss(batch, batch_weights, net, target_net.target_model,\
                                        gamma=params['gamma']**args.n, double=args.double, device=device)
            loss.backward()
            optimizer.step()
            
            buffer.update_priorities(batch_indices, sample_priorities.data.cpu().numpy())
            
            if frame_idx % params['target_net_sync'] == 0:
                target_net.sync()      