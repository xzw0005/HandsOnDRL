'''
Created on Dec 26, 2018

@author: wangxing
'''
import argparse
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import ptan 
import common
import PARAMS
import utils.agent as agent
import utils.actions as actions


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelingDQN, self).__init__()
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
        self.fc_adv = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )
        self.fc_val = nn.Sequential(
                nn.Linear(conv_out_size, 512), 
                nn.ReLU(),
                nn.Linear(512, 1)
            )
        
    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean()
    
if __name__=='__main__':
    params = PARAMS.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
    parser.add_argument('--double', default=False, action='store_true', help='Enable Double DQN')
    parser.add_argument('-n', default=1, type=int, help='Number of steps unrolling Bellman')
    
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')
    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)
    comment_str = '-'+params['run_name']
    if args.double:
        comment_str += '-double'
    if args.n > 1:
        comment_str += '-%d-step'%args.n
    writer = SummaryWriter(comment=comment_str+'-dueling')
    
    net = DuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = agent.TargetNet(net)
    selector = actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = agent.DQNAgent(net, selector, device=device)
    
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, \
                    gamma=params['gamma'], steps_count=args.n)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])
    
    frame_idx = 0
    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break  
            if len(buffer) < params['replay_initial']:
                continue
            
            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            loss = common.calc_loss_dqn(batch, net, target_net.target_model, \
                                gamma=params['gamma']**args.n, double=args.double, device=device)
            loss.backward()
            optimizer.step()
            
            if frame_idx % params['target_net_sync'] == 0:
                target_net.sync()