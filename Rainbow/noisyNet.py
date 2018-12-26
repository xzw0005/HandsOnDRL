'''
Created on Dec 25, 2018

@author: wangxing
'''
import gym
import ptan
import argparse
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import PARAMS
import common
import utils.agent as agent
import utils.actions as actions

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer(name='epsilon_weight', tensor=torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features, ), sigma_init))
            self.register_buffer(name='epsilon_bias', tensor=torch.zeros(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        std = np.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)
        
    def forward(self, x):
        self.epsilon_weight.normal_()
        bias = self.bias 
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data 
        return F.linear(input=x, weight=self.weight+self.sigma_weight*self.epsilon_weight.data, bias=bias)
    
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
        

class NoisyDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(NoisyDQN, self).__init__()
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
        self.noisy_layers = [
                NoisyLinear(conv_out_size, 512),
                NoisyLinear(512, num_actions)
            ]
        self.fc = nn.Sequential(
                self.noisy_layers[0],
                nn.ReLU(),
                self.noisy_layers[1]
            )

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)
    
    def noisy_layers_sigma_snr(self):
        return [( (layer.weight**2).mean().sqrt() / (layer.sigma_weight**2).mean().sqrt() ).item() \
                for layer in self.noisy_layers]

if __name__=='__main__':
    params = PARAMS.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
    parser.add_argument('-n', default=1, type=int, help='Number of steps unrolling Bellman')
    parser.add_argument('--double', default=False, action='store_true', help='Enable double DQN')
    
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)
    
    comment_str = '-'+params['run_name']
    if args.double:
        comment_str += '-double'
    if args.n:
        comment_str += '-%d-step'%args.n
    writer = SummaryWriter(comment=comment_str+'-noisy')
    net = NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = agent.TargetNet(net)
    selector = actions.ArgmaxActionSelector()
    agent = agent.DQNAgent(net, selector, device=device)
    
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, \
                                gamma=params['gamma'], steps_count=args.n)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])
    
    frame_idx = 0
    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx):
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
                
            if frame_idx % 500 == 0:
                for layer_idx, sigma_l2 in enumerate(net.noisy_layers_sigma_snr()):
                    writer.add_scalar("sigma_snr_layer_%d"%(layer_idx+1), sigma_l2, frame_idx) 