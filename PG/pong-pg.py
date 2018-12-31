'''
Created on Dec 30, 2018

@author: wangxing
'''
import time
import sys
import gym 
import numpy as np
import argparse
from collections import deque
from tensorboardX import SummaryWriter 

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import ptan

GAMMA = 0.99
LEARNING_RATE = 1e-4
ENTROPY_BETA = 0.01
BATCH_SIZE = 128

REWARD_STEPS = 10
BASELINE_STEPS = int(1e6)
GRAD_L2_CLIP = 0.1

ENV_COUNT = 32

class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False

class MeanBuffer:
    def __init__(self, capacity):
        self.capacity = capacity 
        self.buffer = deque(maxlen=capacity)
        self.sum = 0.
        
    def add(self, val):
        if len(self.buffer) == self.capacity:
            self.sum -= self.buffer[0]
        self.buffer.append(val)
        self.sum += val 
        
    def mean(self):
        if not self.buffer:
            return 0.
        return self.sum / len(self.buffer)

def make_env(env_name='PongNoFrameskip-v4'):
    env = gym.make(env_name)
    return ptan.common.wrappers.wrap_dqn(env)
    
class AtariPGN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(AtariPGN, self).__init__()
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

class PGAgent:
    def __init__(self, model, apply_softmax=False, device='cpu'):
        self.model = model
        self.device = device 
        self.apply_softmax = apply_softmax
        
    def initial_state(self):
        return None
        
    def __call__(self, states, agent_states=None):
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
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
    
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    envs = [make_env() for _ in range(ENV_COUNT)]
    writer = SummaryWriter(comment='pong-pg')
    
    net = AtariPGN(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print(net)
    
    agent = PGAgent(net, apply_softmax=True, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(\
                    envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
        
    total_rewards = []
    step_idx = 0
    done_episodes = 0
    train_step_idx = 0
    baseline_buffer = MeanBuffer(BASELINE_STEPS)
    
    batch_states, batch_actions, batch_scales = [], [], []
    
    with RewardTracker(writer, stop_reward=18) as tracker:
        for step_idx, exp in enumerate(exp_source):
            baseline_buffer.add(exp.reward)
            b = baseline_buffer.mean()
            
            batch_states.append(np.array(exp.state, copy=False))
            batch_actions.append(int(exp.action))
            batch_scales.append(exp.reward - b)
            
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if tracker.reward(new_rewards[0], step_idx):
                    break 
            if len(batch_states) < BATCH_SIZE:
                continue
            
            train_step_idx += 1
            
            states_v = torch.FloatTensor(batch_states).to(device)
            actions_v = torch.LongTensor(batch_actions).to(device)
            scales_std = np.std(batch_scales)
            scales_v = torch.FloatTensor(batch_scales).to(device)
            
            optimizer.zero_grad()
            logits = net(states_v)
            log_pi = F.log_softmax(logits, dim=1)
            weighted_log_pi = scales_v * log_pi[range(BATCH_SIZE), actions_v]
            loss_policy = -weighted_log_pi.mean()
            
            pi = F.softmax(logits, dim=1)
            entropy = (-pi * log_pi).sum(dim=1).mean()
            loss_entropy = -ENTROPY_BETA * entropy 
            
            loss = loss_policy + loss_entropy
            loss.backward()
            nn.utils.clip_grad_norm_(\
                    parameters=net.parameters(), max_norm=GRAD_L2_CLIP)
            optimizer.step()
            
            ## Calculate KL-divergence
            new_logits = net(states_v)
            new_pi = F.softmax(new_logits, dim=1)
            kl = -((new_pi/pi).log() * pi).sum(dim=1).mean()
            writer.add_scalar("KL-Divergence", kl.item(), step_idx)
            
            grad_max = 0.
            grad_means = 0. 
            grad_count = 0
            for p in net.parameters():
                grad_max = max(grad_max, p.grad.abs().max().item())
                grad_means += (p.grad **2).mean().sqrt().item()
                grad_count += 1
                
            writer.add_scalar("baseline", b, step_idx)
            writer.add_scalar("entropy", entropy.item(), step_idx)
            writer.add_scalar("batch_scales", np.mean(batch_scales), step_idx)
            writer.add_scalar("loss_entropy", loss_entropy.item(), step_idx)
            writer.add_scalar("loss_policy", loss_policy.item(), step_idx)
            writer.add_scalar("loss_total", loss.item(), step_idx)
            writer.add_scalar("grad_L2", grad_means/grad_count, step_idx)
            writer.add_scalar("grad_max", grad_max, step_idx)
            
            batch_states.clear()
            batch_actions.clear()
            batch_scales.clear()
        
    writer.close()    