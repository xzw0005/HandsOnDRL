'''
Created on Dec 30, 2018

@author: wangxing
'''
import time
import sys
import collections
import argparse 
import numpy as np
import gym
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import ptan

GAMMA = 0.99
LEARNING_RATE = 3e-3
ENTROPY_BETA = 0.02
BATCH_SIZE = 32
NUM_ENVS = 50

REWARD_STEPS = 4
CLIP_GRAD = 0.1

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
    
class PGAgent:
    def __init__(self, model, apply_softmax=False, device='cpu'):
        self.model = model
        self.apply_softmax = apply_softmax
        self.device = device 
        
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
            actions.append(np.random.choice(len(p),p=p))
        return np.array(actions), agent_states

def unpack_batch(batch, net, device='cpu'):
    states, actions, rewards, not_dones, next_states = [], [], [], [], []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_dones.append(idx)
            next_states.append(np.array(exp.last_state, copy=False))
    states_v = torch.FloatTensor(states).to(device)
    actions_v = torch.LongTensor(actions).to(device)
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_dones:
        next_states_v = torch.FloatTensor(next_states).to(device)
        next_vals_v = net(next_states_v)[1]
        next_vals = next_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_dones] += GAMMA**REWARD_STEPS * next_vals
    qvals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, qvals_v

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
        
class TBMeanTracker:
    def __init__(self, writer, batch_size):
        assert isinstance(batch_size, int)
        assert writer is not None
        self.writer = writer
        self.batch_size = batch_size

    def __enter__(self):
        self._batches = collections.defaultdict(list)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    @staticmethod
    def _as_float(value):
        assert isinstance(value, (float, int, np.ndarray, np.generic, torch.autograd.Variable)) or torch.is_tensor(value)
        tensor_val = None
        if isinstance(value, torch.autograd.Variable):
            tensor_val = value.data
        elif torch.is_tensor(value):
            tensor_val = value

        if tensor_val is not None:
            return tensor_val.float().mean()
        elif isinstance(value, np.ndarray):
            return float(np.mean(value))
        else:
            return float(value)

    def track(self, param_name, value, iter_index):
        assert isinstance(param_name, str)
        assert isinstance(iter_index, int)

        data = self._batches[param_name]
        data.append(self._as_float(value))

        if len(data) >= self.batch_size:
            self.writer.add_scalar(param_name, np.mean(data), iter_index)
            data.clear()

def make_env(env_name='PongNoFrameskip-v4'):
    env = gym.make(env_name)
    return ptan.common.wrappers.wrap_dqn(env)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
    
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    envs = [make_env() for _ in range(NUM_ENVS)]
    writer = SummaryWriter(comment='-pong-ac')
    
    net = AtariActorCritic(envs[0].observation_space.shape, \
                        envs[0].action_space.n).to(device)
    print(net)
    
    agent = PGAgent(lambda s: net(s)[0], apply_softmax=True, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
    
    batch = []
    with RewardTracker(writer, stop_reward=18) as tracker:
#         with TBMeanTracker(writer, batch_size=10) as tb_tracker:
        for step_idx, exp in enumerate(exp_source):
            batch.append(exp)
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if tracker.reward(new_rewards[0], step_idx):
                    break
            if len(batch) < BATCH_SIZE:
                continue
            
            states_v, actions_v, qvals_v = unpack_batch(batch, net, device=device)
            batch.clear()
            
            optimizer.zero_grad()
            logits, Vs = net(states_v)
            
            loss_value = F.mse_loss(
                    input=Vs.squeeze(-1), target=qvals_v)
            
            log_pi = F.log_softmax(logits, dim=1)
            adv = qvals_v - Vs.detach()
            
            weighted_log_pi = adv * log_pi[range(BATCH_SIZE), actions_v]
            loss_policy = -weighted_log_pi.mean()
            
            pi = F.softmax(logits, dim=1)
            entropy = (-pi * log_pi).sum(dim=1).mean()
            loss_entropy = ENTROPY_BETA * entropy
            
            loss_policy.backward(retain_graph=True)
            grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                for p in net.parameters() if p.grad is not None])
            
            loss = loss_entropy + loss_value
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
            optimizer.step()
            
            loss += loss_policy
                
            writer.add_scalar("advantage", np.mean(adv.data.cpu().numpy()), step_idx)
            writer.add_scalar("batch_rewards", np.mean(qvals_v.data.cpu().numpy()), step_idx)
            writer.add_scalar("values", np.mean(Vs.data.cpu().numpy()), step_idx)
            writer.add_scalar("loss_entropy", loss_entropy.item(), step_idx)
            writer.add_scalar("loss_policy", loss_policy.item(), step_idx)
            writer.add_scalar("loss_total", loss.item(), step_idx)        
    writer.close()             
                
#                 tb_tracker.track("advantage",       adv, step_idx)
#                 tb_tracker.track("values",          Vs, step_idx)
#                 tb_tracker.track("batch_rewards",   qvals_v, step_idx)
#                 tb_tracker.track("loss_entropy",    loss_entropy, step_idx)
#                 tb_tracker.track("loss_policy",     loss_policy, step_idx)
#                 tb_tracker.track("loss_value",      loss_value, step_idx)
#                 tb_tracker.track("loss_total",      loss, step_idx)
#                 tb_tracker.track("grad_l2",         np.sqrt(np.mean(np.square(grads))), step_idx)
#                 tb_tracker.track("grad_max",        np.max(np.abs(grads)), step_idx)
#                 tb_tracker.track("grad_var",        np.var(grads), step_idx)