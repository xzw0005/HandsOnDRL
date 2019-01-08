'''
Created on Jan 7, 2019

@author: wangxing
'''
import gym
import pybullet_envs
import numpy as np
import math
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

import ptan

ENV_ID = "MinitaurBulletEnv-v0"
GAMMA = 0.99
REWARD_STEPS = 2
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
ENTROPY_BETA = 1e-4
HID_SIZE = 128
TEST_ITERS = 1000

class ACModel(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ACModel, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(obs_size, HID_SIZE),
                nn.ReLU(),
            )
        self.mu = nn.Sequential(
                nn.Linear(HID_SIZE, act_size),
                nn.Tanh(),
            )
        self.sigma_sq = nn.Sequential(
                nn.Linear(HID_SIZE, act_size),
                nn.Softplus(),
            )
        self.state_val= nn.Sequential(
                nn.Linear(HID_SIZE, 1)
            )
        
    def forward(self, x):
        fc_out = self.fc(x)
        return self.mu(fc_out), self.sigma_sq(fc_out), self.state_val(fc_out)
        
class ACAgent:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device 
        
    def initial_state(self):
        return None 
    
    def __call__(self, states, agent_states=None):
        states_np = np.array(states, dtype=np.float32)
        states_v = torch.tensor(states_np).to(self.device)
        mu_v, sigma_sq_v, state_vals_v = self.model(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(sigma_sq_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, -1, 1)
        return actions, agent_states


def test_net(net, env, cnt=10, device='cpu'):
    tot_rew = 0.
    steps = 0
    for _ in range(cnt):
        obs = env.reset()
        while True:
            obs_np = np.array([obs], dtype=np.float32)
            obs_v = torch.tensor(obs_np).to(device)
            mu_v = net(obs_v)[0]
            act = mu_v.squeeze(dim=0).data.cpu().numpy()
            act = np.clip(act, -1, 1)
            sp, r, done, _ = env.step(act)
            tot_rew += r
            steps += 1 
            if done:
                break
            obs = sp
    return tot_rew/cnt, steps/cnt

def logprob_gaussian(mu_v, sigma_sq_v, acts_v):
    # Gaussian PDF = 1/sqrt(2*pi*sigma^2) * exp( -(x-mu)^2 / (2*sigma^2) )
    # logprob = -log(sqrt(2*pi*sigma^2))  -(x-mu)^2 / (2 * sigma^2) 
    lp1 = -torch.log(torch.sqrt(2 * math.pi * sigma_sq_v))
#     lp2 = -(acts_v - mu_v)^2 / (2 * sigma_sq_v.clamp(min=1e-3))
    lp2 = -(acts_v - mu_v) * (acts_v - mu_v) / (2 * sigma_sq_v.clamp(min=1e-3))
    return lp1 + lp2

def unpack_batch_ac(batch, net, last_val_gamma, device='cpu'):
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = [] 
    for i, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(i)
            last_states.append(exp.last_state)
    states_np = np.array(states, dtype=np.float32)
    states_v = torch.tensor(states_np).to(device)
    actions_v = torch.FloatTensor(actions).to(device)
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_np = np.array(last_states, dtype=np.float32)
        last_states_v = torch.tensor(last_states_np).to(device)
        last_vals_v = net(last_states_v)[2]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np
    qvals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, qvals_v
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
    
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    save_path = os.path.join('saves', 'ac-')
    os.makedirs(save_path, exist_ok=True)
    
    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID)
        
    net = ACModel(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(net)
    agent = ACAgent(net, device=device)
    
    writer = SummaryWriter(comment='-ac-'+ENV_ID)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)
    
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    batch = []
    best_rew = None 
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track('episode_steps', steps[0], step_idx)
                    tracker.reward(rewards[0], step_idx)
                if step_idx % TEST_ITERS == 0:
                    tic = time.time()
                    rewards, steps = test_net(net, test_env, device=device)
                    print('Test done in %.2f sec, mean episodic rewards=%.3f, mean episodic steps=%d'\
                          %(time.time()-tic, rewards, steps))
                    writer.add_scalar('test_rewards', rewards, step_idx)
                    writer.add_scalar('test_steps', steps, step_idx)
                    if best_rew is None or best_rew < rewards:
                        if best_rew is not None:
                            print("Best episodic reward updated %.3f ==> %.3f"%(best_rew, rewards))
                            name = 'best_%+.3f_%d.dat' % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(net.state_dict(), fname)
                    best_rew = rewards
                    
                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue
                
                states_v, actions_v, qvals_v = unpack_batch_ac(batch, net, GAMMA**REWARD_STEPS, device=device)
                batch.clear()
                
                optimizer.zero_grad()
                mu_v, sigma_sq_v, state_vals_v = net(states_v)
                
                loss_sval = F.mse_loss(input=state_vals_v.squeeze(-1), target=qvals_v)
                
                adv_v = qvals_v.unsqueeze(dim=-1) - state_vals_v.detach()
                logpi = logprob_gaussian(mu_v, sigma_sq_v, actions_v)
                weighted_logpi = adv_v * logpi
                loss_policy = -weighted_logpi.mean()
                
                entropy = (-(torch.log(2 * math.pi * sigma_sq_v) + 1) / 2 ).mean()
                loss_entropy = ENTROPY_BETA * entropy
                
                loss = loss_sval + loss_policy + loss_entropy
                loss.backward()
                optimizer.step()
                
                writer.add_scalar("advantage", np.mean(adv_v.data.cpu().numpy()), step_idx)
                writer.add_scalar("batch_rewards", np.mean(qvals_v.data.cpu().numpy()), step_idx)
                writer.add_scalar("V(s)", np.mean(state_vals_v.data.cpu().numpy()), step_idx)
                writer.add_scalar("loss_entropy", loss_entropy.item(), step_idx)
                writer.add_scalar("loss_policy", loss_policy.item(), step_idx)
                writer.add_scalar("loss_total", loss.item(), step_idx)  
            