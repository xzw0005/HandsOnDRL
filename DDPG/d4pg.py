'''
Created on Jan 9, 2019

@author: wangxing
'''
import gym
import pybullet_envs
import numpy as np
import copy
import os
import time
import argparse
from tensorboardX import SummaryWriter
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import ptan

# ENV_ID = "MinitaurBulletEnv-v0"
ENV_ID = 'Pendulum-v0'
GAMMA = 0.99
REWARD_STEPS = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
REPLAY_SIZE = int(1e5)
REPLAY_INITIAL = int(1e4)
TEST_ITERS = 1000

Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

class ActorD4PG(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ActorD4PG, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(obs_size, 400), nn.ReLU(),
                nn.Linear(400, 300), nn.ReLU(),
                nn.Linear(300, act_size), nn.Tanh(),
            )
        
    def forward(self, x):
        return self.net(x)

class CriticD4PG(nn.Module):
    def __init__(self, obs_size, act_size, n_atoms, v_min, v_max):
        super(CriticD4PG, self).__init__()
        self.obs_net = nn.Sequential(
                nn.Linear(obs_size, 400), nn.ReLU(),
            )
        self.qdist_net = nn.Sequential(
                nn.Linear(400+act_size, 300), nn.ReLU(),
                nn.Linear(300, n_atoms)
            )
        delta_z = (v_max - v_min) / (n_atoms - 1)
        self.register_buffer("supports", \
                torch.arange(v_min, v_max+delta_z, delta_z))
    
    def forward(self, x, a):
        obs_out = self.obs_net(x)
        qdist_in = torch.cat([obs_out, a], dim=1)
        return self.qdist_net(qdist_in)
      
    def dist2qval(self, dist):
        probs = F.softmax(dist, dim=1)
        weighted_qvals = probs * self.supports
        expected_qval = weighted_qvals.sum(dim=1)
        return expected_qval.unsqueeze(dim=-1)
    
class AgentD4PG:
    def __init__(self, net, device='cpu', epsilon=0.3):
        self.net = net
        self.device = device 
        self.epsilon = epsilon 
        
    def initial_state(self):
        return None 

    def __call__(self, states, agent_states):
        states_np = np.array(states, dtype=np.float32)
        states_v = torch.tensor(states_np).to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()
        actions += self.epsilon * np.random.normal(size=actions.shape)
        actions = np.clip(actions, -1, 1)
        return actions, agent_states

class TargetNet:
    def __init__(self, model):
        self.model = copy.deepcopy(model)
        
    def hard_sync(self, model):
        self.model.load_state_dict(model.state_dict())
        
    def soft_sync(self, model, tau=0.99):
        new_state_dict = model.state_dict()
        target_state_dict = self.model.state_dict()
        for k, v in new_state_dict.items():
            target_state_dict[k] = tau*target_state_dict[k] + (1-tau)*v 
        self.model.load_state_dict(target_state_dict)

def dist_projection(next_dists_v, rewards_v, dones_mask_v, gamma, \
                    n_atoms=N_ATOMS, vmin=Vmin, vmax=Vmax, device='cpu'):
    # Z(s, a) = R(s, a) + gamma * Z(s', a')
    # Z(s', a') stands for next_dist
    next_dists = next_dists_v.data.cpu().numpy()
    rewards = rewards_v.data.cpu().numpy()
    dones_mask = dones_mask_v.data.cpu().numpy().astype(np.bool)
    batch_size = len(rewards)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    Zsa = np.zeros([batch_size, n_atoms], dtype=np.float32)
    
    for j in range(n_atoms):
        TZj = rewards + gamma * (vmin + j * delta_z)
        TZj = np.minimum(vmax, np.maximum(vmin, TZj))
        bj = (TZj - vmin) / delta_z
        l = np.floor(bj).astype(np.int)
        u = np.ceil(bj).astype(np.int)
        eq_mask = l==u 
        Zsa[eq_mask, l[eq_mask]] += next_dists[eq_mask, j]
        ne_mask = l!=u 
        Zsa[ne_mask, l[ne_mask]] += next_dists[ne_mask, j] * (u-bj)[ne_mask]
        Zsa[ne_mask, u[ne_mask]] += next_dists[ne_mask, j] * (bj-l)[ne_mask]
    if dones_mask.any():
        Zsa[dones_mask] = 0.
        TZj = rewards[dones_mask]
        TZj = np.minimum(vmax, np.maximum(vmin, TZj))
        bj = (TZj - vmin) / delta_z
        l = np.floor(bj).astype(np.int)
        u = np.ceil(bj).astype(np.int)
        eq_mask = l==u 
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask 
        if eq_dones.any():
            Zsa[eq_dones, l[eq_mask]] = 1.0
        ne_mask = l!=u 
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
        if ne_dones.any():
            Zsa[ne_dones, l[ne_mask]] = (u-bj)[ne_mask]
            Zsa[ne_dones, u[ne_mask]] = (bj-l)[ne_mask]
    return torch.FloatTensor(Zsa).to(device)

def unpack_batch_ddqn(batch, device='cpu'):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = ptan.agent.float32_preprocessor(actions).to(device)
    rewards_v = ptan.agent.float32_preprocessor(rewards).to(device)
    last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
    dones_v = torch.ByteTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_v, last_states_v

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')

    save_path = os.path.join('saves', 'd4pg-'+ENV_ID)
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(comment='-d4pg-'+ENV_ID)
    
    env = gym.make(ENV_ID)
    actor_net = ActorD4PG(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    critic_net = CriticD4PG(env.observation_space.shape[0], env.action_space.shape[0], \
                    N_ATOMS, Vmin, Vmax).to(device)
    target_actor = TargetNet(actor_net)
    target_critic = TargetNet(critic_net)
    agent = AgentD4PG(actor_net, device=device)
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=LEARNING_RATE)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=LEARNING_RATE)    
    
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    
    step_idx = 0 
    best_reward = None 
    with ptan.common.utils.RewardTracker(writer) as tracker:
        while True:
            step_idx += 1
            buffer.populate(1)
            rewards_steps = exp_source.pop_rewards_steps()
            if rewards_steps:
                rewards, steps = zip(*rewards_steps)
                tracker.reward(rewards[0], step_idx)
            if len(buffer) < REPLAY_INITIAL:
                continue
            
            batch = buffer.sample(BATCH_SIZE)
            states_v, actions_v, rewards_v, dones_mask_v, next_states_v \
                    = unpack_batch_ddqn(batch, device)
            
            # Train Critic
            critic_optimizer.zero_grad()
            logits_v = critic_net(states_v, actions_v)
            probs_v = F.
            next_actions_v = target_actor.model(next_states_v)
            next_logits_v = target_critic(next_states_v, next_actions_v)
            next_dists_v = F.softmax(next_logits_v) # Z(s', a')
            Z_v = dist_projection(next_dists_v, rewards_v, dones_mask_v, \
                        gamma=GAMMA**REWARD_STEPS, n_atoms=N_ATOMS, vmin=Vmin, vmax=Vmax, device=device)
            


































