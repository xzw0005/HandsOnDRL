'''
Created on Jan 8, 2019

@author: wangxing
'''

import gym
import pybullet_envs

import numpy as np
import copy
import argparse
import os
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ptan

# ENV_ID = "MinitaurBulletEnv-v0"
ENV_ID = 'Pendulum-v0'
GAMMA = 0.99
REWARD_STEPS = 2
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
REPLAY_SIZE = int(1e5)
REPLAY_INITIAL = int(1e4)
TEST_ITERS = 1000


class ActorNet(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ActorNet, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(obs_size, 400), nn.ReLU(),
                nn.Linear(400, 300), nn.ReLU(),
                nn.Linear(300, act_size), nn.Tanh()
            )
    def forward(self, x):
        return self.net(x)
    
class CriticNet(nn.Module):
    def __init__(self, obs_size, act_size):
        super(CriticNet, self).__init__()
        self.obs_net = nn.Sequential(
                nn.Linear(obs_size, 512), nn.ReLU(),
            )
        self.qval_net = nn.Sequential(
                nn.Linear(400+act_size, 300), nn.ReLU(),
                nn.Linear(300, 1)
            )
    def forward(self, x, act):
        obs_out = self.obs_net(x)
        qval_input = torch.cat([obs_out, act], dim=1)
        return self.qval_net(qval_input)
    
class TargetNet:
    def __init__(self, model):
        self.model = copy.deepcopy(model)
        
    def sync(self, model):
        self.model.load_state_dict(model.state_dict())
        
    def soft_sync(self, model, tau):
        state_dict = model.state_dict()
        target_state_dict = self.model.state_dict()
        for k, v in state_dict.items():
            target_state_dict[k] = tau * target_state_dict[k] + (1-tau) * v 
        self.model.load_state_dict(target_state_dict) 

class AgentDDPG:
    def __init__(self, net, device='cpu', ou_enabled=True,\
                 ou_mu=0., ou_theta=0.15, ou_sigma=0.2, ou_epsilon=1.):
        self.net = net 
        self.device = device 
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_theta = ou_theta 
        self.ou_sigma = ou_sigma 
        self.ou_epsilon = ou_epsilon
        
    def initial_state(self):
        return None 
    
    def __call__(self, states, agent_states):
        states_np = np.array(states, dtype=np.float32)
        states_v = torch.tensor(states_np).to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()   # Deterministic Actions
        ## OU process SDE: dx_t = theta * (mu - x_t) dt + sigma dW
        ## in discrete time: x_{t+1} = x_t + theta * (mu - x_t) + sigma * N(ou_mu, ou_sigma)
        if self.ou_enabled and self.ou_epsilon > 0:
            new_agent_states = []
            for agent_state, action in zip(agent_states, actions):
                if agent_state is None:
                    agent_state = np.zeros(shape=action.shape, dtype=np.float32)
                agent_state += self.ou_theta * (self.ou_mu - agent_state)
                agent_state += self.ou_sigma * np.random.normal(size=action.shape)
                action += self.ou_epsilon * agent_state
                new_agent_states.append(agent_state)
        else:
            new_agent_states = agent_states
        actions = np.clip(actions, -1, 1)
        return actions, new_agent_states
        
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
    dones_t = torch.ByteTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
    
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    save_path = os.path.join("saves", "ddpg-" + ENV_ID)
    os.makedirs(save_path, exist_ok=True)
    
    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID)
    
    actor_net = ActorNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    critic_net = CriticNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    target_actor = TargetNet(actor_net)
    target_critic = TargetNet(critic_net)
    
    writer = SummaryWriter(comment='-ddpg-'+ENV_ID)
    agent = AgentDDPG(actor_net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=LEARNING_RATE)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=LEARNING_RATE)
    
    frame_idx = 0
    best_rew = None 
    with ptan.common.utils.RewardTracker(writer) as tracker:
        while True:
            frame_idx += 1 
            buffer.populate(1)
            rewards_steps = exp_source.pop_rewards_steps()
            if rewards_steps:
                rewards, steps = zip(*rewards_steps)
                tracker.reward(rewards[0], frame_idx)
            if len(buffer) < REPLAY_INITIAL:
                continue
            batch = buffer.sample(batch_size=BATCH_SIZE)
            states_v, actions_v, rewards_v, dones_mask, next_states_v = unpack_batch_ddqn(batch, device)
            
            # Train Critic
            Qs_v = critic_net(states_v, actions_v)
            next_actions_v = target_actor.model(next_states_v).detach()
            Qsp_v = target_critic.model(next_states_v, next_actions_v).detach()
            Qsp_v[dones_mask] = 0.
            Qs_target = rewards_v.unsqueeze(dim=-1) + GAMMA * Qsp_v
            
            critic_optimizer.zero_grad()
            loss_critic = F.mse_loss(input=Qs_v, target=Qs_target)
            loss_critic.backward()
            critic_optimizer.step()
            
            # Train Actor
            actor_optimizer.zero_grad()
            actions_pred = actor_net(states_v)
            loss_actor = -critic_net(states_v, actions_pred).mean()
            loss_actor.backward()
            actor_optimizer.step()
            
            target_actor.soft_sync(actor_net, tau=0.999)
            target_critic.soft_sync(critic_net, tau=0.999)
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
        