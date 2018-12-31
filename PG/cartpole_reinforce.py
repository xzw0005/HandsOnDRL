'''
Created on Dec 29, 2018

@author: wangxing
'''
import gym
import numpy as np
from tensorboardX import SummaryWriter 

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import ptan 

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4

def default_states_preprocessor(states):
    if len(states) == 1:
        np_states = np.expand_dims(states[0], axis=0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states) 

def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)

class ProbabilisticActionSelector:
    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        actions = []
        for p in probs:
            actions.append(np.random.choice(len(p), p=p))
        return np.array(actions)

class PolicyAgent:
    def initial_state(self):
        return None
    
    def __init__(self, model, action_selector=ProbabilisticActionSelector(), device='cpu',
                 apply_softmax=False, preprocessor=default_states_preprocessor):
        self.model = model 
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor
        
    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        probs_v = self.model(states)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        return np.array(actions), agent_states

class PGN(nn.Module):
    def __init__(self, input_size, num_actions):
        super(PGN, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, num_actions)
            )
    def forward(self, x):
        return self.net(x)
    
def calc_qvals(rewards, reward_to_go=True, baseline=False):
    res = []
    q = 0.0
    for r in reversed(rewards): 
        q = GAMMA * q + r 
        res.append(q)
    if reward_to_go:
        res.reverse()
    else:
        res = [q for _ in range(len(res))]
    if baseline:
        mu = np.mean(res)
        sigma = np.std(res)
        res = (res - mu) / sigma
    return res

if __name__=='__main__':
    env = gym.make('CartPole-v0')
    writer = SummaryWriter(comment='-CartPole-REINFORCE')
    
    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)
    
    agent = PolicyAgent(net, preprocessor=float32_preprocessor, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    
    optimizer = optim.Adam(net.parameters(), lr= LEARNING_RATE)
    
    total_rewards = []
    step_idx = 0 
    done_episodes = 0
    
    batch_episodes = 0 
    batch_states, batch_actions, batch_qvals = [], [], []
    cur_rewards = [] 
    
    for step_idx, exp in enumerate(exp_source):
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        cur_rewards.append(exp.reward)
        
        if exp.last_state is None:
            batch_qvals.extend(calc_qvals(cur_rewards))
            cur_rewards.clear()
            batch_episodes += 1 
            
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d"%(
                step_idx, reward, mean_rewards, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!"%(step_idx, done_episodes))
                break 
            
        if batch_episodes < EPISODES_TO_TRAIN:
            continue
        
        optimizer.zero_grad()
        batch_states_v = torch.FloatTensor(batch_states)
        batch_actions_v = torch.LongTensor(batch_actions)
        batch_rets_v = torch.FloatTensor(batch_qvals)
        
        logits = net(batch_states_v)
        log_pi = F.log_softmax(logits, dim=1)
    
        weighted_log_pi = batch_rets_v * log_pi[range(len(batch_states)), batch_actions_v]
        loss = -weighted_log_pi.mean()
        
        loss.backward()
        optimizer.step()
        
        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()
        
    writer.close()