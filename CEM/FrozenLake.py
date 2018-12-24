'''
Created on Dec 23, 2018

@author: wangxing
'''
import gym, gym.spaces
import gym.envs.toy_text.frozen_lake
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 128
BATCH_SIZE = 128
PERCENTILE = 30
LEARNING_RATE = 0.001   
GAMMA = 0.9

class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.observation_space = gym.spaces.Box(0., 1., (env.observation_space.n, ), dtype=np.float32)
        
    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res
    
class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, num_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_actions)
            )
        
    def forward(self, x):
        return self.net(x)
    
Episode = namedtuple(typename='Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple(typename='EpisodeStep', field_names=['observation', 'action'])

def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        obs_v = torch.tensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, done, _ = env.step(action)
        episode_reward += reward 
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch 
                batch = []
        obs = next_obs
        
def filter_batch(batch, percentile):
#     rewards = list(map(lambda s: s.reward, batch))
    rewards = list(map(lambda s: s.reward * GAMMA**len(s.steps), batch))
    reward_bound = np.percentile(rewards, percentile)
#     reward_mean = float(np.mean(rewards))
    
    train_obs = []
    train_act = []
    elite_batch = []
    for example, episode_reward in zip(batch, rewards):
        if episode_reward > reward_bound:
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))
            elite_batch.append(example)
    return elite_batch, train_obs, train_act, reward_bound
#     train_obs_v = torch.tensor(train_obs)
#     train_act_v = torch.LongTensor(train_act)
#     return train_obs_v, train_act_v, reward_bound, reward_mean

if __name__=="__main__":
#     env = DiscreteOneHotWrapper(gym.make('FrozenLake-v0'))
    env = gym.envs.toy_text.frozen_lake.FrozenLakeEnv(is_slippery=False)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
    env = DiscreteOneHotWrapper(env)
    
    obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    net = Net(obs_size, HIDDEN_SIZE, num_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(comment='-frozenlake-naive')
    
    full_batch = []
    for i, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
        full_batch, obs_batch, act_batch, reward_bound = filter_batch(full_batch+batch, PERCENTILE)
        if not full_batch:
            continue
        obs_batch = torch.tensor(obs_batch)
        act_batch = torch.LongTensor(act_batch)
        full_batch = full_batch[-500:]
        
        optimizer.zero_grad()
        action_scores = net(obs_batch)
        loss = objective(action_scores, act_batch)
        loss.backward()
        optimizer.step()
        print("%d: loss=%.3f, mean reward=%.1f, reward bound=%.1f"%(i, loss.item(), reward_mean, reward_bound))
        writer.add_scalar(tag='loss', scalar_value=loss.item(), global_step=i)
        writer.add_scalar(tag='mean reward', scalar_value=reward_mean, global_step=i)
        writer.add_scalar(tag="reward bound", scalar_value=reward_bound, global_step=i)
        if reward_mean > 0.8:
            print("Solved!")
            break
    writer.close()
    
    
    
    
    
    
    
    
    
    
    