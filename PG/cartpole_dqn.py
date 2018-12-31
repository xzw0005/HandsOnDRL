'''
Created on Dec 29, 2018

@author: wangxing
'''
import gym
import numpy as np

import torch 
import torch.nn as nn 
import torch.optim as optim
from tensorboardX import SummaryWriter 

import ptan
GAMMA = 0.99
LEARNING_RATE = 0.01
BATCH_SIZE = 8
EPSILON_START = 1.
EPSILON_END = 0.02
EPSILON_STEPS = 5000
REPLAY_BUFFER = 50000

class DQN(nn.Module):
    def __init__(self, input_size, num_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(input_size, 128), 
                nn.ReLU(),
                nn.Linear(128, num_actions)
            )
    def forward(self, x):
        return self.net(x)
    
def calc_target(net, local_reward, next_state):
    if next_state is None:
        return local_reward
    next_state_v = torch.tensor([next_state], dtype=torch.float32)
    next_Q_v = net(next_state_v)
    bestQ = next_Q_v.max(dim=1)[0].item()
    return local_reward + GAMMA * bestQ

if __name__=='__main__':
    env = gym.make('CartPole-v0')
    writer = SummaryWriter(comment='-cartpole-dqn')
    
    net = DQN(env.observation_space.shape[0], env.action_space.n)
    print(net)
    
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=EPSILON_START)
    agent = ptan.agent.DQNAgent(net, selector, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    replay_buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_BUFFER)
    
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    total_rewards = []
    step_idx = 0
    done_episodes = 0 
    
    while True:
        step_idx += 1
        selector.epsilon = max(EPSILON_END, EPSILON_START-step_idx/EPSILON_STEPS)
        replay_buffer.populate(1)
        
        if len(replay_buffer) < BATCH_SIZE:
            continue
        
        batch = replay_buffer.sample(BATCH_SIZE)
        batch_states = [exp.state for exp in batch]
        batch_actions = [exp.action for exp in batch]
        batch_targets = [calc_target(net, exp.reward, exp.last_state) for exp in batch]
        
        optimizer.zero_grad()
        states = torch.FloatTensor(batch_states)
        Q_v = net(states)
        targetQ = Q_v.data.numpy().copy()
        targetQ[range(BATCH_SIZE), batch_actions] = batch_targets
        targetQ_v = torch.tensor(targetQ)
        loss = nn.MSELoss()(Q_v, targetQ_v)
        loss.backward()
        optimizer.step()
        
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, epsilon: %.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, selector.epsilon, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("epsilon", selector.epsilon, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!"%(step_idx, done_episodes))
                break
    writer.close()   