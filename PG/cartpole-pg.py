'''
Created on Dec 30, 2018

@author: wangxing
'''
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from tensorboardX import SummaryWriter 

import ptan

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 8
REWARD_STEPS = 10

def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)

class PgAgent:
    def __init__(self, model, device='cpu', 
                 apply_softmax=False, preprocessor=None):
        self.model = model 
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
        probs = self.model(states)
        if self.apply_softmax:
            probs = F.softmax(probs, dim=1)
        probs = probs.data.cpu().numpy()
        ## Below is the ProbabilisticActionSelection process
        actions = []
        for p in probs:
            actions.append(np.random.choice(len(p), p=p))
        return np.array(actions), agent_states
    
    def initial_state(self):
        return None

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
    
if __name__=='__main__':
    env = gym.make('CartPole-v0')
    writer = SummaryWriter(comment='-carpole-PG')
    
    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)
    
    agent = PgAgent(net, preprocessor=float32_preprocessor, \
                    apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, \
                    agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    total_rewards = []
    step_rewards = []
    step_idx = 0 
    done_episodes = 0
    reward_sum = 0.
    
    batch_states, batch_actions, batch_scales = [], [], []
    
    for step_idx, exp in enumerate(exp_source):
        reward_sum += exp.reward 
        baseline = reward_sum / (step_idx + 1)
        
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        batch_scales.append(exp.reward - baseline)
        
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

        if len(batch_states) < BATCH_SIZE:
            continue
        
        batch_states_v = torch.FloatTensor(batch_states)
        batch_actions_v = torch.LongTensor(batch_actions)
        batch_scales_v = torch.FloatTensor(batch_scales)
        
        optimizer.zero_grad()
        logits = net(batch_states_v)
        log_pi = F.log_softmax(logits, dim=1)
        weighted_log_pi = batch_scales_v * log_pi[range(BATCH_SIZE), batch_actions_v]
        loss_policy = -weighted_log_pi.mean()
        loss_policy.backward(retain_graph=True)
        
        pi = F.softmax(logits, dim=1)
        entropy = (-pi * log_pi).sum(dim=1).mean()
        loss_entropy = -ENTROPY_BETA * entropy
        loss_entropy.backward()
        
        loss = loss_policy + loss_entropy
#         loss.backward()
        optimizer.step()
        
        ## Calc KL-divergence
        new_logits = net(batch_states_v)
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
            
        writer.add_scalar("baseline", baseline, step_idx)
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
        
        
        
        
        
        
        
        