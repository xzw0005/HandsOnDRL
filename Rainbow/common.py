'''
Created on Dec 25, 2018

@author: wangxing
'''
import sys
import time 
import numpy as np
import torch
import torch.nn as nn

class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']
        self.frame(0)
    
    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = \
            max(self.epsilon_final, self.epsilon_start-frame/self.epsilon_frames)

class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer 
        self.stop_reward = stop_reward
        
    def __enter__(self):
        self.tic = time.time()
        self.tic_frame = 0
        self.total_rewards = []
        return self 
    
    def __exit__(self, *args):
        self.writer.close()
        
    def reward(self, total_reward, frame, epsilon=None):
        self.total_rewards.append(total_reward)
        fps = (frame - self.tic_frame) / (time.time() - self.tic)
        self.tic_frame = frame 
        self.tic = time.time()
        mean_reward_last100 = np.mean(self.total_rewards[-100:])
        eps_str = "" if epsilon is None else ", eps %.2f"%epsilon
        print("%d: done %d games, mean episodic reward %.3f, speed %.2f f/s%s"\
              %(frame, len(self.total_rewards), mean_reward_last100, fps, eps_str))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("fps", fps, frame)
        self.writer.add_scalar("mean reward of last 100 episode", mean_reward_last100, frame)
        self.writer.add_scalar("episodic reward", total_reward, frame)
        if mean_reward_last100 > self.stop_reward:
            print("Solved in %d frames!"%frame)
            return True 
        return False
        
def unpack_batch(batch):
    state_batch, action_batch, reward_batch, done_batch, last_state_batch = [],[],[],[],[]
    for experience in batch:
        state = np.array(experience.state, copy=False)
        state_batch.append(state)
        action_batch.append(experience.action)
        reward_batch.append(experience.reward)
#         done_batch.append(experience.done)
        done_batch.append(experience.last_state is None)

        if experience.last_state is None:
            last_state_batch.append(state) 
        else:
            last_state = np.array(experience.last_state, copy=False)
            last_state_batch.append(last_state)
    state_batch = np.array(state_batch, copy=False)
    action_batch = np.array(action_batch)
    reward_batch = np.array(reward_batch, dtype=np.float32)
    done_batch = np.array(done_batch, dtype=np.uint8)
    last_state_batch = np.array(last_state_batch, copy=False)
    return state_batch, action_batch, reward_batch, done_batch, last_state_batch

def calc_loss_dqn(batch, net, target_net, gamma, device='cpu', double=True):
    sb, ab, rb, db, spb = unpack_batch(batch)
    states = torch.tensor(sb).to(device)
    actions = torch.tensor(ab).to(device)
    rewards = torch.tensor(rb).to(device)
    next_states = torch.tensor(spb).to(device)
    done_mask = torch.ByteTensor(db).to(device)
    
    Qs = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    if double:
        actions_sp = net(next_states).max(1)[1]
        Qsp = target_net(next_states).gather(1, actions_sp.unsqueeze(-1)).squeeze(-1)
    else:
        Qsp = target_net(next_states).max(1)[0]
    Qsp[done_mask] = 0.
    Qs_ = rewards + gamma * Qsp.detach()
    loss = nn.MSELoss()(Qs, Qs_)
    return loss
        
        
        
        
        
        