'''
Created on Dec 23, 2018

@author: wangxing
'''
import wrappers
import dqn_model

import argparse
import time
import numpy as np
import collections 

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter 

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02


Experience = collections.namedtuple('Experience', 
            field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        
    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        sb, ab, rb, doneb, spb = zip(*[self.buffer[idx] for idx in indices])
        return np.array(sb), np.array(ab), np.array(rb, dtype=np.float32), \
            np.array(doneb, dtype=np.uint8), np.array(spb)

class Agent(object):
    def __init__(self, env, replay_buffer):
        self.env = env 
        self.replay_buffer = replay_buffer
        self._reset()
        
    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.
        
    def play_step(self, net, epsilon=0.0, device='cpu'):
        done_reward = None
        if np.random.random() < epsilon:
            a = self.env.action_space.sample()
        else:
            s = np.array([self.state], copy=False)
            s = torch.tensor(s).to(device)
            q_vals = net(s)
            _, a = torch.max(q_vals, dim=1)
            a = int(a.item())
        sp, r, done, _ = self.env.step(a)
        self.total_reward += r 
        experience = Experience(self.state, a, r, done, sp)
        self.replay_buffer.append(experience)
        self.state = sp 
        if done:
            done_reward = self.total_reward
            self._reset()
        return done_reward
    
def calc_loss(batch, net, target_net, device='cpu'):
    s_batch, a_batch, r_batch, done_batch, sp_batch = batch 
    states_v = torch.tensor(s_batch).to(device)
    actions_v = torch.tensor(a_batch).to(device)
    rewards_v = torch.tensor(r_batch).to(device)
    next_states_v = torch.tensor(sp_batch).to(device)
    done_mask = torch.ByteTensor(done_batch).to(device)
    
    Qs = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    
    Qsp = target_net(next_states_v).max(1)[0]
    Qsp[done_mask] = 0.
    Qsp = Qsp.detach()
    Qtarget = rewards_v + GAMMA * Qsp 
    return nn.MSELoss()(Qs, Qtarget)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
    parser.add_argument('--env', default=DEFAULT_ENV_NAME)
    parser.add_argument('--reward', type=float, default=MEAN_REWARD_BOUND, \
                        help='Mean reward bound for stop of training')
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    env = wrappers.make_env(args.env)
    
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment='-'+args.env)
    print(net)
    
    buffer = ReplayBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START
    
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0 
    ts = time.time()
    best_mean_reward = None 
    
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START-frame_idx/EPSILON_DECAY_LAST_FRAME)
        reward = agent.play_step(net, epsilon, device)  # every iter is a single step
        if reward is not None:  # AKA if done (end of an episode)
            total_rewards.append(reward)
            speed = (frame_idx-ts_frame) / (time.time()-ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" \
                  %(frame_idx, len(total_rewards), mean_reward, epsilon, speed) )
            writer.add_scalar('epsilon', epsilon, frame_idx)
            writer.add_scalar('speed', speed, frame_idx)
            writer.add_scalar('mean_reward_100', mean_reward, frame_idx)
            writer.add_scalar('reward', reward, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.env+'-best.dat')  # save model parameters
                if best_mean_reward is not None:
                    print('Best mean reward updated %.3f --> %.3f, model saved'\
                          (best_mean_reward, mean_reward))
                    best_mean_reward = mean_reward
            if mean_reward > args.reward:   # Stopping Criterion
                print('Solved in %d frames!'%frame_idx)
                break
            
        if len(buffer) < REPLAY_START_SIZE:
            continue
        
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            target_net.load_state_dict(net.state_dict())
            
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss = calc_loss(batch, net, target_net, device)
        loss.backward()
        optimizer.step()
        
    writer.close()
        