'''
Created on Dec 25, 2018

@author: wangxing
'''
import gym
import ptan
import argparse
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import PARAMS
import common
import utils.agent as agent
import utils.actions as actions

REWARD_STEPS_DEFAULT = 1
STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        
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
        self.fc = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )
        
    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)
    
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

def eval_state_vals(states, net, device='cpu'):
    mean_vals = []
    for batch in np.array_split(states, 64):
        s_v = torch.tensor(batch).to(device)
        Qs = net(s_v)
        Qstar = Qs.max(1)[0]
        mean_vals.append(Qstar.mean().item())
    return np.mean(mean_vals)

if __name__=='__main__':
    params = PARAMS.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
    parser.add_argument('-n', default=REWARD_STEPS_DEFAULT, type=int, help='Number of steps unrolling Bellman')
    parser.add_argument('--double', default=False, action='store_true', help='Enable double DQN')
    
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)
    
    comment_str = '-'+params['run_name']
    if args.double:
        comment_str += '-double'
    if args.n:
        comment_str += '-%d-step'%args.n
    writer = SummaryWriter(comment=comment_str)
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
#     target_net = ptan.agent.TargetNet(net)
    target_net = agent.TargetNet(net)
#     selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    selector = actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
#     agent = ptan.agent.DQNAgent(net, selector, device=device)
    agent = agent.DQNAgent(net, selector, device=device)
    
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, \
                                gamma=params['gamma'], steps_count=args.n)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])
    
    frame_idx = 0
    eval_states = None
    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)
            
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break
                
            if len(buffer) < params['replay_initial']:
                continue
            if eval_states is None:
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)
            
            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            loss = calc_loss_dqn(batch, net, target_net.target_model, \
                    gamma=params['gamma']**args.n, double=args.double, device=device)
            loss.backward()
            optimizer.step()
            
            if frame_idx % params['target_net_sync'] == 0:
                target_net.sync()   
                     
            if frame_idx % EVAL_EVERY_FRAME == 0:
                mean_val = eval_state_vals(eval_states, net, device=device)
                writer.add_scalar('mean values', mean_val, frame_idx)