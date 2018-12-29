'''
Created on Dec 28, 2018

@author: wangxing
'''
import gym
import numpy as np
import argparse

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from tensorboardX import SummaryWriter

import ptan
import common
import utils.agent as agent
import utils.actions as actions
import PARAMS

N_ATOMS = 51
Vmax = 10
Vmin = -10
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100

SAVE_STATES_IMG = False 
SAVE_TRANSITIONS_IMG = False
if SAVE_STATES_IMG or SAVE_TRANSITIONS_IMG:
    import matplotlib as mpl 
    mpl.use("Agg")
    import matplotlib.pylab as plt

class DistributionalDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DistributionalDQN, self).__init__()
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
                nn.Linear(512, num_actions * N_ATOMS)
            )
        self.register_buffer(name='supports', tensor=torch.arange(Vmin, Vmax+DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() /  256
        conv_out = self.conv(fx).view(batch_size, -1)
        fc_out = self.fc(conv_out)
        return fc_out.view(batch_size, -1, N_ATOMS)
    
    def both(self, x):
        distr = self(x)
        probs = self.apply_softmax(distr)
        weights = probs * self.supports
        Qvals = weights.sum(dim=2)
        return distr, Qvals
    
    def qvals(self, x):
        return self.both(x)[1]
    
    def apply_softmax(self, distr):
        return self.softmax(distr.view(-1, N_ATOMS)).view(distr.size())
    

def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    for j in range(n_atoms):
        Tzj = rewards + gamma * (Vmin + j * delta_z)
        Tzj = np.maximum(Vmin, Tzj)
        Tzj = np.minimum(Vmax, Tzj)
        bj = (Tzj - Vmin) / delta_z
        l = np.floor(bj).astype(np.int64)
        u = np.ceil(bj).astype(np.int64)
        eq_mask = (l==u)
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, j]
        ne_mask = (l != u)
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, j] * (u-bj)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, j] * (bj-l)[ne_mask]
    if dones.any():
        proj_distr[dones] = 0.
        Tzj = rewards[dones]
        Tzj = np.minimum(Vmax, np.maximum(Vmin, Tzj))
        bj = (Tzj - Vmin) / delta_z 
        l = np.floor(bj).astype(np.int64)
        u = np.ceil(bj).astype(np.int64)
        eq_mask = u==l 
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u!=l 
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u-bj)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (bj-l)[ne_mask]
    return proj_distr
        
def calc_loss(batch, net, target_net, gamma, device='cpu', save_prefix=None):
    batch_size = len(batch)
    sb, ab, rb, db, spb = common.unpack_batch(batch)
    
    states = torch.tensor(sb).to(device)
    actions = torch.tensor(ab).to(device)
#     rewards = torch.tensor(rb).to(device)
    next_states = torch.tensor(spb).to(device)
    dones = db.astype(np.bool)
    
    next_distr, next_qvals = target_net.both(next_states)
    next_actions = next_qvals.max(1)[1].data.cpu().numpy()
    next_probs = target_net.apply_softmax(next_distr).data.cpu().numpy()
    
    next_best_action_probs = next_probs[range(batch_size), next_actions]
    # project the distribution using Bellman update
    proj_distr = distr_projection(next_best_action_probs, rb, dones, Vmin, Vmax, N_ATOMS, gamma)
    proj_distr_v = torch.tensor(proj_distr).to(device)
    
    distr = net(states)
    qvals = distr[range(batch_size), actions.data]
    log_sm = F.log_softmax(qvals, dim=1)
    
    if save_prefix is not None:
        pred = F.softmax(qvals, dim=1).data.cpu().numpy()
        save_transition_images(batch_size, pred, proj_distr, next_best_action_probs, dones, rb, save_prefix)
    
    loss = (-proj_distr_v * log_sm).sum(dim=1)      # KL-divergence: -sum p*log(q)
    return loss.mean()
        

def eval_state_vals(states, net, device='cpu'):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        Qs = net(states_v)
        Qstar = Qs.max(1)[0]
        mean_vals.append(Qstar.mean().item())
    return np.mean(mean_vals)

def save_state_images(frame_idx, states, net, device='cpu', max_states=200):
    ofs = 0
    p = np.arange(Vmin, Vmax+DELTA_Z, DELTA_Z)
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_probs = net.apply_softmax(net(states_v)).data.cpu().numpy()
        batch_size, num_actions, _ = action_probs.shape 
        for i in range(batch_size):
            plt.clf()
            for a in range(num_actions):
                plt.subplot(num_actions, 1, a+1)
                plt.bar(p, action_probs[i, a], width=0.5)
            plt.savefig('states%05d_%08d.png'%(ofs+i, frame_idx))
        ofs += batch_size
        if ofs >= max_states:
            break
        
def save_transition_images(batch_size, predicted, projected, next_distr, dones, rewards, save_prefix):
    for i in range(batch_size):
        done = dones[i]
        r = rewards[i]
        plt.clf()
        p = np.arange(Vmin, Vmax+DELTA_Z, DELTA_Z)
        plt.subplot(3, 1, 1)
        plt.bar(p, predicted[i], width=0.5)
        plt.title("Predicted")
        plt.subplot(3, 1, 2)
        plt.bar(p, projected[i], width=0.5)
        plt.title("Projected")
        plt.subplot(3, 1, 3)
        plt.bar(p, next_distr[i], width=0.5)
        plt.title("Next state")
        suffix = ""
        if r != 0.:
            suffix += "_%.0f"%r 
        if done:
            suffix += '_done'
        plt.savefig('%s_%02d%s.png'%(save_prefix, i, suffix))
        
if __name__=='__main__':
    params = PARAMS.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
    
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
           
    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)

    comment_str = '-'+params['run_name']
    writer = SummaryWriter(comment=comment_str+'-distributional')
    
    net = DistributionalDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = agent.TargetNet(net)
    selector = actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start']) 
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = agent.DQNAgent(lambda x: net.qvals(x), selector, device=device)
    
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])
    
    frame_idx = 0 
    eval_states = None
    prev_save = 0 
    save_prefix = None 
    
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


            save_prefix = None
            if SAVE_TRANSITIONS_IMG:
                interesting = any(map(lambda s: s.last_state is None or s.reward != 0.0, batch))
                if interesting and frame_idx // 30000 > prev_save:
                    save_prefix = "images/img_%08d" % frame_idx
                    prev_save = frame_idx // 30000
                    
            
            loss = calc_loss(batch, net, target_net.target_model, \
                            gamma=params['gamma'], device=device, save_prefix=save_prefix)
            loss.backward()
            optimizer.step()
            
            if frame_idx % params['target_net_sync'] == 0:
                target_net.sync()
                
            if frame_idx % EVAL_EVERY_FRAME == 0:
                mean_val = eval_state_vals(eval_states, net, device=device)
                writer.add_scalar('mean_values', mean_val, frame_idx)
                
            if SAVE_STATES_IMG and frame_idx % 10000 == 0:
                save_state_images(frame_idx, eval_states, net, device=device)      
        