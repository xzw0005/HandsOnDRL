'''
Created on Dec 23, 2018
 
@author: wangxing
'''
import gym
import collections
from tensorboardX import SummaryWriter
 
ENV_NAME = 'FrozenLake-v0'
GAMMA = 0.9
ALPHA = 0.2
EPISODES = 20
 
class Agent(object):
 
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)
         
    def sample_env(self):
        a = self.env.action_space.sample()
        s = self.state 
        sp, r, done, _ = self.env.step(a)
        self.state = self.env.reset() if done else sp
        return (s, a, r, sp)
     
    def best_value_and_action(self, s):
        best_value, best_action = None, None
        for a in range(self.env.action_space.n):
            q = self.values[(s, a)]
            if best_value is None or best_value < q:
                best_value = q 
                best_action = a 
        return best_value, best_action
     
    def value_update(self, s, a, r, sp):
        best_value, _ = self.best_value_and_action(sp)
        new_val = r + GAMMA * best_value
        old_val = self.values[(s, a)]
        self.values[(s, a)] = ALPHA * new_val + (1-ALPHA) * old_val
         
    def play_episode(self, env):
        total_reward = 0.0
        s = env.reset()
        while True:
            _, a = self.best_value_and_action(s)
            sp, r, done, _ = env.step(a)
            total_reward += r 
            if done:
                break
            s = sp
        return total_reward
     
if __name__=='__main__':
    env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment='-qlearn')
     
    i = 0 
    best_total_reward = 0.0
    while True:
        i += 1
        s, a, r, sp = agent.sample_env()
        agent.value_update(s, a, r, sp)
         
        episode_reward = 0.0
        for _ in range(EPISODES):
            episode_reward += agent.play_episode(env)
        episode_reward /= EPISODES
        writer.add_scalar("episode_reward", episode_reward, i)
        if episode_reward > best_total_reward:
            print("Best episodic reward updated %.3f ==> %.3f"%(best_total_reward, episode_reward))
            best_total_reward = episode_reward
        if episode_reward > 0.8:
            print('Solved in %d iterations'%i)
            break
    writer.close()