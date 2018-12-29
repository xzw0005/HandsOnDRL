'''
Created on Dec 25, 2018

@author: wangxing
'''
import numpy as np
import torch
import copy

class BaseAgent:
    '''Abstract Agent Interface'''
    def initial_state(self):
        return None
    def __call__(self, states, agent_states):
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(states) == len(agent_states)
        
        raise NotImplementedError
    
def default_states_preprocessor(states):
    if len(states) == 1:
        np_states = np.expand_dims(states[0], axis=0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states)

def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)

class DQNAgent(BaseAgent):
    """ DQNAgent maps observations to actions, 
    first use dqn_model ot calculate Q-values,
    then use action_selector to pick actions"""
    def __init__(self, dqn_model, action_selector, device='cpu', preprocessor=default_states_preprocessor):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device 
    
    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        Qs = self.dqn_model(states)
        Qs = Qs.data.cpu().numpy()
        actions = self.action_selector(Qs)
        return actions, agent_states
    
class TargetNet:
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def alpha_sync(self, alpha):
        assert isinstance(alpha, float)
        assert 0. < alpha <= 1.
        state = self.model.state_dict()
        target_state = self.target_model.state_dict()
        for k, v in state.items():
            target_state[k] = target_state[k] * alpha + (1 - alpha) * v 
        self.target_model.load_state_dict(target_state)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            