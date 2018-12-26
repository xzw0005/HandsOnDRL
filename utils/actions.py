'''
Created on Dec 25, 2018
@author: wangxing
'''
import numpy as np

class ActionSelector:
    ''' Abstract class '''
    def __call__(self, Qs):
        raise NotImplementedError
    
class ArgmaxActionSelector(ActionSelector):
    def __call__(self, Qs):
        assert isinstance(Qs, np.ndarray)
        return np.argmax(Qs, axis=1)
        
class EpsilonGreedyActionSelector(ActionSelector):
    def __init__(self, epsilon=0.05, selector=None):
        self.epsilon = epsilon
        self.selector = selector if selector is not None else ArgmaxActionSelector()
        
    def __call__(self, Qs):
        assert isinstance(Qs, np.ndarray)
        batch_size, num_actions = Qs.shape 
        actions = self.selector(Qs)
        mask = np.random.random(size=batch_size) < self.epsilon
        random_actions = np.random.choice(num_actions, sum(mask))
        actions[mask] = random_actions
        return actions 
    
class ProbabilityActionSelector(ActionSelector):
    """ Sampling actions according to policy distribution """
    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        actions = [np.random.choice(len(pi), p=pi) for pi in probs]
        return np.array(actions)