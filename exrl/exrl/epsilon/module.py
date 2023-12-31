import numpy as np

class Epsilon:
    def __init__(self, epsilon=1.0,  decay=0.999, min_epsilon=1e-3):
        self.decay = decay
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
    
    def _rand(self):
        return np.random.random()
    
    def take_action(self, state):
        if self.rand() < self.epsilon:
            return self.env.action_space.sample()
        return self.get_action(state)
    
    def is_explore(self):
        return self._rand() < self.epsilon
    

    def decrease(self):
        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)