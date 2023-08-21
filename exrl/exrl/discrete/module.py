import numpy as np


class Discrete:
    def __init__(self, env, size: tuple):
        Discrete._assert_shape(env, size)
        self.env = env
        self.low = env.observation_space.low
        self.high = env.observation_space.high
        self.size = np.array(size)

        self.window_size = (self.high - self.low) / self.size

    @staticmethod
    def _assert_shape(env, size):
        msg = "size.shape != observation_space.shape"
        assert env.observation_space.shape == np.array(size).shape, msg

    def __call__(self, state):
        discrete = (state - self.low) / self.window_size
        discrete = discrete.astype(int)
        discrete = np.min((discrete, self.size), axis=0)
        return tuple(discrete)

    def __str__(self):
        return f"""
        \rLOW = {self.low}
        \rHIGH = {self.high}
        \rSIZE = {self.size}
        \rWINDOW = {self.window_size}
        """.strip()

    def generate_table(self, low=-1, high=0):
        return np.random.uniform(
            low=low, high=high, size=([*(self.size + 1)] + [self.env.action_space.n])
        )
