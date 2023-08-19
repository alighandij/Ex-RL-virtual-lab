from gym import Env
from types import FunctionType


class GymWrapper:
    def __init__(self, env: Env, max_episode_step: int, is_goal_reached_fn: FunctionType) -> None:
        self.env = env
        self.episode_step = 0
        self.action_space = self.env.action_space
        self.max_episode_step = max_episode_step
        self.observation_space = self.env.observation_space
        self.is_goal_reached_fn = is_goal_reached_fn

    def reset(self):
        self.episode_step = 0
        s, _ = self.env.reset()
        return s

    def render(self):
        self.env.render()

    def step(self, action):
        self.episode_step += 1
        state, reward, done, info = self.env.step(action)[:4]
        done = (done) or (self.episode_step >= self.max_episode_step)
        return state, reward, done, info

    def is_goal_reached(self):
        return self.is_goal_reached_fn(self)

    def close(self):
        self.env.close()