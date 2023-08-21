import numpy as np
from gym import Env
from .base import Agent
from types import FunctionType
from exrl.discrete import Discrete


class QLearning(Agent):
    def __init__(
        self,
        env: Env,
        discrete: tuple[float],
        gamma: float = 0.99,
        episodes: int = 20000,
        render_each: int = 100000,
        lr: float = 1,
        lr_min: float = 0.01,
        lr_decay: float = 0.999,
        epsilon: float = 1,
        epsilon_decay: float = 0.99,
        epsilon_min: float = 0.001,
        count: int = 20,
        target: float = -120,
        reward_shaper: FunctionType = None,
        break_on_solve: bool = True,
        _tqdm=None,
    ) -> None:
        super().__init__(
            env,
            "Normal Q-Learning",
            gamma,
            episodes,
            render_each,
            lr,
            lr_min,
            lr_decay,
            epsilon,
            epsilon_decay,
            epsilon_min,
            count,
            target,
            reward_shaper,
            break_on_solve,
            _tqdm
        )
        self.discrete = Discrete(env, discrete)
        self.q_table = self.discrete.generate_table()

    def greedy_action(self, state: np.ndarray):
        return np.argmax(self.q_table[self.discrete(state)])

    def get_result(self) -> dict:
        return {"q_table": self.q_table, **self.get_result_general()}

    def update_q_table(self, s: np.ndarray, a, r: float, sn: np.ndarray):
        i = self.discrete(s) + (a,)
        td_err = r
        if not self.env.is_goal_reached():
            max_q = np.max(self.q_table[self.discrete(sn)])
            td_err += self.gamma * max_q - self.q_table[i]
        self.q_table[i] += self.lr * td_err

    def train_episode(self, render: bool) -> tuple[float, np.ndarray]:
        done = False
        state = self.env.reset()
        observations = []
        total_reward = 0
        while not done:
            if render:
                self.env.render()
            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            reward = self.get_reward(state, next_state, reward, done)
            self.update_q_table(state, action, reward, next_state)
            observations.append((state, next_state))
            state = next_state
        return total_reward, np.array(observations)
