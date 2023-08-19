import numpy as np
from gym import Env
from .base import Agent
from typing import Callable
from collections import deque
from exrl.reward_shaper import RewardShaperHMM
from exrl.discrete import Discrete


class ExQLearning(Agent):
    def __init__(
        self,
        env: Env,
        phase_mapping: Callable,
        discrete: tuple[float],
        reward_shaper: RewardShaperHMM,
        discount_mc: float = 0.9,
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
        break_on_solve: bool = True,
    ) -> None:
        super().__init__(
            env,
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
        )
        self.hmm_step = self.reward_shaper.stp
        self.discrete = Discrete(env, discrete)
        self.q_table = self.discrete.generate_table()
        self.discount_mc = discount_mc
        self.phase_mapping = phase_mapping

        self.buffer = deque(maxlen=self.hmm_step)
        self.old_score = 0

    def get_result(self) -> dict:
        return {"q_table": self.q_table, **self.get_result_general()}

    def greedy_action(self, state: np.ndarray):
        return np.argmax(self.q_table[self.discrete(state)])

    def encode(self, s: np.ndarray, sn: np.ndarray) -> int:
        try:
            return self.phase_mapping(s, sn)
        except:
            return self.phase_mapping(self.env, s, sn)

    def _add(self, s: np.ndarray, sn: np.ndarray, d: bool) -> None:
        self.reward_shaper.episode += int(d)
        self.reward_shaper.seq.append(self.encode(s, sn))

    def store(self, s: np.ndarray, a: int, r: float, sn: np.ndarray, d: bool):
        self._add(s, sn, d)
        self.buffer.append((s, a, r, sn, self.env.is_goal_reached()))

    def update_q_table(
        self, s: np.ndarray, a, r: float, sn: np.ndarray, is_goal: bool
    ) -> None:
        i = self.discrete(s) + (a,)
        td_err = r
        if not is_goal:
            max_q = np.max(self.q_table[self.discrete(sn)])
            td_err += self.gamma * max_q - self.q_table[i]
        self.q_table[i] += self.lr * td_err

    def score(self) -> float:
        hmm_score = self.reward_shaper.score()
        diff_score = hmm_score - self.old_score
        self.old_score = hmm_score
        return diff_score

    def update_monte_carlo(self) -> None:
        if len(self.buffer) < self.hmm_step:
            return

        hmm_score = self.score()
        for i, (s, a, r, sn, g) in enumerate(reversed(self.buffer)):
            r = (hmm_score + r) * (self.discount_mc**i)
            self.update_q_table(s, a, r, sn, g)
        self.buffer.clear()

    def reset(self) -> None:
        self.old_score = 0
        self.buffer.clear()
        self.reward_shaper.seq.clear()

    def train_episode(self, render: bool = False) -> tuple[float, np.ndarray]:
        done = False
        state = self.env.reset()
        total_reward = 0
        observations = []
        while not done:
            if render:
                self.env.render()
            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            self.store(state, action, reward, next_state, done)
            self.update_monte_carlo()
            observations.append((state, next_state))
            state = next_state
        self.reset()
        return total_reward, np.array(observations)
