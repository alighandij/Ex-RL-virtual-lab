import numpy as np
from gym import Env
from abc import ABC, abstractmethod
from types import FunctionType
from tqdm.auto import tqdm
from exrl.history import History


class Agent(ABC):
    def __init__(
        self,
        env: Env,
        name: str,
        gamma: float = 0.99,
        episodes: int = 20_000,
        render_each: int = 100_000,
        lr: float = 1,
        lr_min: float = 0.01,
        lr_decay: float = 0.999,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99,
        epsilon_min: float = 1e-3,
        count: int = 20,
        target: float = -120,
        reward_shaper: FunctionType = None,
        break_on_solve: bool = True,
        _tqdm: tqdm = None,
    ) -> None:
        self.env = env
        self.name = name
        self.tqdm = _tqdm or tqdm
        self.gamma = gamma
        self.episodes = episodes

        self.lr = lr
        self.lr_min = lr_min
        self.lr_decay = lr_decay

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.count = count
        self.target = target
        self.reward_shaper = reward_shaper
        self.history = History(count)

        self.solved_on = self.episodes + 1

        self.best_reward = None
        self.best_observation = None

        self.render_each = render_each
        self.break_on_solve = break_on_solve

    def is_explore(self):
        return np.random.rand() < self.epsilon

    def log_break_on_solve(self) -> None:
        if not self.break_on_solve:
            return

        if self.solved_on > self.episodes:
            print("Not Solved")
            return

        print(f"Solved at {self.solved_on} Reward = {self.target} Count = {self.count}")

    def update_bests(self, reward: float, observations):
        if self.best_reward == None or self.best_reward < reward:
            self.best_reward = reward
            self.best_observation = observations

    def last_average(self) -> float:
        arr = self.history.rewards[-self.count :]
        return sum(arr) / len(arr)

    def is_solved(self) -> bool:
        if len(self.history.rewards) < self.count:
            return False
        return self.last_average() >= self.target

    def get_result_general(self) -> dict:
        return {
            "history": self.history,
            "count_last": self.count,
            "best_reward": self.best_reward,
            "best_observation": self.best_observation,
            "reward_last_average": self.last_average(),
            "solved_on": self.solved_on,
            "reward_target": self.target,
        }

    def random_action(self):
        return self.env.action_space.sample()

    def get_action(self, state: np.ndarray):
        if self.is_explore():
            return self.random_action()
        return self.greedy_action(state)

    def decay_params(self):
        self.lr = max(self.lr * self.lr_decay, self.lr_min)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_history(self, reward_episode: float):
        self.history.add(
            lr=self.lr,
            epsilon=self.epsilon,
            reward=reward_episode,
        )

    def update_params(
        self, reward_episode: float, observation_episode: np.ndarray
    ) -> None:
        self.update_bests(reward_episode, observation_episode)
        self.update_history(reward_episode)
        self.decay_params()

    def reset_reward_shaper(self):
        try:
            self.reward_shaper.reset()
        except:
            pass

    def train(self) -> int:
        pbar = self.tqdm(range(self.episodes), unit="Episode", desc=self.name)
        for e in pbar:
            render = (e + 1) % self.render_each == 0
            r, obs = self.train_episode(render)
            self.update_params(r, obs)
            self.reset_reward_shaper()
            pbar.set_postfix(self.history.log())
            if self.is_solved() and self.break_on_solve:
                self.solved_on = e
                break

        self.log_break_on_solve()
        return self.solved_on

    def play(self, render: bool) -> tuple[float, np.ndarray]:
        done = False
        state = self.env.reset()
        observations = []
        total_reward = 0
        while not done:
            if render:
                self.env.render()
            action = self.greedy_action(state)
            next_state, reward, done, _ = self.env.step(action)
            observations.append((state, next_state))
            total_reward += reward
            state = next_state
        return total_reward, observations

    def get_reward(self, state, next_state, reward, done):
        if self.reward_shaper is None:
            return reward
        return self.reward_shaper(state, next_state, reward, done)

    @abstractmethod
    def greedy_action(self, state: np.ndarray):
        ...

    @abstractmethod
    def train_episode(self, render: bool) -> tuple[float, np.ndarray]:
        ...

    @abstractmethod
    def get_result(self) -> dict:
        ...
