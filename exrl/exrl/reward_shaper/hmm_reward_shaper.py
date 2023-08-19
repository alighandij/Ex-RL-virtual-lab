import numpy as np
import matplotlib.pyplot as plt

from gym import Env
from typing import Callable
from collections import deque
from exrl.hmm_model import HMMModel


class RewardShaperHMM:
    def __init__(self, hmm: HMMModel, env: Env, phase_map: Callable, step: int = 5, _inf: float = -100) -> None:
        self.hmm = hmm
        self.stp = step
        self.env = env
        self.fpm = phase_map
        self.seq = deque(maxlen=step)  # Current Episode Sequence
        self.tmp = []  # Episode Scores Sequeces
        self.hst = []
        self.scr = []  # Step Scores
        self._inf = _inf
        self.episode = 0

    def reset(self) -> None:
        self.hst.append(self.tmp.copy())
        self.seq.clear()
        self.tmp.clear()
        
    def score(self) -> float:
        s = self.hmm.score_sequence(self.seq)
        s = self._inf if np.isinf(s) else s
        self.scr.append(s)
        self.tmp.append(s)
        return s
    
    def add(self, s: np.ndarray, sn: np.ndarray, done: bool):
        self.seq.append(self.encode(s, sn))
        self.episode += int(done)

    def __call__(self, s: np.ndarray, sn: np.ndarray, r: float, d: bool) -> float:
        self.add(s, sn, d)
        if not self.can_score():
            return r
        return r + self.score()

    def can_score(self) -> bool:
        n = len(self.seq)
        if n == 0:
            return False
        return n < self.stp

    def encode(self, s: np.ndarray, sn: np.ndarray) -> int:
        try:
            return self.fpm(s, sn)
        except:
            # Mountain Car Environment
            return self.fpm(self.env, s, sn)

    def get_scores_episode(self, episode: int) -> list:
        return self.hst[episode]

    def __len__(self):
        return len(self.len)
