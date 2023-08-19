import os
import pickle
import numpy as np
from hmmlearn.hmm import MultinomialHMM, PoissonHMM


class HMMModel:
    def __init__(
        self,
        sequences: list[list],
        encode_count: int,
        n_components: int,
        n_iter: int,
        _class=PoissonHMM,
        random_state: int = None,
        verbose: bool = True,
        **hmm_params
    ) -> None:
        """_summary_

        Args:
            sequences (list[list]): Encoded Trajectories From Different Agents/Samples
            n_iter (int, optional): HMM Train Iteration Defaults to 10_000.
            verbose (bool, optional): Log HMM Training Process Defaults to True.
            n_components (int, optional): HMMs Number of States.
            random_state (int, optional): HMM Random State. Defaults to 42.
        """
        self.I = np.eye(encode_count).astype(np.int32)
        self.X = np.array(sum(map(list, sequences), [])).reshape((-1,))
        self.X = self.I[self.X]

        self.lengths = tuple(map(len, sequences))

        self.model = _class(
            n_iter=n_iter,
            verbose=verbose,
            random_state=random_state,
            n_components=n_components,
            **hmm_params
        )

    def fit(self):
        self.model.fit(X=self.X, lengths=self.lengths)
        return self

    def score_sequence(self, x: list[int]) -> float:
        return self.model.score(self.I[x]) / len(x)

    def score_sequences(self, xs: list[list[int]]) -> list[float]:
        return [self.score_sequence(x) for x in xs]

    def save_model(self, path: str):
        if not os.path.isdir(path):
            raise Exception("Not A Path")
        path = os.path.join(path, "HMM_Model.pkl")
        with open(path, "wb") as file:
            pickle.dump(self.model, file)
        return self

    def load_model(self, path: str):
        if not os.path.isdir(path):
            raise Exception("Not A Path")
        path = os.path.join(path, "HMM_Model.pkl")
        with open(path, "rb") as file:
            self.model = pickle.load(file)
        return self
