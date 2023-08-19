import os
import numpy as np
import pandas as pd


class DefaultKeys:
    MOUNTAIN_CAR = [
        "angle_perturbation", "q_table",
        "history", "solved_on",
        "best_reward", "count_target",
        "reward_target", "best_observation"
    ]

class OtherKeys: 
    MOUNTAIN_CAR = ["history", "best_observation", "q_table"]

class InfoSave:
    def __init__(self, keys: list[str], other: list[str], folder="results") -> None:
        """_summary_
        For Saving Config & Data of each agent

        Args:
            keys (list[str]): configs that you want to save in log file
            other (list[str]): results for example arrays
            folder (str, optional): save folder Defaults to "results".
        """
        self.keys = keys
        self.data = {k: [] for k in self.keys}
        self.other = other
        self.folder = folder
        self.saved = set()

    def add(self, data: dict):
        for key in self.keys:
            self.data[key].append(data[key])

    def save(self, csv_name: str):
        self._make_dir(os.path.join(os.getcwd(), self.folder))

        for key in self.other:
            self.save_other(key)
        
        pd.DataFrame(data=self.data).to_csv(csv_name)

    def save_other(self, key: str):
        self._make_dir(os.path.join(os.getcwd(), self.folder, key))
        arr = []
        for i, x in enumerate(self.data[key]):
            if i in self.saved:
                continue
            path = os.path.join(self.folder, key, self._get_file_name(i, "_"))
            if key == "history":
                x.save(path)
            else:
                np.save(path, x)

            arr.append(path)
            
        self.data[key] += arr

    def _make_dir(self, path):
        try:
            os.mkdir(path)
        except:
            pass

    def _get_file_name(self, idx: int, sep: str) -> str:
        name = [str(self.data[key][idx]) for key in self.keys]
        return sep.join(name)
