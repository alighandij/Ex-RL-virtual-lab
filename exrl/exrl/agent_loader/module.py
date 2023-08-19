import numpy as np
import pandas as pd

class AgentDataLoader:
    def __init__(self, info_file: str) -> None:
        self.info = pd.read_csv(info_file)

    def get_attributes(self):
        return list(self.info.columns)
    
    def get_all_by_attributes(self, **kwargs):
        df = self.info
        for key, value in kwargs.items():
            df = df[df[key] == value]
        q_table = []
        for name in df["q_table"].values:
            q_table.append(np.load(name))
        return q_table
    