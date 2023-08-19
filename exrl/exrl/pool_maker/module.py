import os
import json
import numpy as np
from itertools import product
from exrl.history import History
from exrl.q_table_agent import QTableAgent
from exrl.custom_mountaincar import AngleModifiedDiscreteMountainCar


class Config:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.result = {}
        self.saveable = []

    def run(self, save_path: str):
        env = AngleModifiedDiscreteMountainCar(
            is_continuous=True,
            angle_perturbation=self.config["angle_perturbation"],
        )
        for name, value in self.config.items():
            setattr(env, name, value)

        agent = QTableAgent(
            env,
            use_tqdm=True,
            episodes=50_000,
            break_on_solve=False
        )
        agent.train()
        self.result = agent.get_result()
        self._parse_result(save_path)

        return self.get_all()

    def get_all(self):
        return {**self.config, **self.result}

    def _parse_result(self, save_path: str):
        for key, value in self.result.items():
            if type(value) == History:
                print("hey")
                self.result[key] = value.save(save_path)
                continue

            if type(value) != np.ndarray:
                continue

            path = os.path.join(save_path, f"{key}.npy")
            np.save(path, value)
            self.result[key] = path.replace(os.sep, "/")

    def __str__(self) -> str:
        return json.dumps(self.get_all(), indent=4)

    def __getitem__(self, key):
        return self.config[key]

    def __getattr__(self, key):
        return self[key]

    def _make_dir(self, path: str):
        try:
            os.mkdir(path)
        except:
            pass


class PoolConfigLoader:
    def __init__(self, filename) -> None:
        self.filename = filename
        self.data = self._load_file()
        self.info = self.data["info"]
        self.configs = list(map(Config, self.data["configs"]))

    def _load_file(self):
        f = open(self.filename)
        data = json.load(f)
        f.close()
        return data

    def run(self, folder: str, save_every: int = 1_000):
        self._mkdir(folder)
        for i, config in enumerate(self.configs):
            print(f"CONFIG #{i}")
            path = os.path.join(folder, f"resultConfig_{i}")
            self._mkdir(path)
            print(config)
            self.data["configs"][i] = config.run(path)
            print(config)
            if i % save_every == 0:
                self.save()
            print("-" * 25)

    def _mkdir(self, folder: str) -> str:
        try:
            os.mkdir(folder)
        except:
            pass

    def save(self):
        f = open(self.filename, "w")
        f.write(json.dumps(self.data))
        f.close()

    def __len__(self):
        return self.info["total"]

    def __getitem__(self, i):
        return self.configs[i]

    def __repr__(self) -> str:
        return json.dumps(self.info, indent=4)

    @staticmethod
    def create_config(**kwargs):
        keys = kwargs.keys()
        prod = product(*[kwargs[k] for k in keys])
        configs = []

        for p in prod:
            config = {}
            for j, k in enumerate(keys):
                config[k] = p[j]
            configs.append(config)

        info = {
            "total": len(configs),
            "parameters": {key: list(kwargs[key]) for key in keys}
        }

        data = {
            "info": info,
            "configs": configs
        }

        return data


if __name__ == "__main__":

    force = np.linspace(0.001 / 2, 0.001 * 2, 5)
    gravity = np.linspace(0.0025 / 2, 0.0025 * 2, 5)
    max_speed = np.linspace(0.035, 0.15, 5)
    min_speed = -max_speed
    angle_perturbation = list(range(-10, 11))

    data = PoolConfigLoader.create_config(
        "config.json",
        force=force,
        gravity=gravity,
        max_speed=max_speed,
        min_speed=min_speed,
        angle_perturbation=angle_perturbation
    )

    pool = PoolConfigLoader("config.json")

    pool.run("test_result")
