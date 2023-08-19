import os
import json
import uuid
from itertools import product

import numpy as np
import pandas as pd

from modules.environments.selector import EnvSelector

from exrl.utils import save_np_arr
from exrl.history import History
from exrl.agents.qlearn import QLearning


class Config:
    def __init__(
        self,
        save_path: str,
        env_maker,
        env_config: list,
        agent_config: dict
    ) -> None:
        self.id = str(uuid.uuid4())

        self.env = env_maker(**env_config)
        self.env_config = env_config

        self.agent = QLearning(
            self.env,
            **agent_config,
        )
        self.result = {}
        self.save_path = os.path.join(save_path, self.id)
        self._mkdir()

    def train(self):
        self.agent.train()
        self.result = {
            "agent_id": self.id,
            **self.agent.get_result(),
            **self.env_config
        }
        self.save()
        return self.result.copy()

    def save(self):
        delete = []
        for key in self.result.keys():
            value = self.result[key]
            if isinstance(value, (str, float, int)):
                continue

            if isinstance(value, np.ndarray):
                path = os.path.join(self.save_path, key)
                save_np_arr(value, path)

            if isinstance(value, History):
                value.save(self.save_path)

            delete.append(key)

        for key in delete:
            self.result.pop(key)

        with open(os.path.join(self.save_path, "env.json"), "w") as f:
            json.dump(self.env_config, f, indent=4)

    def _mkdir(self):
        try:
            os.mkdir(self.save_path)
        except Exception as e:
            pass


class PoolMaker:
    def __init__(
        self,
        save_path: str,
        env_maker,
        env_configs: list[dict],
        agent_config: dict
    ) -> None:
        self._mkdir(save_path)
        self.save_path = os.path.join(save_path, "Agents")
        self._mkdir(self.save_path)

        self.configs = [
            Config(self.save_path, env_maker, env_config, agent_config)
            for env_config in env_configs
        ]
        self.results = {}

    def run(self):
        n = len(self.configs)
        for i, config in enumerate(self.configs):
            print("#" * 50)
            print(f"{i + 1} / {n}")
            print("ID", config.id)
            print(config.env_config)
            result = config.train()
            self._add_result(result)

    def save(self):
        path = os.path.join(self.save_path, "results_information.csv")
        self.results_df = pd.DataFrame(data=self.results)
        self.results_df.to_csv(path, index=None)

    def _mkdir(self, path):
        try:
            os.mkdir(path)
        except Exception as e:
            print(e)

    def _add_result(self, result: dict):
        for key in result.keys():
            if not (key in self.results):
                self.results[key] = []
            self.results[key].append(result[key])

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


class Pool:
    def __init__(self, path: str) -> None:
        self.path = path
        self.config = self._read_config()
        self.agents = self._read_agents()
        self.env_name = self.config["environment"]["name"]
        self.env_make = EnvSelector.get_env_maker(self.env_name)

    def _read_config(self):
        name = os.path.join(self.path, "config.json")
        return self._read_json(name)

    def _read_agents(self):
        name = os.path.join(self.path, "Agents", "results_information.csv")
        return pd.read_csv(name)

    def _read_json(self, name):
        with open(name, "r") as f:
            return json.load(f)

    def find(self, **kwargs):
        df = self.agents.copy()
        for key, value in kwargs.items():

            if not (key in self.agents.columns):
                raise Exception(f"{key} is not valid")

            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    continue
                if len(value) == 2:
                    _min, _max = sorted(value)
                    df = df[(df[key] <= _max) & (_min <= df[key])]
                else:
                    cond = df[key].isin(value)
                    df = df[cond]

            else:
                df = df[df[key] == value]
        return df

    def _load_qtable(self, agent_id: str) -> np.ndarray:
        args = (self.path, "Agents", agent_id, "q_table.npy")
        path = os.path.join(*args)
        q_table = np.load(path)
        return q_table

    def _make_env(self, agent_id: str):
        path = os.path.join(self.path, "Agents", agent_id, "env.json")
        config = self._read_json(path)
        env = self.env_make(**config)
        return env

    def _load_history(self, agent_id: str) -> History:
        path = os.path.join(self.path, "Agents", agent_id)
        count = self.get_agent_info(agent_id).count_last.to_list()[0]

        rewards = os.path.join(path, "history_rewards.npy")
        epsilons = os.path.join(path, "history_epsilons.npy")
        learning_rates = os.path.join(path, "learning_rates.npy")

        rewards = np.load(rewards)
        epsilons = np.load(epsilons)
        try:
            learning_rates = np.load(learning_rates)
        except:
            learning_rates = []

        history = History(w=count)
        history.learning_rates = learning_rates
        history.rewards = rewards
        history.epsilons = epsilons

        return history

    def load_agent(self, agent_id: str):
        env = self._make_env(agent_id)
        agent = QLearning(env, **self.config["agent"])
        agent.q_table = self._load_qtable(agent_id)
        agent.history = self._load_history(agent_id)
        return agent

    def _load_best_observation(self, agent_id: str):
        path = os.path.join(self.path, "Agents", agent_id,
                            "best_observation.npy")
        return np.load(path)

    def _find_agent_ids(self, **kwargs):
        return self.find(**kwargs).agent_id

    def show_agent_info(self, agent_id: str):
        cond = self.agents.agent_id == agent_id
        return self.agents[cond]

    def load_agents(self, **kwargs):
        agents = {}
        for agent_id in self._find_agent_ids(**kwargs):
            agents[agent_id] = self.load_agent(agent_id)
        return agents

    @staticmethod
    def is_valid(path: str):
        if not os.path.isdir(path):
            return False, "Please Enter A Folder"

        config = os.path.join(path, "config.json")
        if not os.path.isfile(config):
            return False, "config.json does not exist"

        config = os.path.join(path, "Agents", "results_information.csv")
        if not os.path.isfile(config):
            return False, "Agents/results_information.csv does not exist"

        return True, None

    def get_parameters(self):
        return self.config["environment"]["info"]["parameters"].items()

    def load_best_observations(self, **kwargs):
        bests = {}
        for agent_id in self._find_agent_ids(**kwargs):
            bests[agent_id] = self._load_best_observation(agent_id)
        return bests

    def get_slope(self, agent_id: str):
        cond = self.agents.agent_id == agent_id
        return list(self.agents[cond].angle_perturbation)[0]

    def get_agent_info(self, agent_id: str):
        cond = self.agents.agent_id == agent_id
        return self.agents[cond]

    def get_agent_property(self, agent_id: str, key: str):
        return self.get_agent_info(agent_id)[key].iloc[0]
