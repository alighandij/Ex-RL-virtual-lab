import os
import glob
import json
import streamlit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gym import Env
from stqdm import stqdm
from modules.pools import Pool
from modules.utils import get_time_str, save_json
from modules.components import Components
from modules.environments import EnvSelector

from exrl.agents import QLearning, ExQLearning
from exrl.hmm_model import HMMModel
from exrl.reward_shaper import RewardShaperHMM, reward_one


def randint(low, high):
    return np.random.randint(low, high)


class ExRLPage:
    @staticmethod
    def set_configs(st: streamlit):
        st.set_page_config(
            page_icon="🏃‍♂️", page_title="Ex-RL", initial_sidebar_state="expanded"
        )
        Components.set_center(st)

    @staticmethod
    def get_configs(st: streamlit):
        pool_path = st.selectbox(
            label="Select Pool",
            options=glob.glob(os.path.join(os.getcwd(), "Experiments", "*")),
            format_func=lambda x: x.split(os.sep)[-1],
        )
        pool = Pool(pool_path)
        hmm_path = st.selectbox(
            label="Select HMM",
            options=glob.glob(os.path.join(pool_path, "HMMs", "*")),
            format_func=lambda x: x.split(os.sep)[-1],
        )

        hmm_config = ExRLPage._hmm_info(st, hmm_path)

        sequences = ExRLPage._load_sequences(hmm_path)
        hmm_model = HMMModel(
            sequences=sequences,
            n_iter=hmm_config["hmm"]["n_iter"],
            n_components=hmm_config["hmm"]["n_components"],
            encode_count=hmm_config["encoder_data"]["encode_count"],
        ).load_model(hmm_path)
        phase_map = EnvSelector.get_phase_map(
            pool.env_name, hmm_config["encoder_data"]["encoder_name"]
        )
        st.write("Pool Configs")
        st.json(pool.config, expanded=False)
        return pool, phase_map, sequences, hmm_model, hmm_config, hmm_path

    @staticmethod
    def get_pool(st: streamlit, key: str):
        pool_path = Components.pool_selector(st, key)
        pool = Pool(pool_path)
        return pool_path, pool

    @staticmethod
    def _load_sequences(hmm_path: str):
        path = os.path.join(hmm_path, "sequences.json")
        return json.load(open(path))

    @staticmethod
    def _hmm_info(st: streamlit, hmm_path: str):
        plot_path = os.path.join(hmm_path, "Plots", "*")
        plot_path = glob.glob(plot_path)
        json_path = os.path.join(hmm_path, "config.json")
        hmm_config = json.load(open(json_path))
        with st.expander("HMM Infos"):
            st.write(hmm_config)
            for image in plot_path:
                st.image(image)

        return hmm_config

    @staticmethod
    def env_name_selector(st: streamlit, env_name: str) -> str:
        envs = EnvSelector.get_env_names()
        idx = envs.index(env_name)
        return st.selectbox(
            label="Please Select Your Environment",
            options=EnvSelector.get_env_names(),
            index=idx,
        )

    @staticmethod
    def env_parameters_selector(st: streamlit, env_name: str) -> dict:
        parameters = {}
        env = ExRLPage.env_name_selector(st, env_name)
        cols = st.columns(2)
        for idx, parameter in enumerate(EnvSelector.get_env_params(env)):
            options, default = EnvSelector.get_env_param(env, parameter)
            with cols[idx % 2]:
                parameters[parameter] = st.selectbox(
                    label=f"Please Select {parameter}:",
                    options=options,
                    index=list(options).index(default),
                )

        return parameters, env

    @staticmethod
    def get_agent(
        env: Env,
        agent_configs: dict,
        reward_shaper,
        agent_type: str | None,
        phase_map=None,
    ):
        configs = agent_configs.copy()
        configs["_tqdm"] = stqdm
        configs["reward_shaper"] = reward_shaper
        if agent_type == "exrl":
            assert phase_map is not None
            return ExQLearning(env, phase_map, **configs)
        return QLearning(env, **configs)

    @staticmethod
    def agent_train(
        q_table: np.ndarray,
        env: Env,
        agent_configs: dict,
        reward_shaper,
        agent_type: str,
        phase_map=None,
    ):
        agent = ExRLPage.get_agent(
            env, agent_configs, reward_shaper, agent_type, phase_map
        )
        agent.q_table = np.copy(q_table)
        solved_on = agent.train()
        return agent, solved_on

    @staticmethod
    def get_reward_shapers(
        env: Env,
        hmm: HMMModel,
        phase_map,
        use_env_reward: bool,
        use_hmm_each_step: bool,
    ):
        reward_shaper_ql = None if use_env_reward else reward_one
        reward_shaper_hmm = RewardShaperHMM(hmm, env, phase_map, use_hmm_each_step)
        return reward_shaper_ql, reward_shaper_hmm

    def get_run_configs(st: streamlit, hmm_path: str):
        cols = st.columns(3)
        result_folder = (
            cols[0].text_input(label="Result Folder", value="QL_HMM_Result")
            + "_"
            + get_time_str()
        )
        result_folder = os.path.join(hmm_path, "Ex-RL", result_folder)
        runs = int(cols[1].number_input(label="Total Runs", value=10))
        save_each = int(cols[2].number_input(label="Save Each", value=5))
        use_hmm_each_step = int(st.number_input("HMM Rewarding Steps", value=10))

        return {
            "runs": runs,
            "save_each": save_each,
            "result_folder": result_folder,
            "use_env_reward": True,
            "use_hmm_each_step": use_hmm_each_step,
        }

    @staticmethod
    def run_test_random_sequences(
        st: streamlit,
        hmm_model: HMMModel,
        sequences: list,
        env: Env,
        agent_configs: dict,
        phase_map,
        result_path: str,
    ):
        lens = list(map(len, sequences))
        count = len(sequences)
        min_l = min(lens) - 20
        max_l = max(lens) + 20
        rand_sequences = ExRLPage.generate_random_sequence(count, min_l, max_l)
        agent_sequences = ExRLPage.generate_agent_trajectory(
            env, agent_configs, count, phase_map
        )

        rand_scores = hmm_model.score_sequences(rand_sequences)
        train_scores = hmm_model.score_sequences(sequences)
        agent_scores = hmm_model.score_sequences(agent_sequences)
        fig = ExRLPage.plot_scores(train_scores, rand_scores, agent_scores)
        fig.savefig(os.path.join(result_path, "SequenceScores.jpeg"))
        st.pyplot(fig)

    @staticmethod
    def plot_scores(
        train_scores: list[float], rand_scores: list[float], agent_scores: list[float]
    ):
        xr = list(range(len(rand_scores)))
        xt = list(range(len(train_scores)))
        xa = list(range(len(agent_scores)))
        fig, ax = plt.subplots(dpi=300)
        ax.scatter(xr, rand_scores, c="r", label="Random")
        ax.scatter(xt, train_scores, c="b", label="Trained")
        ax.scatter(xa, agent_scores, c="k", label="Initial Q")
        ax.set_xlabel("Index")
        ax.set_ylabel("Score")
        ax.set_title("Sequence Scores Scatter Plot")
        ax.grid()
        ax.legend()

        return fig

    @staticmethod
    def generate_random_sequence(count: int, min_length: int, max_length: int) -> list:
        sequences = []
        for _ in range(count):
            n = randint(min_length, max_length)
            a = [randint(0, 4) for _ in range(n)]
            sequences.append(a)
        return sequences

    @staticmethod
    def generate_agent_trajectory(env: Env, agent_params, n: int, phase_map):
        def encode(trajectories):
            try:
                return [phase_map(s, sn) for s, sn in trajectories]
            except:
                return [phase_map(env, s, sn) for s, sn in trajectories]

        def f():
            agent = QLearning(env, **agent_params)
            return encode(agent.play(False)[-1])

        trajectories = [f() for _ in range(n)]
        return trajectories

    @staticmethod
    def save_plots(
        run: int,
        total: int,
        save_each: int,
        result_folder: str,
        export_folder: str,
        agent: QLearning,
        reward_shaper_hmm: RewardShaperHMM = None,
    ):
        run = run + 1
        cnd = (run == 1) or (run == total) or (run % save_each == 0)
        if not cnd:
            return

        path = os.path.join(result_folder, "Runs", export_folder, f"run_{run}")
        os.makedirs(path, exist_ok=True)

        agent.history.save(path)
        plt.close()

        if reward_shaper_hmm is None:
            return

        save_json(reward_shaper_hmm.hst, os.path.join(path, "hmm_rewards.json"))

    @staticmethod
    def mkdir_for_plots(result_folder: str, run: int):
        run_paths = os.path.join(result_folder, "Runs", f"run_{run}")
        os.makedirs(run_paths, exist_ok=True)

        hmm_results = []
        normal_results = []
        for m in ["min", "max"]:
            hmm_results.append(os.path.join(run_paths, f"HMM_{m}_exploration"))
            normal_results.append(os.path.join(run_paths, "QL_{m}_exploration"))
        normal_results.append(os.path.join(run_paths, "QL_Expert"))

        for p in [*hmm_results, normal_results]:
            os.makedirs(p, exist_ok=True)

        return normal_results, hmm_results

    @staticmethod
    def get_describe(solves: list[int], episodes: int):
        return pd.DataFrame(
            data={"solves": list(filter(lambda x: x < episodes, solves))}
        ).describe()

    @staticmethod
    def get_describes(history: dict, episodes: int):
        describes = {}
        for k in history.keys():
            describes[k] = ExRLPage.get_describe(history[k], episodes)
        return describes

    @staticmethod
    def save_describes(describes: dict, result_folder):
        stats_path = os.path.join(result_folder, "statistics")
        os.makedirs(stats_path, exist_ok=True)
        for key in describes.keys():
            csv_path = os.path.join(stats_path, f"{key}.csv")
            describes[key].to_csv(csv_path)
        return

    @staticmethod
    def get_result_table(describes: dict, runs: int):
        data = {}
        for key in describes.keys():
            df = describes[key]
            data[key] = [
                float(df.loc["mean"]),
                float(df.loc["std"]),
                float(df.loc["count"]) / runs,
            ]
        df = pd.DataFrame(data=data)
        df.index = ["mean", "std", "success"]
        return df

    @staticmethod
    def save_history(history: dict, result_folder):
        path = os.path.join(result_folder, "solve_history.json")
        save_json(history, path)

    @staticmethod
    def show_save_run_results(
        st: streamlit,
        history: dict,
        result_folder: str,
        episodes: int,
        runs: int,
        **kwargs,
    ):
        ExRLPage.save_history(history, result_folder)

        describes = ExRLPage.get_describes(history, episodes)
        ExRLPage.save_describes(describes, result_folder)
        result_table = ExRLPage.get_result_table(describes, runs)
        result_table.to_csv(os.path.join(result_folder, "result_table.csv"))
        st.table(result_table)

    @staticmethod
    def agent_configs(st: streamlit, env_name: str) -> dict:
        cols = st.columns(4)

        with cols[0]:
            gamma = float(
                st.number_input(
                    label="Discount Factor (Gamma)",
                    value=EnvSelector.get_agent_gamma(env_name),
                    format="%.5f",
                )
            )

            episodes = int(st.number_input(label="Episodes", value=10_000))

            discrete = (
                int(
                    st.number_input(
                        label="Discrete Size",
                        value=EnvSelector.get_agent_discrete(env_name),
                    )
                ),
            ) * EnvSelector.get_state_shape(env_name)

        with cols[1]:
            epsilon = float(
                st.number_input(
                    label="Epsilon Start",
                    value=EnvSelector.get_agent_epsilon(env_name),
                    format="%.5f",
                )
            )

            epsilon_decay = float(
                st.number_input(label="Epsilon Decay", value=0.99, format="%.5f")
            )

            epsilon_min = float(
                st.number_input(label="Epsilon Minimum", value=0.001, format="%.5f")
            )

        with cols[2]:
            lr = float(
                st.number_input(
                    label="Learning Rate",
                    value=EnvSelector.get_agent_lr(env_name),
                    format="%.5f",
                )
            )

            lr_decay = float(
                st.number_input(
                    label="Learning Rate Decay",
                    value=EnvSelector.get_agent_lr_decay(env_name),
                    format="%.5f",
                )
            )

            lr_min = float(
                st.number_input(
                    label="Learning Rate Minimum",
                    value=EnvSelector.get_agent_lr_min(env_name),
                    format="%.5f",
                )
            )

        with cols[3]:
            count = int(
                st.number_input(
                    label="Count", value=EnvSelector.get_agent_count(env_name)
                )
            )
            target = float(
                st.number_input(
                    label="Target", value=EnvSelector.get_agent_target(env_name)
                )
            )

        return {
            "gamma": gamma,
            "discrete": discrete,
            "episodes": episodes,
            "lr": lr,
            "lr_min": lr_min,
            "lr_decay": lr_decay,
            "epsilon": epsilon,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
            "count": count,
            "target": target,
            "break_on_solve": True,
            "render_each": episodes + 1,
        }

    @staticmethod
    def init_experiment(
        st: streamlit,
        result_folder: str,
        hmm_model: HMMModel,
        sequences,
        env: Env,
        agent_configs: dict,
        phase_map,
        use_env_reward: bool,
        use_hmm_each_step: bool,
        **kwargs,
    ):
        os.makedirs(result_folder)
        reward_shaper_ql, reward_shaper_hmm = ExRLPage.get_reward_shapers(
            env, hmm_model, phase_map, use_env_reward, use_hmm_each_step
        )
        return reward_shaper_ql, reward_shaper_hmm

    @staticmethod
    def init_q_table(env: Env, agent_configs: dict):
        return np.copy(ExRLPage.get_agent(env, agent_configs, None, 0.0).q_table)

    @staticmethod
    def run_experiment(
        st: streamlit,
        env: Env,
        agent_configs: dict,
        reward_shaper_ql,
        reward_shaper_hmm: RewardShaperHMM,
        run: int,
        total: int,
        save_each: int,
        result_folder: str,
        phase_map,
    ):
        args_save = (run, total, save_each, result_folder)
        q_table = ExRLPage.init_q_table(env, agent_configs)
        ql_agent, ql_solve = ExRLPage.agent_train(
            q_table,
            env,
            agent_configs,
            reward_shaper_ql,
            "normal",
            None,
        )
        ExRLPage.save_plots(*args_save, "QLearning", ql_agent)

        reward_shaper_hmm.reset()
        exrl_agent, exrl_solve = ExRLPage.agent_train(
            q_table, env, agent_configs, reward_shaper_hmm, "exrl", phase_map
        )
        ExRLPage.save_plots(
            *args_save, "ExperienceBasedQLearning", exrl_agent, reward_shaper_hmm
        )
        ExRLPage.plot_experiment(st, exrl_agent, ql_agent)

        return ql_solve, exrl_solve

    @staticmethod
    def create_agent(env: Env, q_table: np.ndarray, decay: float, agent_config: dict):
        config = {**agent_config, "decay": decay}
        agent = QLearning(env, **config)
        agent.q_table = q_table.copy()
        return agent

    def plot_experiment(st: streamlit, exrl: ExQLearning, ql: QLearning):
        ql_reward = pd.DataFrame({"Q-Learning": ql.history.rewards})
        exrl_reward = pd.DataFrame({"EX-RL": exrl.history.rewards})
        if len(ql_reward) > len(exrl_reward):
            rewards = ql_reward.join(exrl_reward)
        else:
            rewards = exrl_reward.join(ql_reward)
        st.line_chart(rewards)

    @staticmethod
    def run_experiments(
        st: streamlit,
        runs: int,
        save_each: int,
        result_folder: str,
        env: Env,
        phase_map,
        agent_configs: dict,
        reward_shaper_ql,
        reward_shaper_hmm: RewardShaperHMM,
        **kwargs,
    ):
        history = {"EX-RL": [], "Normal Q-Learning": []}

        for run in range(runs):
            st.write(f"Run {run + 1} / {runs}")
            ql, exrl = ExRLPage.run_experiment(
                st=st,
                env=env,
                agent_configs=agent_configs,
                reward_shaper_ql=reward_shaper_ql,
                reward_shaper_hmm=reward_shaper_hmm,
                run=run,
                total=runs,
                save_each=save_each,
                result_folder=result_folder,
                phase_map=phase_map,
            )
            history["EX-RL"].append(exrl)
            history["Normal Q-Learning"].append(ql)
            Components.separator(st)
            print("-" * 50)

        return history

    @staticmethod
    def save_configs(
        agent_configs,
        env_name,
        env_parameters,
        run_configs,
    ):
        result_folder = run_configs["result_folder"]
        data = {
            "agent": agent_configs,
            "environment": {"name": env_name, "parameters": env_parameters},
            "run_configs": run_configs,
        }
        save_json(data, os.path.join(result_folder, "config.json"))

    @staticmethod
    def start_experiment(
        st: streamlit,
        env: Env,
        parameters: dict,
        result_folder: str,
        agent_configs: dict,
        hmm_model: HMMModel,
        sequences: list,
        phase_map,
        use_env_reward: bool,
        use_hmm_each_step: bool,
        run_configs: dict,
        runs: int,
        save_each: int,
        reward_shaper_ql,
        reward_shaper_hmm: RewardShaperHMM,
    ):
        return
