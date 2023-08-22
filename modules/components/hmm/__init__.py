import os
import streamlit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stqdm import stqdm
from hmmlearn.hmm import PoissonHMM, MultinomialHMM
from modules.pools import Pool
from modules.utils import get_time_str, heatmap, save_json
from exrl.hmm_model import HMMModel
from modules.environments.selector import EnvSelector
from modules.components import Components

NO = "No"
YES = "Yes"
FILTER = "Filter"


class HMMPage:
    @staticmethod
    def set_configs(st: streamlit):
        st.set_page_config(
            page_icon="🕵️", page_title="HMM Trainer", initial_sidebar_state="expanded"
        )
        Components.set_center(st)

    @staticmethod
    def get_pool(st: streamlit):
        pool_path = Components.pool_selector(st)

        is_valid, err = Pool.is_valid(pool_path)
        if not is_valid:
            st.error(err)
            st.stop()

        folder = (
            st.text_input(label="Result Folder", value="HMM") + "_" + get_time_str()
        )

        result_path = os.path.join(pool_path, "HMMs", folder)
        return Pool(pool_path), result_path

    @staticmethod
    def get_configs(st: streamlit):
        cols = st.columns(2)

        n_components = int(cols[0].number_input(label="Hidden States", value=4))

        n_iter = int(cols[1].number_input(label="Iterations", value=100))

        _class_dict = {
            "Poisson": PoissonHMM,
            "Multinomial": MultinomialHMM,
        }
        _class_name = cols[0].selectbox(
            label="HMM Type",
            options=list(_class_dict.keys()),
        )
        _class = _class_dict[_class_name]

        n_steps = int(cols[1].number_input(label="Steps", value=5, min_value=1))

        return n_components, n_iter, n_steps, _class, _class_name

    @staticmethod
    def get_sequence_selection(st: streamlit, pool: Pool):
        cols = st.columns(3)

        best_observation = cols[0].radio("Best Observation Samples", (YES, NO, FILTER))

        agent_sample = cols[1].radio("Agent Samples", (YES, NO, FILTER))

        disabled = agent_sample == NO
        sample_count = int(
            cols[2].number_input(label="Count", value=1, disabled=disabled)
        )

        sample_reward = int(
            cols[2].number_input(label="Minimum Reward", value=0, disabled=disabled)
        )

        max_try = int(
            cols[2].number_input(label="Maximum Try", value=100, disabled=disabled)
        )

        if agent_sample == NO and best_observation == NO:
            st.error("Check At Least One")
            st.stop()

        return {
            "max_try": max_try,
            "agent_sample": agent_sample,
            "sample_count": sample_count,
            "sample_reward": sample_reward,
            "best_observation": best_observation,
        }

    @staticmethod
    def _filter_selects(st: streamlit, pool: Pool) -> dict:
        filters = {
            "reward_last_average": (
                st.number_input(label="minimum reward_last_average", value=0.0),
                np.inf,
            )
        }
        for key, value in pool.get_parameters():
            if len(value) < 2:
                continue
            filters[key] = st.multiselect(label=key, options=value)
        return filters

    @staticmethod
    def _filter_infos(st: streamlit, env_name: str):
        st.write("### Filter Conditions")
        st.write(f"Environment Name: {env_name}")
        st.info(
            """
            + 0 Select: Select All Options
            + 2 Select: Select In Range (Inclusive)
            + n Select: Select From List
            """
        )

    @staticmethod
    def _show_filtered_agents(st: streamlit, agents: pd.DataFrame):
        st.write(f"Total: {len(agents)}")
        st.dataframe(agents)

    @staticmethod
    def filter_pool(st: streamlit, pool: Pool, selections: dict):
        agent_sample = selections.get("agent_sample")
        best_observation = selections.get("best_observation")

        if best_observation != FILTER and agent_sample != FILTER:
            return {}

        HMMPage._filter_infos(st, pool.env_name)
        filters = HMMPage._filter_selects(st, pool)
        agents = pool.find(**filters)
        HMMPage._show_filtered_agents(st, agents)
        return filters

    @staticmethod
    def _load_bests(pool: Pool, filters: dict, selections: dict) -> dict:
        best_observation = selections.get("best_observation")
        if best_observation == NO:
            return {}

        if best_observation == YES:
            return pool.load_best_observations()

        return pool.load_best_observations(**filters)

    @staticmethod
    def _load_agents(pool: Pool, filters: dict, selections: dict) -> dict:
        agent_sample = selections.get("agent_sample")
        if agent_sample == NO:
            return {}
        if agent_sample == YES:
            return pool.load_agents()

        return pool.load_agents(**filters)

    @staticmethod
    def _load_samples(pool: Pool, filters: dict, selections: dict):
        max_try = selections.get("max_try")
        sample_count = selections.get("sample_count")
        sample_reward = selections.get("sample_reward")

        agents = HMMPage._load_agents(pool, filters, selections)
        samples = {}
        for agent_id, agent in agents.items():
            samples[agent_id] = []
            for _ in (pbar := stqdm(range(max_try))):
                reward, obs = agent.play(False)
                if reward >= sample_reward:
                    samples[agent_id].append(obs)
                    pbar.set_postfix(
                        {"sampled": f"{len(samples[agent_id])} / {sample_count}"}
                    )
                if len(samples[agent_id]) == sample_count:
                    break
            if len(samples[agent_id]) == 0:
                del samples[agent_id]

        return samples

    @staticmethod
    def _merge_best_sample(samples: dict, bests: dict):
        merged = samples.copy()
        for agent_id, best in bests.items():
            if agent_id in merged:
                merged[agent_id].append(best)
            else:
                merged[agent_id] = [best]
        return merged

    @staticmethod
    def load_all_samples(
        pool: Pool,
        filters: dict,
        selections: dict,
    ):
        bests = HMMPage._load_bests(pool, filters, selections)
        samples = HMMPage._load_samples(pool, filters, selections)
        merged = HMMPage._merge_best_sample(samples, bests)
        return merged

    @staticmethod
    def get_encoder(st: streamlit, pool: Pool):
        env = pool.env_name
        name = st.selectbox(
            label="Encoder", options=EnvSelector.get_encoders_names(env)
        )
        if name == None:
            st.error("Please Select Encoding")
            st.stop()

        return (
            EnvSelector.get_encoder(env, name),
            name,
            EnvSelector.get_encode_count(env, name),
        )

    @staticmethod
    def _cut_step_encodes(encoded: list, n_steps: int):
        samples = []
        for e in encoded:
            for i in range(len(e) - n_steps):
                samples.append(e[i : i + n_steps])
        return samples

    @staticmethod
    def get_sequences(
        encoder, pool: Pool, filters: dict, selections: dict, n_steps: int
    ):
        samples = HMMPage.load_all_samples(pool, filters, selections)
        encoded = encoder(samples, pool)
        cutstep = HMMPage._cut_step_encodes(encoded, n_steps)
        return cutstep

    @staticmethod
    def save_sequences(st: streamlit, sequences: list, result_path: str):
        st.write(f"Total Sequences: {len(sequences)}")
        os.makedirs(result_path, exist_ok=True)
        path = os.path.join(result_path, "sequences.json")
        save_json(sequences, path)

    @staticmethod
    def train_hmm(
        sequences: list,
        encode_count: int,
        n_iter: int,
        n_components: int,
        result_path: str,
        _class,
    ):
        hmm = HMMModel(
            _class=_class,
            n_iter=n_iter,
            sequences=sequences,
            encode_count=encode_count,
            n_components=n_components,
        )
        hmm.fit()
        hmm.save_model(result_path)
        return hmm

    @staticmethod
    def plot_hmm_params(st: streamlit, hmm: HMMModel, result_path: str):
        folder = os.path.join(result_path, "Plots")
        os.mkdir(folder)

        history = hmm.model.monitor_.history
        fig, ax = plt.subplots(dpi=300)
        ax.grid()
        ax.plot(history)
        ax.set_title("HMM Training History")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("HMM Score")
        ax.set_xlim(0, len(history))
        fig.savefig(os.path.join(folder, "HMM_Training_History.jpg"))
        st.pyplot(fig)

        figs = [
            (hmm.model.transmat_, "Transition Matrix"),
            ([hmm.model.startprob_], "Start Prob Matrix"),
        ]
        try:
            figs.append((hmm.model.emissionprob_, "Emission Prob Matrix"))
        except:
            pass
        try:
            figs.append((hmm.model.lambdas_, "Lambdas Matrix"))
        except:
            pass

        for data, name in figs:
            fig = heatmap(data, name)
            st.pyplot(fig)
            jpg = "_".join(name.split(" ")) + ".jpg"
            fig.savefig(os.path.join(folder, jpg))

    @staticmethod
    def save_configs(
        result_path: str,
        filters: dict,
        selections: dict,
        encoder_name: str,
        encode_count: int,
        n_components: int,
        n_iter: int,
        n_steps: int,
        _class_name: str,
    ):
        data = {
            "filters": filters,
            "selections": selections,
            "encoder_data": {
                "encoder_name": encoder_name,
                "encode_count": encode_count,
            },
            "hmm": {
                "n_components": n_components,
                "n_iter": n_iter,
                "n_steps": n_steps,
                "type": _class_name,
            },
        }
        path = os.path.join(result_path, "config.json")
        save_json(data, path)

    @staticmethod
    def hmm_report(st: streamlit, hmm, result_path: str):
        st.write(f"Total Iteration: {hmm.model.monitor_.iter}")
        HMMPage.plot_hmm_params(st, hmm, result_path)
