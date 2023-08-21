import os
import json
import numpy as np
from modules.pools import PoolMaker
from modules.utils import save_json
from modules.components import Components
from modules.environments import EnvSelector


class PipelinePage:
    @staticmethod
    def env_name_selector(st) -> str:
        env_name = st.selectbox(
            label="Please Select Your Environment", options=EnvSelector.get_env_names()
        )
        return env_name

    @staticmethod
    def env_parameters_selector(st, env_name: str) -> dict:
        parameters = {}

        for parameter in EnvSelector.get_env_params(env_name):
            options, default = EnvSelector.get_env_param(env_name, parameter)
            col1, col2 = st.columns([4, 1], gap="small")

            select_all = col2.checkbox(label="Select All", key=parameter)

            parameters[parameter] = col1.multiselect(
                label=f"Please Select {parameter}:",
                options=options,
                default=options if select_all else default,
            )

        return parameters

    @staticmethod
    def total_configs(st, parameters: dict) -> int:
        total_configs = int(np.prod(list(map(len, parameters.values()))))
        st.write(f"**Total Configs = {total_configs}**")
        return total_configs

    @staticmethod
    def agent_configs(st, env_name: str) -> dict:
        cols = st.columns(4)

        with cols[0]:
            gamma = float(
                st.number_input(
                    label="Gamma",
                    value=EnvSelector.get_agent_gamma(env_name),
                    format="%.5f",
                )
            )

            episodes = int(
                st.number_input(
                    label="Episodes", value=EnvSelector.get_agent_episodes(env_name)
                )
            )

            discrete = (
                int(
                    st.number_input(
                        "Discrete Size", value=EnvSelector.get_agent_discrete(env_name)
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
                st.number_input(
                    label="Epsilon Decay Rate",
                    value=EnvSelector.get_agent_epsilon_decay(env_name),
                    format="%.5f",
                )
            )
            epsilon_min = float(
                st.number_input(
                    label="Epsilon Minimum",
                    value=EnvSelector.get_agent_epsilon_min(env_name),
                    format="%.5f",
                )
            )

        with cols[2]:
            lr = float(
                st.number_input(
                    label="Learning Rate",
                    value=EnvSelector.get_agent_lr(env_name),
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

            lr_decay = float(
                st.number_input(
                    label="Learning Rate Decay",
                    value=EnvSelector.get_agent_lr_decay(env_name),
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
            st.text(" ")
            st.text(" ")

            break_on_solve = st.checkbox(
                label=f"Break On Solve",
                value=EnvSelector.get_agent_break_on_solve(env_name),
            )

        return {
            "gamma": gamma,
            "episodes": episodes,
            "lr": lr,
            "lr_min": lr_min,
            "lr_decay": lr_decay,
            "epsilon": epsilon,
            "epsilon_decay": epsilon_decay,
            "epsilon_min": epsilon_min,
            "discrete": discrete,
            "count": count,
            "target": target,
            "break_on_solve": break_on_solve,
            "render_each": episodes + 1,  # IN ORDER TO AVOID RENDERING
        }

    @staticmethod
    def output_config(st) -> str:
        result_path = os.path.join(os.getcwd(), "Experiments")

        result_folder = st.text_input(label="Folder Name", value="result_pool_agent")

        save_path = None
        if not os.path.isdir(result_path):
            os.makedirs(result_folder)
        else:
            save_path = os.path.join(result_path, result_folder)
            st.info(save_path)

        return result_folder, save_path

    @staticmethod
    def train(st, env_name: str, save_path: str, all_configs: dict):
        cols = st.columns(3)

        validated = cols[0].checkbox("Everything is OK ^~^")
        disabled = not validated

        cols[2].download_button(
            "Download Configs JSON",
            file_name="pipeline_config.json",
            mime="application/json",
            data=json.dumps(all_configs),
            disabled=disabled,
        )

        if cols[1].button("Start Training", disabled=disabled):
            experiment = PoolMaker(
                save_path=save_path,
                env_maker=EnvSelector.get_env_maker(env_name),
                env_configs=all_configs["environment"]["configs"],
                agent_config=all_configs["agent"],
            )

            save_json(
                data=all_configs, file_path=os.path.join(save_path, "config.json")
            )

            experiment.run()
            experiment.save()
            st.dataframe(experiment.results_df)

    @staticmethod
    def set_config(st):
        st.set_page_config(
            page_icon="👷",
            page_title="Agent Pool Creator",
            initial_sidebar_state="expanded",
        )
        Components.set_center(st)

    @staticmethod
    def get_all_configs(env_name: str, parameters: dict, agent_configs: dict) -> dict:
        return {
            "environment": {"name": env_name, **PoolMaker.create_config(**parameters)},
            "agent": agent_configs,
        }
