import streamlit as st
from modules.components import Components
from modules.environments import EnvSelector
from modules.components.exrl import ExRLPage


ExRLPage.set_configs(st)
"# 🏃‍♂️ Ex-RL"
"## HMM Selection"

pool, phase_map, sequences, hmm_model, hmm_config, hmm_path = ExRLPage.get_configs(st)

Components.separator(st)
"## Agent Training Configs"
agent_configs = ExRLPage.agent_configs(st, pool.env_name)


Components.separator(st)
"## Environment Parameters"
parameters, env_name = ExRLPage.env_parameters_selector(st, pool.env_name)
env_maker = EnvSelector.get_env_maker(env_name)


Components.separator(st)
"## Run Configs"
run_configs = ExRLPage.get_run_configs(st, hmm_path)


if st.button("Start Experiment"):
    Components.separator(st)
    "## Results"
    env = env_maker(**parameters)
    reward_shaper_ql, reward_shaper_hmm = ExRLPage.init_experiment(
        st=st,
        hmm_model=hmm_model,
        sequences=sequences,
        env=env,
        agent_configs=agent_configs,
        phase_map=phase_map,
        **run_configs,
    )

    ExRLPage.save_configs(agent_configs, env_name, parameters, run_configs)

    history = ExRLPage.run_experiments(
        st=st,
        env=env,
        phase_map=phase_map,
        agent_configs=agent_configs,
        reward_shaper_ql=reward_shaper_ql,
        reward_shaper_hmm=reward_shaper_hmm,
        **run_configs,
    )

    ExRLPage.show_save_run_results(
        st,
        history,
        episodes=agent_configs["episodes"],
        **run_configs,
    )
