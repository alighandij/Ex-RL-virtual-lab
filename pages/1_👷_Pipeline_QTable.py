import streamlit as st
from modules.components import Components
from modules.components.pipeline import PipelinePage
from stqdm import stqdm as tqdm

PipelinePage.set_config(st)
"# 👷 Pipeline"

"## Environment Configs"
env_name = PipelinePage.env_name_selector(st)
"### Parameters"
parameters = PipelinePage.env_parameters_selector(st, env_name)
total_configs = PipelinePage.total_configs(st, parameters)

Components.seperator(st)
"## Q-Table Agent Training Configs"
agent_configs = PipelinePage.agent_configs(st, env_name)

Components.seperator(st)
"### Output Configs"
result_folder, save_path = PipelinePage.output_config(st)

Components.seperator(st)
"### Train"
all_configs = PipelinePage.get_all_configs(env_name, parameters, agent_configs)
PipelinePage.train(st, env_name, save_path, all_configs)

"#### Configs"
st.json(all_configs, expanded=False)
