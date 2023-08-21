import os
import streamlit
from glob import glob
from modules.pools import Pool
import tqdm
from stqdm import stqdm

tqdm.tqdm = stqdm


class Components:
    @staticmethod
    def set_center(st: streamlit):
        st.markdown(
            """
            <style>
                .css-ocqkz7 {
                    // justify-content: center;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def separator(st: streamlit):
        st.write("-" * 8)

    @staticmethod
    def _pool_selector(st: streamlit, key: str = "pool_selector"):
        path = os.path.join(os.getcwd(), "Experiments", "*")
        return st.selectbox(
            label="Select Pool",
            options=glob(path),
            format_func=lambda x: x.split(os.sep)[-1],
            key=key,
        )

    def get_pool(st: streamlit, key: str = "pool_selector"):
        pool_path = Components._pool_selector(st, key)
        return pool_path, Pool(pool_path)

    def agent_selector(st: streamlit, pool: Pool, show_config: bool = False):
        agent_id = st.selectbox(
            label="Please Select An Agent", options=pool.agents.agent_id
        )
        st.write(pool.get_agent_info(agent_id))
        with st.expander("See All Agents"):
            st.write(pool.agents)
        agent = pool.load_agent(agent_id)
        if show_config:
            st.write("Agent Config:")
            st.json(pool.config["agent"], expanded=False)
        return agent_id, agent
