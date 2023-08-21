import streamlit as st
from modules.components import Components

st.set_page_config(
    page_icon="🎮",
    page_title="Review Agent",
    initial_sidebar_state="expanded",
)

"# 📉🎮 Review Agent"

pool_path, pool = Components.get_pool(st, "pool_hmm_select")
agent_id, agent = Components.agent_selector(st, pool)

if st.button("🎮 Play Episode"):
    reward, obs = agent.play(True)
    agent.env.close()
    

st.pyplot(agent.history.plot())
