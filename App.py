import os
import streamlit as st

st.set_page_config(
    page_icon="🤖",
    page_title="Pipeline doc",
    initial_sidebar_state="expanded",
)

os.makedirs("Experiments", exist_ok=True)
with open("README.md", "r", encoding="utf8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)
