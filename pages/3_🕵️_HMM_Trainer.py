import os
import streamlit as st
from modules.utils import get_time_str
from modules.components import Components
from modules.components.hmm import HMMPage

HMMPage.set_configs(st)
"# 🕵️ HMM Trainer"
"## Pool Selection"
pool_path, pool = Components.get_pool(st)
folder = st.text_input(label="Result Folder", value="HMM") + "_" + get_time_str()
result_path = os.path.join(pool_path, "HMMs", folder)

Components.separator(st)
"## Config"
n_components, n_iter, n_steps, _class, _class_name = HMMPage.get_configs(st)

Components.separator(st)
"## Sequence Selection"
selections = HMMPage.get_sequence_selection(st, pool)
filters = HMMPage.filter_pool(st, pool, selections)

Components.separator(st)
"## Sequence Encoder"
encoder, encoder_name, encode_count = HMMPage.get_encoder(st, pool)
st.write(f"Encode Count: {encode_count}")
if st.button("Train"):
    Components.separator(st)
    st.write("## Report")
    sequences = HMMPage.get_sequences(encoder, pool, filters, selections, n_steps)
    HMMPage.save_sequences(st, sequences, result_path)
    HMMPage.save_configs(
        result_path,
        filters,
        selections,
        encoder_name,
        encode_count,
        n_components,
        n_iter,
        n_steps,
        _class_name
    )

    hmm = HMMPage.train_hmm(
        sequences, encode_count, n_iter, n_components, result_path, _class
    )
    HMMPage.hmm_report(st, hmm, result_path)
