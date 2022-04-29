import streamlit as st


def saved_df():
    if 'df' not in st.session_state:
        st.session_state.df = None
    return st.session_state
