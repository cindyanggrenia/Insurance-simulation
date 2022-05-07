import streamlit as st
from pages import generate_data, loss_analysis, bootstrapping

st.set_page_config(page_title='Insurance Simulation', page_icon='📓')
pages = {
    'Generate Data': generate_data,
    'Loss analysis': loss_analysis,
    'Statistics estimation with bootstrapping': bootstrapping
}

# sidebar pages
if len(pages):
    st.sidebar.subheader("Insurance Portfolio generator")
    selected_page = st.sidebar.radio("Steps:", pages, key=1)
else:
    selected_page = "Generate Data"


# Draw main page
pages[selected_page].show()
