import streamlit as st
from pages import generate_data, bootstrapping

pages = {
    'Generate Data': generate_data,
    'Bootstrapping': bootstrapping
}

# sidebar pages
if len(pages):
    st.sidebar.subheader("Insurance Portfolio generator")
    selected_page = st.sidebar.radio("Steps:", pages, key=1)
else:
    selected_page = "Generate Data"


# Draw main page
pages[selected_page].show()
