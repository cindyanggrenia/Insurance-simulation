import streamlit as st
from pages import generate_data, loss_analysis, bootstrapping


def create_link(links: list):
    piece = """<li><a href="{}" style="color:inherit; text-decoration:underline">{}</a></li>"""
    return ' '.join([piece.format(x, y) for (x, y) in links])


st.set_page_config(page_title='Insurance Simulation', page_icon='📓')
readme = open('README.md', 'r').read()
pages = {
    'Welcome': st.markdown(readme, unsafe_allow_html=True),
    'Generate data': generate_data,
    'Loss analysis': loss_analysis,
    'Simulation and resampling': bootstrapping
}

subsection = {
    'Generate data': [
        ("#sum-under-risk-and-premium-setup", "Setup sum under risk and premium"),
        ("#claim-frequency-generation", "Generate claims frequency"),
        ("#claim-frequency-generation", "Generate claims severity")],
    'Loss analysis': [
        ("#1-estimating-frequency-distribution", "Estimating frequency distribution"),
        ("#2-estimating-severity-distribution", "Estimating severity distribution"),
        ("#3-empirical-estimator", "Empirical estimator")],
    'Simulation and resampling': [
        ("#bootstrap-configuration", "Bootstrap configuration"),
        ("#estimating-loss-ratio-statistics", "Estimating Loss Ratio statistics"),]
}

# sidebar pages
if len(pages):

    st.sidebar.subheader("Insurance portfolio generator and analysis")
    selected_page = st.sidebar.radio("Pages", pages, key=1)
    st.sidebar.write("Jump to section")
    if selected_page != 'Welcome':
        st.sidebar.markdown(f"<ul>{create_link(subsection[selected_page])}</ul>", unsafe_allow_html=True)

else:
    selected_page = "Welcome"

# Draw main page
if selected_page != 'Welcome':
    pages[selected_page].show()
