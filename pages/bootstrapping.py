import streamlit as st
import pandas as pd
import numpy as np
import const
import altair as alt
from helper import saved_df


def show():
    st.title('Bootstrapping')

    if saved_df().df==None:
        st.write('Save data in "Generate Data" step first')
        st.stop()

    df = saved_df().df.copy()

    st.write(f"""
    Summary of the data from step 1 so far:
    
    - Number of policies: {df.shape[0]}
    - Total premium: {df[const.PREMIUM].sum():,.0f}
    - Number of claims: {df[const.HAS_CLAIM].sum()}
    - Total claims: {df[const.CLAIMS].sum():,.0f}
    - Overall Loss Ratio: {df[const.CLAIMS].sum() / df[const.PREMIUM].sum():.0%}
    
    Here, we seem to have had a great year. But what about the years after?
    What is the uncertainty of our portfolio?
    """)

    # =========== Bootstrapping start ===========

    TTL_PREMIUM = const.PREMIUM + '_total'
    TTL_CLAIMS = const.CLAIMS + '_total'
    TTL_LR = 'LR_overall'
    boot_formatting = {TTL_PREMIUM: '{:,.0f}', TTL_CLAIMS: '{:,.0f}',
                       TTL_LR: '{:.0%}', 'loss': '{:,.0f}', }

    @st.cache(allow_output_mutation=True)
    def bootstrap_lr(data_in, nboot: int = 1000, p: float = 1):
        data = data_in.copy()
        rng = np.random.default_rng(42)

        new_size = int(len(data) * p)
        sample_idx = rng.integers(0, new_size, size=(new_size, nboot))
        premium_boot = data[const.PREMIUM].values[sample_idx]
        claims_boot = data[const.CLAIMS].values[sample_idx]

        dfboot = pd.DataFrame({
            TTL_PREMIUM: premium_boot.sum(axis=0),
            TTL_CLAIMS: claims_boot.sum(axis=0)})

        dfboot[TTL_LR] = dfboot[TTL_CLAIMS] / dfboot[TTL_PREMIUM]

        return dfboot

    st.markdown('<h4>Bootstrap to simulate multi-year portfolio</h4>', unsafe_allow_html=True)
    st.write(f"")

    with st.form('Bootstrapping option form') as f:
        n_boot = st.number_input('Number of portfolio simulation', 100, 10000, 1000)
        perc = st.number_input('% of original portfolio size', 0.05, 2.0, 1.0)

        st.form_submit_button('Bootstrap data')

    # if not submit_button:
    #     st.stop()

    df_boot = bootstrap_lr(df, int(n_boot), perc)

    # draw scatter plot
    st.text("")

    df_boot['loss'] = df_boot[TTL_PREMIUM] - df_boot[TTL_CLAIMS]
    c = alt.Chart(df_boot).mark_circle().encode(
        x=alt.X(TTL_PREMIUM, scale=alt.Scale(zero=False)),
        y=alt.Y(TTL_CLAIMS, scale=alt.Scale(zero=False)),
        color=TTL_LR,
        tooltip=[TTL_PREMIUM, TTL_CLAIMS, TTL_LR]
    )
    st.altair_chart(c, use_container_width=True)

    # draw box plot
    st.write('Distribution of bootstrap portfolio loss ratio')
    c = alt.Chart(
        df_boot
    ).transform_density(
        TTL_LR,
        as_=[TTL_LR, "density"],
    ).mark_area(
        opacity=0.1,
        line={'color': 'steelblue'}
    ).encode(
        x=alt.X(TTL_LR, scale=alt.Scale(zero=False)),
        y=alt.Y("density:Q", title=None),
    ).properties(
        height=200
    )
    st.altair_chart(c, use_container_width=True)

    st.write(df_boot.describe().loc[['mean', 'std', 'min', 'max']]
             .style.format(boot_formatting))

    # Summary text
    iq_range = st.slider('Quartile range', 0.0, 1.0, (0.25, 0.75), 0.05)
    range = iq_range[1] - iq_range[0]
    iqr_value = df_boot[TTL_LR].quantile(list(iq_range)).tolist()
    # st.write(str(iqr_value))
    st.write(f"""
    Based on this result, we conclude that our portfolio LR:
    - Has a mean of {df_boot[TTL_LR].mean():.0%}
    - On a given year, will have LR between ({iqr_value[0]:.0%}, {iqr_value[1]:.0%}) with {range:.0%} confidence
     
    """)
