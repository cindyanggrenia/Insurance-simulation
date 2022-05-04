import streamlit as st
import pandas as pd
import numpy as np
import const
import altair as alt


def show():
    rng = np.random.default_rng(42)

    formatting = const.formatting

    st.title("Generate insurance data")

    st.write("""
    This simulator application generate claims data based on various user inputs to allows varying level of complexity of the dataset generated.
    
    Traditionally, claims are modelled with different distibution for frequency and claims severity. Usually, Poisson distribution is used to model frequency and Gamma distibution is used to model severity.
    """)

    st.subheader("Sum Under Risk and Premium setup")
    st.write("""
    The liabilities is biased sampling based on brackets proportion
    """)

    with st.form('SUR and Premium form') as f:
        from st_aggrid import GridOptionsBuilder, AgGrid

        n_policies = int(st.number_input('Number of Policies', 100, 50000, 5000))
        rate_on_line = st.slider('Rate on Line', 0.001, 0.1, 0.02)  # equivalent to Exposure

        df_premium_template = pd.DataFrame({
            const.SUR: [1e6, 2e6, 3e6, 5e6, 10e6, '', ''],
            const.BRACKET_P: [0.4, 0.3, 0.15, 0.12, 0.03, '', '']
        })

        ob = GridOptionsBuilder.from_dataframe(df_premium_template)
        ob.configure_column(const.SUR, type=["numericColumn"], editable=True)
        ob.configure_column(const.BRACKET_P, type=["numericColumn"], editable=True)

        response = AgGrid(df_premium_template, ob.build(), height=250, editable=True, fit_columns_on_grid_load=True)

        st.form_submit_button('Generate data')

    @st.cache
    def genPolicyPremium(n, loss: list, probability: list, rol: float) -> pd.DataFrame:
        """Generate policy dataframe with liability and premium generated from liability table and rate on line"""
        data = pd.DataFrame({const.POLICY_ID: [f'p{i}' for i in range(n)],
                             const.SUR: rng.choice(loss, p=probability, size=n)})
        data.set_index(const.POLICY_ID, inplace=True)
        data[const.ROL] = rng.normal(loc=1 / rol, scale=10, size=n).clip(1, None)
        data[const.PREMIUM] = (np.round(data[const.SUR] / data[const.ROL], 2))
        return data

    dfp = response['data']
    dfp = dfp[dfp[const.SUR] != ''].astype(float).sort_values(const.SUR).reset_index(drop=True)
    dfp[const.BRACKET_P] = dfp[const.BRACKET_P].div(dfp[const.BRACKET_P].sum())  # normalize to total 100%

    dfp = dfp[[const.SUR, const.BRACKET_P]]
    dfp_styler = dfp.style.format(formatting)
    df_policy = genPolicyPremium(n_policies, dfp[const.SUR], dfp[const.BRACKET_P], rate_on_line)

    with st.expander('See explanation', expanded=False):
        st.dataframe(dfp_styler)

        st.write("""
        The premium is based on static rate-on-line rates with added normal 
        distribution noise to simulate a price created via a marginalized risk tuning (rater)
        """)
        st.latex(r'''\small{
             \text{Pure premium} 
             = \frac{\text{Sum under risk}}{\text{Rate on line}} 
             = \text{Frequency} \times \text{Severity} .
             }''')
        st.write('\n\n')

        st.dataframe(df_policy.head(100).style.format('{:,.0f}'))

        alt_hist = alt.Chart(
            df_policy,
            title='Bootstrapped premium distribution'
        ).mark_bar().encode(
            x=alt.X(const.PREMIUM + ':Q', bin=alt.BinParams(maxbins=50)),
            y='count()',
        )
        st.altair_chart(alt_hist, use_container_width=True)

    # ======================= generating claims probability ========================

    st.subheader("Claims generation")
    st.write("""
    The claims generating process is modelled as:
      - Claim frequency model: modelling the number of claims that happen for a partifular policy
      - Gamma: conditional latent probability of the claim severity (measured as losses per unit liability)
    """)

    def freq_beta_binomial():
        from scipy.stats import beta
        st.write('#### Beta-Binomial distribution')
        st.write("This conjugate distribution consists of beta distribution for modelling individual policy's "
                 "probability of making a claim. Binomial (or Bernoulli in this specific case) to model whether "
                 "the claim did happen of not, given the probability")

        col201, col202 = st.columns(2)
        with col201:
            beta_a = st.slider('α parameter', 0.0, 10.0, 5.0, format='%.1f')
        with col202:
            beta_b = st.slider('β parameter', 0, 100, 35)

        x = np.linspace(0.00001, 0.3, 500)
        df_graph = pd.DataFrame({'x': x,
                                 f'a={beta_a}, b={beta_b}': beta.pdf(x, a=beta_a, b=beta_b),
                                 'a=1, b=30': beta.pdf(x, a=1, b=30),
                                 'a=5, b=50': beta.pdf(x, a=5, b=50),
                                 'a=4, b=100': beta.pdf(x, a=4, b=100),
                                 'a=2, b=150': beta.pdf(x, a=2, b=150),
                                 'a=2, b=80': beta.pdf(x, a=2, b=80)
                                 })

        @st.cache
        def genClaimProbabiliy(data_in, a, b):
            """Add latent claims probability to existing policy dataframe"""
            data = data_in.copy()
            data[const.LATENT_CLAIM_P] = rng.beta(a=a, b=b, size=data.shape[0])
            data[const.HAS_CLAIM] = rng.binomial(n=1, p=data[const.LATENT_CLAIM_P])  # assume only 1 claim per policy
            return data

        st.session_state.freq_model = 'Beta-Binomial'
        st.session_state.freq_model_param = {'a': beta_a, 'b': beta_b}

        df = genClaimProbabiliy(df_policy, a=beta_a, b=beta_b)
        return df_graph, df

    def freq_poisson():
        from scipy.stats import poisson
        st.write('#### Poisson distribution')
        st.write("Very commonly used to model random occurrences in a certain time period. "
                 "The rate of occurrence is parameterized a constant, λ")

        col201, col202 = st.columns(2)
        with col201:
            mu = st.slider('λ parameter', 0.0, 1.0, 0.12, format='%.2f')

        x = np.linspace(0, 5, 6)
        df_graph = pd.DataFrame({'x': x,
                                 f'λ={mu}': poisson.pmf(x, mu=mu),
                                 'λ=1': poisson.pmf(x, mu=1),
                                 'λ=2': poisson.pmf(x, mu=2),
                                 'λ=0.5': poisson.pmf(x, mu=0.5),
                                 'λ=0.1': poisson.pmf(x, mu=0.1),
                                 # 'a=2, b=80': poisson.pdf(x, a=2, b=80),
                                 })

        @st.cache
        def genClaimProbabiliy(data_in, mu):
            """Add latent claims probability to existing policy dataframe"""
            data = data_in.copy()
            data[const.HAS_CLAIM] = rng.poisson(lam=mu, size=data.shape[0])
            return data

        st.session_state.freq_model = 'Poisson'
        st.session_state.freq_model_param = {'mu': mu}

        df = genClaimProbabiliy(df_policy, mu)
        return df_graph, df

    st.write('#### Claim Frequency distribution')
    freq_model = {'Beta-Binomial': freq_beta_binomial, 'Poisson': freq_poisson}
    # TODO: add negative binomial, poisson gamma
    selected_freq_model = st.selectbox('Choises of model:', options=freq_model.keys(), index=1)

    df_freq, df = freq_model[selected_freq_model]()

    alt_data = df_freq.melt('x')
    alt_data.rename(columns={'value': 'pdf', 'variable': 'params'}, inplace=True)
    alt_data['colours'] = alt_data['params']
    colours = dict(zip(
        df_freq.columns[1:7],
        ['#FF4B4B', '#3a598a', '#5d729d', '#7d8db0', '#9da8c3', '#bdc4d7']))
    alt_data['colours'] = alt_data['colours'].replace(colours)

    if selected_freq_model == 'Poisson':
        x_axis = alt.X('x', axis=alt.Axis(values=list(range(0, 5, 1))))
    else:
        x_axis = 'x'

    alt_freq = alt.Chart(alt_data).mark_line().encode(
        x=x_axis,
        y='pdf',
        color=alt.Color('colours', scale=None),
        tooltip=[alt.Tooltip('x', format=',.2f'),
                 alt.Tooltip('pdf', format=',.3f')]
    )
    alt_text = alt.Chart(alt_data).mark_text(align='left', dy=-5).encode(
        alt.X('x:Q', aggregate={'argmax': 'pdf'}, title='x'),
        alt.Y('pdf:Q', aggregate='max', title='pdf'),
        alt.Text('params'),
        color=alt.Color('colours', scale=None)
    )
    st.altair_chart((alt_freq + alt_text), use_container_width=True)

    with st.expander('Show generated claims probability and occurance', expanded=False):
        st.dataframe(df.head(100).style.format(formatting))
        st.text(f"""
        Number of policies: {df.shape[0]}
        Total premium: {df[const.PREMIUM].sum():,.0f}
        Number of claims: {df[const.HAS_CLAIM].sum()}""")

        st.write(df[[const.HAS_CLAIM]]
                 .groupby(const.HAS_CLAIM)
                 .agg(counts=pd.NamedAgg(column=const.HAS_CLAIM, aggfunc="count"))
                 .reset_index())

    # ======================= generating claims severity ========================

    @st.cache
    def genClaimSeverity(data_in, a, scale):
        """Add latent claims severity and calculate claims amount"""
        data = data_in.copy()
        data[const.LATENT_SEV] = rng.gamma(shape=a, scale=scale, size=data.shape[0])
        data[const.CLAIMS] = data[const.HAS_CLAIM] * data[const.LATENT_SEV] * data[const.SUR]
        return data

    st.write('#### Gamma distribution')
    st.write("This distribution models claim severity. It has two parameters: shape parameter k and scale parameter θ")

    from scipy.stats import gamma

    col301, col302 = st.columns(2)
    with col301:
        gamma_a = st.slider('Shape parameter k', 0.01, 5.0, 1.6, format='%.2f')
    with col302:
        gamma_scale = st.slider('Scale parameter θ', 0.01, 0.5, 0.1, step=0.005, format='%.3f')

    args = {'a': gamma_a, 'scale': gamma_scale}

    x = np.linspace(0.00001, 1, 500)
    df_gamma = pd.DataFrame({'x': x,
                             f'a={gamma_a}, θ={gamma_scale}': gamma.pdf(x, a=gamma_a, scale=gamma_scale),
                             'a=1, θ=2': gamma.pdf(x, a=1, scale=2),
                             'a=2, θ=0.1': gamma.pdf(x, a=2, scale=0.1),
                             'a=2, θ=0.2': gamma.pdf(x, a=2, scale=0.2),
                             'a=1, θ=1': gamma.pdf(x, a=1, scale=1),
                             'a=1, θ=0.3': gamma.pdf(x, a=1, scale=0.3),
                             # 'a=7.5, θ=1': gamma.pdf(x, a=7.5, scale=1)
                             })

    alt_data = df_gamma.melt('x')
    alt_data.rename(columns={'value': 'pdf', 'variable': 'params'}, inplace=True)
    alt_data['colours'] = alt_data['params']
    colours = dict(zip(
        df_gamma.columns[1:7],
        ['#FF4B4B', '#3a598a', '#5d729d', '#7d8db0', '#9da8c3', '#bdc4d7']))
    alt_data['colours'] = alt_data['colours'].replace(colours)
    alt_beta = alt.Chart(alt_data).mark_line().encode(
        x='x',
        y=alt.Y('pdf'),
        color=alt.Color('colours', scale=None),
        tooltip=[alt.Tooltip('x', format=',.2f'),
                 alt.Tooltip('pdf', format=',.3f')]
    )
    alt_text = alt.Chart(alt_data).mark_text(align='left', dy=-5).encode(
        alt.X('x:Q', aggregate={'argmax': 'pdf'}, title='x'),
        alt.Y('pdf:Q', aggregate='max', title='pdf'),
        alt.Text('params'),
        color=alt.Color('colours', scale=None)
    )
    st.altair_chart((alt_beta + alt_text), use_container_width=True)

    df = genClaimSeverity(df, gamma_a, gamma_scale)

    with st.expander('Show generated claims severity and claims amount', expanded=True):
        st.dataframe(df.head(100).style.format(formatting))
        st.text(f"""
        Number of policies: {df.shape[0]}
        Total premium: {df[const.PREMIUM].sum():,.0f}
        Number of claims: {df[const.HAS_CLAIM].sum()}
        Total claims: {df[const.CLAIMS].sum():,.0f}""")

    # if st.button('Save data and go to bootstrapping 🚀'):
    st.session_state.df = df
    # st.write('`Data saved`')
