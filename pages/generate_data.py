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
    
    Traditionally, claims are modelled with different distibution for frequency and claims severity. 
    In this application, we are going to
    1. **Setup sum under risk and premium:** First we setup the policies, liability, and premium model
    2. **Generate claims frequency:** Next, we generate occurances of claims
    3. **Generate claims severity:** Finally, we generate the level of severity of claims
    """)

    st.subheader("Sum Under Risk and Premium setup")
    st.write("""
    The liabilities is biased sampling based on brackets proportion
    """)

    with st.form('SUR and Premium form') as f:
        from st_aggrid import GridOptionsBuilder, AgGrid, JsCode

        n_policies = int(st.number_input('Number of Policies', 100, 50000, 5000))
        rate_on_line = st.slider('Rate on Line', 0.001, 0.1, 0.02)  # equivalent to Exposure

        df_premium_template = pd.DataFrame({
            const.SUR: [1e6, 2e6, 3e6, 5e6, 10e6, '', ''],
            const.BRACKET_P: [0.4, 0.3, 0.15, 0.12, 0.03, '', '']
        })

        ob = GridOptionsBuilder.from_dataframe(df_premium_template)
        ob.configure_column(const.SUR, type=["numericColumn"], editable=True)
        ob.configure_column(const.BRACKET_P, type=["numericColumn"], editable=True)

        response = AgGrid(df_premium_template, ob.build(), height=250, editable=True, fit_columns_on_grid_load=True,
                          allow_unsafe_jscode=True)

        st.form_submit_button('Generate data')

    @st.experimental_memo
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

    with st.expander('See explanation', expanded=True):
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
            title='Generated premium distribution'
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
      - Claim severity model: modelling latent probability of claim severity (losses per unit liability)
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

        @st.experimental_memo
        def genClaimProbabiliy(data, a, b):
            """Add latent claims probability to existing policy dataframe"""
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

        @st.experimental_memo
        def genClaimProbabiliy(data, mu):
            """Add latent claims probability to existing policy dataframe"""
            data[const.HAS_CLAIM] = rng.poisson(lam=mu, size=data.shape[0])
            return data

        st.session_state.freq_model = 'Poisson'
        st.session_state.freq_model_param = {'mu': mu}

        df = genClaimProbabiliy(df_policy, mu)
        return df_graph, df

    def freq_nbinom():
        from scipy.stats import nbinom
        st.write('#### Negative Binomial distribution')
        st.write("This distribution model the number of successes until we observe the rth failure "
                 "in independent repetitions of binary outcomes"
                 "The negative binomial is parameterized by n and p")

        col201, col202 = st.columns(2)
        with col201:
            n = st.slider('n parameter', 0.0, 10.0, 0.5)
        with col202:
            p = st.slider('p parameter', 0.0, 1.0, 0.85)

        x = np.linspace(0, 10, 11)
        df_graph = pd.DataFrame({'x': x,
                                 f'n={n}, p={p}': nbinom.pmf(x, n=n, p=p),
                                 'n=3, p=0.6': nbinom.pmf(x, n=3, p=.6),
                                 'n=2, p=0.7': nbinom.pmf(x, n=2, p=.7),
                                 'n=4, p=0.5': nbinom.pmf(x, n=4, p=0.5),
                                 'n=1, p=0.1': nbinom.pmf(x, n=1, p=0.1),
                                 # 'a=2, b=80': poisson.pdf(x, a=2, b=80),
                                 })

        @st.experimental_memo
        def genClaimProbabiliy(data, n, p):
            """Add latent claims probability to existing policy dataframe"""
            data[const.HAS_CLAIM] = rng.negative_binomial(n, p, size=data.shape[0])
            return data

        st.session_state.freq_model = 'Negative Binomial'
        st.session_state.freq_model_param = {'n': n, 'p': p}

        df = genClaimProbabiliy(df_policy, n, p)
        return df_graph, df

    st.write('#### Claim Frequency Generation')
    freq_model = {'Beta-Binomial': freq_beta_binomial, 'Poisson': freq_poisson, 'Negative Binomial': freq_nbinom}

    selected_freq_model = st.selectbox('Choises of model:', options=freq_model.keys(), index=1)

    df_freq, df_claim_freq = freq_model[selected_freq_model]()

    alt_data = df_freq.melt('x')
    alt_data.rename(columns={'value': 'pdf', 'variable': 'params'}, inplace=True)
    alt_data['colours'] = alt_data['params']
    colours = dict(zip(
        df_freq.columns[1:7],
        ['#FF4B4B', '#9da8c3', '#7d8db0', '#3a598a', '#bdc4d7', '#5d729d']))
    alt_data['colours'] = alt_data['colours'].replace(colours)

    if selected_freq_model == 'Poisson':
        x_axis = alt.X('x', axis=alt.Axis(values=list(range(0, 5, 1))))
    elif selected_freq_model == 'Negative Binomial':
        x_axis = alt.X('x', axis=alt.Axis(values=list(range(0, 10, 1))))
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

    with st.expander('Show generated claims probability and occurance', expanded=True):
        st.dataframe(df_claim_freq.head(100).style.format(formatting))
        st.write(f"""
        - Number of policies: {df_claim_freq.shape[0]}
        - Total premium: {df_claim_freq[const.PREMIUM].sum():,.0f}
        - Number of claims: {df_claim_freq[const.HAS_CLAIM].sum()}""")

        st.write(df_claim_freq[[const.HAS_CLAIM]]
                 .groupby(const.HAS_CLAIM)
                 .agg(counts=pd.NamedAgg(column=const.HAS_CLAIM, aggfunc="count"))
                 .reset_index())

    # ======================= generating claims severity ========================

    def sev_gamma():
        from scipy.stats import gamma

        st.write("""
        #### Gamma distribution
        This distribution is often used to model claim severity.
        The gamma distribution is considered a light tailed distribution, 
        which may not be suitable to model risky asset with severe losses.
        It has two parameters: shape parameter k and scale parameter θ
        """)

        a_default = 1.6
        scale_default = 0.1

        if 'sev_model' in st.session_state:
            if st.session_state.sev_model == 'Gamma':
                a_default = st.session_state.sev_model_param['a']
                scale_default = st.session_state.sev_model_param['scale']

        col301, col302 = st.columns(2)
        with col301:
            gamma_a = st.slider('Shape parameter k', 0.01, 5.0, a_default, format='%.2f')
        with col302:
            gamma_scale = st.slider('Scale parameter θ', 0.01, 0.5, scale_default, step=0.005, format='%.3f')

        x = np.linspace(0.00001, 1, 500)
        df_graph = pd.DataFrame({'x': x,
                                 f'a={gamma_a}, θ={gamma_scale}': gamma.pdf(x, a=gamma_a, scale=gamma_scale),
                                 'a=1, θ=2': gamma.pdf(x, a=1, scale=2),
                                 'a=2, θ=0.1': gamma.pdf(x, a=2, scale=0.1),
                                 'a=2, θ=0.2': gamma.pdf(x, a=2, scale=0.2),
                                 'a=1, θ=1': gamma.pdf(x, a=1, scale=1),
                                 'a=1, θ=0.3': gamma.pdf(x, a=1, scale=0.3),
                                 # 'a=7.5, θ=1': gamma.pdf(x, a=7.5, scale=1)
                                 })

        @st.experimental_memo
        def genClaimSeverity(data, a, scale):
            """Add latent claims severity and calculate claims amount"""
            data[const.LATENT_SEV] = rng.gamma(shape=a, scale=scale, size=data.shape[0])
            data[const.CLAIMS] = data[const.HAS_CLAIM] * data[const.LATENT_SEV] * data[const.SUR]
            return data

        st.session_state.sev_model = 'Gamma'
        st.session_state.sev_model_param = {'a': gamma_a, 'scale': gamma_scale}

        df = genClaimSeverity(df_claim_freq, a=gamma_a, scale=gamma_scale)
        return df_graph, df


    def sev_pareto():
        from scipy.stats import pareto

        st.write("""
        #### Pareto distribution
        It is a positively skewed and heavy-tailed distribution which makes it suitable for modeling income, 
        high-risk insurance claims and severity of large casualty losses. 
        It has two parameters: shape parameter b and scale parameter θ
        """)

        b_default = 5.0
        scale_default = 0.3

        if 'sev_model' in st.session_state:
            if st.session_state.sev_model == 'Pareto':
                b_default = st.session_state.sev_model_param['b']
                scale_default = st.session_state.sev_model_param['scale']

        col301, col302 = st.columns(2)
        with col301:
            pareto_b = st.slider('Shape parameter b', 0.01, 10.0, b_default, format='%.2f')
        with col302:
            pareto_scale = st.slider('Scale parameter θ', 0.0, 2.0, scale_default, format='%.2f')

        x = np.linspace(-0.1, 0.4, 500)
        df_graph = pd.DataFrame({'x': x,
                                 f'b={pareto_b}, θ={pareto_scale}': pareto.pdf(x+pareto_scale, b=pareto_b, scale=pareto_scale),
                                 # 'b=3, θ=0.2': pareto.pdf(x+0.2, b=3, scale=0.2),
                                 'b=2, θ=0.1': pareto.pdf(x+0.1, b=2, scale=0.1),
                                 'b=2, θ=0.2': pareto.pdf(x+0.2, b=2, scale=0.2),
                                 'b=1, θ=1': pareto.pdf(x+1, b=1, scale=1),
                                 'b=1, θ=0.3': pareto.pdf(x+0.3, b=1, scale=0.3),
                                 # 'b=7.5, θ=1': pareto.pdf(x, b=7.5, scale=1)
                                 })

        @st.experimental_memo
        def genClaimSeverity(data, b, scale):
            """Add latent claims severity and calculate claims amount"""
            data[const.LATENT_SEV] = pareto.rvs(b, scale=scale, size=data.shape[0], random_state=rng)-scale
            data[const.CLAIMS] = data[const.HAS_CLAIM] * data[const.LATENT_SEV] * data[const.SUR]
            return data

        st.session_state.sev_model = 'Pareto'
        st.session_state.sev_model_param = {'b': pareto_b, 'scale': pareto_scale}

        df = genClaimSeverity(df_claim_freq, b=pareto_b, scale=pareto_scale)
        return df_graph, df

    def sev_weibull():
        from scipy.stats import weibull_min

        st.write("""
        #### Weibull distribution
        Weibull is widely used in reliability, life data analysis, weather forecasts and general insurance claims. 
        It is particularly useful to model truncated claims data. 
        It has two parameters: shape parameter c and scale parameter θ
        """)

        c_default = 1.52
        scale_default = 0.12

        if 'sev_model' in st.session_state:
            if st.session_state.sev_model == 'Weibull':
                c_default = st.session_state.sev_model_param['c']
                scale_default = st.session_state.sev_model_param['scale']

        col301, col302 = st.columns(2)
        with col301:
            c = st.slider('Shape parameter c', 0.01, 10.0, c_default, format='%.2f')
        with col302:
            scale = st.slider('Scale parameter θ', 0.0, 1.0, scale_default, format='%.2f')

        x = np.linspace(0, 1, 500)
        df_graph = pd.DataFrame({'x': x,
                                 f'c={c}, θ={scale}': weibull_min.pdf(x, c=c, scale=scale),
                                 'c=3, θ=0.2': weibull_min.pdf(x, c=3, scale=0.2),
                                 'c=2, θ=0.1': weibull_min.pdf(x, c=2, scale=0.1),
                                 'c=2, θ=0.2': weibull_min.pdf(x, c=2, scale=0.2),
                                 'c=1, θ=1': weibull_min.pdf(x, c=1.75, scale=0.5),
                                 # 'c=1, θ=0.3': weibull_min.pdf(x, c=1, scale=0.3),
                                 # 'c=7.5, θ=1': weibull_min.pdf(x, c=7.5, scale=1)
                                 })

        @st.experimental_memo
        def genClaimSeverity(data, c, scale):
            """Add latent claims severity and calculate claims amount"""
            data[const.LATENT_SEV] = weibull_min.rvs(c, scale=scale, size=data.shape[0], random_state=rng)
            data[const.CLAIMS] = data[const.HAS_CLAIM] * data[const.LATENT_SEV] * data[const.SUR]
            return data

        st.session_state.sev_model = 'Weibull'
        st.session_state.sev_model_param = {'c': c, 'scale': scale}

        df = genClaimSeverity(df_claim_freq, c=c, scale=scale)
        return df_graph, df


    st.write('#### Claim Severity Generation')
    sev_model = {'Gamma': sev_gamma, 'Pareto': sev_pareto, 'Weibull': sev_weibull}
    sev_model_index = {'Gamma': 0, 'Pareto': 1, 'Weibull': 2}

    if 'sev_model' in st.session_state:
        sev_model_default = sev_model_index[st.session_state.sev_model]
    else:
        sev_model_default = 0

    selected_sev_model = st.selectbox('Choises of model:', options=sev_model.keys(), index=sev_model_default)

    df_graph, df = sev_model[selected_sev_model]()

    alt_data = df_graph.melt('x')
    alt_data.rename(columns={'value': 'pdf', 'variable': 'params'}, inplace=True)
    alt_data['colours'] = alt_data['params']
    colours = dict(zip(
        df_graph.columns[1:7],
        ['#FF4B4B', '#9da8c3', '#7d8db0', '#bdc4d7', '#3a598a', '#5d729d']))
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

    with st.expander('Show generated claims severity and claims amount', expanded=True):
        st.dataframe(df.head(100).style.format(formatting))
        st.write(f"""
        - Number of policies: {df.shape[0]}
        - Total premium: {df[const.PREMIUM].sum():,.0f}
        - Number of claims: {df[const.HAS_CLAIM].sum()}
        - Total claims: {df[const.CLAIMS].sum():,.0f}""")

    # if st.button('Save data and go to bootstrapping 🚀'):
    st.session_state.df = df
    # st.write('`Data saved`')
