import streamlit as st
import pandas as pd
import numpy as np
import const
import altair as alt
from helper import saved_df


def show():
    st.write('# Loss Analysis')

    if not isinstance(saved_df().df, pd.DataFrame):
        st.write('Save data in "Generate Data" step first')
        st.stop()

    df = saved_df().df
    n = df.shape[0]

    st.write("""
    In this section we will analyse summary statistics and distribution of our generated data.
    """)
    st.write("""
    Given a set of observed data, we often wish to know the most suitable parameter of a distribution function 
    that best represent the data, i.e. maximises the likelihood ob the observed dataset. To do so, we will use 
    the **Maximum Likelihood Estimators (MLE)**, which is the value that maximises the likelihood of the observation.
    Additionally, we will use **Chi-Squared goodness of fit** test to check if the fitted distribution represents the data.
    """)

    st.write("### 1. Estimating frequency distribution")

    alt_freq = alt.Chart(df[[const.HAS_CLAIM]], height=alt.Step(30), ).mark_bar(size=20).encode(
        x='count()',
        y=const.HAS_CLAIM + ':O'
    )
    text = alt_freq.mark_text(align='center', dx=15, color='steelblue').encode(
        text='count()'
    )

    col101, col102 = st.columns([2, 1])
    with col101:
        st.altair_chart((alt_freq + text), use_container_width=True)
    with col102:
        df_counts = (df[[const.HAS_CLAIM]].groupby(const.HAS_CLAIM)
                     .agg(counts=pd.NamedAgg(column=const.HAS_CLAIM, aggfunc="count"))
                     .reset_index())
        df_counts.index = [''] * df_counts.shape[0]
        st.write(df_counts)

    st.write("Summary statistics")
    st.table(df[[const.HAS_CLAIM]].describe().loc[['mean', 'std', '25%', '50%', '75%', 'max']].T)

    # goodness of fit
    st.write("#### Goodness of Fit")

    freq_options = {'Beta-Binomial': 0, 'Poisson': 1, 'Negative-Binomial': 2}

    gen_model = st.session_state.freq_model
    col201, col202 = st.columns([1, 1])
    with col201:
        fitted_freq_model = st.selectbox("Fit data using distribution function:",
                                         options=freq_options,
                                         index=freq_options[gen_model] if gen_model in freq_options else 1)
    with col202:
        cl = st.number_input('Confidence level: ', 0.5, 0.999, 0.95, format='%.3f')

    from scipy.optimize import brute, differential_evolution

    def func(free_params, *args):
        dist, x = args
        # negative log-likelihood function
        ll = -np.log(dist.pmf(x, *free_params)).sum()
        if np.isnan(ll):  # occurs when x is outside of support
            ll = np.inf  # we don't want that
        return ll

    @st.experimental_memo
    def fit_discrete(dist, x, bounds, optimizer=brute):
        return optimizer(func, bounds, args=(dist, x))

    if fitted_freq_model == 'Poisson':
        mle = sum(df[const.HAS_CLAIM]) / n
        st.write(fr"""
        For Poisson distribution, the MLE is 
        $\small \displaystyle \widehat \lambda_\mathrm {{MLE}} =\frac {{1}}{{n}}\sum_{{i=1}}^{{n}}x_{{i}}.$ 
        In our dataset:
        - n = {df.shape[0]:,.0f}
        - sum of claims = {sum(df[const.HAS_CLAIM]):,.0f}
        - so, $\widehat\lambda_\mathrm{{MLE}} $ = {mle:,.3f}
        """)

    elif fitted_freq_model == 'Beta-Binomial':
        mle = sum(df[const.HAS_CLAIM]) / n
        st.write("For Beta-Binomial distribution, because of our assumption that one policy can only can one claim "
                 "(essentially Beta-Binomial wth n=1) the posterior distribution is equivalent to a Bernoulli with parameter "
                 r"$\widehat p_\mathrm{{MLE}} = \frac{\alpha}{\alpha + \beta}$. "
                 r"The MLE is the sample mean, which in our dataset is:")
        st.write(fr"""
        - n = {df.shape[0]:,.0f}
        - sum of claims = {sum(df[const.HAS_CLAIM]):,.0f}
        - $\widehat p_\mathrm{{MLE}} $  = {mle:,.3f}
        """)

    elif fitted_freq_model == 'Negative-Binomial':
        from scipy.stats import nbinom

        n_mle, p_mle = fit_discrete(nbinom, df[const.HAS_CLAIM], [(0, 10), (0, 1)])
        st.write("For Negative-Binomial distribution, the MLE cannot be expressed in closed form. "
                 "However, this can be easily estimated using numerical method or programatically:")
        st.write(fr"""
        - $\widehat n_\mathrm{{MLE}} $ = {n_mle:,.2f}
        - $\widehat p_\mathrm{{MLE}} $ = {p_mle:,.2f}
        """)

    # parameter comparison against generator
    if st.session_state.freq_model == 'Poisson':
        st.write(f"""
        Compare this with the original parameter used for generating the data, 
        λ is {st.session_state.freq_model_param['mu']:,.3f}""")
    elif st.session_state.freq_model == 'Negative-Binomial':
        param = st.session_state.freq_model_param
        st.write(f"""
        Compare this with the original parameter used for generating the data, 
        n is {param['n']:,.3f}, and p is {param['p']:,.3f}""")
    elif st.session_state.freq_model == 'Beta-Binomial':
        param = st.session_state.freq_model_param
        st.write(fr"""
        Compare this with the original parameter used for generating the data, $\frac{{\alpha}}{{\alpha + \beta}}$ is 
        {param['a'] / (param['a'] + param['b']) :,.3f}. Although, because multiple paid of α β satisfy this condition, 
        they are not recoverable.""")

    st.write("""
    We have tried to estimate the best parameter for a chosen distribution. 
    Even so, the fitted distribution may not be a good representation of the underlying distribution. 
    One method to check goodness-of-fit (how well the fitted distribution fits a set of observation) 
    is the Pearson’s chi-square test. This test evaluate how likely is it, that the difference 
    between two data sets happens by chance.
    """)

    if fitted_freq_model == 'Poisson':
        from scipy.stats import poisson
        st.write(fr'If we use the fitted Poisson with $\lambda = \widehat\lambda_\mathrm{{MLE}} $ = {mle:,.3f}')
        df_fit = pd.DataFrame({const.HAS_CLAIM: range(0, 10),
                               'Fitted Counts': [poisson.pmf(x, mle) * n for x in range(0, 10)]})

    elif fitted_freq_model == 'Negative-Binomial':
        st.write(fr'If we use the fitted Poisson with $n = \widehat n_\mathrm{{MLE}} $ = {n_mle:,.3f} and '
                 fr'$p = \widehat p_\mathrm{{MLE}} $ = {p_mle:,.3f}')
        df_fit = pd.DataFrame({const.HAS_CLAIM: range(0, 10),
                               'Fitted Counts': [nbinom.pmf(k=x, n=n_mle, p=p_mle) * n for x in range(0, 10)]})

    elif fitted_freq_model == 'Beta-Binomial':
        st.write(fr'If we use the fitted Beta-Binomial with $\widehat p_\mathrm{{MLE}} $ = {mle:,.3f}')
        df_fit = pd.DataFrame({const.HAS_CLAIM: [0, 1],
                               'Fitted Counts': [(1 - mle) * n, mle * n]})

    capped_at = max(df_fit.loc[df_fit['Fitted Counts'] >= 2, const.HAS_CLAIM].max(), df_counts[const.HAS_CLAIM].max())
    df_fit = df_fit[df_fit[const.HAS_CLAIM] <= capped_at]

    df_fit[const.HAS_CLAIM] = df_fit[const.HAS_CLAIM].astype(str)
    K = capped_at  # for chi-2 test number of category
    if n > df_fit['Fitted Counts'].sum():
        K = K + 1
        df_fit = df_fit.append({const.HAS_CLAIM: f'>{capped_at}',
                                'Fitted Counts': n - df_fit['Fitted Counts'].sum()},
                               ignore_index=True)

    df_counts[const.HAS_CLAIM] = df_counts[const.HAS_CLAIM].astype(str)
    df_diff = df_counts.merge(df_fit, on=const.HAS_CLAIM, how='outer').fillna(0)
    df_diff.index = [''] * df_diff.shape[0]
    df_diff.columns = ['No. claims', 'Obs. counts', 'Fitted counts']
    st.write(df_diff.style.format('{:,.1f}', subset=['Obs. counts', 'Fitted counts']))

    # chi square test
    from scipy.stats import chi2
    chi_square = ((df_diff['Obs. counts'] - df_diff['Fitted counts']) ** 2 / (n * df_diff['Fitted counts']))
    # chi_square.replace([np.inf, -np.inf], 0, inplace=True)
    st.latex(r"\small \frac{(\text{Obs. counts} - \text{Fitted counts})^2 }{n \times \text{Fitted counts}} = "
             f"{chi_square.sum():,.3f}")

    crit_value = chi2.ppf(cl, df=K - 1 - 1)  # number of parameters being estimated is 1 for both cases
    result = 'is a good fit' if crit_value >= chi_square.sum() else 'is not a good fit'
    st.write(f"Compare this to the chi-square distribution critical value at {cl:.1%} confidence "
             f"with {K - 1 - 1} degree of freedom: {crit_value:.2f}. So, we conclude the fitted distribution **{result}**.")

    # TODO: Model selection using test dataset / cross validation
    # TODO: Calculate information criteria such as AIC and BIC
    # TODO: Evaluation of model using qq and probplot

    st.write("""
    ### 2. Estimating severity distribution
    Below is summary of our claims severity measured as losses per unit liability.
    """)
    st.table(df[[const.LATENT_SEV]].describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].T)

    from scipy.stats import gamma, pareto, weibull_min, gaussian_kde

    bins = df[const.LATENT_SEV].max() / 60
    alt_sev = alt.Chart(df[[const.LATENT_SEV]]).transform_joinaggregate(
        total='count(*)'
    ).transform_calculate(
        pct=f'1 / {bins} / datum.total'
    ).mark_bar(opacity=0.2, color='lightslategray').encode(
        x=alt.X(const.LATENT_SEV + ':Q', bin=alt.BinParams(step=bins)),
        y='sum(pct):Q',
    )

    @st.experimental_memo
    def fit_severity(data):
        x = np.linspace(0, max(data), 300)
        gamma_args = gamma.fit(data)
        pareto_args = pareto.fit(data)
        weibull_args = weibull_min.fit(data)

        df_fitted = pd.DataFrame({'x': x,
                                  'data (kde)': gaussian_kde(data).pdf(x),
                                  'weibull': weibull_min.pdf(x, *weibull_args),
                                  'gamma': gamma.pdf(x, *gamma_args),
                                  'pareto': pareto.pdf(x, *pareto_args),
                                  })
        output = df_fitted.melt('x')
        output.rename(columns={'value': 'pdf', 'variable': 'model'}, inplace=True)
        return output

    alt_data = fit_severity(df[const.LATENT_SEV])
    alt_fitted = alt.Chart(alt_data).mark_line().encode(
        x='x',
        y='pdf',
        color=alt.Color('model',
                        scale=alt.Scale(domain=['data (kde)', 'weibull', 'gamma', 'pareto'],
                                        range=['#FF4B4B', '#9da8c3', 'steelblue', '#80c3c2'])),
        tooltip=[alt.Tooltip('model'),
                 alt.Tooltip('x', format=',.2f'),
                 alt.Tooltip('pdf', format=',.3f')]
    )

    st.altair_chart(alt.layer(alt_sev, alt_fitted), use_container_width=True)

    st.write("""
    ### 3. Empirical estimator
    One way to analyse our data is use nonparametric estimator such as Plug-In Principle
    without reference to a parametric model or assuming the underlying distribution
    """)

    st.write("""#### 3.1 Claims and Premium""")
    df_summary = df[[const.PREMIUM, const.CLAIMS, const.CLAIMS]].copy()
    df_summary.columns = [const.PREMIUM, 'zero claims', 'non-zero claims']
    df_summary.loc[df_summary['zero claims'] > 0, ['zero claims']] = np.nan
    df_summary.loc[df_summary['non-zero claims'] == 0, ['non-zero claims']] = np.nan

    st.table(df_summary.describe().T.style.format('{:,.0f}'))

    def trunc_num(x, digit: int):
        return int(str(int(x))[:digit]) * 10 ** (len(str(int(x))) - digit)

    col301, col302 = st.columns(2)
    with col301:
        alt_premium = alt.Chart(df[[const.PREMIUM]], title='Premium histogram').mark_bar().encode(
            x=alt.X(const.PREMIUM + ':Q', bin=alt.BinParams(maxbins=60)),
            y='count():Q',
        )
        st.altair_chart(alt_premium, use_container_width=True)
    with col302:
        alt_claims = alt.Chart(df_summary[['non-zero claims']], title='Non-zero claims histogram').mark_bar().encode(
            x=alt.X('non-zero claims:Q',
                    bin=alt.BinParams(maxbins=60)),
            y='count():Q',
        )
        st.altair_chart(alt_claims, use_container_width=True)

    st.write("#### 3.2. Loss Elimination Ratio")
    st.write("""for a deductible, _d_, and upper limit, _u_, the loss elimination ratio (LER) is the 
    percentage decrease in the expected payment of the insurer as a result of imposing the rules. LER can express it as
    """)
    st.latex(r"LER_n(d, u) = \frac{\sum_{i=1}^n \min(X_i,d) + \max(X_i - u,0)}{\sum_{i=1}^n X_i}")

    col401, col402 = st.columns(2)
    max_input = int(df[const.CLAIMS].max())
    with col401:
        d_default = trunc_num(np.nanquantile(df_summary['non-zero claims'], 0.2), 2)
        d = st.number_input('Deductible', 0, max_input, d_default)
    with col402:
        u_default = trunc_num(np.nanquantile(df_summary['non-zero claims'], 0.95), 2)
        u = st.number_input('Upper limit', 0, max_input, u_default)

    df_trunc = df_summary[['non-zero claims']]
    df_trunc.columns = ['claims']
    df_trunc['color'] = ['steelblue' if (x > d) & (x < u) else 'darkgrey' for x in df_trunc['claims']]

    alt_claims_censor = alt.Chart(df_trunc).mark_bar().encode(
        x=alt.X('claims:Q',
                bin=alt.BinParams(maxbins=100)),
        y='count():Q',
        color=alt.Color('max(color)', scale=None, legend=None)
    )
    st.altair_chart(alt_claims_censor, use_container_width=True)
    ler = (df[const.CLAIMS].values.clip(max=d) + (df[const.CLAIMS] - u).values.clip(min=0)).sum() / df[
        const.CLAIMS].sum()
    st.write(f"""
    - Loss Elimination Ratio = {ler:.1%}
    - Upper limit affects {df[df[const.CLAIMS] > u].shape[0]} claims
    - {df_summary[df_summary['non-zero claims'] < d].shape[0]} claims below deductible amount
    """)

    @st.experimental_memo
    def ler_hold_u(data, u, n_line=100):
        x = np.linspace(0, data.max(), n_line)

        # clip x so that it stops at u, since when d = u, insurer pays nothing
        claims_d = data.values.clip(max=np.array([x.clip(max=u)]).T) + (data.values - u).clip(min=0)

        df = pd.DataFrame({
            'x': x,
            'LER': claims_d.sum(axis=1)
        })
        df['LER'] = df['LER'] / data.sum()

        return df

    @st.experimental_memo
    def ler_hold_d(data, d, n_line=100):
        x = np.linspace(0, data.max(), n_line)

        # clip x at min=d, when u = d, insurer pays nothing
        claims_u = data.values.clip(max=d) + (data.values - np.array([x.clip(min=d)]).T).clip(min=0)

        df = pd.DataFrame({
            'x': x,
            'LER': claims_u.sum(axis=1)
        })
        df['LER'] = df['LER'] / data.sum()

        return df

    with st.expander("See how changing d and u affects the Loss Elimination Ratio"):
        df_d = ler_hold_u(df[const.CLAIMS], u, 100)
        df_u = ler_hold_d(df[const.CLAIMS], d, 100)
        df_ler_point = pd.DataFrame({'d': d, 'u': u, 'LER': ler}, index=[0])

        d_alt = alt.Chart(df_d, title='LER with varying d, holding u').mark_line().encode(
            x=alt.X('x:Q', title='deductible'),
            y=alt.Y('LER:Q', axis=alt.Axis(format='%'))
        )
        d_line = alt.Chart(df_ler_point).mark_rule(color='#FF4B4B', strokeWidth=1.5).encode(x="d:Q")
        d_text = alt.Chart(df_ler_point).mark_text(dx=20, dy=5, color='darkgrey').encode(
            x='d', y='LER', text=alt.Text('LER', format='.1%'),
        )

        alt_u = alt.Chart(df_u, title='LER with varying u, holding d').mark_line().encode(
            x=alt.X('x:Q', title='upper limit'),
            y=alt.Y('LER:Q', axis=alt.Axis(format='%'))
        )
        u_line = alt.Chart(df_ler_point).mark_rule(color='#FF4B4B', strokeWidth=1.5).encode(x="u:Q")
        u_text = alt.Chart(df_ler_point).mark_text(dx=20, dy=-5, color='darkgrey').encode(
            x='u', y='LER', text=alt.Text('LER', format='.1%'),
        )

        st.altair_chart(d_alt + d_line + d_text, use_container_width=True)
        st.altair_chart(alt_u + u_line + u_text, use_container_width=True)

    st.write("#### 3.3. Tail Value-at-Risk (TVaR)")
    st.write(r"""One method to estimate extreme losses is using quantile based approach 
    $\small \text{Value-at-Risk} (VaR)$, which measures the $\small (1-q) \times 100\% $ quantile of losses. 
    Given random variable ${\small X}$ its cumulative distribution function ${\small F_{X}}$. 
    It is Mathematically expressed as""")
    st.latex(r"VaR_q[X]=\inf\{x:F_X(x)\geq q\} = F_{Y}^{-1}(1-q)")

    st.write(r"""However, a downside of $\small VaR$ is it doesn't take into account how long or heavy the tail is. 
    To overcome this, $\text{Tail Value-at-Risk} (TVaR)$ is used to measure the expected loss above the $VaR$ value. 
    This can be expressed as""")
    st.latex(r" TVaR_q[X] = \mathrm{E}[X|X>VaR_q[X]]")

    st.write(r"""Supposing we don't know the underlying distribution function of $X$, we can estimate $TVaR$ using
    the Plug-In Principle as""")
    st.latex(r"TVaR_{n,q}[X] = \frac{\sum_{i=1}^n X_i I(X_i > F^{-1}_n(q))}{\sum_{i=1}^n I(X_i > F^{-1}_n(q))}")
    st.write(r"""Here, the notation $I(⋅)$ is the indicator function, it returns 1 if the event ($⋅)$ 
    is true and 0 otherwise, and $F^{-1}_n(q)$ is the nonparametric estimator of $F_{Y}^{-1}(q)$, 
    which roughly translates to the $q^{th}$ quantile of $X$""")

    @st.experimental_memo
    def tvar_matrix(data, n_line=100):
        x = np.linspace(0, 1, n_line)
        quantiles = np.quantile(data.values, np.array([x]).T)
        tail_claims = np.where(data.values >= quantiles, data.values, False)

        df = pd.DataFrame({
            'x': x,
            'counts': np.count_nonzero(tail_claims, axis=1),
            'TVaR': tail_claims.sum(axis=1)
        })
        df['TVaR'] = df['TVaR'] / df['counts']

        return df

    @st.experimental_memo
    def calc_tvar(data, q):
        quantile = np.quantile(data, q)
        censored_data = [x for x in data if x >= quantile]
        return len(censored_data), np.mean(censored_data)

    q = st.slider('Quantile', 0.0, 1.0, 0.95)

    nonzero_claims = df.loc[df[const.CLAIMS] > 0, const.CLAIMS]
    n_TVaR, TVaR = calc_tvar(nonzero_claims, q)  # TVaR value at given q
    df_tvar = tvar_matrix(nonzero_claims, 21)
    df_tvar_point = pd.DataFrame({'q': q, 'TVaR': TVaR}, index=[0])

    st.write("Here, we will observe only non-zero claims, but this can equally be applied to the full portfolio.")
    st.write(f"""
        - Tail Value-at-Risk = {TVaR:,.0f}
        - Number of claims above {q:.0%} quantile = {n_TVaR}
        """)

    tvar_alt = alt.Chart(df_tvar, title='TVaR with varying quantile').mark_line().encode(
        x=alt.X('x:Q', title='quantile'),
        y=alt.Y('TVaR:Q')
    )
    tvar_line = alt.Chart(df_tvar_point).mark_rule(color='#FF4B4B', strokeWidth=1.5).encode(x="q:Q")
    tvar_text = alt.Chart(df_tvar_point).mark_text(dx=-25, dy=-5, color='darkgrey').encode(
        x='q:Q', y='TVaR:Q', text=alt.Text('TVaR:Q', format=',.0f'),
    )
    st.altair_chart(tvar_alt + tvar_line + tvar_text, use_container_width=True)
