import streamlit as st
import pandas as pd
import numpy as np
import const
import altair as alt
from helper import saved_df


def show():
    st.write('# Analysis')

    if not isinstance(saved_df().df, pd.DataFrame):
        st.write('Save data in "Generate Data" step first')
        st.stop()

    df = saved_df().df.copy()
    n = df.shape[0]

    st.write("""
    In this section we will analyse summary statistics and  distribution of our generated data.
    """)

    st.write("### Estimating frequency distribution")
    st.write("""
    Given a set of observed data, we often wish to know the most suitable parameter of a distribution function 
    that best represent the data, i.e. maximises the likelihood ob the observed dataset. 
    The Maximum Likelihood Estimators (MLE) is precisely that, the value that maximises the likelihood of the observation
    """)

    alt_freq = alt.Chart(df[[const.HAS_CLAIM]]).mark_bar(size=50).encode(
        x=const.HAS_CLAIM + ':O',
        y='count()'
    )
    text = alt_freq.mark_text(align='center', dy=-5, color='steelblue').encode(
        text='count()'
    )

    col101, col102 = st.columns([2,1])
    with col101:
        st.altair_chart((alt_freq + text), use_container_width=True)
    with col102:
        df_counts = (df[[const.HAS_CLAIM]].groupby(const.HAS_CLAIM)
                     .agg(counts=pd.NamedAgg(column=const.HAS_CLAIM, aggfunc="count"))
                     .reset_index())
        df_counts.index = ['']*df_counts.shape[0]
        st.write(df_counts)

    # goodness of fit
    st.write("### Goodness of Fit")

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

    @st.cache
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
        {param['a']/(param['a'] + param['b']) :,.3f}. Although, because multiple paid of α β satisfy this condition, 
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
        K = K+1
        df_fit = df_fit.append({const.HAS_CLAIM: f'>{capped_at}',
                                'Fitted Counts': n-df_fit['Fitted Counts'].sum()},
                               ignore_index=True)

    df_counts[const.HAS_CLAIM] = df_counts[const.HAS_CLAIM].astype(str)
    df_diff = df_counts.merge(df_fit, on=const.HAS_CLAIM, how='outer').fillna(0)
    df_diff.index = [''] * df_diff.shape[0]
    df_diff.columns = ['No. claims', 'Obs. counts', 'Fitted counts']
    st.write(df_diff.style.format('{:,.1f}', subset=['Obs. counts', 'Fitted counts']))

    # chi square test
    from scipy.stats import chi2
    chi_square = ((df_diff['Obs. counts'] - df_diff['Fitted counts'])**2 / (n * df_diff['Fitted counts']))
    # chi_square.replace([np.inf, -np.inf], 0, inplace=True)
    st.latex(r"\small \frac{(\text{Obs. counts} - \text{Fitted counts})^2 }{n \times \text{Fitted counts}} = "
             f"{chi_square.sum():,.3f}")

    crit_value = chi2.ppf(cl, df=K-1-1)  # number of parameters being estimated is 1 for both cases
    result = 'is a good fit' if crit_value >= chi_square.sum() else 'is not a good fit'
    st.write(f"Compare this to the chi-square distribution critical value at {cl:.1%} confidence "
             f"with {K-1-1} degree of freedom: {crit_value:.2f}. So, we conclude the fitted distribution {result}.")

