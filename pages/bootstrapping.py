import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import const
import altair as alt
from helper import saved_df

rng = np.random.default_rng(42)
SEED = 42

TTL_PREMIUM = const.PREMIUM + '_total'
TTL_CLAIMS = const.CLAIMS + '_total'
TTL_LR = 'LR_overall'

d_list = [1e6, 1.5e6, 2e6, 2.5e6]
p_list = [0.7, 0.8, 0.9, 0.95, 0.98]
q_list = [0.8, 0.9, 0.95, 0.98]

ler_col_name = [f'LER {d / 1e6:.1f}m' for d in d_list]
tvar_col_name = [f'TVaR {p:0.2f}' for p in q_list]
boot_formatting = {TTL_PREMIUM: '{:,.0f}', TTL_CLAIMS: '{:,.0f}', TTL_LR: '{:.0%}',
                   **{c: '{:.0%}' for c in ler_col_name},
                   **{c: '{:,.0f}' for c in tvar_col_name}
                   }


@st.cache(allow_output_mutation=True)
def bootstrap(data_in, nboot: int):
    data = data_in.copy()

    sample_idx = rng.integers(0, len(data), size=(len(data), nboot))

    # Total premium, claims, LR
    premium_boot = data[const.PREMIUM].values[sample_idx]
    claims_boot = data[const.CLAIMS].values[sample_idx]

    dfboot = pd.DataFrame({
        TTL_PREMIUM: premium_boot.sum(axis=0),
        TTL_CLAIMS: claims_boot.sum(axis=0)}
    )
    dfboot[TTL_LR] = dfboot[TTL_CLAIMS] / dfboot[TTL_PREMIUM]

    # LER columns
    for d in d_list:
        col_name = f'LER {d/1e6:.1f}m'
        claims_d_boot = (claims_boot - d).clip(0, None)
        dfboot[col_name] = 1 - claims_d_boot.sum(axis=0) / dfboot[TTL_CLAIMS]

    # Tail value at risk columns
    for q in q_list:
        col_name = f'TVaR {q:.2f}'
        threshold = np.quantile(claims_boot, q, axis=0)
        sum_tail_risk = (claims_boot * (claims_boot > threshold)).sum(axis=0)
        count_tail_risk = np.count_nonzero(claims_boot > threshold, axis=0)
        dfboot[col_name] = sum_tail_risk / count_tail_risk

    return dfboot

from typing_extensions import Literal

@st.cache
def conf_int(data: pd.DataFrame,
             size: float = 0.95,
             tail: Literal[1, 2] = 2,
             axis: Literal[0, 1] = 0,
             formatting: callable = None):
    """
    :param data: dataframe input
    :param size: confidence level
    :param tail: 1 or 2 for one or two tail
    :param axis: axis where the statistics are calculated
    :param formatting: string formatting for output numeric range
    :return:
    Output 3 types of Confidence Interval: normal, basic, percentile
    - 'normal' CI formula is (θ−z*σ , θ+z*σ)
    - 'basic' CI formula is (2θ−qU, 2θ−qL)
    - 'percentile' CI formula is (qL, qU)
    """

    m = data.mean(axis=axis)
    std = data.std(ddof=1, axis=axis)

    if tail == 2:
        lower_alpha = 0.5 - size / 2
        upper_alpha = 0.5 + size / 2
    else:
        lower_alpha = 1.0 - size
        upper_alpha = size

    norm_ppf = norm.ppf(upper_alpha)

    upper_normal = m + std * norm_ppf
    upper_basic = (2 * m - np.quantile(data, lower_alpha, axis=axis))
    upper_perc = np.quantile(data, upper_alpha, axis=axis)

    if tail == 2:
        lower_normal = m - std * norm_ppf
        lower_basic = (2 * m - np.quantile(data, upper_alpha, axis=axis))
        lower_perc = np.quantile(data, lower_alpha, axis=axis)

    else:
        lower_normal, lower_basic, lower_perc = [np.zeros(m.shape)] * 3

    out = pd.DataFrame({
        'p': size,
        'Mean': m,
        'Std. Error': std,
        'Lower Normal': lower_normal,
        'Upper Normal': upper_normal,
        'Lower Basic': lower_basic,
        'Upper Basic': upper_basic,
        'Lower Percentile': lower_perc,
        'Upper Percentile': upper_perc,
    })

    def f(x):
        return formatting.format(x)

    out['Normal'] = out['Lower Normal'].map(f) + ' - ' + out['Upper Normal'].map(f)
    out['Basic'] = out['Lower Basic'].map(f) + ' - ' + out['Upper Basic'].map(f)
    out['Percentile'] = out['Lower Percentile'].map(f) + ' - ' + out['Upper Percentile'].map(f)

    return out


def show():
    st.title('Bootstrapping')

    if not isinstance(saved_df().df, pd.DataFrame):
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
    
    In practice, we only have a small set of data. It becomes difficult to estimate the underlying 
    portfolio statistics such as standard deviation, confidence intervals, or value at risk. We can simulate draws 
    from unknown distribution from our sample observation. If the sample is a good representation of the population, 
    we can approximate the population statistics. This process is called bootstrapping
    """)

    # =========== Bootstrapping start ===========

    st.write('#### Bootstrap configuration')
    st.write(f"")

    with st.form('Bootstrapping option form') as f:
        n_boot = st.number_input('Number of portfolio replicates', 100, 10000, 1000)

        st.form_submit_button('Bootstrap data')

    df_boot = bootstrap(df, int(n_boot))

    st.write('#### Analysis of bootstrapped portfolio')

    # draw bar plot
    col1, col2 = st.columns(2)

    with col1:
        alt_hist = alt.Chart(
            df_boot,
            title='Histogram of loss ratio'
        ).mark_bar().encode(
            x=alt.X('LR_overall:Q', bin=alt.BinParams(maxbins=50)),
            y='count()',
        )
        st.altair_chart(alt_hist, use_container_width=True)

    # draw scatter plot
    with col2:
        alt_scatter = alt.Chart(
            df_boot,
            title='Distribution of total premium and claims'
        ).mark_circle().encode(
            x=alt.X(
                TTL_PREMIUM,
                scale=alt.Scale(zero=False),
                axis=alt.Axis(labelExpr='datum.value / 1E6 + "M"'),
            ),
            y=alt.Y(
                TTL_CLAIMS,
                scale=alt.Scale(zero=False),
                axis=alt.Axis(labelExpr='datum.value / 1E6 + "M"'),
            ),
            color=TTL_LR,
            tooltip=[alt.Tooltip(TTL_PREMIUM, format=',.0f'),
                     alt.Tooltip(TTL_CLAIMS, format=',.0f'),
                     alt.Tooltip(TTL_LR, format='.0%')]
        )
        st.altair_chart(alt_scatter, use_container_width=True)

    # Summary statistics
    st.write(df_boot.describe().loc[['mean', 'std', 'min', 'max']]
             .style.format(boot_formatting))

    st.write("#### Estimating Loss Ratio statistics")

    st.write(r'We summarize the nonparametric bootstrap summary measures below.'
             r'Here, $\small\bar{\hat{\theta^*}}$ is the average of '
             r'$\small\{\hat{\theta}_1^*, \ldots,\hat{\theta}_B^*\}$ '
             r'and $\small\hat{\theta}_b^*$ is a statistics of b<sup>th</sup> bootstrap replicate.',
             unsafe_allow_html=True)
    st.latex(r"""{\small \begin{matrix} \begin{array}{l|c|c|c} 
    \hline \text{Population Measure}& \text{Population Definition}&\text{Bootstrap Approximation}\\ 
    \hline \text{Bias} & \mathrm{E}(\hat{\theta})-\theta&\overline{\hat{\theta^*}}-\hat{\theta}\\
    \hline \text{Standard Deviation} & \sqrt{\mathrm{Var}(\hat{\theta})} & 
        \sqrt{\frac{1}{B-1} \sum_{b=1}^{B}\left(\hat{\theta}_b^* -\overline{\hat{\theta^*}} \right) ^2}\\
    \hline \text{Mean Square Error} &\mathrm{E}(\hat{\theta}-\theta)^2 & 
        \frac{1}{B} \sum_{b=1}^{B}\left(\hat{\theta}_b^* -\hat{\theta} \right)^2\\ 
    \hline \end{array}\end{matrix} }""")

    st.write(r"""
    **Confidence Interval**
    Another major benefit of using bootstrap method is avoiding normality assumption when calculating Confidence Interval. This allows alternative confidence intervals:
    1. Normal CI: uses standard deviation, suitable when statistic is normally distributed
    2. Basic CI: uses mean and percentile to calculate upper and lower limit, suitable when bootstrap statistic is normally distributed (**recommended**)
    3. Percentile CI: approximate intervals using quantile. Suitable when standard error of bootstrap statistic and sample statistic are the same
    4. BCa or Bias Corrected Accelerated CI: uses percentile limits with bias correction and estimate acceleration coefficient (**to be implemented**)
    """)

    # =========== Loss Ratio Bootstrap estimate ===========

    df_lr = pd.DataFrame()
    for p in p_list:
        df_lr = df_lr.append(conf_int(df_boot[[TTL_LR]], size=p, formatting='{:.1%}'))

    st.write("""
        #### A.  Bootstrap Estimates of Loss Ratio
        Loss Ratio is calculated as sum of claims / sum of premium
        - Mean = {:.1f}%
        - Bias = {:.1f}%
        - Standard Error: = {:.1f}%
        - Confidence Interval at various confidence level:""".format(
        df_lr['Mean'].values[0] * 100,
        df_lr['Mean'].values[0] * 100 - (df[const.CLAIMS].sum() / df[const.PREMIUM].sum() * 100),
        df_lr['Std. Error'].values[0] * 100
    ))

    df_lr.index = ['{:.0%}'.format(x) for x in df_lr['p']]
    df_lr = df_lr.drop(['Mean', 'Std. Error'], axis=1)

    df_lr = df_lr[['Normal', 'Basic', 'Percentile']]
    st.table(df_lr)


    # =========== Loss Elimination Ratio Bootstrap estimate ===========

    df_ler = pd.DataFrame()
    df_ler = df_ler.append(conf_int(df_boot[ler_col_name], size=0.95, formatting='{:.1%}'))

    st.write("#### B.  Bootstrap Estimates of Loss Elimination Ratio (LER)")

    ler = df_boot[ler_col_name].melt()
    alt_hist = alt.Chart(ler, title=f'Histogram of LER distribution').mark_line().encode(
        x=alt.X('value:Q', bin=alt.BinParams(maxbins=100)),
        y='count()',
        color=alt.Color('variable', scale=alt.Scale(scheme='category20'))
    )
    st.altair_chart(alt_hist, use_container_width=True)

    st.write("""
        Many of reinsurance contain excess of loss coverage. Insurer imposes deductible, _d_, where if the claims 
        amount up to d is paid by policyholder before the insurer makes any payment
        LER is the percentage decrease in the expected payment of the insurer as a result of imposing the deductible 
        calculated as 1 - (sum of claims exceeding deductible / sum of claims)
        Below is the summary statistics of LER with various deductible values at 95% confidence level.
        """)
    df_ler = df_ler[['Mean', 'Std. Error', 'Normal', 'Basic', 'Percentile']]

    st.table(df_ler.style.format({'Mean': '{:.1%}', 'Std. Error': '{:.1%}'}))


    # =========== Loss Quantile Bootstrap estimate ===========

    df_tvar = pd.DataFrame()
    df_tvar = df_tvar.append(conf_int(df_boot[tvar_col_name], size=0.95, formatting='{:,.0f}'))

    st.write("#### C.  Bootstrap Estimates of Tail Value at Risk (TVaR)")

    tvar = df_boot[tvar_col_name].melt()
    alt_hist = alt.Chart(tvar, title=f'Histogram of TVaR distribution').mark_line().encode(
        x=alt.X('value:Q', bin=alt.BinParams(maxbins=100)),
        y='count()',
        color=alt.Color('variable', scale=alt.Scale(scheme='category20'))
    )
    st.altair_chart(alt_hist, use_container_width=True)

    st.write("""
        In insurance, very large claim at the tail-end of the risk can have a devastating impact to portfolio.
        When claims distribution is not known, it is hard to model the tail probability distribution.
        TVaR is calculated as (sum of claims > Q quantile) / (count of claims > Q quantile)
        Bootstrap allows expected value-of-risk above varying claims quantiles to be estimated as below
        """)
    df_tvar = df_tvar[['Mean', 'Std. Error', 'Normal', 'Basic', 'Percentile']]

    st.table(df_tvar.style.format({'Mean': '{:,.0f}', 'Std. Error': '{:,.0f}'}))


