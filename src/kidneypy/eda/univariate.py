import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype, 
    is_float_dtype, 
    is_integer_dtype
)
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def is_binary(x: pd.Series) -> bool:
    x = pd.Series(x)
    return is_integer_dtype(x) and x.dropna().isin([0, 1]).all()


def is_categorical(x: pd.Series) -> bool:
    x = pd.Series(x)
    return x.dtype in ['object', 'category']


def discretize_x(x: pd.Series, nbins: int = 10, quantiles=False, **kwargs) -> pd.Series:
    if x.dtype in ['object', 'category']:
        return x
    if x.nunique() <= nbins:
        return x.astype('object')
    if quantiles:
        return pd.qcut(x, q=nbins, **kwargs)
    return pd.cut(x, bins=nbins, **kwargs)


def plot_feature(
        df: pd.DataFrame, 
        feature_col: str, 
        target_col: str = None, 
        family: str | None = None, 
        alpha: float = 0.05,
        discretize: bool = False,
        nbins: int = 10,
    ) -> tuple[Figure, Axes]:

    if target_col:
        # side-by-side plot if target_col supplied
        fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
        ax1 = ax[0]
        ax2 = ax[1]
    else:
        # otherwise just plot feature distribution
        fig, ax1 = plt.subplots(figsize=(6.4, 4.8))

    fig.suptitle(feature_col, fontsize=16)

    # plot feature distribution -------------------------------------------------------------------

    df_copy = df[[feature_col]].copy()

    if discretize:
        df_copy[feature_col] = discretize_x(df_copy[feature_col], nbins=nbins, quantiles=False)

    if is_categorical(df_copy[feature_col]):

        counts = df_copy[feature_col].value_counts().sort_index()
        dist = counts / counts.sum()
        ax1.bar(
            dist.index.astype(str),
            dist,
        )
        ax1.tick_params(axis='x', labelrotation=90)
    
    else:

        ax1.hist(df_copy[feature_col], bins=nbins, density=True)

    title = 'distribution'
    if discretize: 
        title += f' (nbins={nbins})'
    ax1.set_title(title)

    if not target_col:
        return fig, ax1

    # plot relationship with target ---------------------------------------------------------------

    df_copy = df[[target_col, feature_col]].copy()

    if discretize:
        df_copy[feature_col] = discretize_x(df_copy[feature_col], nbins=nbins, quantiles=True)

    if not family:
        # family = infer_family(y)
        raise NotImplementedError(('infer_family not implemented yet - '
        'please supply a value for the family argument, e.g. family="normal", family="binomial"'))
    if family == 'normal':
        sm_family = sm.families.Gaussian()
    elif family == 'binomial':
        if not is_binary(df[target_col]):
            raise ValueError('binary target expected for family "binomial"')
        sm_family = sm.families.Binomial()
    else:
        raise ValueError(f'invald family {family} - expecting one of "normal" or "binomial"') 

    # use Q() convention in case feature names contain spaces
    # use C() convention for categorical
    target_col_formula = f'Q("{target_col}")'
    feature_col_formula = f'Q("{feature_col}")'

    if is_categorical(df_copy[feature_col]):
        feature_col_formula = f'C({feature_col_formula})'

    null_model = smf.glm(formula=f'{target_col_formula} ~ 1', data=df_copy, family=sm_family).fit()
    full_model = smf.glm(formula=f'{target_col_formula} ~ {feature_col_formula}', data=df_copy, family=sm_family).fit()

    # LR test to get a single p-value for the feature
    ll_ratio = 2 * (full_model.llf - null_model.llf)  
    df_diff = full_model.df_model - null_model.df_model  
    p_value = stats.chi2.sf(2 * ll_ratio, df_diff)

    # distinct values of processed feature col for generating predictions
    df_distinct = df_copy[[feature_col]].drop_duplicates().sort_values(feature_col).reset_index(drop=True)

    # calculate CIs - temporaily disable RuntimeWarnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        preds = full_model.get_prediction(df_distinct).summary_frame(alpha=alpha)

    # fix a peculiar issue with the statsmodels CIs where mean_se=0 and mean_ci is [0, 1]
    zero_se_mask = np.isclose(preds['mean_se'], 0, atol=1e-05)
    preds.loc[zero_se_mask, ['mean_ci_lower', 'mean_ci_upper']] = preds.loc[zero_se_mask, 'mean']

    # append distinct feature values to predictions for plotting
    preds = pd.concat([df_distinct, preds], axis=1)

    if is_categorical(df_copy[feature_col]):

        means = preds["mean"]
        lower = preds["mean_ci_lower"]
        upper = preds["mean_ci_upper"]
        yerr = [means - lower, upper - means]

        ax2.bar(
            preds[feature_col].astype(str), 
            preds["mean"], 
            yerr=yerr, 
            capsize=5,
        )
        ax2.tick_params(axis='x', labelrotation=90)

    else:

        # predicted mean curve
        ax2.plot(
            preds[feature_col],
            preds["mean"],
            label="Mean estimate"
        )

        # confidence interval shading
        ax2.fill_between(
            preds[feature_col],
            preds["mean_ci_lower"],
            preds["mean_ci_upper"],
            alpha=0.25,
            label=f"Mean {1-alpha:.0%} CI"
        )

        # optional: show observed binary data
        ax2.scatter(
            df[feature_col],
            df[target_col],
            alpha=0.4,
            s=25
        )

        ax2.legend(bbox_to_anchor=(1, 1), loc="upper left")
        # ax.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=2)

    if family in ['binomial', 'normal', 'poisson', 'gamma']:
        ax2.axhline(df[target_col].mean(), linestyle=":", color="black")
    
    ax2.set_title(f'{family} GLM (p={p_value:.5f})')
    ax2.set_ylabel(target_col)
    fig.suptitle(feature_col, fontsize=16)

    return fig, ax


def infer_family(y: pd.Series) -> str:
    raise NotImplementedError('planned for future release')

    y = pd.Series(y)

    if not is_numeric_dtype(y):
        return 'category'
    
    y_dtype = y.dtype
    y_nunique = y.dropna().nunique()
    y_min = y.min()
    y_max = y.max()

    if y_nunique == 2 and y_min == 0 and y_max == 1:
        return 'binary'
    
    if y_nunique < 10:
        return 'category'
    
    if y_nunique > 2 and y_min == 0 and y_max == 1:
        return 'beta'
    
    if is_integer_dtype(y) and y_min >= 0:
        return 'count'
    
    if is_float_dtype(y) and y_min < 0:
        return 'normal'

    if is_float_dtype(y) and y_min > 0:
        return 'gamma'
    
    raise NotImplementedError(f'''no family inferred for
dtype {y_dtype}
nunique {y_nunique}
min {y_min}
max {y_max}
''')




