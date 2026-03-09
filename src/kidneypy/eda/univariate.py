import warnings

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


def univariate_glm_plot(
        df: pd.DataFrame, 
        feature_col: str, 
        target_col: str, 
        family: str | None = None, 
        alpha: float = 0.05,
        discretize: bool = False,
        nbins: int = 5,
    ) -> tuple[Figure, Axes]:
    # raise NotImplementedError

    if not family:
        # family = infer_family(y)
        raise NotImplementedError('infer_family not implemented yet - please supply a value for the family argument, e.g. family="normal", family="binomial"')    
    if family == 'normal':
        sm_family = sm.families.Gaussian()
    elif family == 'binomial':
        sm_family = sm.families.Binomial()
    else:
        raise ValueError('expecting family to be one of "normal" or "binomial"')
    
    _ = df[[target_col, feature_col]].copy()

    x_col = feature_col
    discretized = False
    if discretize and _[feature_col].dtype not in ['object', 'category']:
        discretized = True
        if _[feature_col].nunique() > nbins:
            x_col = f'{feature_col}_discretized_{nbins}'
            _.loc[:, x_col] = pd.qcut(_[feature_col], q=nbins)
        else:
            _.loc[:, x_col] = _[x_col].astype(str)
    categorical = _[x_col].dtype in ['object', 'category']
    # print('data following discretization')
    # print(_.head())

    # model = smf.glm(formula=f"{target_col} ~ {feature_col}", data=df, family=sm_family).fit()
    target_formula_name = f'Q("{target_col}")'
    feature_formula_name = f'Q("{x_col}")'
    if categorical:
        feature_formula_name = f'C({feature_formula_name})'
    # print(feature_formula_name)
    null_model = smf.glm(formula=f'{target_formula_name} ~ 1', data=_, family=sm_family).fit()
    full_model = smf.glm(formula=f'{target_formula_name} ~ {feature_formula_name}', data=_, family=sm_family).fit()
    #print(full_model.summary())

    lr_stat = 2 * (full_model.llf - null_model.llf)  # log-likelihood difference
    df_diff = full_model.df_model - null_model.df_model  # difference in number of parameters
    p_value = stats.chi2.sf(lr_stat, df_diff)
    # print(f"LR stat: {lr_stat:.3f}, df: {df_diff}, p-value: {p_value:.4f}")
    # print(p_value)

    _ = _[[x_col]].drop_duplicates().sort_values(x_col).reset_index(drop=True)
    # print()
    # print('data to get preds for')
    # print(_.head())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        preds = full_model.get_prediction(_).summary_frame(alpha=alpha)
    # print()
    # print('preds')
    # print(preds.head())

    # print()
    # print('mean_se')
    # print(preds['mean_se'])
    # print(preds['mean_se'][0])
    # print(np.isclose(preds['mean_se'], 0, atol=1e-05))
    zero_se_mask = np.isclose(preds['mean_se'], 0, atol=1e-05)
    # preds.loc[zero_se_mask, ['mean_ci_lower', 'mean_ci_upper']] = np.nan
    preds.loc[zero_se_mask, ['mean_ci_lower', 'mean_ci_upper']] = preds.loc[zero_se_mask, 'mean']
    # print()
    # print('preds')
    # print(preds.head())

    preds = pd.concat([_, preds], axis=1)#.dropna()
    # print()
    # print('preds joined with data')
    # print(preds.head())
    # print(preds.dtypes)

    fig, ax = plt.subplots()

    # sort values to ensure smooth lines
    # preds = preds.sort_values(x_col)

    if categorical:

        means = preds["mean"]
        lower = preds["mean_ci_lower"]
        upper = preds["mean_ci_upper"]
        yerr = [means - lower, upper - means]

        ax.bar(
            preds[x_col].astype(str), 
            preds["mean"], 
            yerr=yerr, 
            capsize=5,
        )
        plt.xticks(rotation=90)
    
    else:

        # predicted probability curve
        ax.plot(
            preds[x_col],
            preds["mean"],
            label="Mean estimate"
        )

        # confidence interval shading
        ax.fill_between(
            preds[x_col],
            preds["mean_ci_lower"],
            preds["mean_ci_upper"],
            alpha=0.25,
            label=f"Mean {1-alpha:.0%} CI"
        )

        # optional: show observed binary data
        ax.scatter(
            df[feature_col],
            df[target_col],
            alpha=0.4,
            s=25
        )

        # ax.set_ylim(0,1)
        # ax.legend()
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
        # ax.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=2)

    if family in ['binomial', 'normal', 'poisson', 'gamma']:
        ax.axhline(df[target_col].mean(), linestyle=":", color="black")
    
    title = feature_col
    if discretized: 
        title += ' (discretized)'
    fig.suptitle(title, fontsize=16)
    subtitle = f'{family} GLM (p={p_value:.5f})'
    ax.set_title(subtitle)
    ax.set_xlabel(feature_col)
    ax.set_ylabel(target_col)

    return fig, ax


def univariate_glm(x: pd.Series, y: pd.Series, alpha=0.05, family: str | None = None) -> pd.DataFrame:
    """
    Perform a univariate GLM.

    Args:
        x (pd.Series): feature variable (any object tyoe that can be coercible to a pandas series)
        y (pd.Series): target variable (any object tyoe that can be coercible to a pandas series)
        family (str | None, optional): string indicating the GLM family to use. Defaults to None.

    Raises:
        NotImplementedError: if family is not supplied
        ValueError: if family is invalid

    Returns:
        pd.DataFrame: mean predictions and confidence intervals
    """    
    if not family:
        # family = infer_family(y)
        raise NotImplementedError('infer_family not implemented yet - please supply a value for the family argument, e.g. family="normal", family="binary"')    
    if family == 'normal':
        sm_family = sm.families.Normal()
    elif family == 'binary':
        sm_family = sm.families.Binomial()
    else:
        raise ValueError('expecting family to be one of "normal" or "binary"')
    
    df = pd.DataFrame({'x': x, 'y': y}).copy()
    model = smf.glm(formula="y ~ x", data=df, family=sm_family).fit()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        preds = model.get_prediction(df).summary_frame(alpha=alpha)
    preds = pd.concat([df, preds], axis=1)
    return preds


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




