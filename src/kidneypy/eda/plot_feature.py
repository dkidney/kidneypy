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


def plot_feature(
        df: pd.DataFrame, 
        feature_col: str, 
        target_col: str = None, 
        family: str | None = None, 
        alpha: float = 0.05,
        as_category: bool = False,
        explicit_na: bool = False,
        log: bool = False,
        discretize: bool = False,
        nbins: int = 10,
        rot: int = 0,
        figsize = None,
    ) -> tuple[Figure, Axes]:    

    # as_category overrides discretize
    discretize = False if as_category else discretize

    if target_col:
        # side-by-side plot if target_col supplied
        figsize = figsize if figsize else (12.8, 4.8)
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax1 = ax[0]
        ax2 = ax[1]
        df_copy = df[[target_col, feature_col]].copy()
    else:
        figsize = figsize if figsize else (6.4, 4.8)
        fig, ax1 = plt.subplots(figsize=figsize)
        df_copy = df[[feature_col]].copy()

    feature_col_obs = feature_col

    # plot feature distribution -------------------------------------------------------------------

    if as_category:
        # df_copy[feature_col] = df_copy[feature_col].astype(str)
        # df_copy[feature_col] = pd.Categorical(df_copy[feature_col])
        df_copy[feature_col] = as_categorical(df_copy[feature_col], explicit_na=explicit_na)

    if log and is_numeric_dtype(df_copy[feature_col]):
        old_col = df_copy.columns[-1]
        new_col = f'log({old_col})'
        df_copy[new_col] = np.log(df_copy[old_col])
        feature_col_obs = new_col

    if discretize:
        old_col = df_copy.columns[-1]
        new_col = f'{old_col} cut({nbins})'
        df_copy[new_col] = discretize_x(df_copy[old_col], nbins=nbins, quantiles=False)

    if explicit_na and is_categorical(df_copy[df_copy.columns[-1]]):
        df_copy[df_copy.columns[-1]].fillna('MISSING')

    # print(df_copy.head())

    if is_discrete(df_copy[df_copy.columns[-1]]):

        counts = df_copy[df_copy.columns[-1]].value_counts(dropna=False).sort_index()
        dist = counts / counts.sum()
        ax1.bar(
            dist.index.astype(str),
            dist,
        )
        ax1.tick_params(axis='x', labelrotation=90 if discretize else 0)
        ax1.set_ylabel('probability')
    
    else:

        ax1.hist(df_copy[df_copy.columns[-1]], bins=nbins, density=True)
        ax1.set_ylabel('density')

    # main_title = feature_col
    # if logged:
    #     main_title += ' (log)'
    fig.suptitle(feature_col, fontsize=16)

    # title = 'distribution'
    # if discretize: 
    #     title += f' (nbins={nbins})'
    ax1.set_title('distribution')
    ax1.set_xlabel(df_copy.columns[-1])
    ax1.tick_params(axis='x', labelrotation=rot)

    if not target_col:
        return fig, ax1
    
    # plot relationship with target ---------------------------------------------------------------

    df_copy = df_copy.dropna(subset=[target_col])

    # df_copy = df[[target_col, feature_col]].copy().dropna(subset=[target_col])
    # feature_col_obs = feature_col

    # if as_category:
    #     df_copy[feature_col] = df_copy[feature_col].astype(str)

    # if is_numeric_dtype(df_copy[feature_col]) and log:
    #     old_col = df_copy.columns[-1]
    #     new_col = f'log({old_col})'
    #     df_copy[new_col] = np.log(df_copy[old_col])

    if discretize:
        old_col = feature_col_obs
        new_col = f'{old_col} qcut ({nbins})'
        df_copy[new_col] = discretize_x(df_copy[old_col], nbins=nbins, quantiles=True)

    if explicit_na and is_categorical(df_copy[df_copy.columns[-1]]):
        df_copy[df_copy.columns[-1]].fillna('MISSING')

    # print(df_copy.head())

    if not family:
        # family = infer_family(y)
        raise NotImplementedError(('infer_family not implemented yet - '
        'please supply a value for the family argument, e.g. family="normal", family="binomial"'))
    if family == 'normal':
        sm_family = sm.families.Gaussian()
    elif family == 'binomial':
        df_copy[target_col] = as_binary(df_copy[target_col])
        # if not is_binary(df[target_col]):
        #     raise ValueError('binary target expected for family "binomial"')
        sm_family = sm.families.Binomial()
    else:
        raise ValueError(f'invald family {family} - expecting one of "normal" or "binomial"') 

    # use Q() convention in case feature names contain spaces
    # use C() convention for categorical
    target_col_formula = f'Q("{target_col}")'
    feature_col_formula = f'Q("{df_copy.columns[-1]}")'

    if is_discrete(df_copy[df_copy.columns[-1]]):
        feature_col_formula = f'C({feature_col_formula})'

    null_model = smf.glm(formula=f'{target_col_formula} ~ 1', data=df_copy.dropna(), family=sm_family).fit()
    full_model = smf.glm(formula=f'{target_col_formula} ~ {feature_col_formula}', data=df_copy.dropna(), family=sm_family).fit()

    # LR test to get a single p-value for the feature
    ll_ratio = 2 * (full_model.llf - null_model.llf)  
    df_diff = full_model.df_model - null_model.df_model  
    p_value = stats.chi2.sf(2 * ll_ratio, df_diff)

    # distinct values of processed feature col for generating predictions
    df_distinct = df_copy[[df_copy.columns[-1]]].drop_duplicates().sort_values(df_copy.columns[-1]).reset_index(drop=True)

    # calculate CIs - temporaily disable RuntimeWarnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        preds = full_model.get_prediction(df_distinct).summary_frame(alpha=alpha)

    # fix a peculiar issue with the statsmodels CIs where mean_se=0 and mean_ci is [0, 1]
    zero_se_mask = np.isclose(preds['mean_se'], 0, atol=1e-05)
    preds.loc[zero_se_mask, ['mean_ci_lower', 'mean_ci_upper']] = preds.loc[zero_se_mask, 'mean']

    # append distinct feature values to predictions for plotting
    preds = pd.concat([df_distinct, preds], axis=1)

    if is_discrete(df_copy[df_copy.columns[-1]]):

        means = preds["mean"]
        lower = preds["mean_ci_lower"]
        upper = preds["mean_ci_upper"]
        yerr = [means - lower, upper - means]

        ax2.bar(
            preds[df_copy.columns[-1]].astype(str), 
            preds["mean"], 
            yerr=yerr, 
            capsize=5,
        )
        ax2.tick_params(axis='x', labelrotation=90 if discretize else 0)

    else:

        # predicted mean curve
        ax2.plot(
            preds[df_copy.columns[-1]],
            preds["mean"],
            label="Mean estimate"
        )

        # confidence interval shading
        ax2.fill_between(
            preds[df_copy.columns[-1]],
            preds["mean_ci_lower"],
            preds["mean_ci_upper"],
            alpha=0.25,
            label=f"Mean {1-alpha:.0%} CI"
        )

        # optional: show observed data
        ax2.scatter(
            df_copy[feature_col_obs],
            df_copy[target_col],
            alpha=0.4,
            s=25
        )

        ax2.legend(bbox_to_anchor=(1, 1), loc="upper left")
        # ax.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=2)

    if family in ['binomial', 'normal', 'poisson', 'gamma']:
        ax2.axhline(df_copy.dropna(subset=df_copy.columns[-1])[target_col].mean(), linestyle=":", color="black")
    
    ax2.set_title(f'{family} GLM (p={p_value:.5f})')
    ax2.set_ylabel(target_col)
    ax2.set_xlabel(df_copy.columns[-1])
    ax2.tick_params(axis='x', labelrotation=rot)


    return fig, ax


def is_binary(x: pd.Series) -> bool:
    x = pd.Series(x)
    return is_integer_dtype(x) and x.dropna().isin([0, 1]).all()


def as_binary(x: pd.Series) -> bool:
    x = pd.Series(x)
    if is_binary(x):
        return x
    uniques = sorted(x.dropna().unique())
    n_uniques = len(uniques)
    if n_uniques != 2:
        raise ValueError(f'expecting 2 unique values: x has {n_uniques} uniques')
    return x.map(dict(zip(uniques, [0, 1]))).astype(float)


def is_categorical(x: pd.Series) -> bool:
    x = pd.Series(x)
    return x.dtype in ['object', 'category']


def as_categorical(x: pd.Series, explicit_na=False):
    x = pd.Series(x)
    x = pd.Categorical(x)
    if explicit_na:
        x = x.add_categories(['MISSING'])
        x = x.fillna('MISSING')
    return x


def is_discrete(x: pd.Series) -> bool:
    x = pd.Series(x)
    return x.dtype in ['object', 'category', 'bool']


def discretize_x(x: pd.Series, nbins: int = 10, quantiles=False) -> pd.Series:
    if is_discrete(x):
        return x
    if x.nunique() <= nbins:
        return x.astype('str')
    if quantiles:
        return pd.qcut(x, q=nbins, duplicates='drop')
    return pd.cut(x, bins=nbins, duplicates='drop')


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




