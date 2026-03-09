from copy import deepcopy
import warnings

import numpy as np
import pandas as pd


def mode_var(x: pd.Series, dropna=True) -> pd.Series:
    _ = deepcopy(x)
    if dropna:
        _ = _.dropna()
    n = len(_)
    if n == 0:
        return [], None, None
    _ = _.value_counts(dropna=False)
    n_mode = _.max()
    _ = _[_ == n_mode]
    return _.index.to_list(), n_mode, n_mode / n


def mode_df(df: pd.DataFrame) -> pd.DataFrame:
    _ = [mode_var(df[x]) for x in df]
    _ = pd.DataFrame(_, columns=['mode', 'n_mode', 'p_mode'], index=df.columns)
    return _


def any_inf_df(df: pd.DataFrame) -> pd.DataFrame:
    df_numeric = df.select_dtypes('number')
    _ = [(~np.isfinite(df[x].dropna())).any() for x in df_numeric]
    _ = pd.DataFrame(_, columns=['any_inf'], index=df_numeric.columns)
    return _


def flag(prof: pd.DataFrame, mask: pd.Series, message='') -> None:
    features = prof.index[mask].to_list()
    if len(features) > 0:
        print('flag', message, ':', ', '.join(features))


def profile(df: pd.DataFrame, flags=True) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        flags (bool, optional): _description_. Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """    

    n = df.shape[0]
    _ = df.dtypes.to_frame(name='type')

    # missings
    n_na = df.isna().sum()
    p_na = n_na / n

    # uniques
    n_unique = df.nunique(dropna=True)
    p_unique = n_unique / (n - n_na)

    _ = _\
        .merge(n_unique.to_frame(name='n_unique'), left_index=True, right_index=True, how='left')\
        .merge(p_unique.to_frame(name='p_unique'), left_index=True, right_index=True, how='left')\
        .merge(n_na.to_frame(name='n_na'), left_index=True, right_index=True, how='left')\
        .merge(p_na.to_frame(name='p_na'), left_index=True, right_index=True, how='left')

    # any non-finite
    any_inf = any_inf_df(df)
    _ = _.merge(any_inf, left_index=True, right_index=True, how='left')
    _['any_inf'] = pd.Series(_['any_inf'], dtype="boolean").fillna(False)
    
    # mode - only calculate for features where n_unique is less than n rows
    cols = _.index[_['p_unique'] < 0.5]
    mode = mode_df(df[cols])
    _ = _.merge(mode, left_index=True, right_index=True, how='left')

    # pandas description
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        desc = df.describe().T.drop('count', axis=1)
    _ = _.merge(desc, left_index=True, right_index=True, how='left')

    # outliers
    iqr = _['75%'] - _['25%']
    _['out_iqr'] = (_['min'] < (_['25%'] - 1.5 * iqr)) | (_['max'] > (_['75%'] + 1.5 * iqr))
    _['out_std'] = (_['min'] < (_['mean'] - _['std'] * 3)) | (_['max'] > (_['mean'] + _['std'] * 3))

    if flags:
        flag(_, _['n_unique'] == 1, message='zero variance')        
        flag(_, (_['n_unique'] > 1) & ((_['std'] < 1e-5) | (_['p_mode'] > .99)), message='near-zero variance')        
        flag(_, _['p_na'] > .5, message='high missingness')        
        # flag(_, _['any_inf'], message='non-finite values')        
        flag(_, (_['type'].isin(['object', 'category'])) & (_['n_unique'] > 30), message='high cardinality')        

    return _
