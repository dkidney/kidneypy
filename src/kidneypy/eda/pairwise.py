from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats.contingency import association
from sklearn.metrics import adjusted_mutual_info_score


def pairwise_correlation(df: pd.DataFrame, method='pearson') -> pd.DataFrame:
    if method == 'pearson':
        return pairwise_correlation_pearson(df)
    elif method == 'cramers_v':
        return pairwise_correlation_cramers_v(df)
    else:
        raise ValueError('invalid method:', method)


def pairwise_correlation_pearson(df: pd.DataFrame) -> pd.DataFrame:
    df_num = df.select_dtypes('number')
    if df_num.shape[0] == 0:
        print('no numeric columns')
        return pd.DataFrame(columns=['v1', 'v2', 'cor', 'abs', 'method']) 
    cor = np.corrcoef(df_num.to_numpy(), rowvar=False)
    i, j = np.triu_indices(cor.shape[0], k=1)  # k=1 excludes diagonal
    cor = pd.DataFrame({
        "v1": np.array(df_num.columns)[i],
        "v2": np.array(df_num.columns)[j],
        "cor": cor[i, j]
    })
    cor['abs'] = cor['cor'].abs()
    cor['method'] = 'pearson'
    cor = cor.sort_values('abs', ascending=False).reset_index(drop=True)
    return cor


def pairwise_correlation_cramers_v(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes('number').columns
    cat_cols = df.columns[~df.columns.isin(num_cols)].to_list()
    df_cat = df[cat_cols]
    if df_cat.shape[1] == 0:
        return pd.DataFrame(columns=['v1', 'v2', 'cor', 'abs', 'method'])
    cor = []
    for col1 in cat_cols:
        for col2 in cat_cols:
            if col1 <= col2:
                continue
            ctab = pd.crosstab(df[col1], df[col2], dropna=False)
            cor.append({
                'v1': col1,
                'v2': col2,
                'cor': association(ctab, method='cramer')
            })
    cor = pd.DataFrame(cor)
    cor['abs'] = cor['cor'].abs()
    cor['method'] = 'cramers_v'
    cor = cor.sort_values('cor', ascending=False).reset_index(drop=True)
    return cor


def pairwise_mutual_info(X: pd.DataFrame, y: pd.Series, nbins=10) -> pd.DataFrame:
    """
    Compute pairwise mutual information for all columns of a dataframe in parallel.
 
    Args:
        X (pd.DataFrame): features
        y (pd.Series): target
        nbins (int, optional): number of bins for numeric discretization. Defaults to 10.

    Returns:
        pd.DataFrame: symmetric normalized MI matrix with values in [0,1]
    """

    X_copy = deepcopy(X)
    y_copy = deepcopy(y)

    if y_copy.nunique() > nbins:
        y_copy = pd.qcut(y_copy, q=nbins, duplicates='drop')
    y_copy = y_copy.astype(str)

    mi = []
    for col in X_copy:
        x = X_copy[col]
        if x.dtype not in ['object', 'category']:
            if x.nunique() > nbins:
                x = pd.qcut(x, q=nbins, duplicates='drop')
            x = x.astype(str)
        mi.append({
            'col': col,
            'mi': adjusted_mutual_info_score(y_copy, x)
        })

    mi = pd.DataFrame(mi)
    mi = mi.sort_values('mi', ascending=False).reset_index(drop=True)
    return mi
