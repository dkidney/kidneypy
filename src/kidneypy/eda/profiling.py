import pandas as pd


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """    
    n_unique = df.nunique().to_frame(name='n_unique')
    n_na = df.isna().sum().to_frame(name='n_na')
    mode = df.mode(axis=0, dropna=True).T.apply(lambda x: x.dropna().to_list(), axis=1).to_frame(name='mode')
    desc = df.describe().T.drop('count', axis=1)

    _ = df.dtypes.to_frame(name='type')\
        .merge(n_unique, left_index=True, right_index=True)\
        .merge(n_na, left_index=True, right_index=True)\
        .assign(p_na=lambda df: df["n_na"] / df.shape[0])\
        .merge(mode, left_index=True, right_index=True)\
        .merge(desc, left_index=True, right_index=True)
    
    return _