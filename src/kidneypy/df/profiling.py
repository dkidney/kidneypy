from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    NonNegativeFloat,
    NonNegativeInt,
)
# from kidneypy.df._validate import (
#     validate_positive_float,
#     validate_positive_int,
#     validate_proportion,
# )
from kidneypy.utils import format_bytes


class ProfileParams(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
    )
    max_p_missing:NonNegativeFloat=Field(
        description='columns with p_missing > max_p_missing will be identfied as sparse',
        default=0.2,
        le=1.0,
    )
    max_cardinality:PositiveInt=Field(
        description='numeric columns with cardinality > max_cardinality will be identfied as high cardinality',
        default=30
    )
    min_std:PositiveFloat=Field(
        description='numeric columns with std < min_std will be identfied as low variance',
        default=1e-5,
    )
    max_p_mode:PositiveFloat=Field(
        description='non-numeric columns with p_mode > max_p_mode will be identfied as low variance',
        default=0.95,
        le=1.0,
    )


class Profiler:

    def __init__(self, df: pd.DataFrame, *, max_rows: int = 10_000) -> None:
        self.df = df
        self.max_rows = max_rows
        self.params = ProfileParams()
        self.profile = _profile_df(df, max_rows=self.max_rows)

    def clean(self,
              dry_run:bool=True,
              verbose:bool=True,
              **kwargs,
        ) -> None | pd.DataFrame:
        self.params = ProfileParams(**kwargs)
        report = self.report()
        # cols_to_drop = (
        #     self._identify_sparse_cols(verbose=verbose) +
        #     self._identify_constant_cols(verbose=verbose) +
        #     self._identify_low_variance_cols(verbose=verbose) +
        #     self._identify_high_cardinality_cols(verbose=verbose) +
        #     self._identify_any_inf_cols(verbose=verbose)
        # )
        if verbose:
            print(self.params)
            print()
            print(report)
        if dry_run:
            return None
        cols_to_drop = report.columns
        return self.df.loc[:, lambda d: ~d.columns.isin(cols_to_drop)]

    def report(self) -> pd.DataFrame:
        return pd.DataFrame({
            'sparse': pd.Series('True', index=self._identify_sparse_cols()),
            'constant': pd.Series('True', index=self._identify_constant_cols()),
            'low_variance': pd.Series('True', index=self._identify_low_variance_cols()),
            'high_cardinality': pd.Series('True', index=self._identify_high_cardinality_cols()),
            'any_inf': pd.Series('True', index=self._identify_any_inf_cols()),
        }).fillna('.').T


    def _identify_sparse_cols(self) -> list[str]:
        return _identify_sparse_cols(self.profile,
                                     max_p_missing=self.params.max_p_missing)

    def _identify_constant_cols(self) -> list[str]:
        return _identify_constant_cols(self.profile)

    def _identify_low_variance_cols(self) -> list[str]:
        return _identify_low_variance_cols(self.profile,
                                           min_std=self.params.min_std,
                                           max_p_mode=self.params.max_p_mode)

    def _identify_high_cardinality_cols(self) -> list[str]:
        return _identify_high_cardinality_cols(self.profile,
                                               max_cardinality=self.params.max_cardinality)

    def _identify_any_inf_cols(self) -> list[str]:
        return _identify_any_inf_cols(self.profile)

    def __str__(self) -> str:
        return self.profile.__str__()

    def __repr__(self) -> str:
        return self.profile.__repr__()

# _profile_df* -------------------------------------------------------------------------------------

def _profile_df(df, max_rows: None|int = None) -> pd.DataFrame:
    _df = df
    if max_rows is not None:
        # max_rows = validate_positive_int(max_rows)
        if df.shape[0] > max_rows:
            print()
            print(f'profiling the first {max_rows} rows - to profile more rows set the max_rows parameter')
            df = _df.iloc[0:max_rows, :]
    return (df
        .dtypes
        .astype('string')
        .to_frame(name='dtype')
        .pipe(_profile_df_missingness, df)
        .pipe(_profile_df_cardinality, df)
        .pipe(_profile_df_distribution, df)
        .pipe(_profile_df_pd_describe_categorical, df)
        .pipe(_profile_df_outliers, df)
        .pipe(_profile_df_n_inf, df)
        .pipe(_profile_df_memory_usage, df)
    )


def _profile_df_missingness(profile: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    return (
        profile
        .assign(
            n = (df.shape[0] - df.isna().sum()).astype("Int64"),
            n_missing = lambda d: df.shape[0] - d['n'],
            p_missing = lambda d: d['n_missing'] / df.shape[0],
        )
    )


def _profile_df_cardinality(profile: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    return (
        profile
        .assign(
            n_unique = df.nunique(dropna=True).astype("Int64"),
            p_unique = lambda d: d['n_unique'] / d['n'],
        )
    )


def _profile_df_distribution(profile: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    return (
        profile
        .assign(
            mean=df.select_dtypes(include=['number', 'bool']).mean().reindex(df.columns),
            std=df.select_dtypes(include=['number', 'bool']).std().reindex(df.columns),
            min=df.select_dtypes(include=['number']).min().reindex(df.columns),
            q25=df.select_dtypes(include=['number']).quantile(q=0.25).reindex(df.columns),
            q50=df.select_dtypes(include=['number']).quantile(q=0.50).reindex(df.columns),
            q75=df.select_dtypes(include=['number']).quantile(q=0.75).reindex(df.columns),
            max=df.select_dtypes(include=['number']).max().reindex(df.columns),
        )
        .astype({
            "mean": "Float64",
            "std": "Float64",
            "min": "Float64",
            "q25": "Float64",
            "q50": "Float64",
            "q75": "Float64",
            "max": "Float64"
        })
    )


def _profile_df_pd_describe_categorical(profile: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    pd_desc = df.select_dtypes(include=['object', 'str', 'category'])
    if pd_desc.shape[1] == 0:
        return (profile
            .assign(
                mode=pd.NA,
                n_mode=pd.NA,
                p_mode=pd.NA,
            )
        )

    # cols = df.select_dtypes(include=['object', 'str', 'category'])
    pd_desc = (
        df
        # .describe(include=['object', 'str', 'category']).T
        .assign(
            mode=lambda d: d['top'].astype('string'),
            n_mode=lambda d: d['freq'].astype("Int64"),
            p_mode=lambda d: (d['n_mode'] / d['count']).astype("Float64"),
        )
        .loc[:, ['mode', 'n_mode', 'p_mode']]
    )
    return profile.merge(pd_desc, left_index=True, right_index=True, how='left')


def _profile_df_outliers(profile: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    return (
        profile
        .assign(
            iqr = lambda d: d['q75'] - d['q25'],
            out_iqr = lambda d: (d['min'] < (d['q25'] - 1.5 * d['iqr'])) |
                                (d['max'] > (d['q75'] + 1.5 * d['iqr'])),
            out_std = lambda d: (d['min'] < (d['mean'] - d['std'] * 3)) |
                                (d['max'] > (d['mean'] + d['std'] * 3)),
        )
    )


def _profile_df_n_inf(profile: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    return (
        profile
        .assign(
            n_inf = (~np.isfinite(df.select_dtypes('number'))).sum().reindex(df.columns)
                .astype("Int64")
        )
    )


def _profile_df_memory_usage(profile: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    return (profile
        .assign(memory = df.memory_usage(index=False, deep=True))
        .assign(memory = lambda d: d['memory'].map(format_bytes))
    )

# TODO: make a "memory_usage(x: pd.Series) -> pd.Series:" function

# _idenitfy_* --------------------------------------------------------------------------------------

def _identify_cols(profile: pd.DataFrame, query:str) -> list[str]:
    return profile.query(query).index.to_list()


def _identify_sparse_cols(profile: pd.DataFrame, max_p_missing:float) -> list[str]:
    query = f'p_missing > {max_p_missing}'
    return _identify_cols(profile, query=query)


def _identify_constant_cols(profile: pd.DataFrame) -> list[str]:
    query='n_unique == 1'
    return _identify_cols(profile, query=query)


def _identify_low_variance_cols(profile: pd.DataFrame, min_std:float, max_p_mode:float) -> list[str]:
    query = f'(n_unique > 1) & ((std < {min_std}) | (p_mode > {max_p_mode}))'
    return _identify_cols(profile, query=query)


def _identify_high_cardinality_cols(profile: pd.DataFrame, max_cardinality:int) -> list[str]:
    query = f'(dtype in ["str", "object", "category"]) & (n_unique > {max_cardinality})'
    return _identify_cols(profile, query=query)


def _identify_any_inf_cols(profile: pd.DataFrame) -> list[str]:
    query = 'n_inf > 0'
    return _identify_cols(profile, query=query)


if __name__ == '__main__':
    # from pprint import pprint
    from kidneypy.df import generate_example_df

    df = generate_example_df(n=1e6)
    print(df.head())

    profile = Profiler(df)

    print()
    print(profile)

    print()
    profile.clean(dry_run=True)

    print()
    df_cleaned = profile.clean(dry_run=False, verbose=False)

    print()
    print(df_cleaned.head())
