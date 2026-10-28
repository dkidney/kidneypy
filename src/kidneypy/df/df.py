import numpy as np
import pandas as pd
import string


def generate_example_df(n:int=1000, seed:int=42):
    np.random.seed(seed)
    n = int(n)
    return (pd.DataFrame(index=range(n))
        .assign(
            all_empty=None,
            all_ones=1,
            normal=np.random.normal(size=n),
            uniform=np.random.uniform(size=n),
            binary_p50=np.random.binomial(n=1, p=0.5, size=n),
            binary_p01=np.random.binomial(n=1, p=0.01, size=n),
            randint=np.random.randint(low=0, high=10, size=n),
            bool_missing_p50=pd.Series(np.random.choice([1, 0, np.nan],
                                                        size=n,
                                                        p=[0.25, 0.25, 0.50]), dtype='Int8'),
            letters=np.random.choice(list(string.ascii_lowercase)+list(string.ascii_uppercase),
                                     size=n),
        )
    )


# # drop ========================================================================


# def drop_duplicated_rows(df):
#     return df.drop(index=identify_duplicated_rows(df))


# def drop_empty_rows(df):
#     return df.drop(index=identify_empty_rows(df))


# def drop_numeric_cols(df):
#     return df.drop(columns=identify_numeric_cols(df))


# def drop_sparse_cols(df, n, p):
#     return df.drop(columns=identify_sparse_cols(df, n, p))


# def drop_constant_cols(df):
#     return df.drop(columns=identify_constant_cols(df))


# def drop_zero_variance_cols(df):
#     return df.drop(columns=identify_zero_variance_cols(df))


# def drop_near_zero_variance_cols(df, threshold=0.01):
#     return df.drop(columns=identify_near_zero_variance_cols(df, threshold))


# # identify ====================================================================


# def identify_categorical_cols(df):
#     return select_categorical_cols(df).columns.to_list()


# def identify_duplicated_rows(df):
#     return df[df.duplicated(keep=False)].index.to_list()


# def identify_empty_cols(df):
#     nunique = df.nunique()
#     return nunique[nunique == 0].index.to_list()


# def identify_empty_rows(df):
#     return df[df.isnull().all(axis=1)].index.to_list()


# def identify_numeric_cols(df):
#     return select_numeric_cols(df).columns.to_list()


# def identify_sparse_cols(df, n, p):
#     raise NotImplementedError()


# def identify_constant_cols(df):
#     nunique = df.nunique()
#     return nunique[nunique == 1].index.to_list()


# def identify_zero_variance_cols(df):
#     raise NotImplementedError()


# def identify_near_zero_variance_cols(df, threshold=0.01):
#     raise NotImplementedError()


# # select ======================================================================


# def select_categorical_cols(df):
#     return df.select_dtypes(include=['string', 'object'])


# def select_duplicated_rows(df):
#     return df.loc[identify_duplicated_rows(df), :]


# def select_numeric_cols(df):
#     return df.select_dtypes(include=['number'])


# TODO:
# clean_column_names(df) -> df
# profile(df) -> df
# overview(df) -> ser
# pairwise_correlations(df) -> df
# col_means(df) -> ser
# col_sums(df) -> ser
# col_mins(df) -> ser
# col_maxs(df) -> ser
# col_stdevs(df) -> ser
# row_means(df) -> ser
# row_sums(df) -> ser
# describe_numeric(ser) -> bespoke
# describe_categorical(ser) -> bespoke
# detect_outliers_iqr(ser) -> list
# detect_outliers_zscore(ser) -> list
# calculate_skewness(ser) -> float
# calculate_kurtosis(ser) -> float
# memory_usage(ser) -> str
# numeric_to_pandas_numeric(ser) -> ser # e.g. allowing NAs for both int and float
# string_to_category(ser) -> ser # to save memory
# replace_values(ser) -> ser # e.g. replacing sentinel values
# string_to_numeric(ser) -> ser # e.g. following replace_values
# string_to_datetime(format)(ser) -> ser # raising an error if not compatible
# bin(ser) -> ser
# winsorise(ser) -> ser
# format_datetime(ser) -> ser
# is_weekend(ser) -> ser
# weighted_mean(ser) -> ser
# hash_id(ser) -> ser # e.g. hash an id column for printing
# rescale(min=0, max=1)(ser) -> ser # set numeric scale to be (min, max)
# pairwise_correlation(ser1, ser2) -> df
# lag(ser1, ser2) -> ser
# lead(ser1, ser2) -> ser
# rolling(ser1, ser2) -> ser
# time_difference(ser1, ser2) -> ser
# grouped_summary(ser1, ser2) -> ser # e.g. mean of col1 group by col2
# qq_plot_train_test_target(ser1, ser2) -> plt
