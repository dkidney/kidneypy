
import kidneypy as kp

DF = kp.generate_example_df()

PROFILE = kp.df.profiling._profile_df(DF)

# NUMERIC_COLS = DF.columns.to_series()[lambda s: s.str.startswith(('float', 'int'))].index.to_list()

def test_identify_numeric_cols():
    kp.df.profiling._identify_sparse_cols(PROFILE, max_p_missing=0.2)
