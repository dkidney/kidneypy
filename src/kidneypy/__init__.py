__version__ = '0.1.0'

from . import df
from .df import generate_example_df

# from .df.df import (
#     drop_constant_cols,
#     drop_duplicated_rows,
#     drop_empty_rows,
#     drop_near_zero_variance_cols,
#     drop_numeric_cols,u
#     drop_sparse_cols,
#     drop_zero_variance_cols,
#     identify_constant_cols,
#     identify_duplicated_rows,
#     identify_empty_rows,
#     identify_near_zero_variance_cols,
#     identify_numeric_cols,
#     identify_sparse_cols,
#     identify_zero_variance_cols,
# )

__all__ = [
    'df',
    "generate_example_df",
    # 'drop_duplicated_rows',
    # 'drop_empty_rows',
    # 'drop_numeric_cols',
    # 'drop_sparse_cols',
    # 'drop_constant_cols',
    # 'drop_zero_variance_cols',
    # 'drop_near_zero_variance_cols',
    # 'identify_duplicated_rows',
    # 'identify_empty_rows',
    # 'identify_numeric_cols',
    # 'identify_sparse_cols',
    # 'identify_constant_cols',
    # 'identify_zero_variance_cols',
    # 'identify_near_zero_variance_cols',
]
