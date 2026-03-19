import numpy as np
import pandas as pd
from pandas.api.types import (
    is_integer_dtype,
    is_float_dtype,
    is_bool_dtype,
    is_string_dtype,
    is_object_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
)
from sklearn.datasets import load_iris

from kidneypy.eda import (
    replace_na,
    profile_features,
    # infer_family,
)

def test_replace_na():
    series_list = [
        pd.Series(['a', 'b', 'c'], dtype=str),
        pd.Series(['a', 'b', 'c'], dtype=object),
        pd.Series([1.0, 2.0, 3.0], dtype=float),
        pd.Series([1,2,3], dtype=int),
        pd.Series([1,0,1], dtype=bool),
    ]
    for s1 in series_list:
        s2 = replace_na(s1, s1[0])
        if not (
            is_integer_dtype(s1) == is_integer_dtype(s2)
            or is_float_dtype(s1) == is_float_dtype(s2)
            or is_bool_dtype(s1) == is_bool_dtype(s2)
            or is_string_dtype(s1) == is_string_dtype(s2)
            or is_object_dtype(s1) == is_object_dtype(s2)
         ):
            raise ValueError(f'dtype of output series ({s2}) does not match dtype of input series: ({s1})')


def test_profile_features():
    iris = load_iris(as_frame=True)
    prof = profile_features(iris.frame)
    assert prof.shape[0] == iris.frame.shape[1]


# def test_infer_familyy():
#     binomial = pd.Series(np.random.binomial(n=1, p=0.5, size=100))
#     normal   = pd.Series(np.random.normal(size=100))
#     poisson  = pd.Series(np.random.poisson(10, size=100))
#     gamma    = pd.Series(np.random.gamma(2, size=100))
#     assert infer_family(binomial) == 'binary'
#     assert infer_family(normal) == 'normal'
#     assert infer_family(poisson) == 'count'
#     assert infer_family(gamma) == 'gamma'
#     binomial[0] = np.nan
#     normal[0] = np.nan
#     poisson[0] = np.nan
#     gamma[0] = np.nan
#     assert infer_family(binomial) == 'binary'
#     assert infer_family(normal) == 'normal'
#     assert infer_family(poisson) == 'count'
#     assert infer_family(gamma) == 'gamma'
