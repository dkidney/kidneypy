import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from kidneypy.eda import (
    profile_df,
    # infer_family,
)

def test_profile():
    iris = load_iris(as_frame=True)
    prof = profile_df(iris.frame)
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
