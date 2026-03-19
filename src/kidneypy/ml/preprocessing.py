import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Winsorizer(BaseEstimator, TransformerMixin):

    def __init__(self, upper_quantile=0.99):
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        self.cap_ = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        return np.minimum(X, self.cap_)
