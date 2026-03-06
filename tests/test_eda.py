from sklearn.datasets import load_iris

from kidneypy.eda import profile

def test_profile():
    iris = load_iris(as_frame=True)
    prof = profile(iris.frame)
    assert prof.shape[0] == iris.frame.shape[1]
