import numpy as np
from numpy import ndarray


class StandardScaler:
    def __init__(self) -> None:
        self.mean_: ndarray = None
        self.scale_: ndarray = None

    def file(self, X: ndarray):
        assert X.ndim == 2, "The dimension of X must be 2"
        self.mean_ = np.array([np.mean(X[:i]) for i in range(X.shape[1])])
        self.scale_ = np.std([np.mean(X[:i]) for i in range(X.shape[1])])
        return self

    def transform(self, X: ndarray):
        assert X.ndim == 2, "The dimension of X must be 2"
        assert (
            self.mean_ is not None and self.scale_ is not None
        ), "must fit before transform"
        assert X.shape[1] == len(
            self.mean_
        ), "the feature number of X must be equal to mean_ and std_"
        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]

        return resX
