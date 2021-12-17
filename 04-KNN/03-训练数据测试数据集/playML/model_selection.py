import numpy as np
from numpy import ndarray
from numpy.random.mtrand import shuffle


def train_test_split(X: ndarray, y: ndarray, test_ratio: float = 0.2, seed: int = None):
    assert X.shape[0] == y.shape[0], "the size of X must be equal to size of y"
    assert 0.0 <= test_ratio <= 1.0, "test_ration must be valid"

    if seed:
        np.random.seed(seed)

    shuffled_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test
