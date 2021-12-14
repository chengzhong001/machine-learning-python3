import numpy as np
from numpy import ndarray
from math import sqrt
from collections import Counter


def kNN_classfy(k: int, X_train: ndarray, y_train: ndarray, x: ndarray) -> int:
    '''
    X_train: array([[3.39353321, 2.33127338],
                    [3.11007348, 1.78153964],
                    [1.34380883, 3.36836095],
                    [3.58229404, 4.67917911],
                    [2.28036244, 2.86699026],
                    [7.42343694, 4.69652288],
                    [5.745052  , 3.5339898 ],
                    [9.17216862, 2.51110105],
                    [7.79278348, 3.42408894],
                    [7.93982082, 0.79163723]])

    y_train: array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    x: array([8.09360732, 3.36573151])
    '''
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0], "the size of X_train must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0],  "the feature number of x must be equal to X_train"

    distances = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]
    nearest = np.argsort(distances)

    topK_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)
    return votes.most_common(1)[0][0]
