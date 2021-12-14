import numpy as np
from math import sqrt
from collections import Counter
from numpy import ndarray


class KNNClassifier:
    def __init__(self, k: int) -> None:
        assert k >= 1, "k must be valid"
        self.k: int = k
        self._X_train: ndarray = None
        self._y_train: ndarray = None

    def fit(self, X_train: ndarray, y_train: ndarray):
        # 根据训练数据集X_train和y_train训练kNN分类器
        assert (
            X_train.shape[0] == y_train.shape[0]
        ), "the size of X_train must be equal to the size of y_train"
        assert self.k <= X_train.shape[0], "the size of X_train must be at least k."

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict) -> ndarray:
        # 给定代预测数据集X_predict, 返回表示X_predict的结果
        assert (
            self._X_train is not None and self._y_train is not None
        ), "must fit before predict!"
        assert (
            X_predict.shape[1] == self._X_train.shape[1]
        ), "the feature number of X_predict must be equal to X_train"
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x: ndarray) -> int:
        # 给定耽搁代预测的x，返回x_predict的预测结果值
        assert (
            x.shape[0] == self._X_train.shape[1]
        ), "the feature number of x must be equal to X_train"
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[: self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def __repr__(self) -> str:
        return f"KNN(k={self.k})"
