from numpy import ndarray


def accuracy_score(y_true: ndarray, y_predict: ndarray) -> float:
    assert (
        y_true.shape[0] == y_predict.shape[0]
    ), "the size of y_true must be equal to size of y_predict"
    return sum(y_true == y_predict) / len(y_true)
