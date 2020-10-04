import numpy as np
import tensorflow
from typing import Callable


def permutation_importances_LSTM(
    est: tensorflow.python.keras.engine.sequential.Sequential,
    X_eval: np.array,
    y_eval: np.array,
    metric: Callable[[
        tensorflow.python.keras.engine.sequential.Sequential, np.array, np.array], float]
) -> np.array:
    """
    Column by column, shuffle values and observe effect on eval set for LSTM model.

    source: http://explained.ai/rf-importance/index.html
    A similar approach can be done during training. See "Drop-column importance"
    in the above article.
    """
    baseline = metric(est, X_eval, y_eval)
    imp = []
    for col in range(X_eval.shape[2]):
        save = X_eval[:, :, col].copy()
        X_eval[:, :, col] = np.random.permutation(X_eval[:, :, col])
        m = metric(est, X_eval, y_eval)
        X_eval[:, :, col] = save
        imp.append(baseline - m)
    return np.array(imp)


def permutation_importances_logistic_reg(
    est: tensorflow.python.keras.engine.sequential.Sequential,
    X_eval: np.array,
    y_eval: np.array,
    metric: Callable[[
        tensorflow.python.keras.engine.sequential.Sequential, np.array, np.array], float]
) -> np.array:
    """
    Column by column, shuffle values and observe effect on eval set.

    source: http://explained.ai/rf-importance/index.html
    A similar approach can be done during training. See "Drop-column importance"
    in the above article.
    """
    baseline = metric(est, X_eval, y_eval)
    imp = []
    for col in range(X_eval.shape[1]):
        save = X_eval[:, col].copy()
        X_eval[:, col] = np.random.permutation(X_eval[:, col])
        m = metric(est, X_eval, y_eval)
        X_eval[:, col] = save
        imp.append(baseline - m)
    return np.array(imp)


def permutation_importances_multi_cols(
    est: tensorflow.python.keras.engine.sequential.Sequential,
    X_eval: np.array,
    y_eval: np.array,
    metric: Callable[[tensorflow.python.keras.engine.sequential.Sequential, np.array, np.array], float],
    rem_cols: range
) -> np.array:
    """
    Mutlicolumns permutation function for feature importance analysis.

    Parameters
    ---------
    est: tensorflow.python.keras.engine.sequential.Sequential
        tensorflow sequential model
    X_eval: np.array
        numpy array of input for evaluation
    y_eval: np.array
        numpy array of labels
    metric: Callable[[tensorflow.python.keras.engine.sequential.Sequential, np.array, np.array], float]
        function used for computing model's performance
    rem_cols: 
        range of indices representing the columns you want, e.g. range(120, 130), 
        if you want the column indexed 120 until the column indexes 129.
    """
    baseline = metric(est, X_eval, y_eval)
    imp = []
    save = X_eval[:, rem_cols].copy()
    X_eval[:, rem_cols] = np.random.permutation(X_eval[:, rem_cols])
    m = metric(est, X_eval, y_eval)
    X_eval[:, rem_cols] = save
    imp.append(baseline - m)
    return np.array(imp)


def accuracy_metric(
    est: tensorflow.python.keras.engine.sequential.Sequential,
    X: np.array,
    y: np.array
) -> float:
    """
    TensorFlow estimator accuracy
    ."""
    pred_y = (est.predict(X))
    acc = (np.sum(y.astype(int)[0] == pred_y.round().astype(int))) / y.shape[0]
    return acc
