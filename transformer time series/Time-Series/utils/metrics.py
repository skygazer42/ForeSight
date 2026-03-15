import numpy as np


def rse(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def corr(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def mean_absolute_error(pred, true):
    return np.mean(np.abs(pred - true))


def mean_squared_error(pred, true):
    return np.mean((pred - true) ** 2)


def root_mean_squared_error(pred, true):
    return np.sqrt(mean_squared_error(pred, true))


def mean_absolute_percentage_error(pred, true):
    return np.mean(np.abs((pred - true) / true))


def mean_squared_percentage_error(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = mean_absolute_error(pred, true)
    mse = mean_squared_error(pred, true)
    rmse = root_mean_squared_error(pred, true)
    mape = mean_absolute_percentage_error(pred, true)
    mspe = mean_squared_percentage_error(pred, true)

    return mae, mse, rmse, mape, mspe
