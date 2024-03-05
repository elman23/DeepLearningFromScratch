import numpy as np
from model.networks import NeuralNetwork
from numpy import ndarray


def mae(y_true: ndarray, y_pred: ndarray):
    '''
    Compute Mean Absolute Error (MAE) for a neural network.
    '''
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: ndarray, y_pred: ndarray):
    '''
    Compute Root Mean Squared Error (RMSE) for a neural network.
    '''
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))


def eval_regression_model(model: NeuralNetwork,
                          X_test: ndarray,
                          y_test: ndarray):
    '''
    Compute MAE and RMSE for a neural network.
    '''
    preds = model.forward(X_test)
    preds = preds.reshape(-1, 1)
    print("Mean absolute error: {:.2f}".format(mae(preds, y_test)))
    print()
    print("Root mean squared error {:.2f}".format(rmse(preds, y_test)))
