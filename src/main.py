import numpy as np
from numpy import ndarray
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model.layers import Dense
from model.losses import MeanSquaredError
from model.metrics import eval_regression_model
from model.networks import NeuralNetwork
from model.operations import Sigmoid, Linear
from model.optimizers import SGD
from model.trainers import Trainer


def main():
    lr = NeuralNetwork(
        layers=[Dense(neurons=1,
                      activation=Linear())],
        loss=MeanSquaredError(),
        seed=20190501
    )

    nn = NeuralNetwork(
        layers=[Dense(neurons=13,
                      activation=Sigmoid()),
                Dense(neurons=1,
                      activation=Linear())],
        loss=MeanSquaredError(),
        seed=20190501
    )

    dl = NeuralNetwork(
        layers=[Dense(neurons=13,
                      activation=Sigmoid()),
                Dense(neurons=13,
                      activation=Sigmoid()),
                Dense(neurons=1,
                      activation=Linear())],
        loss=MeanSquaredError(),
        seed=20190501
    )

    california = fetch_california_housing()
    data = california.data
    target = california.target
    features = california.feature_names

    # Scaling the data
    s = StandardScaler()
    data = s.fit_transform(data)

    def to_2d_np(a: ndarray,
                 type: str = "col") -> ndarray:
        '''
        Turns a 1D Tensor into 2D
        '''

        assert a.ndim == 1, \
            "Input tensors must be 1 dimensional"

        if type == "col":
            return a.reshape(-1, 1)
        elif type == "row":
            return a.reshape(1, -1)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)

    # make target 2d array
    y_train, y_test = to_2d_np(y_train), to_2d_np(y_test)

    # helper function

    def permute_data(X, y):
        perm = np.random.permutation(X.shape[0])
        return X[perm], y[perm]

    trainer = Trainer(lr, SGD(lr=0.01))

    trainer.fit(X_train, y_train, X_test, y_test,
                epochs=50,
                eval_every=10,
                seed=20190501);
    print()
    eval_regression_model(lr, X_test, y_test)

    trainer = Trainer(nn, SGD(lr=0.01))

    trainer.fit(X_train, y_train, X_test, y_test,
                epochs=50,
                eval_every=10,
                seed=20190501);
    print()
    eval_regression_model(nn, X_test, y_test)

    trainer = Trainer(dl, SGD(lr=0.01))

    trainer.fit(X_train, y_train, X_test, y_test,
                epochs=50,
                eval_every=10,
                seed=20190501);
    print()
    eval_regression_model(dl, X_test, y_test)


if __name__ == '__main__':
    main()
