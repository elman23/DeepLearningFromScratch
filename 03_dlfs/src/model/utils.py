import numpy as np
from numpy import ndarray


def assert_same_shape(array: ndarray,
                      array_grad: ndarray):
    assert array.shape == array_grad.shape, \
        '''
        Two ndarrays should have the same shape;
        instead, first ndarray's shape is {0}
        and second ndarray's shape is {1}.
        '''.format(tuple(array_grad.shape), tuple(array.shape))
    return None


def permute_data(X, y):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]
