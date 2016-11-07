import numpy as np


def get_lr(t, tau=1e-6, k=.5):
    '''Learning rate for iteration t'''
    return tau + (t + 1) ** -k


def relu(x):
    return np.clip(x, 0, None)

def dsigmoid(y):
    return 1.0 - y * y


def dsoftmax(x):
    return np.clip(np.sign(x), 0, 1)


def stable_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def assert_no_nans(struct):
    '''Make sure there are no np.nan floating around in the array/dict'''
    if isinstance(struct, dict):
        ret_val = np.isnan(struct.values()).any()
    else:
        ret_val = np.isnan(struct).any()
    if ret_val:
        raise AssertionError(struct)
