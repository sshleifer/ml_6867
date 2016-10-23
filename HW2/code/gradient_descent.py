from __future__ import division
import numpy as np
import pandas as pd
from collections import defaultdict



def gradient_descent(func, deriv_func=None,
                      init_weights=np.array([5., 5.]), lr=.1, stop_crit=1e-6,
                      h=1e-3, max_iter=10000):
    '''Generic gradient descent function
    Args:
        func: func whose gradient we compute
        deriv_func: func to compute gradient
        init_weights: initial weights
        lr: learning rate
        stop_crit: stopping criterion
        h: step size for computing numerical gradient
        max_iter: how many iterations before stopping
    '''
    if deriv_func is None:
        deriv_func = numerical_gradient
    count = 0
    n = 0
    cur_weights = np.copy(init_weights)
    paths = defaultdict(list)
    for n in range(max_iter):
        local_value = func(cur_weights)
        gradient = deriv_func(cur_weights, func, h)
        cur_weights = cur_weights - lr * gradient
        new_value = func(cur_weights)
        delta = abs((new_value - local_value))
        #print 'cur_weights:{}. local_value: {}, delta: {}'.format(cur_weights,  local_value, delta)
        paths['delta'].append(delta)
        paths['norm'].append(np.linalg.norm(gradient))
        for i, w in enumerate(cur_weights):
            paths['w{}'.format(i)].append(w)
        count = count + 1 if delta < stop_crit else 0
        if count >= 3:
            print('stopping at {}'.format(n))
            break
    return cur_weights, pd.DataFrame(paths)


def numerical_gradient(x, f, h=0.00001):
    '''Numerically evaluate the gradient of f at x using central differences'''
    out = np.zeros(len(x))
    if x.dtype != float:
        x = x.astype(float)

    assert not np.isnan(x).any()
    assert h > 0
    hfix = 2. * h
    for i in range(0, len(x)):
        hplus = np.copy(x)
        hminus = np.copy(x)
        hplus[i] = hplus[i] + h
        hminus[i] = hminus[i] - h
        out[i] = (f(hplus) - f(hminus)) / hfix
        assert not np.isnan(out[i]), 'out:{}, x:{}'.format(out, x)
    return out
