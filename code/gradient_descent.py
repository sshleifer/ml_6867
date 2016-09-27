from __future__ import division
import funcy
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LinearRegression, Lasso, LassoCV

from code.helpers import *
from code.constants import *

def _gradient_descent(func, deriv_func=None,
                      init_weights=np.array([5,5]), lr=1, stop_crit=1e-6,
                      h=1e-3, max_iter=1000):
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
        deriv_func  = numerical_gradient
    count = 0; n = 0; f_call = 0;
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
        paths['w0'].append(cur_weights[0])
        count = count + 1 if delta < stop_crit else 0
        if count >= 3:
            break
    print 'done in {} steps'.format(n)
    return cur_weights, pd.DataFrame(paths)

def gradient_descent(*args, **kwargs):
    weights, _ = _gradient_descent(*args, **kwargs)
    return weights

def numerical_gradient(x, f, h=0.00001):
    '''Numerically evaluate the gradient of f at x'''
    n  = len(x)
    out = np.zeros(len(x))
    assert not np.isnan(x).any()
    hfix =  2 * h
    for i in range(0, len(x)):
        hplus = np.copy(x)
        hminus = np.copy(x)
        hplus[i] += h
        hminus[i] -= h
        out[i] = (f(hplus) - f(hminus)) / hfix
        assert not np.isnan(out[i]), 'out:{}, x:{}'.format(out, x)

    return out


def SGD(init_weights=start_theta, stop_crit=1e-3, h=1e-3, max_iter=10000, lr=1e-5):
    '''Generic gradient descent function
    Args:
  
        init_weights: initial weights
        lr: learning rate
        stop_crit: stopping criterion
        h: step size for computing numerical gradient
        max_iter: how many iterations before stopping
    '''
    count = 0; n = 0; f_call = 0;
    cur_weights = np.copy(init_weights)
    paths = defaultdict(list)
    for n in range(max_iter):
        i = np.random.randint(0, len(X))
        cur_lr = get_lr(n) * lr
        func = funcy.partial(j, i)
        local_value = func(cur_weights)
        gradient = numerical_gradient(cur_weights, func, h)
        cur_weights = cur_weights - cur_lr * gradient
        cur_weights = cur_weights - lr * gradient
        new_value = func(cur_weights)
        delta = abs((new_value - local_value))
        #print 'cur_weights:{}. local_value: {}, delta: {}'.format(cur_weights,  local_value, delta)
        paths['delta'].append(delta)
        paths['norm'].append(np.linalg.norm(gradient))
        paths['w0'].append(cur_weights[0])
        paths['lr'].append(get_lr(n))
        count = count + 1 if delta < stop_crit else 0
        #if count >= 3:break
    print 'done in {} steps'.format(n)
    return cur_weights, pd.DataFrame(paths)