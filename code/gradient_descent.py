from __future__ import division
import funcy
import numpy as np
import pandas as pd
from collections import defaultdict

from code.helpers import f_gauss, d_gauss, f_bowl, d_bowl, get_lr
from code.constants import X1, Y1

def _gradient_descent(func, deriv_func=None,
                      init_weights=np.array([5., 5.]), lr=1, stop_crit=1e-6,
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
        paths['w0'].append(cur_weights[0])
        count = count + 1 if delta < stop_crit else 0
        if count >= 3:
            break
    return cur_weights, pd.DataFrame(paths)


def gradient_descent(*args, **kwargs):
    '''call _gradient_descent and return optimal weights'''
    weights, _ = _gradient_descent(*args, **kwargs)
    return weights


def numerical_gradient(x, f, h=0.00001):
    '''Numerically evaluate the gradient of f at x using central differences'''
    out = np.zeros(len(x))
    if x.dtype != float:
        x = x.astype(float)
    assert not np.isnan(x).any()
    assert h > 0
    hfix = 2 * h
    for i in range(0, len(x)):
        hplus = np.copy(x)
        hminus = np.copy(x)
        hplus[i] = hplus[i] + h
        hminus[i] = hminus[i] - h
        assert hminus[i] != x[i]
        out[i] = (f(hplus) - f(hminus)) / hfix
        assert not np.isnan(out[i]), 'out:{}, x:{}'.format(out, x)
    return out


def SGD(init_weights=np.zeros(10), stop_crit=1e-3, h=1e-3, max_iter=10000, lr=1e-5,
        X=np.copy(X1), Y=np.copy(Y1)):
    '''Generic gradient descent function
    Args:

        init_weights: initial weights
        lr: learning rate
        stop_crit: stopping criterion
        h: step size for computing numerical gradient
        max_iter: how many iterations before stopping
    '''
    def loss_fn(i, w):
        '''Loss for one row of x, y'''
        return np.sum((X[i].dot(w) - Y[i])**2)
    count = 0
    n = 0
    cur_weights = np.copy(init_weights)
    paths = defaultdict(list)
    for n in range(max_iter):
        i = np.random.randint(0, len(X))
        cur_lr = get_lr(n) * lr
        func = funcy.partial(loss_fn, i)
        local_value = func(cur_weights)
        gradient = numerical_gradient(cur_weights, func, h)
        cur_weights = cur_weights - cur_lr * gradient
        cur_weights = cur_weights - lr * gradient
        new_value = func(cur_weights)
        delta = abs((new_value - local_value))
        # print 'cur_weights:{}. local_value: {}, delta: {}'.format(cur_weights,  local_value, delta)
        paths['delta'].append(delta)
        paths['norm'].append(np.linalg.norm(gradient))
        paths['w0'].append(cur_weights[0])
        paths['lr'].append(get_lr(n))
        count = count + 1 if delta < stop_crit else 0
        if count >= 3: break
    print 'done in {} steps'.format(n)
    return cur_weights, pd.DataFrame(paths)


def g_error(start, h=1e-3):
    '''difference betwen numerical and analytical gradient for gaussian'''
    weights = np.array([start, start])
    return np.abs((numerical_gradient(weights, f_gauss, h=h) - d_gauss(weights))[0])


def b_error(start, h=1e-3):
    '''Difference betwen numerical and analytical gradient for quadratic Bowl'''
    weights = np.array([start, start])
    return np.abs((numerical_gradient(weights, f_bowl, h=h) - d_bowl(weights))[0])


if __name__ == '__main__':
    STEP_SIZES = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    gauss_errors = pd.Series({h: g_error(9.995, h=h) for h in STEP_SIZES})
    bowl_errors = pd.Series({h: b_error(9.995, h=h) for h in STEP_SIZES})
    print SGD()
    print gauss_errors, bowl_errors
