data = pl.loadtxt('P1/parametersp1.txt')

gaussMean = data[0,:]
gaussCov = data[1:3,:]

quadBowlA = data[3:5,:]
quadBowlb = data[5,:]
from __future__ import division
import numpy as np
import scipy.optimize as spo
import pandas as pd
from collections import defaultdict

np.set_printoptions(precision=4)

def _gradient_descent(func, deriv_func=None, 
                      init_weights=np.array([5,5]), lr=1, stop_crit=1e-6,
                      h=1e-10, max_iter=1000):
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

def f_gauss(x):
    '''returns a scalar'''
    mu=gaussMean
    sigma=gaussCov 
    # from parameters
    const = np.sqrt((2*np.pi)**len(mu)) * np.linalg.det(sigma)
    delta = x- mu
    expo = np.exp((-1 /2) *(delta.T.dot(np.linalg.inv(sigma)).dot(delta)))
    return - expo / const
    
def d_gauss(x, *args):
    '''returns partial derivatives shape x'''
    mu=gaussMean
    sigma=gaussCov
    return - f_gauss(x) * (np.linalg.inv(sigma)).dot(x - mu)


def f_bowl(x, A=quadBowlA, b=quadBowlb):
    return .5 * x.T.dot(A.dot(x)) - x.T.dot(b)

def d_bowl(x, func=None, h=None, A=quadBowlA, b=quadBowlb):
    return A.dot(x) - b