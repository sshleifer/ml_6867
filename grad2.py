import numpy as np
import scipy.optimize as spo
import pandas as pd

np.set_printoptions(precision=4)

# Various starting values
start1 = np.array([5., -3., 7., 8.])
start2 = np.array([-10., 4., 1., -5.])
start3 = np.array([1., -1., -3., 2.])

def gradient_descent(func, deriv_func, init_weights=start1, lr=.3, stop_crit=.0001, h=.001, max_iter=1000):
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
    count = 0; n = 0; f_call = 0;
    cur_weights = init_weights
    while n < max_iter and count < 2:
        local_value = func(cur_weights)
        gradient = deriv_func(cur_weights, func, h)
        cur_weights = cur_weights - lr * gradient
        new_value = func(cur_weights)
        delta = abs((new_value - local_value))
        n += 1
        f_call += 2 # unclear why
        if deriv_func == numerical_gradient:
            f_call += 2 * len(cur_weights) #unclear why

        count = count + 1 if delta < stop_crit else 0
    return cur_weights

def numerical_gradient(x, f, h=0.00001):
    '''Numerically evaluate the gradient of f at x'''
    n  = len(x)
    out = np.zeros(len(x))
    hplus = np.copy(x)
    hminus = np.copy(x)
    for i in range(0, len(x)):  # TODO: vectorize
        hplus[i] += h
        hminus[i] -= h
        # Calculates a better denominator to address potential problems with floating point arithmetic especially for small values of h
        hfix = hplus[i] - hminus[i]
        out[i] = (f(hplus) - f(hminus))/(hfix)
    return out

# Quadratic bowl
def f1(x): return np.dot(x, x)
def df1(x, *args): return 2*x

# Non-convex function with multiple local minima but easy derivative
def f2(x): return sum(x**2/100 - np.cos(x))
def df2(x, *args): return (x / 50 + np.sin(x))



results = np.empty([1,5])
# could do itertools.combinations
for i in [start1, start2, start3]:
    for lr in [.3, .03, .003]:
        for k in [.1, .001, .00001]:
            results = np.vstack(
                    [results, np.array(
                        [i, lr, k,
                            ['%.3f' % elem for elem in gradient_descent(f1, df1, np.array(i), lr=lr, crit=k)],
                            ['%.3f' % elem for elem in gradient_descent(f2, df2, np.array(i), lr=lr, crit=k)]
                            ]).reshape(1, 5)])

results = pd.DataFrame(results)
