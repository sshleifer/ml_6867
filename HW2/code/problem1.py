import funcy
import numpy as np
from scipy.optimize import minimize


def l2_reg(w):
    return np.sqrt(np.sum(w[1:] ** 2))


def l1_reg(w):
    return np.sum(np.abs(w))


def nll(X, Y, w, L=0., reg_func=l2_reg):
    '''Negative Log Likelihood'''
    yhat = X.dot(w)
    penalty = L * (reg_func(w) ** 2)
    k = 1 + np.exp(-Y * yhat)
    loss = np.sum(np.log1p(k))
    return loss + penalty


class LogReg(object):
    def __init__(self, reg_func=l2_reg, L=0.):
        self.reg_func = reg_func
        self.L = L

    def fit(self, X, y):
        self.coef_ = minimize(funcy.partial(nll, X, y), np.zeros(X.shape[1])).x
        return self

    def predict(self, X):
        return np.dot(X, self.coef_)
