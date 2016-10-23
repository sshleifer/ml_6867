from __future__ import division
import funcy
import numpy as np
from scipy.optimize import minimize

from code.gradient_descent import gradient_descent

def l2_reg(w):
    return np.sqrt(np.sum(w[1:] ** 2))


def l1_reg(w):
    return np.sum(np.abs(w))


def nll(X, Y, w, L=0., reg_func=l2_reg):
    '''Negative Log Likelihood'''
    yhat = X.dot(w)
    penalty = L * (reg_func(w) ** 2)
    print penalty
    k = 1 + np.exp(-Y * yhat)
    loss = np.sum(np.log1p(k))
    return loss + penalty


def sigmoid(x):
    return 1. / (1 + np.exp(x))


class LogReg(object):
    def __init__(self, reg_func=l2_reg, L=0.):
        '''Assumes class labels are 1 and -1'''
        self.reg_func = reg_func
        self.L = L


    def fit(self, X, y):
        #self.optim_ = minimize(funcy.partial(nll, X, y, L=self.L, reg_func=self.reg_func),
        #                       np.zeros(X.shape[1]))
        #self.coef_ = self.optim_.x
        gradient_func = funcy.partial(nll, X, y, L=self.L, reg_func=self.reg_func)
        self.coef_, self.optim_ = gradient_descent(gradient_func, init_weights=np.zeros(X.shape[1]))
        return self

    def predict(self, X):
        return np.sign(np.dot(X, self.coef_))

    def predict_proba(self, X):
        return sigmoid(np.dot(X, self.coef_))

    def score(self, X, y):
        yhat = np.sign(self.predict(X))
        return (yhat == y).mean()
