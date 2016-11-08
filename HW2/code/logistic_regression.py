from __future__ import division
import funcy
import numpy as np
from scipy.optimize import minimize

from gradient_descent import gradient_descent
def l2_reg(w):
    '''returns scalar'''
    return np.sum(w[1:] ** 2)
def l1_reg(w):
    return np.sum(np.abs(w[1:]))
def nll(X, Y, w, L=0., reg_func=l2_reg):
    '''Negative Log Likelihood'''
    yhat = X.dot(w[1:]) + w[0]
    regularization_penalty = L * reg_func(w)
    k = 1 + np.exp(-Y * yhat)
    loss = np.sum(np.log(k))
    return loss + regularization_penalty
def sigmoid(x):
    return 1. / (1 + np.exp(-x))
class LogReg(object):
    def __init__(self, reg_func=l2_reg, L=0.):
        '''Assumes class labels are 1 and -1'''
        self.reg_func = reg_func
        self.L = L
    def logistic_loss(self, weights):
        return nll(self.X, self.y, weights, L=self.L, reg_func=self.reg_func)
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.coef_, self.optim_ = gradient_descent(self.logistic_loss,
                                                   init_weights=np.zeros(X.shape[1]+1))
        return self
    def predict(self, X):
        return np.sign(np.dot(X, self.coef_[1:]) + self.coef_[0])

    def predict_proba(self, X):
        return sigmoid(np.dot(X, self.coef_[1:]) + self.coef_[0])

    def score(self, X, y):
        yhat = np.sign(self.predict(X))
        return (yhat == y).mean()
