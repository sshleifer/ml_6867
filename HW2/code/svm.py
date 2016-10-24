import pylab as pl

import numpy as np
from cvxopt import matrix, solvers
LINEAR_KERNEL = np.dot

from code.helpers import plotDecisionBoundary

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x,y, sigma=1.):
    return np.exp(-np.linalg.norm(x-y)**2 / 2 * (sigma **2))


def get_theta(sv, svy, ind, alpha, K, non_zero_mask):
        theta_0 = 0   #intercept
        for n in range(len(alpha)):
            theta_0 +=  (svy[n] - np.sum(alpha *svy * K[ind[n],  non_zero_mask]))
            theta_0 /= len(alpha)  # seems wierd
        return theta_0


class SVMD(object):
    def __init__(self, kernel=LINEAR_KERNEL, C=1.):
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.X =  X
        self.y = y
        self.K = np.array([[self.kernel(X[i], X[j]) for i in range(n_samples)] for j in range(n_samples)])
        self.n_features = X.shape[1]
        self.solution = self._solve()
        alpha, ind, mask = self.inspect(self.solution)
        svx = self.X[mask]
        svy = self.y[mask]
        self.svx, self.svy, self.ind, self.alpha, self.mask = svx, svy, ind, alpha[mask], mask
        self.coef_ = (self.alpha*self.svy).dot(self.svx)
        self.theta = get_theta(self.svx, self.svy,  self.ind, self.alpha, self.K, self.mask)
        return self


    def _solve(self):
        X,y, kernel, C = self.X, self.y, self.kernel, self.C  # to avoid typing self
        n_samples, n_features = X.shape
        print(n_samples)
        P = matrix(np.outer(y,y) * self.K)
        q = matrix(np.ones(n_samples)*-1.)
        A = matrix(y, (1, n_samples))
        b = matrix(0.)
        G = matrix(np.vstack([np.diag(np.ones(n_samples)) * -1, np.diag(np.ones(n_samples))]))
        h = matrix(np.hstack([np.zeros(n_samples), self.C *np.ones(n_samples)]))
        return solvers.qp(P, q, G, h, A, b)


    @staticmethod
    def inspect(solution, cutoff=1e-5):
        alpha = np.ravel(solution['x'])
        non_zero_mask = alpha > cutoff
        ind = np.arange(len(alpha))[non_zero_mask]
        #print ind.mean()
        return alpha,ind, non_zero_mask

    def predict(self, X_new):
        self.weights = (self.alpha*self.svy).dot(self.svx)
        self.margin = 1. / np.linalg.norm(self.weights)
        if self.kernel ==  LINEAR_KERNEL:
            return np.sign(np.dot(X_new, self.weights) + self.theta)
        else:
            pred_val = [np.sum([a*y*self.kernel(X[i], x) for a,y,x in zip(self.alpha, self.svy, self.svx)])
                        for i in range(X_new.shape[0])]
            self.margin = None
            return np.sign(np.array(pred_val) + self.theta)

    def score(self, X_new, y_new):
        yhat = self.predict(X_new)
        assert yhat.shape ==  y_new.shape, "Shape mismatch"
        return (yhat == y_new).mean()

    def plot_boundary(self, X, y, **kwargs):
        plotDecisionBoundary(X,y, self.predict, [-1,0,1], **kwargs)
        pl.show()


MAX_EPOCHS =1e4

def pegasos(X, y, L=2, max_epochs=MAX_EPOCHS):
    assert L > 0, "pegasos needs L > 0 to build a learning schedule"
    # give a few extra epochs to avoid IndexError
    N = np.array([1. / (t*L) for t in range(1, int(max_epochs * 1.1))])
    w = np.zeros(X.shape[1])
    t = 0
    paths = []
    while t < max_epochs:
        for i, row in enumerate(X):
            t += 1
            w_new = (1 - N[t]*L) * w
            if y[i]* w.T.dot(row) < 1:
                w = w_new + N[t]*y[i]*row
            else:
                w = w_new
            if t % 1e2 ==0:
                paths.append(w_new)

    return w, paths


class Pegasos(object):
    def __init__(self, L=2.):
        self.L = L

    def fit(self, X, y, **kwargs):
        self.coef_, self.paths_ = pegasos(X, y, L=self.L, **kwargs)
        return self

    def predict(self, X):
        return X.dot(self.coef_)

