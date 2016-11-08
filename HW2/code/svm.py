#packages
import pylab    as pl
import numpy    as np
import math     as m
from cvxopt import matrix, solvers
from helpers import plotDecisionBoundary

#kernel functions
linear_kernel = np.dot
# def gaussian_kernel(x,y, gamma=1.):
#     return np.exp(-np.linalg.norm(x-y)**2 / 2 * gamma)

def gaussian_kernel(gamma=1.0):
    return lambda x,y: np.exp(-np.linalg.norm(x-y)**2 / (2 * gamma))

# define the SVM object
class SVMD(object):
    #characteristics of the SVM
    def __init__(self, method = 'qp', kernel=linear_kernel, C=1., L=2.):
        self.kernel = kernel
        self.method = method
        self.C      = C
        self.L      = L

    # get weights of SVM solution
    @staticmethod
    def inspect(solution, thresh, cutoff=1e-5):
        alpha               = np.ravel(solution['x'])
        non_zero_mask       = (alpha > cutoff)
        non_zero_non_C_mask = (alpha > cutoff) & (alpha + cutoff < thresh)
        non_zero_ind        = np.arange(len(alpha))[non_zero_mask]
        non_zero_non_C_ind  = np.arange(len(alpha))[non_zero_non_C_mask]
        #alpha are the lambdas in SVM
        #ind are the indices of the nonzero alphas
        #non_zero_mask indicates for each alpha whether is it < epsilon or larger
        return alpha, non_zero_mask, non_zero_non_C_mask

    # get the bias term of the SVM solution
    @staticmethod
    def get_theta(svy, svalpha, svK):
        # creates the optimal 'b' given the support vectors
        theta_0 = 0  # initial value
        for n in range(len(svalpha)):
            theta_0 += (svy[n] - np.sum(svalpha * svy * svK[n, :]))
            theta_0 /= len(svalpha)  # checks out
        return theta_0

    # get the weights of an SVM solution
    # and compute various other metrics
    def fit(self, X, y):
        # find optimal alphas given method
        # matrix where entry (ij) is <phi(X_i), phi(X_j)>
        n_samples, n_features = X.shape
        self.K = np.array([[self.kernel(X[i], X[j]) for i in range(n_samples)] for j in range(n_samples)])
        self.X = X
        self.y = y
        if self.method == 'qp':
            # solve the quadratic program
            P = matrix(np.array(np.outer(y, y) * self.K))  # outer is pairwise product of two vectors
            q = matrix(np.ones(n_samples) * -1.)
            G = matrix(np.vstack([np.diag(np.ones(n_samples)) * -1, np.diag(np.ones(n_samples))]))
            h = matrix(np.hstack([np.zeros(n_samples), self.C * np.ones(n_samples)]))
            A = matrix(y, (1, n_samples))
            b = matrix(0.)
            self.solution_ = solvers.qp(P, q, G, h, A, b)
            alpha, non_zero_mask, non_zero_non_C_mask = self.inspect(solution=self.solution_,
                                                                     cutoff=1e-5,
                                                                     thresh=self.C)
            self.alpha  = alpha
            self.svY    = self.y[non_zero_mask]
            self.svK    = self.K[non_zero_mask,:][:, non_zero_mask]
            self.svX    = self.X[non_zero_mask, :]
            self.svAlpha = alpha[non_zero_mask]
            self.bias = self.get_theta(self.y[non_zero_non_C_mask],
                                       alpha[non_zero_non_C_mask],
                                       self.K[non_zero_non_C_mask,:][:,non_zero_non_C_mask])
            alpha_times_y = self.svAlpha * self.svY
            weight_norm = np.dot(alpha_times_y, np.dot(self.svK, alpha_times_y))
            self.margin = 1 / m.sqrt(weight_norm+0.1)
            self.weight = np.dot(alpha_times_y, self.svX)
            return self
        elif self.method == 'pegasos':
            if self.kernel == linear_kernel:
                max_epochs  = 10000
                N           = np.array([1. / ((t+1) * self.L) for t in range(0, max_epochs)])
                w           = np.zeros(X.shape[1])  # weight vector
                bias        = 0
                t           = 0  # iteration number
                while t < max_epochs:
                    for i, row in enumerate(X):
                        w_add = (1 - N[t] * self.L) * w
                        if y[i] * (np.dot(w, row) + bias) < 1:
                            w = w_add + N[t] * y[i] * row
                            bias += N[t]*y[i]
                        else:
                            w = w_add
                    t += 1
                self.weight = w
                self.bias   = bias
                self.margin = 1/m.sqrt(np.dot(w, w))
            else:
                max_epochs = 10000
                N = np.array([1. / ((t + 1) * self.L) for t in range(0, max_epochs)])
                alpha = np.zeros(X.shape[0])  # weight vector
                bias = 0
                t = 0  # iteration number
                while t < max_epochs:
                    for i, row in enumerate(X):
                        if y[i] * np.sum(alpha*self.K[i, :]) < 1:
                            alpha[i] += - N[t]*(self.L*alpha[i] - y[i])
                        else:
                            alpha[i] += -N[t]*self.L*alpha[i]
                    t += 1
                #non_zero_mask   = (alpha > 0.00001)
                #self.svAlpha    = alpha[non_zero_mask]
                #self.svY        = y[non_zero_mask]
                #self.svK        = self.K[non_zero_mask,:][:, non_zero_mask]
                #self.svX        = self.X[non_zero_mask,:]
                alpha_times_y   = alpha * y
                weight_norm     = np.dot(alpha_times_y, np.dot(self.K, alpha_times_y))
                self.weight     = np.dot(alpha_times_y, self.X)
                self.margin     = 1 / m.sqrt(weight_norm)
                self.bias       = 0
                self.alpha      = alpha
    #predict for a new input
    def predict(self, X_new):
        if self.method == 'qp':
            K_X_new  = np.array([self.kernel(x_, X_new) for x_ in self.svX])
            return np.sign(np.sum(self.svAlpha * self.svY * K_X_new) + self.bias)
        elif self.method == 'pegasos':
            if self.kernel == linear_kernel:
                return np.sign(np.dot(self.weight, X_new) + self.bias)
            else:
                K_X_new = np.array([self.kernel(x_, X_new) for x_ in self.X])
                return np.sign(np.sum(self.alpha * K_X_new) + self.bias)

