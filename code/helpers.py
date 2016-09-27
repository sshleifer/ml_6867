from __future__ import division
import numpy as np
import pylab as pl

from code.constants import *

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


def J(X, y, w):
    return np.sum((X.dot(w) - y)**2)


def j(i,w):
    '''Loss for one row of x, y'''
    return np.sum((X[i].dot(w) - y[i])**2)