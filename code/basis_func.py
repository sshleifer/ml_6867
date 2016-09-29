from __future__ import division

import numpy as np
import pandas as pd
import random
import scipy as sp
import sys
import matplotlib as mpl
from matplotlib import pylab as pl
from ggplot import *
from sklearn.linear_model import LinearRegression

from code.gradient_descent import *

data = pl.loadtxt('hw1code/P2/curvefittingp2.txt')

X = data[0,:]
Y = data[1,:]
pts = np.array([[p] for p in pl.linspace(min(X), max(X), 100)])


def create_basis(X, M=1):
    '''Create basis for polynomial'''
    return np.array([X**i for i in range(0, M+1)]).T

def actual_process(pts):
    return np.cos(np.pi*pts) + (1.5 * np.cos(2*np.pi*pts))

def plot_df(X=X, Y=Y, M=3):
    '''Plot basis for different a given yhat'''
    pts = np.array([p for p in pl.linspace(min(X), max(X), 100)])
    X_transformed =  create_basis(X, M=M)
    clf= LinearRegression().fit(X_transformed, Y)
    yhat_real = clf.predict(X_transformed)
    yhat = clf.predict(create_basis(pts, M))
    true = actual_process(pts)
    return pd.DataFrame({'X': pd.Series(X.flatten()),
                       'Y': pd.Series(Y.flatten()),
                       'axis': pd.Series(pts.flatten()),
                       'yhat': pd.Series(yhat.flatten()),
                       'yhat_live': pd.Series(yhat_real),
                       'true': pd.Series(true.flatten())}).assign(M=M)


def gg(df):
    return (ggplot(df, aes(x='axis', y='yhat')) +
           geom_line(color="red")  +
           geom_line(aes(y='true'), color='green', size=2) +
           geom_point(aes(x='X', y='Y',size=200), alpha=0.5) +
           xlim(-0, 1.) +
           xlab('X') +
           ylab('Y')
        )

def plotter(M):
    return gg(plot_df(M=M)) + ggtitle('M={}'.format(M))

truth = np.array([ 1.,  1.5,  0.,  0.,  0.,  0.,  0.,  0.,  ]) # coefficients

def create_cos_basis(X, M=1):
    return np.array([np.cos(np.pi * i * X) for i in range(1, M+1)]).T


def cos_descent(M, X=X, Y=Y):
    Xcos = create_cos_basis(X, M=M)
    def loss(w):
        return np.sum((np.array(Xcos).dot(w) - Y)**2)
    start = np.zeros(Xcos.shape[1])
    return gradient_descent(loss,
                            init_weights=start,
                            lr=1e-2)


class BasisSearch(object):

    def __init__(self, X=X, Y=Y, M=3):
        self.M = M
        self.X = X
        self.Xt = create_basis(X, M=M)
        self.clf = LinearRegression(fit_intercept=False).fit(self.Xt, Y)
        self.coef = self.clf.coef_
        self.yhat = self.clf.predict(self.Xt)
        self.Y = Y
        self.sse = np.sum((self.yhat - self.Y)**2)

    def plot(self):
        return plotter(self.M)

if __name__ == '__main__':
    for M in [0, 1, 3, 10]:
        plotter(M).save('figures/2.1M{}.png'.format(M))
    print 'Saved charts for #2.1 to figures directory'
    z_start = pd.Series({n: cos_descent(n) for n in range(1, 9)})
    zdf = z_start.apply(pd.Series).fillna(0)
    print 'Cosine weights'
    print zdf
