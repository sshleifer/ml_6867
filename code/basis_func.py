import numpy as np
import pandas as pd
import random
import scipy as sp
import sys
import matplotlib as mpl
from matplotlib import pylab as pl
from ggplot import *
from sklearn.linear_model import LinearRegression

data = pl.loadtxt('hw1code/P2/curvefittingp2.txt')

X = data[0,:]
Y = data[1,:]
pts = np.array([[p] for p in pl.linspace(min(X), max(X), 100)])


def create_basis(X, M=1):
    return np.matrix([X**i for i in range(0,M+1)]).T

def OLS(X=X, Y=Y, M=1):
    '''Max likelihood weight vector'''
    X_transformed =  create_basis(X, M=M)
    return LinearRegression().fit(X_transformed, Y)

def ypred(X=X, Y=Y, M=1, X2=None):
    '''Max likelihood weight vector'''
    X_transformed =  create_basis(X, M=M)
    clf = LinearRegression().fit(X_transformed, Y)
    return clf.predict(X_transformed)

def coef(X=X, Y=Y, M=1): return OLS(X,Y, M).coef_

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
    theme_bw()
    mpl.rcParams["figure.figsize"] = ".5, .5"
    #mpl.rcParams['axes.facecolor']='b'
    #gsize = theme_bw(rc={"figure.figsize": "2, 2"})
    p = (ggplot(df, aes(x='axis', y='yhat')) +
         # facet_wrap('M') +
           geom_line(color="red")  +
           geom_line(aes(y='true'), color='green', size=2) +
           geom_point(aes(x='X', y='Y',size=200), alpha=0.5) +
           xlim(-0, 1.) +
           xlab('X') +
           ylab('Y')
        # +gsize()
        )
    return p

def plotter(M):
    return gg(plot_df(M=M)) + ggtitle('M={}'.format(M))


if __name__ == '__main__':
    plotter(3)
