from numpy import *
import numpy as np
import pandas as pd
import pylab as pl

# X is data matrix (each row is a data point)
# Y is desired output (1 or -1)
# scoreFn is a function of a data point
# values is a list of values to plot

def plotDecisionBoundary(X, Y, scoreFn, values, title = ""):
    # Plot the decision boundary. For that, we will asign a score to
    # each point in the mesh [x_min, m_max]x[y_min, y_max].
    #print 'here'
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = max((x_max-x_min)/200., (y_max-y_min)/200.)
    xx, yy = meshgrid(arange(x_min, x_max, h),
                      arange(y_min, y_max, h))
    Xt = c_[xx.ravel(), yy.ravel()]
    return Xt
    zz = scoreFN(Xt)
    zz = zz.reshape(xx.shape)
    pl.figure()
    CS = pl.contour(xx, yy, zz, values, colors = 'green', linestyles = 'solid', linewidths = 2)
    pl.clabel(CS, fontsize=9, inline=1)
    # Plot the training points
    pl.scatter(X[:, 0], X[:, 1], c=(1.-Y), s=50, cmap = pl.cm.cool)
    pl.title(title)
    pl.axis('tight')


def make_fname(data=1, suffix='validate'):
    return 'hw2_resources/data/data{}_{}.csv'.format(data, suffix)


def read_in(data=1, suffix='validate'):
    validate = np.loadtxt(make_fname(data, suffix))
    X = validate[:, 0:2]
    Y = np.ravel(validate[:, 2:3])
    return X, Y


base_path = 'hw2_resources/data/mnist_digit_{}.csv'

def read_mnist(digit):
    df = pd.read_csv(base_path.format(digit), header=None)[0]
    return df.apply(lambda x: np.array(map(int, x.split(' ')))).apply(pd.Series).as_matrix()


def mnist_data(digit_true, digit_false):
    Xtrue = read_mnist(digit_true)
    Xfalse = read_mnist(digit_false)
    bigy = np.concatenate([
        np.ones(Xtrue.shape[0]),
        np.zeros(Xfalse.shape[0])
    ])
    bigx = np.vstack([Xtrue, Xfalse])
    return bigx, bigy


