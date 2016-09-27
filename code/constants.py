from __future__ import division
import numpy as np
import pylab as pl


data = pl.loadtxt('hw1code/P1/parametersp1.txt')

gaussMean = data[0,:]
gaussCov = data[1:3,:]

quadBowlA = data[3:5,:]
quadBowlb = data[5,:]


X = pl.loadtxt('hw1code/P1/fittingdatap1_x.txt')
y = pl.loadtxt('hw1code/P1/fittingdatap1_y.txt')

np.set_printoptions(precision=4)

