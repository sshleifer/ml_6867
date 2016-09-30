from __future__ import division
import pylab as pl

data = pl.loadtxt('hw1code/P1/parametersp1.txt')

gaussMean = data[0,:]
gaussCov = data[1:3,:]

quadBowlA = data[3:5,:]
quadBowlb = data[5,:]

X1 = pl.loadtxt('hw1code/P1/fittingdatap1_x.txt')
Y1 = pl.loadtxt('hw1code/P1/fittingdatap1_y.txt')
data2 = pl.loadtxt('hw1code/P2/curvefittingp2.txt')
X2 = data2[0, :]
Y2 = data2[1, :]

