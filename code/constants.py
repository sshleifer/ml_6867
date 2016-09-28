from __future__ import division
import pylab as pl

data = pl.loadtxt('hw1code/P1/parametersp1.txt')

gaussMean = data[0,:]
gaussCov = data[1:3,:]

quadBowlA = data[3:5,:]
quadBowlb = data[5,:]
