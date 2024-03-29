# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 17:41:00 2021

@author: haznom35
"""
from GSKpy.BasicGSK import BasicGSK
from GSKpy.viz import Viz
import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum

def ackley(x,func_args=[20,0.2,2*pi]):
    #The objective function should take x
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    a= func_args[0]
    b= func_args[1]
    c=func_args[2]

    n = len(x[0])
    s1 = sum( x**2 ,axis=1)
    s2 = sum( cos( c * x ),axis=1)
    return -a*exp( -b*sqrt( s1 / n )) - exp( s2 / n ) + a + exp(1)
dim = 10
max_nfes = 1000000
solver_basic = BasicGSK(ackley,dim,100,[-100]*dim, [100]*dim,LPSR=False,max_nfes=max_nfes,func_args=[20,0.2,2*pi],k=10,kf=0.5,kr=0.9,p=0.1)
g,best , best_fit, errors = solver_basic.run(track=True)

print('\nbestfit:',best_fit)


vis = Viz(ackley,-100,100,dim,func_args=[20,0.2,2*pi])
best_hist,fitness_vals, best, middle, worst,junior_dim = solver_basic.getstatistics()

best_hist = np.array(best_hist)
best_hist = np.vstack((best_hist))
best_hist = best_hist.reshape((best_hist.shape[0],dim))
vis.set(dim,best_hist,fitness_vals,best,middle,worst)
vis.build_plot()
x = np.linspace(0,g,100)
