from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math
import random


def odr(x,y):
    #parameters
    zr = 60
    xr = 240
    yr = 150
    A = 0.05
    a = 0.015
    ther = 20
#    phi = math.pi * random.random()
    phi = math.pi
    
    #calculate rchanel values
    exponentr = -((x-xr)/ther)**2 - ((y-yr)/ther)**2

    red =  zr - 1/(a+np.exp(exponentr))
    
    return red

def odgb(x,y):
    #parameters
    k = 40
    xgb = 240
    ygb = 150
    thegb =4
    
    #calculate gbchanel values
    exponentgb = -((x-xgb)/thegb)**2 - ((y-ygb)/thegb)**2
    gb = odr(x,y)+k*np.exp(exponentgb)
    
    return gb

phi = math.pi * random.random()
fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(217, 263, 0.3)
Y = np.arange(123, 177, 0.3)
#X = np.arange(239, 241, 0.3)
#Y = np.arange(149, 151, 0.3)
X, Y = np.meshgrid(X, Y)

Z = odgb(X,Y)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

plt.show()