from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math
import random


def odb(x,y):
    #r parameters
    zr = 90.9403
    xr = 240
    yr = 150
    a = 0.0461424
    sr = 5.79272

    exponentr = -((x-xr)/sr)**2 - ((y-yr)/sr)**2
    r =  zr - 1/(a+np.exp(exponentr))
    
    #parameters
    kb = 2.08531
    xb = 240
    yb = 150
    sb = 3.90212
    
    #calculate bchanel values
    exponentgb = -((x-xb)/sb)**2 - ((y-yb)/sb)**2
    blue = r+kb*np.exp(exponentgb)
    return blue

phi = math.pi * random.random()
fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(217, 263, 0.3)
Y = np.arange(123, 177, 0.3)
X, Y = np.meshgrid(X, Y)

Z = odb(X,Y)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

plt.show()