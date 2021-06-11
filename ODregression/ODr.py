from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math
import random


def odr(x,y):
    #parameters
    zr = 254.211
    xr = 240
    yr = 150
    a = 0.0207176
    sr = 11.9622
    
    #calculate rchanel values
    exponentr = -((x-xr)/sr)**2 - ((y-yr)/sr)**2
    red =  zr - 1/(a+np.exp(exponentr))
    
    return red

phi = math.pi * random.random()
fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(217, 263, 0.3)
Y = np.arange(123, 177, 0.3)
X, Y = np.meshgrid(X, Y)

Z = odr(X,Y)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

plt.show()


