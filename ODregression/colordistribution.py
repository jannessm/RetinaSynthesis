#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 11:54:52 2018

@author: chen
"""
import numpy as np
from skimage import io,transform,color,img_as_ubyte,draw
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math
import random

img = io.imread('25_training.tif')
nimg = img_as_ubyte(transform.resize(img,(300,300)))

Z = nimg[110:190,200:270,2]

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(200, 270, 1)
Y = np.arange(110, 190, 1)

X, Y = np.meshgrid(X, Y)


ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

ax.set_xlabel('x position of the pixel')
ax.set_ylabel('y position of the pixel')
ax.set_zlabel('B Intensity')

#plt.savefig('realodr.jpg')
