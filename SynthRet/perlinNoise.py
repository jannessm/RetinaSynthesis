# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 20:38:17 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import product, count 
from matplotlib.colors import LinearSegmentedColormap


# it produce more vectors pointing diagonally than vectors pointing along
# an axis
# # generate uniform unit vectors
# def generate_unit_vectors(n):
#     'Generates matrix NxN of unit length vectors'
#     v = np.random.uniform(-1, 1, (n, n, 2))
#     l = np.sqrt(v[:, :, 0] ** 2 + v[:, :, 1] ** 2).reshape(n, n, 1)
#     v /= l
#     return v

def generate_unit_vectors(n):
    'Generates matrix NxN of unit length vectors'
    phi = np.random.uniform(0, 2*np.pi, (n, n))
    v = np.stack((np.cos(phi), np.sin(phi)), axis=-1)
    return v


# quintic interpolation
def qz(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


# cubic interpolation
def cz(t):
    return -2 * t * t * t + 3 * t * t


def generate_2D_perlin_noise(size, ns):
    '''
    generate_2D_perlin_noise(size, ns)
    Generate 2D array of size x size filled with Perlin noise.
    Parameters
    ----------
    size : int
        Size of 2D array size x size.
    ns : int
        Distance between nodes.
    Returns
    -------
    m : ndarray
        The 2D array filled with Perlin noise.
    '''
    nc = int(size / ns)  # number of nodes
    grid_size = int(size / ns + 1)  # number of points in grid

    # generate grid of vectors
    v = generate_unit_vectors(grid_size)

    # generate some constans in advance
    ad, ar = np.arange(ns), np.arange(-ns, 0, 1)

    # vectors from each of the 4 nearest nodes to a point in the NSxNS patch
    vd = np.zeros((ns, ns, 4, 1, 2))
    for (l1, l2), c in zip(product((ad, ar), repeat=2), count()):
        vd[:, :, c, 0] = np.stack(np.meshgrid(l2, l1, indexing='xy'), axis=2)

    # interpolation coefficients
    d = qz(np.stack((np.zeros((ns, ns, 2)),
                     np.stack(np.meshgrid(ad, ad, indexing='ij'), axis=2)),
           axis=2) / ns)
    d[:, :, 0] = 1 - d[:, :, 1]
    # make copy and reshape for convenience
    d0 = d[..., 0].copy().reshape(ns, ns, 1, 2)
    d1 = d[..., 1].copy().reshape(ns, ns, 2, 1)

    # make an empy matrix
    #m = np.zeros((size, size,4))
    m=np.zeros((size,size))
    # reshape for convenience
    t = m.reshape(nc, ns, nc, ns)

    # calculate values for a NSxNS patch at a time
    for i, j in product(np.arange(nc), repeat=2):  # loop through the grid
        # get four node vectors
        av = v[i:i+2, j:j+2].reshape(4, 2, 1)
        # 'vector from node to point' dot 'node vector'
        at = np.matmul(vd, av).reshape(ns, ns, 2, 2)
        # horizontal and vertical interpolation
        t[i, :, j, :] = np.matmul(np.matmul(d0, at), d1).reshape(ns, ns)

    return m



def getTexture():
    size=300
    img0 = generate_2D_perlin_noise(size,20)
    img1 = generate_2D_perlin_noise(size,20)
    img2 = generate_2D_perlin_noise(size,50)
    img = (img0 + img1+img2) / 3

    cmap = LinearSegmentedColormap.from_list('cloud', [ '#BD321C','#D9321C','#D93823'])
                                              
    img = cm.ScalarMappable(cmap=cmap).to_rgba(img)
    
    return img
#img=getTexture()
#plt.imshow(img)

