import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import random
from itertools import product, count
from matplotlib.colors import LinearSegmentedColormap

#generate natural texture by using Perlin noise 

# quintic interpolation
def qz(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

# cubic interpolation
def cz(t):
    return -2 * t * t * t + 3 * t * t
	
def generate_unit_vectors(n):
    #Generates matrix NxN of unit length vectors
    phi = np.random.uniform(0, 2*np.pi, (n, n))
    v = np.stack((np.cos(phi), np.sin(phi)), axis=-1)
    return v

#Perlin noise function in 2D
#noise function implementation based on Ken perlin's paper (improving noise) and python version of ruslan karimov
def generate_2D_perlin_noise(size, ns):
    nc = int(round(size / ns) ) # number of nodes
    grid_size = int(round(size / ns + 1)) # number of points in grid

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
    d0 = d[..., 0].copy().reshape(ns, ns, 1, 2)
    d1 = d[..., 1].copy().reshape(ns, ns, 2, 1)
 
    m=np.zeros((size,size), dtype=np.float32)
    t = m.reshape(nc, ns, nc, ns)

    # calculate values for a NSxNS patch at a time
    for i, j in product(np.arange(nc), repeat=2):        
        # 'vector from node to point' dot 'node vector'
        adot = np.matmul(vd, v[i:i+2, j:j+2].reshape(4, 2, 1)).reshape(ns, ns, 2, 2)
        # horizontal and vertical interpolation
        t[i, :, j, :] = np.matmul(np.matmul(d0, adot), d1).reshape(ns, ns)

    return m


#get natural peformance by mixing noise functions
def getTexture(size):
    size=int((size/128)+1)*128
    img0 = generate_2D_perlin_noise(size,int(size/8))
    img1 = generate_2D_perlin_noise(size,int(size/16))
    img2 = generate_2D_perlin_noise(size,int(size/32))
    img3 = generate_2D_perlin_noise(size,int(size/64))
    img4 = generate_2D_perlin_noise(size,int(size/128))
    img = img0*0.1+img1*0.1+img2*0.1+img3*0.2+img4*0.3
    img = img*0.35 * (300 / size)
    #map noise value to RGB value of retinal image's  background
    cmap0 = LinearSegmentedColormap.from_list('cloud', [ '#b66451', '#dab375'])
    cmap1 = LinearSegmentedColormap.from_list('cloud', [ '#cd6836', '#e5b177'])
    cmap2 = LinearSegmentedColormap.from_list('cloud', [ '#ddb061', '#d25b56'])
    cmap3 = LinearSegmentedColormap.from_list('cloud', [ '#d25662', '#cc7553'])
    cmap4 = LinearSegmentedColormap.from_list('cloud', [ '#BD321C', '#D93823'])
    norm = colors.Normalize(vmin=-1, vmax=1, clip=True)
    img = cm.ScalarMappable(cmap=random.choice([cmap0, cmap1, cmap2, cmap3, cmap4]), norm=norm).to_rgba(img)


    img0 = generate_2D_perlin_noise(size,int(size/8))
    img1 = generate_2D_perlin_noise(size,int(size/16))
    img2 = generate_2D_perlin_noise(size,int(size/32))
    img3 = generate_2D_perlin_noise(size,int(size/64))
    img4 = generate_2D_perlin_noise(size,int(size/128))
    brightness = img0*0.2+img1*0.1+img2*0.1+img3*0.2+img4*0.3
    brightness = 1.0 + brightness * 0.04 * (300 / size)
    
    img[:,:,0] *= brightness
    img[:,:,1] *= brightness
    img[:,:,2] *= brightness

    
    return img
