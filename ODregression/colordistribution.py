import numpy as np
from skimage import io, transform, img_as_ubyte
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

X = np.arange(200, 270, 1)
Y = np.arange(110, 190, 1)

def plot(Z, color, real=True):
    fig = plt.figure()
    ax = Axes3D(fig)

    X = np.arange(200, 270, 1)
    Y = np.arange(110, 190, 1)
    X, Y = np.meshgrid(X, Y)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

    ax.set_xlabel('x position of the pixel')
    ax.set_ylabel('y position of the pixel')
    ax.set_zlabel('{} Intensity'.format(color))

    plt.savefig('od_{}_{}.png'.format('real' if real else 'model', color))

def odr(x,y):
    #parameters
    zr = 254.211
    xr = 240
    yr = 150
    
    a = 0.0207176
    srx = 11.9622
    sry = 11.9622
    
    #calculate rchanel values
    exponentr = -((x-xr)/srx)**2 - ((y-yr)/sry)**2
    red =  zr - 1/(a+np.exp(exponentr))
    
    return red

def odb(x,y):
    #r parameters
    zr = 90.9403
    xr = 240
    yr = 150
    a = 0.0461424
    srx = 5.79272
    sry = 5.79272

    exponentr = -((x-xr)/srx)**2 - ((y-yr)/sry)**2
    r =  zr - 1/(a+np.exp(exponentr))
    
    #parameters
    kb = 2.08531
    sbx = 3.90212
    sby = 3.90212
    
    #calculate bchanel values
    exponentgb = -((x-xr)/sbx)**2 - ((y-yr)/sby)**2
    blue = r+kb*np.exp(exponentgb)
    return blue

def odg(x,y):
    #r parameters
    zr = 155.043
    xr = 240
    yr = 150
    a = 0.0403873
    srx = 13.3931
    sry = 13.3931

    exponentr = -((x-xr)/srx)**2 - ((y-yr)/sry)**2
    r =  zr - 1/(a+np.exp(exponentr))
    
    #parameters
    kg = 63.1894
    sgx = 8.05019
    sgy = 8.05019
    
    #calculate gchanel values
    exponentg = -((x-xr)/sgx)**2 - ((y-yr)/sgy)**2
    green = r+kg*np.exp(exponentg)
    return green

img = io.imread('25_training.tif')
nimg = img_as_ubyte(transform.resize(img,(300,300)))

model = np.zeros((300, 300, 3))
for i in X:
    for j in Y:
        model[j,i,0] = odr(i,j)
        model[j,i,1] = odg(i,j) 
        model[j,i,2] = odb(i,j)

colors = ['R', 'G', 'B']

for i in range(3):
    Z = nimg[110:190,200:270,i]
    plot(Z, colors[i])
    Z = model[110:190, 200:270, i]
    plot(Z, colors[i], False)
