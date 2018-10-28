import numpy as np
from skimage import io
import time

def generateOpticDisc(fovea,sizeX,sizeY):
    odimg = np.zeros((sizeY, sizeX, 4), dtype=np.float32)
    
    
    rx = fovea[0] + 77*np.random.choice([1])*round(sizeX/300.0) + np.random.randint(-5,6)*round(sizeX/300.0)
    ry = fovea[1] + np.random.randint(-3,4)*round(sizeX/300.0)
    gbx = rx + np.random.randint(-2,3)*round(sizeX/300.0)
    gby = ry + np.random.randint(-2,3)*round(sizeX/300.0)
    
    for i in np.arange(sizeX):
        for j in np.arange(sizeY):
            if odr(i,j,rx,ry,sizeX,sizeY) > 245:
                odimg[j,i,0] = odr(i,j,rx,ry,sizeX,sizeY)
                odimg[j,i,1] = odg(i,j,rx,ry,gbx,gby,sizeX,sizeY) 
                odimg[j,i,2] = odb(i,j,rx,ry,gbx,gby,sizeX,sizeY)
                odimg[j,i,3] = 255

    return np.transpose(odimg,(1,0,2)), [rx,ry] 

def odr(x,y,rx,ry,sizeX,sizeY):
    #parameters
    zr = 254.211+(np.random.random_sample()-0.5)*1
    
    w=60
    xr = rx+np.cos(w*time.time())
    yr = ry+np.cos(w*time.time())
    
    a = 0.0207176
    srx = 11.9622*sizeX/300
    sry = 11.9622*sizeY/300
    
    #calculate rchanel values
    exponentr = -((x-xr)/srx)**2 - ((y-yr)/sry)**2
    red =  zr - 1/(a+np.exp(exponentr))
    
    return red

def odb(x,y,rx,ry,bx,by,sizeX,sizeY):
    #r parameters
    zr = 90.9403++(np.random.random_sample()-0.5)*10
    xr = rx
    yr = ry
    a = 0.0461424
    srx = 5.79272*sizeX/300
    sry = 5.79272*sizeY/300

    exponentr = -((x-xr)/srx)**2 - ((y-yr)/sry)**2
    r =  zr - 1/(a+np.exp(exponentr))
    
    #parameters
    kb = 2.08531+(np.random.random_sample()-0.5)*0.01
    xb = bx
    yb = by
    sbx = 3.90212*sizeX/300
    sby = 3.90212*sizeY/300
    
    #calculate bchanel values
    exponentgb = -((x-xb)/sbx)**2 - ((y-yb)/sby)**2
    blue = r+kb*np.exp(exponentgb)
    return blue

def odg(x,y,rx,ry,gx,gy,sizeX,sizeY):
    #r parameters
    zr = 155.043+(np.random.random_sample()-0.5)*10
    xr = rx
    yr = ry
    a = 0.0403873
    srx = 13.3931*sizeX/300
    sry = 13.3931*sizeY/300

    exponentr = -((x-xr)/srx)**2 - ((y-yr)/sry)**2
    r =  zr - 1/(a+np.exp(exponentr))
    
    #parameters
    kg = 63.1894+(np.random.random_sample()-0.5)*1
    xg = gx
    yg = gy
    sgx = 8.05019*sizeX/300
    sgy = 8.05019*sizeY/300
    
    #calculate gchanel values
    exponentg = -((x-xg)/sgx)**2 - ((y-yg)/sgy)**2
    green = r+kg*np.exp(exponentg)
    return green