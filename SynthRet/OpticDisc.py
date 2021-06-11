import numpy as np
import math

def generateOpticDisc(fovea,sizeX,sizeY,supersample):
    odimg = np.zeros((sizeY*supersample, sizeX*supersample, 4), dtype=np.float32)
    
    
    rx = fovea[0]*supersample + 77*np.random.choice([1])*sizeX/300.0*supersample + np.random.uniform(-5,6)*sizeX/300.0*supersample
    ry = fovea[1]*supersample + np.random.uniform(-3,4)*sizeX/300.0*supersample
    gbx = rx + np.random.uniform(-2,3)*sizeX/300.0*supersample
    gby = ry + np.random.uniform(-2,3)*sizeX/300.0*supersample

    xrange = range(int(rx - 100*sizeX/300), int(rx + 100*sizeX/300))
    yrange = range(int(ry - 100*sizeX/300), int(ry + 100*sizeX/300))
    
    for i in xrange:
        for j in yrange:
            if odr(i,j,rx,ry,sizeX*supersample,sizeY*supersample) > 245:
                odimg[j,i,0] = odr(i,j,rx,ry,sizeX*supersample,sizeY*supersample)
                odimg[j,i,1] = odg(i,j,rx,ry,gbx,gby,sizeX*supersample,sizeY*supersample) 
                odimg[j,i,2] = odb(i,j,rx,ry,gbx,gby,sizeX*supersample,sizeY*supersample)
                odimg[j,i,3] = 150 # make this a bit transparent

    return np.transpose(odimg,(1,0,2)) / 255, [rx/supersample,ry/supersample] 

def odr(x,y,rx,ry,sizeX,sizeY):
    #parameters
    zr = 254.211+(np.random.random_sample()-0.5)
    
    
    t = math.atan2(y-ry, x-rx)
    
    #w=60
    w = 2
    xr = rx+np.cos(w*t)*sizeX/300
    yr = ry+np.cos(w*t)*sizeY/300
    
    a = 0.0207176
    srx = 11.9622*sizeX/300
    sry = 11.9622*sizeY/300
    
    #calculate rchanel values
    exponentr = -((x-xr)/srx)**2 - ((y-yr)/sry)**2
    red =  zr - 1/(a+math.exp(exponentr))
    
    return red

def odb(x,y,rx,ry,xb,yb,sizeX,sizeY):
    #r parameters
    zr = 90.9403+(np.random.random_sample()-0.5)*10
    xr = rx
    yr = ry
    a = 0.0461424
    srx = 5.79272*sizeX/300
    sry = 5.79272*sizeY/300

    exponentr = -((x-xr)/srx)**2 - ((y-yr)/sry)**2
    r =  zr - 1/(a+np.exp(exponentr))
    
    #parameters
    kb = 2.08531+(np.random.random_sample()-0.5)*0.01
    sbx = 3.90212*sizeX/300
    sby = 3.90212*sizeY/300
    
    #calculate bchanel values
    exponentgb = -((x-xb)/sbx)**2 - ((y-yb)/sby)**2
    blue = r+kb*math.exp(exponentgb)
    return blue

def odg(x,y,xr,yr,xg,yg,sizeX,sizeY):
    #r parameters
    zr = 155.043+(np.random.random_sample()-0.5)*10
    a = 0.0403873
    srx = 13.3931*sizeX/300
    sry = 13.3931*sizeY/300

    exponentr = -((x-xr)/srx)**2 - ((y-yr)/sry)**2
    r =  zr - 1/(a+np.exp(exponentr))
    
    #parameters
    kg = 63.1894+(np.random.random_sample()-0.5)
    sgx = 8.05019*sizeX/300
    sgy = 8.05019*sizeY/300
    
    #calculate gchanel values
    exponentg = -((x-xg)/sgx)**2 - ((y-yg)/sgy)**2
    green = r+kg*math.exp(exponentg)
    return green