import numpy as np
import math
from skimage import io, transform, draw, data

def generateOpticalDisc():
    odimg = np.zeros((300, 300, 4),np.uint8)
    
    rr, cc=draw.ellipse(150, 240, 26, 22)
    draw.set_color(odimg,[rr,cc],np.array([255,255,255,255]))
    
    for i in range(217,263): 
        for j in range(123,177): 
            if np.array_equal(odimg[j,i], [255,255,255,255]): 
                odimg[j,i,0] = odr(i,j) 
                odimg[j,i,1] = odg(i,j) 
                odimg[j,i,2] = odb(i,j) 
    return np.transpose(odimg, (1,0,2)), [240, 150] #TODO select random point according to fovea pos.

def odr(x,y):
    #parameters
    zr = 220
    xr = 240
    yr = 150
    A = 0.05
    a = 0.015
    ther = 20
    phi = math.pi
    
    #calculate rchanel values
    exponentr = -((x-xr+A*math.cos(phi))/ther)**2 - ((y-yr+A*math.cos(phi))/ther)**2
    red =  zr - 1/(a+math.exp(exponentr))
    
    return red

def odb(x,y):
    #parameters
    zr = 40
    xr = 240
    yr = 150
    a = 0.015
    ther = 30

    exponentr = -((x-xr)/ther)**2 - ((y-yr)/ther)**2
    r =  zr - 1/(a+math.exp(exponentr))
    
    #parameters
    k = 40
    xgb = 240
    ygb = 150
    thegb = 8
    
    #calculate bchanel values
    exponentgb = -((x-xgb)/thegb)**2 - ((y-ygb)/thegb)**2
    gb = r+k*math.exp(exponentgb)
    return gb

def odg(x,y):
    #parameters
    zr = 200
    xr = 240
    yr = 150
    a = 0.015
    ther = 30

    exponentr = -((x-xr)/ther)**2 - ((y-yr)/ther)**2
    r =  zr - 1/(a+math.exp(exponentr))
    
    #parameters
    k = 40
    xgb = 240
    ygb = 150
    thegb = 8
    
    #calculate gchanel values
    exponentgb = -((x-xgb)/thegb)**2 - ((y-ygb)/thegb)**2
    gb = r+k*math.exp(exponentgb)
    return gb