import numpy as np
import math
from skimage import io, transform, draw, data

def generateOpticalDisc():
    odimg = np.zeros((300, 300, 4),np.uint8)
    
    rx = 240 + np.random.randint(-20,21)
    ry = 150 + np.random.randint(-5,6)
    ra = 22 + np.random.randint(-3,4)
    rb = ra + np.random.randint(-1,2)
    gbx = rx + np.random.randint(-2,3)
    gby = ry + np.random.randint(-2,3)
    
    rr, cc=draw.ellipse(ry,rx,ra,rb)
    draw.set_color(odimg,[rr,cc],np.array([255,255,255,255]))
    
    for i in range(len(rr)):
            y=rr[i]
            x=cc[i]
            odimg[y,x,0] = odr(x,y,rx,ry) 
            odimg[y,x,1] = odg(x,y,rx,ry,gbx,gby) 
            odimg[y,x,2] = odb(x,y,rx,ry,gbx,gby) 
    return odimg, [rx,ry] #TODO select random point according to fovea pos.

def odr(x,y,rx,ry):
    #parameters
    zr = 220
    xr = rx
    yr = ry
    A = 0.05
    a = 0.015
    ther = 20
    phi = math.pi
    
    #calculate rchanel values
    exponentr = -((x-xr+A*math.cos(phi))/ther)**2 - ((y-yr+A*math.cos(phi))/ther)**2
    red =  zr - 1/(a+math.exp(exponentr))
    
    return red

def odb(x,y,rx,ry,gbx,gby):
    #r parameters
    zr = 40
    xr = rx
    yr = ry
    a = 0.015
    ther = 30

    exponentr = -((x-xr)/ther)**2 - ((y-yr)/ther)**2
    r =  zr - 1/(a+math.exp(exponentr))
    
    #parameters
    k = 40
    xgb = gbx
    ygb = gby
    thegb = 8
    
    #calculate bchanel values
    exponentgb = -((x-xgb)/thegb)**2 - ((y-ygb)/thegb)**2
    gb = r+k*math.exp(exponentgb)
    return gb

def odg(x,y,rx,ry,gbx,gby):
    #r parameters
    zr = 200
    xr = rx
    yr = ry
    a = 0.015
    ther = 30

    exponentr = -((x-xr)/ther)**2 - ((y-yr)/ther)**2
    r =  zr - 1/(a+math.exp(exponentr))
    
    #parameters
    k = 40
    xgb = gbx
    ygb = gby
    thegb = 8
    
    #calculate gchanel values
    exponentgb = -((x-xgb)/thegb)**2 - ((y-ygb)/thegb)**2
    gb = r+k*math.exp(exponentgb)
    return gb

d,p=generateOpticalDisc()
io.imshow(d)