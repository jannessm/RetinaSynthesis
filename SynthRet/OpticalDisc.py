import numpy as np
from skimage import io, draw

def generateOpticalDisc(fovea):
    odimg = np.zeros((300, 300, 4),np.uint8)
    
    
    rx = fovea[0] + 77*np.random.choice([1,-1]) + np.random.randint(-5,6)
    ry = fovea[1] + np.random.randint(-3,4)
    ra = 17 + np.random.randint(-2,3)
    rb = ra + np.random.randint(-2,0)
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
    zr = 254.211+(np.random.random_sample()-0.5)*1
    xr = rx
    yr = ry
    a = 0.0207176
    sr = 11.9622
    
    #calculate rchanel values
    exponentr = -((x-xr)/sr)**2 - ((y-yr)/sr)**2
    red =  zr - 1/(a+np.exp(exponentr))
    
    return red

def odb(x,y,rx,ry,bx,by):
    #r parameters
    zr = 90.9403++(np.random.random_sample()-0.5)*10
    xr = rx
    yr = ry
    a = 0.0461424
    sr = 5.79272

    exponentr = -((x-xr)/sr)**2 - ((y-yr)/sr)**2
    r =  zr - 1/(a+np.exp(exponentr))
    
    #parameters
    kb = 2.08531+(np.random.random_sample()-0.5)*0.01
    xb = bx
    yb = by
    sb = 3.90212
    
    #calculate bchanel values
    exponentgb = -((x-xb)/sb)**2 - ((y-yb)/sb)**2
    blue = r+kb*np.exp(exponentgb)
    return blue

def odg(x,y,rx,ry,gx,gy):
    #r parameters
    zr = 155.043+(np.random.random_sample()-0.5)*10
    xr = rx
    yr = ry
    a = 0.0403873
    ther = 13.3931

    exponentr = -((x-xr)/ther)**2 - ((y-yr)/ther)**2
    r =  zr - 1/(a+np.exp(exponentr))
    
    #parameters
    kg = 63.1894+(np.random.random_sample()-0.5)*1
    xg = gx
    yg = gy
    sg = 8.05019
    
    #calculate gchanel values
    exponentg = -((x-xg)/sg)**2 - ((y-yg)/sg)**2
    green = r+kg*np.exp(exponentg)
    return green

##OD generate test
#d,p=generateOpticalDisc()
#io.imshow(d)