import numpy as np
import math
import random
import matplotlib.pyplot as plt
from skimage import io, transform, draw, data
from perlinNoise import getTexture
# generate an image with the background and fovea
def generateBackgroundAndFovea():
       img=np.zeros((300, 300, 4),np.uint8)
       
       cx=150+np.random.randint(-10,10)
       cy=150+np.random.randint(-10,10)
       img1=getTexture()
       img[:,:,]=img1[:,:,]*255
       PosFovea=(cy,cx)
       rr,cc=draw.circle(cy,cx,25.4)
       gbx=cx + np.random.randint(-2,3)
       gby=cy + np.random.randint(-2,3)
       for i in range(len(rr)):
                y=rr[i]
                x=cc[i]
                img[y,x,0] = rValue(x,y,cx,cy) 
                img[y,x,1] = gValue(x,y,cx,cy,gbx,gby) 
                img[y,x,2] = bValue(x,y,cx,cy,gbx,gby)


       return img,PosFovea

	
def rValue(x,y,cx,cy):
    """
    #parameters
    dx=abs(cx-x)
    dy=abs(cy-y)
    d=np.sqrt(dx**2+dy**2)
    R=np.sqrt(cx**2+cy**2)
    red= 198*math.exp(-0.3*d/(R*2))
    """
    r=178  
    a = 0.05
    ther = 10  
    exponentr = -((x-cx)/ther)**2 - ((y-cy)/ther)**2
    #red =  zr - 1/(a+math.exp(exponentr))
    red = r+1/(a+math.exp(exponentr))
    
    return red	
def bValue(x,y,cx,cy,gbx,gby):
    """
    dx=abs(cx-x)
    dy=abs(cy-y)
    d=np.sqrt(dx**2+dy**2)
    R=np.sqrt(cx**2+cy**2)
    blue= 28*math.exp(-0.6*d/(R*2))
    """
    #r parameters
    zr =28
    a = 0.05
    ther =30
    exponentr = -((x-cx)/ther)**2 - ((y-cy)/ther)**2
    r =  zr+1/(a+math.exp(exponentr))
    
    #parameters
    k = 6
    thegb =10
    #calculate bchanel values
    exponentgb = -((x-gbx)/thegb)**2 - ((y-gby)/thegb)**2
    #blue = r+k*math.exp(exponentgb)
    blue = r-k*math.exp(exponentgb)
    
    return blue

def gValue(x,y,cx,cy,gbx,gby):
    """
    dx=abs(cx-x)
    dy=abs(cy-y)
    d=np.sqrt(dx**2+dy**2)
    R=np.sqrt(cx**2+cy**2)
    green= 50*math.exp(-0.1*d/(R*2))
    """
    #r parameters
    zr = 50
    a = 0.04
    ther = 30
    exponentr = -((x-cx)/ther)**2 - ((y-cy)/ther)**2
    r =  zr+1/(a+math.exp(exponentr))   
    #parameters
    k =12
    thegb =10
    #calculate gchanel values
    exponentgb = -((x-gbx)/thegb)**2 - ((y-gby)/thegb)**2
    #green = r+k*math.exp(exponentgb)
    green = r-k*math.exp(exponentgb)
    
    return green

d,p=generateBackgroundAndFovea()
plt.imshow(d,plt.cm.gray)

plt.savefig("D:/24.png")
#io.imshow(d)