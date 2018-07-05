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
       rr,cc=draw.circle(cy,cx,25.2)
       gbx=cx + np.random.randint(-1,1)
       gby=cy + np.random.randint(-1,1)
       for i in range(len(rr)):
                y=rr[i]
                x=cc[i]
                img[y,x,0] = rValue(x,y,cx,cy) 
                img[y,x,1] = gValue(x,y,cx,cy,gbx,gby) 
                img[y,x,2] = bValue(x,y,cx,cy,gbx,gby)
       
       return img,PosFovea

	
def rValue(x,y,cx,cy):

    r=189.00 
    a = 0.036
    ther = 8.10
    exponentr = -((x-cx)/ther)**2 - ((y-cy)/ther)**2
    red = r+1/(a+math.exp(exponentr))
    
    return red	
def bValue(x,y,cx,cy,gbx,gby):
    

    zr =36.00
    a = 0.1
    ther =30.20
    exponentr = -((x-cx)/ther)**2 - ((y-cy)/ther)**2
    r =  zr+1/(a+math.exp(exponentr))
    k = 10.10
    thegb =40.10
    exponentgb = -((x-gbx)/thegb)**2 - ((y-gby)/thegb)**2
    blue = r-k*math.exp(exponentgb)
    
    return blue

def gValue(x,y,cx,cy,gbx,gby):

    zr = 58.00
    a = 0.1
    ther = 30.10
    exponentr = -((x-cx)/ther)**2 - ((y-cy)/ther)**2
    r =  zr+1/(a+math.exp(exponentr))   
    k =10.10
    thegb =40.10
    exponentgb = -((x-gbx)/thegb)**2 - ((y-gby)/thegb)**2
    green = r-k*math.exp(exponentgb)
    
    return green

#b,p=generateBackgroundAndFovea()
#plt.figure()
#plt.imshow(b)
##plt.savefig("E:/a.png")