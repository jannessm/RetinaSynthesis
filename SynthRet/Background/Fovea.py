import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
from skimage import io, transform, draw, data
from perlinNoise import getTexture
# generate an image with the background and fovea
def generateBackgroundAndFovea(sizeX,sizeY):
    
       img=np.zeros((sizeX, sizeY, 4), dtype=np.float32)  
       #fovea position
       cx=sizeX/2.0+np.random.randint(-10,10)
       cy=sizeX/2.0+np.random.randint(-10,10)
       PosFovea=(cy,cx)
       #Perlin noise texture
       img1=getTexture(sizeX)
       img[:,:,]=img1[:sizeX,:sizeY,]*255
      
       rr,cc=draw.circle(cy,cx,sizeX/12.0)
       gbx=cx + np.random.randint(-1,1)
       gby=cy + np.random.randint(-1,1)
       for i in range(len(rr)):
                y=rr[i]
                x=cc[i]
                #r,g,b channel intensity value
                img[y,x,0] = rValue(x,y,cx,cy,sizeX) 
                img[y,x,1] = gValue(x,y,cx,cy,gbx,gby,sizeX) 
                img[y,x,2] = bValue(x,y,cx,cy,gbx,gby,sizeX)
       
       return img / 255,PosFovea
#RGB intensity model based on equations in the paper of Samuele Fiorini12 et al 	
def rValue(x,y,cx,cy,size):
    #amplitude
    r=189.00 
    a = 0.0364
    #the spread of the distribution surface
    ther =(size/300.0)*8.10
    # oscillating
    h=1.22
    w=2.0387
    exponentr = -((x-cx+h*np.cos(w*time.time()))/ther)**2 - ((y-cy+h*np.cos(w*time.time()))/ther)**2
    #r intensity value
    red = r+1/(a+math.exp(exponentr))  
    return red
	
def bValue(x,y,cx,cy,gbx,gby,size): 
    #amplitude
    zr =36.0
    a = 0.1
    #control the spread
    ther =(size/300.0)*30.20
    exponentr = -((x-cx)/ther)**2 - ((y-cy)/ther)**2
    r =  zr+1/(a+math.exp(exponentr))
    #amplitude of the inner part
    k = 10.10
    thegb =(size/300.0)*40.10
    exponentgb = -((x-gbx)/thegb)**2 - ((y-gby)/thegb)**2
    #b intensity value
    blue = r-k*math.exp(exponentgb)   
    return blue

def gValue(x,y,cx,cy,gbx,gby,size):
    #amplitude
    zr = 58.00
    a = 0.1029
    #control the spread
    ther = (size/300.0)*30.10
    exponentr = -((x-cx)/ther)**2 - ((y-cy)/ther)**2
    r =  zr+1/(a+math.exp(exponentr))
    #amplitude of the inner part
    k =10.10
    thegb =(size/300.0)*40.10
    exponentgb = -((x-gbx)/thegb)**2 - ((y-gby)/thegb)**2
    #g value
    green = r-k*math.exp(exponentgb)  
    return green

#img,pos=generateBackgroundAndFovea(569.596)
#io.imshow(img)
