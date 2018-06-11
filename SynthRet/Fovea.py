import numpy as np
import math
import random
import cv2
import matplotlib.pyplot as plt
from skimage import io, transform, draw, data
#from NoiseUtils import NoiseUtils

# generate an image with the background and fovea
def generateBackgroundAndFovea():
       img=np.zeros((300, 300, 4),np.uint8)
       img[:,:,]=[217,50,28,255]
       """
       size=250
       noise = NoiseUtils(size)
       noise.makeTexture(texture = noise.cloud)
       r1,c1=draw.circle(150,150,100)
       #r2,c2=draw.circle(150,150,50)
       #r3,c3=r1-r2,c1-c2
       #img2 = img.copy()
       for i in range(0, len(r1)):
           y=r1[i]
           x=c1[i]
           c = noise.img[y, x]
           #img2[y, x] = c
           img[y,x] = c
       #img2 = img2.astype('uint8')
       """
       cx=150+np.random.randint(-10,10)
       cy=150+np.random.randint(-10,10)
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

              
       """
       alpha = 0.8
       beta = 1-alpha
       gamma = 0
       img_add = cv2.addWeighted(img, alpha, img2, beta, gamma)
       """
       return img,PosFovea

	
def rValue(x,y,cx,cy):
    """
    #parameters
    dx=abs(cx-x)
    dy=abs(cy-y)
    d=np.sqrt(dx**2+dy**2)
    R=np.sqrt(cx**2+cy**2)
    red= 185*math.exp(-0.3*d/(R*2))
    """
    r=171    
    A = 0.1
    a = 0.05
    ther = 10
    phi = math.pi    
    exponentr = -((x-cx+A*math.cos(phi))/ther)**2 - ((y-cy+A*math.cos(phi))/ther)**2
    #red =  zr - 1/(a+math.exp(exponentr))
    red = r+1/(a+math.exp(exponentr))
    
    return red	
def bValue(x,y,cx,cy,gbx,gby):
    """
    dx=abs(cx-x)
    dy=abs(cy-y)
    d=np.sqrt(dx**2+dy**2)
    R=np.sqrt(cx**2+cy**2)
    blue= 23*math.exp(-0.6*d/(R*2))
    """
    #r parameters
    zr = 28
    a = 0.05
    ther = 30
    exponentr = -((x-cx)/ther)**2 - ((y-cy)/ther)**2
    r =  zr +1/(a+math.exp(exponentr))
    
    #parameters
    k = 10
    thegb = 8   
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
    green= 45*math.exp(-0.1*d/(R*2))
    """
    #r parameters
    zr = 49
    a = 0.05
    ther = 30
    exponentr = -((x-cx)/ther)**2 - ((y-cy)/ther)**2
    r =  zr +1/(a+math.exp(exponentr))   
    #parameters
    k = 10
    thegb = 4    
    #calculate gchanel values
    exponentgb = -((x-gbx)/thegb)**2 - ((y-gby)/thegb)**2
    #green = r+k*math.exp(exponentgb)
    green = r-k*math.exp(exponentgb)
    
    return green

d,p=generateBackgroundAndFovea()
plt.imshow(d,plt.cm.gray)
plt.savefig("D:/3.png")
#io.imshow(d)