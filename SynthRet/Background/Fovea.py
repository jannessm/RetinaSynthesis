import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, transform, draw, data
from perlinNoise import getTexture
# generate an image with the background and fovea
def generateBackgroundAndFovea(sizeX,sizeY,supersample):
    
    img=np.zeros((sizeX*supersample, sizeY*supersample, 4), dtype=np.float32)  
    #fovea position
    cx=(0.5+np.random.uniform(-10/300,10/300))*sizeX*supersample
    cy=(0.5+np.random.uniform(-10/300,10/300))*sizeX*supersample
    PosFovea=(cy,cx)
    #Perlin noise texture
    img1=getTexture(sizeX*supersample)
    img[:,:]=img1[:sizeX*supersample,:sizeY*supersample]
    
    radius = sizeX/12.0*supersample
    rr,cc=draw.circle(cy,cx,radius)
    
    for i in range(len(rr)):
        y=rr[i]
        x=cc[i]
        
        sqr_d = (x-cx)**2 + (y-cy)**2
        f = sqr_d / (radius*radius)
        
        f = min(f, 1.0)
        f = 3.0 * f - 2.0 * f**1.5
        
        f = 0.5 + 0.5 * f
        
        img[y,x,0] = img[y,x,0] * f
        img[y,x,1] = img[y,x,1] * f
        img[y,x,2] = img[y,x,2] * f
        '''
        #r,g,b channel intensity value
        img[y,x,0] = rValue(x,y,cx,cy,sizeX*supersample) 
        img[y,x,1] = gValue(x,y,cx,cy,gbx,gby,sizeX*supersample) 
        img[y,x,2] = bValue(x,y,cx,cy,gbx,gby,sizeX*supersample)
        '''

    return img,(PosFovea[0]/supersample, PosFovea[1]/supersample)


#RGB intensity model based on equations in the paper of Samuele Fiorini et al 	
# def rValue(x,y,cx,cy,size):
#     #amplitude
#     r=189.00 
#     a = 0.0364
#     #the spread of the distribution surface
#     ther =(size/300.0)*8.10
#     # oscillating
#     h=1.22
#     w=2.0387
#     t = math.atan2(y-cy, x-cx)
#     exponentr = -((x-cx+h*np.cos(w*t))/ther)**2 - ((y-cy+h*np.cos(w*t))/ther)**2
#     #r intensity value
#     red = r+1/(a+math.exp(exponentr))  
#     return red
	
# def bValue(x,y,cx,cy,gbx,gby,size): 
#     #amplitude
#     zr =36.0
#     a = 0.1
#     #control the spread
#     ther =(size/300.0)*30.20
#     exponentr = -((x-cx)/ther)**2 - ((y-cy)/ther)**2
#     r =  zr+1/(a+math.exp(exponentr))
#     #amplitude of the inner part
#     k = 10.10
#     thegb =(size/300.0)*40.10
#     exponentgb = -((x-gbx)/thegb)**2 - ((y-gby)/thegb)**2
#     #b intensity value
#     blue = r-k*math.exp(exponentgb)
#     return blue

# def gValue(x,y,cx,cy,gbx,gby,size):
#     #amplitude
#     zr = 58.00
#     a = 0.1029
#     #control the spread
#     ther = (size/300.0)*30.10
#     exponentr = -((x-cx)/ther)**2 - ((y-cy)/ther)**2
#     r =  zr+1/(a+math.exp(exponentr))
#     #amplitude of the inner part
#     k =10.10
#     thegb =(size/300.0)*40.10
#     exponentgb = -((x-gbx)/thegb)**2 - ((y-gby)/thegb)**2
#     #g value
#     green = r-k*math.exp(exponentgb)  
#     return green

#img,pos=generateBackgroundAndFovea(300, 300, 2)
#io.imshow(img)

##### plot f ####
# cx = 25
# cy = 25
# r = 25
# rr,cc=draw.circle(cx,cy,r)
# f_list = []

# for i in range(len(rr)):
#     y=rr[i]
#     x=cc[i]
    
#     sqr_d = (x-cx)**2 + (y-cy)**2
#     f = sqr_d / r**2
    
#     f = min(f, 1.0)
#     f = 3.0 * f - 2.0 * f**1.5
    
#     f = 0.5 + 0.5 * f

#     f_list.append(f)
# plane = np.ones((r*2,r*2)) * 255
# plane[rr,cc] = plane[rr,cc] * np.array(f_list)

# x,y = np.meshgrid(range(r*2), range(r*2))
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x,y,plane, cmap="rainbow")
# ax.set_xlabel('x position in image')
# ax.set_ylabel('y position in image')
# ax.set_zlabel('texture intensity')
# plt.show()
