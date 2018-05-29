# imports
from PIL import Image, ImageEnhance

import cv2
import numpy as np
from skimage import img_as_ubyte
import skimage.io as io
import math
import random

def eval():
    return 0.0

#merge 3-chanel RGB images
def merge3c(collect):
    #***the size of images***
    ncol=300
    nrow=300
    #initial black image
    finalimage = np.zeros((ncol, nrow, 3),np.uint8)
    #merge layers
    for n in range(len(collect)):
        img = collect[n]
        for i in range(ncol):
            for j in range(nrow):
                if (img[i,j,:].max() > 0):
                    finalimage[i,j,:] = img[i,j,:]
    return finalimage

#merge 4-chanel RGBA images
def merge(collect):
    #***the size of images***
    ncol=300
    nrow=300
    #initial black image
    dimg = np.zeros((ncol, nrow, 4),np.uint8)
    dimg[::,::,3] = 255
    #merge layers
    for n in range(len(collect)):
        img = collect[n]
        for i in range(ncol):
            for j in range(nrow):
                #the pixel in new image is not black
                if (img[i,j,0:2].max() > 0):
                    #the pixel in old image is black, replace directly
                    if (dimg[i,j,0:2].max() == 0):
                        dimg[i,j,:] = img[i,j,:]
                    #the pixel in old image is not black, alpha blending
                    else:
                        al0 = dimg[i,j,3]/255.0
                        al1 = img[i,j,3]/255.0
                        dimg[i,j,3] = ((al1+al0*(1-al1))*255).astype(np.uint8)
                        dimg[i,j,0:3] = (
                                (img[i,j,0:3]*al1 + dimg[i,j,0:3]*al0*(1-al1))
                                /al1+al0*(1-al1)
                                ).astype(np.uint8)
                        
    return dimg

#
def addIllumination(image):
    
    # set parameters
    brightness = 2
    color = 3  
    contrast = 3  
    sharpness = 3.0

    # enhance brightness
    image1 = ImageEnhance.Brightness(image).enhance(brightness)  

    # enhance color
    image2 = ImageEnhance.Color(image1).enhance(color)   
    
    # enhance contrase 
    image3 = ImageEnhance.Contrast(image2).enhance(contrast)   
    
    # enhance sharpness 
    img = ImageEnhance.Sharpness(image3).enhance(sharpness)  

    return img

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

##code for merge test
#collect = io.ImageCollection("./*.pic")
#d=merge(collect)
#io.imshow(d)

