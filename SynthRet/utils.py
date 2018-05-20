# imports
from PIL import Image, ImageEnhance

# functions

#
import cv2
import numpy as np
from skimage import img_as_ubyte
import skimage.io as io

def eval():
    return 0.0

#merge 3-chanel RGB images
def merge3c(collect):
    #***the size of images***
    ncol=128
    nrow=128
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
def merge4c(collect):
    #***the size of images***
    ncol=300
    nrow=300
    #initial black image
    dimg = np.zeros((ncol, nrow, 4),np.uint8)
    #merge layers
    for n in range(len(collect)):
        img = collect[n]
        for i in range(ncol):
            for j in range(nrow):
                #the pixel in new image is not black
                if (img[i,j,0:3].max() > 0):
                    #the pixel in old image is black, replace directly
                    if (dimg[i,j,0:3].max() == 0):
                        dimg[i,j,:] = img[i,j,:]
                    #the pixel in old image is not black, alpha blending
                    else:
                        al0 = dimg[i,j,3]/255.0
                        al1 = img[i,j,3]/255.0
                        dimg[i,j,3] = ((al1+al0*(1-al1))*255).astype(np.uint8)
                        print dimg[i,j,3]
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

#collect = io.ImageCollection("./*.png")
#d=merge4c(collect)
#io.imshow(d)
=======
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
>>>>>>> master
