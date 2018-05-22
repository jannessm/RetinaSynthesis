# imports
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from skimage import img_as_ubyte
import skimage.io as io
from Unet.unet import *
from Unet.data import *

def eval():
    data = dataProcess(300, 300, data_path="./syntheticImages/imgs", label_path="./syntheticImages/labels", test_path="../DRIVE/test/images")
    imgs_labels = None #TODO
    unet = myUnet(300, 300)
    unet.train()
    predictions = np.load('results/imgs_mask_test.npy')
    return measure(predictions, imgs_labels)

def measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return (TP / (TP+FN), TN / (TN+FP), (TP+TN)/(TP+TN+FP+FN))

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

##code for merge test
#collect = io.ImageCollection("./*.png")
#d=merge4c(collect)
#io.imshow(d)

