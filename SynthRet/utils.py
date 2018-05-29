# imports
from PIL import Image, ImageEnhance

import cv2
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
from skimage import img_as_ubyte, exposure, io
from scipy.misc import imread
from skimage.morphology import binary_dilation
import math
import random
import os
#from Unet.data import *
#from Unet.unet import *

#def eval():
    #data = dataProcess(300, 300, data_path="./syntheticImages/imgs", label_path="./syntheticImages/labels", test_path="../DRIVE/test/images")
    #imgs_labels = None #TODO
    #unet = myUnet(300, 300)
    #unet.train()
    #predictions = np.load('results/imgs_mask_test.npy')
    #return measure(predictions, imgs_labels)

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
def mergeLayer(collect):
    #***the size of images***
    ncol=300
    nrow=300
    #initial transparent image
    dimg = np.zeros((ncol, nrow, 4),np.uint8)
    #merge layers
    for img in collect:
        if img is None:
            continue
        ids = np.where(img[:,:,3] > 0)
        dimg[ids] = img[ids]
    return dimg

#
def addIllumination(image): # rewrite with skimage
    
    # set parameters (random)
    brightness = np.random.uniform(0.1,3)
    low, high = np.random.randint(low=0,high=30), np.random.randint(low=225,high=255),

    # enhance brightness
    image1 = exposure.adjust_gamma(image, brightness) 
    #exposure.adjust_log(image)   
    
    # enhance contrast 
    img = exposure.rescale_intensity(image1,out_range=(low,high))

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

def showImage(img, points=None):
    if img.ndim == 3:
        plt.imshow(np.transpose(img, (1,0,2)))   #show transposed so x is horizontal and y is vertical
    else:
        plt.imshow(img.T)
    if points is not None:
        x, y = zip(*points)
        plt.scatter(x=x, y=y, c='r')
    plt.show()

def calculateMeanCoverage(path, k=10):
    images = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    means = []
    for f in images:
        binary = imread(path+f)
        for _ in range(k):
            binary = binary_dilation(binary)
        means.append(np.mean(binary))
    return np.mean(np.asarray(means))

if __name__ == '__main__':
    paths = [
        '/../DRIVE/test/1st_manual/',
        '/../DRIVE/test/2nd_manual/'
    ]
    means = []
    k = 10
    for p in paths:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mypath = dir_path + p
        means.append(calculateMeanCoverage(mypath, k))
    print("MEAN COVERAGE WITH DILATION OF " + str(k) + ": " + str(np.mean(np.asarray(means))))
