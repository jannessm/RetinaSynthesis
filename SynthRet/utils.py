# imports
from PIL import Image, ImageEnhance

import cv2
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
from skimage import img_as_ubyte, exposure, io, transform, draw
from scipy.misc import imread
from scipy.ndimage.morphology import binary_dilation
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

def makeBinary(img, threshold):
    if img.shape[2] == 4:
        r = np.multiply(img[:, :, 0], img[:, :, 3])
        g = np.multiply(img[:, :, 1], img[:, :, 3])
        b = np.multiply(img[:, :, 2], img[:, :, 3])
    else:
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
    rgb = np.dstack((r, g, b))
    grey = np.multiply(rgb, [0.21, 0.72, 0.07])
    grey = np.sum(grey, axis=2)
    grey[np.where(grey < threshold)] = 0
    grey[np.where(grey > threshold)] = 255
    return grey

#
def addIllumination(image): # rewrite with skimage
    
    # set parameters (random)
    brightness = np.random.uniform(0.1,3)
    low, high = np.random.randint(low=0,high=30), np.random.randint(low=225,high=255)

    # enhance brightness
    image1 = exposure.adjust_gamma(image, brightness) 
    #exposure.adjust_log(image)   
    
    # enhance contrast 
    img = exposure.rescale_intensity(image1,out_range=(low,high))

    return img

def showImage(img, points=None):
    if type(img) == list:
        points = points if type(points) == list else [None] * len(img)
        rows = np.floor(np.sqrt(len(img)))
        cols = np.ceil(np.sqrt(len(img))) 
        if not rows > 1:
            rows = 1
            cols = len(img)
        for i in range(len(img)):
            plt.subplot(int(rows), int(cols), i+1)
            _plotHelper(img[i], points[i])
    else:
        _plotHelper(img, points)
    plt.show()

def _plotHelper(img, points):
    if img.ndim == 3:
        plt.imshow(np.transpose(img, (1,0,2)))   #show transposed so x is horizontal and y is vertical
    else:
        plt.imshow(img.T)
    if points is not None:
        x, y = zip(*points)
        plt.scatter(x=x, y=y, c='b')

'''
    add black mask on top of the image
'''
def addMask(image):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isfile(dir_path + '/mask.npy'):
        mask = imread(dir_path + '/../DRIVE/test/mask/01_test_mask.gif')
        mask = transform.resize(mask, (300, 300))
        mask = mask.T
        final_mask = np.zeros((300,300,4))
        black = np.where(mask < 0.5)
        transparent = np.where(mask >= 0.5)
        final_mask[black] = [0,0,0,255]
        final_mask[transparent] = [255,255,255,0]
        np.save(dir_path + '/mask.npy', final_mask)
    else:
        final_mask = np.load(dir_path + '/mask.npy')
        if not final_mask.shape == (300,300,4):
            mask = imread(dir_path + '/../DRIVE/test/mask/01_test_mask.gif')
            mask = transform.resize(mask, (300, 300))
            mask = mask.T
            final_mask = np.zeros((300,300,4))
            black = np.where(mask < 0.5)
            transparent = np.where(mask >= 0.5)
            final_mask[black] = [0,0,0,255]
            final_mask[transparent] = [255,255,255,0]
            np.save(dir_path + '/mask.npy', final_mask)
    return mergeLayer([image, final_mask])

def calculateMeanCoverage(path, k=10):
    images = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    means = []
    for f in images:
        binary = imread(path+f)
        binary = transform.resize(binary, (300, 300))
        if binary.ndim == 3:
            binary = makeBinary(binary, 0.5)
        binary = binary_dilation(binary, iterations=k)
        rgba = np.zeros((300, 300, 4)) # make rgba image from binary
        rgba[np.where(binary)] = [0,0,0, 255] # draw binary on it
        # add fovea
        rr, cc = draw.circle(150, 150, 15)
        draw.set_color(rgba, [rr,cc], [0,0,0,255])
        # add mask
        rgba = addMask(rgba)    # add Mask
        rgba = np.abs(rgba - [255, 255, 255, 0])
        binary = makeBinary(rgba, 200)
        means.append(np.mean(binary) / 255)
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
