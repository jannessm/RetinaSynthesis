import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Tree import Tree
from PIL import Image, ImageDraw
from utils import mergeLayer, addIllumination, showImage, addMask
import cv2
from skimage import io, draw, data
from scipy.misc import imread, imsave
import math
import os 
from OpticalDisc import generateOpticalDisc
from multiprocessing.dummy import Pool
import time

'''
    generate synthetic images. if you want to save the generated files adjust save and path to your needs.
    path is relative to this file.
'''
def generateImages(i=0, save=False, path="/../syntheticImages/"):
    bkg, fovea = generateBackgroundAndFovea()
    od_img, od = generateOpticalDisc()
    vt, groundTruth = generateVesselsTree(fovea, od)
    merged = mergeLayer([bkg, od_img, vt])
    image = addIllumination(merged)
    image = addMask(image)
    gt = addMask(groundTruth)
    if save:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + path
        if not os.path.exists(path + "images"):
            os.makedirs(path + "images")
        if not os.path.exists(path + "groundtruth"):
            os.makedirs(path + "groundtruth")
        imsave(path + "images/" + str(i+1) + ".png", image)
        imsave(path + "groundtruth/" + str(i+1) + ".png", gt)
    return addMask(image), addMask(groundTruth)

# generate an image with the background and fovea
def generateBackgroundAndFovea():
    img=np.zeros((300, 300, 4),np.uint8)            
    img[:,:,]=[255,127,36,255]
    #macula
    change=np.random.randint(-20,20)
    for i in range(100):
        rr,cc=draw.circle(150+change,150+change,26-i/4.0)
        draw.set_color(img,[rr,cc],[190-i,190-i,190-i,255])
    
    #fovea
    PosFovea=(150+change,150+change)
    rr,cc=draw.circle(150+change,150+change,15)
    draw.set_color(img,[rr,cc],[139,126,102,255])
    return img, PosFovea

# generate an image containing the vessels tree
def generateVesselsTree(fovea, od):
    tree = Tree(od, fovea)
    tree.growTree()
    return tree.createTreeImage(), tree.createTreeMap()

if __name__ == '__main__':
    k = 2                               # amount of pictures to generate

    if k > 20:                           # limit threads to upper boundary 20
        nthreads = 20
    else:
        nthreads = k
    
    start = time.time()                 # start timer

    threads = Pool(nthreads)            # generate k images in parallel
    res = threads.map(generateImages, range(k))
    threads.close()
    threads.join()

    print("\n\n" + str(k) + " pictures needed " + str(time.time() - start) + " sec!\n")

    showImage(list(np.asarray(res)[:,0])) # show generated images
    showImage(list(np.asarray(res)[:,1])) # show ground truths
