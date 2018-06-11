import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Tree import Tree
from PIL import Image, ImageDraw
from utils import mergeLayer, addIllumination, showImage, addMask
import cv2
from skimage import io, draw, data
import scipy.misc 
import math
import os 
import time
from OpticalDisc import generateOpticalDisc
from Fovea import generateBackgroundAndFovea

'''
    generate synthetic images. if you want to save the generated files adjust save and path to your needs.
    path is relative to this file.
'''
def generateImages(i=0, save=False, path="/../syntheticImages/"):
    bkg, fovea = generateBackgroundAndFovea()
    od_img, od = generateOpticalDisc()
    vt, groundTruth = generateVesselsTree(fovea, od)
    merged = mergeLayer([bkg, np.transpose(od_img,(1,0,2)), vt])
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


# generate an image containing the vessels tree
def generateVesselsTree(fovea, od):
    tree = Tree(od, fovea)
    tree.growTree()
    return tree.createTreeImage(), tree.createTreeMap()

if __name__ == '__main__':
    k = 2                              # amount of pictures to generate

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

    
    showImage('g',list(np.asarray(res)[:,1])) # show ground truths
    showImage('i',list(np.asarray(res)[:,0])) # show generated images

